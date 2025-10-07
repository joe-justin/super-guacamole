# model_pipeline_tft_peakboost.py
"""
TFT + ARIMA pipeline with iterative forecast and configurable peak-boosting.
Outputs per-server CSVs and JSON/pickle artifacts compatible with your UI.
"""
import os
import argparse
import pickle
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from statsmodels.tsa.arima.model import ARIMA

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from pytorch_lightning import Trainer, seed_everything

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", default="output")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--tft_epochs", type=int, default=30)
parser.add_argument("--seq_len", type=int, default=48)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--device", default="cpu")
parser.add_argument("--peak_boost", type=float, default=1.0, help="Multiplier applied to recent peak (1.0=no boost, >1 increases peaks)")
parser.add_argument("--peak_mode", choices=["max","p95"], default="max", help="Which recent peak statistic to use")
parser.add_argument("--clamp_factor_upper", type=float, default=1.2, help="Max allowed factor above observed series max")
parser.add_argument("--min_rows_ratio", type=float, default=1.0, help="min rows required multiplier relative to seq_len + forecast (use <1.0 to relax)")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
n_steps = args.forecast_days * 24

# -----------------------
# Load & prepare data
# -----------------------
df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("timestamp").reset_index(drop=True)

# ensure necessary columns present
required_cols = ["Function","Application","Server","timestamp","cpu_current","mem_current"]
for c in required_cols:
    if c not in df.columns:
        raise SystemExit(f"input CSV missing required column: {c}")

# exogenous defaults
EXOG_FEATURES = [
    "cpu_min","cpu_max",
    "mem_min","mem_max",
    "nwin_current","nwin_min","nwin_max",
    "nwou_current","nwou_min","nwou_max"
]
for c in EXOG_FEATURES:
    if c not in df.columns:
        df[c] = 0.0

# time features & time_idx
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["time_idx"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600).astype(int)

grouped = df.groupby("Server")
servers = sorted(df["Server"].unique().tolist())

# reproducibility
seed_everything(42)
device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

# -----------------------
# storage
# -----------------------
tft_models_store = {}
arima_models_store = {}
server_level_info = {}
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

# helper: decide if server has enough rows for TFT
def has_enough_rows(n_rows, seq_len, horizon):
    required = seq_len + horizon
    return n_rows >= required * args.min_rows_ratio

# -----------------------
# per-server loop
# -----------------------
for server, s in tqdm(grouped, desc="Servers"):
    s = s.sort_values("timestamp").reset_index(drop=True)
    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}

    # prepare preds dataframe for future timestamps
    preds_df = pd.DataFrame({
        "timestamp": pd.date_range(s["timestamp"].max() + timedelta(hours=1), periods=n_steps, freq="H")
    })
    preds = {}

    for metric in ("cpu_current","mem_current"):
        series = s[metric].astype(float).values
        series_min = float(np.nanmin(series)) if len(series)>0 else 0.0
        series_max = float(np.nanmax(series)) if len(series)>0 else 1.0

        # --- ARIMA baseline ---
        try:
            if len(series) >= 10:
                arima = ARIMA(series, order=(5,1,0)).fit()
                preds_arima = np.array(arima.forecast(steps=n_steps)).flatten()
                arima_models_store.setdefault(server, {})[metric] = arima
            else:
                preds_arima = np.repeat(series[-1] if len(series)>0 else 0.0, n_steps)
        except Exception:
            preds_arima = np.repeat(series[-1] if len(series)>0 else 0.0, n_steps)

        # --- TFT with iterative forecast (only if enough rows) ---
        if not has_enough_rows(len(s), args.seq_len, n_steps):
            preds_tft = preds_arima.copy()
        else:
            try:
                # build TimeSeriesDataSet for this server only (GroupNormalizer requires dataset)
                tsd = TimeSeriesDataSet(
                    s,
                    time_idx="time_idx",
                    target=metric,
                    group_ids=["Server"],
                    min_encoder_length=args.seq_len,
                    max_encoder_length=args.seq_len,
                    min_prediction_length=1,
                    max_prediction_length=1,
                    static_categoricals=["Server","Function","Application"],
                    time_varying_known_reals=["time_idx","hour","dayofweek"] + EXOG_FEATURES,
                    time_varying_unknown_reals=[metric],
                    target_normalizer=GroupNormalizer(groups=["Server"], transformation="softplus"),
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True
                )

                train_loader = tsd.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)

                tft = TemporalFusionTransformer.from_dataset(
                    tsd,
                    learning_rate=1e-3,
                    hidden_size=64,
                    attention_head_size=4,
                    dropout=0.1,
                    hidden_continuous_size=32,
                    output_size=1,
                    loss=RMSE(),
                    log_interval=0,
                    reduce_on_plateau_patience=3,
                ).to(device)

                trainer = Trainer(
                    max_epochs=args.tft_epochs,
                    accelerator="gpu" if device.type == "cuda" else "cpu",
                    devices=1 if device.type == "cuda" else None,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                    logger=False,
                )
                trainer.fit(tft, train_loader)

                # iterative forecasting: use the real encoder window and update it each step
                last_window = s[-args.seq_len:].copy().reset_index(drop=True)
                preds_tft = []
                for step in range(n_steps):
                    # compute future time features
                    future_hour = int((last_window["hour"].iloc[-1] + 1) % 24)
                    future_day = int((last_window["dayofweek"].iloc[-1] + ((last_window["hour"].iloc[-1] + 1)//24)) % 7)

                    # build one-row dataframe for prediction consisting of last_window + one future row
                    future_row = last_window.iloc[-1:].copy()
                    future_row["time_idx"] = last_window["time_idx"].iloc[-1] + 1
                    future_row["hour"] = future_hour
                    future_row["dayofweek"] = future_day
                    for c in EXOG_FEATURES:
                        # simple strategy: carry-forward last known exogenous value
                        future_row[c] = last_window[c].iloc[-1]

                    # create a small dataset with last_window + future_row
                    iter_df = pd.concat([last_window, future_row], ignore_index=True)
                    iter_tsd = TimeSeriesDataSet(
                        iter_df,
                        time_idx="time_idx",
                        target=metric,
                        group_ids=["Server"],
                        min_encoder_length=args.seq_len,
                        max_encoder_length=args.seq_len,
                        min_prediction_length=1,
                        max_prediction_length=1,
                        static_categoricals=["Server","Function","Application"],
                        time_varying_known_reals=["time_idx","hour","dayofweek"] + EXOG_FEATURES,
                        time_varying_unknown_reals=[metric],
                        target_normalizer=GroupNormalizer(groups=["Server"], transformation="softplus"),
                        add_relative_time_idx=True,
                        add_target_scales=True,
                        add_encoder_length=True
                    )
                    # dataloader with single batch
                    batch = iter_tsd.to_dataloader(train=False, batch_size=1, num_workers=0)
                    # predict
                    pred_val = None
                    try:
                        for x, y in batch:
                            x = {k: v.to(device) for k, v in x.items()}
                            # using model.predict_item is more stable for single-item prediction
                            out = tft.predict(iter_tsd, mode="raw")
                            # out shape can be (n_samples, seq_len, output_size) depending on mode; take last
                            # but simpler: use tft.predict with dataset -> returns shape (n_samples, prediction_length)
                            out2 = tft.predict(iter_tsd)
                            pred_val = float(out2[0])
                    except Exception:
                        # fallback to last observed value
                        pred_val = float(last_window[metric].iloc[-1])

                    # Peak-boost logic
                    # compute recent peak statistic from last_window (before appending new pred)
                    try:
                        if args.peak_mode == "max":
                            recent_peak = float(np.nanmax(last_window[metric].values))
                        else:
                            recent_peak = float(np.nanpercentile(last_window[metric].values, 95))
                    except Exception:
                        recent_peak = float(np.nanmax(series)) if len(series) > 0 else 0.0

                    # apply peak boost multiplier
                    boosted_threshold = recent_peak * float(args.peak_boost)
                    # choose boosted value = max(pred_val, boosted_threshold)
                    boosted_pred = float(max(pred_val, boosted_threshold))

                    # clamp to plausible range based on historic series
                    upper_limit = series_max * float(args.clamp_factor_upper)
                    if np.isnan(upper_limit) or upper_limit <= 0:
                        upper_limit = boosted_pred * 10.0 + 1.0
                    boosted_pred = float(np.minimum(boosted_pred, upper_limit))
                    boosted_pred = float(np.maximum(boosted_pred, 0.0))

                    preds_tft.append(boosted_pred)

                    # append predicted row to last_window for next step
                    new_row = future_row.copy()
                    new_row[metric] = boosted_pred
                    last_window = pd.concat([last_window.iloc[1:].reset_index(drop=True), new_row.reset_index(drop=True)], ignore_index=True)

                # store model
                tft_models_store.setdefault(server, {})[metric] = tft

            except Exception:
                # on any failure, fall back to ARIMA
                preds_tft = preds_arima.copy()

        # final conservative combined predictions (max of tft/arima)
        preds_combined = np.maximum(np.array(preds_tft), np.array(preds_arima))
        preds[metric] = {"tft": np.array(preds_tft), "arima": np.array(preds_arima), "combined": preds_combined}

        # write into preds_df
        if metric == "cpu_current":
            preds_df["cpu_predicted"] = preds_combined
        else:
            preds_df["mem_predicted"] = preds_combined

    # -----------------------
    # Build final CSV (historic actuals then future predicted)
    hist_df = s[["timestamp","cpu_current","mem_current"]].copy().rename(columns={
        "cpu_current":"cpu_actual","mem_current":"mem_actual"
    })
    future_df = preds_df.copy()
    future_df["cpu_actual"] = np.nan
    future_df["mem_actual"] = np.nan

    final_df = pd.concat([hist_df, future_df], ignore_index=True, sort=False)[
        ["timestamp","cpu_actual","mem_actual","cpu_predicted","mem_predicted"]
    ]
    csv_path = os.path.join(args.outdir, f"{server}_actual_predicted.csv")
    final_df.to_csv(csv_path, index=False)

    # server status
    avg_cpu = float(np.nanmean(preds["cpu_current"]["combined"]))
    avg_mem = float(np.nanmean(preds["mem_current"]["combined"]))
    avg_cpu = 0.0 if np.isnan(avg_cpu) else avg_cpu
    avg_mem = 0.0 if np.isnan(avg_mem) else avg_mem
    worst_avg = max(avg_cpu, avg_mem)
    if worst_avg < 50:
        status = "healthy"
    elif worst_avg <= 80:
        status = "needs_attention"
    else:
        status = "critical"

    server_level_info[server] = {
        "server": server,
        "Function": meta["Function"],
        "Application": meta["Application"],
        "avg_cpu_pred": avg_cpu,
        "avg_mem_pred": avg_mem,
        "worst_avg_pred": worst_avg,
        "status": status,
        "pred_csv": csv_path
    }

    function_summary[meta["Function"]]["total"] += 1
    application_summary[meta["Application"]]["total"] += 1
    key_map = {"healthy":"healthy","needs_attention":"attention","critical":"critical"}
    function_summary[meta["Function"]][key_map[status]] += 1
    application_summary[meta["Application"]][key_map[status]] += 1

# -----------------------
# Save artifacts
with open(os.path.join(args.outdir,"ARIMA_Model.pkl"), "wb") as f:
    pickle.dump(arima_models_store, f)
with open(os.path.join(args.outdir,"TFT_Model.pkl"), "wb") as f:
    pickle.dump(tft_models_store, f)

# Function/Application summary pickles
func_pct = {}
for func, vals in function_summary.items():
    total = vals["total"] or 1
    func_pct[func] = {
        "healthy_pct": round(vals["healthy"]/total*100,2),
        "needs_attention_pct": round(vals["attention"]/total*100,2),
        "critical_pct": round(vals["critical"]/total*100,2),
        "total": total
    }
with open(os.path.join(args.outdir,"Function.pkl"), "wb") as f:
    pickle.dump(func_pct, f)

app_pct = {}
for app, vals in application_summary.items():
    total = vals["total"] or 1
    app_pct[app] = {
        "healthy_pct": round(vals["healthy"]/total*100,2),
        "needs_attention_pct": round(vals["attention"]/total*100,2),
        "critical_pct": round(vals["critical"]/total*100,2),
        "total": total
    }
with open(os.path.join(args.outdir,"Application.pkl"), "wb") as f:
    pickle.dump(app_pct, f)

# per-server pickles
for server, info in server_level_info.items():
    with open(os.path.join(args.outdir, f"{server}.pkl"), "wb") as f:
        pickle.dump(info, f)

with open(os.path.join(args.outdir,"server_status_all.pkl"), "wb") as f:
    pickle.dump(server_level_info, f)

# JSON outputs for UI
pd.DataFrame.from_dict(func_pct, orient="index").reset_index().rename(columns={"index":"Function"}).to_json(
    os.path.join(args.outdir,"functions.json"), orient="records")
pd.DataFrame.from_dict(app_pct, orient="index").reset_index().rename(columns={"index":"Application"}).to_json(
    os.path.join(args.outdir,"applications.json"), orient="records")
pd.DataFrame.from_dict(server_level_info, orient="index").to_json(
    os.path.join(args.outdir,"servers_all.json"), orient="records"
)

print("✅ Done. Outputs in", args.outdir)
