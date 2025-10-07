# model_pipeline_tft_peakboost_quantiles.py
"""
TFT + ARIMA pipeline with iterative forecast, configurable peak-boosting, and prediction confidence intervals.
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
from pytorch_forecasting.metrics import QuantileLoss
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
parser.add_argument("--peak_boost", type=float, default=1.0)
parser.add_argument("--peak_mode", choices=["max", "p95"], default="max")
parser.add_argument("--clamp_factor_upper", type=float, default=1.2)
parser.add_argument("--quantiles", default="0.1,0.5,0.9", help="Comma-separated list of quantiles for confidence intervals")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
n_steps = args.forecast_days * 24

quantiles = [float(q.strip()) for q in args.quantiles.split(",")]

# -----------------------
# Load & prepare data
# -----------------------
df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("timestamp").reset_index(drop=True)

required_cols = ["Function", "Application", "Server", "timestamp", "cpu_current", "mem_current"]
for c in required_cols:
    if c not in df.columns:
        raise SystemExit(f"input CSV missing required column: {c}")

# Exogenous features
EXOG_FEATURES = [
    "cpu_min","cpu_max","mem_min","mem_max",
    "nwin_current","nwin_min","nwin_max",
    "nwou_current","nwou_min","nwou_max"
]
for c in EXOG_FEATURES:
    if c not in df.columns:
        df[c] = 0.0

df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["time_idx"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600).astype(int)

grouped = df.groupby("Server")
servers = sorted(df["Server"].unique().tolist())

seed_everything(42)
device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

# -----------------------
# Storage
# -----------------------
tft_models_store = {}
arima_models_store = {}
server_level_info = {}
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

# -----------------------
# Helpers
# -----------------------
def has_enough_rows(n_rows, seq_len, horizon):
    return n_rows >= seq_len + horizon

# -----------------------
# Per-Server Loop
# -----------------------
for server, s in tqdm(grouped, desc="Servers"):
    s = s.sort_values("timestamp").reset_index(drop=True)
    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}

    preds_df = pd.DataFrame({
        "timestamp": pd.date_range(s["timestamp"].max() + timedelta(hours=1), periods=n_steps, freq="H")
    })
    preds = {}

    for metric in ("cpu_current", "mem_current"):
        series = s[metric].astype(float).values
        series_max = float(np.nanmax(series)) if len(series) else 1.0

        # --- ARIMA baseline ---
        try:
            arima = ARIMA(series, order=(5, 1, 0)).fit()
            preds_arima = np.array(arima.forecast(steps=n_steps)).flatten()
            arima_models_store.setdefault(server, {})[metric] = arima
        except Exception:
            preds_arima = np.repeat(series[-1] if len(series) else 0.0, n_steps)

        # --- TFT ---
        if not has_enough_rows(len(s), args.seq_len, n_steps):
            preds_tft = preds_arima.copy()
            pred_q10 = preds_tft * 0.9
            pred_q90 = preds_tft * 1.1
        else:
            try:
                tsd = TimeSeriesDataSet(
                    s,
                    time_idx="time_idx",
                    target=metric,
                    group_ids=["Server"],
                    min_encoder_length=args.seq_len,
                    max_encoder_length=args.seq_len,
                    min_prediction_length=1,
                    max_prediction_length=1,
                    static_categoricals=["Server", "Function", "Application"],
                    time_varying_known_reals=["time_idx", "hour", "dayofweek"] + EXOG_FEATURES,
                    time_varying_unknown_reals=[metric],
                    target_normalizer=GroupNormalizer(groups=["Server"], transformation="softplus"),
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
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
                    loss=QuantileLoss(quantiles=quantiles),
                    log_interval=0,
                ).to(device)

                trainer = Trainer(
                    max_epochs=args.tft_epochs,
                    accelerator="gpu" if device.type == "cuda" else "cpu",
                    devices=1 if device.type == "cuda" else None,
                    enable_checkpointing=False,
                    logger=False,
                )
                trainer.fit(tft, train_loader)

                # iterative forecast
                last_window = s[-args.seq_len:].copy().reset_index(drop=True)
                preds_tft, pred_q10, pred_q90 = [], [], []

                for step in range(n_steps):
                    future_row = last_window.iloc[-1:].copy()
                    future_row["time_idx"] = last_window["time_idx"].iloc[-1] + 1
                    future_row["hour"] = (last_window["hour"].iloc[-1] + 1) % 24
                    future_row["dayofweek"] = (last_window["dayofweek"].iloc[-1] + 1) % 7
                    for c in EXOG_FEATURES:
                        future_row[c] = last_window[c].iloc[-1]
                    iter_df = pd.concat([last_window, future_row], ignore_index=True)

                    iter_tsd = TimeSeriesDataSet.from_dataset(tsd, iter_df, stop_randomization=True)
                    preds_all = tft.predict(iter_tsd, mode="quantiles")
                    p10, p50, p90 = preds_all[0, 0, 0], preds_all[0, 0, 1], preds_all[0, 0, 2]

                    # Apply peak boosting
                    if args.peak_mode == "max":
                        recent_peak = float(np.nanmax(last_window[metric].values))
                    else:
                        recent_peak = float(np.nanpercentile(last_window[metric].values, 95))
                    boosted_val = max(p50, recent_peak * args.peak_boost)
                    boosted_val = min(boosted_val, series_max * args.clamp_factor_upper)
                    boosted_val = max(boosted_val, 0.0)

                    preds_tft.append(boosted_val)
                    pred_q10.append(p10)
                    pred_q90.append(p90)

                    new_row = future_row.copy()
                    new_row[metric] = boosted_val
                    last_window = pd.concat([last_window.iloc[1:], new_row], ignore_index=True)

                tft_models_store.setdefault(server, {})[metric] = tft

            except Exception:
                preds_tft = preds_arima.copy()
                pred_q10 = preds_tft * 0.9
                pred_q90 = preds_tft * 1.1

        preds[metric] = {"combined": np.maximum(preds_tft, preds_arima),
                         "p10": pred_q10, "p90": pred_q90}

        if metric == "cpu_current":
            preds_df["cpu_predicted"] = preds[metric]["combined"]
            preds_df["cpu_predicted_p10"] = preds[metric]["p10"]
            preds_df["cpu_predicted_p90"] = preds[metric]["p90"]
        else:
            preds_df["mem_predicted"] = preds[metric]["combined"]
            preds_df["mem_predicted_p10"] = preds[metric]["p10"]
            preds_df["mem_predicted_p90"] = preds[metric]["p90"]

    # Combine actual + predicted
    hist_df = s[["timestamp", "cpu_current", "mem_current"]].rename(
        columns={"cpu_current":"cpu_actual","mem_current":"mem_actual"}
    )
    future_df = preds_df.copy()
    future_df["cpu_actual"] = np.nan
    future_df["mem_actual"] = np.nan
    final_df = pd.concat([hist_df, future_df], ignore_index=True)

    csv_path = os.path.join(args.outdir, f"{server}_actual_predicted.csv")
    final_df.to_csv(csv_path, index=False)

    avg_cpu = float(np.nanmean(preds["cpu_current"]["combined"]))
    avg_mem = float(np.nanmean(preds["mem_current"]["combined"]))
    worst_avg = max(avg_cpu, avg_mem)
    if worst_avg < 50: status = "healthy"
    elif worst_avg <= 80: status = "needs_attention"
    else: status = "critical"

    server_level_info[server] = {
        "server": server,
        "Function": meta["Function"],
        "Application": meta["Application"],
        "avg_cpu_pred": avg_cpu,
        "avg_mem_pred": avg_mem,
        "worst_avg_pred": worst_avg,
        "status": status,
        "pred_csv": csv_path,
    }

# Save PKLs + JSONs (same as before)
with open(os.path.join(args.outdir,"ARIMA_Model.pkl"), "wb") as f:
    pickle.dump(arima_models_store, f)
with open(os.path.join(args.outdir,"TFT_Model.pkl"), "wb") as f:
    pickle.dump(tft_models_store, f)

pd.DataFrame.from_dict(server_level_info, orient="index").to_json(
    os.path.join(args.outdir,"servers_all.json"), orient="records"
)

print("✅ Done. Outputs in", args.outdir)
