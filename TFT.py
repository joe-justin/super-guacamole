# model_pipeline_tft.py
import os
import argparse
import pickle
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models import Baseline
from pytorch_forecasting.data import NaNLabelEncoder
from statsmodels.tsa.arima.model import ARIMA

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
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
n_steps = args.forecast_days * 24

# -----------------------
# Data load & prepare
# -----------------------
df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("timestamp")
grouped = df.groupby("Server")
servers = sorted(df["Server"].unique().tolist())

# Ensure exogenous columns exist
EXOG_FEATURES = [
    "cpu_min","cpu_max",
    "mem_min","mem_max",
    "nwin_current","nwin_min","nwin_max",
    "nwou_current","nwou_min","nwou_max"
]
for c in EXOG_FEATURES:
    if c not in df.columns:
        df[c] = 0.0

# Add simple time features
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600
df["time_idx"] = df["time_idx"].astype(int)

# -----------------------
# Pipeline storage
# -----------------------
tft_models_store = {}    # server -> metric -> model
arima_models_store = {}  # server -> metric -> fitted_object
server_level_info = {}
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

# -----------------------
# Iterate per server
# -----------------------
device = torch.device(args.device)

for server, group in tqdm(grouped, desc="Servers"):
    s = group.sort_values("timestamp").copy()
    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}

    preds_df = pd.DataFrame({
        "timestamp": pd.date_range(s["timestamp"].max() + timedelta(hours=1), periods=n_steps, freq="H")
    })
    preds = {}

    for metric in ("cpu_current", "mem_current"):
        series = s[metric].astype(float).values

        # --- ARIMA baseline ---
        try:
            if len(series) >= 10:
                arima = ARIMA(series, order=(5,1,0)).fit()
                arima_models_store.setdefault(server, {})[metric] = arima
                preds_arima = np.array(arima.forecast(steps=n_steps)).flatten()
            else:
                preds_arima = np.repeat(series[-1], n_steps)
        except Exception:
            preds_arima = np.repeat(series[-1], n_steps)

        # --- TFT ---
        # Only proceed if enough rows for TFT
        min_rows = args.seq_len + n_steps
        if len(s) < min_rows:
            preds_tft = preds_arima.copy()  # fallback
        else:
            try:
                # Prepare TimeSeriesDataSet
                tsd = TimeSeriesDataSet(
                    s,
                    time_idx="time_idx",
                    target=metric,
                    group_ids=["Server"],
                    min_encoder_length=args.seq_len,
                    max_encoder_length=args.seq_len,
                    min_prediction_length=n_steps,
                    max_prediction_length=n_steps,
                    static_categoricals=["Server","Function","Application"],
                    time_varying_known_reals=["time_idx","hour","dayofweek"] + EXOG_FEATURES,
                    time_varying_unknown_reals=[metric],
                    target_normalizer=GroupNormalizer(groups=["Server"], transformation="softplus")
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
                    log_interval=10,
                    reduce_on_plateau_patience=3
                ).to(device)

                # Trainer
                from pytorch_lightning import Trainer
                trainer = Trainer(
                    max_epochs=args.tft_epochs,
                    accelerator="gpu" if device.type=="cuda" else "cpu",
                    devices=1 if device.type=="cuda" else None,
                    enable_model_summary=False,
                    enable_checkpointing=False,
                    logger=False
                )
                trainer.fit(tft, train_loader)

                # Iterative forecast
                last_enc = tsd[0][0][:, -args.seq_len:, :]
                last_enc = torch.tensor(last_enc, dtype=torch.float32).to(device)
                preds_tft = []
                window = last_enc.clone()
                for step in range(n_steps):
                    with torch.no_grad():
                        out = tft(window).cpu().numpy().flatten()[0]
                    preds_tft.append(float(out))
                    # shift window: remove oldest, append predicted as future unknown
                    window = torch.cat([window[:,1:,:], torch.tensor(out, dtype=torch.float32).reshape(1,1,1).to(device)], dim=1)
            except Exception:
                preds_tft = preds_arima.copy()

        # --- combined prediction: conservative ---
        preds_combined = np.maximum(preds_arima, preds_tft)
        preds[metric] = {
            "arima": preds_arima,
            "tft": preds_tft,
            "combined": preds_combined
        }

        # Fill preds_df
        if metric=="cpu_current":
            preds_df["cpu_predicted"] = preds_combined
        else:
            preds_df["mem_predicted"] = preds_combined

    # -----------------------
    # Build final CSV
    hist_df = s[["timestamp","cpu_current","mem_current"]].copy().rename(columns={
        "cpu_current":"cpu_actual",
        "mem_current":"mem_actual"
    })
    future_df = preds_df.copy()
    future_df["cpu_actual"] = np.nan
    future_df["mem_actual"] = np.nan
    final_df = pd.concat([hist_df, future_df], ignore_index=True, sort=False)[
        ["timestamp","cpu_actual","mem_actual","cpu_predicted","mem_predicted"]
    ]

    csv_path = os.path.join(args.outdir,f"{server}_actual_predicted.csv")
    final_df.to_csv(csv_path,index=False)

    # --- compute server status ---
    avg_cpu = float(np.nanmean(preds["cpu_current"]["combined"]))
    avg_mem = float(np.nanmean(preds["mem_current"]["combined"]))
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
# Save models & artifacts
# -----------------------
with open(os.path.join(args.outdir,"ARIMA_Model.pkl"),"wb") as f:
    pickle.dump(arima_models_store,f)

with open(os.path.join(args.outdir,"TFT_Model.pkl"),"wb") as f:
    pickle.dump(tft_models_store,f)

# Function / Application summary pickles
func_pct = {}
for func, vals in function_summary.items():
    total = vals["total"] or 1
    func_pct[func] = {
        "healthy_pct": round(vals["healthy"]/total*100,2),
        "needs_attention_pct": round(vals["attention"]/total*100,2),
        "critical_pct": round(vals["critical"]/total*100,2),
        "total": total
    }
with open(os.path.join(args.outdir,"Function.pkl"),"wb") as f:
    pickle.dump(func_pct,f)

app_pct = {}
for app, vals in application_summary.items():
    total = vals["total"] or 1
    app_pct[app] = {
        "healthy_pct": round(vals["healthy"]/total*100,2),
        "needs_attention_pct": round(vals["attention"]/total*100,2),
        "critical_pct": round(vals["critical"]/total*100,2),
        "total": total
    }
with open(os.path.join(args.outdir,"Application.pkl"),"wb") as f:
    pickle.dump(app_pct,f)

# Per-server pickles
for server, info in server_level_info.items():
    with open(os.path.join(args.outdir,f"{server}.pkl"),"wb") as f:
        pickle.dump(info,f)

with open(os.path.join(args.outdir,"server_status_all.pkl"),"wb") as f:
    pickle.dump(server_level_info,f)

# JSON payloads for UI
pd.DataFrame.from_dict(func_pct, orient="index").reset_index().rename(columns={"index":"Function"}).to_json(
    os.path.join(args.outdir,"functions.json"),orient="records")
pd.DataFrame.from_dict(app_pct, orient="index").reset_index().rename(columns={"index":"Application"}).to_json(
    os.path.join(args.outdir,"applications.json"),orient="records")
pd.DataFrame.from_dict(server_level_info, orient="index").to_json(
    os.path.join(args.outdir,"servers_all.json"),orient="records"
)

print("✅ Done. TFT + ARIMA pipeline outputs generated in", args.outdir)
