# model_pipeline_tft_arima.py
import os
import argparse
import pickle
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from statsmodels.tsa.arima.model import ARIMA
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline, GroupNormalizer
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning import Trainer, seed_everything

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", default="output")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--tft_epochs", type=int, default=60)
parser.add_argument("--seq_len", type=int, default=48)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
n_steps = args.forecast_days * 24

# -----------------------
# Load & prepare data
# -----------------------
df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("timestamp")
df["time_idx"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600).astype(int)

grouped = df.groupby("Server")
servers = sorted(df["Server"].unique().tolist())

# Exogenous features
EXOG_FEATURES = [
    "cpu_min","cpu_max",
    "mem_min","mem_max",
    "nwin_current","nwin_min","nwin_max",
    "nwou_current","nwou_min","nwou_max"
]

for c in EXOG_FEATURES:
    if c not in df.columns:
        df[c] = 0.0

# Add time features
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek

device = torch.device(args.device)
seed_everything(42)

# -----------------------
# Pipeline storage
# -----------------------
tft_models_store = {}
arima_models_store = {}
server_level_info = {}
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

# -----------------------
# Iterate per server
# -----------------------
for server, s in tqdm(grouped, desc="Servers"):
    s = s.sort_values("timestamp").reset_index(drop=True)
    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}

    preds_df = pd.DataFrame({
        "timestamp": pd.date_range(s["timestamp"].max() + timedelta(hours=1), periods=n_steps, freq="H")
    })

    for metric in ["cpu_current", "mem_current"]:
        series = s[metric].astype(float).values

        # --- ARIMA baseline ---
        try:
            if len(series) >= 20:
                arima_model = ARIMA(series, order=(5,1,0)).fit()
                arima_forecast = arima_model.forecast(steps=n_steps)
                arima_models_store.setdefault(server, {})[metric] = arima_model
            else:
                arima_forecast = np.repeat(series[-1], n_steps)
        except Exception:
            arima_forecast = np.repeat(series[-1], n_steps)

        # --- Prepare TFT dataset ---
        tft_dataset = TimeSeriesDataSet(
            s,
            time_idx="time_idx",
            target=metric,
            group_ids=["Server"],
            min_encoder_length=args.seq_len,
            max_encoder_length=args.seq_len,
            min_prediction_length=n_steps,
            max_prediction_length=n_steps,
            static_categoricals=["Server", "Function", "Application"],
            time_varying_known_reals=["time_idx", "hour", "dayofweek"] + EXOG_FEATURES,
            time_varying_unknown_reals=[metric],
            target_normalizer=GroupNormalizer(groups=["Server"], transformation="standard"),
        )

        tft_loader = DataLoader(tft_dataset, batch_size=args.batch_size, shuffle=True)

        tft = TemporalFusionTransformer.from_dataset(
            tft_dataset,
            learning_rate=1e-3,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=1,
            loss=SMAPE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        trainer = Trainer(
            max_epochs=args.tft_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            enable_model_summary=True,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(tft, tft_loader)

        # --- Iterative forecast for n_steps ---
        tft.eval()
        last_window = s[-args.seq_len:].copy()
        preds_tft = []
        for step in range(n_steps):
            future_time_idx = last_window["time_idx"].iloc[-1] + 1
            future_hour = (last_window["hour"].iloc[-1] + 1) % 24
            future_day = (last_window["dayofweek"].iloc[-1] + 1//24) % 7

            next_row = last_window.iloc[-1:].copy()
            next_row["time_idx"] = future_time_idx
            next_row["hour"] = future_hour
            next_row["dayofweek"] = future_day
            # carry-forward exogenous
            for c in EXOG_FEATURES:
                next_row[c] = next_row[c].values[0]

            x_pred = TimeSeriesDataSet.from_parameters(
                next_row,
                tft_dataset
            )
            with torch.no_grad():
                pred = tft.predict(x_pred, mode="raw")[0].numpy()[0]
            preds_tft.append(float(pred))
            last_window = pd.concat([last_window.iloc[1:], next_row], ignore_index=True)

        # --- Combine TFT + ARIMA for conservative prediction ---
        preds_combined = np.maximum(preds_tft, arima_forecast)

        # --- Save predictions to dataframe ---
        if metric == "cpu_current":
            preds_df["cpu_predicted"] = preds_combined
            preds_df["cpu_actual"] = list(s["cpu_current"]) + [np.nan]*n_steps
        else:
            preds_df["mem_predicted"] = preds_combined
            preds_df["mem_actual"] = list(s["mem_current"]) + [np.nan]*n_steps

        # Save TFT model metadata
        tft_models_store.setdefault(server, {})[metric] = tft

    # --- Build final CSV ---
    hist_df = s[["timestamp", "cpu_current", "mem_current"]].rename(columns={
        "cpu_current":"cpu_actual",
        "mem_current":"mem_actual"
    })
    future_df = pd.DataFrame({
        "timestamp": preds_df["timestamp"],
        "cpu_actual": np.nan,
        "mem_actual": np.nan,
        "cpu_predicted": preds_df["cpu_predicted"],
        "mem_predicted": preds_df["mem_predicted"],
    })
    final_df = pd.concat([hist_df, future_df], ignore_index=True)
    csv_path = os.path.join(args.outdir, f"{server}_actual_predicted.csv")
    final_df.to_csv(csv_path, index=False)

    # --- Server-level status ---
    avg_cpu = float(np.nanmean(preds_df["cpu_predicted"]))
    avg_mem = float(np.nanmean(preds_df["mem_predicted"]))
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
# Save models and artifacts
# -----------------------
with open(os.path.join(args.outdir, "TFT_Model.pkl"), "wb") as f:
    pickle.dump(tft_models_store, f)

with open(os.path.join(args.outdir, "ARIMA_Model.pkl"), "wb") as f:
    pickle.dump(arima_models_store, f)

# Function/Application summaries
func_pct = {f: {
    "healthy_pct": round(vals["healthy"]/vals["total"]*100,2),
    "needs_attention_pct": round(vals["attention"]/vals["total"]*100,2),
    "critical_pct": round(vals["critical"]/vals["total"]*100,2),
    "total": vals["total"]
} for f, vals in function_summary.items()}

app_pct = {a: {
    "healthy_pct": round(vals["healthy"]/vals["total"]*100,2),
    "needs_attention_pct": round(vals["attention"]/vals["total"]*100,2),
    "critical_pct": round(vals["critical"]/vals["total"]*100,2),
    "total": vals["total"]
} for a, vals in application_summary.items()}

with open(os.path.join(args.outdir, "Function.pkl"), "wb") as f:
    pickle.dump(func_pct, f)
with open(os.path.join(args.outdir, "Application.pkl"), "wb") as f:
    pickle.dump(app_pct, f)

for server, info in server_level_info.items():
    with open(os.path.join(args.outdir, f"{server}.pkl"), "wb") as f:
        pickle.dump(info, f)

with open(os.path.join(args.outdir, "server_status_all.pkl"), "wb") as f:
    pickle.dump(server_level_info, f)

# JSON outputs for UI
pd.DataFrame.from_dict(func_pct, orient="index").reset_index().rename(columns={"index":"Function"}).to_json(
    os.path.join(args.outdir, "functions.json"), orient="records")
pd.DataFrame.from_dict(app_pct, orient="index").reset_index().rename(columns={"index":"Application"}).to_json(
    os.path.join(args.outdir, "applications.json"), orient="records")
pd.DataFrame.from_dict(server_level_info, orient="index").to_json(
    os.path.join(args.outdir, "servers_all.json"), orient="records"
)

print("✅ Done. Outputs written to", args.outdir)
