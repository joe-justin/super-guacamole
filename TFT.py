# model_pipeline_multi_tft.py
import os
import argparse
import pickle
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.arima.model import ARIMA

# TFT model (using PyTorch Forecasting)
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", default="output")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--tft_epochs", type=int, default=50)
parser.add_argument("--seq_len", type=int, default=48)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
n_steps = args.forecast_days * 24
device = torch.device(args.device)

# -----------------------
# Data load & prepare
# -----------------------
df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("timestamp")
grouped = df.groupby("Server")
servers = sorted(df["Server"].unique().tolist())

def resample_server(sdf: pd.DataFrame) -> pd.DataFrame:
    sdf = sdf.set_index("timestamp").sort_index()
    idx = pd.date_range(sdf.index.min(), sdf.index.max(), freq="H")
    sdf = sdf.reindex(idx)
    sdf[["Function","Application","Server"]] = sdf[["Function","Application","Server"]].ffill()
    numeric_cols = [c for c in sdf.columns if c not in ["Function","Application","Server"]]
    sdf[numeric_cols] = sdf[numeric_cols].ffill().bfill()
    sdf = sdf.reset_index().rename(columns={"index":"timestamp"})
    return sdf

# -----------------------
# Pipeline storage
# -----------------------
tft_models_store = {}
arima_models_store = {}
server_level_info = {}
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

# Exogenous feature list
EXOG_FEATURES = [
    "cpu_min","cpu_max",
    "mem_min","mem_max",
    "nwin_current","nwin_min","nwin_max",
    "nwou_current","nwou_min","nwou_max"
]

# -----------------------
# Loop per server
# -----------------------
for server, group in tqdm(grouped, desc="Servers"):
    s = resample_server(group.copy())
    # ensure exog columns exist
    for c in EXOG_FEATURES:
        if c not in s.columns:
            s[c] = 0.0
    s["hour"] = s["timestamp"].dt.hour
    s["dayofweek"] = s["timestamp"].dt.dayofweek

    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}
    preds = {}
    future_index = pd.date_range(s["timestamp"].max() + timedelta(hours=1), periods=n_steps, freq="H")
    preds_df = pd.DataFrame({"timestamp": future_index})

    for metric in ("cpu_current","mem_current"):
        series = s[metric].astype(float).values
        if np.isnan(series).all():
            series = np.zeros(1)

        # -----------------------
        # ARIMA baseline
        # -----------------------
        try:
            if len(series) >= 20:
                arima = ARIMA(series, order=(5,1,0))
                arima_res = arima.fit()
                arima_pred = np.array(arima_res.forecast(steps=n_steps)).flatten()
                arima_models_store.setdefault(server, {})[metric] = arima_res
            else:
                arima_pred = np.repeat(series[-1], n_steps)
        except Exception:
            arima_pred = np.repeat(series[-1], n_steps)

        # -----------------------
        # TFT model
        # -----------------------
        # Prepare dataset for TFT
        df_tft = s.copy()
        df_tft["time_idx"] = np.arange(len(s))
        feature_cols = [metric] + EXOG_FEATURES
        target = metric

        training_cutoff = df_tft["time_idx"].max() - 1  # full series
        tft_dataset = TimeSeriesDataSet(
            df_tft,
            time_idx="time_idx",
            target=target,
            group_ids=["Server"],
            time_varying_known_reals=["hour","dayofweek"] + EXOG_FEATURES,
            max_encoder_length=args.seq_len,
            max_prediction_length=n_steps,
            target_normalizer=GroupNormalizer(groups=["Server"], transformation="robust")
        )

        # DataLoader
        train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)

        # TFT model
        tft_model = TemporalFusionTransformer.from_dataset(
            tft_dataset,
            learning_rate=1e-3,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=1,
            loss=torch.nn.MSELoss(),
            log_interval=0,
            reduce_on_plateau_patience=3
        ).to(device)

        trainer = torch.optim.Adam(tft_model.parameters(), lr=1e-3)

        # Training loop
        tft_model.train()
        for epoch in range(args.tft_epochs):
            for batch in train_dataloader:
                trainer.zero_grad()
                x, y = batch
                loss = tft_model.training_step(batch, 0)
                loss.backward()
                trainer.step()

        # -----------------------
        # Smart iterative forecast
        # -----------------------
        last_encoder_data = df_tft[-args.seq_len:].copy()
        last_encoder_data["time_idx"] = np.arange(len(df_tft)-args.seq_len, len(df_tft))
        seq_features = last_encoder_data[feature_cols + ["time_idx","hour","dayofweek"]].values.astype(float)
        scaler_y = RobustScaler()
        scaler_y.fit(series.reshape(-1,1))
        preds_tft_iter = []

        for step in range(n_steps):
            future_hour = (int(s["hour"].iloc[-1]) + step + 1) % 24
            future_day = (int(s["dayofweek"].iloc[-1]) + ((int(s["hour"].iloc[-1]) + step + 1)//24)) % 7
            future_feats = seq_features[-1].copy()
            future_feats[-2] = future_hour
            future_feats[-1] = future_day
            input_df = pd.DataFrame([future_feats], columns=feature_cols + ["time_idx","hour","dayofweek"])
            input_df["Server"] = server
            input_df["time_idx"] = len(s) + step
            try:
                pred_scaled = tft_model.predict(input_df)
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()[0]
            except Exception:
                pred = series[-1]
            preds_tft_iter.append(pred)
            # update sequence
            next_row = future_feats.copy()
            next_row[0] = pred
            seq_features = np.vstack([seq_features, next_row])
            if len(seq_features) > args.seq_len:
                seq_features = seq_features[-args.seq_len:]

        preds_tft = np.array(preds_tft_iter)

        # Combined predictions
        preds_combined = np.maximum(preds_tft, arima_pred)
        preds[metric] = {"tft": preds_tft, "arima": arima_pred, "combined": preds_combined}

        # Save into preds_df
        if metric=="cpu_current":
            preds_df["cpu_predicted"] = preds_combined
            preds_df["cpu_actual"] = list(s[metric].astype(float)) + [np.nan]*n_steps
        else:
            preds_df["mem_predicted"] = preds_combined
            preds_df["mem_actual"] = list(s[metric].astype(float)) + [np.nan]*n_steps

        # Save TFT model metadata
        tft_models_store.setdefault(server, {})[metric] = {"model": tft_model}

    # -----------------------
    # Final CSV
    # -----------------------
    hist_df = s[["timestamp","cpu_current","mem_current"]].copy().rename(columns={
        "cpu_current":"cpu_actual",
        "mem_current":"mem_actual"
    })
    future_df = pd.DataFrame({"timestamp": preds_df["timestamp"]})
    future_df["cpu_predicted"] = preds_df["cpu_predicted"].astype(float)
    future_df["mem_predicted"] = preds_df["mem_predicted"].astype(float)
    future_df["cpu_actual"] = np.nan
    future_df["mem_actual"] = np.nan

    final_df = pd.concat([hist_df, future_df], ignore_index=True, sort=False)[
        ["timestamp","cpu_actual","mem_actual","cpu_predicted","mem_predicted"]
    ]

    csv_path = os.path.join(args.outdir, f"{server}_actual_predicted.csv")
    final_df.to_csv(csv_path, index=False)

    # -----------------------
    # Server level info
    # -----------------------
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
with open(os.path.join(args.outdir,"TFT_Model.pkl"),"wb") as f:
    pickle.dump(tft_models_store,f)
with open(os.path.join(args.outdir,"ARIMA_Model.pkl"),"wb") as f:
    pickle.dump(arima_models_store,f)

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

for server, info in server_level_info.items():
    with open(os.path.join(args.outdir,f"{server}.pkl"),"wb") as f:
        pickle.dump(info,f)

with open(os.path.join(args.outdir,"server_status_all.pkl"),"wb") as f:
    pickle.dump(server_level_info,f)

# JSON for UI
pd.DataFrame.from_dict(func_pct, orient="index").reset_index().rename(columns={"index":"Function"}).to_json(
    os.path.join(args.outdir,"functions.json"),orient="records")
pd.DataFrame.from_dict(app_pct, orient="index").reset_index().rename(columns={"index":"Application"}).to_json(
    os.path.join(args.outdir,"applications.json"),orient="records")
pd.DataFrame.from_dict(server_level_info, orient="index").to_json(
    os.path.join(args.outdir,"servers_all.json"),orient="records")

print("✅ Done. TFT + ARIMA predictions saved in", args.outdir)
