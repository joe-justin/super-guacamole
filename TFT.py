# model_pipeline_multi_tft.py
import os
import argparse
import pickle
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA

import torch
from pytorch_forecasting import TimeSeriesDataSet, Baseline, Trainer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", default="output")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--tft_epochs", type=int, default=60)
parser.add_argument("--tft_lr", type=float, default=1e-3)
parser.add_argument("--tft_encoder_len", type=int, default=48)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
n_steps = args.forecast_days * 24
device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")

# -----------------------
# Data load & prepare
# -----------------------
df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("timestamp")
grouped = df.groupby("Server")
servers = sorted(df["Server"].unique().tolist())

EXOG_FEATURES = [
    "cpu_min","cpu_max",
    "mem_min","mem_max",
    "nwin_current","nwin_min","nwin_max",
    "nwou_current","nwou_min","nwou_max"
]

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

# -----------------------
# Pipeline per server
# -----------------------
for server, group in tqdm(grouped, desc="Servers"):
    s = resample_server(group.copy())
    for c in EXOG_FEATURES:
        if c not in s.columns:
            s[c] = 0.0
    # time features
    s["hour"] = s["timestamp"].dt.hour
    s["dayofweek"] = s["timestamp"].dt.dayofweek

    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}
    preds = {}
    future_index = pd.date_range(s["timestamp"].max() + timedelta(hours=1), periods=n_steps, freq="H")
    preds_df = pd.DataFrame({"timestamp": future_index})

    for metric in ["cpu_current","mem_current"]:
        series = s[metric].astype(float).values

        # --- ARIMA baseline ---
        try:
            if len(series)>=20:
                arima = ARIMA(series, order=(5,1,0))
                arima_res = arima.fit()
                preds_arima = np.array(arima_res.forecast(steps=n_steps)).flatten()
                arima_models_store.setdefault(server, {})[metric] = arima_res
            else:
                preds_arima = np.repeat(series[-1], n_steps)
        except Exception:
            preds_arima = np.repeat(series[-1], n_steps)

        # --- TFT setup ---
        s_metric = s.copy()
        feature_cols = [metric] + EXOG_FEATURES + ["hour","dayofweek"]
        s_metric["time_idx"] = np.arange(len(s_metric))
        target_normalizer = None
        if np.var(series) > 1e-6:
            target_normalizer = GroupNormalizer(groups=["Server"], transformation="robust")
        dataset = TimeSeriesDataSet(
            s_metric,
            time_idx="time_idx",
            target=metric,
            group_ids=["Server"],
            max_encoder_length=args.tft_encoder_len,
            max_prediction_length=1,
            time_varying_known_reals=["hour","dayofweek"] + EXOG_FEATURES,
            time_varying_unknown_reals=[metric],
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        dataloader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)

        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=args.tft_lr,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=1,
            loss=torch.nn.MSELoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        ).to(device)

        trainer = Trainer(
            max_epochs=args.tft_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            gradient_clip_val=0.1
        )
        trainer.fit(tft, train_dataloaders=dataloader)

        # --- Iterative forecast ---
        last_window = s_metric[-args.tft_encoder_len:].copy()
        preds_tft = []
        for step in range(n_steps):
            future_row = last_window.iloc[-1:].copy()
            future_row["hour"] = (future_row["hour"].values[0]+1)%24
            future_row["dayofweek"] = (future_row["dayofweek"].values[0]+((future_row["hour"].values[0]+1)//24))%7
            for feat in EXOG_FEATURES:
                future_row[feat] = last_window[feat].iloc[-1]
            last_window = pd.concat([last_window,future_row],ignore_index=True)
            future_row["time_idx"] = last_window.index[-1]
            iter_dataset = TimeSeriesDataSet(
                last_window,
                time_idx="time_idx",
                target=metric,
                group_ids=["Server"],
                max_encoder_length=args.tft_encoder_len,
                max_prediction_length=1,
                time_varying_known_reals=["hour","dayofweek"] + EXOG_FEATURES,
                time_varying_unknown_reals=[metric],
                target_normalizer=target_normalizer,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True
            )
            batch = iter_dataset.to_dataloader(train=False, batch_size=1)
            for x, y in batch:
                x = {k:v.to(device) for k,v in x.items()}
                p = tft(x).detach().cpu().numpy().flatten()[0]
            preds_tft.append(p)
            last_window[metric].iloc[-1] = p
            last_window = last_window[-args.tft_encoder_len:]

        # Combined prediction
        preds_combined = np.maximum(preds_tft, preds_arima)

        preds[metric] = {
            "tft": np.array(preds_tft),
            "arima": np.array(preds_arima),
            "combined": preds_combined
        }

        if metric=="cpu_current":
            preds_df["cpu_predicted"] = preds_combined
        else:
            preds_df["mem_predicted"] = preds_combined

    # --- Build final CSV ---
    hist_df = s[["timestamp","cpu_current","mem_current"]].copy().rename(columns={
        "cpu_current":"cpu_actual","mem_current":"mem_actual"
    })
    future_df = pd.DataFrame({"timestamp": preds_df["timestamp"]})
    future_df["cpu_predicted"] = preds_df["cpu_predicted"].astype(float)
    future_df["mem_predicted"] = preds_df["mem_predicted"].astype(float)
    future_df["cpu_actual"] = np.nan
    future_df["mem_actual"] = np.nan

    final_df = pd.concat([hist_df,future_df],ignore_index=True)[
        ["timestamp","cpu_actual","mem_actual","cpu_predicted","mem_predicted"]
    ]
    csv_path = os.path.join(args.outdir,f"{server}_actual_predicted.csv")
    final_df.to_csv(csv_path,index=False)

    # --- Server status ---
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
        "server":server,
        "Function":meta["Function"],
        "Application":meta["Application"],
        "avg_cpu_pred":avg_cpu,
        "avg_mem_pred":avg_mem,
        "worst_avg_pred":worst_avg,
        "status":status,
        "pred_csv":csv_path
    }

    function_summary[meta["Function"]]["total"] += 1
    application_summary[meta["Application"]]["total"] += 1
    key_map = {"healthy":"healthy","needs_attention":"attention","critical":"critical"}
    function_summary[meta["Function"]][key_map[status]] += 1
    application_summary[meta["Application"]][key_map[status]] += 1

    tft_models_store.setdefault(server,{})["cpu_current"] = tft
    tft_models_store.setdefault(server,{})["mem_current"] = tft

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

# per-server pickles
for server, info in server_level_info.items():
    with open(os.path.join(args.outdir,f"{server}.pkl"),"wb") as f:
        pickle.dump(info,f)

with open(os.path.join(args.outdir,"server_status_all.pkl"),"wb") as f:
    pickle.dump(server_level_info,f)

# JSON outputs for UI
pd.DataFrame.from_dict(func_pct,orient="index").reset_index().rename(columns={"index":"Function"}).to_json(
    os.path.join(args.outdir,"functions.json"),orient="records")
pd.DataFrame.from_dict(app_pct,orient="index").reset_index().rename(columns={"index":"Application"}).to_json(
    os.path.join(args.outdir,"applications.json"),orient="records")
pd.DataFrame.from_dict(server_level_info,orient="index").to_json(
    os.path.join(args.outdir,"servers_all.json"),orient="records")

print("✅ Done. TFT + ARIMA pipeline executed, outputs written to", args.outdir)
