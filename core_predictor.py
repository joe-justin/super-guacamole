# model_pipeline_multi.py (enhanced peak-aware version)
import os
import argparse
import pickle
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.arima.model import ARIMA

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", default="output")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--gru_epochs", type=int, default=60)
parser.add_argument("--gru_lr", type=float, default=1e-3)
parser.add_argument("--gru_seq_len", type=int, default=48)  # ↑ 2 days context
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
n_steps = args.forecast_days * 24

df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("timestamp")
grouped = df.groupby("Server")
servers = sorted(df["Server"].unique().tolist())

def resample_server(sdf):
    sdf = sdf.set_index("timestamp").sort_index()
    idx = pd.date_range(sdf.index.min(), sdf.index.max(), freq="H")
    sdf = sdf.reindex(idx)
    sdf[["Function", "Application", "Server"]] = sdf[["Function", "Application", "Server"]].ffill()
    numeric_cols = [c for c in sdf.columns if c not in ["Function","Application","Server"]]
    sdf[numeric_cols] = sdf[numeric_cols].ffill().bfill()
    sdf = sdf.reset_index().rename(columns={"index": "timestamp"})
    return sdf

device = torch.device("cpu")

class SimpleGRU(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(arrX, arrY, seq_len):
    xs, ys = [], []
    for i in range(len(arrY) - seq_len):
        xs.append(arrX[i:i+seq_len])
        ys.append(arrY[i+seq_len])
    return np.array(xs), np.array(ys)

gru_models_store, arima_models_store, server_level_info = {}, {}, {}
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

for server, group in tqdm(grouped, desc="Servers"):
    s = resample_server(group.copy())
    s["hour"] = s["timestamp"].dt.hour
    s["dayofweek"] = s["timestamp"].dt.dayofweek

    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}
    preds = {}
    preds_df = pd.DataFrame({"timestamp": pd.date_range(s["timestamp"].max()+timedelta(hours=1), periods=n_steps, freq="H")})

    for metric in ("cpu_current", "mem_current"):
        if metric not in s.columns:
            s[metric] = 0.0
        series = s[metric].astype(float).values
        if np.isnan(series).all():
            series[:] = 0.0

        scaler_y = RobustScaler()
        scaler_X = RobustScaler()
        feat = s[[metric, "hour", "dayofweek"]].values
        scaled_X = scaler_X.fit_transform(feat)
        scaled_y = scaler_y.fit_transform(series.reshape(-1,1)).flatten()

        seq_len = args.gru_seq_len
        X_seq, y_seq = create_sequences(scaled_X, scaled_y, seq_len)
        if len(X_seq) < 4:
            X_seq = np.repeat(scaled_X[-seq_len:][None, :, :], 4, axis=0)
            y_seq = np.repeat(scaled_y[-1], 4)

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1).to(device)

        model = SimpleGRU(input_size=3, hidden_size=64, num_layers=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.gru_lr)
        loss_fn = nn.MSELoss()

        for epoch in range(args.gru_epochs):
            model.train()
            opt.zero_grad()
            out = model(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()

        model.eval()
        preds_scaled = []
        last_window = scaled_X[-seq_len:].copy()
        last_y_window = scaled_y[-seq_len:].copy()

        for step in range(n_steps):
            future_hour = (s["hour"].iloc[-1] + (step+1)) % 24
            future_day = (s["dayofweek"].iloc[-1] + ((s["hour"].iloc[-1]+(step+1))//24)) % 7
            next_input = np.array([[last_y_window[-1], future_hour, future_day]])
            scaled_input = scaler_X.transform(next_input)
            seq_input = np.concatenate([last_window[-(seq_len-1):], scaled_input], axis=0)
            inp = torch.tensor(seq_input[None, :, :], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_scaled = model(inp).cpu().numpy().flatten()[0]
            preds_scaled.append(pred_scaled)
            last_y_window = np.append(last_y_window, pred_scaled)
            last_window = np.append(last_window, scaled_input, axis=0)

        preds_gru = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()

        try:
            if len(series) >= 10:
                arima = ARIMA(series, order=(5,1,0))
                arima_res = arima.fit()
                arima_models_store.setdefault(server, {})[metric] = arima_res
                preds_arima = arima_res.forecast(steps=n_steps)
            else:
                preds_arima = np.repeat(series[-1], n_steps)
        except Exception:
            preds_arima = np.repeat(series[-1], n_steps)

        preds_metric = np.maximum(preds_gru, preds_arima)
        preds[metric] = preds_metric
        preds_df[f"{metric}_predicted"] = preds_metric
        preds_df[f"{metric}_actual"] = series[-n_steps:] if len(series) >= n_steps else np.concatenate(
            [np.full(n_steps-len(series), np.nan), series])

        gru_models_store.setdefault(server, {})[metric] = {
            "state_dict": model.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_len": seq_len
        }

    csv_path = os.path.join(args.outdir, f"{server}_actual_predicted.csv")
    preds_df.to_csv(csv_path, index=False)

    avg_cpu = float(np.nanmean(preds["cpu_current"]))
    avg_mem = float(np.nanmean(preds["mem_current"]))
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

    key_map = {"healthy":"healthy","needs_attention":"attention","critical":"critical"}
    function_summary[meta["Function"]]["total"] += 1
    application_summary[meta["Application"]]["total"] += 1
    function_summary[meta["Function"]][key_map[status]] += 1
    application_summary[meta["Application"]][key_map[status]] += 1

with open(os.path.join(args.outdir, "GRU_Model.pkl"), "wb") as f:
    pickle.dump(gru_models_store, f)
with open(os.path.join(args.outdir, "ARIMA_Model.pkl"), "wb") as f:
    pickle.dump(arima_models_store, f)

func_pct = {
    func: {
        "healthy_pct": round(v["healthy"]/v["total"]*100,2),
        "needs_attention_pct": round(v["attention"]/v["total"]*100,2),
        "critical_pct": round(v["critical"]/v["total"]*100,2),
        "total": v["total"]
    } for func,v in function_summary.items()
}
app_pct = {
    app: {
        "healthy_pct": round(v["healthy"]/v["total"]*100,2),
        "needs_attention_pct": round(v["attention"]/v["total"]*100,2),
        "critical_pct": round(v["critical"]/v["total"]*100,2),
        "total": v["total"]
    } for app,v in application_summary.items()
}

with open(os.path.join(args.outdir, "Function.pkl"), "wb") as f:
    pickle.dump(func_pct, f)
with open(os.path.join(args.outdir, "Application.pkl"), "wb") as f:
    pickle.dump(app_pct, f)
pd.DataFrame.from_dict(func_pct, orient="index").reset_index().rename(columns={"index":"Function"}).to_json(os.path.join(args.outdir,"functions.json"), orient="records")
pd.DataFrame.from_dict(app_pct, orient="index").reset_index().rename(columns={"index":"Application"}).to_json(os.path.join(args.outdir,"applications.json"), orient="records")
pd.DataFrame.from_dict(server_level_info, orient="index").to_json(os.path.join(args.outdir,"servers_all.json"), orient="records")
with open(os.path.join(args.outdir, "server_status_all.pkl"), "wb") as f:
    pickle.dump(server_level_info, f)

print("✅ Done. Enhanced models saved in", args.outdir)
