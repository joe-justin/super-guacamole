# model_pipeline_multi.py
import os
import argparse
import pickle
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", default="output")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--gru_epochs", type=int, default=50)
parser.add_argument("--gru_lr", type=float, default=1e-3)
parser.add_argument("--gru_seq_len", type=int, default=24)
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
    sdf[["Function","Application","Server"]] = sdf[["Function","Application","Server"]].ffill()
    numeric_cols = [c for c in sdf.columns if c not in ["Function","Application","Server"]]
    sdf[numeric_cols] = sdf[numeric_cols].ffill().bfill()
    sdf = sdf.reset_index().rename(columns={"index":"timestamp"})
    return sdf

device = torch.device("cpu")

class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(arr, seq_len):
    xs = []
    ys = []
    for i in range(len(arr) - seq_len):
        xs.append(arr[i:i+seq_len])
        ys.append(arr[i+seq_len])
    return np.array(xs), np.array(ys)

gru_models_store = {}   # server -> metric -> {state_dict, scaler, seq_len}
arima_models_store = {} # server -> metric -> fitted_object
server_level_info = {}  # server -> status, metadata, csv path
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

for server, group in tqdm(grouped, desc="Servers"):
    s = resample_server(group.copy())
    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}
    preds = {}
    preds_df = pd.DataFrame({"timestamp": pd.date_range(s["timestamp"].max()+timedelta(hours=1), periods=n_steps, freq="H")})
    for metric in ("cpu_current", "mem_current"):
        if metric not in s.columns:
            series = np.zeros(len(s))
        else:
            series = s[metric].astype(float).values
        if np.isnan(series).all():
            series = np.zeros(1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()
        seq_len = max(1, args.gru_seq_len)
        X, y = create_sequences(scaled, seq_len)
        if len(X) < 4:
            X = np.array([scaled[-seq_len:]])
            y = np.array([scaled[-1]])
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
        model = SimpleGRU(input_size=1, hidden_size=32)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.gru_lr)
        loss_fn = nn.MSELoss()
        model.train()
        for epoch in range(args.gru_epochs):
            opt.zero_grad()
            out = model(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()
        model.eval()
        preds_gru_scaled = []
        window = scaled[-seq_len:].tolist()
        for _ in range(n_steps):
            inp = torch.tensor(np.array(window[-seq_len:]), dtype=torch.float32).reshape(1, seq_len, 1).to(device)
            with torch.no_grad():
                p = model(inp).cpu().numpy().flatten()[0]
            preds_gru_scaled.append(float(p))
            window.append(float(p))
        try:
            arima_forecast = None
            if len(series) >= 10:
                arima = ARIMA(series, order=(5,1,0))
                arima_res = arima.fit()
                arima_models_store.setdefault(server, {})[metric] = arima_res
                arima_forecast = arima_res.forecast(steps=n_steps)
            else:
                arima_forecast = np.repeat(series[-1], n_steps)
        except Exception:
            arima_forecast = np.repeat(series[-1], n_steps)
        preds_gru = scaler.inverse_transform(np.array(preds_gru_scaled).reshape(-1,1)).flatten()
        preds_arima = np.array(arima_forecast).astype(float).flatten()
        preds_metric = np.maximum(preds_gru, preds_arima)  # keep max of two predictors per metric
        preds[metric] = {"gru": preds_gru, "arima": preds_arima, "combined": preds_metric}
        preds_df[f"{metric}_pred_gru"] = preds_gru
        preds_df[f"{metric}_pred_arima"] = preds_arima
        preds_df[f"{metric}_pred_combined"] = preds_metric
        gru_models_store.setdefault(server, {})[metric] = {
            "state_dict": model.state_dict(),
            "scaler": scaler,
            "seq_len": seq_len,
            "input_size": 1
        }
    csv_path = os.path.join(args.outdir, f"{server}_predicted.csv")
    preds_df.to_csv(csv_path, index=False)
    # decide status by taking the worst (max) predicted utilization across cpu and mem (avg over horizon)
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
    function_summary[meta["Function"]][ key_map[status] if status in key_map else status ] += 1
    application_summary[meta["Application"]][ key_map[status] if status in key_map else status ] += 1

with open(os.path.join(args.outdir, "GRU_Model.pkl"), "wb") as f:
    pickle.dump(gru_models_store, f)
with open(os.path.join(args.outdir, "ARIMA_Model.pkl"), "wb") as f:
    pickle.dump(arima_models_store, f)
func_pct = {}
for func, vals in function_summary.items():
    total = vals["total"] or 1
    func_pct[func] = {
        "healthy_pct": round(vals["healthy"]/total*100,2),
        "needs_attention_pct": round(vals["attention"]/total*100,2),
        "critical_pct": round(vals["critical"]/total*100,2),
        "total": total
    }
with open(os.path.join(args.outdir, "Function.pkl"), "wb") as f:
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
with open(os.path.join(args.outdir, "Application.pkl"), "wb") as f:
    pickle.dump(app_pct, f)
for server, info in server_level_info.items():
    fname = os.path.join(args.outdir, f"{server}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(info, f)
with open(os.path.join(args.outdir, "server_status_all.pkl"), "wb") as f:
    pickle.dump(server_level_info, f)
# produce JSON payloads for UI
# functions.json -> list of {function, healthy_pct, needs_attention_pct, critical_pct}
pd.DataFrame.from_dict(func_pct, orient="index").reset_index().rename(columns={"index":"Function"}).to_json(os.path.join(args.outdir,"functions.json"), orient="records")
pd.DataFrame.from_dict(app_pct, orient="index").reset_index().rename(columns={"index":"Application"}).to_json(os.path.join(args.outdir,"applications.json"), orient="records")
pd.DataFrame.from_dict(server_level_info, orient="index").to_json(os.path.join(args.outdir,"servers_all.json"), orient="records")
print("Done. Outputs in", args.outdir)
