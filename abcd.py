
# pip install pandas numpy scikit-learn torch statsmodels tqdm
# python model_pipeline.py --input metrics.csv --outdir output


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

df = pd.read_csv(args.input, parse_dates=["timestamp"], dayfirst=False)
df = df.rename(columns={c: c.strip() for c in df.columns})
df = df.sort_values("timestamp")
df = df.loc[:, df.columns.str.strip()]

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

server_series = {}
meta = {}
for server, group in grouped:
    s = resample_server(group.copy())
    server_series[server] = s
    meta[server] = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}

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

gru_models_store = {}
arima_models_store = {}
server_status = {}
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

for server in tqdm(servers, desc="Servers"):
    s = server_series[server].copy()
    series = s["cpu_current"].astype(float).values
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
            arima_forecast = arima_res.forecast(steps=n_steps)
            arima_models_store[server] = {"model": arima_res}
        else:
            arima_forecast = np.repeat(series[-1], n_steps)
    except Exception:
        arima_forecast = np.repeat(series[-1], n_steps)

    preds_gru = scaler.inverse_transform(np.array(preds_gru_scaled).reshape(-1,1)).flatten()
    preds_arima = np.array(arima_forecast).astype(float).flatten()

    combined_pred = (preds_gru + preds_arima) / 2.0

    last_ts = s["timestamp"].max()
    future_idx = pd.date_range(last_ts + timedelta(hours=1), periods=n_steps, freq="H")
    pred_df = pd.DataFrame({"timestamp": future_idx, "cpu_pred_gru": preds_gru, "cpu_pred_arima": preds_arima, "cpu_pred_combined": combined_pred})
    csv_path = os.path.join(args.outdir, f"{server}_predicted.csv")
    pred_df.to_csv(csv_path, index=False)

    avg_pred = float(np.nanmean(combined_pred))
    if avg_pred < 50:
        status = "healthy"
    elif avg_pred <= 80:
        status = "needs_attention"
    else:
        status = "critical"

    server_status[server] = {"server": server, "Function": meta[server]["Function"], "Application": meta[server]["Application"], "avg_pred": avg_pred, "status": status, "pred_csv": csv_path}
    function_summary[meta[server]["Function"]]["total"] += 1
    application_summary[meta[server]["Application"]]["total"] += 1
    function_summary[meta[server]["Function"]][ "healthy" if status=="healthy" else ("attention" if status=="needs_attention" else "critical")] += 1
    application_summary[meta[server]["Application"]][ "healthy" if status=="healthy" else ("attention" if status=="needs_attention" else "critical")] += 1

    gru_models_store[server] = {"state_dict": model.state_dict(), "scaler": scaler, "seq_len": seq_len, "input_size":1}
    arima_models_store[server] = {"fitted": None}

with open(os.path.join(args.outdir, "GRU_Model.pkl"), "wb") as f:
    pickle.dump(gru_models_store, f)

with open(os.path.join(args.outdir, "ARIMA_Model.pkl"), "wb") as f:
    pickle.dump(arima_models_store, f)

func_pct = {}
for func, vals in function_summary.items():
    total = vals["total"] or 1
    func_pct[func] = {"healthy_pct": round(vals["healthy"]/total*100,2), "needs_attention_pct": round(vals["attention"]/total*100,2), "critical_pct": round(vals["critical"]/total*100,2)}
with open(os.path.join(args.outdir, "Function.pkl"), "wb") as f:
    pickle.dump(func_pct, f)

app_pct = {}
for app, vals in application_summary.items():
    total = vals["total"] or 1
    app_pct[app] = {"healthy_pct": round(vals["healthy"]/total*100,2), "needs_attention_pct": round(vals["attention"]/total*100,2), "critical_pct": round(vals["critical"]/total*100,2)}
with open(os.path.join(args.outdir, "Application.pkl"), "wb") as f:
    pickle.dump(app_pct, f)

for server, info in server_status.items():
    fname = os.path.join(args.outdir, f"{server}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(info, f)

with open(os.path.join(args.outdir, "server_status_all.pkl"), "wb") as f:
    pickle.dump(server_status, f)

print("Done. Outputs written to:", args.outdir)
