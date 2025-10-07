# model_pipeline_multi.py
import os
import argparse
import pickle
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.arima.model import ARIMA

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", default="output")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--gru_epochs", type=int, default=60)
parser.add_argument("--gru_lr", type=float, default=1e-3)
parser.add_argument("--gru_seq_len", type=int, default=48)
parser.add_argument("--train_tcn", action="store_true", help="Also train a TCN model and save TCN_Model.pkl")
parser.add_argument("--tcn_epochs", type=int, default=60)
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

def resample_server(sdf: pd.DataFrame) -> pd.DataFrame:
    sdf = sdf.set_index("timestamp").sort_index()
    idx = pd.date_range(sdf.index.min(), sdf.index.max(), freq="H")
    sdf = sdf.reindex(idx)
    sdf[["Function","Application","Server"]] = sdf[["Function","Application","Server"]].ffill()
    numeric_cols = [c for c in sdf.columns if c not in ["Function","Application","Server"]]
    sdf[numeric_cols] = sdf[numeric_cols].ffill().bfill()
    sdf = sdf.reset_index().rename(columns={"index":"timestamp"})
    return sdf

device = torch.device(args.device)

# -----------------------
# Models
# -----------------------
class MultiGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.downsample = (in_channels != out_channels) and nn.Conv1d(in_channels, out_channels, 1) or None
    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        if self.downsample:
            x = self.downsample(x)
        return out + x

class SimpleTCN(nn.Module):
    def __init__(self, input_size, channels=[32, 32, 64], kernel_size=3):
        super().__init__()
        layers = []
        in_ch = input_size
        dilation = 1
        for ch in channels:
            layers.append(TCNBlock(in_ch, ch, kernel_size, dilation))
            in_ch = ch
            dilation *= 2
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, 1)
    def forward(self, x):
        x = x.transpose(1,2)
        y = self.net(x)
        y = y[:, :, -1]
        return self.fc(y)

# -----------------------
# Helpers
# -----------------------
def create_sequences_X_y(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(y) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    if len(xs) == 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))
    return np.array(xs), np.array(ys)

def safe_array(arr, length, fill=np.nan):
    arr = np.asarray(arr)
    if len(arr) >= length:
        return arr[-length:]
    pad = np.full(length - len(arr), fill)
    return np.concatenate([pad, arr])

# -----------------------
# Pipeline storage
# -----------------------
gru_models_store = {}
tcn_models_store = {}
arima_models_store = {}
server_level_info = {}
function_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})
application_summary = defaultdict(lambda: {"healthy":0,"attention":0,"critical":0,"total":0})

# Exogenous feature list (use columns present in feed)
EXOG_FEATURES = [
    "cpu_min","cpu_max",
    "mem_min","mem_max",
    "nwin_current","nwin_min","nwin_max",
    "nwou_current","nwou_min","nwou_max"
]

# -----------------------
# Main loop per server
# -----------------------
for server, group in tqdm(grouped, desc="Servers"):
    s = resample_server(group.copy())
    for c in EXOG_FEATURES:
        if c not in s.columns:
            s[c] = 0.0
    s["hour"] = s["timestamp"].dt.hour
    s["dayofweek"] = s["timestamp"].dt.dayofweek

    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}
    preds = {}
    future_index = pd.date_range(s["timestamp"].max() + timedelta(hours=1), periods=n_steps, freq="H")
    preds_df = pd.DataFrame({"timestamp": future_index})

    for metric in ("cpu_current", "mem_current"):
        if metric not in s.columns:
            preds[metric] = {"gru": np.repeat(0.0, n_steps), "arima": np.repeat(0.0, n_steps), "combined": np.repeat(0.0, n_steps)}
            continue

        series = s[metric].astype(float).values
        if np.isnan(series).all():
            series = np.zeros(1)

        feature_cols = [metric] + EXOG_FEATURES + ["hour", "dayofweek"]
        X_full = s[feature_cols].fillna(0.0).values.astype(float)

        scaler_X = RobustScaler()
        try:
            X_scaled = scaler_X.fit_transform(X_full)
        except Exception:
            X_scaled = X_full

        scaler_y = RobustScaler()
        try:
            y_scaled = scaler_y.fit_transform(series.reshape(-1,1)).flatten()
        except Exception:
            y_scaled = series.copy()

        seq_len = max(1, args.gru_seq_len)
        X_seq, y_seq = create_sequences_X_y(X_scaled, y_scaled, seq_len)
        if len(X_seq) < 4:
            lastX = safe_array(X_scaled, seq_len, fill=0.0)
            lastY = safe_array(y_scaled, seq_len, fill=0.0)
            X_seq = np.repeat(lastX[None,:,:], 4, axis=0)
            y_seq = np.repeat(lastY[-1], 4)

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1).to(device)

        input_size = X_scaled.shape[1]
        model_gru = MultiGRU(input_size=input_size, hidden_size=64, num_layers=2).to(device)
        opt = torch.optim.Adam(model_gru.parameters(), lr=args.gru_lr)
        loss_fn = nn.MSELoss()

        model_gru.train()
        for epoch in range(args.gru_epochs):
            opt.zero_grad()
            out = model_gru(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()

        model_gru.eval()
        last_window = X_scaled[-seq_len:].copy()
        preds_gru_scaled = []
        for step in range(n_steps):
            last_hour = int(s["hour"].iloc[-1])
            last_day = int(s["dayofweek"].iloc[-1])
            future_hour = (last_hour + step + 1) % 24
            future_day = (last_day + ((last_hour + step + 1)//24)) % 7
            feat = last_window[-1].copy()
            feat[-2] = future_hour
            feat[-1] = future_day
            try:
                feat_scaled = scaler_X.transform(feat.reshape(1,-1)).reshape(-1)
            except Exception:
                feat_scaled = feat
            seq_input = np.vstack([last_window[1:], feat_scaled]) if seq_len>1 else feat_scaled[None,:]
            inp = torch.tensor(seq_input[None,:,:], dtype=torch.float32).to(device)
            with torch.no_grad():
                p = model_gru(inp).cpu().numpy().flatten()[0]
            preds_gru_scaled.append(float(p))
            last_window = np.vstack([last_window, feat_scaled])
            if len(last_window) > seq_len:
                last_window = last_window[-seq_len:]

        try:
            preds_gru = scaler_y.inverse_transform(np.array(preds_gru_scaled).reshape(-1,1)).flatten()
        except Exception:
            preds_gru = np.array(preds_gru_scaled).astype(float)

        try:
            if len(series) >= 20:
                arima = ARIMA(series, order=(5,1,0))
                arima_res = arima.fit()
                arima_models_store.setdefault(server, {})[metric] = arima_res
                preds_arima = np.array(arima_res.forecast(steps=n_steps)).astype(float).flatten()
            else:
                preds_arima = np.repeat(series[-1], n_steps)
        except Exception:
            preds_arima = np.repeat(series[-1], n_steps)

        preds_combined = np.maximum(preds_gru, preds_arima)
        preds[metric] = {"gru": preds_gru, "arima": preds_arima, "combined": preds_combined}

        if metric == "cpu_current":
            preds_df["cpu_predicted"] = preds_combined
        else:
            preds_df["mem_predicted"] = preds_combined

        gru_models_store.setdefault(server, {})[metric] = {
            "state_dict": model_gru.state_dict(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_len": seq_len,
            "input_size": input_size
        }

        if args.train_tcn:
            try:
                tcn = SimpleTCN(input_size=input_size, channels=[32,32,64], kernel_size=3).to(device)
                opt_t = torch.optim.Adam(tcn.parameters(), lr=1e-3)
                X_t_tcn = torch.tensor(X_seq, dtype=torch.float32).to(device)
                y_t_tcn = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1).to(device)
                for e in range(args.tcn_epochs):
                    tcn.train()
                    opt_t.zero_grad()
                    out_t = tcn(X_t_tcn)
                    loss_t = loss_fn(out_t, y_t_tcn)
                    loss_t.backward()
                    opt_t.step()
                tcn.eval()
                last_window_tcn = X_scaled[-seq_len:].copy()
                preds_tcn_scaled = []
                for step in range(n_steps):
                    last_hour = int(s["hour"].iloc[-1])
                    last_day = int(s["dayofweek"].iloc[-1])
                    future_hour = (last_hour + step + 1) % 24
                    future_day = (last_day + ((last_hour + step + 1)//24)) % 7
                    feat = last_window_tcn[-1].copy()
                    feat[-2] = future_hour
                    feat[-1] = future_day
                    try:
                        feat_scaled = scaler_X.transform(feat.reshape(1,-1)).reshape(-1)
                    except Exception:
                        feat_scaled = feat
                    seq_input = np.vstack([last_window_tcn[1:], feat_scaled]) if seq_len>1 else feat_scaled[None,:]
                    inp = torch.tensor(seq_input[None,:,:], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        p_t = tcn(inp).cpu().numpy().flatten()[0]
                    preds_tcn_scaled.append(float(p_t))
                    last_window_tcn = np.vstack([last_window_tcn, feat_scaled])
                    if len(last_window_tcn) > seq_len:
                        last_window_tcn = last_window_tcn[-seq_len:]
                try:
                    preds_tcn = scaler_y.inverse_transform(np.array(preds_tcn_scaled).reshape(-1,1)).flatten()
                except Exception:
                    preds_tcn = np.array(preds_tcn_scaled)
                preds[metric]["tcn"] = preds_tcn
                preds_combined = np.maximum(preds_combined, preds_tcn)
                if metric == "cpu_current":
                    preds_df["cpu_predicted"] = preds_combined
                else:
                    preds_df["mem_predicted"] = preds_combined
                tcn_models_store.setdefault(server, {})[metric] = {
                    "state_dict": tcn.state_dict(),
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y,
                    "seq_len": seq_len,
                    "input_size": input_size
                }
            except Exception:
                pass

    # Build final CSV
    hist_df = s[["timestamp", "cpu_current", "mem_current"]].copy().rename(columns={
        "cpu_current": "cpu_actual",
        "mem_current": "mem_actual"
    })
    future_df = pd.DataFrame({"timestamp": preds_df["timestamp"]})
    future_df["cpu_predicted"] = preds_df.get("cpu_predicted", np.repeat(np.nan, n_steps)).astype(float)
    future_df["mem_predicted"] = preds_df.get("mem_predicted", np.repeat(np.nan, n_steps)).astype(float)
    future_df["cpu_actual"] = np.nan
    future_df["mem_actual"] = np.nan

    final_df = pd.concat([hist_df, future_df], ignore_index=True, sort=False)[
        ["timestamp", "cpu_actual", "mem_actual", "cpu_predicted", "mem_predicted"]
    ]

    csv_path = os.path.join(args.outdir, f"{server}_actual_predicted.csv")
    final_df.to_csv(csv_path, index=False)

    avg_cpu = float(np.nanmean(preds.get("cpu_current", {}).get("combined", np.array([0.0]))))
    avg_mem = float(np.nanmean(preds.get("mem_current", {}).get("combined", np.array([0.0]))))
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
with open(os.path.join(args.outdir, "GRU_Model.pkl"), "wb") as f:
    pickle.dump(gru_models_store, f)

if args.train_tcn:
    with open(os.path.join(args.outdir, "TCN_Model.pkl"), "wb") as f:
        pickle.dump(tcn_models_store, f)

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
    with open(os.path.join(args.outdir, f"{server}.pkl"), "wb") as f:
        pickle.dump(info, f)

with open(os.path.join(args.outdir, "server_status_all.pkl"), "wb") as f:
    pickle.dump(server_level_info, f)

pd.DataFrame.from_dict(func_pct, orient="index").reset_index().rename(columns={"index": "Function"}).to_json(
    os.path.join(args.outdir, "functions.json"), orient="records")
pd.DataFrame.from_dict(app_pct, orient="index").reset_index().rename(columns={"index": "Application"}).to_json(
    os.path.join(args.outdir, "applications.json"), orient="records")
pd.DataFrame.from_dict(server_level_info, orient="index").to_json(
    os.path.join(args.outdir, "servers_all.json"), orient="records")

print("✅ Done. Enhanced models and outputs written to", args.outdir)
