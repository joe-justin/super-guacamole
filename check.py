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

# ------------------------- Arguments -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--outdir", default="output")
parser.add_argument("--forecast_days", type=int, default=30)
parser.add_argument("--gru_epochs", type=int, default=80)  # slightly longer training
parser.add_argument("--gru_lr", type=float, default=5e-4)
parser.add_argument("--gru_seq_len", type=int, default=48)  # longer lookback
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
n_steps = args.forecast_days * 24

# ------------------------- Load Input -------------------------
df = pd.read_csv(args.input, parse_dates=["timestamp"])
df.columns = [c.strip() for c in df.columns]
df = df.sort_values("timestamp")
grouped = df.groupby("Server")
servers = sorted(df["Server"].unique().tolist())

# ------------------------- Utilities -------------------------
def resample_server(sdf):
    sdf = sdf.set_index("timestamp").sort_index()
    idx = pd.date_range(sdf.index.min(), sdf.index.max(), freq="H")
    sdf = sdf.reindex(idx)
    sdf[["Function", "Application", "Server"]] = sdf[["Function", "Application", "Server"]].ffill()
    numeric_cols = [c for c in sdf.columns if c not in ["Function", "Application", "Server"]]
    sdf[numeric_cols] = sdf[numeric_cols].ffill().bfill()
    sdf = sdf.reset_index().rename(columns={"index": "timestamp"})
    return sdf

device = torch.device("cpu")

# ------------------------- Model Definition -------------------------
class PeakGRU(nn.Module):
    """Peak-sensitive GRU: deeper + dropout."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# ------------------------- Sequence Builder -------------------------
def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(y) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

# ------------------------- Containers -------------------------
gru_models_store = {}
arima_models_store = {}
server_level_info = {}
function_summary = defaultdict(lambda: {"healthy": 0, "attention": 0, "critical": 0, "total": 0})
application_summary = defaultdict(lambda: {"healthy": 0, "attention": 0, "critical": 0, "total": 0})

# ------------------------- Main Loop -------------------------
for server, group in tqdm(grouped, desc="Servers"):
    s = resample_server(group.copy())
    meta = {"Function": s["Function"].iloc[0], "Application": s["Application"].iloc[0]}

    preds = {}
    preds_df = pd.DataFrame({
        "timestamp": pd.date_range(s["timestamp"].max() + timedelta(hours=1), periods=n_steps, freq="H")
    })

    for metric in ("cpu_current", "mem_current"):
        if metric not in s.columns:
            continue

        series = s[metric].astype(float).values
        if np.isnan(series).all():
            series = np.zeros(1)

        # ----- Optional exogenous variables -----
        exog_cols = [c for c in s.columns if c not in ["Function", "Application", "Server", "timestamp", metric]]
        exog = s[exog_cols].astype(float).fillna(0).values
        scaler_y = MinMaxScaler((0, 1))
        y_scaled = scaler_y.fit_transform(series.reshape(-1, 1)).flatten()
        scaler_x = MinMaxScaler((0, 1))
        X_scaled = scaler_x.fit_transform(exog)

        seq_len = max(1, args.gru_seq_len)
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
        if len(X_seq) < 4:
            continue

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1).to(device)

        # ----- GRU Training -----
        model = PeakGRU(input_size=X_t.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=args.gru_lr)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(args.gru_epochs):
            opt.zero_grad()
            out = model(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()

        # ----- Forecast -----
        model.eval()
        preds_gru_scaled = []
        window_x = X_scaled[-seq_len:].copy()
        for _ in range(n_steps):
            inp = torch.tensor(window_x[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                p = model(inp).cpu().numpy().flatten()[0]
            preds_gru_scaled.append(p)
            # shift window (repeat last exog values)
            window_x = np.vstack([window_x, window_x[-1]])

        preds_gru = scaler_y.inverse_transform(np.array(preds_gru_scaled).reshape(-1, 1)).flatten()

        # ----- ARIMA Fallback -----
        try:
            if len(series) >= 20:
                arima = ARIMA(series, order=(3, 1, 3))
                arima_res = arima.fit()
                arima_forecast = arima_res.forecast(steps=n_steps)
                arima_models_store.setdefault(server, {})[metric] = arima_res
            else:
                arima_forecast = np.repeat(series[-1], n_steps)
        except Exception:
            arima_forecast = np.repeat(series[-1], n_steps)

        preds_arima = np.array(arima_forecast).astype(float)
        preds_metric = np.maximum(preds_gru, preds_arima)

        preds[metric] = preds_metric
        preds_df[f"{metric}_pred_combined"] = preds_metric

        # Store GRU model
        gru_models_store.setdefault(server, {})[metric] = {
            "state_dict": model.state_dict(),
            "scaler_y": scaler_y,
            "scaler_x": scaler_x,
            "seq_len": seq_len,
            "input_size": X_t.shape[-1]
        }

    # ------------------------- Combine actual + predicted -------------------------
    combined_df = s[["timestamp", "cpu_current", "mem_current"]].rename(columns={
        "cpu_current": "cpu_actual",
        "mem_current": "mem_actual"
    })

    preds_df = preds_df.rename(columns={
        "cpu_current_pred_combined": "cpu_predicted",
        "mem_current_pred_combined": "mem_predicted"
    })[["timestamp", "cpu_predicted", "mem_predicted"]]

    final_df = pd.concat([combined_df, preds_df], ignore_index=True)
    csv_path = os.path.join(args.outdir, f"{server}_actual_predicted.csv")
    final_df.to_csv(csv_path, index=False)

    # ------------------------- Status Calculation -------------------------
    avg_cpu = float(np.nanmean(preds.get("cpu_current", [0])))
    avg_mem = float(np.nanmean(preds.get("mem_current", [0])))
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
    key_map = {"healthy": "healthy", "needs_attention": "attention", "critical": "critical"}
    function_summary[meta["Function"]][key_map[status]] += 1
    application_summary[meta["Application"]][key_map[status]] += 1

# ------------------------- Save Models -------------------------
with open(os.path.join(args.outdir, "GRU_Model.pkl"), "wb") as f:
    pickle.dump(gru_models_store, f)
with open(os.path.join(args.outdir, "ARIMA_Model.pkl"), "wb") as f:
    pickle.dump(arima_models_store, f)

# ------------------------- Summaries -------------------------
func_pct = {}
for func, vals in function_summary.items():
    total = vals["total"] or 1
    func_pct[func] = {
        "healthy_pct": round(vals["healthy"]/total*100, 2),
        "needs_attention_pct": round(vals["attention"]/total*100, 2),
        "critical_pct": round(vals["critical"]/total*100, 2),
        "total": total
    }

app_pct = {}
for app, vals in application_summary.items():
    total = vals["total"] or 1
    app_pct[app] = {
        "healthy_pct": round(vals["healthy"]/total*100, 2),
        "needs_attention_pct": round(vals["attention"]/total*100, 2),
        "critical_pct": round(vals["critical"]/total*100, 2),
        "total": total
    }

with open(os.path.join(args.outdir, "Function.pkl"), "wb") as f:
    pickle.dump(func_pct, f)
with open(os.path.join(args.outdir, "Application.pkl"), "wb") as f:
    pickle.dump(app_pct, f)

for server, info in server_level_info.items():
    with open(os.path.join(args.outdir, f"{server}.pkl"), "wb") as f:
        pickle.dump(info, f)

with open(os.path.join(args.outdir, "server_status_all.pkl"), "wb") as f:
    pickle.dump(server_level_info, f)

# ------------------------- JSON for UI -------------------------
pd.DataFrame.from_dict(func_pct, orient="index").reset_index().rename(columns={"index": "Function"}).to_json(
    os.path.join(args.outdir, "functions.json"), orient="records")
pd.DataFrame.from_dict(app_pct, orient="index").reset_index().rename(columns={"index": "Application"}).to_json(
    os.path.join(args.outdir, "applications.json"), orient="records")
pd.DataFrame.from_dict(server_level_info, orient="index").to_json(
    os.path.join(args.outdir, "servers_all.json"), orient="records")

print("✅ Done. Outputs in", args.outdir)
