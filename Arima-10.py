import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Try to import pmdarima
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except:
    PMDARIMA_AVAILABLE = False


# ============================================================
# 1) LOAD & PREPARE
# ============================================================

def load_and_prepare(filepath):
    df = pd.read_csv(filepath)

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
    df = df.sort_values('timestamp')

    expected = {'Function','Application','server','timestamp','cpu_current','mem_current'}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV missing expected columns. Found: {df.columns.tolist()}")

    df = df.set_index('timestamp')
    return df


# ============================================================
# 2) FIT & FORECAST SINGLE SERIES
# ============================================================

def fit_and_forecast_series(y, steps=2160, seasonal_period=144):
    y = y.asfreq("10T")

    if len(y.dropna()) < 50:
        mean_val = y.mean()
        future_idx = pd.date_range(y.index[-1] + pd.Timedelta("10 min"), periods=steps, freq="10T")
        return pd.Series(mean_val, index=future_idx), None

    # Auto ARIMA first attempt
    if PMDARIMA_AVAILABLE:
        try:
            auto = pm.auto_arima(
                y.dropna(),
                seasonal=True,
                m=seasonal_period,
                max_p=3, max_q=3, max_P=1, max_Q=1,
                stepwise=True, suppress_warnings=True,
                error_action="ignore"
            )
            order = auto.order
            P, D, Q, m = auto.seasonal_order
            seasonal_order = (P, D, Q, seasonal_period)

            model = SARIMAX(
                y.dropna(),
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)

            pred = res.get_forecast(steps=steps)
            return pred.predicted_mean, res

        except Exception as e:
            print("Auto ARIMA failed; fallback to simple SARIMAX.", e)

    # Fallback SARIMAX
    model = SARIMAX(
        y.dropna(),
        order=(1,0,1),
        seasonal_order=(1,0,1,seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    return pred.predicted_mean, res


# ============================================================
# 3) FORECAST PER SERVER (AGGREGATED)
# ============================================================

def forecast_per_server(df, steps=2160, seasonal_period=144,
                         forecast_dir="forecasts", plot_dir="plots"):

    os.makedirs(forecast_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Aggregate only per-server
    servers = df.groupby("server")

    results = {}

    for server, sdf in servers:
        print(f"\nProcessing server: {server}")

        # Aggregate CPU + Memory across all Functions / Applications
        sdf = sdf.resample("10T").mean()

        # Prepare forecast frame
        future_index = pd.date_range(
            sdf.index[-1] + pd.Timedelta("10 min"),
            periods=steps,
            freq="10T"
        )
        forecast_df = pd.DataFrame(index=future_index)

        # Forecast CPU
        cpu_pred, _ = fit_and_forecast_series(sdf["cpu_current"], steps, seasonal_period)
        forecast_df["cpu_current"] = cpu_pred

        # Forecast Memory
        mem_pred, _ = fit_and_forecast_series(sdf["mem_current"], steps, seasonal_period)
        forecast_df["mem_current"] = mem_pred

        forecast_df["server"] = server

        # Save forecast CSV
        out_path = os.path.join(forecast_dir, f"{server}_forecast.csv")
        forecast_df.to_csv(out_path, index_label="timestamp")
        print(f"Saved forecast → {out_path}")

        # --------------------------
        # 📊 PLOT GENERATION PER SERVER
        # --------------------------
        plt.figure(figsize=(12,5))

        # last 3 days historical
        hist_window = 3 * 24 * 6  # 3 days @ 10-min
        hist_cpu = sdf["cpu_current"].iloc[-hist_window:]
        hist_mem = sdf["mem_current"].iloc[-hist_window:]

        plt.plot(hist_cpu.index, hist_cpu.values, label="CPU - last 3 days")
        plt.plot(hist_mem.index, hist_mem.values, label="Memory - last 3 days")

        # forecast
        plt.plot(cpu_pred.index, cpu_pred.values, label="CPU Forecast (15 days)", linestyle="--")
        plt.plot(mem_pred.index, mem_pred.values, label="Memory Forecast (15 days)", linestyle="--")

        plt.title(f"Server {server} – CPU & Memory Forecast")
        plt.xlabel("Timestamp")
        plt.ylabel("Utilization")
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f"{server}_forecast.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot → {plot_path}")

        # store result
        results[server] = {
            "forecast_csv": out_path,
            "plot": plot_path,
            "forecast_df": forecast_df
        }

    return results


# ============================================================
# HOW TO RUN
# ============================================================

# df = load_and_prepare("yourfile.csv")
# results = forecast_per_server(df)
