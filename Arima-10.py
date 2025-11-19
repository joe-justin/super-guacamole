# forecast_10min_arima.py
# Place your CSV in the same folder (or edit the path) and run this script.
# Expects columns: Function, Application, server, timestamp, cpu_current, mem_current

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# try to use pmdarima for automatic order selection if available
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

def load_and_prepare(filepath):
    """
    Loads CSV with expected columns and returns DataFrame indexed by timestamp.
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True, utc=False)
    df = df.sort_values('timestamp')
    expected = {'Function','Application','server','timestamp','cpu_current','mem_current'}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV missing expected columns. Found columns: {df.columns.tolist()}")
    df = df.set_index('timestamp')
    return df

def fit_and_forecast_series(y, steps=2160, seasonal_period=144, use_auto=True):
    """
    Fit ARIMA-like model and return forecast series (mean) for `steps`.
    seasonal_period default 144 -> daily seasonality at 10-min frequency.
    """
    # ensure freq
    y = y.asfreq('10T')
    if len(y.dropna()) < 50:
        mean_val = y.mean()
        index = pd.date_range(start=y.index[-1] + pd.Timedelta(minutes=10), periods=steps, freq='10T')
        return pd.Series(mean_val, index=index), None

    if PMDARIMA_AVAILABLE and use_auto:
        try:
            model_auto = pm.auto_arima(y.dropna(), seasonal=True, m=seasonal_period,
                                      max_p=3, max_q=3, max_P=1, max_Q=1, max_order=5,
                                      stepwise=True, suppress_warnings=True, error_action='ignore')
            order = model_auto.order
            seasonal_order = model_auto.seasonal_order
            sarima_seasonal = (seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_period)
            model = SARIMAX(y.dropna(), order=order, seasonal_order=sarima_seasonal,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=steps)
            return pred.predicted_mean, res
        except Exception as e:
            print("auto_arima failed; falling back to default SARIMAX. Error:", e)

    # fallback simple SARIMAX
    order = (1,0,1)
    seasonal_order = (1,0,1, seasonal_period)
    model = SARIMAX(y.dropna(), order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    return pred.predicted_mean, res

def forecast_per_group(df, group_cols=('Function','Application','server'),
                       value_cols=('cpu_current','mem_current'),
                       steps=2160, seasonal_period=144, top_n_groups=None, out_dir='forecasts'):
    """
    Forecast value_cols for each unique combination of group_cols.
    Saves CSV per group into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    grouped = df.groupby(list(group_cols))
    group_sizes = grouped.size().sort_values(ascending=False)
    if top_n_groups is not None:
        group_sizes = group_sizes.iloc[:top_n_groups]

    results = {}
    for grp_key in group_sizes.index:
        grp_df = grouped.get_group(grp_key).copy()
        grp_df = grp_df.resample('10T').mean()  # regularize to 10-minute bins
        forecast_index = pd.date_range(start=grp_df.index[-1] + pd.Timedelta(minutes=10), periods=steps, freq='10T')
        forecast_df = pd.DataFrame(index=forecast_index)

        for col in value_cols:
            series = grp_df[col]
            pred_mean, model_fit = fit_and_forecast_series(series, steps=steps, seasonal_period=seasonal_period)
            forecast_df[col] = pred_mean

        for i, colname in enumerate(group_cols):
            forecast_df[colname] = grp_key[i]

        fname = f"group_{'_'.join([str(x) for x in grp_key])}.csv".replace(' ','_')
        out_path = os.path.join(out_dir, fname)
        forecast_df.to_csv(out_path, index_label='timestamp')
        print("Saved forecast for group", grp_key, "->", out_path)
        results[grp_key] = {'forecast': forecast_df, 'file': out_path}
    return results

# ---------------------- How to run on your CSV ----------------------
# Example usage:
# df = load_and_prepare('data.csv')
# results = forecast_per_group(df, top_n_groups=None, steps=2160, seasonal_period=144, out_dir='forecasts')
#
# The above will create forecast CSV files in the 'forecasts' folder.

# ---------------------- Quick demo (optional) ----------------------
# If you'd like to quickly test the script without your CSV,
# create a short simulated dataset (uncomment to run).
#
# import numpy as np
# rng = pd.date_range('2025-09-01', periods=1440, freq='10T')  # smaller demo dataset
# cpu = 10 + 3*np.sin(2*np.pi*(rng.hour*60 + rng.minute)/(24*60)) + np.random.normal(scale=0.5, size=len(rng))
# mem = 40 + 1.5*np.sin(2*np.pi*(rng.hour*60 + rng.minute)/(24*60)) + np.random.normal(scale=0.2, size=len(rng))
# demo = pd.DataFrame({'Function':'F1','Application':'AppA','server':'srv-01','cpu_current':cpu,'mem_current':mem}, index=rng)
# demo.index.name='timestamp'
# results = forecast_per_group(demo, top_n_groups=1, steps=288, seasonal_period=144, out_dir='forecasts_demo')
# print('Demo forecast saved to', results[list(results.keys())[0]]['file'])
