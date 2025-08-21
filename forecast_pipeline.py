#!/usr/bin/env python3
import os, argparse, json
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_lagged_features(df, cfg):
    for lag in cfg["features"]["include_lag_hours"]:
        df[f"demand_lag_{lag}"] = df["demand_mw"].shift(lag)
    for w in cfg["features"]["rolling_windows"]:
        df[f"irr_roll_{w}"] = df["irradiance_wm2"].rolling(w, min_periods=1).mean()
        df[f"temp_roll_{w}"] = df["temp_c"].rolling(w, min_periods=1).mean()
    return df

def split_train_test(df, ratio=0.8):
    n = len(df)
    cut = int(n*ratio)
    return df.iloc[:cut], df.iloc[cut:]

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(1e-6, np.abs(y_true)))) * 100

def baseline_naive(last_value, length):
    return np.array([last_value]*length)

def train_sarimax(train_df, cfg):
    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels not available. Try linear model instead.")
    order = tuple(cfg["model"]["sarimax"]["order"])
    sorder= tuple(cfg["model"]["sarimax"]["seasonal_order"])
    model = SARIMAX(train_df["demand_mw"], order=order, seasonal_order=sorder, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def train_linear(train_df, features):
    X = train_df[features].fillna(method="bfill").fillna(method="ffill")
    y = train_df["demand_mw"].values
    reg = LinearRegression().fit(X, y)
    return reg

def main(args):
    cfg = load_config(args.config)
    proc_path = Path(cfg["paths"]["processed"]) / "merged_features.csv"
    df = pd.read_csv(proc_path, parse_dates=["timestamp"]).sort_values("timestamp")

    df = build_lagged_features(df, cfg)
    # Drop initial rows with NaNs from lags
    df = df.dropna().reset_index(drop=True)

    # Baseline forecast (naive last value)
    X_train, X_test = split_train_test(df, ratio=cfg["model"].get("train_ratio", 0.8))
    naive_pred = baseline_naive(X_train["demand_mw"].iloc[-1], len(X_test))

    # Choose model
    algo = cfg["model"]["algorithm"]
    if algo == "sarimax" and HAS_STATSMODELS:
        sarimax_res = train_sarimax(X_train, cfg)
        sarimax_forecast = sarimax_res.forecast(steps=len(X_test))
        model_name = "sarimax"
        model_pred = sarimax_forecast.values
    else:
        # Fallback to linear regression with features
        features = [c for c in df.columns if c not in ["timestamp","demand_mw"]]
        reg = train_linear(X_train, features)
        model_pred = reg.predict(X_test[features].fillna(method="bfill").fillna(method="ffill"))
        model_name = "linear_regression"

    # Metrics
    y_true = X_test["demand_mw"].values
    metrics = {
        "model": model_name,
        "MAE": float(mean_absolute_error(y_true, model_pred)),
        "RMSE": float(sqrt(mean_squared_error(y_true, model_pred))),
        "MAPE": float(mape(y_true, model_pred)),
        "Baseline_MAPE": float(mape(y_true, naive_pred)),
        "Improvement_vs_Baseline_pct": float( (mape(y_true, naive_pred) - mape(y_true, model_pred)) / max(1e-6, mape(y_true, naive_pred)) * 100 )
    }

    # Save metrics & forecast
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    out = X_test[["timestamp","demand_mw"]].copy()
    out["forecast_mw"] = model_pred
    out.to_csv(Path(cfg["paths"]["processed"]) / "forecast.csv", index=False)

    print("[OK] Metrics:", json.dumps(metrics, indent=2))
    print(f"[OK] Wrote forecast to {Path(cfg['paths']['processed']) / 'forecast.csv'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--train", action="store_true")
    p.add_argument("--forecast_horizon", type=int, default=48, help="For demo only; test set size is determined by split.")
    args = p.parse_args()
    main(args)
