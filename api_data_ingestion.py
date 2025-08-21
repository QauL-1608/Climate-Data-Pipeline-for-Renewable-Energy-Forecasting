#!/usr/bin/env python3
import os, argparse, sys, json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(paths):
    for p in paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)

def read_weather(cfg):
    raw_path = Path(cfg["paths"]["raw"]) / "weather_timeseries.csv"
    if cfg["use_synthetic_weather"]:
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing synthetic weather at {raw_path}")
        return pd.read_csv(raw_path, parse_dates=["timestamp"])
    else:
        # Placeholder for real API call
        # Use os.environ for WEATHER_API_URL and WEATHER_API_TOKEN
        raise NotImplementedError("Real API ingestion not implemented in sample. Use synthetic.")

def read_demand(cfg):
    if cfg["use_synthetic_demand"]:
        raw_path = Path(cfg["paths"]["raw"]) / "demand_timeseries.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing synthetic demand at {raw_path}")
        return pd.read_csv(raw_path, parse_dates=["timestamp"])
    else:
        src = cfg["demand_source"]["path"]
        return pd.read_csv(src, parse_dates=["timestamp"])

def quality_checks(df, ts_col="timestamp"):
    issues = {}
    issues["missing_timestamps"] = df[ts_col].isna().sum()
    issues["duplicate_timestamps"] = df[ts_col].duplicated().sum()
    # Simple outlier flags using z-score
    num_cols = df.select_dtypes(include=[np.number]).columns
    z = (df[num_cols] - df[num_cols].mean())/df[num_cols].std(ddof=0)
    outlier_counts = (z.abs() > 3).sum().to_dict()
    issues["zscore_outliers"] = {k:int(v) for k,v in outlier_counts.items()}
    return issues

def clean_weather(df):
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    # Impute small missing values if any
    for col in ["irradiance_wm2", "temp_c", "wind_ms"]:
        if col in df.columns:
            df[col] = df[col].interpolate(limit_direction="both")
    # Normalize units examples (already in desired units)
    return df

def enrich_calendar(df, ts_col="timestamp"):
    dt = pd.to_datetime(df[ts_col])
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"]>=5).astype(int)
    return df

def main(args):
    cfg = load_config(args.config)
    ensure_dirs(cfg["paths"])

    weather = read_weather(cfg)
    demand = read_demand(cfg)

    qc_weather = quality_checks(weather)
    qc_demand  = quality_checks(demand)

    weather = clean_weather(weather)
    demand  = demand.sort_values("timestamp").drop_duplicates("timestamp")

    # Join on timestamp (hourly alignment assumed)
    df = pd.merge_asof(
        demand.sort_values("timestamp"),
        weather.sort_values("timestamp"),
        on="timestamp"
    )

    df = enrich_calendar(df)

    proc_path = Path(cfg["paths"]["processed"]) / "merged_features.csv"
    df.to_csv(proc_path, index=False)

    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "quality_report.json", "w") as f:
        json.dump({"weather": qc_weather, "demand": qc_demand}, f, indent=2)

    print(f"[OK] Wrote processed features → {proc_path}")
    print(f"[OK] Quality report → {reports_dir / 'quality_report.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()
    main(args)
