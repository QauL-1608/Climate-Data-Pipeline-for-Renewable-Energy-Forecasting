import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Renewable Forecasting", layout="wide")

st.title("🔆 Renewable Energy Forecasting – Demo App")
st.caption("ETL ➜ features ➜ forecast ➜ visualization (synthetic data)")

proc_dir = Path("data/processed")
reports_dir = Path("reports")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Metrics")
    metrics_path = reports_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.json(metrics)
    else:
        st.info("Run training to see metrics (python scripts/forecast_pipeline.py --train).")

with col2:
    st.subheader("Forecast vs Actual")
    fc_path = proc_dir / "forecast.csv"
    if fc_path.exists():
        df = pd.read_csv(fc_path, parse_dates=["timestamp"])
        st.line_chart(df.set_index("timestamp")[["demand_mw","forecast_mw"]])
    else:
        st.info("No forecast found. Train first.")

st.divider()
st.subheader("Feature Snapshot")
mf = Path(proc_dir / "merged_features.csv")
if mf.exists():
    df = pd.read_csv(mf, parse_dates=["timestamp"]).tail(100)
    st.dataframe(df)
else:
    st.info("Run ETL first (python scripts/api_data_ingestion.py --run_all).")
