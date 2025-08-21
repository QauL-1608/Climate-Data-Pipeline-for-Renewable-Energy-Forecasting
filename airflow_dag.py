from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="renewable_etl_forecast",
    default_args=default_args,
    start_date=datetime(2025, 8, 1),
    schedule_interval="@daily",
    catchup=False,
    description="ETL + Forecast pipeline for renewable energy",
) as dag:

    etl = BashOperator(
        task_id="etl_ingestion",
        bash_command="python scripts/api_data_ingestion.py --run_all"
    )

    train = BashOperator(
        task_id="train_and_forecast",
        bash_command="python scripts/forecast_pipeline.py --train"
    )

    etl >> train
