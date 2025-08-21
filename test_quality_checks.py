import json, os

def test_quality_report_exists():
    assert os.path.exists("reports/quality_report.json"), "Run ETL before tests."
