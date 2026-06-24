"""
RetailPulse - Basic validation tests
=======================================
Lightweight smoke tests for the pipeline outputs. Run with: pytest tests/
(Per submission guidelines: "Basic tests / validation scripts appreciated")
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def _require(path):
    if not path.exists():
        pytest.skip(f"{path} not found - run `python run_pipeline.py` first")
    return path


class TestDataQuality:
    def test_clean_transactions_no_missing_keys(self):
        df = pd.read_csv(_require(PROCESSED / "transactions_clean.csv"))
        assert df["customer_id"].isna().sum() == 0
        assert df["product_id"].isna().sum() == 0
        assert df["unit_price"].isna().sum() == 0

    def test_no_duplicate_rows(self):
        df = pd.read_csv(_require(PROCESSED / "transactions_clean.csv"))
        assert df.duplicated().sum() == 0


class TestSegmentation:
    def test_segment_count_in_business_range(self):
        with open(_require(PROCESSED / "segmentation_metrics.json")) as f:
            metrics = json.load(f)
        assert 6 <= metrics["n_clusters"] <= 8, "F02 requires 6-8 meaningful segments"

    def test_every_customer_has_a_segment(self):
        df = pd.read_csv(_require(PROCESSED / "customer_segments.csv"))
        assert df["segment_name"].isna().sum() == 0


class TestForecasting:
    def test_mape_meets_business_target(self):
        with open(_require(PROCESSED / "forecast_metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["ensemble_mape"] <= 12.0, "F03 requires MAPE <= 12%"

    def test_forecast_has_30_days(self):
        df = pd.read_csv(_require(PROCESSED / "demand_forecast_30day.csv"))
        assert len(df) == 30

    def test_forecast_non_negative(self):
        df = pd.read_csv(_require(PROCESSED / "demand_forecast_30day.csv"))
        assert (df["forecasted_demand"] >= 0).all()


class TestChurn:
    def test_auc_meets_business_target(self):
        with open(_require(PROCESSED / "churn_metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["auc_roc"] >= 0.88, "F04 requires AUC-ROC >= 0.88"

    def test_precision_at_20_meets_target(self):
        with open(_require(PROCESSED / "churn_metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["precision_at_top_20pct"] >= 0.75, "F04 requires precision@top20% >= 0.75"

    def test_churn_probabilities_in_valid_range(self):
        df = pd.read_csv(_require(PROCESSED / "churn_scores.csv"))
        assert df["churn_probability"].between(0, 1).all()


class TestInventory:
    def test_every_product_has_a_recommendation(self):
        rec = pd.read_csv(_require(PROCESSED / "inventory_recommendations.csv"))
        products = pd.read_csv(_require(PROCESSED / "products_clean.csv"))
        assert set(rec["product_id"]) == set(products["product_id"])

    def test_recommended_order_qty_non_negative(self):
        rec = pd.read_csv(_require(PROCESSED / "inventory_recommendations.csv"))
        assert (rec["recommended_order_qty"] >= 0).all()

    def test_status_values_valid(self):
        rec = pd.read_csv(_require(PROCESSED / "inventory_recommendations.csv"))
        valid = {"Critical - Reorder Immediately", "Reorder Soon", "Healthy"}
        assert set(rec["status"].unique()).issubset(valid)
