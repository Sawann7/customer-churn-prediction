"""
Unit tests for churn prediction, risk tiering, edge cases, and missing inputs.
"""

import pandas as pd
import pytest
from src.predict import (
    DEFAULT_RISK_THRESHOLDS,
    SAMPLE_PROFILES,
    classify_risk_tier,
    predict_churn,
    validate_customer_input,
)
from src.train import load_best_model


@pytest.fixture(scope="module")
def trained_pipeline():
    pipe = load_best_model()
    if pipe is None:
        from src.data_loader import generate_demo_dataset
        from src.train import train_and_compare_models
        demo_df = generate_demo_dataset(50)
        _, pipe, _, _ = train_and_compare_models(df=demo_df, cv_folds=2)
    return pipe


def test_classify_risk_tier():
    tier_low, color_low = classify_risk_tier(0.20, DEFAULT_RISK_THRESHOLDS)
    assert tier_low == "Low Risk"
    assert color_low == "#2ca02c"

    tier_mod, color_mod = classify_risk_tier(0.45, DEFAULT_RISK_THRESHOLDS)
    assert tier_mod == "Moderate Risk"
    assert color_mod == "#e67e22"

    tier_high, color_high = classify_risk_tier(0.75, DEFAULT_RISK_THRESHOLDS)
    assert tier_high == "High Risk"
    assert color_high == "#d9534f"


def test_predict_churn_single_customer(trained_pipeline):
    customer = SAMPLE_PROFILES["High Risk Profile"]
    res = predict_churn(customer, pipeline=trained_pipeline)

    assert "predicted_class" in res
    assert res["predicted_class"] in [0, 1]
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert 0.0 <= res["retention_probability"] <= 1.0
    assert abs(res["churn_probability"] + res["retention_probability"] - 1.0) < 1e-4
    assert res["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert len(res["disclaimer"]) > 10


def test_predict_churn_high_vs_low_profiles(trained_pipeline):
    high_res = predict_churn(SAMPLE_PROFILES["High Risk Profile"], pipeline=trained_pipeline)
    low_res = predict_churn(SAMPLE_PROFILES["Low Risk Profile"], pipeline=trained_pipeline)

    assert high_res["churn_probability"] > low_res["churn_probability"]
    assert high_res["churn_probability"] > 0.50
    assert low_res["churn_probability"] < 0.50


def test_predict_churn_missing_fields_and_unseen_categories(trained_pipeline):
    # Missing several optional fields, and an unseen payment method category
    incomplete_profile = {
        "tenure": 5,
        "MonthlyCharges": 85.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Cryptocurrency (Novel Unseen Category)",
        "InternetService": "Fiber optic",
    }
    res = predict_churn(incomplete_profile, pipeline=trained_pipeline)
    assert "churn_probability" in res
    assert 0.0 <= res["churn_probability"] <= 1.0


def test_validate_customer_input_negative_tenure():
    bad_input = {"tenure": -10, "MonthlyCharges": 50.0}
    clean_df = validate_customer_input(bad_input)
    assert clean_df.loc[0, "tenure"] == 0
