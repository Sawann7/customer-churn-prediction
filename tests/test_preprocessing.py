"""
Unit tests for data loader and preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    clean_raw_data,
    generate_demo_dataset,
    validate_raw_data,
)
from src.preprocessing import (
    build_preprocessor,
    get_feature_names,
    identify_feature_types,
    prepare_train_test_data,
    separate_features_target,
)


def test_validate_raw_data_valid():
    df = generate_demo_dataset(50)
    is_valid, errors = validate_raw_data(df)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_raw_data_empty():
    empty_df = pd.DataFrame()
    is_valid, errors = validate_raw_data(empty_df)
    assert is_valid is False
    assert any("empty" in e.lower() for e in errors)


def test_validate_raw_data_missing_target():
    df = generate_demo_dataset(20).drop(columns=["Churn"])
    is_valid, errors = validate_raw_data(df)
    assert is_valid is False
    assert any("churn" in e.lower() for e in errors)


def test_clean_raw_data_total_charges_spaces():
    raw_df = pd.DataFrame({
        "customerID": ["1", "2"],
        "TotalCharges": ["  ", "100.50"],
        "tenure": ["0", "5"],
        "MonthlyCharges": ["20.0", "50.0"],
        "Churn": ["No", "Yes"],
    })
    cleaned = clean_raw_data(raw_df)
    assert pd.isna(cleaned.loc[0, "TotalCharges"])
    assert cleaned.loc[1, "TotalCharges"] == 100.50
    assert cleaned.loc[0, "tenure"] == 0


def test_separate_features_target():
    df = generate_demo_dataset(40)
    X, y = separate_features_target(df)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert y is not None
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y) == 40


def test_build_preprocessor_transformation():
    df = generate_demo_dataset(60)
    X, y = separate_features_target(df)
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    assert X_trans.shape[0] == 60
    assert X_trans.shape[1] > 20
    assert not np.isnan(X_trans).any()

    feature_names = get_feature_names(preprocessor)
    assert len(feature_names) == X_trans.shape[1]
    assert "tenure" in feature_names


def test_prepare_train_test_split_stratification():
    df = generate_demo_dataset(100, random_state=42)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df, test_size=0.25, random_state=42)

    assert len(X_train) == 75
    assert len(X_test) == 25
    # Stratification check: Churn rate should be approximately identical
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.05
