"""
Inference and prediction module for Telco Customer Churn.
Validates input features, executes the trained pipeline, calculates churn probabilities,
and assigns operational risk tiers.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.train import BEST_MODEL_PATH, load_best_model

DEFAULT_RISK_THRESHOLDS = {
    "low_moderate": 0.35,
    "moderate_high": 0.60,
}

SAMPLE_PROFILES = {
    "High Risk Profile": {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00,
    },
    "Low Risk Profile": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.20,
        "TotalCharges": 3852.00,
    },
    "Average / Moderate Profile": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 75.00,
        "TotalCharges": 1800.00,
    },
}


def validate_customer_input(data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """
    Validates and normalizes input customer features into a clean single-row DataFrame.
    Gracefully handles missing or extra columns.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Customer data must be a dictionary or pandas DataFrame.")

    # Convert numeric fields safely
    for num_col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
        else:
            df[num_col] = np.nan

    # Ensure tenure is non-negative
    if not df["tenure"].isna().all():
        df["tenure"] = df["tenure"].clip(lower=0)

    # Ensure string types for categorical fields
    cat_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        else:
            df[c] = "Unknown"

    return df


def classify_risk_tier(
    probability: float,
    thresholds: Optional[Dict[str, float]] = None
) -> Tuple[str, str]:
    """
    Assigns an operational risk tier and color badge based on estimated churn probability.
    """
    if thresholds is None:
        thresholds = DEFAULT_RISK_THRESHOLDS

    low_mod = thresholds.get("low_moderate", 0.35)
    mod_high = thresholds.get("moderate_high", 0.60)

    if probability < low_mod:
        return "Low Risk", "#2ca02c"
    elif probability < mod_high:
        return "Moderate Risk", "#e67e22"
    else:
        return "High Risk", "#d9534f"


def predict_churn(
    customer_input: Union[Dict[str, Any], pd.DataFrame],
    pipeline: Optional[Any] = None,
    thresholds: Optional[Dict[str, float]] = None,
    decision_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Generates churn prediction and probability for a customer profile.

    Returns:
        Dict with predicted_class, probability, risk_tier, color, and operational disclaimer.
    """
    if pipeline is None:
        pipeline = load_best_model()
        if pipeline is None:
            raise FileNotFoundError(
                f"No trained model found at '{BEST_MODEL_PATH}'. Please train the model first."
            )

    df_clean = validate_customer_input(customer_input)

    # Probability estimation
    if hasattr(pipeline, "predict_proba"):
        prob = float(pipeline.predict_proba(df_clean)[0, 1])
    elif hasattr(pipeline, "decision_function"):
        score = float(pipeline.decision_function(df_clean)[0])
        prob = float(1.0 / (1.0 + np.exp(-score)))
    else:
        prob = float(pipeline.predict(df_clean)[0])

    pred_class = int(prob >= decision_threshold)
    pred_label = "Likely to Churn" if pred_class == 1 else "Likely to Stay"
    risk_tier, risk_color = classify_risk_tier(prob, thresholds)

    return {
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "churn_probability": prob,
        "retention_probability": 1.0 - prob,
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "decision_threshold": decision_threshold,
        "disclaimer": (
            "This prediction is an algorithmic estimate based on statistical patterns in historical data. "
            "It is not a certainty or guarantee of customer behavior. Retention actions should be evaluated "
            "alongside customer service history and business policy."
        ),
        "input_data": df_clean.to_dict(orient="records")[0],
    }
