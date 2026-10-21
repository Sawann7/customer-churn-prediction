"""
Data loading and raw data validation module for Telco Customer Churn.
Handles local file retrieval, optional remote download, schema verification,
and safe type casting.
"""

import os
import urllib.request
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Telco-Customer-Churn.csv")
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"


def validate_raw_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates the structure, required fields, and basic integrity of the raw dataset.

    Returns:
        (is_valid, error_messages)
    """
    errors: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return False, ["Input is not a valid pandas DataFrame."]

    if df.empty:
        return False, ["Dataset is completely empty (0 rows)."]

    if TARGET_COLUMN not in df.columns:
        errors.append(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing expected columns: {', '.join(missing_cols)}")

    if TARGET_COLUMN in df.columns:
        unique_targets = set(df[TARGET_COLUMN].dropna().unique())
        valid_targets = {"Yes", "No", 1, 0, "1", "0", True, False}
        if not unique_targets.issubset(valid_targets):
            errors.append(
                f"Unexpected target labels in '{TARGET_COLUMN}': {unique_targets}. Expected Yes/No or 1/0."
            )

    return (len(errors) == 0), errors


def download_dataset(url: str = DEFAULT_DATA_URL, dest_path: str = DEFAULT_DATA_PATH, force: bool = False) -> str:
    """
    Downloads the official Telco churn dataset from a remote URL to dest_path.
    Verifies that the file is non-empty and accessible.
    """
    if os.path.exists(dest_path) and not force:
        if os.path.getsize(dest_path) > 1000:
            return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read()
            if len(data) < 1000:
                raise ValueError(f"Downloaded content from {url} is suspiciously small ({len(data)} bytes).")
            with open(dest_path, "wb") as f:
                f.write(data)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset from '{url}': {str(e)}") from e

    return dest_path


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Strips whitespace from string columns
    - Safely converts TotalCharges to numeric, coercing spaces/blanks to NaN
    - Preserves all original columns and rows without altering raw source
    """
    cleaned = df.copy()

    # Clean whitespace in string columns safely across pandas versions
    str_cols = [c for c in cleaned.columns if pd.api.types.is_string_dtype(cleaned[c]) or cleaned[c].dtype == "object"]
    for c in str_cols:
        cleaned[c] = cleaned[c].astype(str).str.strip()

    # Safely convert TotalCharges to numeric
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    # Ensure tenure and MonthlyCharges are numeric if present
    if "tenure" in cleaned.columns:
        cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")
    if "MonthlyCharges" in cleaned.columns:
        cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")
    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="coerce")

    return cleaned


def load_data(
    source: Optional[Union[str, pd.DataFrame]] = None,
    allow_download: bool = True,
    clean: bool = True
) -> pd.DataFrame:
    """
    Primary data loader.
    Supports file path (str), in-memory DataFrame, or downloads if absent and allowed.
    Returns cleaned pandas DataFrame ready for EDA or preprocessing.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = source or DEFAULT_DATA_PATH
        if not os.path.exists(path):
            if allow_download:
                download_dataset(dest_path=path)
            else:
                raise FileNotFoundError(
                    f"Dataset not found at '{path}'. Please provide a valid file path or enable downloading."
                )
        df = pd.read_csv(path)

    is_valid, errors = validate_raw_data(df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed:\n - " + "\n - ".join(errors))

    if clean:
        df = clean_raw_data(df)

    return df


def generate_demo_dataset(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a small synthetic Telco Customer Churn dataset.
    Explicitly labeled for DEMO / TESTING purposes only when real data is unavailable.
    """
    rng = np.random.RandomState(random_state)
    genders = rng.choice(["Male", "Female"], size=n_samples)
    senior = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples)
    dependents = rng.choice(["Yes", "No"], size=n_samples)
    tenure = rng.randint(0, 73, size=n_samples)
    phone = rng.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
    multiple_lines = [
        "No phone service" if p == "No" else rng.choice(["No", "Yes"])
        for p in phone
    ]
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])

    def get_service(int_srv):
        return ["No internet service" if int_srv == "No" else rng.choice(["No", "Yes"]) for _ in range(n_samples)]

    online_sec = ["No internet service" if s == "No" else rng.choice(["No", "Yes"]) for s in internet]
    online_bkp = ["No internet service" if s == "No" else rng.choice(["No", "Yes"]) for s in internet]
    dev_prot = ["No internet service" if s == "No" else rng.choice(["No", "Yes"]) for s in internet]
    tech_supp = ["No internet service" if s == "No" else rng.choice(["No", "Yes"]) for s in internet]
    stream_tv = ["No internet service" if s == "No" else rng.choice(["No", "Yes"]) for s in internet]
    stream_mov = ["No internet service" if s == "No" else rng.choice(["No", "Yes"]) for s in internet]
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples
    )
    monthly = np.round(rng.uniform(18.25, 118.75, size=n_samples), 2)
    total = np.round(monthly * tenure + rng.uniform(0, 10, size=n_samples), 2)
    total = np.where(tenure == 0, np.nan, total)

    # Churn probability higher for month-to-month and fiber optic
    churn_prob = 0.25 + (contract == "Month-to-month") * 0.25 + (internet == "Fiber optic") * 0.15 - (tenure > 36) * 0.2
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = [("Yes" if rng.rand() < p else "No") for p in churn_prob]

    demo_df = pd.DataFrame({
        "customerID": [f"DEMO-{1000 + i}" for i in range(n_samples)],
        "gender": genders,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bkp,
        "DeviceProtection": dev_prot,
        "TechSupport": tech_supp,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })
    return demo_df
