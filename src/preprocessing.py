"""
Data preprocessing and pipeline building module.
Implements Scikit-learn ColumnTransformer, missing value imputation,
scaling, one-hot encoding, and stratified data partitioning.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMNS = ["customerID"]

# Standard telco feature categories
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
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


def separate_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Separates feature matrix X and target vector y.
    Removes identifier columns from X.
    """
    if id_cols is None:
        id_cols = IDENTIFIER_COLUMNS

    df_copy = df.copy()
    drop_cols = [c for c in id_cols if c in df_copy.columns]

    if target_col in df_copy.columns:
        y_raw = df_copy[target_col]
        # Encode target consistently as binary integers: 1 for Churn/Yes, 0 for Retained/No
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0, 1: 1, 0: 0, True: 1, False: 0}
        mapped = y_raw.map(mapping)
        if mapped.isna().all() or mapped.isna().any():
            # In case some rows are already numeric or unmapped
            try:
                y = pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int)
            except Exception:
                y = mapped.fillna(0).astype(int)
        else:
            y = mapped.astype(int)
        drop_cols.append(target_col)
    else:
        y = None

    X = df_copy.drop(columns=drop_cols, errors="ignore")
    return X, y


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and categorical features programmatically from feature matrix X.
    """
    num_cols = []
    cat_cols = []

    for col in X.columns:
        if col in NUMERICAL_FEATURES:
            num_cols.append(col)
        elif col in CATEGORICAL_FEATURES:
            cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > 5:
            num_cols.append(col)
        else:
            cat_cols.append(col)

    return num_cols, cat_cols


def build_preprocessor(
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer for preprocessing:
    - Numerical: median imputation + standard scaling
    - Categorical: most frequent imputation + one-hot encoding (handle_unknown='ignore')
    """
    if num_cols is None:
        num_cols = NUMERICAL_FEATURES
    if cat_cols is None:
        cat_cols = CATEGORICAL_FEATURES

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts post-transformation feature names from a fitted ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction if get_feature_names_out is unavailable
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
                ohe = trans.named_steps["ohe"]
                if hasattr(ohe, "get_feature_names_out"):
                    feature_names.extend(list(ohe.get_feature_names_out(cols)))
                else:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
        return feature_names


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a stratified train/test split on cleaned dataframe.
    Ensures test data is strictly held out and untouched.
    """
    X, y = separate_features_target(df)
    if y is None:
        raise ValueError("Cannot perform train/test split without target column 'Churn'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
