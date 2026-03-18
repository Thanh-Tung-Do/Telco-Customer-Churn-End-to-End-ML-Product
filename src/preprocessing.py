"""
Reusable preprocessing components for the Telco churn pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# Feature groups — keep in sync with the raw data column layout
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling",
]

MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# SeniorCitizen is already 0/1, so it goes straight to numeric
ALL_FEATURE_COLS = ["SeniorCitizen"] + BINARY_COLS + MULTI_CAT_COLS + NUMERIC_COLS


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs and fix the TotalCharges dtype issue."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    for df in (train, test):
        # TotalCharges can be whitespace strings for new customers (tenure=0)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return train, test


def encode_target(series: pd.Series) -> pd.Series:
    """Convert 'Yes'/'No' churn labels to 1/0."""
    return series.map({"Yes": 1, "No": 0}).astype(int)


def build_preprocessor() -> ColumnTransformer:
    """
    Scikit-learn ColumnTransformer:
      - Binary categoricals  → OrdinalEncoder (0/1)
      - Multi-class cats     → OrdinalEncoder
      - Numerics             → median imputation + StandardScaler
    """
    binary_pipe = Pipeline([
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    cat_pipe = Pipeline([
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", binary_pipe, BINARY_COLS),
            ("cat", cat_pipe, MULTI_CAT_COLS),
            ("num", num_pipe, ["SeniorCitizen"] + NUMERIC_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return ordered feature names after fitting the preprocessor."""
    return (
        BINARY_COLS
        + MULTI_CAT_COLS
        + ["SeniorCitizen"] + NUMERIC_COLS
    )