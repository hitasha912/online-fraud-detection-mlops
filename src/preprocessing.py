import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ============================================================
# RAW FEATURES
# ============================================================

RAW_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameDest"
]


# ============================================================
# CUSTOM FEATURE ENGINEERING
# ============================================================

class FraudFeatureEngineer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self, use_post_balances=True):
        self.use_post_balances = use_post_balances

    def fit(self, X, y=None):

        required = [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "nameDest"
        ]

        missing = [
            c for c in required
            if c not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        return self

    def transform(self, X):

        d = pd.DataFrame(index=X.index)

        amt = X["amount"].astype(float)

        d["step"] = X["step"].astype(float)
        d["amount"] = amt

        d["oldbalanceOrg"] = (
            X["oldbalanceOrg"].astype(float)
        )

        d["newbalanceOrig"] = (
            X["newbalanceOrig"].astype(float)
        )

        d["oldbalanceDest"] = (
            X["oldbalanceDest"].astype(float)
        )

        d["newbalanceDest"] = (
            X["newbalanceDest"].astype(float)
        )

        # Time features
        d["hour"] = (
            X["step"] % 24
        ).astype(float)

        d["day"] = (
            X["step"] // 24
        ).astype(float)

        # Balance discrepancy features
        d["errorBalanceOrig"] = (
            d["newbalanceOrig"]
            + amt
            - d["oldbalanceOrg"]
        )

        d["errorBalanceDest"] = (
            d["oldbalanceDest"]
            + amt
            - d["newbalanceDest"]
        )

        # Balance indicators
        d["orig_emptied"] = (
            d["newbalanceOrig"] == 0
        ).astype(float)

        d["dest_is_merchant"] = (
            X["nameDest"]
            .astype(str)
            .str.startswith("M")
        ).astype(float)

        d["dest_balance_zero_before"] = (
            d["oldbalanceDest"] == 0
        ).astype(float)

        # Amount relative to origin balance
        d["amount_to_orig_balance"] = (
            amt /
            (d["oldbalanceOrg"] + 1)
        )

        # Log transformed amount
        d["log_amount"] = np.log1p(amt)

        # Categorical feature
        d["type"] = X["type"].astype(str)

        return d


# ============================================================
# PROCESSED FEATURE LISTS
# ============================================================

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "hour",
    "day",
    "errorBalanceOrig",
    "errorBalanceDest",
    "orig_emptied",
    "dest_is_merchant",
    "dest_balance_zero_before",
    "amount_to_orig_balance",
    "log_amount"
]

CATEGORICAL_FEATURES = [
    "type"
]


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "cat",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])


# ============================================================
# CREATE COMPLETE PREPROCESSING PIPELINE
# ============================================================

def create_feature_pipeline(model):

    return Pipeline([
        (
            "features",
            FraudFeatureEngineer()
        ),
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            model
        )
    ])
