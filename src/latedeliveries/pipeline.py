"""Feature selection, data splitting, and preprocessing pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET = "late"

NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "prep_minutes",
    "courier_months",
]

CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
    "vehicle",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_orders(
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a repeatable, stratified 75/25 train-test split."""
    features = orders[FEATURES]
    target = orders[TARGET]

    return train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_pipeline(model) -> Pipeline:
    """Return preprocessing and a model joined in one pipeline."""
    preprocessing = ColumnTransformer(
        [
            (
                "numbers",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
