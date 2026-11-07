"""Test saving and reloading the complete model pipeline."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, split_orders


def test_reloaded_model_makes_same_predictions(
    orders: pd.DataFrame,
    tmp_path,
) -> None:
    x_train, x_test, y_train, _ = split_orders(orders)

    pipeline = build_pipeline(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=10.0,
            solver="liblinear",
        )
    )
    pipeline.fit(x_train, y_train)

    sample = x_test.head(10)
    predictions_before = pipeline.predict_proba(sample)

    model_path = tmp_path / "late_orders.joblib"
    joblib.dump(pipeline, model_path)
    reloaded_pipeline = joblib.load(model_path)

    predictions_after = reloaded_pipeline.predict_proba(sample)

    assert np.allclose(predictions_before, predictions_after)
