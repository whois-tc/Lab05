"""Tests for feature selection, splitting, and preprocessing."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import FEATURES, build_pipeline, split_orders


@pytest.mark.parametrize(
    "excluded_column",
    ["order_id", "customer_rating", "late"],
)
def test_feature_list_excludes_invalid_columns(excluded_column: str) -> None:
    assert excluded_column not in FEATURES


def test_split_uses_75_percent_for_training(orders: pd.DataFrame) -> None:
    x_train, x_test, _, _ = split_orders(orders)

    assert len(x_train) == 300
    assert len(x_test) == 100


def test_split_is_repeatable(orders: pd.DataFrame) -> None:
    first_split = split_orders(orders)
    second_split = split_orders(orders)

    assert first_split[0].index.tolist() == second_split[0].index.tolist()
    assert first_split[1].index.tolist() == second_split[1].index.tolist()


def test_split_is_stratified(orders: pd.DataFrame) -> None:
    _, _, y_train, y_test = split_orders(orders)
    full_rate = orders["late"].mean()

    assert y_train.mean() == pytest.approx(full_rate, abs=0.01)
    assert y_test.mean() == pytest.approx(full_rate, abs=0.01)


def test_pipeline_returns_valid_probabilities(orders: pd.DataFrame) -> None:
    x_train, x_test, y_train, _ = split_orders(orders)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]

    assert len(probabilities) == len(x_test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_pipeline_handles_missing_value_and_unseen_category(
    orders: pd.DataFrame,
) -> None:
    x_train, x_test, y_train, _ = split_orders(orders)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(x_train, y_train)

    unusual_order = x_test.iloc[[0]].copy()
    unusual_order["prep_minutes"] = float("nan")
    unusual_order["weather"] = "hail"

    prediction = pipeline.predict(unusual_order)

    assert prediction.shape == (1,)
