"""Model comparison, tuning, and evaluation functions."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)

from latedeliveries.pipeline import RANDOM_STATE, build_pipeline

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def make_folds() -> StratifiedKFold:
    """Return the repeatable five-fold cross-validation plan."""
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def candidate_models() -> dict:
    """Return the baseline and three real model candidates."""
    return {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Balanced logistic regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Balanced random forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def compare_models(x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Compare all candidates with five-fold stratified cross-validation."""
    rows = []

    for model_name, model in candidate_models().items():
        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=make_folds(),
            scoring=SCORING,
            n_jobs=-1,
        )

        row = {"model": model_name}

        for metric_name in SCORING:
            metric_results = scores[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = round(metric_results.mean(), 4)
            row[f"{metric_name}_std"] = round(metric_results.std(), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def tune_balanced_logistic(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Tune balanced logistic regression using training data only."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    pipeline = build_pipeline(model)

    parameter_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__solver": ["liblinear", "lbfgs"],
    }

    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="roc_auc",
        cv=make_folds(),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)

    return search
