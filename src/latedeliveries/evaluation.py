"""Model comparison, tuning, and evaluation functions."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

VOUCHER_COST = 4
MISSED_LATE_COST = 15
THRESHOLDS = [round(value / 100, 2) for value in range(5, 100, 5)]


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


def order_cost(y_true, flagged) -> int:
    """Return voucher costs plus costs from missed late orders."""
    true_values = np.asarray(y_true)
    flagged_values = np.asarray(flagged, dtype=bool)

    voucher_total = int(flagged_values.sum()) * VOUCHER_COST
    missed_late = (true_values == 1) & ~flagged_values
    missed_late_total = int(missed_late.sum()) * MISSED_LATE_COST

    return voucher_total + missed_late_total


def choose_threshold(
    y_true,
    probabilities,
    thresholds=None,
) -> tuple[float, pd.DataFrame]:
    """Return the threshold with the lowest total business cost."""
    true_values = np.asarray(y_true)
    probability_values = np.asarray(probabilities)

    if thresholds is None:
        thresholds = THRESHOLDS

    rows = []

    for threshold in thresholds:
        flagged = probability_values >= threshold
        missed_late = (true_values == 1) & ~flagged

        rows.append(
            {
                "threshold": threshold,
                "cost": order_cost(true_values, flagged),
                "flagged_orders": int(flagged.sum()),
                "missed_late_orders": int(missed_late.sum()),
            }
        )

    results = pd.DataFrame(rows)
    best_index = results["cost"].idxmin()
    best_threshold = float(results.loc[best_index, "threshold"])

    return best_threshold, results


def evaluate_test_predictions(
    y_true,
    probabilities,
    threshold,
) -> dict:
    """Evaluate final predictions at the chosen threshold."""
    true_values = np.asarray(y_true)
    probability_values = np.asarray(probabilities)
    flagged = probability_values >= threshold

    flag_nobody = np.zeros(len(true_values), dtype=bool)
    flag_everybody = np.ones(len(true_values), dtype=bool)

    return {
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(true_values, flagged).tolist(),
        "precision": round(
            precision_score(true_values, flagged, zero_division=0),
            4,
        ),
        "recall": round(
            recall_score(true_values, flagged, zero_division=0),
            4,
        ),
        "f1": round(
            f1_score(true_values, flagged, zero_division=0),
            4,
        ),
        "roc_auc": round(
            roc_auc_score(true_values, probability_values),
            4,
        ),
        "flagged_orders": int(flagged.sum()),
        "costs": {
            "flag_nobody": order_cost(true_values, flag_nobody),
            "flag_everybody": order_cost(true_values, flag_everybody),
            "model": order_cost(true_values, flagged),
        },
    }
