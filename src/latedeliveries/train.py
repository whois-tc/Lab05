"""Train and evaluate late-delivery models."""

from pathlib import Path

from sklearn.model_selection import cross_val_predict

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    choose_threshold,
    compare_models,
    make_folds,
    tune_balanced_logistic,
)
from latedeliveries.pipeline import split_orders


def main() -> int:
    """Compare, tune, and select a threshold using training data."""
    orders = make_orders()
    x_train, _, y_train, _ = split_orders(orders)

    results = compare_models(x_train, y_train)

    comparison_path = Path("reports/cv_results.csv")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(comparison_path, index=False)

    print("Cross-validation results:")
    print(results.to_string(index=False))
    print()

    search = tune_balanced_logistic(x_train, y_train)

    balanced_results = results[results["model"] == "Balanced logistic regression"].iloc[
        0
    ]

    untuned_mean = balanced_results["roc_auc_mean"]
    untuned_spread = balanced_results["roc_auc_std"]
    improvement = search.best_score_ - untuned_mean
    beat_fold_spread = improvement > untuned_spread

    print("Tuning results:")
    print(f"Best settings: {search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {search.best_score_:.4f}")
    print(f"Untuned ROC-AUC mean: {untuned_mean:.4f}")
    print(f"Untuned fold-to-fold spread: {untuned_spread:.4f}")
    print(f"Improvement from tuning: {improvement:.4f}")
    print(f"Tuning beat the fold-to-fold spread: {beat_fold_spread}")
    print()

    out_of_fold_probabilities = cross_val_predict(
        search.best_estimator_,
        x_train,
        y_train,
        cv=make_folds(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    best_threshold, threshold_results = choose_threshold(
        y_train,
        out_of_fold_probabilities,
    )

    threshold_path = Path("reports/threshold_costs.csv")
    threshold_results.to_csv(threshold_path, index=False)

    minimum_cost = int(threshold_results["cost"].min())

    print("Threshold results:")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Lowest cross-validated training cost: ${minimum_cost}")
    print(f"Threshold table saved to: {threshold_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
