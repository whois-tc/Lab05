"""Train and evaluate late-delivery models."""

from pathlib import Path

from latedeliveries.data import make_orders
from latedeliveries.evaluation import compare_models, tune_balanced_logistic
from latedeliveries.pipeline import split_orders


def main() -> int:
    """Compare and tune models using only the training data."""
    orders = make_orders()
    x_train, _, y_train, _ = split_orders(orders)

    results = compare_models(x_train, y_train)

    report_path = Path("reports/cv_results.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_path, index=False)

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
    print(f"Comparison saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
