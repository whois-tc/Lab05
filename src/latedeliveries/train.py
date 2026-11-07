"""Train, evaluate, and save the late-delivery model."""

import json
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import cross_val_predict

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    choose_threshold,
    compare_models,
    evaluate_test_predictions,
    make_folds,
    tune_balanced_logistic,
)
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    split_orders,
)

DATA_SEED = 7


def main() -> int:
    """Run the complete late-delivery training workflow."""
    orders = make_orders(seed=DATA_SEED)
    x_train, x_test, y_train, y_test = split_orders(orders)

    comparison_results = compare_models(x_train, y_train)

    comparison_path = Path("reports/cv_results.csv")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_results.to_csv(comparison_path, index=False)

    print("Cross-validation results:")
    print(comparison_results.to_string(index=False))
    print()

    search = tune_balanced_logistic(x_train, y_train)

    balanced_results = comparison_results[
        comparison_results["model"] == "Balanced logistic regression"
    ].iloc[0]

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

    minimum_training_cost = int(threshold_results["cost"].min())

    print("Threshold results:")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Lowest cross-validated training cost: ${minimum_training_cost}")
    print()

    final_pipeline = search.best_estimator_
    final_pipeline.fit(x_train, y_train)

    test_probabilities = final_pipeline.predict_proba(x_test)[:, 1]
    test_results = evaluate_test_predictions(
        y_test,
        test_probabilities,
        best_threshold,
    )

    metrics_path = Path("reports/metrics.json")
    metrics_path.write_text(
        json.dumps(test_results, indent=2),
        encoding="utf-8",
    )

    model_path = Path("models/late_orders.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)

    model_info = {
        "scikit_learn_version": sklearn.__version__,
        "seed": RANDOM_STATE,
        "data_seed": DATA_SEED,
        "threshold": best_threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_settings": search.best_params_,
        "test_results": test_results,
    }

    model_info_path = Path("models/model_info.json")
    model_info_path.write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    print("Final test-set result:")
    print(json.dumps(test_results, indent=2))
    print()
    print(f"Model saved to: {model_path}")
    print(f"Model information saved to: {model_info_path}")
    print(f"Test metrics saved to: {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
