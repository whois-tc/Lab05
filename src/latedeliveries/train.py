"""Train and evaluate late-delivery models."""

from pathlib import Path

from latedeliveries.data import make_orders
from latedeliveries.evaluation import compare_models
from latedeliveries.pipeline import split_orders


def main() -> int:
    """Compare candidate models using only the training data."""
    orders = make_orders()
    x_train, _, y_train, _ = split_orders(orders)

    results = compare_models(x_train, y_train)

    report_path = Path("reports/cv_results.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_path, index=False)

    print("Cross-validation results:")
    print(results.to_string(index=False))
    print()
    print(f"Results saved to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
