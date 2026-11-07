# Late Deliveries

This project predicts whether a QuickBite food-delivery order will arrive late. It compares several classification models, selects a probability threshold using the company’s voucher and missed-delivery costs, and saves the final preprocessing-and-model pipeline.

## Setup

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .

```

## Run

```bash
python -m latedeliveries.train
```

This recreates the cross-validation results, threshold-cost table, final metrics, saved pipeline and model information.

## Check it

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

### Columns used

Numeric features:

- `distance_km`
- `items`
- `hour`
- `prep_minutes`
- `courier_months`

Categorical features:

- `weather`
- `day_type`
- `vehicle`

### Columns excluded

- `order_id` was excluded because it is an identifier rather than meaningful predictive evidence.
- `customer_rating` was excluded because it is provided after delivery and would leak information unavailable when an order is placed.
- `late` was excluded because it is the target being predicted.

The project treats `prep_minutes` as the preparation-time field available when an order is placed. If it represents the actual completed preparation duration rather than an available estimate, it should be excluded from a production model.

### Model comparison

Models were compared using five-fold stratified cross-validation on the training data only.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy baseline | 0.8440 ± 0.0009 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| Logistic regression | 0.8622 ± 0.0085 | 0.6541 ± 0.0653 | 0.2420 ± 0.0526 | 0.3516 ± 0.0644 | 0.7951 ± 0.0342 |
| Balanced logistic regression | 0.7369 ± 0.0187 | 0.3328 ± 0.0267 | 0.6809 ± 0.0493 | 0.4469 ± 0.0337 | 0.7947 ± 0.0338 |
| Balanced random forest | 0.8387 ± 0.0116 | 0.4853 ± 0.0371 | 0.4042 ± 0.0683 | 0.4363 ± 0.0350 | 0.7748 ± 0.0213 |

The dummy model’s 84.4% accuracy hides the fact that it catches no late deliveries. Balanced logistic regression was selected because it achieved the highest recall and F1 while maintaining essentially the same ROC-AUC as standard logistic regression. This suits the business scenario because missing a late delivery costs $15 while flagging an order costs $4.

### Tuning

Balanced logistic regression was tuned using ROC-AUC because the final threshold was selected separately using business costs. Grid search tested eight combinations:

- `model__C`: 0.01, 0.1, 1.0 and 10.0
- `model__solver`: `liblinear` and `lbfgs`

The best settings were `C=10.0` and `solver="liblinear"`. The tuned ROC-AUC was 0.7949, compared with the untuned mean of 0.7947. The improvement of 0.0002 was much smaller than the fold-to-fold spread of 0.0338, so tuning did not produce a meaningful improvement.

### Threshold

Out-of-fold training probabilities were tested at thresholds from 0.05 to 0.95 in steps of 0.05. Each flagged order costs $4, while each unflagged late order costs $15. The 0.70 threshold produced the lowest training cost of $4,098. The test set did not influence this decision.

## Result

The final pipeline was fitted on the complete training set and evaluated on the test set once using the fixed 0.70 threshold.

| | Predicted on time | Flagged late |
|---|---:|---:|
| Actually on time | 578 | 55 |
| Actually late | 64 | 53 |

| Metric | Result |
|---|---:|
| Precision | 0.4907 |
| Recall | 0.4530 |
| F1 | 0.4711 |
| ROC-AUC | 0.7951 |

| Policy | Cost |
|---|---:|
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| Use the model | $1,392 |

The model reduced the observed cost by $363 compared with flagging nobody. This is promising but should be treated cautiously because it comes from invented data and one fixed test set.

## What the tests cover

The tests protect:

- excluded identifier, leakage and target columns;
- the repeatable stratified 75/25 split;
- valid predicted probabilities;
- missing numeric values and unseen categories;
- the hand-calculated cost rule;
- thresholds where flagging nobody or everybody is cheapest;
- identical predictions before and after saving and reloading the pipeline using `tmp_path`.

The tests do not verify whether vouchers change real customer behaviour because the project has no real customer-response data.

## Known limitations

- The dataset is invented.
- The cost rule assumes every voucher works.
- Using `prep_minutes` depends on it being available when prediction occurs.
- The result comes from one fixed test split.
- The model missed 64 of the 117 late test orders.
- Production drift and changing delivery conditions are not tested.