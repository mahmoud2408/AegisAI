# MetroPT Failure Prediction

Phase 7 uses MetroPT-3 for a realistic temporal predictive-maintenance case study. The raw CSV does not contain a machine-readable failure target. Failure intervals are curated from the local `Data Description_Metro.pdf` report and used to define an early-warning target.

## Current Run

- Run: `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912/metropt`
- Script: `scripts/experiments/run_failure_prediction.py`
- Config: `configs/experiments/failure_prediction.yaml`
- Target: `failure_within_6h`
- Split: event-aware chronological train/validation/test
- Threshold policy: validation-only threshold selection

## Dataset Audit

| Item | Value |
| --- | ---: |
| Rows | 1516948 |
| Columns | 17 |
| Timestamp start | 2020-02-01 00:00:00 |
| Timestamp end | 2020-09-01 03:59:50 |
| Median cadence seconds | 10 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate timestamps | 0 |
| Cadence gaps above 1.5x median | 363 |
| Curated failure events | 4 |
| Rows inside curated active failure periods | 29954 |

## Curated Failure Events

| Event | Start | End | Type | Split role |
| --- | --- | --- | --- | --- |
| `metropt_air_leak_2020_04_18` | 2020-04-18 00:00 | 2020-04-18 23:59 | Air leak | Train |
| `metropt_air_leak_2020_05_29` | 2020-05-29 23:30 | 2020-05-30 06:00 | Air leak | Train |
| `metropt_air_leak_2020_06_05` | 2020-06-05 10:00 | 2020-06-07 14:30 | Air leak | Validation |
| `metropt_air_leak_2020_07_15` | 2020-07-15 14:30 | 2020-07-15 19:00 | Air leak | Test |

The label source is the local MetroPT description PDF, not the telemetry CSV. The experiment excludes rows during active failure periods from supervised training and evaluation.

## Target Definition

At timestamp `t`, the target is `1` when a curated failure start occurs within the next six hours. Rows inside active failure intervals are marked as ineligible. This makes the task early-warning prediction rather than post-failure detection.

## Features

Features use only current and historical observations:

- Current sensor values for pressure, oil temperature, reservoir pressure, and motor current.
- Lagged values at 1, 6, and 60 rows.
- One-step differences and rates.
- Rolling mean, standard deviation, min, and max over 5-minute and 30-minute windows.
- Causal trailing-window rate over 30 minutes.
- Digital-state current value, one-step lag, transition flag, and seconds since transition.
- Hour-of-day and day-of-week cyclical features.

## Splits

| Split | Evaluation rows | Positive target rows |
| --- | ---: | ---: |
| Train | 139605 | 661 |
| Validation | 6877 | 363 |
| Test | 101150 | 363 |

Training negatives are downsampled chronologically after retaining all positive training rows. The final training sample contains 17186 rows.

## Measured Test Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | False positives | False negatives | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.1282 | 0.0020 | 0.0507 | 5108 | 363 | 0.3 |
| Random Forest | 0.0000 | 0.0000 | 0.0000 | 0.2509 | 0.0031 | 0.0142 | 1436 | 363 | 0.1 |
| XGBoost | 0.0000 | 0.0000 | 0.0000 | 0.1795 | 0.0035 | 0.0126 | 1274 | 363 | 0.1 |

All classical baselines missed the held-out July failure at their validation-selected thresholds. XGBoost is the artifact summary's best test-ranked model only because all F1 values are zero and it has the highest PR-AUC with the lowest false-positive rate among the zero-recall models. This should not be interpreted as a deployable predictive-maintenance model.

## Explainability

SHAP artifacts were generated for the validation-selected tree explanation model, Random Forest:

- `shap/global_feature_importance.csv`
- `shap/local_explanations.csv`
- `shap/shap_summary_bar.png`

Top global SHAP features:

| Feature | Mean absolute SHAP |
| --- | ---: |
| Caudal_impulses__seconds_since_transition | 0.0721 |
| LPS__seconds_since_transition | 0.0487 |
| time__dayofweek_cos | 0.0426 |
| time__dayofweek_sin | 0.0391 |
| time__hour_cos | 0.0385 |

These features describe what the fitted model used. They do not establish root cause.

## Interpretation

The negative result is useful. It shows that a simple event-derived early-warning target with only four documented air-leak events is not enough for reliable generalization across time. The held-out event's maximum probability inside the warning horizon remained below the selected thresholds for all three models.

## Limitations

- Only four documented failure events are available.
- All curated events are air-leak events, so failure-mode diversity is absent.
- The source labels are report-derived intervals, not per-row operator annotations.
- The test set contains one failure episode; event-level confidence intervals would be very wide.
- Deep learning is not justified yet for MetroPT failure prediction. The priority should be label curation, horizon sensitivity, and stronger classical temporal validation.
