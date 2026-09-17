# Lead-Time Analysis

Phase 7 introduces event-level early-warning evaluation for predictive maintenance. This complements row-level precision, recall, F1, ROC-AUC, and PR-AUC by asking whether a model produced at least one warning before a held-out failure event.

## Method

For each failure event, the warning window is the interval from `event_start - horizon` up to, but not including, `event_start`. A model warns when its predicted failure probability is greater than or equal to the threshold selected on validation data.

Reported metrics:

- Warning coverage: fraction of failure events with at least one pre-failure warning.
- Lead time: time between the first warning and the failure start.
- Missed events: failure events with no warning inside the horizon.
- False warnings: warnings outside protected pre-failure windows.
- False alarm rate: false warnings divided by negative prediction rows.
- False alarms per day: false warnings normalized by prediction time span.

Rows during active failure periods are excluded before evaluation, so this is not scoring post-failure detection.

## Current MetroPT Run

- Run: `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912/metropt`
- Horizon: 6 hours
- Held-out test event: `metropt_air_leak_2020_07_15`
- Test prediction rows in warning horizon: 363

## Measured Event-Level Results

| Model | Warning coverage | Warned events | Missed events | Median lead hours | False warnings | False alarms per day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.0000 | 0 | 1 | n/a | 5108 | 60.3255 |
| Random Forest | 0.0000 | 0 | 1 | n/a | 1436 | 16.9592 |
| XGBoost | 0.0000 | 0 | 1 | n/a | 1274 | 15.0460 |

No model produced a warning for the held-out July failure inside the six-hour warning horizon.

## Held-Out Failure Detail

| Model | Failure start | Warning produced | Warnings in horizon | Max probability in horizon |
| --- | --- | --- | ---: | ---: |
| Logistic Regression | 2020-07-15 14:30 | No | 0 | 0.0000 |
| Random Forest | 2020-07-15 14:30 | No | 0 | 0.0078 |
| XGBoost | 2020-07-15 14:30 | No | 0 | 0.0001 |

## Takeaway

The early-warning analysis prevents an inflated interpretation of probability-ranking metrics. Even when a model has nonzero ROC-AUC or PR-AUC, it can still fail operationally if the validation-selected threshold does not fire before the failure.

Future work should evaluate multiple horizons and event-aware cross-validation folds after more failure intervals are curated. Any threshold expansion must be selected using validation folds only, never by inspecting the held-out test failure.
