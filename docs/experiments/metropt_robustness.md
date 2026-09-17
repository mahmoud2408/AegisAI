# MetroPT Robustness

Phase 7B evaluates whether MetroPT-3 supports reliable supervised early-warning prediction after correcting the Phase 7 target-analysis gaps. The result is a constrained scientific conclusion, not a tuned success story.

## Current Run

- Run: `experiments/prediction/metropt_robustness/phase7b_metropt_robustness_20260914`
- Script: `scripts/experiments/run_metropt_robustness.py`
- Config: `configs/experiments/metropt_robustness.yaml`
- Models: Logistic Regression, Random Forest, XGBoost
- Horizons: 0.5, 2.0, and 6.0 hours
- Thresholds: selected from validation only
- Deep learning: not implemented in this phase

## Candidate Selection

The selected candidate is `h6p0__random_forest`, chosen by validation warning coverage, validation F1, validation PR-AUC, and validation false-alarm burden. Test labels were not used for model, horizon, or threshold selection.

| Metric | Value |
| --- | ---: |
| Horizon | 6.0 hours |
| Model | Random Forest |
| Threshold | 0.005 |
| Validation precision | 0.0694 |
| Validation recall | 0.4242 |
| Validation F1 | 0.1193 |
| Validation PR-AUC | 0.0567 |
| Validation event detection rate | 1.0000 |
| Validation median lead time hours | 5.9517 |
| Validation false alarms per day | 211.7724 |

## Held-Out Test Result

| Metric | Value |
| --- | ---: |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.2439 |
| PR-AUC | 0.0036 |
| False-positive rate | 0.4376 |
| False positives | 44102 |
| False negatives | 363 |
| Event detection rate | 0.0000 |
| Median lead time hours | n/a |
| False alarms per day | 377.2708 |

The selected candidate detected the validation failure but missed the held-out July failure entirely, while producing an impractically high false-alarm burden.

## Horizon Comparison

| Candidate | Validation F1 | Validation event detection | Test F1 | Test PR-AUC | Test event detection | Test false alarms per day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5h Logistic Regression | 0.0471 | 1.0000 | 0.0000 | 0.0002 | 0.0000 | 19.9589 |
| 0.5h Random Forest | 0.0833 | 1.0000 | 0.0000 | 0.0006 | 0.0000 | 22.5099 |
| 0.5h XGBoost | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 0.0000 | 4.7831 |
| 2.0h Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.0006 | 0.0000 | 15.8845 |
| 2.0h Random Forest | 0.0800 | 1.0000 | 0.0000 | 0.0012 | 0.0000 | 82.8945 |
| 2.0h XGBoost | 0.0000 | 0.0000 | 0.0000 | 0.0013 | 0.0000 | 7.7592 |
| 6.0h Logistic Regression | 0.0177 | 1.0000 | 0.0000 | 0.0021 | 0.0000 | 133.5358 |
| 6.0h Random Forest | 0.1193 | 1.0000 | 0.0000 | 0.0036 | 0.0000 | 377.2708 |
| 6.0h XGBoost | 0.0000 | 0.0000 | 0.0000 | 0.0044 | 0.0000 | 24.2460 |

No validation-selected candidate warned for the July test event. A non-selected 0.5h Random Forest threshold of `0.005` can trigger on the July event, but it produces about `453.6816` false alarms per day on the test split and has validation F1 `0.0133`. That is a threshold-sensitivity warning, not a deployable result.

## July Failure Investigation

For the selected 6h Random Forest candidate:

| Item | Value |
| --- | ---: |
| Warning-window rows | 363 |
| Warnings in horizon | 0 |
| First threshold crossing | n/a |
| Baseline probability median | 0.0000 |
| Baseline probability p95 | 0.0050 |
| Warning probability max | 0.0000 |
| Probability increased before failure | No |

The model did not receive a transferable pre-failure probability signal for July. The event is not simply missed because of a conservative threshold; the selected model's probability stayed flat inside the warning horizon.

## Distribution Shift Evidence

The July pre-failure rows differ strongly from train pre-failure rows. The largest standardized mean differences for the selected candidate are:

| Feature | Train positive mean | July positive mean | Standardized difference |
| --- | ---: | ---: | ---: |
| `time__dayofweek_sin` | -0.4339 | 0.9749 | 140881159.9016 |
| `time__dayofweek_cos` | -0.9010 | -0.2225 | 67844793.2005 |
| `Motor_current__rolling_min_30min` | 0.0392 | 3.5601 | 2164.8140 |
| `Caudal_impulses__seconds_since_transition` | 81538.1406 | 1750879.2500 | 22.5575 |
| `Oil_level__seconds_since_transition` | 81538.1406 | 1750879.2500 | 22.5575 |

The enormous time-feature differences occur because the train positive windows cover a small set of days. These features are a warning sign for event-identity overfitting, not causal predictive evidence.

## Feature Contribution

SHAP artifacts were generated for the selected horizon's tree model. Top model-contributing features:

| Feature | Mean absolute SHAP |
| --- | ---: |
| `Caudal_impulses__seconds_since_transition` | 0.0610 |
| `LPS__seconds_since_transition` | 0.0463 |
| `time__dayofweek_sin` | 0.0433 |
| `Oil_level__seconds_since_transition` | 0.0419 |
| `time__dayofweek_cos` | 0.0417 |

These are model-contributing features only. They are not root-cause factors.

## Final Decision

Decision **B** is supported: MetroPT supports limited failure prediction but has strong generalization limitations.

Operational interpretation: keep MetroPT in AegisAI as a realistic telemetry, forecasting, anomaly-explanation, and robustness dataset. Do not present the current MetroPT supervised failure-prediction model as deployable or reliable. Under the current setup, the available labels are too sparse and too distribution-shifted for a trustworthy early-warning classifier.

## Next Step

Do not move to deep learning for MetroPT failure prediction yet. Better next work is either stronger label curation and event-aware validation, or a separate forecasting phase where MetroPT is better suited.
