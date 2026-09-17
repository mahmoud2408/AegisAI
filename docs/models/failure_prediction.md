# Failure Prediction Model Card

## Status

Phase 7 creates experimental supervised failure-prediction baselines only. No model is registered for production inference.

## Intended Use

These models support portfolio-grade evaluation of incident and predictive-maintenance classifiers. They are intended for offline benchmarking, feature-attribution inspection, calibration checks, and comparison against future forecasting or agentic investigation components.

## Out-of-Scope Use

- Production alerting.
- Root-cause claims.
- Safety-critical maintenance decisions.
- Generalizing AI4I performance to real industrial telemetry.
- Treating MetroPT report-derived event labels as dense per-row ground truth.

## Data

| Dataset | Role | Label |
| --- | --- | --- |
| AI4I 2020 | Controlled supervised tabular baseline | Native `Machine failure` column |
| MetroPT-3 | Temporal predictive-maintenance case study | Six-hour future-failure target from curated report intervals |

AI4I is split with stratification because it has no meaningful time axis. MetroPT is split chronologically by event: two train failures, one validation failure, and one held-out test failure.

## Model Families

| Model | Purpose |
| --- | --- |
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Nonlinear classical baseline |
| XGBoost | Strong gradient-boosted tabular baseline |

Thresholds are selected on validation predictions only. Test labels are used only for final evaluation.

## Current Artifacts

- Run summary: `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912/metrics.json`
- AI4I artifacts: `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912/ai4i/`
- MetroPT artifacts: `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912/metropt/`
- Reproduction command: `.\.venv\Scripts\python.exe scripts\experiments\run_failure_prediction.py --run-id phase7_failure_prediction_20260912`

## Current Measured Results

| Dataset | Test-ranked model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AI4I | Random Forest | 0.6596 | 0.6078 | 0.6327 | 0.9674 | 0.6757 | 0.0110 |
| MetroPT | XGBoost | 0.0000 | 0.0000 | 0.0000 | 0.1795 | 0.0035 | 0.0126 |

The MetroPT model is listed as test-ranked only for artifact summarization. Because all MetroPT F1 values are zero, there is no deployable winner in Phase 7.

## Explainability

SHAP is generated for a validation-selected tree model where available:

- AI4I explanation model: XGBoost.
- MetroPT explanation model: Random Forest.

SHAP values are model-attribution evidence. They are not causal evidence and should not be presented as root-cause analysis.

## Known Failure Modes

- Class imbalance can produce high ranking scores but poor operational thresholds.
- MetroPT has too few curated failure episodes for confident supervised generalization.
- Digital-state timing features can dominate attribution without proving causal mechanisms.
- Calibration remains experimental; no calibrated model is registered.

## Next Steps

1. Add horizon sensitivity for MetroPT using validation-only model selection.
2. Add event-aware cross-validation once more failure intervals are curated.
3. Compare against SMD-derived incident-window prediction after a leakage-safe target is defined.
4. Register a model only after validation and test performance are both operationally acceptable.
