# AI4I Failure Prediction

Phase 7 uses AI4I 2020 as the controlled supervised predictive-maintenance baseline. This dataset is useful for validating the tabular classification, calibration, and explainability machinery, but it is synthetic and has no meaningful timestamp axis.

## Current Run

- Run: `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912/ai4i`
- Script: `scripts/experiments/run_failure_prediction.py`
- Config: `configs/experiments/failure_prediction.yaml`
- Target: `Machine failure == 1`
- Split: stratified train/validation/test because AI4I has row index only
- Threshold policy: threshold selected on validation data from the configured grid

## Dataset Audit

| Item | Value |
| --- | ---: |
| Rows | 10000 |
| Columns | 14 |
| Positive failures | 339 |
| Positive rate | 0.0339 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Product type H | 1003 |
| Product type L | 6000 |
| Product type M | 2997 |

The predictive features are the five numeric process variables plus `Type`. The identifiers `UDI` and `Product ID` are excluded. The failure-mode columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are also excluded because they are target-side labels, not deployable input features for predicting `Machine failure`.

## Models

| Model | Notes |
| --- | --- |
| Logistic Regression | Scaled numeric features, one-hot `Type`, balanced class weights |
| Random Forest | Balanced class weights, validation-selected threshold |
| XGBoost | `scale_pos_weight` from the training split, validation-selected threshold |

## Measured Test Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.2782 | 0.7255 | 0.4022 | 0.9101 | 0.4219 | 0.0662 | 0.7 |
| Random Forest | 0.6596 | 0.6078 | 0.6327 | 0.9674 | 0.6757 | 0.0110 | 0.7 |
| XGBoost | 0.5606 | 0.7255 | 0.6325 | 0.9659 | 0.7450 | 0.0200 | 0.7 |

The best test F1 is Random Forest at `0.6327`. XGBoost has the strongest PR-AUC at `0.7450`, but slightly lower precision at the selected operating point.

## Explainability

SHAP artifacts were generated for the validation-selected tree explanation model, XGBoost:

- `shap/global_feature_importance.csv`
- `shap/local_explanations.csv`
- `shap/shap_summary_bar.png`

Top global SHAP features:

| Feature | Mean absolute SHAP |
| --- | ---: |
| Torque [Nm] | 1.6150 |
| Tool wear [min] | 1.4204 |
| Rotational speed [rpm] | 1.1278 |
| Air temperature [K] | 0.8852 |
| Process temperature [K] | 0.3770 |

These are model-contributing features, not causal proof of machine failure.

## Limitations

- AI4I is synthetic and compact, so it is not evidence that the same model will transfer to production telemetry.
- There is no real timestamp, so temporal early-warning evaluation is not appropriate.
- The failure modes are provided as labels and are intentionally excluded from the input feature set.
- No model is registered for production inference in this phase.
