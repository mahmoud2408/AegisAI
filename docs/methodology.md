# Methodology

Phase status: experimental methodology proposal only.

## Principles

- Establish simple baselines before deep models.
- Use chronological train/validation/test splits for time-series data.
- Fit preprocessing and feature engineering only on training data where learning is involved.
- Avoid future-looking rolling windows and label leakage.
- Track all parameters, datasets, metrics, and artifacts.
- Report uncertainty and limitations with every benchmark.

## Anomaly Detection

Planned models:

- Statistical baseline
- Isolation Forest
- Autoencoder
- LSTM Autoencoder

Planned metrics:

- Precision
- Recall
- F1
- PR-AUC when applicable
- False positive rate
- Detection delay
- NAB score when applicable

## Incident Prediction

The supervised task predicts whether an incident occurs within a configurable future window.

Planned models:

- Logistic Regression
- Random Forest
- XGBoost
- Optional LightGBM

Feature groups:

- Current telemetry values
- Rolling statistics
- Trend features
- Lagged values
- Anomaly scores
- Error rates
- Traffic signals
- Service-level indicators

## Forecasting

Forecasting experiments must define:

- Target metric
- Horizon
- Input window
- Resampling frequency
- Missing-data policy
- Model family
- Backtesting strategy

Metrics:

- MAE
- RMSE
- MAPE when values make percentage error meaningful

## Explainability

Use SHAP or model-native feature importance only where appropriate. Explanations must come from the fitted model and input features used for the prediction.

The UI and reports must separate:

- Observed evidence
- Model inference
- Probable cause
- Recommendation
