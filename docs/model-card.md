# Model Card

Phase status: experimental forecasting baselines exist, but no model has been registered for production inference.

Experimental model cards:

- `docs/models/isolation_forest_nab.md`
- `docs/models/autoencoder_nab.md`
- `docs/models/lstm_autoencoder_nab.md`
- `docs/models/smd_isolation_forest.md`
- `docs/models/smd_autoencoder.md`
- `docs/models/smd_lstm_autoencoder.md`
- `docs/models/failure_prediction.md`
- `docs/models/forecasting_baselines.md`

This document is a template for future registered models.

## Model Details

- Model name:
- Version:
- Model family:
- Owner:
- Training date:
- Intended use:
- Out-of-scope use:

## Data

- Training dataset:
- Validation dataset:
- Test dataset:
- Dataset version or commit:
- Split policy:
- Known data limitations:

## Metrics

No metrics until an evaluation artifact exists.

Required metrics depend on model type:

- Anomaly detection: precision, recall, F1, PR-AUC, false positive rate, detection delay, NAB score when applicable
- Incident prediction: precision, recall, F1, ROC-AUC, PR-AUC, calibration
- Forecasting: MAE, RMSE, MAPE when appropriate

## Explainability

- Explanation method:
- Feature groups:
- Known explanation limitations:

## Operational Considerations

- Expected inference latency:
- Monitoring signals:
- Retraining triggers:
- Failure modes:
