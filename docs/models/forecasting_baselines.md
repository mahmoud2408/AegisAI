# Forecasting Baselines Model Card

## Status

Phase 8 creates experimental MetroPT forecasting baselines only. No forecasting model is registered for production inference.

## Intended Use

The models support offline evaluation of short-horizon telemetry forecasting. They are intended to help AegisAI estimate future sensor trajectories and produce future degradation evidence for later risk-engine work.

## Out-of-Scope Use

- Production alerting.
- Predicting compressor failures directly.
- Root-cause claims.
- Safety-critical maintenance decisions.
- Forecasting binary actuator or state variables with continuous-regression models.

## Data

| Dataset | Role | Target |
| --- | --- | --- |
| MetroPT-3 | High-volume compressor telemetry | Continuous sensor values at future horizons |

Phase 8 uses the raw MetroPT CSV, resamples selected continuous sensors to one-minute median values, and creates split-local forecasting windows. Scalers are fit only on training rows.

## Target Variables

- `TP2`
- `TP3`
- `H1`
- `Reservoirs`
- `Oil_temperature`
- `Motor_current`

`DV_pressure` is excluded from the first target set because it is intermittent and spike-oriented. Binary state variables are audited but not forecast.

## Horizons

- 5 minutes
- 15 minutes
- 30 minutes

The raw median cadence is `10.0` seconds. The forecasting frame uses a one-minute grid, so the horizons are exact row offsets of 5, 15, and 30 steps.

## Model Families

| Model | Parameters | Purpose |
| --- | ---: | --- |
| Persistence | 0 | Last-value baseline |
| Moving average | 0 | Strong simple smoother |
| Rolling linear trend | 0 | Causal statistical extrapolation |
| Ridge autoregression | 1086 | Multivariate lag baseline |
| LSTM univariate | 4513 | Pooled single-sensor sequence model |
| LSTM multivariate | 5318 | Cross-sensor sequence model |

ARIMA was not selected for the first rolling-origin benchmark because the resampled telemetry has long gaps, operating-regime changes, and highly correlated sensors. Rolling linear trend and ridge autoregression provide lower-cost statistical baselines while preserving strict temporal evaluation.

## Current Artifacts

- Run summary: `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915/metrics.json`
- Metrics: `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915/metrics.csv`
- Forecasts: `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915/forecasts.parquet`
- Model artifacts: `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915/models/`
- Reproduction command: `.\.venv\Scripts\python.exe scripts\experiments\run_metropt_forecasting.py --run-id phase8_metropt_forecasting_20260915`

## Current Measured Result

Validation-selected global candidate:

| Model | Horizon | Test MAE | Test RMSE | Test MAPE |
| --- | ---: | ---: | ---: | ---: |
| LSTM multivariate | 5 min | 1.2845 | 2.1687 | 10.6241 |

Best global test model by horizon:

| Horizon | Model | MAE | RMSE | MAPE |
| ---: | --- | ---: | ---: | ---: |
| 5 min | LSTM multivariate | 1.2845 | 2.1687 | 10.6241 |
| 15 min | Moving average | 1.4778 | 2.3374 | 12.8960 |
| 30 min | Moving average | 1.6402 | 2.5160 | 14.2721 |

## Known Limitations

- Point forecasts only; no prediction intervals yet.
- Risk-signal analysis uses train-only normal ranges, not failure labels.
- Ridge autoregression has numerical-stability warnings caused by correlated lag features.
- MAPE is unstable for sensors that can approach zero; results report MAPE only where `abs(y_true) >= 0.1`.
- LSTM training is bounded for local reproducibility and is not hyperparameter-tuned.

## Next Steps

1. Add residual-based forecast anomaly scores.
2. Add uncertainty intervals.
3. Evaluate forecast residuals near curated MetroPT failure events.
4. Integrate forecasting outputs into a deterministic degradation evidence engine.
