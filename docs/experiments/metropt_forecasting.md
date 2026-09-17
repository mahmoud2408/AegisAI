# MetroPT Forecasting

Phase 8 moves MetroPT-3 from supervised failure prediction to time-series forecasting. The experiment asks:

> How accurately can AegisAI forecast future MetroPT machine telemetry, and how does horizon length affect forecast-based degradation signals?

## Current Run

- Run: `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915`
- Script: `scripts/experiments/run_metropt_forecasting.py`
- Config: `configs/experiments/metropt_forecasting.yaml`
- Raw rows: `1516948`
- Raw timestamp coverage: 2020-02-01 00:00:00 to 2020-09-01 03:59:50
- Raw median cadence: `10.0` seconds
- Forecasting cadence: one-minute median resample
- Resampled valid rows: `253371`
- Segment count after gap handling: `309`

## Target Sensors

Selected continuous sensors:

- `TP2`
- `TP3`
- `H1`
- `Reservoirs`
- `Oil_temperature`
- `Motor_current`

`DV_pressure` remains in the audit context but is excluded from the first forecasting target set because it is intermittent and spike-oriented. Binary state variables are audited but not forecast with continuous-regression models.

## Horizons And Splits

The 10-second raw cadence supports a one-minute forecasting view. Phase 8 evaluates:

| Horizon | Resampled steps | Rationale |
| ---: | ---: | --- |
| 5 minutes | 5 | Short operational forecast |
| 15 minutes | 15 | Medium local trajectory |
| 30 minutes | 30 | Longer local trajectory |

Splits reuse the Phase 7B chronology:

| Split | Rule | Forecast windows at 5 min |
| --- | --- | ---: |
| Train | timestamp < 2020-05-31 00:00 | 26712 |
| Validation | 2020-05-31 00:00 <= timestamp < 2020-06-08 00:00 | 1846 |
| Test | timestamp >= 2020-06-08 00:00 | 19137 |

Window audits reported zero timestamp-alignment errors and zero target-step errors for all horizons.

## Models

| Model | Type | Notes |
| --- | --- | --- |
| Persistence | Univariate deterministic baseline | Future equals last observed value |
| Moving average | Univariate deterministic baseline | Future equals causal tail average |
| Rolling linear trend | Statistical baseline | Per-window least-squares extrapolation |
| Ridge autoregression | Multivariate statistical/ML baseline | Flattened lag window across sensors |
| LSTM univariate | Deep sequence model | Pooled one-sensor histories |
| LSTM multivariate | Deep sequence model | All selected sensors jointly |

ARIMA was not used as the default statistical baseline because the resampled MetroPT series contains long temporal gaps, operating-regime changes, and highly correlated sensor pairs. The phase uses causal rolling linear trend and multivariate ridge autoregression as lower-cost statistical baselines for rolling-origin evaluation.

The Transformer was excluded in Phase 8. Lower-complexity baselines and LSTM diagnostics were required first, and the current run already shows useful model-complexity tradeoffs.

## Global Test Results

| Model | Horizon min | MAE | RMSE | MAPE | Training seconds | Inference seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM multivariate | 5 | 1.2845 | 2.1687 | 10.6241 | 12.6751 | 3.5419 |
| Ridge autoregression | 5 | 1.3298 | 2.2318 | 11.5940 | 0.1078 | 0.0910 |
| LSTM univariate | 5 | 1.3978 | 2.3315 | 11.3082 | 102.7025 | 28.2508 |
| Moving average | 15 | 1.4778 | 2.3374 | 12.8960 | 0.0000 | 0.0279 |
| LSTM multivariate | 15 | 1.5428 | 2.4194 | 12.4527 | 14.7372 | 5.3645 |
| LSTM univariate | 15 | 1.5771 | 2.4454 | 14.1648 | 87.2528 | 22.6292 |
| Moving average | 30 | 1.6402 | 2.5160 | 14.2721 | 0.0000 | 0.0177 |
| LSTM univariate | 30 | 1.6911 | 2.5511 | 14.1575 | 82.5675 | 20.0383 |
| Ridge autoregression | 15 | 1.7139 | 2.5559 | 14.7732 | 0.0811 | 0.0919 |
| Rolling linear trend | 15 | 1.6267 | 2.6096 | 14.2482 | 0.0000 | 0.2292 |
| Ridge autoregression | 30 | 1.7834 | 2.6612 | 15.4329 | 0.0610 | 0.0871 |
| Moving average | 5 | 1.8163 | 2.7046 | 15.8871 | 0.0000 | 0.0215 |
| LSTM multivariate | 30 | 1.8924 | 2.7468 | 15.7056 | 9.5295 | 1.5511 |
| Rolling linear trend | 5 | 1.9618 | 2.9285 | 17.1532 | 0.0000 | 0.2791 |
| Persistence | 15 | 1.8832 | 3.3641 | 17.4954 | 0.0000 | 0.0049 |
| Rolling linear trend | 30 | 2.3316 | 3.5512 | 20.1893 | 0.0000 | 0.2069 |
| Persistence | 5 | 2.1515 | 3.5896 | 20.1259 | 0.0000 | 0.0050 |
| Persistence | 30 | 2.2850 | 3.6951 | 21.0455 | 0.0000 | 0.0045 |

The validation-selected global candidate is also the best test-ranked global candidate:

| Selection | Model | Horizon | Test MAE | Test RMSE | Test MAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation-selected | LSTM multivariate | 5 min | 1.2845 | 2.1687 | 10.6241 |

## Horizon Effect

| Horizon | Best global test model | MAE | RMSE | MAPE |
| ---: | --- | ---: | ---: | ---: |
| 5 min | LSTM multivariate | 1.2845 | 2.1687 | 10.6241 |
| 15 min | Moving average | 1.4778 | 2.3374 | 12.8960 |
| 30 min | Moving average | 1.6402 | 2.5160 | 14.2721 |

Longer horizons are harder globally. The simple moving-average baseline is competitive at 15 and 30 minutes, which means future sequence models must beat a strong low-compute baseline before they are worth operational integration.

## Per-Sensor Findings

Best test result per sensor:

| Sensor | Best model | Horizon | MAE | RMSE | MAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| Reservoirs | LSTM univariate | 5 min | 0.2981 | 0.4373 | 3.3464 |
| TP3 | LSTM univariate | 5 min | 0.2987 | 0.4382 | 3.3529 |
| Motor_current | LSTM multivariate | 5 min | 1.1042 | 1.4812 | 24.5475 |
| TP2 | LSTM multivariate | 5 min | 1.7485 | 2.6536 | 55.9764 |
| H1 | LSTM multivariate | 5 min | 1.8632 | 2.6792 | 17.5906 |
| Oil_temperature | Ridge autoregression | 5 min | 2.3259 | 3.1982 | 3.4660 |

`TP3` and `Reservoirs` are easiest because they are tightly coupled and comparatively smooth. `TP2`, `H1`, and `Motor_current` are harder because they react more sharply to compressor operating transitions. `Oil_temperature` has large absolute-scale excursions, so RMSE is high even when relative error is modest.

## Multivariate Versus Univariate

Multivariate modeling improves the best global 5-minute result:

| Model | Horizon | Global test RMSE |
| --- | ---: | ---: |
| LSTM multivariate | 5 min | 2.1687 |
| LSTM univariate | 5 min | 2.3315 |
| Ridge autoregression | 5 min | 2.2318 |

The improvement is not uniform per sensor. `TP3` and `Reservoirs` are slightly better with the pooled univariate LSTM, while `TP2`, `H1`, and `Motor_current` benefit from multivariate context.

## Error Analysis

Hard cases concentrate around:

- rapid compressor state transitions;
- pressure and current shifts;
- large oil-temperature excursions;
- long temporal gaps that require segment-local windowing;
- sensor pairs with near-duplicate behavior, especially `TP3` and `Reservoirs`.

The ridge baseline emitted ill-conditioned matrix warnings during the full run, consistent with highly correlated lag features. Those warnings are retained as a numerical-stability limitation, not hidden.

## Uncertainty

Prediction intervals are not implemented in Phase 8. All results are point forecasts, so they should not be presented as certain. A future phase should add residual quantile intervals, conformal intervals, or probabilistic forecasting.

## Conclusion

Forecasting is worth integrating into the future AegisAI risk engine as an evidence signal, not as a failure classifier. The 5-minute multivariate LSTM has the best measured global accuracy, while moving average remains a strong longer-horizon baseline. The next phase should connect forecast residuals and normal-range violations to a deterministic degradation evidence layer before adding RCA or agentic workflows.
