# Forecast-Based Degradation Signal

Phase 8 investigates whether future sensor forecasts can support a degradation/risk signal. This is not a failure-prediction model.

## Definition

For each forecasted future sensor value:

```text
forecast warning = forecasted value outside train non-failure quantile range
future out-of-range = actual future value outside train non-failure quantile range
```

The normal range is estimated from training data only, excluding curated active MetroPT failure intervals. Phase 8 uses the 1st and 99th training quantiles.

## Current Artifact

- Run: `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915`
- Risk rows: `forecast_risk_signal.csv`
- Summary: `risk_signal_summary.json`
- Plot: `figures/forecast_risk_signal.png`

The saved risk-signal rows are based on a deterministic forecast sample used for visualization and investigation, not every possible test forecast row.

## Summary

Across sampled validation-selected forecast rows:

| Metric | Value |
| --- | ---: |
| Mean precision | 0.3296 |
| Mean recall | 0.0698 |
| Mean F1 | 0.1030 |

This means the signal is conservative for most sensors: when it warns, it can be informative for some variables, but it misses many future out-of-range values.

## Sensor Patterns

| Horizon | Sensor | Forecast warning rate | Future out-of-range rate | Precision | Recall | F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | TP2 | 0.1013 | 0.0240 | 0.0000 | 0.0000 | 0.0000 |
| 15 | Oil_temperature | 0.0060 | 0.0093 | 0.8333 | 0.5357 | 0.6522 |
| 15 | Reservoirs | 0.0023 | 0.0200 | 1.0000 | 0.1167 | 0.2090 |
| 15 | TP3 | 0.0023 | 0.0200 | 1.0000 | 0.1167 | 0.2090 |
| 30 | Oil_temperature | 0.0033 | 0.0093 | 0.9000 | 0.3214 | 0.4737 |
| 30 | Reservoirs | 0.0017 | 0.0230 | 0.6000 | 0.0435 | 0.0811 |
| 30 | TP3 | 0.0017 | 0.0227 | 0.6000 | 0.0441 | 0.0822 |

The oil-temperature signal is the strongest measured case. TP2 produces warnings at the 5-minute horizon but those sampled warnings do not align with future out-of-range values under the train-quantile definition.

## Interpretation

Forecast trajectories can provide an additional risk signal, especially when a forecast moves outside the learned train-normal range before the actual future value does. The current signal is not strong enough to use alone. It should be combined with:

- current anomaly scores;
- forecast residuals;
- state-transition context;
- service or machine dependency context;
- later evidence from logs, traces, and documentation.

## Limitations

- No probabilistic uncertainty or prediction intervals yet.
- Normal ranges are empirical quantiles, not physics-informed safety thresholds.
- The analysis does not use failure labels and does not claim failure prediction.
- Saved risk rows are sampled for investigation. Full production scoring should evaluate every online forecast.

## Next Step

Use the Phase 8 forecasts to build a deterministic degradation evidence layer: current abnormality, forecasted abnormality, residual magnitude, sensor trend, and operating-state transition should be separate evidence fields.
