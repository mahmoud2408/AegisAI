# MetroPT Target Analysis

Phase 7B audits whether the MetroPT early-warning target is scientifically appropriate before tuning models. The answer is mixed: the target can be constructed without detected temporal leakage, but the positive class is extremely sparse and depends on only four report-curated events.

## Horizon Selection

The raw MetroPT cadence is near 10 seconds. Phase 7B keeps the Phase 7 prediction stride of 6 raw rows, so model opportunities occur roughly once per minute. The tested horizons were chosen from that measured cadence:

| Horizon | Approximate prediction rows per event | Rationale |
| ---: | ---: | --- |
| 0.5 hours | 30 | Immediate precursor window |
| 2.0 hours | 120 | Short operational warning |
| 6.0 hours | 360 | Medium horizon used by Phase 7 |

No horizon was selected from test performance. The best horizon/model candidate is selected with validation metrics only.

## Target Definition

For each timestamp `t`:

```text
target(t) = 1 if a curated failure starts in (t, t + horizon]
target(t) = 0 otherwise
```

Rows during active failure intervals are marked ineligible and removed from supervised training/evaluation. Features use only current and historical observations.

## Leakage Audit

| Horizon | Positive raw rows | Modeling rows | Active failures in modeling | Invalid positive labels | Split-window boundary crossings | Leakage detected |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 694 | 247632 | 0 | 0 | 0 | No |
| 2.0 | 2776 | 247632 | 0 | 0 | 0 | No |
| 6.0 | 8327 | 247632 | 0 | 0 | 0 | No |

The temporal split boundaries do not fall inside failure intervals, and no tested target window crosses from validation into train or from test into validation.

## Class Imbalance

| Horizon | Split | Samples | Positives | Positive ratio |
| ---: | --- | ---: | ---: | ---: |
| 0.5 | Train | 139605 | 56 | 0.000401 |
| 0.5 | Validation | 6877 | 31 | 0.004508 |
| 0.5 | Test | 101150 | 30 | 0.000297 |
| 2.0 | Train | 139605 | 220 | 0.001576 |
| 2.0 | Validation | 6877 | 121 | 0.017595 |
| 2.0 | Test | 101150 | 120 | 0.001186 |
| 6.0 | Train | 139605 | 661 | 0.004735 |
| 6.0 | Validation | 6877 | 363 | 0.052785 |
| 6.0 | Test | 101150 | 363 | 0.003589 |

The validation split has a much higher positive ratio than train and test because the June failure is long and near the validation window. This is an important distribution issue, not a model-quality improvement.

## Per-Event Positive Rows

| Horizon | April train | May train | June validation | July test |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 25 | 31 | 31 | 30 |
| 2.0 | 99 | 121 | 121 | 120 |
| 6.0 | 298 | 363 | 363 | 363 |

The May, June, and July counts are close to the expected one-minute stride. The April counts are lower because the first event starts at midnight and the configured rolling/lag feature warm-up removes some early rows.

## Pre-Failure Buffer Findings

Phase 7B compares `pre_10min_to_start` against `pre_6h_to_1h` for each event. The largest standardized changes are:

| Event | Sensor | Mean delta | Standardized delta |
| --- | --- | ---: | ---: |
| June validation | DV_pressure | 2.2202 | 14.6981 |
| May train | DV_pressure | 1.9467 | 11.5326 |
| May train | H1 | -7.9892 | -2.7946 |
| June validation | H1 | -7.9122 | -2.6341 |
| May train | TP2 | 7.2014 | 2.4896 |
| July test | Oil_temperature | 5.4258 | 2.4630 |

The May and June failures show strong pressure-related immediate pre-failure changes. The July held-out event's strongest immediate shift is oil temperature, with weaker pressure changes. That mismatch helps explain why the supervised model did not generalize.

## Target Assessment

The target is technically valid under the implemented checks, but it is statistically fragile. Four events cannot support robust supervised learning, and the validation/test event behavior differs enough that validation-selected thresholds do not transfer.
