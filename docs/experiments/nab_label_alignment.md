# NAB Label Alignment Audit

Phase: 4
Status: completed from processed local NAB artifacts
Run artifact: `experiments/anomaly/nab/deep_autoencoders/run_phase4_deep_anomaly/label_alignment.json`

## Summary

| Field | Value |
| --- | ---: |
| Series | 58 |
| Observations | 365,558 |
| Labeled series | 52 |
| Unlabeled series | 6 |
| Anomaly windows | 116 |
| Point labels | 120 |
| Series with multiple windows | 35 |
| Windows with observations | 116 |
| Windows without observations | 0 |
| Windows outside observation range | 0 |
| Windows with null boundaries | 0 |
| Windows with exact start timestamp | 116 |
| Windows with exact end timestamp | 116 |
| Point labels with exact timestamp match | 120 |
| Point labels without exact timestamp match | 0 |

## Label Format

Processed NAB labels are represented in the canonical label table:

- point labels: `label_value = anomaly`, using `timestamp`
- anomaly windows: `label_value = anomaly_window`, using `timestamp_start` and `timestamp_end`

Phase 4 uses NAB anomaly windows as the main temporal evaluation target. Point labels remain available and are used only as a fallback for a series without anomaly windows.

## Boundary Policy

NAB windows are mapped to observations as closed intervals:

```text
timestamp >= timestamp_start and timestamp <= timestamp_end
```

All 116 anomaly windows have exact start and end timestamp matches in the processed observations. A half-open interval would exclude exactly 116 endpoint observations, one per window. The Phase 3 closed-interval policy is therefore retained.

## Correctly Aligned Examples

| Entity | First Positive Timestamp | Last Positive Timestamp | Positive Observations |
| --- | --- | --- | ---: |
| `nab:artificialWithAnomaly/art_daily_flatmiddle` | 2014-04-10 07:15:00 | 2014-04-11 16:45:00 | 403 |
| `nab:artificialWithAnomaly/art_daily_jumpsdown` | 2014-04-10 16:15:00 | 2014-04-12 01:45:00 | 403 |
| `nab:artificialWithAnomaly/art_daily_jumpsup` | 2014-04-10 16:15:00 | 2014-04-12 01:45:00 | 403 |
| `nab:artificialWithAnomaly/art_daily_nojump` | 2014-04-10 16:15:00 | 2014-04-12 01:45:00 | 403 |
| `nab:artificialWithAnomaly/art_increase_spike_density` | 2014-04-07 06:25:00 | 2014-04-08 15:55:00 | 403 |
| `nab:artificialWithAnomaly/art_load_balancer_spikes` | 2014-04-10 11:50:00 | 2014-04-11 21:20:00 | 403 |

## Edge Cases

Six processed series have no positive observations after alignment because they have no point labels and no anomaly windows:

| Entity | Windows | Point Labels |
| --- | ---: | ---: |
| `nab:artificialNoAnomaly/art_daily_no_noise` | 0 | 0 |
| `nab:artificialNoAnomaly/art_daily_perfect_square_wave` | 0 | 0 |
| `nab:artificialNoAnomaly/art_daily_small_noise` | 0 | 0 |
| `nab:artificialNoAnomaly/art_flatline` | 0 | 0 |
| `nab:artificialNoAnomaly/art_noisy` | 0 | 0 |
| `nab:realAWSCloudwatch/ec2_cpu_utilization_c6585a` | 0 | 0 |

These are valid negative/no-window series, not alignment failures.

## Evaluation Impact

The label audit did not require changing the underlying Phase 3 label mapping.

Phase 4 does change the comparison protocol:

- autoencoder scores are assigned to the window-ending timestamp
- baselines are evaluated on the same window-ending test timestamps
- scalers are fit on training data only
- labeled anomalous training observations/windows are excluded from unsupervised fitting and train-percentile threshold estimation
- test labels are still evaluation-only

Because Phase 4 uses a common window-end test row set and train-percentile thresholds, its baseline numbers should not be treated as a direct replacement for the Phase 3 baseline run.
