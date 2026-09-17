# NAB Error Analysis

Phase: 5  
Source run: `experiments/anomaly/nab/deep_autoencoders/run_phase4_deep_anomaly`  
Analysis run: `experiments/anomaly/nab/robustness/run_phase5_nab_robustness_final`

This report analyzes Phase 4 NAB failures using existing predictions, model scores, thresholds, and saved checkpoints. It does not add a new production model and does not report official NAB normalized scores.

## Evaluation Scope

- Evaluated series: 58
- Test rows scored: 107,897
- Test positive points: 11,514
- Test anomaly windows: 42
- Score alignment: neural window scores are assigned to the window-ending timestamp; baselines are evaluated at the same test timestamps.
- Main artifact: `per_series_error_analysis.csv`

## Difficulty Criteria

For series with test anomalies:

| Class | Objective criterion |
| --- | --- |
| easy | best F1 >= 0.50 and best recall >= 0.40 |
| moderate | best F1 >= 0.20 and not easy |
| difficult | best F1 > 0 and < 0.20 |
| complete_failure | best F1 = 0 |

For series with no test anomalies:

| Class | Objective criterion |
| --- | --- |
| easy | max model FPR <= 0.01 |
| moderate | max model FPR <= 0.05 |
| difficult | max model FPR <= 0.15 |
| complete_failure | max model FPR > 0.15 |

Measured classification:

| Class | Series |
| --- | ---: |
| easy | 10 |
| moderate | 31 |
| difficult | 12 |
| complete_failure | 5 |

Two positive-test series were complete point-metric failures for all four models:

| Series | Test positive points | Anomaly windows |
| --- | ---: | ---: |
| `nab:artificialWithAnomaly/art_daily_nojump` | 337 | 1 |
| `nab:realAWSCloudwatch/ec2_disk_write_bytes_c0d644` | 135 | 3 |

## Representative Failure Cases

Plots are stored in `experiments/anomaly/nab/robustness/run_phase5_nab_robustness_final/figures/`.

| Case | Evidence | Supported interpretation |
| --- | --- | --- |
| LSTM false positive on `Twitter_volume_AAPL` | score 2808.93 vs threshold 28.34; target label 0; local context std 1928.93 | Very high non-stationary traffic burst outside the NAB scoring window; the model treats it as anomalous, but the point labels mark it normal. |
| Dense missed `ec2_cpu_utilization_24ae8d` window | 201 positive points; max score 1.17 vs threshold 8.87; score/threshold ratio 0.13 | Reconstruction score never approaches the learned threshold, so this is primarily a threshold and signal-strength miss. |
| Isolation Forest missed same `ec2_cpu_utilization_24ae8d` window | max score 0.6666 vs threshold 0.7093; score/threshold ratio 0.94 | The anomaly was close to the operating threshold but still below it; a lower threshold may detect it at the cost of more false positives. |
| LSTM delayed `cpu_utilization_asg_misconfiguration` | 1499 positive points; first detection after 375 steps; max score 5.24 vs threshold 2.46 | The model eventually reacts to the sustained shift, but not near the window start. |
| Dense delayed `machine_temperature_system_failure` | 567 positive points; first detection after 543 steps; score/threshold ratio 8.75 at peak | Reconstruction error rises late in the labeled interval, producing poor incident timeliness despite eventual detection. |
| Dense noisy-normal false positive on `Twitter_volume_AAPL` | metric value 9310; score 2085.27 vs threshold 25.96; label 0 | Large unlabelled bursts create operationally plausible anomalies that are counted as false positives under point labels. |

## Supported Failure Drivers

Threshold mismatch is strongly supported. Dense AE improved from F1 0.2835 with train-p99 to 0.3141 with validation-p99, while train-MAD over-fired with FPR 0.1854. IF performed best at train-p99; validation-p99 reduced FPR but also reduced recall and F1.

Heterogeneity is strongly supported. Median per-series F1 is 0 for every model even though positive-test nonzero-F1 rates are 86.7% for IF, Dense AE, and LSTM AE. This means many zero-F1 rows are no-anomaly or high-FPR normal series, while positive series still include several hard misses.

Short or low-magnitude anomaly intervals are supported as a failure mode. The `ec2_cpu_utilization_24ae8d` miss stayed below both Dense and IF thresholds, while `nyc_taxi` contains a 32-point missed window in the IF/Dense overlap table.

Non-stationarity and noisy normal periods are supported by the `Twitter_volume_AAPL` false positives, where very large bursts occur outside labeled anomaly windows.

Univariate feature limits remain a likely limitation, but Phase 5 cannot prove this from NAB alone. NAB provides one metric per series, so it cannot test whether cross-metric dependencies would disambiguate noise, workload shifts, or root-cause patterns.

## Limitations

- These are point/window metrics derived from NAB labels; they are not official NAB scores.
- Failure explanations are observational. They do not claim causality.
- Window sensitivity is a bounded diagnostic on six labeled series, not a full benchmark rerun.
