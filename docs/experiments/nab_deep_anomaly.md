# NAB Deep Anomaly Detection

Phase: 4
Status: completed with measured local artifacts
Run ID: `run_phase4_deep_anomaly`
Run date: 2026-09-04

## Objective

This experiment tests whether temporal representation learning improves NAB anomaly detection compared with Phase 3 classical baselines. It compares:

- rolling z-score
- Isolation Forest
- Dense Autoencoder
- LSTM Autoencoder

The experiment does not assume that deep learning will win. It measures the result under a common leakage-aware protocol.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_nab_deep_anomaly.py --run-id run_phase4_deep_anomaly
```

Configuration:

```text
configs/experiments/nab_deep_anomaly.yaml
```

Primary artifacts:

```text
experiments/anomaly/nab/deep_autoencoders/run_phase4_deep_anomaly/
```

The experiment directory is ignored by Git and can be regenerated.

## Label Alignment

Before training, NAB labels were audited against processed observations. The audit found:

- 58 series
- 365,558 observations
- 116 anomaly windows
- 120 point labels
- 52 labeled series
- 6 unlabeled/no-window series
- no windows outside observation ranges
- no empty anomaly windows
- all point labels matched exact observation timestamps
- all anomaly windows had exact start and end timestamp matches

Conclusion: the Phase 3 closed-interval label mapping was retained.

Detailed audit: `docs/experiments/nab_label_alignment.md`

## Protocol

Each series is split chronologically:

| Split | Fraction |
| --- | ---: |
| Train | 0.50 |
| Validation | 0.20 |
| Test | 0.30 |

Window protocol:

| Field | Value |
| --- | ---: |
| Sequence length | 32 |
| Stride | 1 |
| Deep feature set | raw metric value |
| Normalization | standard scaling |
| Scaler fit scope | training rows excluding labeled anomalies |
| Threshold strategy | 99th percentile of training scores |

Leakage controls:

- no random observation splits
- no windows cross train/validation/test boundaries
- scaler statistics are learned from training data only
- labeled anomalous training rows/windows are excluded from model fitting
- thresholds are selected without test labels
- test metrics are computed only on window-ending timestamps shared by all four models

Score mapping:

- Dense AE and LSTM AE produce one reconstruction-error score per window.
- Each window score is assigned to the window-ending observation timestamp.
- Baseline models are evaluated on the same window-ending test timestamps.

This yields 107,897 evaluated test rows and 11,514 positive test points.

## Model Configurations

### Rolling Z-Score

Scores are absolute long-window rolling z-scores. The threshold is the 99th percentile of training scores after excluding labeled anomalous training rows.

### Isolation Forest

One Isolation Forest is trained per series on causal rolling features.

| Parameter | Value |
| --- | --- |
| `n_estimators` | 200 |
| `max_samples` | `auto` |
| `contamination` | `auto` |
| `random_state` | 42 |
| `n_jobs` | -1 |

### Dense Autoencoder

One dense autoencoder is trained per series on standardized raw-value windows.

| Parameter | Value |
| --- | --- |
| Hidden dimensions | `[32, 16]` |
| Latent dimension | 8 |
| Learning rate | 0.001 |
| Batch size | 128 |
| Max epochs | 4 |
| Early stopping patience | 2 |
| Random seed | 42 |

### LSTM Autoencoder

One LSTM autoencoder is trained per series on standardized raw-value windows.

| Parameter | Value |
| --- | --- |
| Hidden size | 16 |
| Layers | 1 |
| Dropout | 0.0 |
| Learning rate | 0.001 |
| Batch size | 128 |
| Max epochs | 4 |
| Early stopping patience | 2 |
| Random seed | 42 |

## Global Test Metrics

Measured on `run_phase4_deep_anomaly`:

| Model | Precision | Recall | F1 | PR-AUC | FPR | Detected Windows | Missed Windows | Median Delay Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.0660 | 0.0142 | 0.0233 | 0.1175 | 0.0239 | 37 | 5 | 49 |
| Isolation Forest | 0.2511 | 0.1397 | 0.1795 | 0.2093 | 0.0498 | 36 | 6 | 60 |
| Dense Autoencoder | 0.2964 | 0.2718 | 0.2835 | 0.1796 | 0.0771 | 35 | 7 | 77 |
| LSTM Autoencoder | 0.2523 | 0.2245 | 0.2376 | 0.1913 | 0.0795 | 35 | 7 | 79 |

Winners by global metric:

| Criterion | Best Model | Value |
| --- | --- | ---: |
| F1 | Dense Autoencoder | 0.2835 |
| Recall | Dense Autoencoder | 0.2718 |
| Precision | Dense Autoencoder | 0.2964 |
| PR-AUC | Isolation Forest | 0.2093 |

## Per-Series Performance

| Model | Mean F1 | Median F1 | F1=0 Series | Positive Test Series With F1=0 |
| --- | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.0163 | 0.0000 | 31 | 3 |
| Isolation Forest | 0.0834 | 0.0000 | 32 | 4 |
| Dense Autoencoder | 0.1521 | 0.0000 | 32 | 4 |
| LSTM Autoencoder | 0.1475 | 0.0000 | 32 | 4 |

The median per-series F1 remains zero for every approach. Deep learning improves aggregate detection but does not make the detector robust across all NAB series.

## Strongest Series

Dense Autoencoder strongest examples:

| Entity | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| `nab:realKnownCause/cpu_utilization_asg_misconfiguration` | 0.9953 | 0.7038 | 0.8245 | 0.8786 |
| `nab:artificialWithAnomaly/art_load_balancer_spikes` | 0.8559 | 0.6901 | 0.7641 | 0.8349 |
| `nab:artificialWithAnomaly/art_daily_jumpsup` | 1.0000 | 0.3947 | 0.5660 | 0.5737 |

LSTM Autoencoder strongest examples:

| Entity | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| `nab:artificialWithAnomaly/art_load_balancer_spikes` | 0.8489 | 0.6725 | 0.7505 | 0.8361 |
| `nab:realKnownCause/machine_temperature_system_failure` | 0.7324 | 0.4947 | 0.5905 | 0.7423 |
| `nab:realAdExchange/exchange-4_cpm_results` | 0.6563 | 0.5122 | 0.5753 | 0.4986 |

## Error Analysis

LSTM Autoencoder false positives were concentrated in `nab:realTweets/Twitter_volume_AAPL`, where very high reconstruction scores appeared outside the labeled windows. This suggests threshold sensitivity and possible unlabeled distribution shifts in high-volume social traffic.

Representative LSTM missed windows:

| Entity | Window Start | Window End | Points | Max Score |
| --- | --- | --- | ---: | ---: |
| `nab:realAWSCloudwatch/ec2_cpu_utilization_24ae8d` | 2014-02-27 08:55:00 | 2014-02-28 01:35:00 | 201 | 1.1473 |
| `nab:realKnownCause/nyc_taxi` | 2014-12-29 21:30:00 | 2015-01-03 04:30:00 | 207 | 1.0325 |
| `nab:artificialWithAnomaly/art_daily_flatmiddle` | 2014-04-10 21:45:00 | 2014-04-11 16:45:00 | 229 | 0.6104 |
| `nab:artificialWithAnomaly/art_daily_jumpsdown` | 2014-04-10 21:45:00 | 2014-04-12 01:45:00 | 337 | 0.2652 |

Representative LSTM delayed detections:

| Entity | Delay Steps | First Detection |
| --- | ---: | --- |
| `nab:realKnownCause/cpu_utilization_asg_misconfiguration` | 375 | 2014-07-11 19:44:00 |
| `nab:realKnownCause/machine_temperature_system_failure` | 301 | 2014-01-28 15:25:00 |
| `nab:realTweets/Twitter_volume_IBM` | 290 | 2015-04-20 11:12:53 |
| `nab:realTweets/Twitter_volume_KO` | 264 | 2015-04-14 14:52:53 |
| `nab:realAWSCloudwatch/ec2_cpu_utilization_ac20cd` | 206 | 2014-04-15 01:14:00 |

Observable difficulty patterns:

- some anomalies start gradually, so reconstruction error crosses threshold late
- Twitter-volume series contain large unlabeled distribution shifts
- artificial daily series can be sensitive to threshold choice and sequence alignment
- point-expanded interval evaluation penalizes late in-window detections heavily

These are hypotheses grounded in score/timestamp behavior, not causal claims.

## Ablation Study

The ablation study uses a deterministic six-series labeled subset and trains Dense Autoencoder variants only. It is intended to test protocol sensitivity without exploding compute.

| Variant | F1 | Precision | Recall | PR-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main protocol | 0.3072 | 0.8612 | 0.1869 | 0.3998 | 0.0113 |
| Window 16 | 0.2779 | 0.8943 | 0.1645 | 0.3737 | 0.0076 |
| No normalization | 0.3030 | 0.8513 | 0.1843 | 0.2871 | 0.0120 |
| Causal features | 0.2667 | 0.7319 | 0.1630 | 0.3636 | 0.0223 |
| Validation threshold | 0.3670 | 0.7126 | 0.2471 | 0.3998 | 0.0373 |

Findings:

- shorter windows reduced F1 on the ablation subset
- no normalization was close in F1 but reduced PR-AUC
- adding causal rolling features to Dense AE hurt this small subset
- validation-percentile threshold improved F1 and recall but increased false positives

The validation-threshold result is promising but should be rerun at full scale before changing the main protocol.

## Computational Cost

| Model | Total Train Seconds | Total Inference Seconds | Median Parameter Count | Max RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.00 | 0.04 | 0 | 802.86 |
| Isolation Forest | 74.39 | 6.82 | N/A | 802.88 |
| Dense Autoencoder | 60.79 | 3.03 | 3,464 | 801.69 |
| LSTM Autoencoder | 192.68 | 9.31 | 3,409 | 810.14 |

The LSTM is substantially more expensive than the Dense AE in this CPU run and did not improve global F1.

## Conclusion

Dense Autoencoder produced the best global F1, precision, and recall in this run. LSTM Autoencoder improved over Isolation Forest on F1 and recall, but it did not outperform Dense AE and had higher computational cost. Isolation Forest retained the best global PR-AUC.

The most honest conclusion is mixed: temporal reconstruction models help on aggregate, but robustness remains weak across heterogeneous NAB series. The next phase should not assume deep learning is universally superior.

## Limitations

- official normalized NAB scoring is not implemented
- model capacity and training epochs are intentionally small CPU-friendly baselines
- one model is trained per series, so cross-series representation learning is not tested
- thresholds are simple percentiles
- no multivariate service context is available in NAB
- no forecasting, incident prediction, SHAP, RAG, agent, API, or dashboard work is included in this phase

## Artifacts

| Artifact | Description |
| --- | --- |
| `config.json` | Resolved Phase 4 configuration |
| `label_alignment.json` | Label audit summary |
| `metrics.json` | Global, per-series, ranking, and efficiency metrics |
| `per_series_metrics.csv` | Metrics for every model and series |
| `predictions.csv` | Test scores and predictions |
| `error_analysis.json` | False positives, missed windows, delayed detections |
| `efficiency.csv` | Training/inference time and memory by series/model |
| `protocol_audit.csv` | Split/window/exclusion counts per series |
| `ablation_results.json` | Dense AE ablation summary |
| `models/` | Per-series serialized models |
| `figures/` | Saved visualizations |

## Next Work

Recommended next milestone: Phase 5, SMD multivariate anomaly detection.
