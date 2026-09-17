# NAB Anomaly Detection Baseline

Phase: 3
Status: completed with measured local artifacts
Run ID: `run_phase3_baseline`
Run date: 2026-09-03

## Objective

This experiment establishes the first real anomaly-detection benchmark for AegisAI using the processed Numenta Anomaly Benchmark (NAB) dataset. It compares a causal rolling z-score baseline against an Isolation Forest detector using leakage-safe chronological splits.

The goal is not to maximize performance yet. The goal is to create a reproducible, inspectable baseline with honest metrics, error analysis, figures, and model artifacts.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_nab_baseline.py --run-id run_phase3_baseline
```

Primary artifacts are written under:

```text
experiments/anomaly/nab/isolation_forest/run_phase3_baseline/
```

The `experiments/` directory is intentionally ignored by Git except for `.gitkeep`. The run can be regenerated from processed dataset artifacts.

## Dataset

Source dataset: Numenta Anomaly Benchmark (NAB)

Input path:

```text
data/processed/metrics/nab/
data/processed/labels/nab/part-00000.parquet
```

Processed dataset summary:

| Field | Value |
| --- | ---: |
| Series | 58 |
| Observations | 365,558 |
| Point labels | 120 |
| Scoring windows | 116 |
| Evaluated test rows | 109,695 |
| Positive test points from windows | 11,820 |

Label policy:

- Point labels come from NAB `combined_labels.json`.
- Interval labels come from NAB `combined_windows.json`.
- This experiment uses interval labels as a generic temporal point/window evaluation target.
- If a series has no interval window labels, exact point labels are used as a fallback.

Important: this experiment does not report the official normalized NAB score. Official NAB scoring has its own scoring profiles and reward/penalty windows. The current implementation reports standard temporal anomaly metrics only.

## Split Protocol

Each series is split independently and chronologically:

| Split | Fraction |
| --- | ---: |
| Train | 0.50 |
| Validation | 0.20 |
| Test | 0.30 |

Leakage controls:

- Models fit only on the train split.
- Thresholds are selected from validation anomaly scores using the 99th percentile.
- Labels are evaluation-only and are not used for fitting or threshold selection.
- Features are causal: rolling statistics use shifted historical values, so the current timestamp is not included in its own historical baseline.
- Final metrics are reported only on the test split.

## Features

The experiment uses univariate causal time-series features:

- raw value
- first difference
- percentage change
- short rolling mean
- short rolling standard deviation
- short rolling z-score
- long rolling mean
- long rolling standard deviation
- long rolling z-score
- short trend estimate

Default feature configuration:

| Parameter | Value |
| --- | ---: |
| Short window | 5 |
| Long window | 20 |
| Trend window | 5 |
| Minimum periods | 3 |
| Epsilon | 1e-8 |

## Models

### Rolling Z-Score

The statistical baseline scores each point by the absolute long-window rolling z-score. The decision threshold is the 99th percentile of validation scores for the same series.

### Isolation Forest

One Isolation Forest model is trained per series using the causal feature matrix from that series.

Hyperparameters:

| Parameter | Value |
| --- | --- |
| `n_estimators` | 200 |
| `max_samples` | `auto` |
| `contamination` | `auto` |
| `random_state` | 42 |
| `n_jobs` | -1 |

Isolation Forest scores are transformed so larger scores mean more anomalous.

## Global Test Metrics

Measured on `run_phase3_baseline`:

| Model | Precision | Recall | F1 | PR-AUC | FPR | Detected Windows | Missed Windows | Median Delay Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.0589 | 0.0139 | 0.0225 | 0.1185 | 0.0268 | 38 | 6 | 41.0 |
| Isolation Forest | 0.2928 | 0.1316 | 0.1815 | 0.2147 | 0.0384 | 36 | 8 | 47.0 |

Detailed Isolation Forest counts:

| Count | Value |
| --- | ---: |
| True positives | 1,555 |
| False positives | 3,756 |
| True negatives | 94,119 |
| False negatives | 10,265 |
| Test positives | 11,820 |
| Predicted positives | 5,311 |

## Per-Series Distribution

| Model | Mean F1 | Median F1 | Mean PR-AUC | Median PR-AUC | Mean FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.0159 | 0.0000 | 0.2268 | 0.2170 | 0.0329 |
| Isolation Forest | 0.0874 | 0.0000 | 0.3364 | 0.3036 | 0.0566 |

The median F1 is zero for both models, which is a meaningful result. It shows that a single unsupervised thresholding strategy is brittle across heterogeneous NAB series.

## Strongest Isolation Forest Series

| Entity | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| `nab:artificialWithAnomaly/art_load_balancer_spikes` | 0.8344 | 0.4159 | 0.5551 | 0.7025 |
| `nab:realAWSCloudwatch/rds_cpu_utilization_cc0c53` | 0.3169 | 0.7537 | 0.4462 | 0.3422 |
| `nab:realAdExchange/exchange-4_cpm_results` | 0.5263 | 0.3659 | 0.4317 | 0.4520 |
| `nab:realKnownCause/cpu_utilization_asg_misconfiguration` | 0.7024 | 0.2141 | 0.3282 | 0.6010 |
| `nab:realAWSCloudwatch/ec2_cpu_utilization_ac20cd` | 0.4870 | 0.2333 | 0.3154 | 0.4820 |

## Error Analysis

Isolation Forest missed several windows where the maximum score stayed below the validation-derived threshold. Examples include:

| Entity | Window Start | Window End | Points | Max Score |
| --- | --- | --- | ---: | ---: |
| `nab:realAWSCloudwatch/ec2_cpu_utilization_24ae8d` | 2014-02-27 08:55:00 | 2014-02-28 01:35:00 | 201 | 0.6666 |
| `nab:artificialWithAnomaly/art_daily_jumpsdown` | 2014-04-10 19:10:00 | 2014-04-12 01:45:00 | 368 | 0.6610 |
| `nab:artificialWithAnomaly/art_daily_flatmiddle` | 2014-04-10 19:10:00 | 2014-04-11 16:45:00 | 260 | 0.6483 |
| `nab:artificialWithAnomaly/art_daily_nojump` | 2014-04-10 19:10:00 | 2014-04-12 01:45:00 | 368 | 0.6027 |
| `nab:realKnownCause/rogue_agent_key_hold` | 2014-07-17 12:30:00 | 2014-07-18 06:45:00 | 18 | 0.5993 |

Long delayed detections were common in some Twitter-volume series:

| Entity | Delay Steps | First Detection |
| --- | ---: | --- |
| `nab:realTweets/Twitter_volume_IBM` | 291 | 2015-04-20 11:17:53 |
| `nab:realTweets/Twitter_volume_KO` | 263 | 2015-04-14 14:47:53 |
| `nab:realTweets/Twitter_volume_CVS` | 178 | 2015-04-14 20:57:53 |
| `nab:artificialWithAnomaly/art_daily_jumpsup` | 166 | 2014-04-11 09:00:00 |
| `nab:realTweets/Twitter_volume_PFE` | 155 | 2015-04-07 20:07:53 |

High-scoring false positives appeared in `nab:realTweets/Twitter_volume_AAPL`, where large traffic changes were scored as anomalous outside the evaluation windows.

## Interpretation

Isolation Forest outperformed rolling z-score on global precision, recall, F1, and PR-AUC. It also produced a substantially stronger global F1 score: 0.1815 versus 0.0225.

The improvement came with tradeoffs. Isolation Forest increased false positive rate from 0.0268 to 0.0384 and detected fewer labeled windows than the rolling z-score baseline. Both models showed long detection delays, and both had median per-series F1 of zero.

This is a credible Phase 3 baseline, not a solved anomaly detector.

## Limitations

- Official NAB normalized scoring is not implemented yet.
- The point-expanded window target can penalize late detections heavily, even if a model fires inside the window.
- Per-series 99th percentile thresholds are simple and unsupervised, but brittle.
- Seasonal structure is only weakly represented by rolling statistics.
- No multivariate context is available in this NAB experiment.
- No root-cause, incident severity, RAG, forecasting, agentic investigation, or explainability layer is included in Phase 3.

## Artifacts

| Artifact | Description |
| --- | --- |
| `metrics.json` | Global metrics, per-series distributions, best/worst series |
| `per_series_metrics.csv` | Metrics for each model and NAB series |
| `predictions.csv` | Test predictions and anomaly scores |
| `error_analysis.json` | False-positive, missed-window, and delayed-window examples |
| `models/*.joblib` | One serialized Isolation Forest per series |
| `figures/*.png` | Time-series, score, and comparison plots |
| `run_report.md` | Compact generated run report |

## Next Work

Recommended next milestone: Phase 4, SMD multivariate anomaly detection.

Before Phase 4, useful refinements include:

- add official NAB scoring support if the original NAB scoring implementation is vendored or reproduced faithfully
- add threshold sensitivity analysis
- add robust seasonal baselines
- add model calibration plots for score thresholds
- improve visualization by sampling or faceting dense long series
