# Model Card: Isolation Forest NAB Baseline

Model name: `isolation_forest_nab`
Version: `run_phase3_baseline`
Status: experimental baseline, not registered for production inference
Date: 2026-09-03

## Model Details

This model card documents the Phase 3 Isolation Forest anomaly-detection baseline for NAB. The experiment trains one unsupervised Isolation Forest per NAB time series using causal univariate time-series features.

Model family: scikit-learn Isolation Forest

Implementation:

```text
src/aegis_ai/ml/anomaly/isolation_forest.py
src/aegis_ai/ml/anomaly/nab_experiment.py
scripts/experiments/run_nab_baseline.py
```

## Intended Use

Intended use:

- establish a reproducible unsupervised anomaly-detection baseline
- compare against a transparent statistical rolling z-score baseline
- produce inspectable metrics, figures, predictions, and error-analysis artifacts
- support interview discussion around leakage-safe time-series evaluation

Out-of-scope use:

- production incident paging
- root-cause analysis
- incident severity classification
- multivariate service diagnosis
- causal claims about incidents
- official NAB leaderboard comparison

## Data

Dataset: processed Numenta Anomaly Benchmark (NAB)

Input artifacts:

```text
data/processed/metrics/nab/
data/processed/labels/nab/part-00000.parquet
```

Evaluation target:

- NAB window labels from `combined_windows.json` are used as generic temporal evaluation windows.
- NAB point labels from `combined_labels.json` are used only when a series has no window labels.
- Labels are not used for model fitting or threshold selection.

Split policy:

| Split | Fraction | Use |
| --- | ---: | --- |
| Train | 0.50 | Fit the per-series Isolation Forest |
| Validation | 0.20 | Select the 99th percentile score threshold |
| Test | 0.30 | Final evaluation only |

## Inputs

Each model receives a finite causal feature matrix with these columns:

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

Rolling features use shifted historical values so the current timestamp is not included in its own baseline.

## Outputs

The detector returns:

- anomaly score, where larger means more anomalous
- binary prediction, using a validation-derived threshold

Threshold policy:

```text
threshold = validation_score_quantile(0.99)
```

## Hyperparameters

| Parameter | Value |
| --- | --- |
| `n_estimators` | 200 |
| `max_samples` | `auto` |
| `contamination` | `auto` |
| `random_state` | 42 |
| `n_jobs` | -1 |

## Metrics

Measured on `run_phase3_baseline` test split:

| Metric | Value |
| --- | ---: |
| Precision | 0.2928 |
| Recall | 0.1316 |
| F1 | 0.1815 |
| PR-AUC | 0.2147 |
| False positive rate | 0.0384 |
| True positives | 1,555 |
| False positives | 3,756 |
| True negatives | 94,119 |
| False negatives | 10,265 |
| Detected windows | 36 |
| Missed windows | 8 |
| Median delay steps | 47.0 |
| Median delay seconds | 20,400.0 |

Baseline comparison:

| Model | Precision | Recall | F1 | PR-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling z-score | 0.0589 | 0.0139 | 0.0225 | 0.1185 | 0.0268 |
| Isolation Forest | 0.2928 | 0.1316 | 0.1815 | 0.2147 | 0.0384 |

## Known Limitations

- Median per-series F1 is zero, indicating uneven performance across NAB series.
- Recall is low, so many anomalous points inside evaluation windows are missed.
- Detection delays are high for several windows.
- False positives increase relative to the rolling z-score baseline.
- The model is univariate per series and does not use service dependencies, logs, traces, or multivariate context.
- The experiment does not compute the official NAB normalized score.
- Isolation Forest feature importances are not directly interpretable; explainability is deferred to later phases.

## Failure Modes

Observed failure modes:

- delayed detection in long Twitter-volume anomaly windows
- missed artificial daily anomalies when validation thresholds are too high
- high-scoring false positives during large but unlabeled traffic changes
- threshold brittleness across heterogeneous metric distributions

## Monitoring Considerations

If promoted beyond experiment status, monitor:

- inference latency per series
- score distribution drift
- predicted anomaly rate
- false positive feedback from incident reviews
- missed incident windows
- retraining data coverage and freshness

## Reproducibility

Run command:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_nab_baseline.py --run-id run_phase3_baseline
```

Primary artifacts:

```text
experiments/anomaly/nab/isolation_forest/run_phase3_baseline/
```

Serialized models:

```text
experiments/anomaly/nab/isolation_forest/run_phase3_baseline/models/*.joblib
```

The serialized models are local experiment artifacts and are not committed to Git.
