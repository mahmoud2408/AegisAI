# Model Card: Dense Autoencoder NAB Baseline

Model name: `dense_autoencoder_nab`
Version: `run_phase4_deep_anomaly`
Status: experimental baseline, not registered for production inference
Date: 2026-09-04

## Model Details

The Dense Autoencoder is a per-series reconstruction model trained on standardized raw-value windows from NAB. It learns to reconstruct historical normal windows, and reconstruction error is used as the anomaly score.

Implementation:

```text
src/aegis_ai/ml/anomaly/autoencoder.py
src/aegis_ai/ml/anomaly/nab_deep_experiment.py
scripts/experiments/run_nab_deep_anomaly.py
```

Architecture:

```text
32 x 1 input window -> Dense(32) -> Dense(16) -> latent(8) -> Dense(16) -> Dense(32) -> reconstruction
```

## Intended Use

Intended use:

- benchmark reconstruction-based anomaly detection on NAB
- compare against rolling z-score, Isolation Forest, and LSTM Autoencoder
- inspect per-series reconstruction failures

Out-of-scope use:

- production paging
- multivariate service diagnosis
- root-cause analysis
- official NAB leaderboard comparison
- causal explanation of incidents

## Data and Protocol

Dataset: processed NAB

Split:

- 50% train
- 20% validation
- 30% test

Training policy:

- train one model per series
- fit scaler on training rows only
- exclude labeled anomalous training windows from model fitting
- select threshold from the 99th percentile of training reconstruction errors
- evaluate on test window-ending timestamps only

## Hyperparameters

| Parameter | Value |
| --- | --- |
| Window length | 32 |
| Stride | 1 |
| Hidden dimensions | `[32, 16]` |
| Latent dimension | 8 |
| Learning rate | 0.001 |
| Batch size | 128 |
| Max epochs | 4 |
| Early stopping patience | 2 |
| Random seed | 42 |
| Device | CPU |

## Metrics

Measured on `run_phase4_deep_anomaly`:

| Metric | Value |
| --- | ---: |
| Precision | 0.2964 |
| Recall | 0.2718 |
| F1 | 0.2835 |
| PR-AUC | 0.1796 |
| False positive rate | 0.0771 |
| True positives | 3,129 |
| False positives | 7,428 |
| True negatives | 88,955 |
| False negatives | 8,385 |
| Detected windows | 35 |
| Missed windows | 7 |
| Median delay steps | 77 |
| Median delay seconds | 41,100 |

Per-series summary:

- mean F1: 0.1521
- median F1: 0.0000
- F1=0 series: 32 of 58
- positive test series with F1=0: 4 of 30

## Computational Cost

| Field | Value |
| --- | ---: |
| Total training time | 60.79 seconds |
| Mean training time per series | 1.05 seconds |
| Total inference time | 3.03 seconds |
| Median trainable parameters | 3,464 |
| Max process RSS | 801.69 MiB |

## Failure Modes

Observed failure modes:

- late threshold crossing in long anomaly windows
- false positives on unlabeled traffic shifts
- missed artificial daily anomalies under train-percentile thresholding
- uneven performance across heterogeneous NAB series

## Limitations

- univariate only
- per-series model, no shared representation
- simple percentile threshold
- no official NAB score
- no interpretability method beyond reconstruction error and observable score traces

## Reproducibility

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_nab_deep_anomaly.py --run-id run_phase4_deep_anomaly
```
