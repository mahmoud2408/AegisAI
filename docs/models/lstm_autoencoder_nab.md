# Model Card: LSTM Autoencoder NAB Baseline

Model name: `lstm_autoencoder_nab`
Version: `run_phase4_deep_anomaly`
Status: experimental baseline, not registered for production inference
Date: 2026-09-04

## Model Details

The LSTM Autoencoder is a per-series temporal reconstruction model trained on standardized raw-value windows from NAB. The encoder summarizes each sequence with an LSTM hidden state; the decoder repeats that context over the sequence length and reconstructs the input sequence.

Implementation:

```text
src/aegis_ai/ml/anomaly/lstm_autoencoder.py
src/aegis_ai/ml/anomaly/nab_deep_experiment.py
scripts/experiments/run_nab_deep_anomaly.py
```

Architecture:

```text
32 x 1 input window -> LSTM encoder(hidden=16) -> repeated context -> LSTM decoder(hidden=16) -> linear output -> reconstruction
```

## Intended Use

Intended use:

- evaluate sequence-aware reconstruction on NAB
- compare temporal representation learning against a Dense Autoencoder and classical baselines
- inspect computational cost versus detection benefit

Out-of-scope use:

- production incident paging
- multivariate service diagnosis
- root-cause inference
- official NAB leaderboard comparison
- causal claims

## Data and Protocol

Dataset: processed NAB

Split:

- 50% train
- 20% validation
- 30% test

Training policy:

- train one LSTM Autoencoder per series
- fit scaler on training rows only
- exclude labeled anomalous training windows from model fitting
- select threshold from the 99th percentile of training reconstruction errors
- evaluate on test window-ending timestamps only

## Hyperparameters

| Parameter | Value |
| --- | --- |
| Window length | 32 |
| Stride | 1 |
| Hidden size | 16 |
| Layers | 1 |
| Dropout | 0.0 |
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
| Precision | 0.2523 |
| Recall | 0.2245 |
| F1 | 0.2376 |
| PR-AUC | 0.1913 |
| False positive rate | 0.0795 |
| True positives | 2,585 |
| False positives | 7,661 |
| True negatives | 88,722 |
| False negatives | 8,929 |
| Detected windows | 35 |
| Missed windows | 7 |
| Median delay steps | 79 |
| Median delay seconds | 36,600 |

Per-series summary:

- mean F1: 0.1475
- median F1: 0.0000
- F1=0 series: 32 of 58
- positive test series with F1=0: 4 of 30

## Computational Cost

| Field | Value |
| --- | ---: |
| Total training time | 192.68 seconds |
| Mean training time per series | 3.32 seconds |
| Total inference time | 9.31 seconds |
| Median trainable parameters | 3,409 |
| Max process RSS | 810.14 MiB |

## Failure Modes

Observed failure modes:

- delayed detections in long incident windows
- false positives on large unlabeled Twitter-volume shifts
- missed windows where reconstruction error remains below train-percentile threshold
- higher compute cost without global F1 improvement over Dense Autoencoder

## Limitations

- univariate only
- small CPU-friendly architecture
- one model per series
- simple percentile threshold
- no official NAB score
- no causal or root-cause evidence

## Reproducibility

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_nab_deep_anomaly.py --run-id run_phase4_deep_anomaly
```
