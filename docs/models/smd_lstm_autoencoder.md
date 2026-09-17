# SMD LSTM Autoencoder

## Purpose

The LSTM autoencoder is the first temporal deep-learning detector for SMD. It tests whether sequence reconstruction over multivariate telemetry windows improves anomaly detection compared with point-wise and dense-window baselines.

## Input

- Dataset: Server Machine Dataset
- Entity: one model per machine
- Window shape: `[64, 38]`
- Training data: source `train` training-fit windows
- Validation data: source `train` validation windows
- Test data: source `test`, evaluated at causal window endpoints

## Architecture

Current Phase 6 configuration:

- Encoder: LSTM
- Decoder: LSTM
- Hidden size: 32
- Layers: 1
- Dropout: 0.0
- Loss: mean squared reconstruction error
- Optimizer: Adam
- Epochs: 1
- Batch size: 256
- Training window cap: 4096 windows per machine
- Device: CPU

This is an intentionally small baseline intended to establish the experiment interface and artifact schema before tuning.

## Scores

The detector exposes:

- Global score: mean reconstruction error over the full window and all metrics.
- Per-metric score: reconstruction error averaged over time for each metric.

These scores provide candidate contributing signals for investigation. They do not prove causality.

## Measured Performance

Run: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3`

| Threshold | Precision | Recall | F1 | PR-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_percentile | 0.0936 | 0.5577 | 0.1603 | 0.1097 | 0.2349 |
| validation_percentile | 0.0771 | 0.3620 | 0.1272 | 0.1097 | 0.1883 |

The LSTM autoencoder did not beat the point-wise Isolation Forest globally in this first run. It did, however, become the best model for 6 individual machines, which suggests temporal modeling may help for a subset after tuning.

## Window Sensitivity

The run includes a small two-machine diagnostic comparing windows 32 and 64. Window size 32 was better for both neural models on both diagnostic machines. This is not a global hyperparameter result, but it supports running a broader window-size search before using the LSTM as a candidate production detector.

## Artifacts

- Models: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3/models/lstm_autoencoder/`
- Per-metric scores: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3/per_metric_scores/lstm_autoencoder/`
- Window sensitivity: `window_sensitivity.csv`
- Top metric evidence: `per_metric_top_signals.csv`

## Limitations

- One epoch is not enough to conclude that LSTM autoencoders are weak for SMD.
- Training uses capped windows for laptop-scale reproducibility.
- Sequence length was not globally optimized.
- Reconstruction evidence is useful for investigation ranking, not causal root-cause analysis.
