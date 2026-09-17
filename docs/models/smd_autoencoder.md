# SMD Dense Autoencoder

## Purpose

The dense autoencoder is the first reconstruction-based multivariate detector for SMD. It learns to reconstruct 64-step telemetry windows and treats high reconstruction error as anomalous behavior.

## Input

- Dataset: Server Machine Dataset
- Entity: one model per machine
- Window shape: `[64, 38]`
- Training data: normal source `train` windows from the training-fit segment
- Validation data: source `train` validation segment for threshold diagnostics
- Test data: source `test`, evaluated at causal window endpoints

## Architecture

Current Phase 6 configuration:

- Flattened input dimension: 2432
- Hidden dimensions: 128, 64
- Latent dimension: 16
- Loss: mean squared reconstruction error
- Optimizer: Adam
- Epochs: 1
- Batch size: 256
- Training window cap: 4096 windows per machine
- Device: CPU

This is a bounded first-pass baseline, not a tuned final neural model.

## Scores

The detector exposes two score views:

- Global score: mean reconstruction error over the full window and all metrics.
- Per-metric score: reconstruction error averaged over time for each metric.

Per-metric scores are used to rank candidate contributing telemetry signals. They are not root-cause labels.

## Measured Performance

Run: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3`

| Threshold | Precision | Recall | F1 | PR-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_percentile | 0.0960 | 0.5644 | 0.1641 | 0.1131 | 0.2310 |
| validation_percentile | 0.0749 | 0.4170 | 0.1270 | 0.1131 | 0.2239 |

The dense autoencoder achieved high recall under train-percentile thresholding but produced many false positives. This is expected for a minimally trained reconstruction model and should be revisited with tuned training, validation-aware model selection, and feature-variance handling.

## Artifacts

- Models: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3/models/dense_autoencoder/`
- Per-metric scores: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3/per_metric_scores/dense_autoencoder/`
- Top metric evidence: `per_metric_top_signals.csv`
- Machine overview plots: `figures/*dense_autoencoder*_overview.png`

## Limitations

- One training epoch is intentionally conservative for run time.
- Training uses a deterministic cap of 4096 windows per machine.
- The model currently reconstructs flattened windows, so it does not explicitly model sequence order.
- Percentile thresholds are simple and may not match an operational alert budget.
