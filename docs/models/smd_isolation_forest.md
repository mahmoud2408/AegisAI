# SMD Isolation Forest

## Purpose

This model is the first multivariate non-neural detector for SMD. It tests whether modeling the 38 machine metrics jointly at each time step improves detection over an independent per-feature z-score baseline.

## Input

- Dataset: Server Machine Dataset
- Entity: one model per machine
- Features: 38 train-scaled telemetry metrics
- Training split: first 80 percent of source `train` for each machine
- Validation split: final 20 percent of source `train` for threshold selection diagnostics
- Test split: source `test`, labels used only for evaluation

## Design Decision

The Phase 6 Isolation Forest is point-wise, not window-flattened. A flattened 64 by 38 window would produce 2432 features and make a tree baseline less interpretable and more sensitive to sparse high-dimensional effects. The point-wise design aligns directly with SMD point labels and provides a strong multivariate baseline before temporal models are tuned.

## Configuration

Current run configuration:

- `n_estimators`: 100
- `max_samples`: `auto`
- `contamination`: `auto`
- `random_state`: 42
- `train_row_limit`: 12000 per machine
- `threshold_percentile`: 99.0

## Measured Performance

Run: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3`

| Threshold | Precision | Recall | F1 | PR-AUC | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_percentile | 0.1622 | 0.3636 | 0.2243 | 0.2022 | 0.0817 |
| validation_percentile | 0.1471 | 0.3508 | 0.2072 | 0.2022 | 0.0885 |

Best global model in Phase 6: Isolation Forest with train-percentile thresholding.

## Artifacts

- Models: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3/models/isolation_forest/`
- Scores and predictions: `predictions.parquet`
- Global metrics: `global_metrics.csv`
- Per-machine metrics: `per_machine_metrics.csv`

## Limitations

- The model does not directly expose per-feature contribution scores.
- It does not model temporal dynamics beyond the current endpoint.
- Thresholding is percentile-based and has not yet been optimized for operational precision/recall tradeoffs.
- Results are SMD-specific and should not be generalized to NAB or production telemetry without evaluation.
