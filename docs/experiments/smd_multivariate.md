# SMD Multivariate Anomaly Detection

Phase 6 evaluates whether multivariate telemetry modeling improves anomaly detection on the Server Machine Dataset (SMD) compared with a per-feature statistical baseline.

## Current Run

- Run: `experiments/anomaly/smd/phase6_smd_multivariate_20260911_eps1e3`
- Script: `scripts/experiments/run_smd_multivariate.py`
- Config: `configs/experiments/smd_multivariate.yaml`
- Notebook: `notebooks/04_smd_multivariate_anomaly.ipynb`
- Scaler epsilon: `1e-3`
- Window size: `64`
- Stride: `1`
- Threshold policies: train percentile and validation percentile, both at percentile `99.0`

## Dataset Audit

The run loaded raw SMD matrices directly from `data/raw/smd`; it did not assume processed parquet data was present.

| Item | Value |
| --- | ---: |
| Machines | 28 |
| Metrics per machine | 38 |
| Train rows | 708405 |
| Test rows | 708420 |
| Window-end evaluation rows | 706656 |
| Positive test labels | 29444 |
| Positive label rate | 0.0416 |
| Anomaly segments | 327 |
| Missing values | 0 |
| Duplicate train rows | 0 |
| Duplicate test rows | 0 |
| Machines with train near-constant features at epsilon `1e-3` | 28 |
| Machines with test near-constant features at epsilon `1e-3` | 28 |

SMD has no native timestamps in this local copy, so all delay values are reported in sequence steps, not seconds.

## Evaluation Protocol

The source SMD split is preserved. For each machine, source `train` is split chronologically into an 80 percent training-fit segment and a 20 percent validation segment. Source `test` labels are held out until final evaluation.

Normalization is fitted on the training-fit segment only, then applied to validation and test. The `1e-3` scaler epsilon prevents near-constant normalized SMD dimensions from producing unstable, extremely large standardized values. This choice is recorded in `config.json` and `dataset_audit.json`.

Windowed neural models use causal windows shaped `[batch, 64, 38]`. Window scores are assigned to the window endpoint and evaluated against the endpoint test label. No window crosses the train/validation/test boundary.

Thresholds are selected from either training scores or validation scores. Test labels are not used for threshold selection.

## Models Compared

| Model | Input | Score |
| --- | --- | --- |
| Per-feature z-score | Current 38-dimensional endpoint | Max absolute train-scaled feature value |
| Isolation Forest | Current 38-dimensional endpoint | Negative Isolation Forest sample score |
| Dense autoencoder | 64 by 38 window | Mean reconstruction error |
| LSTM autoencoder | 64 by 38 window | Mean reconstruction error |

The Isolation Forest is intentionally point-wise rather than flattened-window based. This keeps the comparison aligned with SMD point labels and avoids pushing a sparse baseline into a 2432-dimensional flattened window space before stronger temporal models are established.

## Global Results

Measured results from `global_metrics.csv`:

| Model | Threshold | Precision | Recall | F1 | PR-AUC | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest | train_percentile | 0.1622 | 0.3636 | 0.2243 | 0.2022 | 0.0817 |
| Isolation Forest | validation_percentile | 0.1471 | 0.3508 | 0.2072 | 0.2022 | 0.0885 |
| Per-feature z-score | validation_percentile | 0.1147 | 0.4944 | 0.1861 | 0.1002 | 0.1660 |
| Dense autoencoder | train_percentile | 0.0960 | 0.5644 | 0.1641 | 0.1131 | 0.2310 |
| Per-feature z-score | train_percentile | 0.0960 | 0.5390 | 0.1630 | 0.1002 | 0.2207 |
| LSTM autoencoder | train_percentile | 0.0936 | 0.5577 | 0.1603 | 0.1097 | 0.2349 |
| LSTM autoencoder | validation_percentile | 0.0771 | 0.3620 | 0.1272 | 0.1097 | 0.1883 |
| Dense autoencoder | validation_percentile | 0.0749 | 0.4170 | 0.1270 | 0.1131 | 0.2239 |

Best global model by F1: Isolation Forest with train-percentile thresholding.

Its detection delay over SMD anomaly segments was:

- Detected segments: 209 of 327
- Missed segments: 118 of 327
- Mean delay: 16.74 sequence steps
- Median delay: 1.00 sequence step

## Machine-Level Robustness

Measured summary from `per_machine_summary.csv`:

| Model | Threshold | Mean F1 | Median F1 | Std F1 | Machines with F1=0 |
| --- | --- | ---: | ---: | ---: | ---: |
| Per-feature z-score | train_percentile | 0.2469 | 0.1691 | 0.2109 | 0 |
| Per-feature z-score | validation_percentile | 0.2289 | 0.1497 | 0.1875 | 0 |
| LSTM autoencoder | train_percentile | 0.2229 | 0.1732 | 0.1975 | 0 |
| Isolation Forest | train_percentile | 0.2174 | 0.1910 | 0.1610 | 0 |
| Dense autoencoder | train_percentile | 0.2172 | 0.1570 | 0.1937 | 0 |
| Isolation Forest | validation_percentile | 0.1949 | 0.1473 | 0.1433 | 0 |
| Dense autoencoder | validation_percentile | 0.1792 | 0.0963 | 0.1739 | 2 |
| LSTM autoencoder | validation_percentile | 0.1781 | 0.0890 | 0.1771 | 2 |

The best model per machine was distributed as follows:

- Per-feature z-score: 12 machines
- Isolation Forest: 9 machines
- LSTM autoencoder: 6 machines
- Dense autoencoder: 1 machine

This means the global point-weighted winner is not the same as the machine-balanced winner. That distinction matters in incident systems because a model can perform well globally while being less robust across machine types.

## Multivariate Benefit

Compared with the best per-feature z-score result per machine:

- Multivariate models improved F1 on 16 machines.
- Multivariate models underperformed on 12 machines.
- Mean F1 delta, best multivariate minus z-score: 0.0344.
- Median F1 delta: 0.0343.

This is evidence that multivariate structure helps on a subset of SMD machines, but it is not universal. It is also not a causal statement about incidents; it is only an empirical detection comparison inside SMD.

## Candidate Contributing Metrics

The experiment records top per-metric signals for the z-score and autoencoder models. These are candidate contributing signals, not root causes.

Most frequent top-k metric mentions across the run:

| Metric | Mentions |
| --- | ---: |
| metric_02 | 10457 |
| metric_06 | 10137 |
| metric_03 | 10036 |
| metric_25 | 9400 |
| metric_01 | 8665 |
| metric_23 | 8300 |
| metric_10 | 7664 |
| metric_15 | 7300 |
| metric_12 | 6751 |
| metric_00 | 6284 |

Detailed evidence is stored in `per_metric_top_signals.csv`, `per_metric_signal_frequency.csv`, and compressed arrays under `per_metric_scores/`.

## Computational Cost

Measured totals from `runtime.csv`:

| Model | Total train seconds | Total inference seconds | Mean train seconds per machine | Mean inference seconds per machine |
| --- | ---: | ---: | ---: | ---: |
| Per-feature z-score | 0.00 | 0.17 | 0.00 | 0.006 |
| Isolation Forest | 6.36 | 7.07 | 0.23 | 0.253 |
| Dense autoencoder | 10.99 | 39.70 | 0.39 | 1.418 |
| LSTM autoencoder | 49.58 | 134.67 | 1.77 | 4.810 |

The full run took 348.04 seconds including model training, inference, artifact writing, figures, and window-sensitivity diagnostics.

## Window Sensitivity

The small diagnostic subset compared neural windows of 32 and 64 on `machine-1-1` and `machine-1-2`. Window size 32 performed better for both dense and LSTM autoencoders on both diagnostic machines. This is not enough to claim a global optimum, but it justifies making window size a tuned experiment parameter in the next anomaly iteration.

## Limitations

- Neural models used a bounded first-pass budget: one epoch and at most 4096 training windows per machine.
- Thresholds are simple percentiles; no adaptive per-machine operating-point selection has been implemented yet.
- SMD labels are point-wise; interpretation labels are available but not yet used as supervised attribution targets.
- Per-metric evidence is reconstruction or z-score evidence, not a causal RCA result.
- NAB scores are not reported for SMD because SMD does not define NAB anomaly windows.
- Results are not directly comparable with NAB because the datasets, labels, telemetry structure, and evaluation setup differ.

## Next Phase

Proceed to incident prediction only after deciding whether to run a stronger SMD follow-up with tuned window sizes, more epochs, and model selection against validation scores. A small follow-up should preserve the same leakage-safe split and artifact schema introduced here.
