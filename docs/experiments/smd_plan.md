# SMD Experimental Plan

Phase: 5 planning only, implemented in Phase 6  
Profile artifact: `experiments/smd/profile/run_phase5_smd_profile_final/smd_profile.json`

No SMD anomaly model was implemented in Phase 5. The planned experiment is now implemented by `scripts/experiments/run_smd_multivariate.py`; measured results are documented in `docs/experiments/smd_multivariate.md`.

## Dataset Profile

| Property | Value |
| --- | ---: |
| Machines | 28 |
| Metrics per machine | 38 |
| Train rows | 708,405 |
| Test rows | 708,420 |
| Test label rows | 708,420 |
| Positive test labels | 29,444 |
| Positive label rate | 0.0416 |
| Machines with anomalies | 28 |
| Anomaly segments | 327 |
| Interpretation intervals | 327 |
| Raw data size | 463.47 MiB |
| Processed metric records | 53,839,350 |
| Processed metric size | 555.54 MiB |
| Estimated wide float32 matrix size | 205.38 MiB |
| Estimated wide float64 matrix size | 410.76 MiB |

SMD uses a sequence index rather than timestamps. The existing Phase 2 canonical adapter stores long-format metric records for provenance, but model training should read or reconstruct per-machine wide matrices shaped `[time, 38 metrics]`.

Highest anomaly-rate machines:

| Machine | Positive rate | Positive labels | Segments | Max segment length |
| --- | ---: | ---: | ---: | ---: |
| `machine-1-6` | 0.1565 | 3,708 | 30 | 3,161 |
| `machine-2-2` | 0.1195 | 2,833 | 11 | 872 |
| `machine-1-7` | 0.1012 | 2,398 | 13 | 1,215 |
| `machine-1-1` | 0.0946 | 2,694 | 8 | 721 |
| `machine-2-4` | 0.0715 | 1,694 | 20 | 401 |
| `machine-2-9` | 0.0611 | 1,755 | 10 | 414 |

## Research Question

Does explicitly modeling multivariate dependencies between machine metrics improve anomaly detection on SMD compared with independent univariate detectors and classical multivariate baselines?

## Experimental Strategy

Train/validation/test:

- Use SMD train files for fitting only. Treat them as normal training data, while documenting that train labels are unavailable.
- Split each machine train matrix chronologically, for example 80% model fit and 20% validation-threshold calibration.
- Use the official SMD test split only for final evaluation.
- Do not use test labels for threshold selection, scaling, early stopping, or model selection.

Canonical input:

- Entity: one machine.
- Features: 38 synchronized metric columns.
- Temporal axis: integer sequence index.
- Labels: point-wise binary test labels, plus interpretation intervals for post-hoc affected-metric analysis only.
- Preferred model tensor: `[windows, window_size, 38]`.

Baseline sequence:

1. Univariate baseline: rolling z-score per metric, aggregate by max score across metrics.
2. Classical multivariate baseline: Isolation Forest over current values plus causal rolling features.
3. First deep model: multivariate LSTM Autoencoder or Dense Autoencoder over 38-metric windows.

Initial window plan:

- Start with window sizes 32 and 64.
- Use 64 as the first serious neural default because the Phase 5 NAB diagnostic improved neural F1 at 64, while keeping 32 for comparison.
- Keep stride 1 for first measurements, then consider larger stride only for efficiency.

Metrics:

- Precision
- Recall
- F1
- PR-AUC
- False positive rate
- Detection delay by contiguous positive segment
- Segment hit rate
- Per-machine metrics and global pooled metrics

Leakage risks:

- Fit scalers per machine on train-fit rows only.
- Do not pivot or normalize using test statistics.
- Do not tune thresholds using test labels.
- Keep validation windows chronological and avoid future-looking rolling features.
- Do not use interpretation labels during model training; reserve them for root-cause/affected-metric analysis.

## Exact Next Experiment

Implement the next phase as `run_smd_multivariate_baseline`:

- Load raw SMD machine matrices and test labels.
- Build leak-free train/validation/test windows.
- Train a per-machine multivariate Isolation Forest baseline and a per-machine neural autoencoder baseline.
- Evaluate global and per-machine point/segment metrics.
- Write artifacts under `experiments/anomaly/smd/`.

Do not add forecasting, RAG, agents, dashboards, or MLOps until the first SMD anomaly benchmark is measured.
