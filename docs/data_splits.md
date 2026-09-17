# Data Splits and Leakage Policy

Phase 2 does not train models, but it establishes split rules that later ML phases must follow.

## Core Principles

- Use source-provided chronological splits when available.
- Never fit preprocessing transforms on validation or test data.
- Build rolling features using only historical values available at the prediction time.
- Treat label windows carefully so future labels do not leak into features.
- Keep entity-based splits explicit when evaluating generalization to unseen machines or services.

## Dataset Split Policy

| Dataset | Split strategy | Notes |
| --- | --- | --- |
| `nab` | Chronological per series | Train/evaluate using early history or rolling prequential protocols; labels are anomaly points/windows. |
| `smd` | Source split | Use `train` for fitting unsupervised detectors and `test` plus labels for evaluation. |
| `metropt` | Chronological split to be defined in ML phase | No labels, so use for forecasting, unsupervised detection, and synthetic/curated scenarios only. |
| `ai4i` | Chronological row-index split for supervised practice | Avoid random split because row order is the only temporal proxy available. |
| `smap`/`msl` | Blocked until telemetry exists | Labels are not model-ready without telemetry. |
| `loghub-openstack` | Chronological line order | No row labels; use for log feature extraction and investigation context. |

## Implemented Utilities

`src/aegis_ai/data/preprocessing/splitting.py` contains:

- `chronological_split`
- `entity_split`
- `assert_no_overlapping_windows`

These are low-level utilities only. Dataset-specific model splits and feature-window generation will be implemented in the ML/feature phases.

## Next Split Work

The next phase should add model-ready views for NAB and SMD:

- NAB per-series chronological windows with point/window label joins.
- SMD wide matrix reconstruction from canonical metrics for multivariate detectors.
- Explicit metadata recording fit/evaluation boundaries.
- Tests that ensure rolling windows do not cross validation/test cut points.
