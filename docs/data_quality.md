# Data Quality and Lineage

The Phase 2 data layer favors honest data representation over convenience. Missing timestamps, missing labels, and weak causal evidence are represented explicitly instead of patched with assumptions.

## Quality Flags

Canonical records include a `quality_flag` field. Current values are:

- `valid`
- `missing`
- `invalid`
- `duplicate`
- `out_of_order`
- `derived`
- `synthetic`

The timestamp utilities flag missing, invalid, duplicate, and out-of-order timestamps where the source provides a timestamp axis. Row-index datasets such as SMD and AI4I use `sequence_index` and do not fabricate timestamps.

## Timestamp Policy

Rules:

- Preserve the source timestamp text in `timestamp_original`.
- Parse a normalized timestamp only when the source value is parseable.
- Do not invent a timezone when the source lacks one.
- Store `timezone_assumption` whenever parsing requires a documented or conservative assumption.

For NAB and MetroPT, timestamps are parsed as naive datetimes with:

```text
timezone unavailable in source; preserving naive timestamp
```

For SMD and AI4I, the canonical temporal axis is `sequence_index`.

## Lineage

Every generated dataset writes a lineage manifest under `data/manifests/lineage/<dataset>.json`.

Lineage includes:

- dataset id
- adapter name and version
- processing timestamp
- source path
- source files
- input record count
- output record counts
- output files
- transformations applied
- warnings

Lineage manifests are generated artifacts and are ignored by Git.

## Generated Artifact Policy

The repository tracks code, configuration, documentation, tests, and placeholder `.gitkeep` files. It does not track:

- raw datasets
- processed Parquet files
- generated profiles
- generated lineage manifests
- model artifacts
- experiment runs

This prevents accidental dataset redistribution and keeps results reproducible from scripts.

## Known Data Quality Issues

| Dataset | Issue | Handling |
| --- | --- | --- |
| `smap` | Telemetry bundle missing locally | Emit labels only and mark status `LABELS_ONLY`. |
| `msl` | Telemetry bundle missing locally | Emit labels only and mark status `LABELS_ONLY`. |
| `metropt` | No machine-readable incident/failure labels in local raw file | Emit metrics only; do not fabricate labels. |
| `loghub-openstack` | No anomaly label column in the 2k sample | Emit logs only. |
| `smd` | No timestamps | Use `sequence_index` and source split. |
| `ai4i` | Tabular rows are not naturally time-series telemetry | Use row index only; reserve it for prediction/XAI practice. |

## Validation Performed

The processed validation command verified that each generated Parquet file is readable and has a consistent schema count per canonical stream. It does not yet validate semantic constraints such as feature leakage or train/test joins; those checks belong to the next feature-engineering and model-evaluation phases.
