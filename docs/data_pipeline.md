# Canonical Data Pipeline

Phase 2 implements the canonical data layer for AegisAI. It converts local raw datasets into typed, source-preserving Parquet streams that later ML phases can consume without mixing training logic into ingestion code.

## Scope

Implemented in this phase:

- Canonical Pydantic records for metrics, logs, traces, labels, failures, and incidents.
- Dataset adapter interface with validation, profiling, canonical conversion, and lineage.
- Dataset adapters for NAB, SMD, MetroPT-3, AI4I, SMAP/MSL labels, and reusable LogHub structured logs.
- Streaming Parquet writer with conservative partitioning.
- Local generated manifests for profiles, lineage, preprocessing summaries, and processed-output validation.

Not implemented in this phase:

- Feature windows for model training.
- Statistical or ML anomaly detection.
- Forecasting, incident prediction, root-cause scoring, RAG, agent logic, API, or dashboard integration.

## Flow

```text
data/raw/<dataset>
  -> DatasetAdapter.load_raw()
  -> DatasetAdapter.validate()
  -> DatasetAdapter.profile()
  -> DatasetAdapter.to_canonical()
  -> data/processed/<canonical_type>/<dataset>/*.parquet
  -> data/manifests/profiles/<dataset>.json
  -> data/manifests/lineage/<dataset>.json
```

The pipeline is intentionally file-based for Phase 2. Database ingestion will be added later, after the canonical records and model-ready feature views stabilize.

## Output Contract

Canonical streams are stored by type:

- `data/processed/metrics/<dataset>/`
- `data/processed/logs/<dataset>/`
- `data/processed/traces/<dataset>/`
- `data/processed/incidents/<dataset>/`
- `data/processed/failures/<dataset>/`
- `data/processed/labels/<dataset>/`

Generated profiles and lineage are stored under:

- `data/manifests/profiles/`
- `data/manifests/lineage/`
- `data/manifests/preprocessing_summary.json`
- `data/manifests/preprocessing_report.md`
- `data/manifests/processed_validation.json`

These generated files are ignored by Git because they are local artifacts derived from local datasets.

## Measured Local Phase 2 Run

Last measured run: 2026-09-03.

| Dataset | Status | Canonical outputs |
| --- | --- | --- |
| `nab` | `READY` | 365,558 metric records; 236 label records |
| `smd` | `READY` | 53,839,350 metric records; 708,747 label records |
| `metropt` | `READY` | 22,754,220 metric records |
| `ai4i` | `READY` | 50,000 metric records; 10,000 failure records; 60,000 label records |
| `smap` | `LABELS_ONLY` | 69 label records |
| `msl` | `LABELS_ONLY` | 36 label records |
| `loghub-openstack` | `READY` | 2,000 log records |

These are preprocessing counts only. They are not model-performance results.

## Commands

```powershell
python scripts/data/profile_all.py
python scripts/data/preprocess_all.py --batch-size 25000
python scripts/data/validate_processed.py
```

To run one dataset:

```powershell
python scripts/data/preprocess_dataset.py --dataset nab
python scripts/data/preprocess_dataset.py --dataset smd --batch-size 25000
```

## Design Notes

The canonical metric representation is long format: one row per entity, time/index, and metric name. This keeps the model input contract consistent across NAB, SMD, MetroPT, and AI4I, but it expands wide datasets substantially. SMD expands 1,416,825 source rows into 53,839,350 metric records.

The writer uses Snappy-compressed Parquet and streams batches without loading whole datasets into memory. SMD metrics are partitioned by `split` and `entity_id`; NAB metrics are partitioned by `entity_id`; SMD labels are partitioned by `entity_id`.

## Limitations

- SMAP/MSL telemetry is not present locally, so only label intervals are emitted.
- MetroPT-3 has telemetry but no machine-readable incident/failure labels in the local raw file.
- OpenStack LogHub 2k sample has structured logs and templates but no row anomaly labels.
- No scalers, imputers, feature transforms, model fits, or learned preprocessing steps are created in Phase 2.
