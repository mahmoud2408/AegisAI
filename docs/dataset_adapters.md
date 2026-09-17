# Dataset Adapters

AegisAI uses dataset adapters to isolate source-specific parsing from canonical data contracts. Adapters live in `src/aegis_ai/data/adapters` and expose the same lifecycle:

```text
load_raw -> validate -> profile -> to_canonical -> write lineage
```

## Shared Adapter Contract

Every adapter must:

- Read from `data/raw` without modifying raw files.
- Validate required source files before preprocessing.
- Return a structured profile with measured rows, labels, and source-specific facts.
- Emit canonical records with `source_dataset`, `source_file`, `source_row_id`, `quality_flag`, and stable ids.
- Preserve timestamps as source values plus parsed values; never invent timezone precision.
- Return empty streams for canonical types the source does not contain.

## Implemented Adapters

| Adapter | Dataset id | Canonical streams | Notes |
| --- | --- | --- | --- |
| `NABAdapter` | `nab` | metrics, labels | Parses 58 univariate series plus point labels and scoring windows. |
| `SMDAdapter` | `smd` | metrics, labels | Converts 38-column train/test matrices into long metrics and aligned point labels. |
| `MetroPTAdapter` | `metropt` | metrics | Emits continuous sensor metrics and binary state metrics. No failure labels are emitted. |
| `AI4IAdapter` | `ai4i` | metrics, failures, labels | Separates numeric equipment features from row-level failure labels. |
| `SMAPAdapter` | `smap` | labels | Emits labels only until telemetry bundle is available. |
| `MSLAdapter` | `msl` | labels | Emits labels only until telemetry bundle is available. |
| `LogHubAdapter` | `loghub-openstack` and LogHub family aliases | logs, optional labels | Reusable parser for structured LogHub 2k samples. BGL labels are supported when present. |

## Dataset Decisions

NAB:

- `entity_id` is derived from the series path, for example `nab:realAWSCloudwatch/ec2_cpu_utilization_24ae8d`.
- `metric_name` is the source series filename stem.
- Labels are stored as `LabelEvent` records, separating point labels from scoring windows through metadata.

SMD:

- Machine filenames become `entity_id` and `host_id`.
- `train` and `test` are preserved in the canonical `split` field.
- Source rows do not contain timestamps; `sequence_index` is the temporal axis.
- `test_label` files produce point-wise anomaly/normal labels.
- `interpretation_label` files produce interval labels used later as evidence, not causal truth.

MetroPT-3:

- The compressor is represented as `metropt_air_compressor_01`.
- Sensor columns are classified as continuous or binary state.
- Source timestamps are parsed but kept timezone-naive with an explicit assumption string.
- No incident labels are fabricated from documentation or timestamps.

AI4I:

- Identifier columns such as `UDI` and `Product ID` are preserved as lineage/entity metadata, not model features.
- Numeric sensor/process fields become metric observations.
- `Machine failure` becomes the supervised failure target stream.
- Failure mode columns become label events for later explainability and classification work.

SMAP/MSL:

- `labeled_anomalies.csv` is parsed into interval labels.
- Because the telemetry archive is absent locally, the adapters return `LABELS_ONLY`.
- Label rows include metadata noting that they are not yet joinable to canonical metric records.

LogHub:

- Structured CSV files provide line id, template id, template text, level, component, message, and optional labels.
- OpenStack request ids are extracted from explicit request-like fields or content when present.
- Templates are treated as metadata and future features, not as anomaly labels.

## Adding a New Adapter

1. Add a dataset entry to `config/datasets.yaml`.
2. Implement a subclass of `BaseDatasetAdapter`.
3. Register it in `src/aegis_ai/data/adapters/__init__.py`.
4. Add a small fixture test under `tests/unit`.
5. Run:

```powershell
python scripts/data/profile_all.py --dataset <dataset-id>
python scripts/data/preprocess_dataset.py --dataset <dataset-id>
python scripts/data/validate_processed.py --dataset <dataset-id>
```
