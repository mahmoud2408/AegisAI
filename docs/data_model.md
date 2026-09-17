# Unified Data Model

Phase 2 implementation note: the canonical records described here now have Pydantic models under `src/aegis_ai/data/models` and PyArrow storage schemas under `src/aegis_ai/data/preprocessing/schemas.py`.

AegisAI must support heterogeneous telemetry without pretending that every dataset contains the same information. The correct design is a canonical core, dataset-specific extensions, and adapters that document every transformation.

## Design Principles

- Preserve raw data exactly in `data/raw`.
- Convert source-specific records into typed canonical events in `data/processed`.
- Reserve `data/interim` for future partially transformed files that are not yet canonical.
- Build task-specific feature views and split metadata in `data/processed` in later phases.
- Track provenance for every row: source dataset, source file, entity, original row index, and adapter version.
- Keep real labels, derived labels, and synthetic labels separate.
- Never force missing fields into fake relationships. Null is better than invented linkage.

## Implemented Phase 2 Storage Schemas

The Phase 2 Parquet schemas use `metadata_json` for small adapter-specific metadata, while first-class fields remain relationally shaped for later PostgreSQL ingestion.

### `MetricObservation`

| Field group | Fields |
| --- | --- |
| Identity | `event_id`, `entity_id`, `service_id`, `host_id` |
| Time | `timestamp`, `timestamp_original`, `timezone_assumption`, `sequence_index` |
| Metric | `metric_name`, `metric_value`, `metric_type`, `unit`, `split` |
| Lineage | `source_dataset`, `source_file`, `source_row_id`, `quality_flag`, `metadata_json` |

### `LogEvent`

| Field group | Fields |
| --- | --- |
| Identity | `event_id`, `entity_id`, `service_id`, `host_id`, `process_id`, `request_id`, `trace_id` |
| Time | `timestamp`, `timestamp_original`, `timezone_assumption`, `sequence_index` |
| Log content | `log_level`, `event_type`, `event_template`, `message` |
| Lineage | `source_dataset`, `source_file`, `source_row_id`, `quality_flag`, `metadata_json` |

### `LabelEvent`

| Field group | Fields |
| --- | --- |
| Identity | `label_id`, `event_id`, `entity_id`, `service_id` |
| Semantics | `label_kind`, `label_source_type`, `label_value`, `label_source` |
| Time/index | `timestamp`, `timestamp_original`, `timezone_assumption`, `sequence_index`, `timestamp_start`, `timestamp_end`, `start_index`, `end_index` |
| Lineage | `source_dataset`, `source_file`, `source_row_id`, `quality_flag`, `metadata_json` |

### `FailureObservation`

| Field group | Fields |
| --- | --- |
| Identity | `event_id`, `entity_id` |
| Target | `target`, `failure_type`, `label_source`, `label_source_type` |
| Time | `timestamp`, `timestamp_original`, `timezone_assumption`, `sequence_index` |
| Lineage | `source_dataset`, `source_file`, `source_row_id`, `quality_flag`, `metadata_json` |

`TraceEvent` and `IncidentEvent` schemas are implemented for future OpenTelemetry/synthetic and incident-curation phases, but none of the Phase 2 local datasets produced trace or incident rows.

## Canonical Entities

### SourceDataset

Represents a dataset and version.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `source_dataset` | string | yes | Example: `nab`, `smd`, `metropt`, `loghub-bgl`. |
| `dataset_version` | string | no | Upstream tag/hash/date when known. |
| `source_path` | string | yes | Raw file path relative to repository. |
| `ingested_at` | datetime | yes | Adapter run timestamp. |
| `adapter_name` | string | yes | Example: `NABAdapter`. |
| `adapter_version` | string | yes | Project-controlled adapter version. |

### Entity

Represents the thing producing data.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `entity_id` | string | yes | Machine, service, channel, node, product, or synthetic entity. |
| `entity_type` | enum | yes | `machine`, `service`, `sensor_channel`, `node`, `request`, `product`, `unknown`. |
| `service_id` | string | no | Only when the source provides or adapter can safely map service. |
| `host_id` | string | no | Machine or host identifier. |
| `source_dataset` | string | yes | Provenance. |
| `metadata` | object | no | Small adapter-specific metadata only. |

### MetricObservation

Canonical representation for numerical and state telemetry.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `timestamp` | datetime | no | Required only when the source has real timestamps. |
| `sequence_index` | integer | no | Required when timestamp is absent, such as SMD. |
| `entity_id` | string | yes | Machine/channel/product id. |
| `service_id` | string | no | Null when unavailable. |
| `host_id` | string | no | Null when unavailable. |
| `metric_name` | string | yes | Source column or generated stable metric name. |
| `metric_value` | float | yes | Numeric value. |
| `metric_type` | enum | yes | `continuous`, `binary_state`, `count`, `rate`, `unknown`. |
| `unit` | string | no | Example: `K`, `rpm`, `Nm`, `seconds`. |
| `source_dataset` | string | yes | Provenance. |
| `source_file` | string | yes | Provenance. |
| `source_row_index` | integer | yes | Raw row offset after header if applicable. |

Rules:

- NAB emits one metric per timestamp with a series-derived `metric_name`.
- SMD emits 38 metric observations per source row, using stable names such as `metric_00`.
- MetroPT emits continuous sensor metrics plus binary state metrics.
- AI4I emits equipment observations; it should not be treated as timestamped telemetry.

### LogEvent

Canonical representation for logs.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `timestamp` | datetime | no | Parsed from source date/time when available. |
| `sequence_index` | integer | yes | Preserves source line ordering. |
| `entity_id` | string | no | Node/process/service if available. |
| `service_id` | string | no | Component/service if safely inferable. |
| `host_id` | string | no | Node/host field when available. |
| `process_id` | string | no | `Pid`, `Process`, or source equivalent. |
| `request_id` | string | no | Only if parsed from explicit request field/content. |
| `trace_id` | string | no | Only for OpenTelemetry/synthetic data with real trace ids. |
| `log_level` | string | no | `INFO`, `WARN`, `ERROR`, etc. |
| `component` | string | no | Source component/logger. |
| `event_id` | string | no | Template id from structured LogHub files. |
| `event_template` | string | no | Source-provided or parser-produced template. |
| `log_message` | string | yes | Raw message/content. |
| `source_dataset` | string | yes | Provenance. |
| `source_file` | string | yes | Provenance. |

Rules:

- LogHub template files are metadata/features, not anomaly labels.
- OpenStack `ADDR` may contain request-like identifiers, but those must be parsed and validated before use.

### TraceSpan

Canonical representation for traces. Current real datasets do not contain static trace rows; this schema is for future synthetic/OpenTelemetry data.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `trace_id` | string | yes | Trace identifier. |
| `span_id` | string | yes | Span identifier. |
| `parent_span_id` | string | no | Parent id. |
| `service_id` | string | yes | Service name. |
| `operation_name` | string | yes | Span operation. |
| `start_time` | datetime | yes | Start timestamp. |
| `end_time` | datetime | yes | End timestamp. |
| `duration_ms` | float | yes | Derived from start/end. |
| `status_code` | string | no | Source status. |
| `source_dataset` | string | yes | Provenance. |

### LabelEvent

Canonical label representation.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `label_id` | string | yes | Stable generated id. |
| `label_type` | enum | yes | `real`, `derived`, `synthetic`. |
| `task_type` | enum | yes | `anomaly`, `incident`, `failure`, `root_cause`, `severity`. |
| `entity_id` | string | no | Required if label is entity-specific. |
| `timestamp` | datetime | no | Point label timestamp when available. |
| `sequence_index` | integer | no | Point label index when timestamp absent. |
| `start_time` | datetime | no | Interval/window start. |
| `end_time` | datetime | no | Interval/window end. |
| `start_index` | integer | no | Interval start in row-index datasets. |
| `end_index` | integer | no | Interval end in row-index datasets. |
| `label_value` | string | yes | Example: `anomaly`, `normal`, `failure`. |
| `label_source` | string | yes | Upstream file or derivation procedure. |
| `source_dataset` | string | yes | Provenance. |

Rules:

- NAB point labels and scoring windows become `real` label events.
- SMD test labels become point-wise `real` label events on `sequence_index`.
- SMD interpretation-label files become interval evidence labels, not definitive root-cause proof.
- AI4I `Machine failure` and failure modes become row-level `real` labels.
- Synthetic incident labels must be `synthetic`.

### IncidentEvent

Represents known or generated incidents.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `incident_id` | string | yes | Stable id. |
| `incident_type` | string | yes | Failure, anomaly cluster, service outage, etc. |
| `severity` | enum | no | Known for synthetic/curated incidents only. |
| `start_time` / `end_time` | datetime | no | Timestamped incidents. |
| `start_index` / `end_index` | integer | no | Row-index incidents. |
| `affected_entities` | array | yes | Entity ids. |
| `root_cause_label` | string | no | Only when real/curated/synthetic label exists. |
| `label_type` | enum | yes | `real`, `derived`, `synthetic`, or `unknown`. |
| `source_dataset` | string | yes | Provenance. |

### EvidenceItem

Common record used by RCA, RAG, and agent reporting.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `evidence_id` | string | yes | Stable id. |
| `incident_id` | string | no | Optional link. |
| `evidence_type` | enum | yes | `metric`, `log`, `trace`, `label`, `document`, `model_output`. |
| `observed_at` | datetime | no | Timestamp if available. |
| `entity_id` | string | no | Evidence entity. |
| `summary` | string | yes | Concise evidence statement. |
| `value` | string | no | Metric value, log event id, score, etc. |
| `source_ref` | string | yes | File, row, chunk, or model run reference. |
| `confidence` | float | no | For model-derived evidence only. |

## Dataset Adaptation

| Dataset | Canonical Objects | Entity Strategy | Temporal Strategy | Label Strategy |
| --- | --- | --- | --- | --- |
| NAB | `MetricObservation`, `LabelEvent` | Series path as entity | Real timestamp | Real point/window anomaly labels |
| SMD | `MetricObservation`, `LabelEvent` | Machine filename as entity | Row index | Real point labels and interpretation intervals |
| MetroPT-3 | `MetricObservation` | Air compressor as entity | Real timestamp | None until curated |
| AI4I | Equipment observation view, `LabelEvent` | Product/row id as entity | Row index only | Real row failure labels |
| LogHub | `LogEvent` | Node/process/component when available | Event timestamp plus line order | BGL only has real row labels locally |
| OpenTelemetry | Future `MetricObservation`, `LogEvent`, `TraceSpan` | Service/host/request ids from generated telemetry | Real timestamp in generated data | Synthetic scenario labels |
| SMAP/MSL | Future `MetricObservation`, `LabelEvent` | Channel id | Row index if telemetry restored | Real interval labels, currently unjoinable |

## Adapter Interface

Adapters should share one protocol:

```python
class DatasetAdapter(Protocol):
    dataset_id: str
    adapter_version: str

    def load_raw(self) -> RawDataset: ...

    def validate(self, raw: RawDataset) -> ValidationReport: ...

    def profile(self, raw: RawDataset) -> DatasetProfile: ...

    def transform(self, raw: RawDataset) -> InterimDataset: ...

    def to_canonical(self, interim: InterimDataset) -> CanonicalDataset: ...

    def get_labels(self, raw: RawDataset) -> list[LabelEvent]: ...
```

Required adapter behavior:

- Validate expected files before reading.
- Emit structured validation errors.
- Preserve source row indices.
- Include source path and adapter version in every output record.
- Return empty labels explicitly when no real labels exist.
- Never fit scalers, templates, imputers, or encoders on test data.

## Planned Adapters

| Adapter | First Implementation Scope | Deferred Scope |
| --- | --- | --- |
| `NABAdapter` | Load CSVs, parse timestamps, load labels/windows, canonicalize series. | NAB scoring integration. |
| `SMDAdapter` | Load train/test/test labels, align row counts, canonicalize machine metrics. | Interpretation-label evidence views. |
| `MetroPTAdapter` | Stream CSV, classify continuous vs binary columns, resample policy metadata. | Curated maintenance incident labels. |
| `AI4IAdapter` | Load tabular rows, separate identifiers/features/labels. | Calibration/evaluation reports. |
| `LogHubAdapter` | Load raw/structured/template files, emit log events and template metadata. | Parser benchmarking against templates. |
| `OpenTelemetryAdapter` | Treat current files as generator/reference config. | Ingest generated OTLP-style telemetry. |
| `SMAPMSLAdapter` | Validate label CSV and missing telemetry status. | Full telemetry if legal archive is restored. |

## Relational Storage Mapping

Canonical records should later map cleanly to PostgreSQL tables:

- `source_datasets`
- `entities`
- `metric_observations`
- `log_events`
- `trace_spans`
- `label_events`
- `incident_events`
- `feature_windows`
- `model_runs`
- `model_predictions`
- `evidence_items`
- `documents`
- `document_chunks`

Do not collapse these into arbitrary JSON blobs. Use JSON only for small source metadata, parser attributes, or provider-specific payloads that do not deserve first-class relational columns yet.
