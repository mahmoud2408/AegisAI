"""PyArrow schemas for canonical processed datasets."""

from __future__ import annotations

import pyarrow as pa

METRIC_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("timestamp", pa.string()),
        ("timestamp_original", pa.string()),
        ("timezone_assumption", pa.string()),
        ("sequence_index", pa.int64()),
        ("entity_id", pa.string()),
        ("service_id", pa.string()),
        ("host_id", pa.string()),
        ("metric_name", pa.string()),
        ("metric_value", pa.float64()),
        ("metric_type", pa.string()),
        ("unit", pa.string()),
        ("split", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_row_id", pa.string()),
        ("quality_flag", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

LOG_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("timestamp", pa.string()),
        ("timestamp_original", pa.string()),
        ("timezone_assumption", pa.string()),
        ("sequence_index", pa.int64()),
        ("entity_id", pa.string()),
        ("service_id", pa.string()),
        ("host_id", pa.string()),
        ("process_id", pa.string()),
        ("request_id", pa.string()),
        ("trace_id", pa.string()),
        ("log_level", pa.string()),
        ("event_type", pa.string()),
        ("event_template", pa.string()),
        ("message", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_row_id", pa.string()),
        ("quality_flag", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

TRACE_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("timestamp", pa.string()),
        ("timestamp_original", pa.string()),
        ("timezone_assumption", pa.string()),
        ("trace_id", pa.string()),
        ("span_id", pa.string()),
        ("service_id", pa.string()),
        ("parent_span_id", pa.string()),
        ("operation_name", pa.string()),
        ("duration_ms", pa.float64()),
        ("status", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_row_id", pa.string()),
        ("quality_flag", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

FAILURE_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("timestamp", pa.string()),
        ("timestamp_original", pa.string()),
        ("timezone_assumption", pa.string()),
        ("sequence_index", pa.int64()),
        ("entity_id", pa.string()),
        ("target", pa.int8()),
        ("failure_type", pa.string()),
        ("label_source", pa.string()),
        ("label_source_type", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_row_id", pa.string()),
        ("quality_flag", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

LABEL_SCHEMA = pa.schema(
    [
        ("label_id", pa.string()),
        ("label_kind", pa.string()),
        ("label_source_type", pa.string()),
        ("label_value", pa.string()),
        ("label_source", pa.string()),
        ("event_id", pa.string()),
        ("timestamp", pa.string()),
        ("timestamp_original", pa.string()),
        ("timezone_assumption", pa.string()),
        ("sequence_index", pa.int64()),
        ("timestamp_start", pa.string()),
        ("timestamp_end", pa.string()),
        ("start_index", pa.int64()),
        ("end_index", pa.int64()),
        ("entity_id", pa.string()),
        ("service_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("source_row_id", pa.string()),
        ("quality_flag", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

INCIDENT_SCHEMA = pa.schema(
    [
        ("incident_id", pa.string()),
        ("timestamp_start", pa.string()),
        ("timestamp_end", pa.string()),
        ("start_index", pa.int64()),
        ("end_index", pa.int64()),
        ("entity_id", pa.string()),
        ("service_id", pa.string()),
        ("incident_type", pa.string()),
        ("severity", pa.string()),
        ("label_source", pa.string()),
        ("label_source_type", pa.string()),
        ("source_dataset", pa.string()),
        ("source_file", pa.string()),
        ("evidence_reference", pa.string()),
        ("quality_flag", pa.string()),
        ("metadata_json", pa.string()),
    ]
)

SCHEMAS = {
    "metrics": METRIC_SCHEMA,
    "logs": LOG_SCHEMA,
    "traces": TRACE_SCHEMA,
    "failures": FAILURE_SCHEMA,
    "labels": LABEL_SCHEMA,
    "incidents": INCIDENT_SCHEMA,
}
