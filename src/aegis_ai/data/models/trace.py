"""Canonical trace event schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from aegis_ai.data.models.common import CanonicalModel, JsonMetadata, QualityFlag


class TraceEvent(CanonicalModel):
    """Canonical representation of a trace span."""

    event_id: str = Field(min_length=1)
    timestamp: datetime
    timestamp_original: str | None = None
    timezone_assumption: str | None = None
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    service_id: str = Field(min_length=1)
    parent_span_id: str | None = None
    operation_name: str = Field(min_length=1)
    duration_ms: float = Field(ge=0)
    status: str | None = None
    source_dataset: str = Field(min_length=1)
    source_file: str | None = None
    source_row_id: str | None = None
    quality_flag: QualityFlag = QualityFlag.VALID
    metadata: JsonMetadata = Field(default_factory=dict)
