"""Canonical log event schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from aegis_ai.data.models.common import CanonicalModel, JsonMetadata, QualityFlag


class LogEvent(CanonicalModel):
    """Canonical representation of a structured or parsed log line."""

    event_id: str = Field(min_length=1)
    timestamp: datetime | None = None
    timestamp_original: str | None = None
    timezone_assumption: str | None = None
    sequence_index: int | None = Field(default=None, ge=0)
    entity_id: str | None = None
    service_id: str | None = None
    host_id: str | None = None
    process_id: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    log_level: str | None = None
    event_type: str | None = None
    event_template: str | None = None
    message: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_row_id: str | None = None
    quality_flag: QualityFlag = QualityFlag.VALID
    metadata: JsonMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_temporal_reference(self) -> LogEvent:
        if self.timestamp is None and self.sequence_index is None:
            raise ValueError("LogEvent requires timestamp or sequence_index")
        return self
