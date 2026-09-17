"""Canonical label event schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from aegis_ai.data.models.common import (
    CanonicalModel,
    JsonMetadata,
    LabelKind,
    LabelSourceType,
    QualityFlag,
)


class LabelEvent(CanonicalModel):
    """Canonical point or interval label with explicit semantics."""

    label_id: str = Field(min_length=1)
    label_kind: LabelKind
    label_source_type: LabelSourceType
    label_value: str = Field(min_length=1)
    label_source: str = Field(min_length=1)
    event_id: str | None = None
    timestamp: datetime | None = None
    timestamp_original: str | None = None
    timezone_assumption: str | None = None
    sequence_index: int | None = Field(default=None, ge=0)
    timestamp_start: datetime | None = None
    timestamp_end: datetime | None = None
    start_index: int | None = Field(default=None, ge=0)
    end_index: int | None = Field(default=None, ge=0)
    entity_id: str | None = None
    service_id: str | None = None
    source_dataset: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_row_id: str | None = None
    quality_flag: QualityFlag = QualityFlag.VALID
    metadata: JsonMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_point_or_interval(self) -> LabelEvent:
        has_point = self.timestamp is not None or self.sequence_index is not None
        has_time_interval = self.timestamp_start is not None or self.timestamp_end is not None
        has_index_interval = self.start_index is not None or self.end_index is not None
        if not has_point and not has_time_interval and not has_index_interval:
            raise ValueError("LabelEvent requires point or interval information")
        return self
