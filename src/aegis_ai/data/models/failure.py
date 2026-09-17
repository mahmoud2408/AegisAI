"""Canonical failure observation schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from aegis_ai.data.models.common import (
    CanonicalModel,
    JsonMetadata,
    LabelSourceType,
    QualityFlag,
)


class FailureObservation(CanonicalModel):
    """A row-level failure target or failure-mode target."""

    event_id: str = Field(min_length=1)
    timestamp: datetime | None = None
    timestamp_original: str | None = None
    timezone_assumption: str | None = None
    sequence_index: int | None = Field(default=None, ge=0)
    entity_id: str = Field(min_length=1)
    target: int = Field(ge=0, le=1)
    failure_type: str = Field(min_length=1)
    label_source: str = Field(min_length=1)
    label_source_type: LabelSourceType = LabelSourceType.REAL
    source_dataset: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_row_id: str | None = None
    quality_flag: QualityFlag = QualityFlag.VALID
    metadata: JsonMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_temporal_reference(self) -> FailureObservation:
        if self.timestamp is None and self.sequence_index is None:
            raise ValueError("FailureObservation requires timestamp or sequence_index")
        return self
