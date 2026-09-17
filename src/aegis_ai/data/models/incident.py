"""Canonical incident event schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from aegis_ai.data.models.common import (
    CanonicalModel,
    IncidentSeverity,
    JsonMetadata,
    LabelSourceType,
    QualityFlag,
)


class IncidentEvent(CanonicalModel):
    """Known, derived, or synthetic incident interval."""

    incident_id: str = Field(min_length=1)
    timestamp_start: datetime | None = None
    timestamp_end: datetime | None = None
    start_index: int | None = Field(default=None, ge=0)
    end_index: int | None = Field(default=None, ge=0)
    entity_id: str | None = None
    service_id: str | None = None
    incident_type: str = Field(min_length=1)
    severity: IncidentSeverity = IncidentSeverity.UNKNOWN
    label_source: str = Field(min_length=1)
    label_source_type: LabelSourceType = LabelSourceType.UNKNOWN
    source_dataset: str = Field(min_length=1)
    source_file: str | None = None
    evidence_reference: str | None = None
    quality_flag: QualityFlag = QualityFlag.VALID
    metadata: JsonMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_interval_reference(self) -> IncidentEvent:
        has_time = self.timestamp_start is not None or self.timestamp_end is not None
        has_index = self.start_index is not None or self.end_index is not None
        if not has_time and not has_index:
            raise ValueError("IncidentEvent requires time or index interval information")
        return self
