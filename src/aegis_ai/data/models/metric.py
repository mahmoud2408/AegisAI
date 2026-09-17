"""Canonical metric observation schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from aegis_ai.data.models.common import (
    CanonicalModel,
    JsonMetadata,
    MetricType,
    QualityFlag,
)


class MetricObservation(CanonicalModel):
    """Long-format canonical representation of one metric value."""

    event_id: str = Field(min_length=1)
    timestamp: datetime | None = None
    timestamp_original: str | None = None
    timezone_assumption: str | None = None
    sequence_index: int | None = Field(default=None, ge=0)
    entity_id: str = Field(min_length=1)
    service_id: str | None = None
    host_id: str | None = None
    metric_name: str = Field(min_length=1)
    metric_value: float
    metric_type: MetricType = MetricType.UNKNOWN
    unit: str | None = None
    split: str | None = None
    source_dataset: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_row_id: str | None = None
    quality_flag: QualityFlag = QualityFlag.VALID
    metadata: JsonMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_temporal_reference(self) -> MetricObservation:
        if self.timestamp is None and self.sequence_index is None:
            raise ValueError("MetricObservation requires timestamp or sequence_index")
        return self
