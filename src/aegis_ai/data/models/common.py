"""Shared canonical data model primitives."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QualityFlag(StrEnum):
    """Record-level quality state preserved through preprocessing."""

    VALID = "VALID"
    MISSING = "MISSING"
    INVALID = "INVALID"
    DUPLICATE = "DUPLICATE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    IMPUTED = "IMPUTED"
    DERIVED = "DERIVED"


class MetricType(StrEnum):
    """Canonical metric families."""

    CONTINUOUS = "continuous"
    BINARY_STATE = "binary_state"
    COUNT = "count"
    RATE = "rate"
    UNKNOWN = "unknown"


class LabelKind(StrEnum):
    """Meaning of a label value."""

    ANOMALY = "anomaly"
    FAILURE = "failure"
    NORMAL = "normal"
    EVENT = "event"
    WEAK_LABEL = "weak_label"
    SYNTHETIC_LABEL = "synthetic_label"
    UNKNOWN = "unknown"


class LabelSourceType(StrEnum):
    """Provenance class for labels."""

    REAL = "real"
    DERIVED = "derived"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class IncidentSeverity(StrEnum):
    """Incident severity values used by curated or synthetic incidents."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class CanonicalModel(BaseModel):
    """Base class for all canonical records."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class SourceLineage(CanonicalModel):
    """Where a canonical record came from."""

    source_dataset: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_row_id: str | None = None
    source_record_id: str | None = None


class TemporalRecord(CanonicalModel):
    """Common timestamp and sequence-index fields."""

    timestamp: datetime | None = None
    timestamp_original: str | None = None
    timezone_assumption: str | None = None
    sequence_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_temporal_reference(self) -> TemporalRecord:
        if self.timestamp is None and self.sequence_index is None:
            raise ValueError("canonical records require a timestamp or sequence_index")
        return self


JsonMetadata = dict[str, str | int | float | bool | None | list[Any] | dict[str, Any]]
