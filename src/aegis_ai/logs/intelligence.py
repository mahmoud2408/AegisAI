"""Phase 10 deterministic LogHub intelligence and log evidence."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aegis_ai.data.adapters import create_adapter
from aegis_ai.data.dataset_registry import project_root
from aegis_ai.evidence import (
    EvidenceScoringConfig,
    EvidenceSeverity,
    EvidenceSignal,
    EvidenceSignalType,
    EvidenceValidity,
    ReliabilityLevel,
    aggregate_evidence_signals,
)

LOG_SIGNAL_TYPES = (
    EvidenceSignalType.LOG_RARE_EVENT,
    EvidenceSignalType.LOG_FREQUENCY_ANOMALY,
    EvidenceSignalType.LOG_LEVEL_ANOMALY,
    EvidenceSignalType.LOG_SEQUENCE_ANOMALY,
    EvidenceSignalType.LOG_COMPONENT_ANOMALY,
)
DEFAULT_LOGHUB_FAMILIES = ("openstack", "hdfs", "bgl", "hadoop", "spark", "zookeeper")
ANOMALOUS_LEVELS = frozenset({"WARN", "WARNING", "ERROR", "FATAL", "SEVERE", "CRITICAL"})


@dataclass(frozen=True)
class LogIntelligenceConfig:
    """Configuration for deterministic LogHub profiling and anomaly detection."""

    families: tuple[str, ...] = DEFAULT_LOGHUB_FAMILIES
    telemetry_evidence_run_dir: Path | None = Path(
        "experiments/evidence/phase8b_evidence_engine_20260916"
    )
    bucket_frequency_overrides: dict[str, str] = field(default_factory=dict)
    rare_event_max_frequency: float = 0.01
    rare_event_max_count: int = 3
    rare_event_max_occurrences_per_event: int = 3
    frequency_z_threshold: float = 4.0
    frequency_min_count: int = 5
    level_rate_z_threshold: float = 3.0
    level_rate_absolute_threshold: float = 0.75
    level_min_count: int = 5
    component_z_threshold: float = 4.0
    component_min_count: int = 5
    transition_max_frequency: float = 0.005
    transition_max_count: int = 2
    transition_max_occurrences: int = 3
    max_signals_per_detector_per_dataset: int = 250
    severity_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 0.90}
    )
    log_signal_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            EvidenceSignalType.LOG_RARE_EVENT.value: 0.25,
            EvidenceSignalType.LOG_FREQUENCY_ANOMALY.value: 0.45,
            EvidenceSignalType.LOG_LEVEL_ANOMALY.value: 0.40,
            EvidenceSignalType.LOG_SEQUENCE_ANOMALY.value: 0.35,
            EvidenceSignalType.LOG_COMPONENT_ANOMALY.value: 0.40,
        }
    )
    log_detector_reliability: dict[str, str] = field(
        default_factory=lambda: {
            "rare_event_detector": ReliabilityLevel.LOW.value,
            "frequency_deviation_detector": ReliabilityLevel.LOW.value,
            "log_level_distribution_detector": ReliabilityLevel.LOW.value,
            "transition_rarity_detector": ReliabilityLevel.LOW.value,
            "component_activity_detector": ReliabilityLevel.LOW.value,
        }
    )

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("families must not be empty")
        if not 0.0 < self.rare_event_max_frequency <= 1.0:
            raise ValueError("rare_event_max_frequency must be in (0, 1]")
        if self.rare_event_max_count < 1:
            raise ValueError("rare_event_max_count must be positive")
        if self.rare_event_max_occurrences_per_event < 1:
            raise ValueError("rare_event_max_occurrences_per_event must be positive")
        if self.frequency_z_threshold <= 0:
            raise ValueError("frequency_z_threshold must be positive")
        if self.frequency_min_count < 1:
            raise ValueError("frequency_min_count must be positive")
        if self.level_rate_z_threshold <= 0:
            raise ValueError("level_rate_z_threshold must be positive")
        if not 0.0 < self.level_rate_absolute_threshold <= 1.0:
            raise ValueError("level_rate_absolute_threshold must be in (0, 1]")
        if self.level_min_count < 1:
            raise ValueError("level_min_count must be positive")
        if self.component_z_threshold <= 0:
            raise ValueError("component_z_threshold must be positive")
        if self.component_min_count < 1:
            raise ValueError("component_min_count must be positive")
        if not 0.0 < self.transition_max_frequency <= 1.0:
            raise ValueError("transition_max_frequency must be in (0, 1]")
        if self.transition_max_count < 1:
            raise ValueError("transition_max_count must be positive")
        if self.transition_max_occurrences < 1:
            raise ValueError("transition_max_occurrences must be positive")
        if self.max_signals_per_detector_per_dataset < 1:
            raise ValueError("max_signals_per_detector_per_dataset must be positive")
        _validate_thresholds("severity_thresholds", self.severity_thresholds)
        _validate_weights("log_signal_type_weights", self.log_signal_type_weights)


@dataclass(frozen=True)
class LogEvidenceSignal:
    """Log-specific evidence with provenance, convertible to EvidenceSignal."""

    signal_id: str
    timestamp: pd.Timestamp
    entity_id: str
    component: str | None
    signal_type: EvidenceSignalType
    event_id: str | None
    value: float
    normalized_value: float
    confidence: float
    severity: EvidenceSeverity
    source_dataset: str
    source_file: str
    source_event_ids: tuple[str, ...]
    detector_name: str
    evidence_description: str
    validity: EvidenceValidity
    limitations: str
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signal_type not in LOG_SIGNAL_TYPES:
            raise ValueError("LogEvidenceSignal requires a log-specific signal type")
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        if not 0.0 <= self.normalized_value <= 1.0:
            raise ValueError("normalized_value must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def metric_name(self) -> str:
        """Return the EvidenceSignal metric name representation."""

        if self.component and self.event_id:
            return f"{self.component}:{self.event_id}"
        if self.event_id:
            return self.event_id
        if self.component:
            return self.component
        return "log_activity"

    def to_evidence_signal(self) -> EvidenceSignal:
        """Convert to the shared EvidenceSignal object."""

        return EvidenceSignal(
            signal_id=self.signal_id,
            timestamp=self.timestamp,
            entity_id=self.entity_id,
            signal_type=self.signal_type,
            metric_name=self.metric_name(),
            value=self.value,
            normalized_value=self.normalized_value,
            severity=self.severity,
            confidence=self.confidence,
            source_model=self.detector_name,
            source_dataset=self.source_dataset,
            evidence_description=self.evidence_description,
            validity=self.validity,
            limitations=self.limitations,
        )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly log-evidence record."""

        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "entity_id": self.entity_id,
            "component": self.component,
            "signal_type": self.signal_type.value,
            "event_id": self.event_id,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "source_dataset": self.source_dataset,
            "source_file": self.source_file,
            "source_event_ids": "|".join(self.source_event_ids),
            "detector_name": self.detector_name,
            "evidence_description": self.evidence_description,
            "validity": self.validity.value,
            "limitations": self.limitations,
            "metadata": json.dumps(self.metadata, sort_keys=True),
        }


@dataclass(frozen=True)
class LogIntelligenceExperimentResult:
    """Location and summary for a completed Phase 10 run."""

    run_dir: Path
    metrics: dict[str, Any]


def load_log_intelligence_config(path: Path) -> LogIntelligenceConfig:
    """Load Phase 10 log intelligence YAML configuration."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("log intelligence config must be a YAML mapping")
    default = LogIntelligenceConfig()
    telemetry_run = payload.get("telemetry_evidence_run_dir", default.telemetry_evidence_run_dir)
    return LogIntelligenceConfig(
        families=_tuple_strings(payload.get("families"), default.families),
        telemetry_evidence_run_dir=None if telemetry_run is None else Path(str(telemetry_run)),
        bucket_frequency_overrides=_string_mapping(
            payload.get("bucket_frequency_overrides"),
            default.bucket_frequency_overrides,
        ),
        rare_event_max_frequency=float(
            payload.get("rare_event_max_frequency", default.rare_event_max_frequency)
        ),
        rare_event_max_count=int(payload.get("rare_event_max_count", default.rare_event_max_count)),
        rare_event_max_occurrences_per_event=int(
            payload.get(
                "rare_event_max_occurrences_per_event",
                default.rare_event_max_occurrences_per_event,
            )
        ),
        frequency_z_threshold=float(
            payload.get("frequency_z_threshold", default.frequency_z_threshold)
        ),
        frequency_min_count=int(payload.get("frequency_min_count", default.frequency_min_count)),
        level_rate_z_threshold=float(
            payload.get("level_rate_z_threshold", default.level_rate_z_threshold)
        ),
        level_rate_absolute_threshold=float(
            payload.get(
                "level_rate_absolute_threshold",
                default.level_rate_absolute_threshold,
            )
        ),
        level_min_count=int(payload.get("level_min_count", default.level_min_count)),
        component_z_threshold=float(
            payload.get("component_z_threshold", default.component_z_threshold)
        ),
        component_min_count=int(payload.get("component_min_count", default.component_min_count)),
        transition_max_frequency=float(
            payload.get("transition_max_frequency", default.transition_max_frequency)
        ),
        transition_max_count=int(payload.get("transition_max_count", default.transition_max_count)),
        transition_max_occurrences=int(
            payload.get("transition_max_occurrences", default.transition_max_occurrences)
        ),
        max_signals_per_detector_per_dataset=int(
            payload.get(
                "max_signals_per_detector_per_dataset",
                default.max_signals_per_detector_per_dataset,
            )
        ),
        severity_thresholds=_float_mapping(
            payload.get("severity_thresholds"),
            default.severity_thresholds,
        ),
        log_signal_type_weights=_float_mapping(
            payload.get("log_signal_type_weights"),
            default.log_signal_type_weights,
        ),
        log_detector_reliability=_string_mapping(
            payload.get("log_detector_reliability"),
            default.log_detector_reliability,
        ),
    )


def load_loghub_events(
    family: str,
    *,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load one LogHub family through the existing canonical adapter."""

    adapter = create_adapter(family, root)
    raw = adapter.load_raw()
    profile = adapter.profile(raw)
    records = list(adapter.iter_log_records(raw))
    labels = list(adapter.get_labels(raw))
    events = normalize_log_records(records)
    label_frame = normalize_log_labels(labels)
    return (
        events,
        label_frame,
        {
            "adapter_dataset_id": adapter.dataset_id,
            "family": family,
            "adapter_profile": profile.profile,
            "adapter_status": profile.status,
        },
    )


def normalize_log_records(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Normalize adapter log records into the Phase 10 internal representation."""

    if not records:
        return _empty_events_frame()
    frame = pd.DataFrame(records).copy()
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["timestamp"] = timestamps.dt.tz_convert(None)
    normalized = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "dataset": frame["source_dataset"].astype(str),
            "source_dataset": frame["source_dataset"].astype(str),
            "source_file": frame["source_file"].astype(str),
            "canonical_log_event_id": frame["event_id"].astype(str),
            "sequence_index": pd.to_numeric(frame["sequence_index"], errors="coerce").astype(
                "Int64"
            ),
            "entity_id": _string_or_missing(frame.get("entity_id"), "unknown_log_entity"),
            "component": _string_or_missing(frame.get("service_id"), None),
            "process_id": _string_or_missing(frame.get("process_id"), None),
            "request_id": _string_or_missing(frame.get("request_id"), None),
            "log_level": _string_or_missing(frame.get("log_level"), "UNKNOWN").str.upper(),
            "event_id": _string_or_missing(frame.get("event_type"), "UNKNOWN_EVENT"),
            "event_template": _string_or_missing(frame.get("event_template"), None),
            "message": _string_or_missing(frame.get("message"), ""),
            "source_row_id": _string_or_missing(frame.get("source_row_id"), None),
            "quality_flag": _string_or_missing(frame.get("quality_flag"), "UNKNOWN"),
        }
    )
    normalized["component"] = normalized["component"].fillna(normalized["entity_id"])
    normalized = normalized.dropna(subset=["timestamp"]).sort_values(
        ["dataset", "timestamp", "sequence_index"]
    )
    return normalized.reset_index(drop=True)


def normalize_log_labels(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Normalize adapter label records for optional labeled evaluation."""

    if not records:
        return pd.DataFrame(
            columns=[
                "canonical_log_event_id",
                "timestamp",
                "sequence_index",
                "label_kind",
                "label_value",
                "source_dataset",
            ]
        )
    frame = pd.DataFrame(records).copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    return pd.DataFrame(
        {
            "canonical_log_event_id": frame["event_id"].astype(str),
            "timestamp": frame["timestamp"],
            "sequence_index": pd.to_numeric(frame["sequence_index"], errors="coerce").astype(
                "Int64"
            ),
            "label_kind": frame["label_kind"].astype(str),
            "label_value": frame["label_value"].astype(str),
            "source_dataset": frame["source_dataset"].astype(str),
        }
    )


def select_bucket_frequency(events: pd.DataFrame, config: LogIntelligenceConfig) -> str:
    """Select a dataset-specific aggregation bucket from observed timestamp coverage."""

    if events.empty:
        return "1min"
    dataset = str(events["source_dataset"].iloc[0])
    if dataset in config.bucket_frequency_overrides:
        return config.bucket_frequency_overrides[dataset]
    timestamps = pd.to_datetime(events["timestamp"], errors="coerce").dropna().sort_values()
    if timestamps.empty:
        return "1min"
    duration_seconds = max(float((timestamps.max() - timestamps.min()).total_seconds()), 0.0)
    if duration_seconds <= 120:
        return "1s"
    if duration_seconds <= 3600:
        return "10s"
    if duration_seconds <= 7 * 24 * 3600:
        return "1h"
    if duration_seconds <= 45 * 24 * 3600:
        return "1D"
    return "7D"


def aggregate_log_events(
    events: pd.DataFrame,
    bucket_frequency: str,
) -> dict[str, pd.DataFrame]:
    """Aggregate log events into bucket, event, level, component, and transition tables."""

    _require_columns(
        events,
        {
            "timestamp",
            "source_dataset",
            "event_id",
            "component",
            "log_level",
            "canonical_log_event_id",
        },
    )
    if events.empty:
        return {
            "bucket_summary": pd.DataFrame(),
            "event_frequency": pd.DataFrame(),
            "level_frequency": pd.DataFrame(),
            "component_frequency": pd.DataFrame(),
            "transition_statistics": pd.DataFrame(),
            "event_sequences": pd.DataFrame(),
        }
    frame = events.copy()
    frame["bucket_timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise").dt.floor(
        bucket_frequency
    )
    frame["is_anomalous_level"] = frame["log_level"].astype(str).str.upper().isin(ANOMALOUS_LEVELS)
    bucket_summary = (
        frame.groupby(["source_dataset", "bucket_timestamp"], observed=True)
        .agg(
            event_count=("canonical_log_event_id", "size"),
            anomalous_level_count=("is_anomalous_level", "sum"),
            unique_event_ids=("event_id", "nunique"),
            unique_components=("component", "nunique"),
        )
        .reset_index()
    )
    bucket_summary["anomalous_level_rate"] = bucket_summary[
        "anomalous_level_count"
    ] / bucket_summary["event_count"].clip(lower=1)
    bucket_summary["event_diversity"] = bucket_summary["unique_event_ids"] / bucket_summary[
        "event_count"
    ].clip(lower=1)
    event_frequency = (
        frame.groupby(["source_dataset", "bucket_timestamp", "event_id"], observed=True)
        .agg(
            event_count=("canonical_log_event_id", "size"),
            source_event_ids=("canonical_log_event_id", lambda values: "|".join(map(str, values))),
            component=("component", _mode_string),
            source_file=("source_file", _mode_string),
        )
        .reset_index()
    )
    level_frequency = (
        frame.groupby(["source_dataset", "bucket_timestamp", "log_level"], observed=True)
        .agg(
            level_count=("canonical_log_event_id", "size"),
            source_event_ids=("canonical_log_event_id", lambda values: "|".join(map(str, values))),
            source_file=("source_file", _mode_string),
        )
        .reset_index()
    )
    component_frequency = (
        frame.groupby(["source_dataset", "bucket_timestamp", "component"], observed=True)
        .agg(
            event_count=("canonical_log_event_id", "size"),
            source_event_ids=("canonical_log_event_id", lambda values: "|".join(map(str, values))),
            source_file=("source_file", _mode_string),
        )
        .reset_index()
    )
    sequences = build_event_sequences(frame)
    transition_statistics = (
        sequences.groupby(["source_dataset", "previous_event_id", "event_id"], observed=True)
        .agg(
            transition_count=("canonical_log_event_id", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            component=("component", _mode_string),
            source_file=("source_file", _mode_string),
            source_event_ids=("canonical_log_event_id", lambda values: "|".join(map(str, values))),
        )
        .reset_index()
        if not sequences.empty
        else pd.DataFrame()
    )
    if not transition_statistics.empty:
        totals = transition_statistics.groupby("source_dataset", observed=True)[
            "transition_count"
        ].transform("sum")
        transition_statistics["transition_frequency"] = transition_statistics[
            "transition_count"
        ] / totals.clip(lower=1)
    return {
        "bucket_summary": bucket_summary,
        "event_frequency": event_frequency,
        "level_frequency": level_frequency,
        "component_frequency": component_frequency,
        "transition_statistics": transition_statistics,
        "event_sequences": sequences,
    }


def build_event_sequences(events: pd.DataFrame) -> pd.DataFrame:
    """Represent adjacent event transitions while preserving timestamps."""

    if events.empty:
        return pd.DataFrame()
    ordered = events.sort_values(["source_dataset", "timestamp", "sequence_index"]).copy()
    ordered["previous_event_id"] = ordered.groupby("source_dataset", observed=True)[
        "event_id"
    ].shift(1)
    ordered["previous_component"] = ordered.groupby("source_dataset", observed=True)[
        "component"
    ].shift(1)
    sequences = ordered.dropna(subset=["previous_event_id"]).copy()
    sequences["transition"] = (
        sequences["previous_event_id"].astype(str) + "->" + sequences["event_id"].astype(str)
    )
    sequences["component_transition"] = (
        sequences["previous_component"].astype(str)
        + ":"
        + sequences["previous_event_id"].astype(str)
        + "->"
        + sequences["component"].astype(str)
        + ":"
        + sequences["event_id"].astype(str)
    )
    return sequences.reset_index(drop=True)


def build_log_evidence_signals(
    events: pd.DataFrame,
    config: LogIntelligenceConfig,
) -> tuple[list[LogEvidenceSignal], dict[str, pd.DataFrame]]:
    """Build all deterministic log-evidence signals for a single dataset frame."""

    bucket_frequency = select_bucket_frequency(events, config)
    aggregates = aggregate_log_events(events, bucket_frequency)
    signals: list[LogEvidenceSignal] = []
    signals.extend(detect_rare_events(events, config))
    signals.extend(detect_frequency_anomalies(events, aggregates["event_frequency"], config))
    signals.extend(detect_level_anomalies(events, aggregates["bucket_summary"], config))
    signals.extend(detect_sequence_anomalies(aggregates["event_sequences"], config))
    signals.extend(detect_component_anomalies(events, aggregates["component_frequency"], config))
    return signals, aggregates


def detect_rare_events(
    events: pd.DataFrame,
    config: LogIntelligenceConfig,
) -> list[LogEvidenceSignal]:
    """Detect event IDs that are rare relative to the dataset sample."""

    if events.empty:
        return []
    total = int(events.shape[0])
    counts = (
        events.groupby(["source_dataset", "event_id"], observed=True)
        .agg(
            event_count=("canonical_log_event_id", "size"),
            first_timestamp=("timestamp", "min"),
            component=("component", _mode_string),
            source_file=("source_file", _mode_string),
        )
        .reset_index()
    )
    counts["reference_frequency"] = counts["event_count"] / max(total, 1)
    rare = counts[
        counts["event_count"].le(config.rare_event_max_count)
        | counts["reference_frequency"].le(config.rare_event_max_frequency)
    ].sort_values(["reference_frequency", "event_count", "event_id"])
    signals: list[LogEvidenceSignal] = []
    for row in rare.itertuples(index=False):
        event_rows = events[events["event_id"].astype(str).eq(str(row.event_id))].sort_values(
            ["timestamp", "sequence_index"]
        )
        event_rows = event_rows.head(config.rare_event_max_occurrences_per_event)
        for occurrence in event_rows.itertuples(index=False):
            frequency = float(row.reference_frequency)
            rarity_score = 1.0 - min(frequency / config.rare_event_max_frequency, 1.0)
            normalized = _clamp(max(rarity_score, 1.0 / max(float(row.event_count), 1.0)), 0.0, 1.0)
            signals.append(
                _make_log_signal(
                    timestamp=pd.Timestamp(occurrence.timestamp),
                    entity_id=_log_entity(
                        str(occurrence.source_dataset), str(occurrence.component)
                    ),
                    component=str(occurrence.component),
                    signal_type=EvidenceSignalType.LOG_RARE_EVENT,
                    event_id=str(occurrence.event_id),
                    value=frequency,
                    normalized_value=normalized,
                    source_dataset=str(occurrence.source_dataset),
                    source_file=str(occurrence.source_file),
                    source_event_ids=(str(occurrence.canonical_log_event_id),),
                    detector_name="rare_event_detector",
                    evidence_description=(
                        f"Event {occurrence.event_id} occurred with reference frequency "
                        f"{frequency:.4f} in {occurrence.source_dataset}; "
                        f"component={occurrence.component}."
                    ),
                    validity=EvidenceValidity.LIMITED,
                    limitations=(
                        "Rare log events are investigation evidence only. Rarity does not imply "
                        "incident, root cause, or error."
                    ),
                    metadata={
                        "event_count": int(row.event_count),
                        "reference_frequency": frequency,
                    },
                    config=config,
                )
            )
    return _limit_signals(signals, config.max_signals_per_detector_per_dataset)


def detect_frequency_anomalies(
    events: pd.DataFrame,
    event_frequency: pd.DataFrame,
    config: LogIntelligenceConfig,
) -> list[LogEvidenceSignal]:
    """Detect event-frequency spikes with a robust per-event baseline."""

    if events.empty or event_frequency.empty:
        return []
    dataset = str(events["source_dataset"].iloc[0])
    bucket_frequency = select_bucket_frequency(events, config)
    buckets = pd.date_range(
        events["timestamp"].min().floor(bucket_frequency),
        events["timestamp"].max().ceil(bucket_frequency),
        freq=bucket_frequency,
    )
    signals: list[LogEvidenceSignal] = []
    for event_id, group in event_frequency.groupby("event_id", observed=True):
        counts = (
            group.set_index("bucket_timestamp")["event_count"]
            .reindex(buckets, fill_value=0)
            .astype(float)
        )
        baseline = float(counts.median())
        mad = _median_absolute_deviation(counts.to_numpy(dtype=float))
        scale = max(1.0, 1.4826 * mad, math.sqrt(max(baseline, 1.0)))
        z_scores = (counts - baseline) / scale
        anomalous = counts[
            (counts >= config.frequency_min_count) & (z_scores >= config.frequency_z_threshold)
        ]
        for timestamp, observed in anomalous.items():
            row = group[group["bucket_timestamp"].eq(timestamp)].iloc[0]
            z_score = float(z_scores.loc[timestamp])
            normalized = _clamp(z_score / (config.frequency_z_threshold * 2.0), 0.0, 1.0)
            signals.append(
                _make_log_signal(
                    timestamp=pd.Timestamp(timestamp),
                    entity_id=_log_entity(dataset, str(row["component"])),
                    component=str(row["component"]),
                    signal_type=EvidenceSignalType.LOG_FREQUENCY_ANOMALY,
                    event_id=str(event_id),
                    value=float(observed),
                    normalized_value=normalized,
                    source_dataset=dataset,
                    source_file=str(row["source_file"]),
                    source_event_ids=_split_ids(str(row["source_event_ids"])),
                    detector_name="frequency_deviation_detector",
                    evidence_description=(
                        f"Event {event_id} count rose to {int(observed)} in bucket "
                        f"{pd.Timestamp(timestamp).isoformat()} versus baseline {baseline:.2f} "
                        f"(robust z={z_score:.2f})."
                    ),
                    validity=EvidenceValidity.LIMITED,
                    limitations=(
                        "Frequency spikes describe abnormal log volume relative to this sample. "
                        "They are not calibrated incident probabilities."
                    ),
                    metadata={
                        "baseline_rate": baseline,
                        "observed_rate": float(observed),
                        "robust_z": z_score,
                        "bucket_frequency": bucket_frequency,
                    },
                    config=config,
                )
            )
    return _limit_signals(signals, config.max_signals_per_detector_per_dataset)


def detect_level_anomalies(
    events: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    config: LogIntelligenceConfig,
) -> list[LogEvidenceSignal]:
    """Detect unusual WARN/ERROR/FATAL/SEVERE level-rate buckets."""

    if events.empty or bucket_summary.empty:
        return []
    rates = bucket_summary["anomalous_level_rate"].astype(float)
    baseline = float(rates.median())
    mad = _median_absolute_deviation(rates.to_numpy(dtype=float))
    scale = max(0.01, 1.4826 * mad)
    z_scores = (rates - baseline) / scale
    selected = bucket_summary[
        bucket_summary["anomalous_level_count"].ge(config.level_min_count)
        & (
            z_scores.ge(config.level_rate_z_threshold)
            | bucket_summary["anomalous_level_rate"].ge(config.level_rate_absolute_threshold)
        )
    ].copy()
    signals: list[LogEvidenceSignal] = []
    for index, row in selected.iterrows():
        z_score = float(z_scores.loc[index])
        deviation_score = z_score / (config.level_rate_z_threshold * 2.0)
        absolute_score = float(row["anomalous_level_rate"]) / config.level_rate_absolute_threshold
        normalized = _clamp(max(deviation_score, absolute_score), 0.0, 1.0)
        bucket_events = events[
            pd.to_datetime(events["timestamp"])
            .dt.floor(select_bucket_frequency(events, config))
            .eq(row["bucket_timestamp"])
        ]
        anomalous_events = bucket_events[
            bucket_events["log_level"].astype(str).str.upper().isin(ANOMALOUS_LEVELS)
        ]
        component = (
            _mode_string(anomalous_events["component"]) if not anomalous_events.empty else None
        )
        source_ids = anomalous_events["canonical_log_event_id"].astype(str).head(20).tolist()
        signals.append(
            _make_log_signal(
                timestamp=pd.Timestamp(row["bucket_timestamp"]),
                entity_id=_log_entity(str(row["source_dataset"]), component),
                component=component,
                signal_type=EvidenceSignalType.LOG_LEVEL_ANOMALY,
                event_id=None,
                value=float(row["anomalous_level_rate"]),
                normalized_value=normalized,
                source_dataset=str(row["source_dataset"]),
                source_file=_mode_string(anomalous_events["source_file"])
                if not anomalous_events.empty
                else "",
                source_event_ids=tuple(source_ids),
                detector_name="log_level_distribution_detector",
                evidence_description=(
                    f"Anomalous log-level rate reached {row['anomalous_level_rate']:.3f} "
                    f"with {int(row['anomalous_level_count'])} WARN/ERROR-like events "
                    f"versus baseline {baseline:.3f}."
                ),
                validity=EvidenceValidity.LIMITED,
                limitations=(
                    "Log-level distribution shifts are evidence of abnormal logging behavior, "
                    "not root-cause proof."
                ),
                metadata={
                    "baseline_rate": baseline,
                    "observed_rate": float(row["anomalous_level_rate"]),
                    "robust_z": z_score,
                    "absolute_rate_threshold": config.level_rate_absolute_threshold,
                    "anomalous_level_count": int(row["anomalous_level_count"]),
                },
                config=config,
            )
        )
    return _limit_signals(signals, config.max_signals_per_detector_per_dataset)


def detect_sequence_anomalies(
    sequences: pd.DataFrame,
    config: LogIntelligenceConfig,
) -> list[LogEvidenceSignal]:
    """Detect rare adjacent EventId transitions."""

    if sequences.empty:
        return []
    stats = (
        sequences.groupby(["source_dataset", "previous_event_id", "event_id"], observed=True)
        .agg(
            transition_count=("canonical_log_event_id", "size"),
            component=("component", _mode_string),
            source_file=("source_file", _mode_string),
        )
        .reset_index()
    )
    stats["transition_frequency"] = stats["transition_count"] / max(int(sequences.shape[0]), 1)
    rare = stats[
        stats["transition_count"].le(config.transition_max_count)
        | stats["transition_frequency"].le(config.transition_max_frequency)
    ].sort_values(["transition_frequency", "transition_count", "previous_event_id", "event_id"])
    signals: list[LogEvidenceSignal] = []
    for row in rare.itertuples(index=False):
        transition_rows = sequences[
            sequences["previous_event_id"].astype(str).eq(str(row.previous_event_id))
            & sequences["event_id"].astype(str).eq(str(row.event_id))
        ].sort_values(["timestamp", "sequence_index"])
        for occurrence in transition_rows.head(config.transition_max_occurrences).itertuples(
            index=False
        ):
            frequency = float(row.transition_frequency)
            normalized = 1.0 - min(frequency / config.transition_max_frequency, 1.0)
            normalized = _clamp(
                max(normalized, 1.0 / max(float(row.transition_count), 1.0)), 0.0, 1.0
            )
            signals.append(
                _make_log_signal(
                    timestamp=pd.Timestamp(occurrence.timestamp),
                    entity_id=_log_entity(
                        str(occurrence.source_dataset), str(occurrence.component)
                    ),
                    component=str(occurrence.component),
                    signal_type=EvidenceSignalType.LOG_SEQUENCE_ANOMALY,
                    event_id=f"{occurrence.previous_event_id}->{occurrence.event_id}",
                    value=frequency,
                    normalized_value=normalized,
                    source_dataset=str(occurrence.source_dataset),
                    source_file=str(occurrence.source_file),
                    source_event_ids=(str(occurrence.canonical_log_event_id),),
                    detector_name="transition_rarity_detector",
                    evidence_description=(
                        "Rare log transition "
                        f"{occurrence.previous_event_id}->{occurrence.event_id} occurred "
                        f"with reference frequency {frequency:.4f}."
                    ),
                    validity=EvidenceValidity.LIMITED,
                    limitations=(
                        "Transition rarity is n-gram evidence only; sequence rarity does not "
                        "prove incident or causality."
                    ),
                    metadata={
                        "previous_event_id": str(occurrence.previous_event_id),
                        "current_event_id": str(occurrence.event_id),
                        "transition_frequency": frequency,
                        "transition_count": int(row.transition_count),
                    },
                    config=config,
                )
            )
    return _limit_signals(signals, config.max_signals_per_detector_per_dataset)


def detect_component_anomalies(
    events: pd.DataFrame,
    component_frequency: pd.DataFrame,
    config: LogIntelligenceConfig,
) -> list[LogEvidenceSignal]:
    """Detect component-level activity spikes."""

    if events.empty or component_frequency.empty:
        return []
    dataset = str(events["source_dataset"].iloc[0])
    bucket_frequency = select_bucket_frequency(events, config)
    buckets = pd.date_range(
        events["timestamp"].min().floor(bucket_frequency),
        events["timestamp"].max().ceil(bucket_frequency),
        freq=bucket_frequency,
    )
    signals: list[LogEvidenceSignal] = []
    for component, group in component_frequency.groupby("component", observed=True):
        counts = (
            group.set_index("bucket_timestamp")["event_count"]
            .reindex(buckets, fill_value=0)
            .astype(float)
        )
        baseline = float(counts.median())
        mad = _median_absolute_deviation(counts.to_numpy(dtype=float))
        scale = max(1.0, 1.4826 * mad, math.sqrt(max(baseline, 1.0)))
        z_scores = (counts - baseline) / scale
        anomalous = counts[
            (counts >= config.component_min_count) & (z_scores >= config.component_z_threshold)
        ]
        for timestamp, observed in anomalous.items():
            row = group[group["bucket_timestamp"].eq(timestamp)].iloc[0]
            z_score = float(z_scores.loc[timestamp])
            normalized = _clamp(z_score / (config.component_z_threshold * 2.0), 0.0, 1.0)
            signals.append(
                _make_log_signal(
                    timestamp=pd.Timestamp(timestamp),
                    entity_id=_log_entity(dataset, str(component)),
                    component=str(component),
                    signal_type=EvidenceSignalType.LOG_COMPONENT_ANOMALY,
                    event_id=None,
                    value=float(observed),
                    normalized_value=normalized,
                    source_dataset=dataset,
                    source_file=str(row["source_file"]),
                    source_event_ids=_split_ids(str(row["source_event_ids"])),
                    detector_name="component_activity_detector",
                    evidence_description=(
                        f"Component {component} emitted {int(observed)} events in bucket "
                        f"{pd.Timestamp(timestamp).isoformat()} versus baseline {baseline:.2f} "
                        f"(robust z={z_score:.2f})."
                    ),
                    validity=EvidenceValidity.LIMITED,
                    limitations=(
                        "Component activity spikes identify abnormal log volume by component; "
                        "they do not infer service dependencies."
                    ),
                    metadata={
                        "baseline_rate": baseline,
                        "observed_rate": float(observed),
                        "robust_z": z_score,
                        "bucket_frequency": bucket_frequency,
                    },
                    config=config,
                )
            )
    return _limit_signals(signals, config.max_signals_per_detector_per_dataset)


def run_log_intelligence_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: LogIntelligenceConfig | None = None,
) -> LogIntelligenceExperimentResult:
    """Run the Phase 10 deterministic LogHub intelligence experiment."""

    cfg = config or LogIntelligenceConfig()
    project = project_root() if root is None else root
    resolved_run_id = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    base_output = output_root or project / "experiments" / "logs"
    run_dir = base_output / resolved_run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "config.json", _config_to_json(cfg))

    started = time.perf_counter()
    all_events: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []
    profile_rows: list[dict[str, Any]] = []
    event_statistics: list[pd.DataFrame] = []
    bucket_summaries: list[pd.DataFrame] = []
    event_frequencies: list[pd.DataFrame] = []
    level_frequencies: list[pd.DataFrame] = []
    component_frequencies: list[pd.DataFrame] = []
    transition_statistics: list[pd.DataFrame] = []
    event_sequences: list[pd.DataFrame] = []
    all_log_signals: list[LogEvidenceSignal] = []

    for family in cfg.families:
        events, labels, metadata = load_loghub_events(family, root=project)
        if events.empty:
            continue
        bucket_frequency = select_bucket_frequency(events, cfg)
        signals, aggregates = build_log_evidence_signals(events, cfg)
        all_events.append(events)
        all_labels.append(labels)
        all_log_signals.extend(signals)
        profile_rows.append(profile_log_dataset(events, labels, metadata, bucket_frequency))
        event_statistics.append(build_event_statistics(events))
        bucket_summaries.append(aggregates["bucket_summary"])
        event_frequencies.append(aggregates["event_frequency"])
        level_frequencies.append(aggregates["level_frequency"])
        component_frequencies.append(aggregates["component_frequency"])
        transition_statistics.append(aggregates["transition_statistics"])
        event_sequences.append(aggregates["event_sequences"])

    events_frame = pd.concat(all_events, ignore_index=True) if all_events else _empty_events_frame()
    labels_frame = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()
    profiles = pd.DataFrame(profile_rows)
    event_stats = _concat_or_empty(event_statistics)
    bucket_summary = _concat_or_empty(bucket_summaries)
    event_frequency = _concat_or_empty(event_frequencies)
    level_frequency = _concat_or_empty(level_frequencies)
    component_frequency = _concat_or_empty(component_frequencies)
    transition_stats = _concat_or_empty(transition_statistics)
    sequences = _concat_or_empty(event_sequences)

    evidence_signals = [signal.to_evidence_signal() for signal in all_log_signals]
    risk_config = _log_risk_config(cfg)
    risk_signals = aggregate_evidence_signals(evidence_signals, risk_config)
    label_evaluation = evaluate_labeled_log_signals(events_frame, labels_frame, all_log_signals)
    structural_evaluation = evaluate_structural_log_signals(events_frame, all_log_signals)
    comparison = compare_telemetry_and_log_evidence(project, cfg, all_log_signals, risk_signals)
    case_studies = select_log_case_studies(
        events_frame,
        bucket_summary,
        all_log_signals,
        cfg,
    )

    profiles.to_csv(run_dir / "profiles.csv", index=False)
    _save_json(run_dir / "profiles.json", profiles.to_dict(orient="records"))
    event_stats.to_csv(run_dir / "event_statistics.csv", index=False)
    bucket_summary.to_csv(run_dir / "bucket_summary.csv", index=False)
    event_frequency.to_csv(run_dir / "event_frequency_by_bucket.csv", index=False)
    level_frequency.to_csv(run_dir / "log_level_frequency_by_bucket.csv", index=False)
    component_frequency.to_csv(run_dir / "component_activity_by_bucket.csv", index=False)
    transition_stats.to_csv(run_dir / "transition_statistics.csv", index=False)
    sequences.head(5000).to_csv(run_dir / "event_sequences.csv", index=False)
    log_signal_frame = pd.DataFrame([signal.to_record() for signal in all_log_signals])
    evidence_frame = pd.DataFrame([signal.to_record() for signal in evidence_signals])
    risk_frame = pd.DataFrame([signal.to_record() for signal in risk_signals])
    log_signal_frame.to_csv(run_dir / "log_anomaly_signals.csv", index=False)
    evidence_frame.to_csv(run_dir / "evidence_signals.csv", index=False)
    risk_frame.to_csv(run_dir / "risk_signals.csv", index=False)
    if not evidence_frame.empty:
        evidence_frame.to_parquet(run_dir / "evidence_signals.parquet", index=False)
    _save_json(run_dir / "case_studies.json", case_studies)
    _save_json(run_dir / "comparison.json", comparison)
    pd.DataFrame([comparison]).to_csv(run_dir / "telemetry_log_comparison.csv", index=False)

    _plot_event_frequency(bucket_summary, figures_dir / "event_frequency_over_time.png")
    _plot_event_distribution(event_stats, figures_dir / "event_type_distribution.png")
    _plot_log_level_distribution(events_frame, figures_dir / "log_level_distribution.png")
    _plot_anomaly_timeline(log_signal_frame, figures_dir / "anomaly_events_timeline.png")
    _plot_component_activity(component_frequency, figures_dir / "component_activity.png")
    _plot_sequence_anomalies(log_signal_frame, figures_dir / "sequence_anomaly_examples.png")
    _plot_telemetry_log_comparison(
        comparison, figures_dir / "telemetry_log_evidence_comparison.png"
    )

    metrics = {
        "run_id": resolved_run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "research_question": (
            "Can structured log-event behavior provide additional evidence for detecting "
            "and characterizing incidents beyond telemetry-only evidence?"
        ),
        "datasets_analyzed": sorted(profiles["source_dataset"].tolist())
        if not profiles.empty
        else [],
        "event_counts": profiles.set_index("source_dataset")["row_count"].to_dict()
        if not profiles.empty
        else {},
        "bucket_frequencies": profiles.set_index("source_dataset")["bucket_frequency"].to_dict()
        if not profiles.empty
        else {},
        "event_distribution_top": _top_event_distribution(event_stats),
        "log_evidence_signal_count": int(len(all_log_signals)),
        "evidence_counts_by_type": _counts_by_signal_type(all_log_signals),
        "log_risk_signal_count": int(len(risk_signals)),
        "label_evaluation": label_evaluation,
        "structural_evaluation": structural_evaluation,
        "comparison": comparison,
        "case_studies": case_studies,
        "limitations": [
            "LogHub safe-default files are 2k samples, not full production log streams.",
            "Only BGL has real row labels in the local LogHub samples.",
            "Rare events, frequency spikes, level shifts, and rare transitions are evidence only.",
            "OpenStack/HDFS/Hadoop/Spark/Zookeeper signals are not ground-truth incidents.",
            "Log evidence is low-reliability and low-weight so it cannot dominate risk scoring.",
        ],
    }
    _save_json(run_dir / "metrics.json", metrics)
    _save_json(
        run_dir / "metadata.json",
        {
            "phase": "10",
            "name": "deterministic_log_intelligence",
            "created_at_utc": metrics["created_at_utc"],
            "source": "LogHub structured 2k samples",
        },
    )
    _write_run_report(run_dir, metrics)
    return LogIntelligenceExperimentResult(run_dir=run_dir, metrics=metrics)


def profile_log_dataset(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: dict[str, Any],
    bucket_frequency: str,
) -> dict[str, Any]:
    """Profile one normalized LogHub event table."""

    timestamps = pd.to_datetime(events["timestamp"], errors="coerce")
    deltas = timestamps.sort_values().diff().dropna().dt.total_seconds()
    label_counts = labels["label_kind"].value_counts().to_dict() if not labels.empty else {}
    return {
        "family": metadata["family"],
        "source_dataset": str(events["source_dataset"].iloc[0]),
        "row_count": int(events.shape[0]),
        "timestamp_start": timestamps.min().isoformat(),
        "timestamp_end": timestamps.max().isoformat(),
        "duration_seconds": float((timestamps.max() - timestamps.min()).total_seconds()),
        "median_delta_seconds": float(deltas.median()) if not deltas.empty else 0.0,
        "bucket_frequency": bucket_frequency,
        "event_id_count": int(events["event_id"].nunique()),
        "component_count": int(events["component"].nunique()),
        "log_level_count": int(events["log_level"].nunique()),
        "request_id_count": int(events["request_id"].dropna().nunique()),
        "process_id_count": int(events["process_id"].dropna().nunique()),
        "label_count": int(labels.shape[0]),
        "label_counts": json.dumps(label_counts, sort_keys=True),
        "source_file": str(events["source_file"].iloc[0]),
        "adapter_status": metadata["adapter_status"],
    }


def build_event_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """Build source-dataset event frequency statistics."""

    if events.empty:
        return pd.DataFrame()
    stats = (
        events.groupby(["source_dataset", "event_id"], observed=True)
        .agg(
            event_count=("canonical_log_event_id", "size"),
            component=("component", _mode_string),
            log_level=("log_level", _mode_string),
            event_template=("event_template", _mode_string),
            source_file=("source_file", _mode_string),
        )
        .reset_index()
    )
    totals = stats.groupby("source_dataset", observed=True)["event_count"].transform("sum")
    stats["reference_frequency"] = stats["event_count"] / totals.clip(lower=1)
    return stats.sort_values(["source_dataset", "event_count"], ascending=[True, False])


def evaluate_labeled_log_signals(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    signals: Sequence[LogEvidenceSignal],
) -> dict[str, Any]:
    """Evaluate log evidence against real row labels where they exist."""

    if labels.empty:
        return {"available": False, "reason": "no local LogHub row labels were loaded"}
    labeled_datasets = sorted(labels["source_dataset"].astype(str).unique().tolist())
    predicted_ids = {
        source_id
        for signal in signals
        for source_id in signal.source_event_ids
        if signal.source_dataset in labeled_datasets
    }
    label_map = labels.set_index("canonical_log_event_id")["label_kind"].astype(str).to_dict()
    event_ids = set(label_map)
    anomaly_ids = {event_id for event_id, kind in label_map.items() if kind == "anomaly"}
    true_positive = len(predicted_ids.intersection(anomaly_ids))
    false_positive = len(predicted_ids.intersection(event_ids.difference(anomaly_ids)))
    false_negative = len(anomaly_ids.difference(predicted_ids))
    true_negative = len(event_ids.difference(anomaly_ids).difference(predicted_ids))
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alarm_rate = (
        false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
    )
    return {
        "available": True,
        "labeled_datasets": labeled_datasets,
        "label_type": "row-level LogHub BGL labels",
        "predicted_event_rows": int(len(predicted_ids)),
        "anomaly_rows": int(len(anomaly_ids)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "accuracy_claimed": False,
    }


def evaluate_structural_log_signals(
    events: pd.DataFrame,
    signals: Sequence[LogEvidenceSignal],
) -> dict[str, Any]:
    """Evaluate deterministic structural properties where labels are absent."""

    signal_frame = pd.DataFrame([signal.to_record() for signal in signals])
    if signal_frame.empty:
        return {"signal_count": 0}
    return {
        "signal_count": int(signal_frame.shape[0]),
        "datasets_with_signals": sorted(signal_frame["source_dataset"].unique().tolist()),
        "signals_per_1000_events": {
            dataset: float(
                count / max(int(events[events["source_dataset"].eq(dataset)].shape[0]), 1) * 1000
            )
            for dataset, count in signal_frame["source_dataset"].value_counts().to_dict().items()
        },
        "mean_normalized_value": float(signal_frame["normalized_value"].mean()),
        "max_normalized_value": float(signal_frame["normalized_value"].max()),
        "provenance_completeness": float(
            (
                signal_frame["source_file"].astype(str).ne("")
                & signal_frame["source_event_ids"].astype(str).ne("")
            ).mean()
        ),
        "deterministic_reproducibility": True,
    }


def compare_telemetry_and_log_evidence(
    project: Path,
    config: LogIntelligenceConfig,
    log_signals: Sequence[LogEvidenceSignal],
    log_risk_signals: Sequence[Any],
) -> dict[str, Any]:
    """Compare telemetry/evidence-only artifacts against log evidence outputs."""

    telemetry_dir = (
        None
        if config.telemetry_evidence_run_dir is None
        else _resolve_project_path(project, config.telemetry_evidence_run_dir)
    )
    telemetry_evidence_count = 0
    telemetry_risk_count = 0
    telemetry_evidence_types: list[str] = []
    if telemetry_dir is not None and telemetry_dir.exists():
        telemetry_evidence_path = telemetry_dir / "evidence_signals.csv"
        telemetry_risk_path = telemetry_dir / "risk_signals.csv"
        if telemetry_evidence_path.exists():
            telemetry_evidence = pd.read_csv(telemetry_evidence_path)
            telemetry_evidence_count = int(telemetry_evidence.shape[0])
            telemetry_evidence_types = sorted(
                telemetry_evidence["signal_type"].astype(str).unique().tolist()
            )
        if telemetry_risk_path.exists():
            telemetry_risk_count = int(pd.read_csv(telemetry_risk_path).shape[0])
    log_types = sorted({signal.signal_type.value for signal in log_signals})
    return {
        "telemetry_evidence_run_dir": str(telemetry_dir) if telemetry_dir else None,
        "telemetry_evidence_count": telemetry_evidence_count,
        "telemetry_risk_signal_count": telemetry_risk_count,
        "telemetry_evidence_types": telemetry_evidence_types,
        "log_evidence_count": int(len(log_signals)),
        "log_risk_signal_count": int(len(log_risk_signals)),
        "log_evidence_types": log_types,
        "new_modalities_added": ["logs"] if log_signals else [],
        "complementary_information": bool(log_signals),
        "claim": (
            "Logs add complementary evidence types and log entities. No cross-dataset "
            "incident-diagnosis improvement is claimed because LogHub and MetroPT are not "
            "the same system."
        ),
    }


def select_log_case_studies(
    events: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    signals: Sequence[LogEvidenceSignal],
    config: LogIntelligenceConfig,
) -> dict[str, Any]:
    """Select normal, rare, frequency, level, sequence, and noisy log cases."""

    signal_by_type: dict[str, list[LogEvidenceSignal]] = {}
    for signal in signals:
        signal_by_type.setdefault(signal.signal_type.value, []).append(signal)
    cases: dict[str, Any] = {}
    normal_bucket = _normal_log_bucket(events, bucket_summary, signals)
    cases["normal_log_period"] = normal_bucket
    for key, signal_type in [
        ("rare_event_example", EvidenceSignalType.LOG_RARE_EVENT.value),
        ("event_frequency_spike", EvidenceSignalType.LOG_FREQUENCY_ANOMALY.value),
        ("error_rate_spike", EvidenceSignalType.LOG_LEVEL_ANOMALY.value),
        ("sequence_anomaly", EvidenceSignalType.LOG_SEQUENCE_ANOMALY.value),
    ]:
        selected = signal_by_type.get(signal_type, [])
        cases[key] = (
            _case_from_signal(events, bucket_summary, selected[0], config)
            if selected
            else {"available": False, "reason": f"no {signal_type} signal generated"}
        )
    noisy_candidates = sorted(
        signals,
        key=lambda signal: (
            signal.validity.value != EvidenceValidity.VALID.value,
            abs(signal.normalized_value - config.severity_thresholds["low"]),
            signal.timestamp,
        ),
        reverse=True,
    )
    cases["noisy_difficult_case"] = (
        _case_from_signal(events, bucket_summary, noisy_candidates[0], config)
        if noisy_candidates
        else {"available": False, "reason": "no log evidence signals generated"}
    )
    return cases


def _normal_log_bucket(
    events: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    signals: Sequence[LogEvidenceSignal],
) -> dict[str, Any]:
    if bucket_summary.empty:
        return {"available": False}
    signal_buckets = {(signal.source_dataset, signal.timestamp) for signal in signals}
    candidates = bucket_summary[
        ~bucket_summary.apply(
            lambda row: (
                (str(row["source_dataset"]), pd.Timestamp(row["bucket_timestamp"]))
                in signal_buckets
            ),
            axis=1,
        )
    ].sort_values(["anomalous_level_rate", "event_count"])
    if candidates.empty:
        candidates = bucket_summary.sort_values(["anomalous_level_rate", "event_count"])
    row = candidates.iloc[0]
    bucket_events = events[
        events["source_dataset"].eq(row["source_dataset"])
        & pd.to_datetime(events["timestamp"])
        .dt.floor("1s")
        .ge(pd.Timestamp(row["bucket_timestamp"]))
    ].head(8)
    return {
        "available": True,
        "source_dataset": str(row["source_dataset"]),
        "timestamp": pd.Timestamp(row["bucket_timestamp"]).isoformat(),
        "aggregated_behavior": {
            "event_count": int(row["event_count"]),
            "anomalous_level_rate": float(row["anomalous_level_rate"]),
            "unique_event_ids": int(row["unique_event_ids"]),
        },
        "detected_log_signal": None,
        "raw_events": _raw_event_preview(bucket_events),
    }


def _case_from_signal(
    events: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    signal: LogEvidenceSignal,
    config: LogIntelligenceConfig,
) -> dict[str, Any]:
    dataset_events = events[events["source_dataset"].eq(signal.source_dataset)]
    bucket_frequency = select_bucket_frequency(dataset_events, config)
    bucket = signal.timestamp.floor(bucket_frequency)
    raw_events = dataset_events[
        pd.to_datetime(dataset_events["timestamp"]).dt.floor(bucket_frequency).eq(bucket)
    ].head(10)
    bucket_rows = bucket_summary[
        bucket_summary["source_dataset"].eq(signal.source_dataset)
        & pd.to_datetime(bucket_summary["bucket_timestamp"]).eq(bucket)
    ]
    aggregated = bucket_rows.iloc[0].to_dict() if not bucket_rows.empty else {}
    for key, value in list(aggregated.items()):
        if isinstance(value, pd.Timestamp):
            aggregated[key] = value.isoformat()
    return {
        "available": True,
        "source_dataset": signal.source_dataset,
        "timestamp": signal.timestamp.isoformat(),
        "raw_events": _raw_event_preview(raw_events),
        "aggregated_behavior": aggregated,
        "detected_log_signal": signal.to_record(),
        "evidence_signal": signal.to_evidence_signal().to_record(),
    }


def _make_log_signal(
    *,
    timestamp: pd.Timestamp,
    entity_id: str,
    component: str | None,
    signal_type: EvidenceSignalType,
    event_id: str | None,
    value: float,
    normalized_value: float,
    source_dataset: str,
    source_file: str,
    source_event_ids: tuple[str, ...],
    detector_name: str,
    evidence_description: str,
    validity: EvidenceValidity,
    limitations: str,
    metadata: dict[str, str | int | float | bool | None],
    config: LogIntelligenceConfig,
) -> LogEvidenceSignal:
    normalized = _clamp(normalized_value, 0.0, 1.0)
    severity = _severity_from_normalized(normalized, config.severity_thresholds)
    reliability = ReliabilityLevel(config.log_detector_reliability.get(detector_name, "LOW"))
    confidence = _clamp(
        _reliability_weight(reliability) * (0.5 + 0.5 * normalized),
        0.0,
        1.0,
    )
    signal_id = _signal_id(
        timestamp,
        entity_id,
        signal_type.value,
        event_id or component or "log_activity",
        detector_name,
        value,
    )
    return LogEvidenceSignal(
        signal_id=signal_id,
        timestamp=timestamp,
        entity_id=entity_id,
        component=component,
        signal_type=signal_type,
        event_id=event_id,
        value=value,
        normalized_value=normalized,
        confidence=confidence,
        severity=severity,
        source_dataset=source_dataset,
        source_file=source_file,
        source_event_ids=source_event_ids,
        detector_name=detector_name,
        evidence_description=evidence_description,
        validity=validity,
        limitations=limitations,
        metadata=metadata,
    )


def _log_risk_config(config: LogIntelligenceConfig) -> EvidenceScoringConfig:
    base = EvidenceScoringConfig()
    source_reliability = dict(base.source_reliability)
    for family in config.families:
        dataset = f"loghub-{family}"
        for detector, reliability in config.log_detector_reliability.items():
            source_reliability[f"{dataset}:{detector}"] = reliability
    signal_type_weights = {**base.signal_type_weights, **config.log_signal_type_weights}
    return EvidenceScoringConfig(
        signal_type_weights=signal_type_weights,
        source_reliability=source_reliability,
    )


def _severity_from_normalized(
    value: float,
    thresholds: dict[str, float],
) -> EvidenceSeverity:
    if value >= thresholds["critical"]:
        return EvidenceSeverity.CRITICAL
    if value >= thresholds["high"]:
        return EvidenceSeverity.HIGH
    if value >= thresholds["medium"]:
        return EvidenceSeverity.MEDIUM
    if value >= thresholds["low"]:
        return EvidenceSeverity.LOW
    return EvidenceSeverity.INFO


def _reliability_weight(reliability: ReliabilityLevel) -> float:
    return {
        ReliabilityLevel.HIGH: 1.0,
        ReliabilityLevel.MEDIUM: 0.75,
        ReliabilityLevel.LOW: 0.45,
        ReliabilityLevel.RESEARCH_ONLY: 0.15,
    }[reliability]


def _median_absolute_deviation(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _limit_signals(
    signals: list[LogEvidenceSignal],
    limit: int,
) -> list[LogEvidenceSignal]:
    if len(signals) <= limit:
        return signals
    ordered = sorted(
        signals,
        key=lambda signal: (signal.normalized_value, signal.confidence, signal.timestamp),
        reverse=True,
    )
    return sorted(ordered[:limit], key=lambda signal: signal.timestamp)


def _log_entity(source_dataset: str, component: str | None) -> str:
    safe_component = (component or "logs").replace(" ", "_")
    return f"{source_dataset}:{safe_component}"


def _signal_id(
    timestamp: pd.Timestamp,
    entity_id: str,
    signal_type: str,
    event_id: str,
    detector_name: str,
    value: float,
) -> str:
    payload = "|".join(
        [
            timestamp.isoformat(),
            entity_id,
            signal_type,
            event_id,
            detector_name,
            f"{value:.12g}",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _mode_string(values: Iterable[Any]) -> str:
    series = pd.Series(list(values)).dropna().astype(str)
    if series.empty:
        return ""
    return str(series.value_counts().index[0])


def _split_ids(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split("|") if item)


def _raw_event_preview(events: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    columns = [
        "timestamp",
        "source_dataset",
        "event_id",
        "component",
        "log_level",
        "message",
        "canonical_log_event_id",
    ]
    preview = (
        events.head(limit)[columns].copy() if not events.empty else pd.DataFrame(columns=columns)
    )
    records = preview.to_dict(orient="records")
    for record in records:
        record["timestamp"] = pd.Timestamp(record["timestamp"]).isoformat()
    return cast(list[dict[str, Any]], records)


def _counts_by_signal_type(signals: Sequence[LogEvidenceSignal]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signal in signals:
        counts[signal.signal_type.value] = counts.get(signal.signal_type.value, 0) + 1
    return counts


def _top_event_distribution(event_stats: pd.DataFrame) -> dict[str, Any]:
    if event_stats.empty:
        return {}
    rows: dict[str, Any] = {}
    for dataset, group in event_stats.groupby("source_dataset", observed=True):
        rows[str(dataset)] = group.head(10)[
            ["event_id", "event_count", "reference_frequency", "component", "log_level"]
        ].to_dict(orient="records")
    return rows


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "dataset",
            "source_dataset",
            "source_file",
            "canonical_log_event_id",
            "sequence_index",
            "entity_id",
            "component",
            "process_id",
            "request_id",
            "log_level",
            "event_id",
            "event_template",
            "message",
            "source_row_id",
            "quality_flag",
        ]
    )


def _string_or_missing(series: Any, missing: str | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    result = pd.Series(series, copy=False).astype("object")
    result = result.where(result.notna(), missing)
    result = result.replace({"": missing, "nan": missing, "None": missing})
    return result


def _concat_or_empty(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def _tuple_strings(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list | tuple):
        raise ValueError("expected a list of strings")
    return tuple(str(item) for item in value)


def _float_mapping(value: object, default: dict[str, float]) -> dict[str, float]:
    if value is None:
        return dict(default)
    if not isinstance(value, dict):
        raise ValueError("expected a mapping of string keys to numeric values")
    return {str(key): float(item) for key, item in value.items()}


def _string_mapping(value: object, default: dict[str, str]) -> dict[str, str]:
    if value is None:
        return dict(default)
    if not isinstance(value, dict):
        raise ValueError("expected a mapping of string keys to string values")
    return {str(key): str(item) for key, item in value.items()}


def _validate_thresholds(name: str, thresholds: dict[str, float]) -> None:
    required = ("low", "medium", "high", "critical")
    missing = sorted(set(required).difference(thresholds))
    if missing:
        raise ValueError(f"{name} missing threshold keys: {missing}")
    values = [thresholds[key] for key in required]
    if values != sorted(values):
        raise ValueError(f"{name} thresholds must be ordered")


def _validate_weights(name: str, weights: dict[str, float]) -> None:
    for key, value in weights.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}.{key} must be in [0, 1]")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _resolve_project_path(project: Path, path: Path) -> Path:
    return path if path.is_absolute() else project / path


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"data frame is missing required columns: {missing}")


def _config_to_json(config: LogIntelligenceConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["telemetry_evidence_run_dir"] = (
        None
        if payload["telemetry_evidence_run_dir"] is None
        else str(payload["telemetry_evidence_run_dir"])
    )
    return payload


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.integer | np.floating):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _plot_event_frequency(bucket_summary: pd.DataFrame, output_path: Path) -> None:
    if bucket_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for dataset, group in bucket_summary.groupby("source_dataset", observed=True):
        ordered = group.sort_values("bucket_timestamp")
        ax.plot(
            ordered["bucket_timestamp"], ordered["event_count"], label=str(dataset), linewidth=1.1
        )
    ax.set_title("Log Event Frequency over Time")
    ax.set_ylabel("events per bucket")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # type: ignore[no-untyped-call]
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_event_distribution(event_stats: pd.DataFrame, output_path: Path) -> None:
    if event_stats.empty:
        return
    top = event_stats.sort_values("event_count", ascending=False).head(20)
    labels = top["source_dataset"].astype(str) + ":" + top["event_id"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels[::-1], top["event_count"].iloc[::-1], color="#2563eb")
    ax.set_title("Top Log Event Types")
    ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_log_level_distribution(events: pd.DataFrame, output_path: Path) -> None:
    if events.empty:
        return
    counts = (
        events.groupby(["source_dataset", "log_level"], observed=True).size().unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Log Level Distribution")
    ax.set_ylabel("events")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_anomaly_timeline(signals: pd.DataFrame, output_path: Path) -> None:
    if signals.empty:
        return
    frame = signals.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", format="mixed")
    counts = (
        frame.groupby([pd.Grouper(key="timestamp", freq="1D"), "signal_type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(ax=ax, linewidth=1.1)
    ax.set_title("Log Evidence Signals on Timeline")
    ax.set_ylabel("signals")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_component_activity(component_frequency: pd.DataFrame, output_path: Path) -> None:
    if component_frequency.empty:
        return
    top = (
        component_frequency.groupby(["source_dataset", "component"], observed=True)["event_count"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    labels = [f"{dataset}:{component}" for dataset, component in top.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels[::-1], top.iloc[::-1], color="#0f766e")
    ax.set_title("Top Component Activity")
    ax.set_xlabel("events")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_sequence_anomalies(signals: pd.DataFrame, output_path: Path) -> None:
    if signals.empty:
        return
    selected = signals[signals["signal_type"].eq(EvidenceSignalType.LOG_SEQUENCE_ANOMALY.value)]
    if selected.empty:
        return
    top = selected["event_id"].value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind="barh", ax=ax, color="#9333ea")
    ax.set_title("Sequence Anomaly Examples")
    ax.set_xlabel("signals")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_telemetry_log_comparison(comparison: dict[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["telemetry evidence", "log evidence", "telemetry risk", "log risk"]
    values = [
        comparison.get("telemetry_evidence_count", 0),
        comparison.get("log_evidence_count", 0),
        comparison.get("telemetry_risk_signal_count", 0),
        comparison.get("log_risk_signal_count", 0),
    ]
    ax.bar(labels, values, color=["#2563eb", "#0f766e", "#60a5fa", "#14b8a6"])
    ax.set_title("Telemetry Evidence vs Log Evidence")
    ax.set_ylabel("artifact rows")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_run_report(run_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Phase 10 Log Intelligence Run",
        "",
        f"- Run ID: `{metrics['run_id']}`",
        f"- Datasets analyzed: {', '.join(metrics['datasets_analyzed'])}",
        f"- Log evidence signals: {metrics['log_evidence_signal_count']}",
        f"- Log risk buckets: {metrics['log_risk_signal_count']}",
        "- RCA/incident improvement claimed: false",
        "",
        "## Evidence Counts",
        "",
    ]
    for signal_type, count in sorted(metrics["evidence_counts_by_type"].items()):
        lines.append(f"- {signal_type}: {count}")
    lines.extend(
        [
            "",
            "## Label Evaluation",
            "",
            json.dumps(metrics["label_evaluation"], indent=2, sort_keys=True),
            "",
            "## Comparison",
            "",
            metrics["comparison"]["claim"],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in metrics["limitations"])
    lines.append("")
    run_dir.joinpath("run_report.md").write_text("\n".join(lines), encoding="utf-8")
