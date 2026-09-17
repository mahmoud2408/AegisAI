"""Phase 8B evidence engine and deterministic risk-signal aggregation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from aegis_ai.data.adapters.metropt import BINARY_STATE_COLUMNS
from aegis_ai.data.dataset_registry import project_root
from aegis_ai.ml.prediction.failure_prediction import MetroPTExperimentConfig


class EvidenceSignalType(StrEnum):
    """Supported evidence categories for incident investigation."""

    CURRENT_ANOMALY = "CURRENT_ANOMALY"
    FORECAST_DEVIATION = "FORECAST_DEVIATION"
    TREND = "TREND"
    FAILURE_SIGNAL = "FAILURE_SIGNAL"
    STATE_TRANSITION = "STATE_TRANSITION"
    HISTORICAL_CONTEXT = "HISTORICAL_CONTEXT"


class EvidenceSeverity(StrEnum):
    """Ordinal evidence severity used by deterministic scoring."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceValidity(StrEnum):
    """Signal validity labels exposed to downstream users."""

    VALID = "VALID"
    LIMITED = "LIMITED"
    RESEARCH_ONLY = "RESEARCH_ONLY"


class ReliabilityLevel(StrEnum):
    """Objective reliability tier for a source model or rule."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    RESEARCH_ONLY = "RESEARCH_ONLY"


@dataclass(frozen=True)
class EvidenceSignal:
    """Single auditable evidence item produced at or before its timestamp."""

    signal_id: str
    timestamp: pd.Timestamp
    entity_id: str
    signal_type: EvidenceSignalType
    metric_name: str
    value: float
    normalized_value: float
    severity: EvidenceSeverity
    confidence: float
    source_model: str
    source_dataset: str
    evidence_description: str
    validity: EvidenceValidity
    limitations: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        if not 0.0 <= self.normalized_value <= 1.0:
            raise ValueError("normalized_value must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_record(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly representation."""

        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "entity_id": self.entity_id,
            "signal_type": self.signal_type.value,
            "metric_name": self.metric_name,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "source_model": self.source_model,
            "source_dataset": self.source_dataset,
            "evidence_description": self.evidence_description,
            "validity": self.validity.value,
            "limitations": self.limitations,
        }


@dataclass(frozen=True)
class RiskSignal:
    """Aggregated evidence score for an entity and time bucket."""

    timestamp: pd.Timestamp
    entity_id: str
    risk_score: float
    severity: EvidenceSeverity
    evidence_count: int
    dominant_evidence_types: tuple[str, ...]
    confidence: float
    model_sources: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_record(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly representation."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "entity_id": self.entity_id,
            "risk_score": self.risk_score,
            "severity": self.severity.value,
            "evidence_count": self.evidence_count,
            "dominant_evidence_types": "|".join(self.dominant_evidence_types),
            "confidence": self.confidence,
            "model_sources": "|".join(self.model_sources),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class EvidenceScoringConfig:
    """Configurable thresholds and artifact locations for the evidence engine."""

    forecasting_run_dir: Path = Path(
        "experiments/forecasting/metropt/phase8_metropt_forecasting_20260915"
    )
    robustness_prediction_path: Path | None = Path(
        "experiments/prediction/metropt_robustness/"
        "phase7b_metropt_robustness_20260914/predictions.parquet"
    )
    entity_id: str = "metropt_compressor"
    source_dataset: str = "metropt"
    selected_forecast_model: str = "lstm_multivariate"
    selected_forecast_horizon_minutes: int = 5
    selected_split: str = "test"
    aggregation_frequency: str = "5min"
    trend_lookback_minutes: int = 30
    trend_min_points: int = 3
    historical_context_lookback_minutes: int = 60
    historical_context_min_prior_signals: int = 3
    max_signals_per_type: int = 5000
    max_state_transition_signals: int = 2000
    max_failure_signals: int = 1500
    pre_failure_context_hours: float = 6.0
    current_anomaly_z_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 2.5, "medium": 3.0, "high": 4.0, "critical": 5.0}
    )
    forecast_residual_z_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 1.0, "medium": 1.5, "high": 2.5, "critical": 4.0}
    )
    trend_z_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.8, "medium": 1.2, "high": 2.0, "critical": 3.0}
    )
    failure_probability_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.02, "medium": 0.05, "high": 0.10, "critical": 0.20}
    )
    historical_count_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 3.0, "medium": 5.0, "high": 8.0, "critical": 12.0}
    )
    risk_severity_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.15, "medium": 0.35, "high": 0.60, "critical": 0.80}
    )
    signal_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            EvidenceSignalType.CURRENT_ANOMALY.value: 1.00,
            EvidenceSignalType.FORECAST_DEVIATION.value: 0.90,
            EvidenceSignalType.TREND.value: 0.65,
            EvidenceSignalType.FAILURE_SIGNAL.value: 0.30,
            EvidenceSignalType.STATE_TRANSITION.value: 0.35,
            EvidenceSignalType.HISTORICAL_CONTEXT.value: 0.45,
        }
    )
    reliability_weights: dict[str, float] = field(
        default_factory=lambda: {
            ReliabilityLevel.HIGH.value: 1.00,
            ReliabilityLevel.MEDIUM.value: 0.75,
            ReliabilityLevel.LOW.value: 0.45,
            ReliabilityLevel.RESEARCH_ONLY.value: 0.15,
        }
    )
    source_reliability: dict[str, str] = field(
        default_factory=lambda: {
            "metropt:train_normal_range_rule": ReliabilityLevel.MEDIUM.value,
            "metropt:lstm_multivariate": ReliabilityLevel.MEDIUM.value,
            "metropt:moving_average": ReliabilityLevel.MEDIUM.value,
            "metropt:rolling_trend_rule": ReliabilityLevel.LOW.value,
            "metropt:state_transition_rule": ReliabilityLevel.LOW.value,
            "metropt:historical_context_rule": ReliabilityLevel.LOW.value,
            "metropt:metropt_failure_prediction": ReliabilityLevel.RESEARCH_ONLY.value,
        }
    )

    def __post_init__(self) -> None:
        if self.selected_forecast_horizon_minutes <= 0:
            raise ValueError("selected_forecast_horizon_minutes must be positive")
        if self.trend_lookback_minutes <= 0:
            raise ValueError("trend_lookback_minutes must be positive")
        if self.trend_min_points < 2:
            raise ValueError("trend_min_points must be at least 2")
        if self.historical_context_lookback_minutes <= 0:
            raise ValueError("historical_context_lookback_minutes must be positive")
        if self.historical_context_min_prior_signals < 1:
            raise ValueError("historical_context_min_prior_signals must be positive")
        if self.max_signals_per_type < 1:
            raise ValueError("max_signals_per_type must be positive")
        if self.max_state_transition_signals < 1:
            raise ValueError("max_state_transition_signals must be positive")
        if self.max_failure_signals < 1:
            raise ValueError("max_failure_signals must be positive")
        if self.pre_failure_context_hours <= 0:
            raise ValueError("pre_failure_context_hours must be positive")
        for name, thresholds in [
            ("current_anomaly_z_thresholds", self.current_anomaly_z_thresholds),
            ("forecast_residual_z_thresholds", self.forecast_residual_z_thresholds),
            ("trend_z_thresholds", self.trend_z_thresholds),
            ("failure_probability_thresholds", self.failure_probability_thresholds),
            ("historical_count_thresholds", self.historical_count_thresholds),
            ("risk_severity_thresholds", self.risk_severity_thresholds),
        ]:
            _validate_thresholds(name, thresholds)
        for name, weights in [
            ("signal_type_weights", self.signal_type_weights),
            ("reliability_weights", self.reliability_weights),
        ]:
            _validate_weights(name, weights)


@dataclass(frozen=True)
class EvidenceEngineExperimentResult:
    """Location and summary for a completed Phase 8B run."""

    run_dir: Path
    metrics: dict[str, Any]


def load_evidence_scoring_config(path: Path) -> EvidenceScoringConfig:
    """Load the Phase 8B evidence-scoring YAML config."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("evidence scoring config must be a YAML mapping")
    default = EvidenceScoringConfig()
    robustness_value = payload.get("robustness_prediction_path", default.robustness_prediction_path)
    return EvidenceScoringConfig(
        forecasting_run_dir=Path(
            str(payload.get("forecasting_run_dir", default.forecasting_run_dir))
        ),
        robustness_prediction_path=(
            None if robustness_value is None else Path(str(robustness_value))
        ),
        entity_id=str(payload.get("entity_id", default.entity_id)),
        source_dataset=str(payload.get("source_dataset", default.source_dataset)),
        selected_forecast_model=str(
            payload.get("selected_forecast_model", default.selected_forecast_model)
        ),
        selected_forecast_horizon_minutes=int(
            payload.get(
                "selected_forecast_horizon_minutes",
                default.selected_forecast_horizon_minutes,
            )
        ),
        selected_split=str(payload.get("selected_split", default.selected_split)),
        aggregation_frequency=str(
            payload.get("aggregation_frequency", default.aggregation_frequency)
        ),
        trend_lookback_minutes=int(
            payload.get("trend_lookback_minutes", default.trend_lookback_minutes)
        ),
        trend_min_points=int(payload.get("trend_min_points", default.trend_min_points)),
        historical_context_lookback_minutes=int(
            payload.get(
                "historical_context_lookback_minutes",
                default.historical_context_lookback_minutes,
            )
        ),
        historical_context_min_prior_signals=int(
            payload.get(
                "historical_context_min_prior_signals",
                default.historical_context_min_prior_signals,
            )
        ),
        max_signals_per_type=int(payload.get("max_signals_per_type", default.max_signals_per_type)),
        max_state_transition_signals=int(
            payload.get("max_state_transition_signals", default.max_state_transition_signals)
        ),
        max_failure_signals=int(payload.get("max_failure_signals", default.max_failure_signals)),
        pre_failure_context_hours=float(
            payload.get("pre_failure_context_hours", default.pre_failure_context_hours)
        ),
        current_anomaly_z_thresholds=_float_mapping(
            payload.get("current_anomaly_z_thresholds"),
            default.current_anomaly_z_thresholds,
        ),
        forecast_residual_z_thresholds=_float_mapping(
            payload.get("forecast_residual_z_thresholds"),
            default.forecast_residual_z_thresholds,
        ),
        trend_z_thresholds=_float_mapping(
            payload.get("trend_z_thresholds"),
            default.trend_z_thresholds,
        ),
        failure_probability_thresholds=_float_mapping(
            payload.get("failure_probability_thresholds"),
            default.failure_probability_thresholds,
        ),
        historical_count_thresholds=_float_mapping(
            payload.get("historical_count_thresholds"),
            default.historical_count_thresholds,
        ),
        risk_severity_thresholds=_float_mapping(
            payload.get("risk_severity_thresholds"),
            default.risk_severity_thresholds,
        ),
        signal_type_weights=_float_mapping(
            payload.get("signal_type_weights"),
            default.signal_type_weights,
        ),
        reliability_weights=_float_mapping(
            payload.get("reliability_weights"),
            default.reliability_weights,
        ),
        source_reliability=_string_mapping(
            payload.get("source_reliability"),
            default.source_reliability,
        ),
    )


def build_current_anomaly_signals(
    observations: pd.DataFrame,
    normal_ranges: pd.DataFrame,
    config: EvidenceScoringConfig,
) -> list[EvidenceSignal]:
    """Build current anomaly evidence from observed values and train-only normal ranges."""

    _require_columns(observations, {"target_timestamp", "sensor", "y_true"})
    ranges = _normal_range_lookup(normal_ranges)
    signals: list[EvidenceSignal] = []
    for row in observations.drop_duplicates(["target_timestamp", "sensor"]).itertuples(index=False):
        sensor = str(row.sensor)
        if sensor not in ranges:
            continue
        timestamp = pd.Timestamp(row.target_timestamp)
        value = float(row.y_true)
        stats = ranges[sensor]
        abs_z = _abs_z(value, stats)
        outside_range = value < stats["low_quantile"] or value > stats["high_quantile"]
        if abs_z < config.current_anomaly_z_thresholds["low"] and not outside_range:
            continue
        normalized = _normalize(abs_z, config.current_anomaly_z_thresholds["critical"])
        severity = severity_from_value(abs_z, config.current_anomaly_z_thresholds)
        reliability = resolve_reliability("train_normal_range_rule", config.source_dataset, config)
        signals.append(
            _make_signal(
                timestamp=timestamp,
                entity_id=config.entity_id,
                signal_type=EvidenceSignalType.CURRENT_ANOMALY,
                metric_name=sensor,
                value=value,
                normalized_value=normalized,
                severity=severity,
                reliability=reliability,
                source_model="train_normal_range_rule",
                source_dataset=config.source_dataset,
                evidence_description=(
                    f"{sensor} observed value {value:.4g} has train-normal absolute z-score "
                    f"{abs_z:.2f}; empirical range breach={outside_range}."
                ),
                validity=EvidenceValidity.VALID,
                limitations=(
                    "Normal range is empirical and train-only; it is not a physics safety limit "
                    "or causal explanation."
                ),
                config=config,
            )
        )
    return _limit_signals(signals, config.max_signals_per_type)


def build_forecast_deviation_signals(
    forecasts: pd.DataFrame,
    normal_ranges: pd.DataFrame,
    config: EvidenceScoringConfig,
) -> list[EvidenceSignal]:
    """Build residual evidence from Phase 8 forecasts without timestamp leakage."""

    _require_columns(
        forecasts,
        {
            "model",
            "horizon_minutes",
            "split",
            "sensor",
            "input_end_timestamp",
            "target_timestamp",
            "y_true",
            "y_pred",
            "absolute_error",
        },
    )
    ranges = _normal_range_lookup(normal_ranges)
    selected = _selected_forecasts(forecasts, config)
    signals: list[EvidenceSignal] = []
    for row in selected.itertuples(index=False):
        sensor = str(row.sensor)
        if sensor not in ranges:
            continue
        target_timestamp = pd.Timestamp(row.target_timestamp)
        input_end_timestamp = pd.Timestamp(row.input_end_timestamp)
        if target_timestamp < input_end_timestamp:
            raise ValueError("forecast target timestamp is before input end timestamp")
        actual = float(row.y_true)
        forecast = float(row.y_pred)
        residual = actual - forecast
        stats = ranges[sensor]
        residual_z = abs(residual) / max(stats["std"], 1e-8)
        if residual_z < config.forecast_residual_z_thresholds["low"]:
            continue
        normalized = _normalize(residual_z, config.forecast_residual_z_thresholds["critical"])
        severity = severity_from_value(residual_z, config.forecast_residual_z_thresholds)
        reliability = resolve_reliability(
            config.selected_forecast_model,
            config.source_dataset,
            config,
        )
        signals.append(
            _make_signal(
                timestamp=target_timestamp,
                entity_id=config.entity_id,
                signal_type=EvidenceSignalType.FORECAST_DEVIATION,
                metric_name=sensor,
                value=residual,
                normalized_value=normalized,
                severity=severity,
                reliability=reliability,
                source_model=config.selected_forecast_model,
                source_dataset=config.source_dataset,
                evidence_description=(
                    f"{sensor} forecast residual at target time is {residual:.4g} "
                    f"({residual_z:.2f} train-standard-deviation units) for "
                    f"{config.selected_forecast_horizon_minutes} minute horizon."
                ),
                validity=EvidenceValidity.VALID,
                limitations=(
                    "Residual evidence is timestamped at the observed target time. It is not "
                    "available at forecast input time and must not be interpreted as advance "
                    "failure probability."
                ),
                config=config,
            )
        )
    return _limit_signals(signals, config.max_signals_per_type)


def build_trend_signals(
    observations: pd.DataFrame,
    normal_ranges: pd.DataFrame,
    config: EvidenceScoringConfig,
) -> list[EvidenceSignal]:
    """Build rolling trend evidence using only observations at or before each timestamp."""

    _require_columns(observations, {"target_timestamp", "sensor", "y_true"})
    ranges = _normal_range_lookup(normal_ranges)
    rows = observations[["target_timestamp", "sensor", "y_true"]].drop_duplicates().copy()
    rows["target_timestamp"] = pd.to_datetime(rows["target_timestamp"], errors="raise")
    lookback = pd.Timedelta(minutes=config.trend_lookback_minutes)
    signals: list[EvidenceSignal] = []
    for sensor, group in rows.sort_values("target_timestamp").groupby("sensor", observed=True):
        sensor_name = str(sensor)
        if sensor_name not in ranges:
            continue
        stats = ranges[sensor_name]
        group = group.reset_index(drop=True)
        timestamps = pd.to_datetime(group["target_timestamp"], errors="raise")
        values = pd.to_numeric(group["y_true"], errors="coerce")
        for index, timestamp in enumerate(timestamps):
            window_mask = (timestamps >= timestamp - lookback) & (timestamps <= timestamp)
            window_values = values[window_mask].dropna()
            if int(window_values.shape[0]) < config.trend_min_points:
                continue
            first = float(window_values.iloc[0])
            current = float(values.iloc[index])
            delta = current - first
            trend_z = abs(delta) / max(stats["std"], 1e-8)
            if trend_z < config.trend_z_thresholds["low"]:
                continue
            normalized = _normalize(trend_z, config.trend_z_thresholds["critical"])
            severity = severity_from_value(trend_z, config.trend_z_thresholds)
            reliability = resolve_reliability("rolling_trend_rule", config.source_dataset, config)
            direction = "upward" if delta >= 0 else "downward"
            signals.append(
                _make_signal(
                    timestamp=pd.Timestamp(timestamp),
                    entity_id=config.entity_id,
                    signal_type=EvidenceSignalType.TREND,
                    metric_name=sensor_name,
                    value=delta,
                    normalized_value=normalized,
                    severity=severity,
                    reliability=reliability,
                    source_model="rolling_trend_rule",
                    source_dataset=config.source_dataset,
                    evidence_description=(
                        f"{sensor_name} moved {direction} by {delta:.4g} over the prior "
                        f"{config.trend_lookback_minutes} minutes "
                        f"({trend_z:.2f} train-standard-deviation units)."
                    ),
                    validity=EvidenceValidity.LIMITED,
                    limitations=(
                        "Trend is descriptive and uses past/current observations only; trend "
                        "direction is not causal attribution."
                    ),
                    config=config,
                )
            )
    return _limit_signals(signals, config.max_signals_per_type)


def build_state_transition_signals(
    raw_metropt: pd.DataFrame,
    config: EvidenceScoringConfig,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[EvidenceSignal]:
    """Build operating-state transition evidence from binary MetroPT state columns."""

    available = [column for column in BINARY_STATE_COLUMNS if column in raw_metropt.columns]
    if not available:
        return []
    _require_columns(raw_metropt, {"timestamp", *available})
    indexed = raw_metropt.sort_values("timestamp").set_index("timestamp")
    states = indexed.loc[:, available].apply(pd.to_numeric, errors="coerce")
    states = states.resample(config.aggregation_frequency).median().ffill().dropna(how="all")
    states = states.loc[(states.index >= start) & (states.index <= end)]
    if states.empty:
        return []
    binary = (states >= 0.5).astype(int)
    changed = binary.ne(binary.shift(1))
    signals: list[EvidenceSignal] = []
    reliability = resolve_reliability("state_transition_rule", config.source_dataset, config)
    for timestamp, changed_row in changed.iloc[1:].iterrows():
        changed_metrics = [column for column in available if bool(changed_row[column])]
        for column in changed_metrics:
            new_value = float(binary.loc[timestamp, column])
            normalized = 0.65 if new_value >= 0.5 else 0.45
            severity = EvidenceSeverity.MEDIUM if new_value >= 0.5 else EvidenceSeverity.LOW
            state_name = "active" if new_value >= 0.5 else "inactive"
            signals.append(
                _make_signal(
                    timestamp=pd.Timestamp(timestamp),
                    entity_id=config.entity_id,
                    signal_type=EvidenceSignalType.STATE_TRANSITION,
                    metric_name=column,
                    value=new_value,
                    normalized_value=normalized,
                    severity=severity,
                    reliability=reliability,
                    source_model="state_transition_rule",
                    source_dataset=config.source_dataset,
                    evidence_description=f"{column} operating state transitioned to {state_name}.",
                    validity=EvidenceValidity.LIMITED,
                    limitations=(
                        "State transitions provide operating context only. They do not imply "
                        "fault, causation, or incident probability."
                    ),
                    config=config,
                )
            )
    return _limit_signals(signals, config.max_state_transition_signals)


def build_research_failure_signals(
    predictions: pd.DataFrame,
    config: EvidenceScoringConfig,
) -> list[EvidenceSignal]:
    """Build tightly capped failure-prediction evidence from Phase 7B robustness artifacts."""

    if predictions.empty:
        return []
    _require_columns(
        predictions,
        {"candidate_id", "model", "split", "timestamp", "probability", "threshold", "y_pred"},
    )
    selected = predictions[predictions["split"].astype(str).eq(config.selected_split)].copy()
    if selected.empty:
        return []
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="raise")
    selected["probability"] = pd.to_numeric(selected["probability"], errors="coerce")
    selected = selected.dropna(subset=["probability"])
    if selected.empty:
        return []
    if "h6p0__random_forest" in set(selected["candidate_id"].astype(str)):
        selected = selected[selected["candidate_id"].astype(str).eq("h6p0__random_forest")]
    if "random_forest" in set(selected["model"].astype(str)):
        selected = selected[selected["model"].astype(str).eq("random_forest")]
    selected = selected[
        selected["probability"].ge(config.failure_probability_thresholds["low"])
    ].sort_values("probability", ascending=False)
    selected = _limit_dataframe(selected, config.max_failure_signals)
    reliability = ReliabilityLevel.RESEARCH_ONLY
    signals: list[EvidenceSignal] = []
    for row in selected.itertuples(index=False):
        probability = float(row.probability)
        candidate_id = str(row.candidate_id)
        model = str(row.model)
        normalized = _normalize(probability, config.failure_probability_thresholds["critical"])
        severity = severity_from_value(probability, config.failure_probability_thresholds)
        source_model = "metropt_failure_prediction"
        signals.append(
            _make_signal(
                timestamp=pd.Timestamp(row.timestamp),
                entity_id=config.entity_id,
                signal_type=EvidenceSignalType.FAILURE_SIGNAL,
                metric_name="failure_prediction_probability",
                value=probability,
                normalized_value=normalized,
                severity=severity,
                reliability=reliability,
                source_model=source_model,
                source_dataset=config.source_dataset,
                evidence_description=(
                    f"Phase 7B {model} candidate {candidate_id} emitted a calibrated research "
                    f"score of {probability:.4f}."
                ),
                validity=EvidenceValidity.RESEARCH_ONLY,
                limitations=(
                    "MetroPT failure prediction is marked RESEARCH_ONLY due to target fragility, "
                    "class imbalance, and limited failure-event validation. It is capped by a "
                    "low aggregation weight and must not dominate decisions."
                ),
                config=config,
            )
        )
    return signals


def build_historical_context_signals(
    base_signals: Sequence[EvidenceSignal],
    config: EvidenceScoringConfig,
) -> list[EvidenceSignal]:
    """Create context signals from repeated prior evidence without peeking forward."""

    if not base_signals:
        return []
    ordered = sorted(base_signals, key=lambda signal: signal.timestamp)
    lookback = pd.Timedelta(minutes=config.historical_context_lookback_minutes)
    prior_by_key: dict[tuple[str, str], list[pd.Timestamp]] = defaultdict(list)
    emitted_keys: set[tuple[pd.Timestamp, str, str]] = set()
    signals: list[EvidenceSignal] = []
    reliability = resolve_reliability("historical_context_rule", config.source_dataset, config)
    for signal in ordered:
        if signal.signal_type is EvidenceSignalType.HISTORICAL_CONTEXT:
            continue
        key = (signal.metric_name, signal.signal_type.value)
        prior = [timestamp for timestamp in prior_by_key[key] if timestamp < signal.timestamp]
        cutoff = signal.timestamp - lookback
        prior = [timestamp for timestamp in prior if timestamp >= cutoff]
        prior_by_key[key] = [*prior, signal.timestamp]
        count = len(prior)
        if count < config.historical_context_min_prior_signals:
            continue
        emit_key = (
            signal.timestamp.floor(config.aggregation_frequency),
            signal.metric_name,
            signal.signal_type.value,
        )
        if emit_key in emitted_keys:
            continue
        emitted_keys.add(emit_key)
        normalized = _normalize(float(count), config.historical_count_thresholds["critical"])
        severity = severity_from_value(float(count), config.historical_count_thresholds)
        signals.append(
            _make_signal(
                timestamp=signal.timestamp,
                entity_id=config.entity_id,
                signal_type=EvidenceSignalType.HISTORICAL_CONTEXT,
                metric_name=signal.metric_name,
                value=float(count),
                normalized_value=normalized,
                severity=severity,
                reliability=reliability,
                source_model="historical_context_rule",
                source_dataset=config.source_dataset,
                evidence_description=(
                    f"{count} prior {signal.signal_type.value} signals for {signal.metric_name} "
                    f"occurred within the previous "
                    f"{config.historical_context_lookback_minutes} minutes."
                ),
                validity=EvidenceValidity.LIMITED,
                limitations=(
                    "Historical context counts only prior evidence. It does not prove recurrence "
                    "cause or incident probability."
                ),
                config=config,
            )
        )
    return _limit_signals(signals, config.max_signals_per_type)


def aggregate_evidence_signals(
    signals: Sequence[EvidenceSignal],
    config: EvidenceScoringConfig,
) -> list[RiskSignal]:
    """Aggregate normalized evidence into deterministic risk signals."""

    if not signals:
        return []
    frame = pd.DataFrame([signal.to_record() for signal in signals])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["bucket_timestamp"] = frame["timestamp"].dt.floor(config.aggregation_frequency)
    risk_signals: list[RiskSignal] = []
    grouped = frame.groupby(["entity_id", "bucket_timestamp"], sort=True, observed=True)
    for (entity_id, bucket_timestamp), group in grouped:
        contributions: list[float] = []
        confidence_terms: list[float] = []
        contribution_by_type: dict[str, float] = defaultdict(float)
        sources = sorted(set(group["source_model"].astype(str).tolist()))
        for row in group.itertuples(index=False):
            signal_type = str(row.signal_type)
            source_model = str(row.source_model)
            source_dataset = str(row.source_dataset)
            validity = str(row.validity)
            reliability = resolve_reliability(source_model, source_dataset, config)
            if validity == EvidenceValidity.RESEARCH_ONLY.value:
                reliability = ReliabilityLevel.RESEARCH_ONLY
            type_weight = config.signal_type_weights.get(signal_type, 0.0)
            reliability_weight = config.reliability_weights[reliability.value]
            normalized_value = float(row.normalized_value)
            contribution = _clamp(normalized_value * type_weight * reliability_weight, 0.0, 0.95)
            contributions.append(contribution)
            confidence_terms.append(float(row.confidence))
            contribution_by_type[signal_type] += contribution
        risk_score = 1.0 - float(np.prod([1.0 - value for value in contributions]))
        severity = severity_from_value(risk_score, config.risk_severity_thresholds)
        dominant = tuple(
            key
            for key, _value in sorted(
                contribution_by_type.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        )
        confidence = _clamp(
            float(np.mean(confidence_terms)) * _evidence_count_confidence(len(contributions)),
            0.0,
            1.0,
        )
        explanation = (
            f"Weighted evidence score {risk_score:.3f} from {len(contributions)} signals. "
            f"Dominant evidence: {', '.join(dominant) if dominant else 'none'}. "
            "This score is deterministic evidence aggregation, not incident probability."
        )
        risk_signals.append(
            RiskSignal(
                timestamp=pd.Timestamp(bucket_timestamp),
                entity_id=str(entity_id),
                risk_score=_clamp(risk_score, 0.0, 1.0),
                severity=severity,
                evidence_count=len(contributions),
                dominant_evidence_types=dominant,
                confidence=confidence,
                model_sources=tuple(sources),
                explanation=explanation,
            )
        )
    return risk_signals


def run_evidence_engine_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: EvidenceScoringConfig | None = None,
) -> EvidenceEngineExperimentResult:
    """Run the Phase 8B evidence engine on measured Phase 7B/8 artifacts."""

    cfg = config or EvidenceScoringConfig()
    project = project_root() if root is None else root
    resolved_run_id = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    base_output = output_root or project / "experiments" / "evidence"
    run_dir = base_output / resolved_run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "config.json", _config_to_json(cfg))

    started = time.perf_counter()
    forecasting_run = _resolve_project_path(project, cfg.forecasting_run_dir)
    forecasts_path = forecasting_run / "forecasts.parquet"
    normal_ranges_path = forecasting_run / "normal_ranges.csv"
    if not forecasts_path.exists():
        raise FileNotFoundError(f"Phase 8 forecasts not found: {forecasts_path}")
    if not normal_ranges_path.exists():
        raise FileNotFoundError(f"Phase 8 normal ranges not found: {normal_ranges_path}")
    forecasts = pd.read_parquet(forecasts_path)
    normal_ranges = pd.read_csv(normal_ranges_path)
    selected_forecasts = _selected_forecasts(forecasts, cfg)
    if selected_forecasts.empty:
        raise ValueError(
            "No Phase 8 forecasts matched the configured model, horizon, and split: "
            f"{cfg.selected_forecast_model}, {cfg.selected_forecast_horizon_minutes}, "
            f"{cfg.selected_split}"
        )

    start = pd.Timestamp(selected_forecasts["target_timestamp"].min()).floor(
        cfg.aggregation_frequency
    )
    end = pd.Timestamp(selected_forecasts["target_timestamp"].max()).ceil(cfg.aggregation_frequency)
    signal_groups: dict[str, list[EvidenceSignal]] = {}
    signal_groups[EvidenceSignalType.FORECAST_DEVIATION.value] = build_forecast_deviation_signals(
        selected_forecasts,
        normal_ranges,
        cfg,
    )
    signal_groups[EvidenceSignalType.CURRENT_ANOMALY.value] = build_current_anomaly_signals(
        selected_forecasts,
        normal_ranges,
        cfg,
    )
    signal_groups[EvidenceSignalType.TREND.value] = build_trend_signals(
        selected_forecasts,
        normal_ranges,
        cfg,
    )

    try:
        raw_metropt = _load_raw_metropt(project)
    except FileNotFoundError:
        raw_metropt = pd.DataFrame()
    signal_groups[EvidenceSignalType.STATE_TRANSITION.value] = (
        build_state_transition_signals(raw_metropt, cfg, start=start, end=end)
        if not raw_metropt.empty
        else []
    )

    failure_predictions = _load_failure_predictions(project, cfg)
    signal_groups[EvidenceSignalType.FAILURE_SIGNAL.value] = build_research_failure_signals(
        failure_predictions,
        cfg,
    )

    base_signals = [signal for group in signal_groups.values() for signal in group]
    signal_groups[EvidenceSignalType.HISTORICAL_CONTEXT.value] = build_historical_context_signals(
        base_signals,
        cfg,
    )
    all_signals = [signal for group in signal_groups.values() for signal in group]
    risk_signals = aggregate_evidence_signals(all_signals, cfg)
    case_studies = select_case_studies(all_signals, risk_signals, cfg)
    leakage_audit = forecast_residual_leakage_audit(selected_forecasts, signal_groups)

    evidence_frame = pd.DataFrame([signal.to_record() for signal in all_signals])
    risk_frame = pd.DataFrame([risk_signal.to_record() for risk_signal in risk_signals])
    evidence_frame.to_csv(run_dir / "evidence_signals.csv", index=False)
    risk_frame.to_csv(run_dir / "risk_signals.csv", index=False)
    if not evidence_frame.empty:
        evidence_frame.to_parquet(run_dir / "evidence_signals.parquet", index=False)
    if not risk_frame.empty:
        risk_frame.to_parquet(run_dir / "risk_signals.parquet", index=False)
    timeline = build_evidence_timeline(evidence_frame, risk_frame, cfg.aggregation_frequency)
    timeline.to_csv(run_dir / "evidence_timeline.csv", index=False)
    per_metric = summarize_per_metric_evidence(evidence_frame)
    per_metric.to_csv(run_dir / "per_metric_evidence.csv", index=False)
    reliability = build_reliability_table(cfg)
    reliability.to_csv(run_dir / "model_reliability.csv", index=False)
    _save_json(run_dir / "case_studies.json", case_studies)

    _plot_risk_timeline(risk_frame, figures_dir / "risk_score_timeline.png")
    _plot_evidence_timeline(evidence_frame, figures_dir / "evidence_timeline.png")
    _plot_type_counts(evidence_frame, figures_dir / "top_evidence_types.png")
    _plot_per_metric(per_metric, figures_dir / "per_metric_evidence.png")
    _plot_forecast_deviation_vs_risk(
        evidence_frame,
        risk_frame,
        cfg,
        figures_dir / "forecast_deviation_vs_risk.png",
    )
    _plot_anomaly_score_vs_risk(
        evidence_frame,
        risk_frame,
        cfg,
        figures_dir / "current_anomaly_vs_risk.png",
    )

    metrics = {
        "run_id": resolved_run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "input_artifacts": {
            "forecasting_run_dir": str(forecasting_run),
            "forecasts": str(forecasts_path),
            "normal_ranges": str(normal_ranges_path),
            "robustness_prediction_path": str(
                _resolve_project_path(project, cfg.robustness_prediction_path)
            )
            if cfg.robustness_prediction_path is not None
            else None,
        },
        "selected_forecast_model": cfg.selected_forecast_model,
        "selected_forecast_horizon_minutes": cfg.selected_forecast_horizon_minutes,
        "selected_split": cfg.selected_split,
        "risk_score_definition": (
            "risk_score = 1 - product(1 - normalized_value * signal_type_weight * "
            "reliability_weight) across evidence in each time bucket. It is not a "
            "probability of incident, failure, or causation."
        ),
        "evidence_signal_count": int(len(all_signals)),
        "risk_signal_count": int(len(risk_signals)),
        "evidence_counts_by_type": _count_by_evidence_type(all_signals),
        "risk_severity_counts": risk_frame["severity"].value_counts().to_dict()
        if "severity" in risk_frame
        else {},
        "mean_risk_score": float(risk_frame["risk_score"].mean())
        if "risk_score" in risk_frame
        else 0.0,
        "max_risk_score": float(risk_frame["risk_score"].max())
        if "risk_score" in risk_frame
        else 0.0,
        "mean_confidence": float(risk_frame["confidence"].mean())
        if "confidence" in risk_frame
        else 0.0,
        "leakage_audit": leakage_audit,
        "reliability_policy": {
            "weights": cfg.reliability_weights,
            "source_reliability": cfg.source_reliability,
            "metropt_failure_prediction": (
                "RESEARCH_ONLY and capped by reliability and signal-type weights."
            ),
        },
        "case_studies": case_studies,
        "limitations": [
            "Risk score is an evidence aggregation score, not a calibrated incident probability.",
            "Current anomaly evidence uses empirical train-only ranges, not causal labels.",
            "Forecast residual evidence is only available at the target timestamp.",
            "MetroPT failure-prediction signals remain RESEARCH_ONLY and cannot dominate.",
            "State transitions and historical context are descriptive support signals.",
        ],
    }
    _save_json(run_dir / "metrics.json", metrics)
    _write_run_report(run_dir, metrics)
    return EvidenceEngineExperimentResult(run_dir=run_dir, metrics=metrics)


def severity_from_value(
    value: float,
    thresholds: dict[str, float],
) -> EvidenceSeverity:
    """Map a scalar value to an ordinal severity using configured thresholds."""

    if value >= thresholds["critical"]:
        return EvidenceSeverity.CRITICAL
    if value >= thresholds["high"]:
        return EvidenceSeverity.HIGH
    if value >= thresholds["medium"]:
        return EvidenceSeverity.MEDIUM
    if value >= thresholds["low"]:
        return EvidenceSeverity.LOW
    return EvidenceSeverity.INFO


def resolve_reliability(
    source_model: str,
    source_dataset: str,
    config: EvidenceScoringConfig,
) -> ReliabilityLevel:
    """Resolve the reliability tier for a source model and dataset."""

    direct_key = f"{source_dataset}:{source_model}"
    value = config.source_reliability.get(direct_key)
    if value is None and source_model.startswith("metropt_failure_prediction"):
        value = config.source_reliability.get(f"{source_dataset}:metropt_failure_prediction")
    if value is None:
        value = ReliabilityLevel.LOW.value
    return ReliabilityLevel(value)


def confidence_from_reliability(
    reliability: ReliabilityLevel,
    normalized_value: float,
    config: EvidenceScoringConfig,
) -> float:
    """Convert source reliability and signal strength into a bounded confidence score."""

    weight = config.reliability_weights[reliability.value]
    return _clamp(weight * (0.5 + 0.5 * normalized_value), 0.0, 1.0)


def forecast_residual_leakage_audit(
    selected_forecasts: pd.DataFrame,
    signal_groups: dict[str, list[EvidenceSignal]],
) -> dict[str, Any]:
    """Audit that residual evidence is not timestamped before future actuals exist."""

    if selected_forecasts.empty:
        return {"leakage_detected": False, "reason": "no selected forecasts"}
    selected = selected_forecasts.copy()
    selected["input_end_timestamp"] = pd.to_datetime(
        selected["input_end_timestamp"],
        errors="raise",
    )
    selected["target_timestamp"] = pd.to_datetime(selected["target_timestamp"], errors="raise")
    alignment_errors = int(selected["target_timestamp"].lt(selected["input_end_timestamp"]).sum())
    residual_signals = signal_groups.get(EvidenceSignalType.FORECAST_DEVIATION.value, [])
    input_end_min_by_target = selected.groupby("target_timestamp", observed=True)[
        "input_end_timestamp"
    ].min()
    residual_before_target = 0
    for signal in residual_signals:
        if signal.timestamp not in input_end_min_by_target.index:
            continue
        input_end = pd.Timestamp(input_end_min_by_target.loc[signal.timestamp])
        if signal.timestamp < input_end:
            residual_before_target += 1
    return {
        "leakage_detected": bool(alignment_errors or residual_before_target),
        "forecast_rows_checked": int(selected.shape[0]),
        "forecast_alignment_errors": alignment_errors,
        "forecast_residual_signals_checked": int(len(residual_signals)),
        "residual_signals_timestamped_before_input_end": residual_before_target,
        "policy": (
            "Forecast residual evidence uses target_timestamp because it requires observed "
            "future actuals. Forecast input timestamps are retained only for audit."
        ),
    }


def build_evidence_timeline(
    evidence: pd.DataFrame,
    risk: pd.DataFrame,
    aggregation_frequency: str,
) -> pd.DataFrame:
    """Build a compact joined timeline for inspection and plotting."""

    if evidence.empty or risk.empty:
        return pd.DataFrame()
    evidence = evidence.copy()
    risk = risk.copy()
    evidence["timestamp"] = pd.to_datetime(evidence["timestamp"], errors="raise")
    evidence["timestamp"] = evidence["timestamp"].dt.floor(aggregation_frequency)
    risk["timestamp"] = pd.to_datetime(risk["timestamp"], errors="raise")
    counts = (
        evidence.groupby(["timestamp", "signal_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    return risk.merge(counts, on="timestamp", how="left").fillna(0)


def summarize_per_metric_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    """Summarize evidence counts and mean strength by metric."""

    if evidence.empty:
        return pd.DataFrame(
            columns=[
                "metric_name",
                "signal_count",
                "mean_normalized_value",
                "max_normalized_value",
                "dominant_signal_type",
            ]
        )
    grouped = evidence.groupby("metric_name", observed=True)
    rows: list[dict[str, Any]] = []
    for metric_name, group in grouped:
        type_counts = group["signal_type"].value_counts()
        rows.append(
            {
                "metric_name": str(metric_name),
                "signal_count": int(group.shape[0]),
                "mean_normalized_value": float(group["normalized_value"].mean()),
                "max_normalized_value": float(group["normalized_value"].max()),
                "dominant_signal_type": str(type_counts.index[0]) if not type_counts.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["signal_count", "max_normalized_value"],
        ascending=False,
    )


def build_reliability_table(config: EvidenceScoringConfig) -> pd.DataFrame:
    """Return the configured reliability policy as a tabular artifact."""

    rows = []
    for source_key, reliability in sorted(config.source_reliability.items()):
        rows.append(
            {
                "source": source_key,
                "reliability": reliability,
                "weight": config.reliability_weights[reliability],
                "rule": _reliability_rule_text(ReliabilityLevel(reliability)),
            }
        )
    return pd.DataFrame(rows)


def select_case_studies(
    evidence_signals: Sequence[EvidenceSignal],
    risk_signals: Sequence[RiskSignal],
    config: EvidenceScoringConfig,
) -> dict[str, Any]:
    """Select normal, anomalous, difficult, and pre-failure case studies."""

    if not risk_signals:
        return {}
    risk_frame = pd.DataFrame([risk.to_record() for risk in risk_signals])
    risk_frame["timestamp"] = pd.to_datetime(risk_frame["timestamp"], errors="raise")
    risk_frame = risk_frame.sort_values("timestamp").reset_index(drop=True)
    evidence_frame = pd.DataFrame([signal.to_record() for signal in evidence_signals])
    evidence_frame["timestamp"] = pd.to_datetime(evidence_frame["timestamp"], errors="raise")
    evidence_frame["bucket_timestamp"] = evidence_frame["timestamp"].dt.floor(
        config.aggregation_frequency
    )
    cases: dict[str, Any] = {}
    normal_row = risk_frame.sort_values(["risk_score", "evidence_count"]).iloc[0]
    cases["normal"] = _case_payload(
        normal_row,
        evidence_frame,
        "Lowest aggregate evidence bucket in the measured timeline.",
    )
    anomalous_row = risk_frame.sort_values(["risk_score", "evidence_count"], ascending=False).iloc[
        0
    ]
    cases["anomalous"] = _case_payload(
        anomalous_row,
        evidence_frame,
        "Highest aggregate evidence bucket in the measured timeline.",
    )
    forecast_evidence = evidence_frame[
        evidence_frame["signal_type"].eq(EvidenceSignalType.FORECAST_DEVIATION.value)
    ]
    if not forecast_evidence.empty:
        difficult_timestamp = forecast_evidence.sort_values(
            "normalized_value",
            ascending=False,
        ).iloc[0]["bucket_timestamp"]
        difficult_candidates = risk_frame[risk_frame["timestamp"].eq(difficult_timestamp)]
        if not difficult_candidates.empty:
            cases["difficult"] = _case_payload(
                difficult_candidates.iloc[0],
                evidence_frame,
                "Largest normalized forecast-residual evidence bucket.",
            )
    event = MetroPTExperimentConfig().failure_events[-1]
    pre_failure_start = event.start - pd.Timedelta(hours=config.pre_failure_context_hours)
    pre_failure_end = event.start
    pre_failure = risk_frame[
        risk_frame["timestamp"].ge(pre_failure_start) & risk_frame["timestamp"].le(pre_failure_end)
    ]
    if pre_failure.empty:
        prior = risk_frame[risk_frame["timestamp"].le(event.start)]
        if not prior.empty:
            selected = prior.sort_values("timestamp").iloc[-1]
            cases["pre_failure"] = _case_payload(
                selected,
                evidence_frame,
                (
                    "Closest available bucket before the July MetroPT failure event; no sampled "
                    "Phase 8 forecast bucket fell inside the configured pre-failure window."
                ),
            )
        else:
            cases["pre_failure"] = {
                "selection_rule": "No risk bucket exists before the July failure event.",
                "event_id": event.event_id,
                "event_start": event.start.isoformat(),
                "available": False,
            }
    else:
        selected = pre_failure.sort_values(["risk_score", "evidence_count"], ascending=False).iloc[
            0
        ]
        cases["pre_failure"] = _case_payload(
            selected,
            evidence_frame,
            (
                "Highest aggregate evidence bucket within the configured pre-failure window for "
                f"{event.event_id}."
            ),
        )
    return cases


def _make_signal(
    *,
    timestamp: pd.Timestamp,
    entity_id: str,
    signal_type: EvidenceSignalType,
    metric_name: str,
    value: float,
    normalized_value: float,
    severity: EvidenceSeverity,
    reliability: ReliabilityLevel,
    source_model: str,
    source_dataset: str,
    evidence_description: str,
    validity: EvidenceValidity,
    limitations: str,
    config: EvidenceScoringConfig,
) -> EvidenceSignal:
    normalized_timestamp = pd.Timestamp(timestamp)
    signal_id = _signal_id(
        normalized_timestamp,
        entity_id,
        signal_type,
        metric_name,
        source_model,
        value,
    )
    return EvidenceSignal(
        signal_id=signal_id,
        timestamp=normalized_timestamp,
        entity_id=entity_id,
        signal_type=signal_type,
        metric_name=metric_name,
        value=value,
        normalized_value=_clamp(normalized_value, 0.0, 1.0),
        severity=severity,
        confidence=confidence_from_reliability(reliability, normalized_value, config),
        source_model=source_model,
        source_dataset=source_dataset,
        evidence_description=evidence_description,
        validity=validity,
        limitations=limitations,
    )


def _selected_forecasts(forecasts: pd.DataFrame, config: EvidenceScoringConfig) -> pd.DataFrame:
    selected = forecasts[
        forecasts["model"].astype(str).eq(config.selected_forecast_model)
        & forecasts["horizon_minutes"].astype(int).eq(config.selected_forecast_horizon_minutes)
        & forecasts["split"].astype(str).eq(config.selected_split)
    ].copy()
    selected["input_end_timestamp"] = pd.to_datetime(
        selected["input_end_timestamp"],
        errors="raise",
    )
    selected["target_timestamp"] = pd.to_datetime(selected["target_timestamp"], errors="raise")
    return selected.sort_values(["target_timestamp", "sensor"]).reset_index(drop=True)


def _load_failure_predictions(project: Path, config: EvidenceScoringConfig) -> pd.DataFrame:
    if config.robustness_prediction_path is None:
        return pd.DataFrame()
    path = _resolve_project_path(project, config.robustness_prediction_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(
            path,
            filters=[
                ("split", "==", config.selected_split),
                ("candidate_id", "==", "h6p0__random_forest"),
            ],
        )
    except Exception:
        predictions = pd.read_parquet(path)
        return predictions[
            predictions["split"].astype(str).eq(config.selected_split)
            & predictions["candidate_id"].astype(str).eq("h6p0__random_forest")
        ].copy()


def _load_raw_metropt(project: Path) -> pd.DataFrame:
    from aegis_ai.ml.prediction.failure_prediction import load_metropt_frame

    return load_metropt_frame(project)


def _normal_range_lookup(normal_ranges: pd.DataFrame) -> dict[str, dict[str, float]]:
    _require_columns(normal_ranges, {"sensor", "mean", "std", "low_quantile", "high_quantile"})
    lookup: dict[str, dict[str, float]] = {}
    for row in normal_ranges.itertuples(index=False):
        lookup[str(row.sensor)] = {
            "mean": float(row.mean),
            "std": max(float(row.std), 1e-8),
            "low_quantile": float(row.low_quantile),
            "high_quantile": float(row.high_quantile),
        }
    return lookup


def _case_payload(
    risk_row: pd.Series[Any],
    evidence_frame: pd.DataFrame,
    selection_rule: str,
) -> dict[str, Any]:
    timestamp = pd.Timestamp(risk_row["timestamp"])
    bucket_evidence = evidence_frame[evidence_frame["bucket_timestamp"].eq(timestamp)].copy()
    top = bucket_evidence.sort_values(
        ["normalized_value", "confidence"],
        ascending=False,
    ).head(8)
    return {
        "available": True,
        "selection_rule": selection_rule,
        "timestamp": timestamp.isoformat(),
        "risk_score": float(risk_row["risk_score"]),
        "severity": str(risk_row["severity"]),
        "confidence": float(risk_row["confidence"]),
        "evidence_count": int(risk_row["evidence_count"]),
        "dominant_evidence_types": str(risk_row["dominant_evidence_types"]).split("|")
        if str(risk_row["dominant_evidence_types"])
        else [],
        "top_evidence": top[
            [
                "signal_type",
                "metric_name",
                "value",
                "normalized_value",
                "severity",
                "source_model",
                "validity",
                "evidence_description",
            ]
        ].to_dict(orient="records"),
    }


def _signal_id(
    timestamp: pd.Timestamp,
    entity_id: str,
    signal_type: EvidenceSignalType,
    metric_name: str,
    source_model: str,
    value: float,
) -> str:
    payload = "|".join(
        [
            timestamp.isoformat(),
            entity_id,
            signal_type.value,
            metric_name,
            source_model,
            f"{value:.12g}",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _limit_dataframe(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.shape[0] <= limit:
        return frame
    indices = np.linspace(0, frame.shape[0] - 1, num=limit, dtype=int)
    return frame.iloc[indices].copy()


def _limit_signals(signals: list[EvidenceSignal], limit: int) -> list[EvidenceSignal]:
    if len(signals) <= limit:
        return signals
    ordered = sorted(
        signals,
        key=lambda signal: (signal.normalized_value, signal.confidence, signal.timestamp),
        reverse=True,
    )
    return sorted(ordered[:limit], key=lambda signal: signal.timestamp)


def _count_by_evidence_type(signals: Sequence[EvidenceSignal]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signal in signals:
        counts[signal.signal_type.value] = counts.get(signal.signal_type.value, 0) + 1
    return counts


def _abs_z(value: float, stats: dict[str, float]) -> float:
    return abs((value - stats["mean"]) / max(stats["std"], 1e-8))


def _normalize(value: float, critical_threshold: float) -> float:
    return _clamp(value / max(critical_threshold, 1e-8), 0.0, 1.0)


def _evidence_count_confidence(count: int) -> float:
    return _clamp(math.log1p(count) / math.log1p(5.0), 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _resolve_project_path(project: Path, path: Path | None) -> Path:
    if path is None:
        raise ValueError("path must not be None")
    return path if path.is_absolute() else project / path


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"data frame is missing required columns: {missing}")


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
    if any(value < 0 for value in values):
        raise ValueError(f"{name} thresholds must be non-negative")
    if values != sorted(values):
        raise ValueError(f"{name} thresholds must be ordered low <= medium <= high <= critical")
    if thresholds["critical"] <= 0:
        raise ValueError(f"{name} critical threshold must be positive")


def _validate_weights(name: str, weights: dict[str, float]) -> None:
    for key, value in weights.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}.{key} must be in [0, 1]")


def _config_to_json(config: EvidenceScoringConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("forecasting_run_dir", "robustness_prediction_path"):
        value = payload[key]
        payload[key] = None if value is None else str(value)
    return payload


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
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
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _reliability_rule_text(reliability: ReliabilityLevel) -> str:
    if reliability is ReliabilityLevel.HIGH:
        return "Validated on matching data with stable metrics and production-quality monitoring."
    if reliability is ReliabilityLevel.MEDIUM:
        return "Measured on matching data, but not calibrated enough to act alone."
    if reliability is ReliabilityLevel.LOW:
        return "Rule-based or descriptive signal that supports investigation context."
    return "Research-only signal with known target or validation limitations."


def _plot_risk_timeline(risk: pd.DataFrame, output_path: Path) -> None:
    if risk.empty:
        return
    frame = risk.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame["timestamp"], frame["risk_score"], color="#2563eb", linewidth=1.5)
    ax.set_title("Aggregated Evidence Risk Score")
    ax.set_ylabel("risk score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # type: ignore[no-untyped-call]
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_evidence_timeline(evidence: pd.DataFrame, output_path: Path) -> None:
    if evidence.empty:
        return
    frame = evidence.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    counts = (
        frame.groupby([pd.Grouper(key="timestamp", freq="1h"), "signal_type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    counts.plot(ax=ax, linewidth=1.1)
    ax.set_title("Evidence Signal Timeline")
    ax.set_ylabel("signals per hour")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # type: ignore[no-untyped-call]
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_type_counts(evidence: pd.DataFrame, output_path: Path) -> None:
    if evidence.empty:
        return
    counts = evidence["signal_type"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="barh", ax=ax, color="#0f766e")
    ax.set_title("Evidence Signal Counts")
    ax.set_xlabel("signals")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_per_metric(per_metric: pd.DataFrame, output_path: Path) -> None:
    if per_metric.empty:
        return
    frame = per_metric.head(12).sort_values("signal_count")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(frame["metric_name"], frame["signal_count"], color="#7c3aed")
    ax.set_title("Evidence by Metric")
    ax.set_xlabel("signals")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_forecast_deviation_vs_risk(
    evidence: pd.DataFrame,
    risk: pd.DataFrame,
    config: EvidenceScoringConfig,
    output_path: Path,
) -> None:
    _plot_signal_strength_vs_risk(
        evidence,
        risk,
        config,
        EvidenceSignalType.FORECAST_DEVIATION,
        "Forecast Deviation Evidence vs Risk",
        output_path,
    )


def _plot_anomaly_score_vs_risk(
    evidence: pd.DataFrame,
    risk: pd.DataFrame,
    config: EvidenceScoringConfig,
    output_path: Path,
) -> None:
    _plot_signal_strength_vs_risk(
        evidence,
        risk,
        config,
        EvidenceSignalType.CURRENT_ANOMALY,
        "Current Anomaly Evidence vs Risk",
        output_path,
    )


def _plot_signal_strength_vs_risk(
    evidence: pd.DataFrame,
    risk: pd.DataFrame,
    config: EvidenceScoringConfig,
    signal_type: EvidenceSignalType,
    title: str,
    output_path: Path,
) -> None:
    if evidence.empty or risk.empty:
        return
    evidence_frame = evidence[evidence["signal_type"].eq(signal_type.value)].copy()
    if evidence_frame.empty:
        return
    evidence_frame["timestamp"] = pd.to_datetime(evidence_frame["timestamp"], errors="raise")
    evidence_frame["bucket_timestamp"] = evidence_frame["timestamp"].dt.floor(
        config.aggregation_frequency
    )
    strength = (
        evidence_frame.groupby("bucket_timestamp", observed=True)["normalized_value"]
        .max()
        .reset_index()
        .rename(columns={"bucket_timestamp": "timestamp", "normalized_value": "signal_strength"})
    )
    risk_frame = risk.copy()
    risk_frame["timestamp"] = pd.to_datetime(risk_frame["timestamp"], errors="raise")
    joined = strength.merge(risk_frame[["timestamp", "risk_score"]], on="timestamp", how="inner")
    if joined.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(joined["signal_strength"], joined["risk_score"], alpha=0.45, s=16, color="#dc2626")
    ax.set_title(title)
    ax.set_xlabel("max normalized signal")
    ax.set_ylabel("risk score")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_run_report(run_dir: Path, metrics: dict[str, Any]) -> None:
    case_names = ", ".join(sorted(metrics["case_studies"].keys()))
    lines = [
        "# Phase 8B Evidence Engine Run",
        "",
        f"- Run ID: `{metrics['run_id']}`",
        f"- Evidence signals: {metrics['evidence_signal_count']}",
        f"- Risk buckets: {metrics['risk_signal_count']}",
        f"- Mean risk score: {metrics['mean_risk_score']:.4f}",
        f"- Max risk score: {metrics['max_risk_score']:.4f}",
        f"- Leakage detected: {metrics['leakage_audit']['leakage_detected']}",
        f"- Case studies: {case_names}",
        "",
        "## Risk Score",
        "",
        metrics["risk_score_definition"],
        "",
        "## Evidence Counts",
        "",
    ]
    for signal_type, count in sorted(metrics["evidence_counts_by_type"].items()):
        lines.append(f"- {signal_type}: {count}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in metrics["limitations"]],
            "",
        ]
    )
    run_dir.joinpath("run_report.md").write_text("\n".join(lines), encoding="utf-8")
