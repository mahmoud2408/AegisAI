"""Deterministic incident grouping and RCA candidate ranking."""

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

from aegis_ai.data.dataset_registry import project_root
from aegis_ai.evidence import EvidenceSeverity
from aegis_ai.ml.prediction.failure_prediction import MetroPTExperimentConfig

SEVERITY_RANK: dict[str, int] = {
    EvidenceSeverity.INFO.value: 0,
    EvidenceSeverity.LOW.value: 1,
    EvidenceSeverity.MEDIUM.value: 2,
    EvidenceSeverity.HIGH.value: 3,
    EvidenceSeverity.CRITICAL.value: 4,
}


@dataclass(frozen=True)
class RCAScoringConfig:
    """Configuration for deterministic incident grouping and candidate scoring."""

    evidence_run_dir: Path = Path("experiments/evidence/phase8b_evidence_engine_20260916")
    entity_id: str = "metropt_compressor"
    aggregation_frequency: str = "5min"
    min_risk_threshold: float = 0.80
    min_bucket_evidence_count: int = 6
    min_window_evidence_count: int = 8
    min_persistence_minutes: float = 5.0
    critical_single_bucket_threshold: float = 0.80
    max_gap_minutes: float = 15.0
    merge_gap_minutes: float = 20.0
    merge_pattern_jaccard_threshold: float = 0.25
    persistence_cap_minutes: float = 60.0
    temporal_precedence_reference: str = "peak_risk"
    signal_count_cap: int = 12
    multi_signal_type_cap: int = 4
    top_candidates_per_incident: int = 8
    max_evidence_per_candidate: int = 12
    case_study_pre_failure_hours: float = 6.0
    candidate_feature_weights: dict[str, float] = field(
        default_factory=lambda: {
            "magnitude": 0.24,
            "persistence": 0.16,
            "signal_count": 0.12,
            "multi_signal_support": 0.14,
            "forecast_deviation": 0.10,
            "trend_strength": 0.08,
            "state_transition": 0.04,
            "temporal_precedence": 0.06,
            "historical_recurrence": 0.04,
            "confidence": 0.02,
        }
    )
    signal_type_role_weights: dict[str, float] = field(
        default_factory=lambda: {
            "CURRENT_ANOMALY": 1.00,
            "FORECAST_DEVIATION": 0.95,
            "TREND": 0.75,
            "HISTORICAL_CONTEXT": 0.55,
            "STATE_TRANSITION": 0.45,
            "FAILURE_SIGNAL": 0.20,
        }
    )
    candidate_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "metric": 1.00,
            "component": 0.95,
            "entity": 0.75,
        }
    )
    metric_component_map: dict[str, str] = field(
        default_factory=lambda: {
            "TP2": "compressor_pressure_path",
            "TP3": "compressor_pressure_path",
            "H1": "compressor_pressure_path",
            "Reservoirs": "air_reservoir_system",
            "Oil_temperature": "lubrication_thermal_system",
            "Motor_current": "motor_electrical_system",
            "COMP": "operating_state_control",
            "DV_eletric": "operating_state_control",
            "Towers": "operating_state_control",
            "MPG": "operating_state_control",
            "LPS": "operating_state_control",
            "Pressure_switch": "operating_state_control",
            "Oil_level": "lubrication_thermal_system",
            "Caudal_impulses": "air_flow_signal",
            "failure_prediction_probability": "research_failure_signal",
        }
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_risk_threshold <= 1.0:
            raise ValueError("min_risk_threshold must be in [0, 1]")
        if self.min_bucket_evidence_count < 1:
            raise ValueError("min_bucket_evidence_count must be positive")
        if self.min_window_evidence_count < 1:
            raise ValueError("min_window_evidence_count must be positive")
        if self.min_persistence_minutes <= 0:
            raise ValueError("min_persistence_minutes must be positive")
        if not 0.0 <= self.critical_single_bucket_threshold <= 1.0:
            raise ValueError("critical_single_bucket_threshold must be in [0, 1]")
        if self.max_gap_minutes < 0:
            raise ValueError("max_gap_minutes must be non-negative")
        if self.merge_gap_minutes < 0:
            raise ValueError("merge_gap_minutes must be non-negative")
        if not 0.0 <= self.merge_pattern_jaccard_threshold <= 1.0:
            raise ValueError("merge_pattern_jaccard_threshold must be in [0, 1]")
        if self.persistence_cap_minutes <= 0:
            raise ValueError("persistence_cap_minutes must be positive")
        if self.signal_count_cap < 1:
            raise ValueError("signal_count_cap must be positive")
        if self.multi_signal_type_cap < 1:
            raise ValueError("multi_signal_type_cap must be positive")
        if self.top_candidates_per_incident < 1:
            raise ValueError("top_candidates_per_incident must be positive")
        if self.max_evidence_per_candidate < 1:
            raise ValueError("max_evidence_per_candidate must be positive")
        if self.case_study_pre_failure_hours <= 0:
            raise ValueError("case_study_pre_failure_hours must be positive")
        _validate_non_negative_weights("candidate_feature_weights", self.candidate_feature_weights)
        _validate_non_negative_weights("signal_type_role_weights", self.signal_type_role_weights)
        _validate_non_negative_weights("candidate_type_weights", self.candidate_type_weights)


@dataclass(frozen=True)
class IncidentWindow:
    """Deduplicated time window of related evidence and risk buckets."""

    incident_id: str
    entity_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    severity: EvidenceSeverity
    risk_score: float
    mean_risk_score: float
    evidence_count: int
    risk_bucket_count: int
    persistence_minutes: float
    metric_names: tuple[str, ...]
    evidence_types: tuple[str, ...]
    signal_ids: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """Return a flat artifact row."""

        return {
            "incident_id": self.incident_id,
            "entity_id": self.entity_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "severity": self.severity.value,
            "risk_score": self.risk_score,
            "mean_risk_score": self.mean_risk_score,
            "evidence_count": self.evidence_count,
            "risk_bucket_count": self.risk_bucket_count,
            "persistence_minutes": self.persistence_minutes,
            "metric_names": "|".join(self.metric_names),
            "evidence_types": "|".join(self.evidence_types),
            "signal_ids": "|".join(self.signal_ids),
        }


@dataclass(frozen=True)
class RootCauseCandidate:
    """Evidence-based root-cause candidate, not a causal proof."""

    candidate_id: str
    incident_id: str
    entity: str
    metric_component: str
    candidate_type: str
    score: float
    rank: int
    supporting_signal_ids: tuple[str, ...]
    observed_evidence: tuple[str, ...]
    inference: str
    uncertainty: str
    limitations: str
    data_sources: tuple[str, ...]
    source_models: tuple[str, ...]
    first_evidence_time: pd.Timestamp
    last_evidence_time: pd.Timestamp
    persistence_minutes: float
    feature_scores: dict[str, float]

    def to_record(self) -> dict[str, Any]:
        """Return a flat artifact row."""

        return {
            "candidate_id": self.candidate_id,
            "incident_id": self.incident_id,
            "entity": self.entity,
            "metric_component": self.metric_component,
            "candidate_type": self.candidate_type,
            "score": self.score,
            "rank": self.rank,
            "supporting_signal_ids": "|".join(self.supporting_signal_ids),
            "observed_evidence": json.dumps(list(self.observed_evidence)),
            "inference": self.inference,
            "uncertainty": self.uncertainty,
            "limitations": self.limitations,
            "data_sources": "|".join(self.data_sources),
            "source_models": "|".join(self.source_models),
            "first_evidence_time": self.first_evidence_time.isoformat(),
            "last_evidence_time": self.last_evidence_time.isoformat(),
            "persistence_minutes": self.persistence_minutes,
            "feature_scores": json.dumps(self.feature_scores, sort_keys=True),
        }


@dataclass(frozen=True)
class IncidentCandidate:
    """Structured incident report assembled from a window and ranked candidates."""

    incident_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    severity: EvidenceSeverity
    risk_score: float
    affected_entities: tuple[str, ...]
    affected_services: tuple[str, ...]
    top_metric_signals: tuple[dict[str, Any], ...]
    root_cause_candidates: tuple[RootCauseCandidate, ...]
    supporting_evidence: tuple[dict[str, Any], ...]
    uncertainty: str
    data_sources: tuple[str, ...]
    timeline: tuple[dict[str, Any], ...]
    summary: str

    def to_record(self) -> dict[str, Any]:
        """Return a JSON friendly representation."""

        return {
            "incident_id": self.incident_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "severity": self.severity.value,
            "risk_score": self.risk_score,
            "affected_entities": list(self.affected_entities),
            "affected_services": list(self.affected_services),
            "top_metric_signals": list(self.top_metric_signals),
            "root_cause_candidates": [
                candidate.to_record() for candidate in self.root_cause_candidates
            ],
            "supporting_evidence": list(self.supporting_evidence),
            "uncertainty": self.uncertainty,
            "data_sources": list(self.data_sources),
            "timeline": list(self.timeline),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class IncidentWindowDetectionResult:
    """Preliminary and merged incident-window outputs."""

    preliminary_windows: tuple[IncidentWindow, ...]
    merged_windows: tuple[IncidentWindow, ...]


@dataclass(frozen=True)
class RCAExperimentResult:
    """Location and summary for a completed Phase 9 RCA run."""

    run_dir: Path
    metrics: dict[str, Any]


def load_rca_scoring_config(path: Path) -> RCAScoringConfig:
    """Load RCA scoring configuration from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("RCA config must be a YAML mapping")
    default = RCAScoringConfig()
    return RCAScoringConfig(
        evidence_run_dir=Path(str(payload.get("evidence_run_dir", default.evidence_run_dir))),
        entity_id=str(payload.get("entity_id", default.entity_id)),
        aggregation_frequency=str(
            payload.get("aggregation_frequency", default.aggregation_frequency)
        ),
        min_risk_threshold=float(payload.get("min_risk_threshold", default.min_risk_threshold)),
        min_bucket_evidence_count=int(
            payload.get("min_bucket_evidence_count", default.min_bucket_evidence_count)
        ),
        min_window_evidence_count=int(
            payload.get("min_window_evidence_count", default.min_window_evidence_count)
        ),
        min_persistence_minutes=float(
            payload.get("min_persistence_minutes", default.min_persistence_minutes)
        ),
        critical_single_bucket_threshold=float(
            payload.get(
                "critical_single_bucket_threshold",
                default.critical_single_bucket_threshold,
            )
        ),
        max_gap_minutes=float(payload.get("max_gap_minutes", default.max_gap_minutes)),
        merge_gap_minutes=float(payload.get("merge_gap_minutes", default.merge_gap_minutes)),
        merge_pattern_jaccard_threshold=float(
            payload.get(
                "merge_pattern_jaccard_threshold",
                default.merge_pattern_jaccard_threshold,
            )
        ),
        persistence_cap_minutes=float(
            payload.get("persistence_cap_minutes", default.persistence_cap_minutes)
        ),
        temporal_precedence_reference=str(
            payload.get("temporal_precedence_reference", default.temporal_precedence_reference)
        ),
        signal_count_cap=int(payload.get("signal_count_cap", default.signal_count_cap)),
        multi_signal_type_cap=int(
            payload.get("multi_signal_type_cap", default.multi_signal_type_cap)
        ),
        top_candidates_per_incident=int(
            payload.get("top_candidates_per_incident", default.top_candidates_per_incident)
        ),
        max_evidence_per_candidate=int(
            payload.get("max_evidence_per_candidate", default.max_evidence_per_candidate)
        ),
        case_study_pre_failure_hours=float(
            payload.get("case_study_pre_failure_hours", default.case_study_pre_failure_hours)
        ),
        candidate_feature_weights=_float_mapping(
            payload.get("candidate_feature_weights"),
            default.candidate_feature_weights,
        ),
        signal_type_role_weights=_float_mapping(
            payload.get("signal_type_role_weights"),
            default.signal_type_role_weights,
        ),
        candidate_type_weights=_float_mapping(
            payload.get("candidate_type_weights"),
            default.candidate_type_weights,
        ),
        metric_component_map=_string_mapping(
            payload.get("metric_component_map"),
            default.metric_component_map,
        ),
    )


def detect_incident_windows(
    risk_signals: pd.DataFrame,
    evidence_signals: pd.DataFrame,
    config: RCAScoringConfig,
) -> IncidentWindowDetectionResult:
    """Group risk buckets into deduplicated incident windows."""

    _require_columns(
        risk_signals,
        {
            "timestamp",
            "entity_id",
            "risk_score",
            "severity",
            "evidence_count",
        },
    )
    _require_columns(
        evidence_signals,
        {
            "signal_id",
            "timestamp",
            "entity_id",
            "signal_type",
            "metric_name",
            "normalized_value",
            "severity",
            "source_model",
            "source_dataset",
        },
    )
    risk = _prepare_risk_frame(risk_signals)
    evidence = _prepare_evidence_frame(evidence_signals, config.aggregation_frequency)
    selected = risk[
        risk["risk_score"].ge(config.min_risk_threshold)
        & risk["evidence_count"].ge(config.min_bucket_evidence_count)
    ].sort_values(["entity_id", "timestamp"])
    preliminary: list[IncidentWindow] = []
    for _entity_id, group in selected.groupby("entity_id", sort=True, observed=True):
        current_rows: list[dict[str, Any]] = []
        previous: pd.Timestamp | None = None
        for row in group.to_dict(orient="records"):
            timestamp = pd.Timestamp(row["timestamp"])
            if (
                previous is not None
                and _minutes_between(previous, timestamp) > config.max_gap_minutes
            ):
                window = _window_from_risk_rows(current_rows, risk, evidence, config)
                if window is not None:
                    preliminary.append(window)
                current_rows = []
            current_rows.append(row)
            previous = timestamp
        window = _window_from_risk_rows(current_rows, risk, evidence, config)
        if window is not None:
            preliminary.append(window)
    merged = merge_incident_windows(preliminary, risk, evidence, config)
    return IncidentWindowDetectionResult(
        preliminary_windows=tuple(preliminary),
        merged_windows=tuple(merged),
    )


def merge_incident_windows(
    windows: Sequence[IncidentWindow],
    risk_signals: pd.DataFrame,
    evidence_signals: pd.DataFrame,
    config: RCAScoringConfig,
) -> list[IncidentWindow]:
    """Merge adjacent related windows using gap and evidence-pattern overlap."""

    if not windows:
        return []
    risk = _prepare_risk_frame(risk_signals)
    evidence = _prepare_evidence_frame(evidence_signals, config.aggregation_frequency)
    sorted_windows = sorted(windows, key=lambda item: (item.entity_id, item.start_time))
    merged: list[IncidentWindow] = [sorted_windows[0]]
    for window in sorted_windows[1:]:
        last = merged[-1]
        gap = _minutes_between(last.end_time, window.start_time)
        overlap = _pattern_jaccard(last, window)
        if (
            last.entity_id == window.entity_id
            and gap <= config.merge_gap_minutes
            and overlap >= config.merge_pattern_jaccard_threshold
        ):
            replacement = _window_from_time_range(
                entity_id=last.entity_id,
                start_time=min(last.start_time, window.start_time),
                end_time=max(last.end_time, window.end_time),
                risk_signals=risk,
                evidence_signals=evidence,
                config=config,
            )
            if replacement is not None:
                merged[-1] = replacement
        else:
            merged.append(window)
    return merged


def rank_root_cause_candidates(
    window: IncidentWindow,
    evidence_signals: pd.DataFrame,
    risk_signals: pd.DataFrame,
    config: RCAScoringConfig,
) -> list[RootCauseCandidate]:
    """Rank metric and component candidates for a single incident window."""

    evidence = _window_evidence(
        _prepare_evidence_frame(evidence_signals, config.aggregation_frequency),
        window.entity_id,
        window.start_time,
        window.end_time,
    )
    risk = _window_risk(
        _prepare_risk_frame(risk_signals),
        window.entity_id,
        window.start_time,
        window.end_time,
    )
    return _rank_root_cause_candidates_from_prepared(window, evidence, risk, config)


def _rank_root_cause_candidates_from_prepared(
    window: IncidentWindow,
    window_evidence: pd.DataFrame,
    window_risk: pd.DataFrame,
    config: RCAScoringConfig,
) -> list[RootCauseCandidate]:
    """Rank candidates from already window-filtered frames."""

    if window_evidence.empty:
        return []
    candidate_groups = _candidate_groups(window_evidence, config)
    scored = [
        _score_candidate(
            window=window,
            evidence=group,
            risk=window_risk,
            candidate_type=candidate_type,
            metric_component=metric_component,
            config=config,
        )
        for candidate_type, metric_component, group in candidate_groups
    ]
    scored = sorted(
        scored,
        key=lambda item: (
            item.score,
            len(item.supporting_signal_ids),
            item.persistence_minutes,
            item.metric_component,
        ),
        reverse=True,
    )
    ranked: list[RootCauseCandidate] = []
    for rank, candidate in enumerate(scored[: config.top_candidates_per_incident], start=1):
        ranked.append(
            RootCauseCandidate(
                **{
                    **asdict(candidate),
                    "rank": rank,
                }
            )
        )
    return ranked


def build_incident_reports(
    windows: Sequence[IncidentWindow],
    evidence_signals: pd.DataFrame,
    risk_signals: pd.DataFrame,
    config: RCAScoringConfig,
) -> list[IncidentCandidate]:
    """Build deterministic incident reports with separated evidence and inference."""

    evidence = _prepare_evidence_frame(evidence_signals, config.aggregation_frequency)
    risk = _prepare_risk_frame(risk_signals)
    reports: list[IncidentCandidate] = []
    for window in windows:
        window_evidence = _window_evidence(
            evidence,
            window.entity_id,
            window.start_time,
            window.end_time,
        )
        window_risk = _window_risk(risk, window.entity_id, window.start_time, window.end_time)
        candidates = _rank_root_cause_candidates_from_prepared(
            window,
            window_evidence,
            window_risk,
            config,
        )
        top_metrics = _top_metric_signals(window_evidence)
        supporting = _supporting_evidence_records(window_evidence, limit=20)
        data_sources = tuple(
            sorted(window_evidence["source_dataset"].astype(str).unique().tolist())
        )
        timeline = tuple(_incident_timeline(window_evidence, window_risk))
        uncertainty = _incident_uncertainty(window_evidence, candidates)
        summary = deterministic_incident_summary(window, top_metrics, candidates)
        reports.append(
            IncidentCandidate(
                incident_id=window.incident_id,
                start_time=window.start_time,
                end_time=window.end_time,
                severity=window.severity,
                risk_score=window.risk_score,
                affected_entities=(window.entity_id,),
                affected_services=(),
                top_metric_signals=tuple(top_metrics),
                root_cause_candidates=tuple(candidates),
                supporting_evidence=tuple(supporting),
                uncertainty=uncertainty,
                data_sources=data_sources,
                timeline=timeline,
                summary=summary,
            )
        )
    return reports


def deterministic_incident_summary(
    window: IncidentWindow,
    top_metric_signals: Sequence[dict[str, Any]],
    candidates: Sequence[RootCauseCandidate],
) -> str:
    """Create a deterministic summary from structured RCA data."""

    metric_names = [str(item["metric_name"]) for item in top_metric_signals[:3]]
    top_candidate = candidates[0].metric_component if candidates else "none"
    metric_text = ", ".join(metric_names) if metric_names else "no metric evidence"
    return (
        f"{window.severity.value} evidence window for {window.entity_id} from "
        f"{window.start_time.isoformat()} to {window.end_time.isoformat()}. "
        f"Top metric signals: {metric_text}. Top candidate signal group: {top_candidate}. "
        "This is evidence-based candidate ranking, not causal proof."
    )


def evaluate_rca_outputs(
    *,
    preliminary_windows: Sequence[IncidentWindow],
    merged_windows: Sequence[IncidentWindow],
    reports: Sequence[IncidentCandidate],
    evidence_signals: pd.DataFrame,
    risk_signals: pd.DataFrame,
    config: RCAScoringConfig,
) -> dict[str, Any]:
    """Evaluate deterministic RCA properties without claiming RCA accuracy."""

    evidence = _prepare_evidence_frame(evidence_signals, config.aggregation_frequency)
    risk = _prepare_risk_frame(risk_signals)
    incident_signal_ids = {
        signal_id for window in merged_windows for signal_id in window.signal_ids
    }
    all_signal_ids = set(evidence["signal_id"].astype(str).tolist())
    high_risk = risk[
        risk["risk_score"].ge(config.min_risk_threshold)
        & risk["evidence_count"].ge(config.min_bucket_evidence_count)
    ]
    covered_risk = 0
    for row in high_risk.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        entity_id = str(row.entity_id)
        if any(
            window.entity_id == entity_id and window.start_time <= timestamp <= window.end_time
            for window in merged_windows
        ):
            covered_risk += 1
    candidates = [candidate for report in reports for candidate in report.root_cause_candidates]
    provenance_complete = [
        bool(candidate.supporting_signal_ids)
        and bool(candidate.data_sources)
        and bool(candidate.source_models)
        for candidate in candidates
    ]
    bucket_delta = pd.Timedelta(config.aggregation_frequency)
    temporal_consistent = [
        report.start_time
        <= candidate.first_evidence_time
        <= candidate.last_evidence_time
        <= report.end_time + bucket_delta
        for report in reports
        for candidate in report.root_cause_candidates
    ]
    rank_consistent = [
        [candidate.rank for candidate in report.root_cause_candidates]
        == list(range(1, len(report.root_cause_candidates) + 1))
        for report in reports
    ]
    duplicate_rate = (
        1.0 - len(merged_windows) / len(preliminary_windows) if preliminary_windows else 0.0
    )
    return {
        "preliminary_incident_windows": int(len(preliminary_windows)),
        "merged_incident_windows": int(len(merged_windows)),
        "duplicate_incident_rate": float(max(0.0, duplicate_rate)),
        "evidence_coverage": (
            len(incident_signal_ids.intersection(all_signal_ids)) / len(all_signal_ids)
            if all_signal_ids
            else 0.0
        ),
        "high_risk_bucket_coverage": (
            covered_risk / int(high_risk.shape[0]) if int(high_risk.shape[0]) else 0.0
        ),
        "average_evidence_count_per_incident": (
            float(np.mean([window.evidence_count for window in merged_windows]))
            if merged_windows
            else 0.0
        ),
        "average_candidates_per_incident": (
            float(np.mean([len(report.root_cause_candidates) for report in reports]))
            if reports
            else 0.0
        ),
        "provenance_completeness": (
            sum(provenance_complete) / len(provenance_complete) if provenance_complete else 0.0
        ),
        "temporal_consistency": (
            sum(temporal_consistent) / len(temporal_consistent) if temporal_consistent else 0.0
        ),
        "rule_consistency": sum(rank_consistent) / len(rank_consistent) if rank_consistent else 0.0,
        "ground_truth_rca_available": False,
        "accuracy_claimed": False,
    }


def run_rca_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: RCAScoringConfig | None = None,
) -> RCAExperimentResult:
    """Run the deterministic Phase 9 incident grouping and RCA experiment."""

    cfg = config or RCAScoringConfig()
    project = project_root() if root is None else root
    resolved_run_id = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    base_output = output_root or project / "experiments" / "rca"
    run_dir = base_output / resolved_run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "config.json", _config_to_json(cfg))

    started = time.perf_counter()
    evidence_run = _resolve_project_path(project, cfg.evidence_run_dir)
    evidence = _read_evidence_signals(evidence_run)
    risk = _read_risk_signals(evidence_run)
    detection = detect_incident_windows(risk, evidence, cfg)
    reports = build_incident_reports(detection.merged_windows, evidence, risk, cfg)
    evaluation = evaluate_rca_outputs(
        preliminary_windows=detection.preliminary_windows,
        merged_windows=detection.merged_windows,
        reports=reports,
        evidence_signals=evidence,
        risk_signals=risk,
        config=cfg,
    )
    case_studies = select_rca_case_studies(reports, evidence, risk, cfg)

    windows_frame = pd.DataFrame([window.to_record() for window in detection.merged_windows])
    preliminary_frame = pd.DataFrame(
        [window.to_record() for window in detection.preliminary_windows]
    )
    candidates = [candidate for report in reports for candidate in report.root_cause_candidates]
    candidate_frame = pd.DataFrame([candidate.to_record() for candidate in candidates])
    evidence_mapping = build_evidence_mapping(reports, evidence)
    timeline_frame = build_incident_timeline_frame(reports)

    windows_frame.to_csv(run_dir / "incident_windows.csv", index=False)
    preliminary_frame.to_csv(run_dir / "preliminary_incident_windows.csv", index=False)
    candidate_frame.to_csv(run_dir / "root_cause_candidates.csv", index=False)
    candidate_frame.to_csv(run_dir / "candidate_rankings.csv", index=False)
    evidence_mapping.to_csv(run_dir / "evidence_mappings.csv", index=False)
    timeline_frame.to_csv(run_dir / "incident_timeline.csv", index=False)
    _save_json(run_dir / "incident_reports.json", [report.to_record() for report in reports])
    _save_json(run_dir / "case_studies.json", case_studies)

    _plot_incident_timeline(risk, detection.merged_windows, figures_dir / "incident_timeline.png")
    _plot_risk_score(risk, detection.merged_windows, figures_dir / "risk_score.png")
    _plot_evidence_signals(
        evidence, cfg.aggregation_frequency, figures_dir / "evidence_signals.png"
    )
    _plot_top_candidate_metrics(candidate_frame, figures_dir / "top_candidate_metrics.png")
    _plot_candidate_ranking(candidate_frame, figures_dir / "candidate_ranking.png")
    _plot_metric_trajectories(evidence, reports, figures_dir / "metric_trajectories.png")
    _plot_anomaly_forecast_overlap(
        evidence,
        cfg.aggregation_frequency,
        figures_dir / "anomaly_forecast_overlap.png",
    )

    metrics = {
        "run_id": resolved_run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "input_artifacts": {
            "evidence_run_dir": str(evidence_run),
            "evidence_signals": str(_evidence_path(evidence_run)),
            "risk_signals": str(_risk_path(evidence_run)),
        },
        "candidate_ranking_methodology": _candidate_formula_text(cfg),
        "persistence_methodology": (
            "Persistence is measured as the span between first and last supporting evidence "
            "for a candidate, plus one aggregation bucket, capped by persistence_cap_minutes."
        ),
        "temporal_precedence_methodology": (
            "Temporal precedence rewards candidates whose first evidence occurs earlier than "
            "the configured incident reference point. It is a ranking feature only and does "
            "not imply causality."
        ),
        "incident_windowing": {
            "min_risk_threshold": cfg.min_risk_threshold,
            "min_bucket_evidence_count": cfg.min_bucket_evidence_count,
            "min_window_evidence_count": cfg.min_window_evidence_count,
            "max_gap_minutes": cfg.max_gap_minutes,
            "merge_gap_minutes": cfg.merge_gap_minutes,
            "merge_pattern_jaccard_threshold": cfg.merge_pattern_jaccard_threshold,
        },
        "preliminary_incident_windows": evaluation["preliminary_incident_windows"],
        "merged_incident_windows": evaluation["merged_incident_windows"],
        "average_evidence_count_per_incident": evaluation["average_evidence_count_per_incident"],
        "severity_counts": windows_frame["severity"].value_counts().to_dict()
        if "severity" in windows_frame
        else {},
        "candidate_count": int(candidate_frame.shape[0]),
        "evaluation": evaluation,
        "case_studies": case_studies,
        "limitations": [
            "The engine ranks evidence-supported candidates, not proven causes.",
            "No causal graph, intervention data, or RCA ground-truth labels are available.",
            "MetroPT service dependencies are not fabricated; affected_services is empty.",
            "Temporal precedence is a weak ordering signal and does not prove causality.",
            "Research-only failure-prediction signals remain low-weight support evidence.",
        ],
    }
    _save_json(run_dir / "metrics.json", metrics)
    _save_json(
        run_dir / "metadata.json",
        {
            "phase": "9",
            "name": "deterministic_incident_grouping_and_rca",
            "created_at_utc": metrics["created_at_utc"],
            "source": "Phase 8B evidence artifacts",
        },
    )
    _write_run_report(run_dir, metrics)
    return RCAExperimentResult(run_dir=run_dir, metrics=metrics)


def build_evidence_mapping(
    reports: Sequence[IncidentCandidate],
    evidence_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Create candidate-to-evidence lineage rows."""

    evidence = evidence_signals.copy()
    evidence["timestamp"] = pd.to_datetime(evidence["timestamp"], errors="raise")
    evidence_by_id = {str(row["signal_id"]): row for row in evidence.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for report in reports:
        for candidate in report.root_cause_candidates:
            for signal_id in candidate.supporting_signal_ids:
                source = evidence_by_id.get(signal_id, {})
                rows.append(
                    {
                        "incident_id": report.incident_id,
                        "candidate_id": candidate.candidate_id,
                        "signal_id": signal_id,
                        "timestamp": source.get("timestamp"),
                        "metric_name": source.get("metric_name"),
                        "signal_type": source.get("signal_type"),
                        "source_dataset": source.get("source_dataset"),
                        "source_model": source.get("source_model"),
                    }
                )
    return pd.DataFrame(rows)


def build_incident_timeline_frame(reports: Sequence[IncidentCandidate]) -> pd.DataFrame:
    """Flatten per-incident timeline entries."""

    rows: list[dict[str, Any]] = []
    for report in reports:
        for event in report.timeline:
            rows.append({"incident_id": report.incident_id, **event})
    return pd.DataFrame(rows)


def select_rca_case_studies(
    reports: Sequence[IncidentCandidate],
    evidence_signals: pd.DataFrame,
    risk_signals: pd.DataFrame,
    config: RCAScoringConfig,
) -> dict[str, Any]:
    """Select normal, isolated, persistent, pre-failure, and noisy case studies."""

    risk = _prepare_risk_frame(risk_signals)
    evidence = _prepare_evidence_frame(evidence_signals, config.aggregation_frequency)
    cases: dict[str, Any] = {}
    if risk.empty:
        return cases
    normal_row = risk.sort_values(["risk_score", "evidence_count"]).iloc[0]
    normal_timestamp = pd.Timestamp(normal_row["timestamp"])
    normal_evidence = evidence[
        evidence["bucket_timestamp"].eq(normal_timestamp)
        & evidence["entity_id"].astype(str).eq(str(normal_row["entity_id"]))
    ]
    cases["normal_period"] = {
        "timestamp": normal_timestamp.isoformat(),
        "risk_score": float(normal_row["risk_score"]),
        "incident_created": False,
        "evidence_count": int(normal_evidence.shape[0]),
        "uncertainty": "Below configured incident threshold; no RCA candidate is produced.",
    }
    if not reports:
        return cases
    shortest = min(
        reports, key=lambda report: (report.end_time - report.start_time, report.risk_score)
    )
    longest = max(
        reports, key=lambda report: (report.end_time - report.start_time, report.risk_score)
    )
    cases["isolated_anomaly"] = _case_from_report(
        shortest,
        "Shortest deduplicated incident window that passed incident criteria.",
    )
    cases["persistent_anomaly"] = _case_from_report(
        longest,
        "Longest deduplicated incident window by persistence duration.",
    )
    event = MetroPTExperimentConfig().failure_events[-1]
    pre_start = event.start - pd.Timedelta(hours=config.case_study_pre_failure_hours)
    pre_failure = [
        report
        for report in reports
        if report.start_time <= event.start and report.end_time >= pre_start
    ]
    if pre_failure:
        selected = max(pre_failure, key=lambda report: (report.risk_score, len(report.timeline)))
        cases["pre_failure_period"] = _case_from_report(
            selected,
            f"Highest-risk RCA window overlapping the pre-failure context for {event.event_id}.",
        )
    else:
        cases["pre_failure_period"] = {
            "available": False,
            "event_id": event.event_id,
            "uncertainty": (
                "No deduplicated incident window overlaps the configured pre-failure context."
            ),
        }
    noisy = max(
        reports,
        key=lambda report: (
            _report_uncertainty_score(report),
            len(report.root_cause_candidates),
            report.risk_score,
        ),
    )
    cases["difficult_noisy_case"] = _case_from_report(
        noisy,
        "Highest uncertainty score from mixed evidence validity and close candidate ranking.",
    )
    return cases


def _window_from_risk_rows(
    rows: Sequence[dict[str, Any]],
    risk_signals: pd.DataFrame,
    evidence_signals: pd.DataFrame,
    config: RCAScoringConfig,
) -> IncidentWindow | None:
    if not rows:
        return None
    entity_id = str(rows[0]["entity_id"])
    start = min(pd.Timestamp(row["timestamp"]) for row in rows)
    end = max(pd.Timestamp(row["timestamp"]) for row in rows)
    return _window_from_time_range(
        entity_id=entity_id,
        start_time=start,
        end_time=end,
        risk_signals=risk_signals,
        evidence_signals=evidence_signals,
        config=config,
    )


def _window_from_time_range(
    *,
    entity_id: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    risk_signals: pd.DataFrame,
    evidence_signals: pd.DataFrame,
    config: RCAScoringConfig,
) -> IncidentWindow | None:
    risk = _window_risk(risk_signals, entity_id, start_time, end_time)
    evidence = _window_evidence(evidence_signals, entity_id, start_time, end_time)
    if risk.empty or evidence.empty:
        return None
    persistence = _window_persistence_minutes(start_time, end_time, config.aggregation_frequency)
    max_risk = float(risk["risk_score"].max())
    evidence_count = int(evidence.shape[0])
    accepted = evidence_count >= config.min_window_evidence_count and (
        persistence >= config.min_persistence_minutes
        or max_risk >= config.critical_single_bucket_threshold
    )
    if not accepted:
        return None
    severity = _max_severity(risk["severity"].astype(str).tolist())
    metrics = tuple(sorted(evidence["metric_name"].astype(str).unique().tolist()))
    evidence_types = tuple(sorted(evidence["signal_type"].astype(str).unique().tolist()))
    signal_ids = tuple(sorted(evidence["signal_id"].astype(str).unique().tolist()))
    incident_id = _incident_id(entity_id, start_time, end_time, signal_ids)
    return IncidentWindow(
        incident_id=incident_id,
        entity_id=entity_id,
        start_time=start_time,
        end_time=end_time,
        severity=severity,
        risk_score=max_risk,
        mean_risk_score=float(risk["risk_score"].mean()),
        evidence_count=evidence_count,
        risk_bucket_count=int(risk.shape[0]),
        persistence_minutes=persistence,
        metric_names=metrics,
        evidence_types=evidence_types,
        signal_ids=signal_ids,
    )


def _score_candidate(
    *,
    window: IncidentWindow,
    evidence: pd.DataFrame,
    risk: pd.DataFrame,
    candidate_type: str,
    metric_component: str,
    config: RCAScoringConfig,
) -> RootCauseCandidate:
    evidence = evidence.sort_values(["timestamp", "signal_id"]).copy()
    first_time = pd.Timestamp(evidence["timestamp"].min())
    last_time = pd.Timestamp(evidence["timestamp"].max())
    persistence = _window_persistence_minutes(first_time, last_time, config.aggregation_frequency)
    signal_types = set(evidence["signal_type"].astype(str).tolist())
    magnitude = float(
        (evidence["normalized_value"].max() + evidence["normalized_value"].mean()) / 2.0
    )
    signal_count_score = _log_score(int(evidence.shape[0]), config.signal_count_cap)
    features = {
        "magnitude": _clamp(magnitude, 0.0, 1.0),
        "persistence": _clamp(persistence / config.persistence_cap_minutes, 0.0, 1.0),
        "signal_count": signal_count_score,
        "multi_signal_support": _clamp(
            len(signal_types) / config.multi_signal_type_cap,
            0.0,
            1.0,
        ),
        "forecast_deviation": _signal_type_max(evidence, "FORECAST_DEVIATION"),
        "trend_strength": _signal_type_max(evidence, "TREND"),
        "state_transition": _signal_type_max(evidence, "STATE_TRANSITION"),
        "temporal_precedence": _temporal_precedence_score(first_time, window, risk, config),
        "historical_recurrence": _signal_type_max(evidence, "HISTORICAL_CONTEXT"),
        "confidence": _clamp(float(evidence["confidence"].mean()), 0.0, 1.0),
    }
    weighted_sum = sum(
        config.candidate_feature_weights[name] * features.get(name, 0.0)
        for name in config.candidate_feature_weights
    )
    weight_total = sum(config.candidate_feature_weights.values())
    role_multiplier = _role_multiplier(signal_types, config)
    type_multiplier = config.candidate_type_weights.get(candidate_type, 1.0)
    score = _clamp(
        (weighted_sum / max(weight_total, 1e-8)) * role_multiplier * type_multiplier,
        0.0,
        1.0,
    )
    supporting = tuple(evidence["signal_id"].astype(str).tolist())
    data_sources = tuple(sorted(evidence["source_dataset"].astype(str).unique().tolist()))
    source_models = tuple(sorted(evidence["source_model"].astype(str).unique().tolist()))
    observed = tuple(
        evidence.sort_values(["normalized_value", "confidence"], ascending=False)
        .head(config.max_evidence_per_candidate)["evidence_description"]
        .astype(str)
        .tolist()
    )
    inference = _candidate_inference(
        metric_component,
        candidate_type,
        evidence_count=int(evidence.shape[0]),
        signal_types=signal_types,
        persistence_minutes=persistence,
        first_time=first_time,
    )
    uncertainty = _candidate_uncertainty(evidence, candidate_type)
    limitations = (
        "Candidate ranking is deterministic evidence aggregation. It does not establish "
        "causality, root-cause ground truth, or remediation certainty."
    )
    candidate_id = _candidate_id(
        window.incident_id,
        candidate_type,
        metric_component,
        supporting,
    )
    return RootCauseCandidate(
        candidate_id=candidate_id,
        incident_id=window.incident_id,
        entity=window.entity_id,
        metric_component=metric_component,
        candidate_type=candidate_type,
        score=score,
        rank=0,
        supporting_signal_ids=supporting,
        observed_evidence=observed,
        inference=inference,
        uncertainty=uncertainty,
        limitations=limitations,
        data_sources=data_sources,
        source_models=source_models,
        first_evidence_time=first_time,
        last_evidence_time=last_time,
        persistence_minutes=persistence,
        feature_scores={key: float(value) for key, value in sorted(features.items())},
    )


def _candidate_groups(
    evidence: pd.DataFrame,
    config: RCAScoringConfig,
) -> list[tuple[str, str, pd.DataFrame]]:
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for metric_name, group in evidence.groupby("metric_name", sort=True, observed=True):
        groups.append(("metric", str(metric_name), group.copy()))
    component_values = (
        evidence["metric_name"]
        .astype(str)
        .map(lambda metric: config.metric_component_map.get(metric, metric))
    )
    with_component = evidence.assign(component=component_values)
    for component, group in with_component.groupby("component", sort=True, observed=True):
        metric_count = int(group["metric_name"].nunique())
        if metric_count > 1 or str(component) != str(group["metric_name"].iloc[0]):
            groups.append(("component", str(component), group.drop(columns=["component"]).copy()))
    entity_group = evidence.copy()
    groups.append(("entity", str(evidence["entity_id"].iloc[0]), entity_group))
    return groups


def _top_metric_signals(evidence: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if evidence.empty:
        return []
    rows: list[dict[str, Any]] = []
    for metric, group in evidence.groupby("metric_name", sort=True, observed=True):
        rows.append(
            {
                "metric_name": str(metric),
                "max_normalized_value": float(group["normalized_value"].max()),
                "mean_normalized_value": float(group["normalized_value"].mean()),
                "evidence_count": int(group.shape[0]),
                "signal_types": sorted(group["signal_type"].astype(str).unique().tolist()),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            float(item["max_normalized_value"]),
            int(item["evidence_count"]),
            str(item["metric_name"]),
        ),
        reverse=True,
    )[:limit]


def _supporting_evidence_records(evidence: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    if evidence.empty:
        return []
    columns = [
        "signal_id",
        "timestamp",
        "entity_id",
        "signal_type",
        "metric_name",
        "value",
        "normalized_value",
        "severity",
        "confidence",
        "source_model",
        "source_dataset",
        "evidence_description",
        "validity",
        "limitations",
    ]
    selected = evidence.sort_values(
        ["normalized_value", "confidence", "timestamp"],
        ascending=[False, False, True],
    ).head(limit)
    records = selected[columns].to_dict(orient="records")
    for record in records:
        record["timestamp"] = pd.Timestamp(record["timestamp"]).isoformat()
    return cast(list[dict[str, Any]], records)


def _incident_timeline(evidence: pd.DataFrame, risk: pd.DataFrame) -> list[dict[str, Any]]:
    timestamps = sorted(
        set(evidence["bucket_timestamp"].tolist()).union(set(risk["timestamp"].tolist()))
    )
    rows: list[dict[str, Any]] = []
    for timestamp in timestamps:
        bucket_evidence = evidence[evidence["bucket_timestamp"].eq(timestamp)]
        bucket_risk = risk[risk["timestamp"].eq(timestamp)]
        risk_score = float(bucket_risk["risk_score"].max()) if not bucket_risk.empty else 0.0
        severity = (
            _max_severity(bucket_risk["severity"].astype(str).tolist()).value
            if not bucket_risk.empty
            else EvidenceSeverity.INFO.value
        )
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "risk_score": risk_score,
                "severity": severity,
                "evidence_count": int(bucket_evidence.shape[0]),
                "evidence_types": sorted(
                    bucket_evidence["signal_type"].astype(str).unique().tolist()
                ),
                "metrics": sorted(bucket_evidence["metric_name"].astype(str).unique().tolist()),
            }
        )
    return rows


def _incident_uncertainty(
    evidence: pd.DataFrame,
    candidates: Sequence[RootCauseCandidate],
) -> str:
    validity = set(evidence["validity"].astype(str).tolist()) if "validity" in evidence else set()
    top_gap = 1.0
    if len(candidates) >= 2:
        top_gap = candidates[0].score - candidates[1].score
    notes = [
        "No causal ground truth is available for this RCA ranking.",
        "Candidates are supported by observed telemetry evidence only.",
    ]
    if "RESEARCH_ONLY" in validity:
        notes.append("Some supporting evidence is research-only and low weighted.")
    if top_gap < 0.05:
        notes.append("Top candidates have close deterministic scores.")
    return " ".join(notes)


def _candidate_inference(
    metric_component: str,
    candidate_type: str,
    *,
    evidence_count: int,
    signal_types: set[str],
    persistence_minutes: float,
    first_time: pd.Timestamp,
) -> str:
    signal_text = ", ".join(sorted(signal_types))
    return (
        f"{candidate_type} candidate '{metric_component}' is ranked from {evidence_count} "
        f"supporting signals ({signal_text}) persisting for {persistence_minutes:.1f} minutes. "
        f"First supporting evidence appears at {first_time.isoformat()}."
    )


def _candidate_uncertainty(evidence: pd.DataFrame, candidate_type: str) -> str:
    validity = sorted(evidence["validity"].astype(str).unique().tolist())
    source_models = sorted(evidence["source_model"].astype(str).unique().tolist())
    return (
        f"Validity labels: {', '.join(validity)}. Source models/rules: "
        f"{', '.join(source_models)}. {candidate_type} grouping is evidence-based and "
        "does not prove causality."
    )


def _temporal_precedence_score(
    first_time: pd.Timestamp,
    window: IncidentWindow,
    risk: pd.DataFrame,
    config: RCAScoringConfig,
) -> float:
    if risk.empty:
        reference = window.end_time
    elif config.temporal_precedence_reference == "peak_risk":
        peak_index = risk["risk_score"].astype(float).idxmax()
        reference = pd.Timestamp(risk.loc[peak_index, "timestamp"])
    else:
        reference = window.end_time
    total = max(_minutes_between(window.start_time, window.end_time), 1e-8)
    lead = max(_minutes_between(first_time, reference), 0.0)
    return _clamp(lead / total, 0.0, 1.0)


def _signal_type_max(evidence: pd.DataFrame, signal_type: str) -> float:
    selected = evidence[evidence["signal_type"].astype(str).eq(signal_type)]
    return float(selected["normalized_value"].max()) if not selected.empty else 0.0


def _role_multiplier(signal_types: set[str], config: RCAScoringConfig) -> float:
    if not signal_types:
        return 1.0
    role_scores = [
        config.signal_type_role_weights.get(signal_type, 0.5) for signal_type in signal_types
    ]
    return _clamp(float(np.mean(role_scores)) + 0.15, 0.35, 1.0)


def _pattern_jaccard(left: IncidentWindow, right: IncidentWindow) -> float:
    left_pattern = set(left.metric_names).union(left.evidence_types)
    right_pattern = set(right.metric_names).union(right.evidence_types)
    if not left_pattern and not right_pattern:
        return 1.0
    union = left_pattern.union(right_pattern)
    if not union:
        return 0.0
    return len(left_pattern.intersection(right_pattern)) / len(union)


def _prepare_risk_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, {"timestamp", "entity_id", "risk_score", "severity", "evidence_count"})
    risk = frame.copy()
    risk["timestamp"] = pd.to_datetime(risk["timestamp"], errors="raise")
    risk["risk_score"] = pd.to_numeric(risk["risk_score"], errors="raise")
    risk["evidence_count"] = pd.to_numeric(risk["evidence_count"], errors="raise")
    return risk.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)


def _prepare_evidence_frame(frame: pd.DataFrame, aggregation_frequency: str) -> pd.DataFrame:
    _require_columns(
        frame,
        {
            "signal_id",
            "timestamp",
            "entity_id",
            "signal_type",
            "metric_name",
            "normalized_value",
            "severity",
            "confidence",
            "source_model",
            "source_dataset",
        },
    )
    evidence = frame.copy()
    evidence["timestamp"] = pd.to_datetime(evidence["timestamp"], errors="raise")
    evidence["bucket_timestamp"] = evidence["timestamp"].dt.floor(aggregation_frequency)
    evidence["normalized_value"] = pd.to_numeric(evidence["normalized_value"], errors="raise")
    evidence["confidence"] = pd.to_numeric(evidence["confidence"], errors="raise")
    return evidence.sort_values(["entity_id", "timestamp", "signal_id"]).reset_index(drop=True)


def _window_evidence(
    evidence: pd.DataFrame,
    entity_id: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    return evidence[
        evidence["entity_id"].astype(str).eq(entity_id)
        & evidence["bucket_timestamp"].ge(start_time)
        & evidence["bucket_timestamp"].le(end_time)
    ].copy()


def _window_risk(
    risk: pd.DataFrame,
    entity_id: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    return risk[
        risk["entity_id"].astype(str).eq(entity_id)
        & risk["timestamp"].ge(start_time)
        & risk["timestamp"].le(end_time)
    ].copy()


def _max_severity(values: Iterable[str]) -> EvidenceSeverity:
    ranked = sorted(values, key=lambda value: SEVERITY_RANK.get(str(value), -1), reverse=True)
    if not ranked:
        return EvidenceSeverity.INFO
    return EvidenceSeverity(str(ranked[0]))


def _window_persistence_minutes(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    aggregation_frequency: str,
) -> float:
    bucket_minutes = max(float(pd.Timedelta(aggregation_frequency).total_seconds()) / 60.0, 0.0)
    elapsed_minutes = float((end_time - start_time).total_seconds()) / 60.0
    return max(elapsed_minutes + bucket_minutes, bucket_minutes)


def _minutes_between(start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
    return float((end_time - start_time).total_seconds()) / 60.0


def _log_score(count: int, cap: int) -> float:
    return _clamp(math.log1p(count) / math.log1p(cap), 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _incident_id(
    entity_id: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    signal_ids: Sequence[str],
) -> str:
    payload = "|".join([entity_id, start_time.isoformat(), end_time.isoformat(), *signal_ids[:20]])
    return f"inc_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def _candidate_id(
    incident_id: str,
    candidate_type: str,
    metric_component: str,
    signal_ids: Sequence[str],
) -> str:
    payload = "|".join([incident_id, candidate_type, metric_component, *signal_ids[:20]])
    return f"rca_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def _report_uncertainty_score(report: IncidentCandidate) -> float:
    validity_terms = [
        "RESEARCH_ONLY" in json.dumps(candidate.observed_evidence)
        or "RESEARCH_ONLY" in candidate.uncertainty
        for candidate in report.root_cause_candidates
    ]
    close_score = 0.0
    if len(report.root_cause_candidates) >= 2:
        close_score = max(
            0.0,
            0.05 - (report.root_cause_candidates[0].score - report.root_cause_candidates[1].score),
        )
    return float(sum(validity_terms)) + close_score


def _case_from_report(report: IncidentCandidate, selection_rule: str) -> dict[str, Any]:
    top_candidate = report.root_cause_candidates[0] if report.root_cause_candidates else None
    return {
        "available": True,
        "selection_rule": selection_rule,
        "incident_id": report.incident_id,
        "start_time": report.start_time.isoformat(),
        "end_time": report.end_time.isoformat(),
        "severity": report.severity.value,
        "risk_score": report.risk_score,
        "top_metric_signals": list(report.top_metric_signals[:5]),
        "top_candidate": top_candidate.to_record() if top_candidate is not None else None,
        "uncertainty": report.uncertainty,
    }


def _candidate_formula_text(config: RCAScoringConfig) -> str:
    terms = " + ".join(
        f"{weight:.2f}*{name}" for name, weight in config.candidate_feature_weights.items()
    )
    return (
        f"candidate_score = role_multiplier * weighted_mean({terms}). Features are normalized "
        "to [0, 1]. The score ranks evidence-supported candidates and is not a causality "
        "probability."
    )


def _read_evidence_signals(evidence_run: Path) -> pd.DataFrame:
    path = _evidence_path(evidence_run)
    if not path.exists():
        raise FileNotFoundError(f"Evidence signals not found: {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _read_risk_signals(evidence_run: Path) -> pd.DataFrame:
    path = _risk_path(evidence_run)
    if not path.exists():
        raise FileNotFoundError(f"Risk signals not found: {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _evidence_path(evidence_run: Path) -> Path:
    parquet_path = evidence_run / "evidence_signals.parquet"
    return parquet_path if parquet_path.exists() else evidence_run / "evidence_signals.csv"


def _risk_path(evidence_run: Path) -> Path:
    parquet_path = evidence_run / "risk_signals.parquet"
    return parquet_path if parquet_path.exists() else evidence_run / "risk_signals.csv"


def _resolve_project_path(project: Path, path: Path) -> Path:
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


def _validate_non_negative_weights(name: str, weights: dict[str, float]) -> None:
    if not weights:
        raise ValueError(f"{name} must not be empty")
    for key, value in weights.items():
        if value < 0:
            raise ValueError(f"{name}.{key} must be non-negative")


def _config_to_json(config: RCAScoringConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["evidence_run_dir"] = str(payload["evidence_run_dir"])
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
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _plot_incident_timeline(
    risk: pd.DataFrame,
    windows: Sequence[IncidentWindow],
    output_path: Path,
) -> None:
    frame = _prepare_risk_frame(risk)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame["timestamp"], frame["risk_score"], color="#2563eb", linewidth=1.1)
    for window in windows:
        ax.axvspan(window.start_time, window.end_time, color="#dc2626", alpha=0.16)
    ax.set_title("Incident Windows over Risk Score")
    ax.set_ylabel("risk score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # type: ignore[no-untyped-call]
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_risk_score(
    risk: pd.DataFrame,
    windows: Sequence[IncidentWindow],
    output_path: Path,
) -> None:
    _plot_incident_timeline(risk, windows, output_path)


def _plot_evidence_signals(
    evidence: pd.DataFrame,
    aggregation_frequency: str,
    output_path: Path,
) -> None:
    frame = _prepare_evidence_frame(evidence, aggregation_frequency)
    if frame.empty:
        return
    counts = frame["signal_type"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="barh", ax=ax, color="#0f766e")
    ax.set_title("RCA Evidence Signal Counts")
    ax.set_xlabel("signals")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_top_candidate_metrics(candidates: pd.DataFrame, output_path: Path) -> None:
    if candidates.empty:
        return
    metrics = candidates[candidates["candidate_type"].eq("metric")].copy()
    if metrics.empty:
        return
    top = (
        metrics.groupby("metric_component", observed=True)["score"]
        .max()
        .sort_values(ascending=True)
        .tail(12)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", ax=ax, color="#7c3aed")
    ax.set_title("Top Metric RCA Candidate Scores")
    ax.set_xlabel("candidate score")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_candidate_ranking(candidates: pd.DataFrame, output_path: Path) -> None:
    if candidates.empty:
        return
    first_incident = str(candidates.sort_values(["incident_id", "rank"])["incident_id"].iloc[0])
    top = candidates[candidates["incident_id"].eq(first_incident)].sort_values("rank").head(10)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["metric_component"], top["score"], color="#ea580c")
    ax.invert_yaxis()
    ax.set_title(f"Candidate Ranking: {first_incident}")
    ax.set_xlabel("candidate score")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_metric_trajectories(
    evidence: pd.DataFrame,
    reports: Sequence[IncidentCandidate],
    output_path: Path,
) -> None:
    if not reports:
        return
    top_report = max(reports, key=lambda report: report.risk_score)
    metrics = [item["metric_name"] for item in top_report.top_metric_signals[:4]]
    frame = _prepare_evidence_frame(evidence, "5min")
    frame = frame[
        frame["metric_name"].astype(str).isin(metrics)
        & frame["bucket_timestamp"].ge(top_report.start_time)
        & frame["bucket_timestamp"].le(top_report.end_time)
    ]
    if frame.empty:
        return
    pivot = (
        frame.groupby(["bucket_timestamp", "metric_name"], observed=True)["normalized_value"]
        .max()
        .unstack(fill_value=0.0)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(ax=ax, linewidth=1.2)
    ax.set_title("Metric Evidence Trajectories")
    ax.set_ylabel("max normalized evidence")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_anomaly_forecast_overlap(
    evidence: pd.DataFrame,
    aggregation_frequency: str,
    output_path: Path,
) -> None:
    frame = _prepare_evidence_frame(evidence, aggregation_frequency)
    selected = frame[
        frame["signal_type"].astype(str).isin(["CURRENT_ANOMALY", "FORECAST_DEVIATION"])
    ]
    if selected.empty:
        return
    pivot = (
        selected.groupby(["bucket_timestamp", "signal_type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    pivot.plot(ax=ax, linewidth=1.1)
    ax.set_title("Current Anomaly and Forecast Deviation Overlap")
    ax.set_ylabel("signals per bucket")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_run_report(run_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Phase 9 RCA Run",
        "",
        f"- Run ID: `{metrics['run_id']}`",
        f"- Preliminary windows: {metrics['preliminary_incident_windows']}",
        f"- Merged windows: {metrics['merged_incident_windows']}",
        f"- Candidate count: {metrics['candidate_count']}",
        "- Accuracy claimed: false",
        "",
        "## Candidate Scoring",
        "",
        metrics["candidate_ranking_methodology"],
        "",
        "## Evaluation",
        "",
    ]
    for key, value in metrics["evaluation"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in metrics["limitations"])
    lines.append("")
    run_dir.joinpath("run_report.md").write_text("\n".join(lines), encoding="utf-8")
