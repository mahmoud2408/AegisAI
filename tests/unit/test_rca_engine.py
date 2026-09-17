from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aegis_ai.evidence import EvidenceSeverity
from aegis_ai.rca import (
    RCAScoringConfig,
    build_evidence_mapping,
    build_incident_reports,
    detect_incident_windows,
    load_rca_scoring_config,
    rank_root_cause_candidates,
)


def test_config_loader_parses_phase9_yaml(tmp_path: Path) -> None:
    path = tmp_path / "rca.yaml"
    path.write_text(
        "\n".join(
            [
                "evidence_run_dir: experiments/evidence/example",
                "min_risk_threshold: 0.7",
                "max_gap_minutes: 9",
                "candidate_feature_weights:",
                "  magnitude: 1.0",
                "  temporal_precedence: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_rca_scoring_config(path)

    assert config.evidence_run_dir == Path("experiments/evidence/example")
    assert config.min_risk_threshold == pytest.approx(0.7)
    assert config.max_gap_minutes == pytest.approx(9.0)
    assert config.candidate_feature_weights["temporal_precedence"] == pytest.approx(0.5)


def test_incident_window_grouping_and_deduplication() -> None:
    config = _config(max_gap_minutes=10)
    risk = _risk_frame(
        [
            ("2020-01-01 00:00:00", 0.7, "HIGH", 2),
            ("2020-01-01 00:05:00", 0.8, "CRITICAL", 2),
            ("2020-01-01 00:10:00", 0.7, "HIGH", 2),
            ("2020-01-01 00:40:00", 0.7, "HIGH", 2),
        ]
    )
    evidence = _evidence_frame(
        [
            ("s1", "2020-01-01 00:00:00", "TP2", "CURRENT_ANOMALY", 0.8),
            ("s2", "2020-01-01 00:05:00", "TP2", "TREND", 0.7),
            ("s3", "2020-01-01 00:10:00", "TP2", "FORECAST_DEVIATION", 0.8),
            ("s4", "2020-01-01 00:40:00", "TP3", "CURRENT_ANOMALY", 0.8),
            ("s5", "2020-01-01 00:40:00", "TP3", "TREND", 0.8),
        ]
    )

    result = detect_incident_windows(risk, evidence, config)

    assert len(result.preliminary_windows) == 2
    assert len(result.merged_windows) == 2
    assert result.merged_windows[0].start_time == pd.Timestamp("2020-01-01 00:00:00")
    assert result.merged_windows[0].end_time == pd.Timestamp("2020-01-01 00:10:00")
    assert result.merged_windows[0].severity is EvidenceSeverity.CRITICAL
    assert result.merged_windows[0].persistence_minutes == pytest.approx(15.0)


def test_incident_merging_uses_gap_and_pattern_overlap() -> None:
    config = _config(max_gap_minutes=1, merge_gap_minutes=30, merge_pattern_jaccard_threshold=0.2)
    risk = _risk_frame(
        [
            ("2020-01-01 00:00:00", 0.8, "HIGH", 2),
            ("2020-01-01 00:20:00", 0.8, "HIGH", 2),
        ]
    )
    evidence = _evidence_frame(
        [
            ("s1", "2020-01-01 00:00:00", "TP2", "CURRENT_ANOMALY", 0.8),
            ("s2", "2020-01-01 00:00:00", "TP2", "TREND", 0.7),
            ("s3", "2020-01-01 00:20:00", "TP2", "CURRENT_ANOMALY", 0.9),
            ("s4", "2020-01-01 00:20:00", "TP2", "FORECAST_DEVIATION", 0.8),
        ]
    )

    result = detect_incident_windows(risk, evidence, config)

    assert len(result.preliminary_windows) == 2
    assert len(result.merged_windows) == 1
    assert result.merged_windows[0].start_time == pd.Timestamp("2020-01-01 00:00:00")
    assert result.merged_windows[0].end_time == pd.Timestamp("2020-01-01 00:20:00")


def test_candidate_scoring_prefers_persistent_multi_signal_evidence() -> None:
    config = _config()
    risk = _risk_frame(
        [
            ("2020-01-01 00:00:00", 0.8, "HIGH", 2),
            ("2020-01-01 00:05:00", 0.9, "CRITICAL", 2),
            ("2020-01-01 00:10:00", 0.8, "HIGH", 2),
        ]
    )
    evidence = _evidence_frame(
        [
            ("a1", "2020-01-01 00:00:00", "Motor_current", "CURRENT_ANOMALY", 0.8),
            ("a2", "2020-01-01 00:05:00", "Motor_current", "TREND", 0.8),
            ("a3", "2020-01-01 00:10:00", "Motor_current", "FORECAST_DEVIATION", 0.8),
            ("b1", "2020-01-01 00:05:00", "Oil_temperature", "CURRENT_ANOMALY", 0.95),
        ]
    )
    window = detect_incident_windows(risk, evidence, config).merged_windows[0]

    candidates = rank_root_cause_candidates(window, evidence, risk, config)

    assert candidates[0].metric_component in {"Motor_current", "motor_electrical_system"}
    assert candidates[0].score > candidates[-1].score
    assert len(candidates[0].supporting_signal_ids) >= 3


def test_temporal_precedence_feature_rewards_earlier_signal() -> None:
    config = _config(
        candidate_feature_weights={
            "magnitude": 0.1,
            "persistence": 0.0,
            "signal_count": 0.0,
            "multi_signal_support": 0.0,
            "forecast_deviation": 0.0,
            "trend_strength": 0.0,
            "state_transition": 0.0,
            "temporal_precedence": 1.0,
            "historical_recurrence": 0.0,
            "confidence": 0.0,
        }
    )
    risk = _risk_frame(
        [
            ("2020-01-01 00:00:00", 0.7, "HIGH", 2),
            ("2020-01-01 00:10:00", 0.95, "CRITICAL", 2),
        ]
    )
    evidence = _evidence_frame(
        [
            ("early", "2020-01-01 00:00:00", "TP2", "CURRENT_ANOMALY", 0.8),
            ("late", "2020-01-01 00:10:00", "TP3", "CURRENT_ANOMALY", 0.8),
        ]
    )
    window = detect_incident_windows(risk, evidence, config).merged_windows[0]

    candidates = rank_root_cause_candidates(window, evidence, risk, config)
    score_by_metric = {candidate.metric_component: candidate.score for candidate in candidates}

    assert score_by_metric["TP2"] > score_by_metric["TP3"]


def test_reports_preserve_provenance_and_uncertainty() -> None:
    config = _config()
    risk = _risk_frame(
        [
            ("2020-01-01 00:00:00", 0.9, "CRITICAL", 2),
            ("2020-01-01 00:05:00", 0.8, "HIGH", 2),
        ]
    )
    evidence = _evidence_frame(
        [
            ("s1", "2020-01-01 00:00:00", "TP2", "CURRENT_ANOMALY", 0.8),
            ("s2", "2020-01-01 00:05:00", "TP2", "FORECAST_DEVIATION", 0.8),
            ("s3", "2020-01-01 00:05:00", "failure_prediction_probability", "FAILURE_SIGNAL", 0.5),
        ],
        validity_by_signal={"s3": "RESEARCH_ONLY"},
        source_model_by_signal={"s3": "metropt_failure_prediction"},
    )
    windows = detect_incident_windows(risk, evidence, config).merged_windows

    reports = build_incident_reports(windows, evidence, risk, config)
    mapping = build_evidence_mapping(reports, evidence)

    assert reports[0].severity is EvidenceSeverity.CRITICAL
    assert "not causal proof" in reports[0].summary
    assert "No causal ground truth" in reports[0].uncertainty
    assert not mapping.empty
    assert set(mapping["signal_id"]).issubset(set(evidence["signal_id"]))
    assert mapping["source_dataset"].notna().all()
    assert reports[0].root_cause_candidates[0].limitations.startswith("Candidate ranking")


def _config(
    *,
    max_gap_minutes: float = 15.0,
    merge_gap_minutes: float = 20.0,
    merge_pattern_jaccard_threshold: float = 0.25,
    candidate_feature_weights: dict[str, float] | None = None,
) -> RCAScoringConfig:
    return RCAScoringConfig(
        min_risk_threshold=0.6,
        min_bucket_evidence_count=1,
        min_window_evidence_count=1,
        max_gap_minutes=max_gap_minutes,
        merge_gap_minutes=merge_gap_minutes,
        merge_pattern_jaccard_threshold=merge_pattern_jaccard_threshold,
        candidate_feature_weights=candidate_feature_weights
        or RCAScoringConfig().candidate_feature_weights,
    )


def _risk_frame(rows: list[tuple[str, float, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(timestamp),
                "entity_id": "metropt_compressor",
                "risk_score": risk_score,
                "severity": severity,
                "evidence_count": evidence_count,
                "dominant_evidence_types": "CURRENT_ANOMALY",
                "confidence": 0.5,
                "model_sources": "test_rule",
                "explanation": "test risk",
            }
            for timestamp, risk_score, severity, evidence_count in rows
        ]
    )


def _evidence_frame(
    rows: list[tuple[str, str, str, str, float]],
    *,
    validity_by_signal: dict[str, str] | None = None,
    source_model_by_signal: dict[str, str] | None = None,
) -> pd.DataFrame:
    validity_by_signal = validity_by_signal or {}
    source_model_by_signal = source_model_by_signal or {}
    return pd.DataFrame(
        [
            {
                "signal_id": signal_id,
                "timestamp": pd.Timestamp(timestamp),
                "entity_id": "metropt_compressor",
                "signal_type": signal_type,
                "metric_name": metric_name,
                "value": normalized_value,
                "normalized_value": normalized_value,
                "severity": "HIGH",
                "confidence": 0.75,
                "source_model": source_model_by_signal.get(signal_id, "test_rule"),
                "source_dataset": "metropt",
                "evidence_description": (
                    f"{metric_name} {signal_type} evidence at strength {normalized_value}"
                ),
                "validity": validity_by_signal.get(signal_id, "VALID"),
                "limitations": "test evidence limitation",
            }
            for signal_id, timestamp, metric_name, signal_type, normalized_value in rows
        ]
    )
