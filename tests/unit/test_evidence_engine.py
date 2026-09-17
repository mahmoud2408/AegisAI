from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aegis_ai.evidence import (
    EvidenceScoringConfig,
    EvidenceSeverity,
    EvidenceSignal,
    EvidenceSignalType,
    EvidenceValidity,
    ReliabilityLevel,
    aggregate_evidence_signals,
    build_current_anomaly_signals,
    build_forecast_deviation_signals,
    build_historical_context_signals,
    build_research_failure_signals,
    build_state_transition_signals,
    build_trend_signals,
    load_evidence_scoring_config,
)
from aegis_ai.evidence.engine import (
    build_evidence_timeline,
    forecast_residual_leakage_audit,
    resolve_reliability,
    severity_from_value,
)


def test_config_loader_parses_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "evidence.yaml"
    config_path.write_text(
        "\n".join(
            [
                "forecasting_run_dir: experiments/example",
                "robustness_prediction_path: null",
                "selected_forecast_model: moving_average",
                "selected_forecast_horizon_minutes: 15",
                "aggregation_frequency: 1min",
                "risk_severity_thresholds:",
                "  low: 0.1",
                "  medium: 0.2",
                "  high: 0.5",
                "  critical: 0.9",
            ]
        ),
        encoding="utf-8",
    )

    config = load_evidence_scoring_config(config_path)

    assert config.forecasting_run_dir == Path("experiments/example")
    assert config.robustness_prediction_path is None
    assert config.selected_forecast_model == "moving_average"
    assert config.selected_forecast_horizon_minutes == 15
    assert config.risk_severity_thresholds["critical"] == pytest.approx(0.9)


def test_forecast_deviation_schema_normalization_and_no_leakage() -> None:
    config = EvidenceScoringConfig()
    forecasts = _forecast_fixture(y_true=20.0, y_pred=10.0)

    signals = build_forecast_deviation_signals(forecasts, _normal_ranges(), config)
    audit = forecast_residual_leakage_audit(
        forecasts,
        {EvidenceSignalType.FORECAST_DEVIATION.value: signals},
    )

    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type is EvidenceSignalType.FORECAST_DEVIATION
    assert signal.timestamp == pd.Timestamp("2020-01-01 00:05:00")
    assert signal.value == pytest.approx(10.0)
    assert signal.normalized_value == pytest.approx(1.0)
    assert signal.severity is EvidenceSeverity.CRITICAL
    assert signal.validity is EvidenceValidity.VALID
    assert audit["leakage_detected"] is False


def test_current_anomaly_preserves_metric_attribution() -> None:
    config = EvidenceScoringConfig()
    forecasts = _forecast_fixture(y_true=16.5, y_pred=15.0)

    signals = build_current_anomaly_signals(forecasts, _normal_ranges(), config)

    assert len(signals) == 1
    assert signals[0].metric_name == "TP2"
    assert signals[0].source_model == "train_normal_range_rule"
    assert "TP2 observed value" in signals[0].evidence_description


def test_trend_uses_prior_and_current_observations_only() -> None:
    config = EvidenceScoringConfig(
        trend_lookback_minutes=10,
        trend_min_points=3,
        trend_z_thresholds={"low": 0.5, "medium": 1.0, "high": 2.0, "critical": 3.0},
    )
    observations = pd.DataFrame(
        {
            "model": ["lstm_multivariate"] * 3,
            "horizon_minutes": [5, 5, 5],
            "split": ["test", "test", "test"],
            "sensor": ["TP2", "TP2", "TP2"],
            "input_end_timestamp": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 00:05:00", "2020-01-01 00:10:00"]
            ),
            "target_timestamp": pd.to_datetime(
                ["2020-01-01 00:05:00", "2020-01-01 00:10:00", "2020-01-01 00:15:00"]
            ),
            "y_true": [10.0, 12.0, 16.0],
            "y_pred": [10.0, 11.0, 13.0],
            "absolute_error": [0.0, 1.0, 3.0],
        }
    )

    signals = build_trend_signals(observations, _normal_ranges(), config)

    assert len(signals) == 1
    assert signals[0].timestamp == pd.Timestamp("2020-01-01 00:15:00")
    assert signals[0].value == pytest.approx(6.0)
    assert signals[0].validity is EvidenceValidity.LIMITED


def test_state_transition_signal_metric_attribution() -> None:
    config = EvidenceScoringConfig(aggregation_frequency="1min")
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01 00:00:00", periods=4, freq="min"),
            "COMP": [0, 0, 1, 1],
            "DV_eletric": [0, 0, 0, 0],
            "Towers": [0, 0, 0, 0],
            "MPG": [0, 0, 0, 0],
            "LPS": [0, 0, 0, 0],
            "Pressure_switch": [0, 0, 0, 0],
            "Oil_level": [0, 0, 0, 0],
            "Caudal_impulses": [0, 0, 0, 0],
        }
    )

    signals = build_state_transition_signals(
        raw,
        config,
        start=pd.Timestamp("2020-01-01 00:00:00"),
        end=pd.Timestamp("2020-01-01 00:03:00"),
    )

    assert len(signals) == 1
    assert signals[0].signal_type is EvidenceSignalType.STATE_TRANSITION
    assert signals[0].metric_name == "COMP"
    assert signals[0].value == pytest.approx(1.0)


def test_research_failure_signal_is_capped_in_aggregation() -> None:
    config = EvidenceScoringConfig()
    predictions = pd.DataFrame(
        {
            "candidate_id": ["h6p0__random_forest"],
            "horizon_hours": [6.0],
            "model": ["random_forest"],
            "split": ["test"],
            "timestamp": [pd.Timestamp("2020-01-01 00:00:00")],
            "sequence_index": [1],
            "future_failure_event_id": [""],
            "target": [0],
            "y_true": [0],
            "probability": [1.0],
            "threshold": [0.3],
            "y_pred": [1],
        }
    )

    signals = build_research_failure_signals(predictions, config)
    risk = aggregate_evidence_signals(signals, config)

    assert signals[0].validity is EvidenceValidity.RESEARCH_ONLY
    assert signals[0].source_model == "metropt_failure_prediction"
    assert risk[0].risk_score == pytest.approx(0.045)


def test_aggregation_combines_weighted_evidence_and_tracks_dominant_type() -> None:
    config = EvidenceScoringConfig()
    signals = [
        _signal(
            EvidenceSignalType.CURRENT_ANOMALY,
            "TP2",
            0.8,
            "train_normal_range_rule",
            ReliabilityLevel.MEDIUM,
        ),
        _signal(
            EvidenceSignalType.TREND,
            "TP2",
            0.5,
            "rolling_trend_rule",
            ReliabilityLevel.LOW,
        ),
    ]

    risk = aggregate_evidence_signals(signals, config)

    assert len(risk) == 1
    assert risk[0].risk_score > 0.6
    assert risk[0].dominant_evidence_types[0] == EvidenceSignalType.CURRENT_ANOMALY.value
    assert "not incident probability" in risk[0].explanation


def test_severity_thresholds_are_configurable() -> None:
    thresholds = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 0.9}

    assert severity_from_value(0.05, thresholds) is EvidenceSeverity.INFO
    assert severity_from_value(0.3, thresholds) is EvidenceSeverity.MEDIUM
    assert severity_from_value(0.91, thresholds) is EvidenceSeverity.CRITICAL


def test_historical_context_counts_prior_signals_only() -> None:
    config = EvidenceScoringConfig(
        aggregation_frequency="5min",
        historical_context_lookback_minutes=30,
        historical_context_min_prior_signals=3,
    )
    timestamps = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:05:00",
            "2020-01-01 00:10:00",
            "2020-01-01 00:15:00",
        ]
    )
    base_signals = [
        _signal(
            EvidenceSignalType.CURRENT_ANOMALY,
            "TP2",
            0.7,
            "train_normal_range_rule",
            ReliabilityLevel.MEDIUM,
            timestamp=timestamp,
        )
        for timestamp in timestamps
    ]

    context = build_historical_context_signals(base_signals, config)

    assert len(context) == 1
    assert context[0].timestamp == pd.Timestamp("2020-01-01 00:15:00")
    assert context[0].value == pytest.approx(3.0)
    assert "3 prior" in context[0].evidence_description


def test_resolve_reliability_defaults_unknown_sources_to_low() -> None:
    config = EvidenceScoringConfig()

    assert resolve_reliability("unregistered_detector", "metropt", config) is ReliabilityLevel.LOW
    assert (
        resolve_reliability("metropt_failure_prediction_v1", "metropt", config)
        is ReliabilityLevel.RESEARCH_ONLY
    )


def test_evidence_timeline_buckets_evidence_before_joining_risk() -> None:
    evidence = pd.DataFrame(
        [
            {
                "timestamp": "2020-01-01T00:04:00",
                "signal_type": EvidenceSignalType.CURRENT_ANOMALY.value,
            }
        ]
    )
    risk = pd.DataFrame(
        [
            {
                "timestamp": "2020-01-01T00:00:00",
                "entity_id": "metropt_compressor",
                "risk_score": 0.5,
                "severity": "MEDIUM",
                "evidence_count": 1,
                "dominant_evidence_types": EvidenceSignalType.CURRENT_ANOMALY.value,
                "confidence": 0.5,
                "model_sources": "train_normal_range_rule",
                "explanation": "test",
            }
        ]
    )

    timeline = build_evidence_timeline(evidence, risk, "5min")

    assert int(timeline.loc[0, EvidenceSignalType.CURRENT_ANOMALY.value]) == 1


def _normal_ranges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor": ["TP2"],
            "mean": [10.0],
            "std": [2.0],
            "low_quantile": [6.0],
            "high_quantile": [14.0],
        }
    )


def _forecast_fixture(*, y_true: float, y_pred: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["lstm_multivariate"],
            "horizon_minutes": [5],
            "split": ["test"],
            "sensor": ["TP2"],
            "input_end_timestamp": [pd.Timestamp("2020-01-01 00:00:00")],
            "target_timestamp": [pd.Timestamp("2020-01-01 00:05:00")],
            "segment_id": [1],
            "y_true": [y_true],
            "y_pred": [y_pred],
            "absolute_error": [abs(y_true - y_pred)],
        }
    )


def _signal(
    signal_type: EvidenceSignalType,
    metric_name: str,
    normalized_value: float,
    source_model: str,
    reliability: ReliabilityLevel,
    *,
    timestamp: pd.Timestamp | None = None,
) -> EvidenceSignal:
    config = EvidenceScoringConfig()
    confidence = config.reliability_weights[reliability.value]
    resolved_timestamp = timestamp or pd.Timestamp("2020-01-01 00:00:00")
    return EvidenceSignal(
        signal_id=f"{signal_type.value}-{metric_name}-{resolved_timestamp.isoformat()}",
        timestamp=resolved_timestamp,
        entity_id="metropt_compressor",
        signal_type=signal_type,
        metric_name=metric_name,
        value=1.0,
        normalized_value=normalized_value,
        severity=EvidenceSeverity.HIGH,
        confidence=confidence,
        source_model=source_model,
        source_dataset="metropt",
        evidence_description="test evidence",
        validity=EvidenceValidity.VALID,
        limitations="test-only signal",
    )
