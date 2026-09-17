from __future__ import annotations

import pandas as pd
import pytest

from aegis_ai.evidence import EvidenceSignalType, EvidenceValidity
from aegis_ai.logs import (
    LogIntelligenceConfig,
    aggregate_log_events,
    build_log_evidence_signals,
    detect_component_anomalies,
    detect_frequency_anomalies,
    detect_level_anomalies,
    detect_rare_events,
    detect_sequence_anomalies,
    evaluate_labeled_log_signals,
    evaluate_structural_log_signals,
    normalize_log_labels,
    normalize_log_records,
    select_bucket_frequency,
)


def test_normalize_log_records_preserves_missing_values_and_timestamps() -> None:
    records = [
        {
            "timestamp": "2020-01-01T00:00:00Z",
            "source_dataset": "loghub-test",
            "source_file": "test.log_structured.csv",
            "event_id": "canonical-1",
            "sequence_index": 1,
            "entity_id": "node-a",
            "service_id": None,
            "process_id": None,
            "request_id": None,
            "log_level": "warn",
            "event_type": "E_WARN",
            "event_template": None,
            "message": "warning without optional IDs",
            "source_row_id": "1",
            "quality_flag": "ok",
        }
    ]

    events = normalize_log_records(records)

    assert events.shape[0] == 1
    assert pd.Timestamp(events.loc[0, "timestamp"]).tzinfo is None
    assert events.loc[0, "component"] == "node-a"
    assert pd.isna(events.loc[0, "process_id"])
    assert pd.isna(events.loc[0, "request_id"])
    assert events.loc[0, "log_level"] == "WARN"
    assert events.loc[0, "event_id"] == "E_WARN"


def test_select_bucket_frequency_uses_dataset_coverage_and_overrides() -> None:
    short = _events(
        [
            ("2020-01-01 00:00:00", "E1", "INFO", "api"),
            ("2020-01-01 00:00:30", "E1", "INFO", "api"),
        ]
    )
    long = _events(
        [
            ("2020-01-01 00:00:00", "E1", "INFO", "api"),
            ("2020-01-20 00:00:00", "E1", "INFO", "api"),
        ]
    )
    override = LogIntelligenceConfig(
        families=("test",),
        telemetry_evidence_run_dir=None,
        bucket_frequency_overrides={"loghub-test": "5min"},
    )
    auto = LogIntelligenceConfig(families=("test",), telemetry_evidence_run_dir=None)

    assert select_bucket_frequency(short, auto) == "1s"
    assert select_bucket_frequency(long, auto) == "1D"
    assert select_bucket_frequency(short, override) == "5min"


def test_aggregate_log_events_builds_bucket_counts_and_sequences() -> None:
    events = _events(
        [
            ("2020-01-01 00:00:00", "E1", "INFO", "api"),
            ("2020-01-01 00:00:01", "E2", "WARN", "api"),
            ("2020-01-01 00:00:01", "E2", "WARN", "worker"),
        ]
    )

    aggregates = aggregate_log_events(events, "1s")

    assert aggregates["bucket_summary"]["event_count"].sum() == 3
    assert aggregates["bucket_summary"]["anomalous_level_count"].sum() == 2
    assert set(aggregates["event_frequency"]["event_id"]) == {"E1", "E2"}
    assert set(aggregates["level_frequency"]["log_level"]) == {"INFO", "WARN"}
    assert set(aggregates["component_frequency"]["component"]) == {"api", "worker"}
    assert aggregates["event_sequences"].shape[0] == 2
    assert "E1->E2" in set(aggregates["event_sequences"]["transition"])


def test_log_detectors_generate_typed_evidence_with_provenance() -> None:
    events = _detector_events()
    config = _config()
    aggregates = aggregate_log_events(events, "1s")

    rare_signals = detect_rare_events(events, config)
    frequency_signals = detect_frequency_anomalies(
        events,
        aggregates["event_frequency"],
        config,
    )
    level_signals = detect_level_anomalies(events, aggregates["bucket_summary"], config)
    sequence_signals = detect_sequence_anomalies(aggregates["event_sequences"], config)
    component_signals = detect_component_anomalies(
        events,
        aggregates["component_frequency"],
        config,
    )

    assert any(signal.event_id == "E_RARE" for signal in rare_signals)
    assert any(
        signal.signal_type is EvidenceSignalType.LOG_FREQUENCY_ANOMALY
        for signal in frequency_signals
    )
    assert any(
        signal.signal_type is EvidenceSignalType.LOG_LEVEL_ANOMALY for signal in level_signals
    )
    assert any("->" in str(signal.event_id) for signal in sequence_signals)
    assert any(signal.component == "worker" for signal in component_signals)

    signal = rare_signals[0]
    evidence_signal = signal.to_evidence_signal()
    assert evidence_signal.signal_type in {
        EvidenceSignalType.LOG_RARE_EVENT,
        EvidenceSignalType.LOG_SEQUENCE_ANOMALY,
    }
    assert evidence_signal.source_dataset == "loghub-test"
    assert evidence_signal.source_model in {"rare_event_detector", "transition_rarity_detector"}
    assert evidence_signal.validity is EvidenceValidity.LIMITED
    assert signal.source_file == "test.log_structured.csv"
    assert signal.source_event_ids


def test_build_log_evidence_signals_and_evaluations_are_reproducible() -> None:
    events = _detector_events()
    labels = normalize_log_labels(
        [
            {
                "event_id": "row-23",
                "timestamp": "2020-01-01T00:00:31Z",
                "sequence_index": 23,
                "label_kind": "anomaly",
                "label_value": "anomaly",
                "source_dataset": "loghub-test",
            },
            {
                "event_id": "row-0",
                "timestamp": "2020-01-01T00:00:00Z",
                "sequence_index": 0,
                "label_kind": "normal",
                "label_value": "normal",
                "source_dataset": "loghub-test",
            },
        ]
    )

    first_signals, first_aggregates = build_log_evidence_signals(events, _config())
    second_signals, second_aggregates = build_log_evidence_signals(events, _config())
    label_eval = evaluate_labeled_log_signals(events, labels, first_signals)
    structural_eval = evaluate_structural_log_signals(events, first_signals)

    assert [signal.signal_id for signal in first_signals] == [
        signal.signal_id for signal in second_signals
    ]
    assert not first_aggregates["event_sequences"].empty
    assert not second_aggregates["event_sequences"].empty
    assert label_eval["available"] is True
    assert label_eval["accuracy_claimed"] is False
    assert label_eval["true_positive"] >= 1
    assert structural_eval["deterministic_reproducibility"] is True
    assert structural_eval["provenance_completeness"] == pytest.approx(1.0)


def _config() -> LogIntelligenceConfig:
    return LogIntelligenceConfig(
        families=("test",),
        telemetry_evidence_run_dir=None,
        bucket_frequency_overrides={"loghub-test": "1s"},
        rare_event_max_frequency=0.01,
        rare_event_max_count=1,
        frequency_z_threshold=2.0,
        frequency_min_count=5,
        level_rate_z_threshold=2.0,
        level_min_count=5,
        component_z_threshold=2.0,
        component_min_count=5,
        transition_max_frequency=0.05,
        transition_max_count=1,
        transition_max_occurrences=2,
    )


def _detector_events() -> pd.DataFrame:
    rows: list[tuple[str, str, str, str]] = []
    rows.extend(
        (f"2020-01-01 00:00:{second:02d}", "E_COMMON", "INFO", "api") for second in range(10)
    )
    rows.extend(("2020-01-01 00:00:20", "E_SPIKE", "INFO", "worker") for _ in range(8))
    rows.extend(("2020-01-01 00:00:30", "E_ERROR", "ERROR", "db") for _ in range(5))
    rows.append(("2020-01-01 00:00:31", "E_RARE", "INFO", "api"))
    return _events(rows)


def _events(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    records = []
    for index, (timestamp, event_id, level, component) in enumerate(rows):
        records.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "dataset": "loghub-test",
                "source_dataset": "loghub-test",
                "source_file": "test.log_structured.csv",
                "canonical_log_event_id": f"row-{index}",
                "sequence_index": index,
                "entity_id": "node-a",
                "component": component,
                "process_id": None,
                "request_id": None,
                "log_level": level,
                "event_id": event_id,
                "event_template": f"{event_id} template",
                "message": f"{event_id} message",
                "source_row_id": str(index),
                "quality_flag": "ok",
            }
        )
    return pd.DataFrame(records)
