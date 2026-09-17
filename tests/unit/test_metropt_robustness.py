from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis_ai.data.adapters.metropt import BINARY_STATE_COLUMNS, CONTINUOUS_SENSORS
from aegis_ai.ml.prediction.failure_prediction import (
    FailureEvent,
    MetroPTExperimentConfig,
    build_metropt_causal_features,
    build_metropt_failure_target,
)
from aegis_ai.ml.prediction.metropt_robustness import (
    audit_horizon_target_integrity,
    characterize_failure_events,
    distribution_shift_for_candidate,
    false_alarm_analysis,
    load_metropt_robustness_config,
    prepare_horizon_modeling_frame,
    select_best_candidate,
    summarize_failure_event_structure,
)


def test_characterize_failure_events_counts_gaps_and_split_roles() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=80, freq="min")
    events = (
        _event("event-a", "2020-01-01 00:10:00", "2020-01-01 00:12:00"),
        _event("event-b", "2020-01-01 00:30:00", "2020-01-01 00:34:00"),
        _event("event-c", "2020-01-01 01:00:00", "2020-01-01 01:02:00"),
    )

    table = characterize_failure_events(
        _metropt_fixture(timestamps),
        failure_events=events,
        train_end=pd.Timestamp("2020-01-01 00:20:00"),
        validation_end=pd.Timestamp("2020-01-01 00:50:00"),
    )
    summary = summarize_failure_event_structure(table)

    assert table["number_of_observations"].tolist() == [3, 5, 3]
    assert table["split_role"].tolist() == ["train", "validation", "test"]
    assert table.loc[1, "hours_since_previous_start"] == pytest.approx(20 / 60)
    assert table["overlaps_previous_event"].sum() == 0
    assert summary["failure_event_count"] == 3
    assert summary["held_out_test_event_count"] == 1
    assert summary["held_out_july_failure_is_unique"] is False
    assert (
        summary["supervised_learning_sufficiency"]
        == "insufficient_for_reliable_supervised_learning"
    )


def test_target_integrity_rules_out_prefailure_and_split_leakage() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=75, freq="min")
    events = (
        _event("train-event", "2020-01-01 00:15:00", "2020-01-01 00:16:00"),
        _event("validation-event", "2020-01-01 00:35:00", "2020-01-01 00:36:00"),
        _event("test-event", "2020-01-01 00:55:00", "2020-01-01 00:56:00"),
    )
    labeled = build_metropt_failure_target(
        _metropt_fixture(timestamps),
        failure_events=events,
        horizon_hours=5 / 60,
    )
    features = build_metropt_causal_features(
        labeled,
        continuous_columns=CONTINUOUS_SENSORS,
        state_columns=BINARY_STATE_COLUMNS,
        lag_steps=(1,),
        rolling_windows=("3min",),
        trend_window="3min",
        min_periods=1,
        state_transition_count_window="3min",
    )
    modeling = prepare_horizon_modeling_frame(
        features,
        MetroPTExperimentConfig(
            horizon_hours=5 / 60,
            prediction_stride_rows=1,
            train_end="2020-01-01 00:25:00",
            validation_end="2020-01-01 00:45:00",
            lag_steps=(1,),
            rolling_windows=("3min",),
            trend_window="3min",
            min_periods=1,
            failure_events=events,
        ),
    )

    audit = audit_horizon_target_integrity(
        labeled,
        modeling,
        failure_events=events,
        horizon_hours=5 / 60,
        train_end=pd.Timestamp("2020-01-01 00:25:00"),
        validation_end=pd.Timestamp("2020-01-01 00:45:00"),
    )

    assert audit["leakage_detected"] is False
    assert audit["active_failure_rows_in_modeling_frame"] == 0
    assert audit["invalid_positive_target_rows"] == 0
    assert audit["target_window_crosses_split_boundary_count"] == 0


def test_target_integrity_flags_boundary_contamination() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=40, freq="min")
    events = (_event("validation-event", "2020-01-01 00:22:00", "2020-01-01 00:23:00"),)
    labeled = build_metropt_failure_target(
        _metropt_fixture(timestamps),
        failure_events=events,
        horizon_hours=10 / 60,
    )
    modeling = labeled.assign(
        split=np.where(labeled["timestamp"] < timestamps[20], "train", "validation")
    )

    audit = audit_horizon_target_integrity(
        labeled,
        modeling,
        failure_events=events,
        horizon_hours=10 / 60,
        train_end=timestamps[20],
        validation_end=timestamps[30],
    )

    assert audit["leakage_detected"] is True
    assert audit["target_window_crosses_split_boundary_count"] == 1


def test_transition_count_feature_is_causal() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=5, freq="min")
    frame = _metropt_fixture(timestamps).assign(
        target=0,
        future_failure_event_id="",
        is_failure_period=False,
        eligible_for_prediction=True,
    )
    frame["COMP"] = [0, 0, 1, 1, 0]

    features = build_metropt_causal_features(
        frame,
        continuous_columns=CONTINUOUS_SENSORS,
        state_columns=BINARY_STATE_COLUMNS,
        lag_steps=(1,),
        rolling_windows=("3min",),
        trend_window="3min",
        min_periods=1,
        state_transition_count_window="3min",
    ).set_index("timestamp")

    assert features.loc[timestamps[4], "COMP__transition_count_3min"] == pytest.approx(2.0)


def test_false_alarm_analysis_counts_persistent_warning_episodes() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=15, freq="min")
    event = _event("event-1", "2020-01-01 00:10:00", "2020-01-01 00:12:00")
    probabilities = np.zeros(timestamps.shape[0])
    probabilities[[0, 1, 2, 6, 13]] = 0.8
    target = np.zeros(timestamps.shape[0], dtype=int)
    target[5:10] = 1

    result = false_alarm_analysis(
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "probability": probabilities,
                "target": target,
            }
        ),
        failure_events=(event,),
        horizon_hours=5 / 60,
        threshold=0.5,
    )

    assert result["false_warnings"] == 4
    assert result["persistent_warning_episodes"] == 2
    assert result["false_alarm_rate"] > 0


def test_select_best_candidate_uses_validation_not_test() -> None:
    metrics = pd.DataFrame(
        [
            {
                "candidate_id": "validation_winner",
                "validation_warning_coverage": 1.0,
                "validation_f1": 0.4,
                "validation_pr_auc": 0.5,
                "validation_false_alarms_per_day": 5.0,
                "validation_false_positive_rate": 0.1,
                "test_f1": 0.0,
            },
            {
                "candidate_id": "test_winner",
                "validation_warning_coverage": 0.0,
                "validation_f1": 0.2,
                "validation_pr_auc": 0.3,
                "validation_false_alarms_per_day": 1.0,
                "validation_false_positive_rate": 0.0,
                "test_f1": 1.0,
            },
        ]
    )

    assert select_best_candidate(metrics)["candidate_id"] == "validation_winner"


def test_distribution_shift_compares_train_and_test_positive_rows() -> None:
    modeling = pd.DataFrame(
        {
            "split": ["train", "train", "test", "test", "test"],
            "target": [1, 1, 1, 1, 0],
            "feature_a": [1.0, 2.0, 10.0, 12.0, 3.0],
            "feature_b": [4.0, 5.0, 5.0, 5.5, 0.0],
        }
    )

    shift = distribution_shift_for_candidate(
        modeling,
        feature_columns=("feature_a", "feature_b"),
        candidate_id="candidate",
        horizon_hours=1.0,
        model="random_forest",
    )

    assert shift.iloc[0]["feature"] == "feature_a"
    assert shift.iloc[0]["standardized_mean_difference"] > 1


def test_load_metropt_robustness_config_parses_horizons_and_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "horizons_hours: [0.5, 2.0]",
                "threshold_grid: [0.01, 0.1]",
                "state_transition_count_window: 15min",
                "models:",
                "  random_forest_estimators: 5",
                "shap:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_metropt_robustness_config(config_path)

    assert config.horizons_hours == (0.5, 2.0)
    assert config.threshold_grid == (0.01, 0.1)
    assert config.state_transition_count_window == "15min"
    assert config.models.random_forest_estimators == 5
    assert config.shap.enabled is False


def _event(event_id: str, start_time: str, end_time: str) -> FailureEvent:
    return FailureEvent(
        event_id=event_id,
        start_time=start_time,
        end_time=end_time,
        failure_type="air_leak",
        severity="high",
        source="fixture",
    )


def _metropt_fixture(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    row_count = timestamps.shape[0]
    rows: dict[str, object] = {"timestamp": timestamps}
    base = np.arange(row_count, dtype=float)
    for offset, column in enumerate(CONTINUOUS_SENSORS):
        rows[column] = base + float(offset)
    for offset, column in enumerate(BINARY_STATE_COLUMNS):
        rows[column] = ((base // 4 + offset) % 2).astype(float)
    return pd.DataFrame(rows)
