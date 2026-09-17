from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis_ai.data.adapters.ai4i import FAILURE_COLUMNS, NUMERIC_FEATURES
from aegis_ai.data.adapters.metropt import BINARY_STATE_COLUMNS, CONTINUOUS_SENSORS
from aegis_ai.ml.prediction.failure_prediction import (
    AI4IExperimentConfig,
    FailureEvent,
    FailureModelConfig,
    FailurePredictionExperimentConfig,
    MetroPTExperimentConfig,
    SHAPConfig,
    _select_final_tree_model,
    build_metropt_causal_features,
    build_metropt_failure_target,
    classification_metrics,
    compute_early_warning_metrics,
    optimize_threshold,
    prepare_metropt_modeling_frame,
    run_ai4i_failure_prediction,
    sample_temporal_training_rows,
    stratified_ai4i_split,
)


def test_stratified_ai4i_split_preserves_class_presence() -> None:
    y = np.asarray([0] * 100 + [1] * 10, dtype=int)

    split = stratified_ai4i_split(
        y,
        train_fraction=0.6,
        validation_fraction=0.2,
        random_seed=7,
    )

    assert len(np.intersect1d(split["train"], split["validation"])) == 0
    assert len(np.intersect1d(split["train"], split["test"])) == 0
    assert len(np.intersect1d(split["validation"], split["test"])) == 0
    assert sum(indices.shape[0] for indices in split.values()) == y.shape[0]
    for indices in split.values():
        assert set(y[indices].tolist()) == {0, 1}


def test_metropt_target_marks_future_horizon_and_excludes_active_failure() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=15, freq="min")
    event = FailureEvent(
        event_id="event-1",
        start_time="2020-01-01 00:10:00",
        end_time="2020-01-01 00:12:00",
        failure_type="air_leak",
        severity="high",
        source="fixture",
    )

    labeled = build_metropt_failure_target(
        _metropt_fixture(timestamps),
        failure_events=(event,),
        horizon_hours=5 / 60,
    )

    target_by_time = labeled.set_index("timestamp")["target"].to_dict()
    assert [target_by_time[timestamps[index]] for index in range(5, 10)] == [1, 1, 1, 1, 1]
    assert [target_by_time[timestamps[index]] for index in range(0, 5)] == [0, 0, 0, 0, 0]
    assert bool(labeled.loc[labeled["timestamp"].eq(timestamps[10]), "is_failure_period"].iloc[0])
    assert not bool(
        labeled.loc[labeled["timestamp"].eq(timestamps[10]), "eligible_for_prediction"].iloc[0]
    )
    assert target_by_time[timestamps[10]] == 0


def test_metropt_causal_features_use_past_and_current_values_only() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=8, freq="min")
    labeled = _metropt_fixture(timestamps).assign(
        target=0,
        future_failure_event_id="",
        is_failure_period=False,
        eligible_for_prediction=True,
    )

    features = build_metropt_causal_features(
        labeled,
        continuous_columns=CONTINUOUS_SENSORS,
        state_columns=BINARY_STATE_COLUMNS,
        lag_steps=(1,),
        rolling_windows=("3min",),
        trend_window="3min",
        min_periods=1,
    )

    source = labeled.set_index("timestamp")
    row = features.set_index("timestamp").loc[timestamps[2]]
    assert row["TP2__lag_1"] == pytest.approx(float(source.loc[timestamps[1], "TP2"]))
    assert row["TP2__rolling_mean_3min"] == pytest.approx(
        float(source.loc[timestamps[:3], "TP2"].mean())
    )
    assert row["TP2__current"] == pytest.approx(float(source.loc[timestamps[2], "TP2"]))


def test_prepare_metropt_modeling_frame_assigns_event_aware_splits() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=65, freq="min")
    events = (
        _event("train-event", "2020-01-01 00:15:00", "2020-01-01 00:16:00"),
        _event("validation-event", "2020-01-01 00:30:00", "2020-01-01 00:31:00"),
        _event("test-event", "2020-01-01 00:50:00", "2020-01-01 00:51:00"),
    )
    labeled = build_metropt_failure_target(
        _metropt_fixture(timestamps),
        failure_events=events,
        horizon_hours=6 / 60,
    )
    features = build_metropt_causal_features(
        labeled,
        continuous_columns=CONTINUOUS_SENSORS,
        state_columns=BINARY_STATE_COLUMNS,
        lag_steps=(1,),
        rolling_windows=("3min",),
        trend_window="3min",
        min_periods=1,
    )

    modeling = prepare_metropt_modeling_frame(
        features,
        MetroPTExperimentConfig(
            horizon_hours=6 / 60,
            prediction_stride_rows=1,
            train_end="2020-01-01 00:20:00",
            validation_end="2020-01-01 00:40:00",
            lag_steps=(1,),
            rolling_windows=("3min",),
            trend_window="3min",
            min_periods=1,
            failure_events=events,
        ),
    )

    assert set(modeling["split"].unique().tolist()) == {"train", "validation", "test"}
    assert not modeling["is_failure_period"].any()
    positive_counts = modeling.groupby("split", observed=True)["target"].sum().to_dict()
    assert positive_counts == {"test": 6, "train": 6, "validation": 6}


def test_sample_temporal_training_rows_keeps_positives_and_limits_negatives() -> None:
    timestamps = pd.date_range("2020-01-01", periods=25, freq="min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "target": [1] * 5 + [0] * 20,
        }
    )

    sampled = sample_temporal_training_rows(
        frame,
        target_column="target",
        max_rows=10,
        negative_to_positive_ratio=2,
    )

    assert int(sampled["target"].sum()) == 5
    assert sampled.shape[0] == 10
    assert sampled["timestamp"].is_monotonic_increasing


def test_optimize_threshold_uses_validation_grid() -> None:
    selection = optimize_threshold(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.05, 0.40, 0.45, 0.90]),
        thresholds=(0.3, 0.5),
    )

    assert selection.threshold == 0.3
    assert selection.validation_recall == pytest.approx(1.0)


def test_classification_metrics_include_confusion_and_auc_values() -> None:
    metrics = classification_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 1, 0, 1]),
        np.asarray([0.1, 0.8, 0.2, 0.9]),
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positive_rate == pytest.approx(0.5)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.pr_auc is not None


def test_early_warning_metrics_measure_lead_time_and_false_alarms() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=14, freq="min")
    event = _event("event-1", "2020-01-01 00:10:00", "2020-01-01 00:12:00")
    probabilities = np.zeros(timestamps.shape[0], dtype=float)
    probabilities[6] = 0.9
    probabilities[13] = 0.8
    target = np.zeros(timestamps.shape[0], dtype=int)
    target[5:10] = 1
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "probability": probabilities,
            "target": target,
        }
    )

    summary, episodes = compute_early_warning_metrics(
        predictions,
        failure_events=(event,),
        horizon_hours=5 / 60,
        threshold=0.5,
    )

    assert summary.warning_coverage == pytest.approx(1.0)
    assert summary.missed_events == 0
    assert summary.false_warnings == 1
    assert summary.median_lead_time_hours == pytest.approx(4 / 60)
    assert bool(episodes.loc[0, "warning_produced"])


def test_final_tree_selection_uses_validation_metrics_only() -> None:
    chosen = _select_final_tree_model(
        [
            {
                "model": "random_forest",
                "validation_f1": 0.8,
                "threshold_selection": {
                    "validation_precision": 0.7,
                    "validation_recall": 0.9,
                    "validation_false_positive_rate": 0.1,
                },
                "test_metrics": {"pr_auc": 0.1},
            },
            {
                "model": "xgboost",
                "validation_f1": 0.7,
                "threshold_selection": {
                    "validation_precision": 0.95,
                    "validation_recall": 0.95,
                    "validation_false_positive_rate": 0.0,
                },
                "test_metrics": {"pr_auc": 0.9},
            },
        ]
    )

    assert chosen["model"] == "random_forest"


def test_ai4i_experiment_writes_metrics_and_model_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "ai4i"
    raw_dir.mkdir(parents=True)
    _ai4i_fixture(rows=80).to_csv(raw_dir / "ai4i2020.csv", index=False)
    config = FailurePredictionExperimentConfig(
        random_seed=7,
        ai4i=AI4IExperimentConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            threshold_grid=(0.3, 0.5),
        ),
        models=FailureModelConfig(
            random_seed=7,
            logistic_max_iter=200,
            random_forest_estimators=5,
            random_forest_min_samples_leaf=1,
            xgboost_estimators=5,
            xgboost_max_depth=2,
            xgboost_learning_rate=0.1,
            n_jobs=1,
        ),
        shap=SHAPConfig(enabled=False),
    )

    result = run_ai4i_failure_prediction(tmp_path, tmp_path / "experiments", config)

    metrics = pd.read_csv(result.output_dir / "metrics.csv")
    assert {"logistic_regression", "random_forest"}.issubset(set(metrics["model"].tolist()))
    assert (result.output_dir / "models" / "logistic_regression.joblib").exists()
    assert (result.output_dir / "predictions.parquet").exists() or (
        result.output_dir / "predictions.csv.gz"
    ).exists()
    assert result.metrics["failure_mode_columns_excluded_from_features"] == list(FAILURE_COLUMNS)


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


def _ai4i_fixture(*, rows: int) -> pd.DataFrame:
    index = np.arange(rows)
    target = (index % 8 == 0).astype(int)
    frame = pd.DataFrame(
        {
            "UDI": index + 1,
            "Product ID": [f"PID-{value:04d}" for value in index],
            "Type": np.where(index % 3 == 0, "H", np.where(index % 3 == 1, "M", "L")),
            "Air temperature [K]": 298.0 + (index % 5) * 0.1,
            "Process temperature [K]": 308.0 + (index % 7) * 0.1,
            "Rotational speed [rpm]": 1500 - target * 120 + (index % 11),
            "Torque [Nm]": 40.0 + target * 8.0 + (index % 4),
            "Tool wear [min]": 20 + index % 180,
            "Machine failure": target,
            "TWF": target,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )
    return frame.loc[:, ["UDI", "Product ID", "Type", *NUMERIC_FEATURES, *FAILURE_COLUMNS]]
