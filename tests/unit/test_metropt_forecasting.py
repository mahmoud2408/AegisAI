from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis_ai.data.adapters.metropt import BINARY_STATE_COLUMNS, CONTINUOUS_SENSORS
from aegis_ai.forecasting.metropt import (
    ForecastWindowConfig,
    LSTMForecastingConfig,
    MetroPTForecastingExperimentConfig,
    assign_temporal_splits_and_segments,
    audit_window_sets,
    compute_forecasting_metrics,
    fit_forecasting_scaler,
    horizon_minutes_to_steps,
    load_metropt_forecasting_config,
    make_forecasting_windows,
    moving_average_forecast,
    persistence_forecast,
    prepare_metropt_forecasting_frame,
    rolling_linear_trend_forecast,
    train_multivariate_lstm,
    transform_forecasting_frame,
)


def test_temporal_splits_and_segments_break_on_gaps() -> None:
    timestamps = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:01:00",
            "2020-01-01 00:02:00",
            "2020-01-01 00:10:00",
            "2020-01-01 00:11:00",
            "2020-01-01 00:20:00",
        ]
    )
    frame = pd.DataFrame({"timestamp": timestamps, "TP2": np.arange(len(timestamps))})

    split = assign_temporal_splits_and_segments(
        frame,
        train_end=pd.Timestamp("2020-01-01 00:05:00"),
        validation_end=pd.Timestamp("2020-01-01 00:15:00"),
        frequency="1min",
    )

    assert split["split"].tolist() == [
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
    ]
    assert split["segment_id"].tolist() == [0, 0, 0, 1, 1, 2]


def test_forecasting_windows_use_exact_future_horizon_without_crossing_gap() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=10, freq="min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "segment_id": [0] * 5 + [1] * 5,
            "TP2": np.arange(10, dtype=float),
            "TP3": np.arange(10, dtype=float) + 100.0,
        }
    )

    windows = make_forecasting_windows(
        frame,
        value_columns=("TP2", "TP3"),
        config=ForecastWindowConfig(sequence_length=3, horizon_steps=2, stride=1),
    )

    assert windows.inputs.shape == (2, 3, 2)
    assert windows.targets[:, 0].tolist() == [4.0, 9.0]
    assert windows.input_end_timestamps == [timestamps[2], timestamps[7]]
    assert windows.target_timestamps == [timestamps[4], timestamps[9]]


def test_window_audit_confirms_timestamp_alignment() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=8, freq="min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "segment_id": 0,
            "TP2": np.arange(8, dtype=float),
        }
    )
    windows = make_forecasting_windows(
        frame,
        value_columns=("TP2",),
        config=ForecastWindowConfig(sequence_length=3, horizon_steps=2, stride=1),
    )

    audit = audit_window_sets(
        {"train": windows, "validation": windows, "test": windows},
        horizon_minutes=2,
        horizon_steps=2,
        expected_frequency="1min",
    )

    assert audit["leakage_detected"] is False
    assert audit["splits"]["train"]["alignment_errors"] == 0


def test_scaler_is_fit_on_training_rows_only() -> None:
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
            "TP2": [1.0, 2.0, 3.0],
            "TP3": [10.0, 20.0, 30.0],
        }
    )
    validation = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-02", periods=2, freq="min"),
            "TP2": [100.0, 200.0],
            "TP3": [1000.0, 2000.0],
        }
    )
    frame = pd.concat([train.assign(split="train"), validation.assign(split="validation")])

    scaler = fit_forecasting_scaler(train, ("TP2", "TP3"))
    transformed = transform_forecasting_frame(frame, ("TP2", "TP3"), scaler)

    assert scaler.mean_.tolist() == pytest.approx([2.0, 20.0])
    assert transformed.loc[transformed["split"].eq("validation"), "TP2"].mean() > 100


def test_baseline_predictions_and_metrics_have_expected_shape() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=8, freq="min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "segment_id": 0,
            "TP2": np.arange(8, dtype=float),
            "TP3": np.arange(8, dtype=float) * 2,
        }
    )
    windows = make_forecasting_windows(
        frame,
        value_columns=("TP2", "TP3"),
        config=ForecastWindowConfig(sequence_length=3, horizon_steps=1, stride=1),
    )

    persistence = persistence_forecast(windows)
    moving = moving_average_forecast(windows, window_steps=2)
    trend = rolling_linear_trend_forecast(windows, window_steps=3, horizon_steps=1)
    metrics = compute_forecasting_metrics(
        y_true=windows.targets,
        y_pred=trend,
        sensors=("TP2", "TP3"),
        split="test",
        model="rolling_linear_trend",
        horizon_minutes=1,
        runtime_seconds=0.1,
        training_seconds=0.0,
        parameter_count=0,
        mape_min_abs_value=0.1,
    )

    assert persistence.shape == windows.targets.shape
    assert moving.shape == windows.targets.shape
    assert trend.shape == windows.targets.shape
    assert set(metrics["sensor"].tolist()) == {"__global__", "TP2", "TP3"}
    assert float(metrics.loc[metrics["sensor"].eq("__global__"), "rmse"].iloc[0]) == pytest.approx(
        0.0
    )


def test_prepare_forecasting_frame_excludes_binary_targets_and_keeps_splits() -> None:
    raw = _metropt_fixture(pd.date_range("2020-01-01 00:00:00", periods=120, freq="10s"))
    config = MetroPTForecastingExperimentConfig(
        selected_sensors=("TP2", "TP3"),
        horizons_minutes=(1,),
        train_end="2020-01-01 00:06:00",
        validation_end="2020-01-01 00:12:00",
        small_gap_interpolate_limit=0,
        lstm=LSTMForecastingConfig(sequence_length=3, epochs=1),
    )

    frame, audit = prepare_metropt_forecasting_frame(raw, config)

    assert audit["selected_sensors"] == ["TP2", "TP3"]
    assert "COMP" in audit["binary_state_columns"]
    assert set(frame["split"].unique().tolist()) == {"train", "validation", "test"}
    assert "COMP" not in frame.columns


def test_tiny_multivariate_lstm_prediction_shape() -> None:
    timestamps = pd.date_range("2020-01-01 00:00:00", periods=20, freq="min")
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "segment_id": 0,
            "split": ["train"] * 10 + ["validation"] * 5 + ["test"] * 5,
            "TP2": np.sin(np.arange(20, dtype=float) / 4.0) + 2.0,
            "TP3": np.cos(np.arange(20, dtype=float) / 5.0) + 5.0,
        }
    )
    scaler = fit_forecasting_scaler(base[base["split"].eq("train")], ("TP2", "TP3"))
    scaled = transform_forecasting_frame(base, ("TP2", "TP3"), scaler)
    windows = {
        split: make_forecasting_windows(
            scaled[scaled["split"].eq(split)],
            value_columns=("TP2", "TP3"),
            config=ForecastWindowConfig(sequence_length=3, horizon_steps=1, stride=1),
        )
        for split in ("train", "validation", "test")
    }

    result = train_multivariate_lstm(
        windows,
        scaler=scaler,
        horizon_minutes=1,
        sensors=("TP2", "TP3"),
        config=LSTMForecastingConfig(
            sequence_length=3,
            hidden_size=4,
            batch_size=2,
            epochs=1,
            patience=1,
            max_train_windows=4,
            max_validation_windows=2,
        ),
        random_seed=7,
        output_path=Path("unused_tiny_lstm.pt"),
    )

    assert result.predictions["test"].shape == (2, 2)
    Path("unused_tiny_lstm.pt").unlink(missing_ok=True)


def test_config_loader_parses_phase8_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "resample_frequency: 1min",
                "selected_sensors: [TP2, Oil_temperature]",
                "horizons_minutes: [5, 30]",
                "window_stride: 2",
                "lstm:",
                "  enabled: false",
                "  sequence_length: 12",
            ]
        ),
        encoding="utf-8",
    )

    config = load_metropt_forecasting_config(config_path)

    assert config.selected_sensors == ("TP2", "Oil_temperature")
    assert config.horizons_minutes == (5, 30)
    assert config.window_stride == 2
    assert config.lstm.enabled is False
    assert horizon_minutes_to_steps(5, "1min") == 5


def _metropt_fixture(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    row_count = timestamps.shape[0]
    rows: dict[str, object] = {"timestamp": timestamps}
    base = np.arange(row_count, dtype=float)
    for offset, column in enumerate(CONTINUOUS_SENSORS):
        rows[column] = base + float(offset)
    for offset, column in enumerate(BINARY_STATE_COLUMNS):
        rows[column] = ((base // 4 + offset) % 2).astype(float)
    return pd.DataFrame(rows)
