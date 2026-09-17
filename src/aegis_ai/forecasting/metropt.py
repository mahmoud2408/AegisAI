"""MetroPT-3 time-series forecasting experiment utilities."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aegis_ai.data.adapters.metropt import BINARY_STATE_COLUMNS, CONTINUOUS_SENSORS
from aegis_ai.data.dataset_registry import project_root
from aegis_ai.features.normalization import TimeSeriesStandardScaler
from aegis_ai.ml.prediction.failure_prediction import (
    FailureEvent,
    MetroPTExperimentConfig,
    _active_failure_mask,
    _duration_days,
    _save_json,
    _write_dataframe,
    audit_metropt_frame,
    load_metropt_frame,
)

DEFAULT_FORECAST_SENSORS = (
    "TP2",
    "TP3",
    "H1",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)
DEFAULT_FORECAST_HORIZONS_MINUTES = (5, 15, 30)

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class ForecastWindowConfig:
    """Configuration for split-local forecasting windows."""

    sequence_length: int = 60
    horizon_steps: int = 5
    stride: int = 5

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")


@dataclass(frozen=True)
class ForecastWindowSet:
    """Forecasting inputs, future targets, and timestamp alignment metadata."""

    inputs: np.ndarray
    targets: np.ndarray
    input_start_timestamps: list[pd.Timestamp]
    input_end_timestamps: list[pd.Timestamp]
    target_timestamps: list[pd.Timestamp]
    segment_ids: np.ndarray
    start_indices: np.ndarray
    end_indices: np.ndarray
    target_indices: np.ndarray
    feature_columns: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return bool(self.inputs.shape[0] == 0)


@dataclass(frozen=True)
class LSTMForecastingConfig:
    """Training configuration for lightweight LSTM forecasting models."""

    enabled: bool = True
    sequence_length: int = 60
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 5
    patience: int = 2
    max_train_windows: int = 12000
    max_validation_windows: int = 4000
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.max_train_windows < 1:
            raise ValueError("max_train_windows must be positive")
        if self.max_validation_windows < 1:
            raise ValueError("max_validation_windows must be positive")


@dataclass(frozen=True)
class MetroPTForecastingExperimentConfig:
    """Configuration for Phase 8 MetroPT forecasting."""

    random_seed: int = 42
    resample_frequency: str = "1min"
    selected_sensors: tuple[str, ...] = DEFAULT_FORECAST_SENSORS
    horizons_minutes: tuple[int, ...] = DEFAULT_FORECAST_HORIZONS_MINUTES
    train_end: str = "2020-05-31 00:00:00"
    validation_end: str = "2020-06-08 00:00:00"
    window_stride: int = 5
    moving_average_window_steps: int = 15
    statistical_window_steps: int = 30
    ridge_lag_steps: int = 30
    normal_quantile_low: float = 0.01
    normal_quantile_high: float = 0.99
    normal_z_threshold: float = 3.0
    mape_min_abs_value: float = 0.1
    small_gap_interpolate_limit: int = 2
    max_gap_multiplier: float = 1.5
    max_saved_prediction_rows_per_model: int = 3000
    failure_events: tuple[FailureEvent, ...] = field(
        default_factory=lambda: MetroPTExperimentConfig().failure_events
    )
    lstm: LSTMForecastingConfig = field(default_factory=LSTMForecastingConfig)

    def __post_init__(self) -> None:
        if not self.selected_sensors:
            raise ValueError("selected_sensors must not be empty")
        invalid = sorted(set(self.selected_sensors).difference(CONTINUOUS_SENSORS))
        if invalid:
            raise ValueError(f"selected_sensors must be continuous MetroPT sensors: {invalid}")
        if not self.horizons_minutes:
            raise ValueError("horizons_minutes must not be empty")
        if any(horizon <= 0 for horizon in self.horizons_minutes):
            raise ValueError("horizons_minutes must contain positive values")
        if tuple(sorted(self.horizons_minutes)) != self.horizons_minutes:
            raise ValueError("horizons_minutes must be sorted from shortest to longest")
        if self.window_stride < 1:
            raise ValueError("window_stride must be positive")
        if self.moving_average_window_steps < 1:
            raise ValueError("moving_average_window_steps must be positive")
        if self.statistical_window_steps < 2:
            raise ValueError("statistical_window_steps must be at least 2")
        if self.ridge_lag_steps < 1:
            raise ValueError("ridge_lag_steps must be positive")
        if pd.Timestamp(self.train_end) >= pd.Timestamp(self.validation_end):
            raise ValueError("train_end must be before validation_end")
        if not 0 < self.normal_quantile_low < self.normal_quantile_high < 1:
            raise ValueError("normal quantiles must satisfy 0 < low < high < 1")
        if self.normal_z_threshold <= 0:
            raise ValueError("normal_z_threshold must be positive")
        if self.mape_min_abs_value < 0:
            raise ValueError("mape_min_abs_value must be non-negative")
        if self.small_gap_interpolate_limit < 0:
            raise ValueError("small_gap_interpolate_limit must be non-negative")
        if self.max_gap_multiplier <= 1:
            raise ValueError("max_gap_multiplier must be greater than 1")
        if self.max_saved_prediction_rows_per_model < 1:
            raise ValueError("max_saved_prediction_rows_per_model must be positive")


@dataclass(frozen=True)
class MetroPTForecastingExperimentResult:
    """Location and summary for a completed Phase 8 forecasting run."""

    run_dir: Path
    metrics: dict[str, Any]


class LSTMForecaster(nn.Module):
    """Small sequence-to-vector LSTM forecaster."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return cast(torch.Tensor, self.head(hidden[-1]))


def load_metropt_forecasting_config(path: Path) -> MetroPTForecastingExperimentConfig:
    """Load Phase 8 forecasting config from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a YAML mapping")
    default = MetroPTForecastingExperimentConfig()
    events_payload = payload.get("failure_events")
    events = (
        tuple(FailureEvent(**event) for event in events_payload)
        if events_payload is not None
        else default.failure_events
    )
    return MetroPTForecastingExperimentConfig(
        random_seed=int(payload.get("random_seed", default.random_seed)),
        resample_frequency=str(payload.get("resample_frequency", default.resample_frequency)),
        selected_sensors=_tuple_strings(
            payload.get("selected_sensors"),
            default.selected_sensors,
        ),
        horizons_minutes=_tuple_ints(
            payload.get("horizons_minutes"),
            default.horizons_minutes,
        ),
        train_end=str(payload.get("train_end", default.train_end)),
        validation_end=str(payload.get("validation_end", default.validation_end)),
        window_stride=int(payload.get("window_stride", default.window_stride)),
        moving_average_window_steps=int(
            payload.get("moving_average_window_steps", default.moving_average_window_steps)
        ),
        statistical_window_steps=int(
            payload.get("statistical_window_steps", default.statistical_window_steps)
        ),
        ridge_lag_steps=int(payload.get("ridge_lag_steps", default.ridge_lag_steps)),
        normal_quantile_low=float(payload.get("normal_quantile_low", default.normal_quantile_low)),
        normal_quantile_high=float(
            payload.get("normal_quantile_high", default.normal_quantile_high)
        ),
        normal_z_threshold=float(payload.get("normal_z_threshold", default.normal_z_threshold)),
        mape_min_abs_value=float(payload.get("mape_min_abs_value", default.mape_min_abs_value)),
        small_gap_interpolate_limit=int(
            payload.get("small_gap_interpolate_limit", default.small_gap_interpolate_limit)
        ),
        max_gap_multiplier=float(payload.get("max_gap_multiplier", default.max_gap_multiplier)),
        max_saved_prediction_rows_per_model=int(
            payload.get(
                "max_saved_prediction_rows_per_model",
                default.max_saved_prediction_rows_per_model,
            )
        ),
        failure_events=events,
        lstm=LSTMForecastingConfig(**_mapping(payload.get("lstm"))),
    )


def run_metropt_forecasting_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: MetroPTForecastingExperimentConfig | None = None,
) -> MetroPTForecastingExperimentResult:
    """Run the Phase 8 MetroPT forecasting experiment."""

    cfg = config or MetroPTForecastingExperimentConfig()
    _set_reproducible_seed(cfg.random_seed)
    project = project_root() if root is None else root
    resolved_run_id = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    base_output = output_root or project / "experiments" / "forecasting" / "metropt"
    run_dir = base_output / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    models_dir = run_dir / "models"
    for directory in (figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "config.json", asdict(cfg))

    started = time.perf_counter()
    raw = load_metropt_frame(project)
    raw_audit = audit_metropt_frame(raw, failure_events=cfg.failure_events)
    frame, data_audit = prepare_metropt_forecasting_frame(raw, cfg)
    _save_json(run_dir / "raw_dataset_audit.json", raw_audit)
    _save_json(run_dir / "forecasting_data_audit.json", data_audit)

    split_summary = split_segment_summary(frame)
    split_summary.to_csv(run_dir / "split_segment_summary.csv", index=False)
    _plot_sensor_overview(frame, cfg.selected_sensors, figures_dir / "sensor_overview.png")

    train_frame = frame[frame["split"].eq("train")].copy()
    scaler = fit_forecasting_scaler(train_frame, cfg.selected_sensors)
    _save_json(run_dir / "scaler.json", scaler.to_dict())
    scaled_frame = transform_forecasting_frame(frame, cfg.selected_sensors, scaler)
    normal_ranges = compute_normal_ranges(
        train_frame,
        sensors=cfg.selected_sensors,
        failure_events=cfg.failure_events,
        low_quantile=cfg.normal_quantile_low,
        high_quantile=cfg.normal_quantile_high,
    )
    normal_ranges.to_csv(run_dir / "normal_ranges.csv", index=False)

    metrics_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    training_rows: list[dict[str, Any]] = []
    risk_frames: list[pd.DataFrame] = []

    for horizon_minutes in cfg.horizons_minutes:
        horizon_started = time.perf_counter()
        horizon_steps = horizon_minutes_to_steps(horizon_minutes, cfg.resample_frequency)
        horizon_dir = run_dir / f"horizon_{horizon_minutes}min"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        original_windows = {
            split: make_forecasting_windows(
                frame[frame["split"].eq(split)].copy(),
                value_columns=cfg.selected_sensors,
                config=ForecastWindowConfig(
                    sequence_length=cfg.lstm.sequence_length,
                    horizon_steps=horizon_steps,
                    stride=cfg.window_stride,
                ),
            )
            for split in ("train", "validation", "test")
        }
        scaled_windows = {
            split: make_forecasting_windows(
                scaled_frame[scaled_frame["split"].eq(split)].copy(),
                value_columns=cfg.selected_sensors,
                config=ForecastWindowConfig(
                    sequence_length=cfg.lstm.sequence_length,
                    horizon_steps=horizon_steps,
                    stride=cfg.window_stride,
                ),
            )
            for split in ("train", "validation", "test")
        }
        window_audit = audit_window_sets(
            original_windows,
            horizon_minutes=horizon_minutes,
            horizon_steps=horizon_steps,
            expected_frequency=cfg.resample_frequency,
        )
        _save_json(horizon_dir / "window_audit.json", window_audit)

        horizon_metrics, horizon_predictions, horizon_training = _run_horizon_models(
            original_windows=original_windows,
            scaled_windows=scaled_windows,
            scaler=scaler,
            normal_ranges=normal_ranges,
            config=cfg,
            horizon_minutes=horizon_minutes,
            horizon_steps=horizon_steps,
            models_dir=models_dir,
        )
        metrics_frames.append(horizon_metrics)
        prediction_frames.append(horizon_predictions)
        training_rows.extend(horizon_training)
        horizon_validation_global = horizon_metrics[
            horizon_metrics["split"].eq("validation") & horizon_metrics["sensor"].eq("__global__")
        ]
        best_horizon_record = select_best_forecasting_candidate(horizon_validation_global)
        risk = forecast_risk_signal(
            horizon_predictions[
                horizon_predictions["model"].eq(best_horizon_record["model"])
                & horizon_predictions["split"].eq("test")
            ],
            normal_ranges=normal_ranges,
            horizon_minutes=horizon_minutes,
            model=str(best_horizon_record["model"]),
        )
        risk_frames.append(risk)
        _plot_actual_vs_predicted(
            horizon_predictions,
            horizon_minutes=horizon_minutes,
            model=str(best_horizon_record["model"]),
            sensor=str(best_horizon_record["sensor"])
            if best_horizon_record["sensor"] != "__global__"
            else cfg.selected_sensors[0],
            output_path=figures_dir / f"actual_vs_predicted_{horizon_minutes}min.png",
        )
        _plot_difficult_cases(
            horizon_predictions,
            horizon_minutes=horizon_minutes,
            model=str(best_horizon_record["model"]),
            output_path=figures_dir / f"difficult_cases_{horizon_minutes}min.png",
        )
        _save_json(
            horizon_dir / "runtime.json",
            {
                "horizon_minutes": horizon_minutes,
                "horizon_steps": horizon_steps,
                "elapsed_seconds": float(time.perf_counter() - horizon_started),
            },
        )

    metrics = pd.concat(metrics_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    training = pd.DataFrame(training_rows)
    risk_signals = pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame()

    metrics.to_csv(run_dir / "metrics.csv", index=False)
    training.to_csv(run_dir / "training_runtime.csv", index=False)
    _write_dataframe(predictions, run_dir / "forecasts")
    risk_signals.to_csv(run_dir / "forecast_risk_signal.csv", index=False)

    global_metrics = metrics[metrics["sensor"].eq("__global__")].copy()
    best = select_best_forecasting_candidate(global_metrics[global_metrics["split"].eq("test")])
    validation_best = select_best_forecasting_candidate(
        global_metrics[global_metrics["split"].eq("validation")]
    )
    validation_selected_test = _matching_test_candidate(global_metrics, validation_best)
    per_sensor = metrics[metrics["split"].eq("test") & ~metrics["sensor"].eq("__global__")]
    hardest = (
        per_sensor.sort_values(["rmse", "mae"], ascending=False).head(10).to_dict(orient="records")
    )
    easiest = (
        per_sensor.sort_values(["rmse", "mae"], ascending=True).head(10).to_dict(orient="records")
    )
    risk_summary = summarize_risk_signal(risk_signals)
    _save_json(run_dir / "risk_signal_summary.json", risk_summary)
    _plot_metrics(metrics, figures_dir / "forecast_metrics.png")
    _plot_per_sensor_error(metrics, figures_dir / "per_sensor_error.png")
    _plot_risk_signal(risk_signals, figures_dir / "forecast_risk_signal.png")

    summary = {
        "run_id": resolved_run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "research_question": (
            "How accurately can AegisAI forecast future MetroPT machine telemetry, "
            "and how does horizon length affect forecast-based degradation signals?"
        ),
        "sampling_interval_seconds": data_audit["resampled_cadence_seconds"],
        "raw_median_cadence_seconds": data_audit["raw_median_cadence_seconds"],
        "selected_sensors": list(cfg.selected_sensors),
        "excluded_continuous_sensors": data_audit["excluded_continuous_sensors"],
        "horizons_minutes": list(cfg.horizons_minutes),
        "models": sorted(metrics["model"].unique().tolist()),
        "best_test_global": best,
        "best_validation_global": validation_best,
        "validation_selected_test_global": validation_selected_test,
        "per_sensor_hardest_test_cases": hardest,
        "per_sensor_easiest_test_cases": easiest,
        "training_runtime": training.to_dict(orient="records"),
        "risk_signal": risk_summary,
        "transformer": {
            "implemented": False,
            "reason": (
                "Excluded in Phase 8 because persistence, rolling statistical baselines, "
                "ridge autoregression, and LSTM baselines are the required first evidence "
                "layer. A Transformer would add compute before the lower-complexity "
                "models are fully interpreted."
            ),
        },
        "uncertainty": {
            "implemented": False,
            "reason": (
                "Prediction intervals are documented as a limitation. Phase 8 stores "
                "point forecasts and normal-range risk comparisons only."
            ),
        },
    }
    _save_json(run_dir / "metrics.json", summary)
    _write_run_report(run_dir, summary=summary, metrics=metrics, training=training)
    return MetroPTForecastingExperimentResult(run_dir=run_dir, metrics=summary)


def prepare_metropt_forecasting_frame(
    raw: pd.DataFrame,
    config: MetroPTForecastingExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resample raw MetroPT telemetry to a regular grid and add split/segment metadata."""

    selected_sensors = validate_forecasting_sensors(raw, config.selected_sensors)
    ordered = raw.sort_values("timestamp").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(ordered["timestamp"], errors="raise")
    raw_deltas = timestamps.diff().dropna().dt.total_seconds()
    indexed = ordered.set_index("timestamp", drop=False)
    continuous = indexed.loc[:, list(selected_sensors)].apply(pd.to_numeric, errors="coerce")
    resampled = continuous.resample(config.resample_frequency).median()
    missing_before = resampled.isna()
    if config.small_gap_interpolate_limit > 0:
        resampled = resampled.interpolate(
            method="time",
            limit=config.small_gap_interpolate_limit,
            limit_area="inside",
        )
    imputed = missing_before & resampled.notna()
    valid = resampled.dropna(axis=0, how="any").reset_index()
    valid = valid.rename(columns={"index": "timestamp"})
    if "timestamp" not in valid.columns:
        valid = valid.rename(columns={valid.columns[0]: "timestamp"})
    valid["timestamp"] = pd.to_datetime(valid["timestamp"], errors="raise")
    valid = assign_temporal_splits_and_segments(
        valid,
        train_end=pd.Timestamp(config.train_end),
        validation_end=pd.Timestamp(config.validation_end),
        frequency=config.resample_frequency,
        max_gap_multiplier=config.max_gap_multiplier,
    )
    raw_gap_count = int((raw_deltas > config.max_gap_multiplier * float(raw_deltas.median())).sum())
    resampled_deltas = valid["timestamp"].diff().dropna().dt.total_seconds()
    coverage_minutes = int(
        pd.date_range(
            valid["timestamp"].min(),
            valid["timestamp"].max(),
            freq=config.resample_frequency,
        ).shape[0]
    )
    sensor_audit = sensor_quality_table(raw, resampled, selected_sensors)
    excluded = [
        sensor
        for sensor in CONTINUOUS_SENSORS
        if sensor not in selected_sensors and sensor in raw.columns
    ]
    audit = {
        "raw_rows": int(raw.shape[0]),
        "raw_timestamp_start": timestamps.min().isoformat(),
        "raw_timestamp_end": timestamps.max().isoformat(),
        "raw_median_cadence_seconds": float(raw_deltas.median()),
        "raw_dominant_cadence_seconds": {
            str(key): int(value) for key, value in raw_deltas.value_counts().head(10).items()
        },
        "raw_gaps_gt_expected": raw_gap_count,
        "raw_missing_values_selected_sensors": int(continuous.isna().sum().sum()),
        "raw_duplicate_timestamps": int(timestamps.duplicated().sum()),
        "resample_frequency": config.resample_frequency,
        "resampled_cadence_seconds": frequency_to_seconds(config.resample_frequency),
        "resampled_grid_rows": int(resampled.shape[0]),
        "resampled_valid_rows": int(valid.shape[0]),
        "resampled_coverage_minutes": coverage_minutes,
        "resampled_missing_values_before_interpolation": int(missing_before.sum().sum()),
        "resampled_missing_rows_before_interpolation": int(missing_before.any(axis=1).sum()),
        "resampled_values_interpolated": int(imputed.sum().sum()),
        "resampled_rows_after_dropna": int(valid.shape[0]),
        "resampled_gap_count_after_dropna": int(
            (
                resampled_deltas
                > config.max_gap_multiplier * frequency_to_seconds(config.resample_frequency)
            ).sum()
        ),
        "selected_sensors": list(selected_sensors),
        "excluded_continuous_sensors": excluded,
        "excluded_sensor_reason": (
            "Phase 8 keeps a compact set of continuous, high-value compressor sensors. "
            "Binary state columns are not forecast with continuous-regression models; "
            "DV_pressure is retained for audit/risk context but excluded from the first "
            "target set because it is highly intermittent and spike-oriented."
        ),
        "binary_state_columns": list(BINARY_STATE_COLUMNS),
        "constant_selected_sensors": [
            sensor for sensor in selected_sensors if raw[sensor].nunique(dropna=False) <= 1
        ],
        "split_counts": valid["split"].value_counts().sort_index().to_dict(),
        "segment_count": int(valid["segment_id"].nunique()),
        "sensor_quality": sensor_audit.to_dict(orient="records"),
        "state_summary": state_summary(raw),
    }
    return valid.reset_index(drop=True), audit


def validate_forecasting_sensors(
    raw: pd.DataFrame,
    sensors: tuple[str, ...],
    *,
    min_unique_values: int = 10,
    min_std: float = 1e-6,
) -> tuple[str, ...]:
    """Validate continuous MetroPT forecasting targets against actual data quality."""

    missing = sorted(set(sensors).difference(raw.columns))
    if missing:
        raise ValueError(f"raw MetroPT frame missing selected sensors: {missing}")
    invalid = sorted(set(sensors).difference(CONTINUOUS_SENSORS))
    if invalid:
        raise ValueError(f"forecasting targets must be continuous sensors: {invalid}")
    accepted: list[str] = []
    rejected: list[str] = []
    for sensor in sensors:
        values = pd.to_numeric(raw[sensor], errors="coerce")
        too_few_values = int(values.nunique(dropna=True)) < min_unique_values
        too_little_variance = float(values.std(ddof=0)) <= min_std
        if too_few_values or too_little_variance:
            rejected.append(sensor)
        else:
            accepted.append(sensor)
    if rejected:
        raise ValueError(f"selected sensors are not forecastable continuous variables: {rejected}")
    return tuple(accepted)


def assign_temporal_splits_and_segments(
    frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    frequency: str,
    max_gap_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Assign strict chronological splits and contiguous segment identifiers."""

    if train_end >= validation_end:
        raise ValueError("train_end must be before validation_end")
    if "timestamp" not in frame.columns:
        raise ValueError("frame must include timestamp")
    result = frame.sort_values("timestamp").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="raise")
    result["split"] = np.select(
        [
            timestamps < train_end,
            (timestamps >= train_end) & (timestamps < validation_end),
        ],
        ["train", "validation"],
        default="test",
    )
    expected_seconds = frequency_to_seconds(frequency)
    deltas = timestamps.diff().dt.total_seconds()
    gap_break = deltas.gt(max_gap_multiplier * expected_seconds).fillna(False)
    split_break = pd.Series(result["split"]).ne(pd.Series(result["split"]).shift()).fillna(True)
    result["segment_id"] = (gap_break | split_break).cumsum().astype(int) - 1
    return result


def make_forecasting_windows(
    frame: pd.DataFrame,
    *,
    value_columns: tuple[str, ...],
    config: ForecastWindowConfig,
    timestamp_column: str = "timestamp",
    segment_column: str = "segment_id",
) -> ForecastWindowSet:
    """Create leakage-safe forecasting windows inside contiguous split segments."""

    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    if segment_column not in frame.columns:
        raise ValueError(f"missing segment column: {segment_column}")
    missing = sorted(set(value_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing value columns: {missing}")
    ordered = frame.sort_values([segment_column, timestamp_column]).copy()
    values_list: list[np.ndarray] = []
    input_start_timestamps: list[pd.Timestamp] = []
    input_end_timestamps: list[pd.Timestamp] = []
    target_timestamps: list[pd.Timestamp] = []
    segment_ids: list[int] = []
    start_indices: list[int] = []
    end_indices: list[int] = []
    target_indices: list[int] = []

    for _, segment in ordered.groupby(segment_column, sort=True, observed=True):
        segment = segment.reset_index(drop=False)
        array = segment.loc[:, list(value_columns)].to_numpy(dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("forecasting window input contains NaN or infinite values")
        timestamps = pd.to_datetime(segment[timestamp_column], errors="raise").reset_index(
            drop=True
        )
        segment_length = array.shape[0]
        last_start = segment_length - config.sequence_length - config.horizon_steps
        if last_start < 0:
            continue
        starts = np.arange(0, last_start + 1, config.stride, dtype=int)
        ends = starts + config.sequence_length - 1
        targets = ends + config.horizon_steps
        values_list.extend(array[start : end + 1] for start, end in zip(starts, ends, strict=True))
        input_start_timestamps.extend(timestamps.iloc[starts].tolist())
        input_end_timestamps.extend(timestamps.iloc[ends].tolist())
        target_timestamps.extend(timestamps.iloc[targets].tolist())
        segment_value = int(segment[segment_column].iloc[0])
        segment_ids.extend([segment_value] * starts.shape[0])
        original_indices = segment["index"].to_numpy(dtype=int)
        start_indices.extend(original_indices[starts].tolist())
        end_indices.extend(original_indices[ends].tolist())
        target_indices.extend(original_indices[targets].tolist())

    feature_count = len(value_columns)
    if not values_list:
        empty_inputs = np.empty((0, config.sequence_length, feature_count), dtype=np.float32)
        empty_targets = np.empty((0, feature_count), dtype=np.float32)
        return ForecastWindowSet(
            inputs=empty_inputs,
            targets=empty_targets,
            input_start_timestamps=[],
            input_end_timestamps=[],
            target_timestamps=[],
            segment_ids=np.empty(0, dtype=int),
            start_indices=np.empty(0, dtype=int),
            end_indices=np.empty(0, dtype=int),
            target_indices=np.empty(0, dtype=int),
            feature_columns=value_columns,
        )

    inputs = np.stack(values_list).astype(np.float32)
    target_values = ordered.loc[
        np.asarray(target_indices, dtype=int),
        list(value_columns),
    ].to_numpy(dtype=np.float32)
    return ForecastWindowSet(
        inputs=inputs,
        targets=target_values,
        input_start_timestamps=input_start_timestamps,
        input_end_timestamps=input_end_timestamps,
        target_timestamps=target_timestamps,
        segment_ids=np.asarray(segment_ids, dtype=int),
        start_indices=np.asarray(start_indices, dtype=int),
        end_indices=np.asarray(end_indices, dtype=int),
        target_indices=np.asarray(target_indices, dtype=int),
        feature_columns=value_columns,
    )


def fit_forecasting_scaler(
    train_frame: pd.DataFrame,
    sensors: tuple[str, ...],
) -> TimeSeriesStandardScaler:
    """Fit a standard scaler on training rows only."""

    values = train_frame.loc[:, list(sensors)].to_numpy(dtype=float)
    return TimeSeriesStandardScaler().fit(values)


def transform_forecasting_frame(
    frame: pd.DataFrame,
    sensors: tuple[str, ...],
    scaler: TimeSeriesStandardScaler,
) -> pd.DataFrame:
    """Transform selected sensor columns while preserving timestamps and split metadata."""

    result = frame.copy()
    transformed = scaler.transform(result.loc[:, list(sensors)].to_numpy(dtype=float))
    result.loc[:, list(sensors)] = transformed
    return result


def horizon_minutes_to_steps(horizon_minutes: int, frequency: str) -> int:
    """Convert a minute horizon to rows for a fixed-frequency frame."""

    seconds = frequency_to_seconds(frequency)
    horizon_seconds = float(horizon_minutes * 60)
    steps = horizon_seconds / seconds
    rounded = int(round(steps))
    if not np.isclose(steps, rounded):
        raise ValueError(
            f"horizon {horizon_minutes} minutes is not an integer number of {frequency} steps"
        )
    if rounded < 1:
        raise ValueError("horizon must be at least one step")
    return rounded


def frequency_to_seconds(frequency: str) -> float:
    """Return the number of seconds represented by a pandas frequency string."""

    seconds = pd.Timedelta(frequency).total_seconds()
    if seconds <= 0:
        raise ValueError("frequency must be positive")
    return float(seconds)


def compute_forecasting_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensors: tuple[str, ...],
    split: str,
    model: str,
    horizon_minutes: int,
    runtime_seconds: float,
    training_seconds: float,
    parameter_count: int | None,
    mape_min_abs_value: float,
) -> pd.DataFrame:
    """Compute global and per-sensor MAE/RMSE/MAPE metrics."""

    true = _as_2d_array(y_true, name="y_true")
    pred = _as_2d_array(y_pred, name="y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.shape[1] != len(sensors):
        raise ValueError("sensor count must match prediction width")

    rows = [
        _metric_row(
            true.ravel(),
            pred.ravel(),
            sensor="__global__",
            split=split,
            model=model,
            horizon_minutes=horizon_minutes,
            runtime_seconds=runtime_seconds,
            training_seconds=training_seconds,
            parameter_count=parameter_count,
            mape_min_abs_value=mape_min_abs_value,
        )
    ]
    for index, sensor in enumerate(sensors):
        rows.append(
            _metric_row(
                true[:, index],
                pred[:, index],
                sensor=sensor,
                split=split,
                model=model,
                horizon_minutes=horizon_minutes,
                runtime_seconds=runtime_seconds,
                training_seconds=training_seconds,
                parameter_count=parameter_count,
                mape_min_abs_value=mape_min_abs_value,
            )
        )
    return pd.DataFrame(rows)


def compute_normal_ranges(
    train_frame: pd.DataFrame,
    *,
    sensors: tuple[str, ...],
    failure_events: tuple[FailureEvent, ...],
    low_quantile: float,
    high_quantile: float,
) -> pd.DataFrame:
    """Estimate train-only normal ranges, excluding curated active failure rows."""

    timestamps = pd.to_datetime(train_frame["timestamp"], errors="raise")
    non_failure = train_frame.loc[~_active_failure_mask(timestamps, failure_events), list(sensors)]
    if non_failure.empty:
        non_failure = train_frame.loc[:, list(sensors)]
    rows = []
    for sensor in sensors:
        values = pd.to_numeric(non_failure[sensor], errors="coerce").dropna()
        rows.append(
            {
                "sensor": sensor,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "low_quantile": float(values.quantile(low_quantile)),
                "high_quantile": float(values.quantile(high_quantile)),
                "low_quantile_level": low_quantile,
                "high_quantile_level": high_quantile,
            }
        )
    return pd.DataFrame(rows)


def forecast_risk_signal(
    predictions: pd.DataFrame,
    *,
    normal_ranges: pd.DataFrame,
    horizon_minutes: int,
    model: str,
) -> pd.DataFrame:
    """Compare forecasts with train-only normal ranges as a degradation/risk signal."""

    if predictions.empty:
        return pd.DataFrame()
    ranges = normal_ranges.set_index("sensor").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for row in predictions.itertuples(index=False):
        sensor = str(row.sensor)
        bounds = ranges[sensor]
        std = max(float(bounds["std"]), 1e-8)
        forecast_z = abs((float(row.y_pred) - float(bounds["mean"])) / std)
        actual_z = abs((float(row.y_true) - float(bounds["mean"])) / std)
        forecast_outside = bool(
            float(row.y_pred) < float(bounds["low_quantile"])
            or float(row.y_pred) > float(bounds["high_quantile"])
        )
        actual_outside = bool(
            float(row.y_true) < float(bounds["low_quantile"])
            or float(row.y_true) > float(bounds["high_quantile"])
        )
        rows.append(
            {
                "model": model,
                "horizon_minutes": horizon_minutes,
                "split": str(row.split),
                "sensor": sensor,
                "input_end_timestamp": row.input_end_timestamp,
                "target_timestamp": row.target_timestamp,
                "forecast_value": float(row.y_pred),
                "actual_value": float(row.y_true),
                "absolute_error": float(row.absolute_error),
                "forecast_outside_train_range": forecast_outside,
                "actual_outside_train_range": actual_outside,
                "forecast_abs_z": forecast_z,
                "actual_abs_z": actual_z,
            }
        )
    return pd.DataFrame(rows)


def summarize_risk_signal(risk: pd.DataFrame) -> dict[str, Any]:
    """Summarize forecast-based normal-range warnings against future out-of-range values."""

    if risk.empty:
        return {"available": False}
    rows: list[dict[str, Any]] = []
    for (horizon, sensor), group in risk.groupby(["horizon_minutes", "sensor"], observed=True):
        y_true = group["actual_outside_train_range"].to_numpy(dtype=bool)
        y_pred = group["forecast_outside_train_range"].to_numpy(dtype=bool)
        true_positives = int((y_true & y_pred).sum())
        false_positives = int((~y_true & y_pred).sum())
        false_negatives = int((y_true & ~y_pred).sum())
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "horizon_minutes": int(horizon),
                "sensor": str(sensor),
                "rows": int(group.shape[0]),
                "forecast_warning_rate": float(y_pred.mean()) if y_pred.size else 0.0,
                "future_out_of_range_rate": float(y_true.mean()) if y_true.size else 0.0,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "mean_forecast_abs_z": float(group["forecast_abs_z"].mean()),
                "mean_actual_abs_z": float(group["actual_abs_z"].mean()),
            }
        )
    summary_frame = pd.DataFrame(rows)
    return {
        "available": True,
        "definition": (
            "Forecast-based degradation/risk signal: predicted future value outside "
            "train non-failure quantile range. This is not a failure-prediction model."
        ),
        "rows": rows,
        "global_mean_precision": float(summary_frame["precision"].mean()),
        "global_mean_recall": float(summary_frame["recall"].mean()),
        "global_mean_f1": float(summary_frame["f1"].mean()),
    }


def select_best_forecasting_candidate(metrics: pd.DataFrame) -> dict[str, Any]:
    """Select the lowest-error candidate from a metrics frame."""

    if metrics.empty:
        raise ValueError("cannot select a candidate from empty metrics")
    sortable = metrics.copy()
    sortable["parameter_sort"] = sortable["parameter_count"].fillna(0)
    chosen = sortable.sort_values(
        ["rmse", "mae", "mape", "parameter_sort", "runtime_seconds"],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).iloc[0]
    return {str(key): _json_safe(value) for key, value in chosen.to_dict().items()}


def _matching_test_candidate(
    global_metrics: pd.DataFrame,
    validation_best: dict[str, Any],
) -> dict[str, Any]:
    match = global_metrics[
        global_metrics["split"].eq("test")
        & global_metrics["model"].eq(validation_best["model"])
        & global_metrics["horizon_minutes"].eq(validation_best["horizon_minutes"])
    ]
    if match.empty:
        return {}
    row = match.iloc[0]
    return {str(key): _json_safe(value) for key, value in row.to_dict().items()}


def split_segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize rows and temporal coverage by split and contiguous segment."""

    rows: list[dict[str, Any]] = []
    for (split, segment_id), group in frame.groupby(["split", "segment_id"], observed=True):
        rows.append(
            {
                "split": str(split),
                "segment_id": int(segment_id),
                "rows": int(group.shape[0]),
                "start_time": pd.Timestamp(group["timestamp"].min()).isoformat(),
                "end_time": pd.Timestamp(group["timestamp"].max()).isoformat(),
                "duration_days": _duration_days(group["timestamp"]),
            }
        )
    return pd.DataFrame(rows)


def audit_window_sets(
    windows_by_split: dict[str, ForecastWindowSet],
    *,
    horizon_minutes: int,
    horizon_steps: int,
    expected_frequency: str,
) -> dict[str, Any]:
    """Check split-local windows for timestamp alignment and leakage-risk boundaries."""

    expected_delta = pd.Timedelta(expected_frequency) * horizon_steps
    rows = {}
    leakage_detected = False
    for split, windows in windows_by_split.items():
        if windows.is_empty:
            rows[split] = {"window_count": 0, "alignment_errors": 0}
            continue
        actual_deltas = [
            pd.Timestamp(target) - pd.Timestamp(end)
            for end, target in zip(
                windows.input_end_timestamps,
                windows.target_timestamps,
                strict=True,
            )
        ]
        alignment_errors = sum(delta != expected_delta for delta in actual_deltas)
        segment_errors = int(
            ((windows.target_indices - windows.end_indices) != horizon_steps).sum()
        )
        leakage_detected = leakage_detected or alignment_errors > 0 or segment_errors > 0
        rows[split] = {
            "window_count": int(windows.inputs.shape[0]),
            "feature_count": int(windows.inputs.shape[2]),
            "alignment_errors": int(alignment_errors),
            "target_step_errors": segment_errors,
            "first_input_end": pd.Timestamp(windows.input_end_timestamps[0]).isoformat(),
            "first_target": pd.Timestamp(windows.target_timestamps[0]).isoformat(),
            "last_input_end": pd.Timestamp(windows.input_end_timestamps[-1]).isoformat(),
            "last_target": pd.Timestamp(windows.target_timestamps[-1]).isoformat(),
        }
    return {
        "horizon_minutes": horizon_minutes,
        "horizon_steps": horizon_steps,
        "expected_frequency": expected_frequency,
        "leakage_detected": leakage_detected,
        "splits": rows,
    }


def sensor_quality_table(
    raw: pd.DataFrame,
    resampled: pd.DataFrame,
    sensors: tuple[str, ...],
) -> pd.DataFrame:
    """Return a compact audit table for selected continuous sensors."""

    rows = []
    for sensor in sensors:
        raw_values = pd.to_numeric(raw[sensor], errors="coerce")
        resampled_values = pd.to_numeric(resampled[sensor], errors="coerce")
        rows.append(
            {
                "sensor": sensor,
                "raw_missing": int(raw_values.isna().sum()),
                "raw_unique_values": int(raw_values.nunique(dropna=True)),
                "raw_mean": float(raw_values.mean()),
                "raw_std": float(raw_values.std(ddof=0)),
                "raw_min": float(raw_values.min()),
                "raw_max": float(raw_values.max()),
                "resampled_missing": int(resampled_values.isna().sum()),
                "resampled_std": float(resampled_values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def state_summary(raw: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize binary/state variables without forecasting them."""

    rows = []
    for column in BINARY_STATE_COLUMNS:
        values = pd.to_numeric(raw[column], errors="coerce")
        counts = values.value_counts(normalize=True).sort_index()
        rows.append(
            {
                "state": column,
                "unique_values": sorted(
                    float(value) for value in values.dropna().unique().tolist()
                ),
                "zero_rate": float(counts.get(0.0, 0.0)),
                "one_rate": float(counts.get(1.0, 0.0)),
                "transition_count": int(values.ne(values.shift()).sum() - 1),
            }
        )
    return rows


def _run_horizon_models(
    *,
    original_windows: dict[str, ForecastWindowSet],
    scaled_windows: dict[str, ForecastWindowSet],
    scaler: TimeSeriesStandardScaler,
    normal_ranges: pd.DataFrame,
    config: MetroPTForecastingExperimentConfig,
    horizon_minutes: int,
    horizon_steps: int,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    sensors = config.selected_sensors
    metrics: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    training_rows: list[dict[str, Any]] = []

    baseline_specs: tuple[tuple[str, Callable[[ForecastWindowSet], np.ndarray]], ...] = (
        ("persistence", lambda windows: persistence_forecast(windows)),
        (
            "moving_average",
            lambda windows: moving_average_forecast(
                windows,
                window_steps=config.moving_average_window_steps,
            ),
        ),
        (
            "rolling_linear_trend",
            lambda windows: rolling_linear_trend_forecast(
                windows,
                window_steps=config.statistical_window_steps,
                horizon_steps=horizon_steps,
            ),
        ),
    )
    for model_name, predictor in baseline_specs:
        started = time.perf_counter()
        split_predictions: dict[str, np.ndarray] = {}
        for split, windows in original_windows.items():
            split_predictions[split] = predictor(windows)
        runtime = float(time.perf_counter() - started)
        metrics.extend(
            _metrics_for_splits(
                original_windows,
                split_predictions,
                sensors=sensors,
                model=model_name,
                horizon_minutes=horizon_minutes,
                runtime_seconds=runtime,
                training_seconds=0.0,
                parameter_count=0,
                mape_min_abs_value=config.mape_min_abs_value,
            )
        )
        predictions.append(
            _prediction_frame_for_splits(
                original_windows,
                split_predictions,
                sensors=sensors,
                model=model_name,
                horizon_minutes=horizon_minutes,
                max_rows_per_model=config.max_saved_prediction_rows_per_model,
            )
        )
        training_rows.append(
            {
                "model": model_name,
                "horizon_minutes": horizon_minutes,
                "training_seconds": 0.0,
                "inference_seconds": runtime,
                "parameter_count": 0,
                "notes": "deterministic baseline",
            }
        )

    ridge_result = train_ridge_autoregression(
        original_windows,
        horizon_minutes=horizon_minutes,
        lag_steps=config.ridge_lag_steps,
        max_train_windows=config.lstm.max_train_windows,
        random_seed=config.random_seed,
        output_path=models_dir / f"ridge_autoregression_{horizon_minutes}min.joblib",
    )
    metrics.extend(
        _metrics_for_splits(
            original_windows,
            ridge_result.predictions,
            sensors=sensors,
            model="ridge_autoregression",
            horizon_minutes=horizon_minutes,
            runtime_seconds=ridge_result.inference_seconds,
            training_seconds=ridge_result.training_seconds,
            parameter_count=ridge_result.parameter_count,
            mape_min_abs_value=config.mape_min_abs_value,
        )
    )
    predictions.append(
        _prediction_frame_for_splits(
            original_windows,
            ridge_result.predictions,
            sensors=sensors,
            model="ridge_autoregression",
            horizon_minutes=horizon_minutes,
            max_rows_per_model=config.max_saved_prediction_rows_per_model,
        )
    )
    training_rows.append(ridge_result.to_row(horizon_minutes=horizon_minutes))

    if config.lstm.enabled:
        uni_result = train_pooled_univariate_lstm(
            scaled_windows,
            scaler=scaler,
            horizon_minutes=horizon_minutes,
            sensors=sensors,
            config=config.lstm,
            random_seed=config.random_seed,
            output_path=models_dir / f"lstm_univariate_{horizon_minutes}min.pt",
        )
        metrics.extend(
            _metrics_for_splits(
                original_windows,
                uni_result.predictions,
                sensors=sensors,
                model="lstm_univariate",
                horizon_minutes=horizon_minutes,
                runtime_seconds=uni_result.inference_seconds,
                training_seconds=uni_result.training_seconds,
                parameter_count=uni_result.parameter_count,
                mape_min_abs_value=config.mape_min_abs_value,
            )
        )
        predictions.append(
            _prediction_frame_for_splits(
                original_windows,
                uni_result.predictions,
                sensors=sensors,
                model="lstm_univariate",
                horizon_minutes=horizon_minutes,
                max_rows_per_model=config.max_saved_prediction_rows_per_model,
            )
        )
        training_rows.append(uni_result.to_row(horizon_minutes=horizon_minutes))

        multi_result = train_multivariate_lstm(
            scaled_windows,
            scaler=scaler,
            horizon_minutes=horizon_minutes,
            sensors=sensors,
            config=config.lstm,
            random_seed=config.random_seed,
            output_path=models_dir / f"lstm_multivariate_{horizon_minutes}min.pt",
        )
        metrics.extend(
            _metrics_for_splits(
                original_windows,
                multi_result.predictions,
                sensors=sensors,
                model="lstm_multivariate",
                horizon_minutes=horizon_minutes,
                runtime_seconds=multi_result.inference_seconds,
                training_seconds=multi_result.training_seconds,
                parameter_count=multi_result.parameter_count,
                mape_min_abs_value=config.mape_min_abs_value,
            )
        )
        predictions.append(
            _prediction_frame_for_splits(
                original_windows,
                multi_result.predictions,
                sensors=sensors,
                model="lstm_multivariate",
                horizon_minutes=horizon_minutes,
                max_rows_per_model=config.max_saved_prediction_rows_per_model,
            )
        )
        training_rows.append(multi_result.to_row(horizon_minutes=horizon_minutes))

    metrics_frame = pd.concat(metrics, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    risk_example = forecast_risk_signal(
        prediction_frame[prediction_frame["split"].eq("test")],
        normal_ranges=normal_ranges,
        horizon_minutes=horizon_minutes,
        model="all_models",
    )
    if not risk_example.empty:
        risk_example.to_csv(
            models_dir.parent / f"risk_candidates_{horizon_minutes}min.csv",
            index=False,
        )
    return metrics_frame, prediction_frame, training_rows


def persistence_forecast(windows: ForecastWindowSet) -> np.ndarray:
    """Forecast the future as the last observed value."""

    if windows.is_empty:
        return np.empty_like(windows.targets)
    return windows.inputs[:, -1, :].astype(np.float32)


def moving_average_forecast(windows: ForecastWindowSet, *, window_steps: int) -> np.ndarray:
    """Forecast the future as a causal moving average over the input tail."""

    if window_steps < 1:
        raise ValueError("window_steps must be positive")
    if windows.is_empty:
        return np.empty_like(windows.targets)
    tail = windows.inputs[:, -min(window_steps, windows.inputs.shape[1]) :, :]
    return tail.mean(axis=1).astype(np.float32)


def rolling_linear_trend_forecast(
    windows: ForecastWindowSet,
    *,
    window_steps: int,
    horizon_steps: int,
) -> np.ndarray:
    """Forecast with a per-window least-squares linear trend."""

    if window_steps < 2:
        raise ValueError("window_steps must be at least 2")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    if windows.is_empty:
        return np.empty_like(windows.targets)
    tail = windows.inputs[:, -min(window_steps, windows.inputs.shape[1]) :, :].astype(float)
    length = tail.shape[1]
    x = np.arange(length, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered * x_centered))
    y_mean = tail.mean(axis=1)
    slope = np.sum((tail - y_mean[:, None, :]) * x_centered[None, :, None], axis=1) / denominator
    target_x = float(length - 1 + horizon_steps)
    prediction = y_mean + slope * (target_x - x.mean())
    return prediction.astype(np.float32)


@dataclass(frozen=True)
class ModelRunResult:
    """Predictions and runtime metadata for one fitted forecasting model."""

    predictions: dict[str, np.ndarray]
    training_seconds: float
    inference_seconds: float
    parameter_count: int
    best_validation_loss: float | None
    epochs_trained: int | None
    model_path: str
    notes: str

    def to_row(self, *, horizon_minutes: int) -> dict[str, Any]:
        return {
            "model": self.notes,
            "horizon_minutes": horizon_minutes,
            "training_seconds": self.training_seconds,
            "inference_seconds": self.inference_seconds,
            "parameter_count": self.parameter_count,
            "best_validation_loss": self.best_validation_loss,
            "epochs_trained": self.epochs_trained,
            "model_path": self.model_path,
        }


def train_ridge_autoregression(
    windows_by_split: dict[str, ForecastWindowSet],
    *,
    horizon_minutes: int,
    lag_steps: int,
    max_train_windows: int,
    random_seed: int,
    output_path: Path,
) -> ModelRunResult:
    """Train a multivariate ridge autoregression baseline on flattened lagged windows."""

    train = windows_by_split["train"]
    if train.is_empty:
        raise ValueError("train windows are empty")
    train_indices = _sample_indices(train.inputs.shape[0], max_train_windows, random_seed)
    x_train = _ridge_features(train.inputs[train_indices], lag_steps=lag_steps)
    y_train = train.targets[train_indices]
    model = Ridge(alpha=1.0)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = float(time.perf_counter() - started)
    started = time.perf_counter()
    predictions = {}
    for split, windows in windows_by_split.items():
        features = _ridge_features(windows.inputs, lag_steps=lag_steps)
        predictions[split] = cast(np.ndarray, model.predict(features)).astype(np.float32)
    inference_seconds = float(time.perf_counter() - started)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    parameter_count = int(np.asarray(model.coef_).size + np.asarray(model.intercept_).size)
    return ModelRunResult(
        predictions=predictions,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=parameter_count,
        best_validation_loss=None,
        epochs_trained=None,
        model_path=str(output_path),
        notes="ridge_autoregression",
    )


def train_pooled_univariate_lstm(
    windows_by_split: dict[str, ForecastWindowSet],
    *,
    scaler: TimeSeriesStandardScaler,
    horizon_minutes: int,
    sensors: tuple[str, ...],
    config: LSTMForecastingConfig,
    random_seed: int,
    output_path: Path,
) -> ModelRunResult:
    """Train one pooled univariate LSTM using one sensor history at a time."""

    train = windows_by_split["train"]
    validation = windows_by_split["validation"]
    train_indices = _sample_indices(train.inputs.shape[0], config.max_train_windows, random_seed)
    validation_indices = _sample_indices(
        validation.inputs.shape[0],
        config.max_validation_windows,
        random_seed,
    )
    x_train, y_train = _pooled_univariate_arrays(
        train.inputs[train_indices],
        train.targets[train_indices],
    )
    x_validation, y_validation = _pooled_univariate_arrays(
        validation.inputs[validation_indices],
        validation.targets[validation_indices],
    )
    model = LSTMForecaster(
        input_size=1,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        output_size=1,
    )
    training = _fit_lstm(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        config=config,
        random_seed=random_seed,
    )
    started = time.perf_counter()
    predictions = {
        split: _predict_pooled_univariate_lstm(
            training.model,
            windows,
            scaler=scaler,
            config=config,
        )
        for split, windows in windows_by_split.items()
    }
    inference_seconds = float(time.perf_counter() - started)
    _save_torch_checkpoint(
        output_path,
        model=training.model,
        config=config,
        sensors=sensors,
        horizon_minutes=horizon_minutes,
        model_name="lstm_univariate",
        training=training,
    )
    return ModelRunResult(
        predictions=predictions,
        training_seconds=training.training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=count_torch_parameters(training.model),
        best_validation_loss=training.best_validation_loss,
        epochs_trained=training.epochs_trained,
        model_path=str(output_path),
        notes="lstm_univariate",
    )


def train_multivariate_lstm(
    windows_by_split: dict[str, ForecastWindowSet],
    *,
    scaler: TimeSeriesStandardScaler,
    horizon_minutes: int,
    sensors: tuple[str, ...],
    config: LSTMForecastingConfig,
    random_seed: int,
    output_path: Path,
) -> ModelRunResult:
    """Train a multivariate LSTM that sees all selected sensors jointly."""

    train = windows_by_split["train"]
    validation = windows_by_split["validation"]
    train_indices = _sample_indices(train.inputs.shape[0], config.max_train_windows, random_seed)
    validation_indices = _sample_indices(
        validation.inputs.shape[0],
        config.max_validation_windows,
        random_seed,
    )
    model = LSTMForecaster(
        input_size=len(sensors),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        output_size=len(sensors),
    )
    training = _fit_lstm(
        model,
        train.inputs[train_indices],
        train.targets[train_indices],
        validation.inputs[validation_indices],
        validation.targets[validation_indices],
        config=config,
        random_seed=random_seed,
    )
    started = time.perf_counter()
    predictions_scaled = {
        split: _predict_lstm(training.model, windows.inputs, config=config)
        for split, windows in windows_by_split.items()
    }
    predictions = {
        split: scaler.inverse_transform(values).astype(np.float32)
        for split, values in predictions_scaled.items()
    }
    inference_seconds = float(time.perf_counter() - started)
    _save_torch_checkpoint(
        output_path,
        model=training.model,
        config=config,
        sensors=sensors,
        horizon_minutes=horizon_minutes,
        model_name="lstm_multivariate",
        training=training,
    )
    return ModelRunResult(
        predictions=predictions,
        training_seconds=training.training_seconds,
        inference_seconds=inference_seconds,
        parameter_count=count_torch_parameters(training.model),
        best_validation_loss=training.best_validation_loss,
        epochs_trained=training.epochs_trained,
        model_path=str(output_path),
        notes="lstm_multivariate",
    )


@dataclass(frozen=True)
class LSTMTrainingState:
    """Training outcome for a fitted LSTM model."""

    model: LSTMForecaster
    training_seconds: float
    best_validation_loss: float
    epochs_trained: int
    train_loss_history: list[float]
    validation_loss_history: list[float]


def _fit_lstm(
    model: LSTMForecaster,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    config: LSTMForecastingConfig,
    random_seed: int,
) -> LSTMTrainingState:
    _set_reproducible_seed(random_seed)
    device = torch.device(config.device)
    model = model.to(device)
    train_dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    validation_x = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
    validation_y = torch.as_tensor(y_validation, dtype=torch.float32, device=device)
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    train_history: list[float] = []
    validation_history: list[float] = []
    started = time.perf_counter()

    epochs_trained = 0
    for _epoch in range(config.epochs):
        epochs_trained += 1
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y).detach().cpu())
        train_history.append(train_loss)
        validation_history.append(validation_loss)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return LSTMTrainingState(
        model=model,
        training_seconds=float(time.perf_counter() - started),
        best_validation_loss=best_validation_loss,
        epochs_trained=epochs_trained,
        train_loss_history=train_history,
        validation_loss_history=validation_history,
    )


def _predict_lstm(
    model: LSTMForecaster,
    inputs: np.ndarray,
    *,
    config: LSTMForecastingConfig,
) -> np.ndarray:
    if inputs.shape[0] == 0:
        return np.empty((0, model.head.out_features), dtype=np.float32)
    device = torch.device(config.device)
    model.eval()
    outputs: list[np.ndarray] = []
    dataset = TensorDataset(torch.as_tensor(inputs, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_prediction = model(batch_x.to(device)).detach().cpu().numpy()
            outputs.append(batch_prediction)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _predict_pooled_univariate_lstm(
    model: LSTMForecaster,
    windows: ForecastWindowSet,
    *,
    scaler: TimeSeriesStandardScaler,
    config: LSTMForecastingConfig,
) -> np.ndarray:
    if windows.is_empty:
        return np.empty_like(windows.targets)
    predictions = np.empty_like(windows.targets, dtype=np.float32)
    for sensor_index in range(windows.inputs.shape[2]):
        sensor_inputs = windows.inputs[:, :, sensor_index : sensor_index + 1]
        sensor_prediction = _predict_lstm(model, sensor_inputs, config=config).reshape(-1)
        predictions[:, sensor_index] = sensor_prediction
    return _inverse_transform_scaled_predictions(predictions, scaler)


def _inverse_transform_scaled_predictions(
    predictions: np.ndarray,
    scaler: TimeSeriesStandardScaler,
) -> np.ndarray:
    return scaler.inverse_transform(predictions).astype(np.float32)


def _pooled_univariate_arrays(
    inputs: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples, sequence_length, feature_count = inputs.shape
    x = inputs.transpose(0, 2, 1).reshape(samples * feature_count, sequence_length, 1)
    y = targets.reshape(samples * feature_count, 1)
    return x.astype(np.float32), y.astype(np.float32)


def _ridge_features(inputs: np.ndarray, *, lag_steps: int) -> np.ndarray:
    if inputs.shape[0] == 0:
        return np.empty((0, min(lag_steps, inputs.shape[1]) * inputs.shape[2]), dtype=np.float32)
    tail = inputs[:, -min(lag_steps, inputs.shape[1]) :, :]
    return tail.reshape(tail.shape[0], tail.shape[1] * tail.shape[2]).astype(np.float32)


def count_torch_parameters(model: nn.Module) -> int:
    """Count trainable PyTorch parameters."""

    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


def _metrics_for_splits(
    windows_by_split: dict[str, ForecastWindowSet],
    predictions_by_split: dict[str, np.ndarray],
    *,
    sensors: tuple[str, ...],
    model: str,
    horizon_minutes: int,
    runtime_seconds: float,
    training_seconds: float,
    parameter_count: int | None,
    mape_min_abs_value: float,
) -> list[pd.DataFrame]:
    return [
        compute_forecasting_metrics(
            y_true=windows_by_split[split].targets,
            y_pred=predictions_by_split[split],
            sensors=sensors,
            split=split,
            model=model,
            horizon_minutes=horizon_minutes,
            runtime_seconds=runtime_seconds,
            training_seconds=training_seconds,
            parameter_count=parameter_count,
            mape_min_abs_value=mape_min_abs_value,
        )
        for split in ("train", "validation", "test")
        if not windows_by_split[split].is_empty
    ]


def _prediction_frame_for_splits(
    windows_by_split: dict[str, ForecastWindowSet],
    predictions_by_split: dict[str, np.ndarray],
    *,
    sensors: tuple[str, ...],
    model: str,
    horizon_minutes: int,
    max_rows_per_model: int,
) -> pd.DataFrame:
    frames = []
    for split in ("validation", "test"):
        windows = windows_by_split[split]
        if windows.is_empty:
            continue
        indices = _sample_indices(windows.inputs.shape[0], max_rows_per_model, random_seed=17)
        frames.append(
            forecast_prediction_frame(
                windows,
                predictions_by_split[split],
                sensors=sensors,
                model=model,
                horizon_minutes=horizon_minutes,
                split=split,
                window_indices=indices,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def forecast_prediction_frame(
    windows: ForecastWindowSet,
    predictions: np.ndarray,
    *,
    sensors: tuple[str, ...],
    model: str,
    horizon_minutes: int,
    split: str,
    window_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return long-format forecast rows with recoverable timestamps."""

    pred = _as_2d_array(predictions, name="predictions")
    if pred.shape != windows.targets.shape:
        raise ValueError("predictions must match window targets")
    indices = (
        np.arange(windows.targets.shape[0], dtype=int)
        if window_indices is None
        else np.asarray(window_indices, dtype=int)
    )
    rows: list[dict[str, Any]] = []
    for row_index in indices:
        for sensor_index, sensor in enumerate(sensors):
            actual = float(windows.targets[row_index, sensor_index])
            forecast = float(pred[row_index, sensor_index])
            rows.append(
                {
                    "model": model,
                    "horizon_minutes": horizon_minutes,
                    "split": split,
                    "sensor": sensor,
                    "input_start_timestamp": windows.input_start_timestamps[row_index],
                    "input_end_timestamp": windows.input_end_timestamps[row_index],
                    "target_timestamp": windows.target_timestamps[row_index],
                    "segment_id": int(windows.segment_ids[row_index]),
                    "y_true": actual,
                    "y_pred": forecast,
                    "absolute_error": abs(actual - forecast),
                }
            )
    return pd.DataFrame(rows)


def _metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sensor: str,
    split: str,
    model: str,
    horizon_minutes: int,
    runtime_seconds: float,
    training_seconds: float,
    parameter_count: int | None,
    mape_min_abs_value: float,
) -> dict[str, Any]:
    errors = y_true - y_pred
    absolute = np.abs(errors)
    squared = errors * errors
    mape_mask = np.abs(y_true) >= mape_min_abs_value
    mape = (
        float(np.mean(absolute[mape_mask] / np.abs(y_true[mape_mask])) * 100)
        if mape_mask.any()
        else None
    )
    return {
        "model": model,
        "horizon_minutes": horizon_minutes,
        "split": split,
        "sensor": sensor,
        "rows": int(y_true.shape[0]),
        "mae": float(np.mean(absolute)) if absolute.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(squared))) if squared.size else float("nan"),
        "mape": mape,
        "mape_applicable_ratio": float(mape_mask.mean()) if mape_mask.size else 0.0,
        "runtime_seconds": runtime_seconds,
        "training_seconds": training_seconds,
        "parameter_count": parameter_count,
    }


def _as_2d_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _sample_indices(row_count: int, max_rows: int, random_seed: int) -> np.ndarray:
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if row_count <= max_rows:
        return np.arange(row_count, dtype=int)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(row_count, size=max_rows, replace=False)).astype(int)


def _save_torch_checkpoint(
    output_path: Path,
    *,
    model: LSTMForecaster,
    config: LSTMForecastingConfig,
    sensors: tuple[str, ...],
    horizon_minutes: int,
    model_name: str,
    training: LSTMTrainingState,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "sensors": list(sensors),
            "horizon_minutes": horizon_minutes,
            "best_validation_loss": training.best_validation_loss,
            "epochs_trained": training.epochs_trained,
            "train_loss_history": training.train_loss_history,
            "validation_loss_history": training.validation_loss_history,
        },
        output_path,
    )


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _plot_sensor_overview(
    frame: pd.DataFrame,
    sensors: tuple[str, ...],
    output_path: Path,
) -> None:
    sample = _sample_frame(frame, max_rows=5000)
    rows = min(3, len(sensors))
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(12, 2.6 * rows),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    for axis, sensor in zip(axes_array, sensors[:rows], strict=False):
        axis.plot(sample["timestamp"], sample[sensor], linewidth=0.8, label=sensor)
        axis.set_ylabel(sensor)
        axis.legend(loc="upper right")
    axes_array[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore[no-untyped-call]
    figure.savefig(output_path)
    plt.close(figure)


def _plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    *,
    horizon_minutes: int,
    model: str,
    sensor: str,
    output_path: Path,
) -> None:
    frame = predictions[
        predictions["horizon_minutes"].eq(horizon_minutes)
        & predictions["model"].eq(model)
        & predictions["split"].eq("test")
        & predictions["sensor"].eq(sensor)
    ].copy()
    if frame.empty:
        return
    frame = _sample_frame(frame.sort_values("target_timestamp"), max_rows=1500)
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.plot(frame["target_timestamp"], frame["y_true"], label="actual", linewidth=0.9)
    axis.plot(frame["target_timestamp"], frame["y_pred"], label="forecast", linewidth=0.9)
    axis.set_title(f"{model} {sensor} forecast at {horizon_minutes} min")
    axis.set_ylabel(sensor)
    axis.legend(loc="upper right")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore[no-untyped-call]
    figure.savefig(output_path)
    plt.close(figure)


def _plot_difficult_cases(
    predictions: pd.DataFrame,
    *,
    horizon_minutes: int,
    model: str,
    output_path: Path,
) -> None:
    frame = predictions[
        predictions["horizon_minutes"].eq(horizon_minutes)
        & predictions["model"].eq(model)
        & predictions["split"].eq("test")
    ].copy()
    if frame.empty:
        return
    top = frame.sort_values("absolute_error", ascending=False).head(30)
    labels = top["sensor"].astype(str) + "\n" + top["target_timestamp"].astype(str).str.slice(5, 16)
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    axis.barh(labels, top["absolute_error"], color="#b8322b")
    axis.set_title(f"Difficult forecast cases at {horizon_minutes} min")
    axis.set_xlabel("absolute error")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    global_metrics = metrics[metrics["sensor"].eq("__global__") & metrics["split"].eq("test")]
    if global_metrics.empty:
        return
    pivot = global_metrics.pivot_table(
        index="model",
        columns="horizon_minutes",
        values="rmse",
        aggfunc="mean",
    )
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Global test RMSE by model and horizon")
    axis.set_ylabel("RMSE")
    axis.legend(title="horizon min")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_per_sensor_error(metrics: pd.DataFrame, output_path: Path) -> None:
    frame = metrics[metrics["split"].eq("test") & ~metrics["sensor"].eq("__global__")].copy()
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="sensor",
        columns="horizon_minutes",
        values="rmse",
        aggfunc="min",
    )
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Best per-sensor test RMSE by horizon")
    axis.set_ylabel("RMSE")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_risk_signal(risk: pd.DataFrame, output_path: Path) -> None:
    if risk.empty:
        return
    frame = (
        risk.groupby(["horizon_minutes", "sensor"], observed=True)[
            ["forecast_outside_train_range", "actual_outside_train_range"]
        ]
        .mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for sensor, group in frame.groupby("sensor", observed=True):
        axis.plot(
            group["horizon_minutes"],
            group["forecast_outside_train_range"],
            marker="o",
            label=str(sensor),
        )
    axis.set_title("Forecast-based normal-range warning rate")
    axis.set_xlabel("horizon minutes")
    axis.set_ylabel("forecast outside train range")
    axis.legend(loc="upper left", ncols=2)
    figure.savefig(output_path)
    plt.close(figure)


def _sample_frame(frame: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if frame.shape[0] <= max_rows:
        return frame.copy()
    indices = np.linspace(0, frame.shape[0] - 1, num=max_rows, dtype=int)
    return frame.iloc[indices].copy()


def _write_run_report(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    training: pd.DataFrame,
) -> None:
    best = summary["best_test_global"]
    validation = summary["best_validation_global"]
    selected_test = summary["validation_selected_test_global"]
    preview = (
        metrics[metrics["split"].eq("test") & metrics["sensor"].eq("__global__")]
        .sort_values(["rmse", "mae"])
        .head(12)
    )
    lines = [
        "# MetroPT Forecasting Phase 8",
        "",
        f"- Sampling interval: {summary['sampling_interval_seconds']} seconds",
        f"- Selected sensors: {', '.join(summary['selected_sensors'])}",
        f"- Horizons: {', '.join(str(value) for value in summary['horizons_minutes'])} minutes",
        (
            f"- Best validation global model: {validation['model']} "
            f"at {validation['horizon_minutes']} min"
        ),
        (f"- Validation-selected test RMSE: {_fmt(selected_test.get('rmse'))}"),
        f"- Best test global model: {best['model']} at {best['horizon_minutes']} min",
        f"- Best test global MAE: {_fmt(best['mae'])}",
        f"- Best test global RMSE: {_fmt(best['rmse'])}",
        f"- Transformer implemented: {summary['transformer']['implemented']}",
        "",
        "## Global Test Metrics Preview",
        "",
        *_markdown_table(
            preview[
                [
                    "model",
                    "horizon_minutes",
                    "mae",
                    "rmse",
                    "mape",
                    "runtime_seconds",
                    "training_seconds",
                ]
            ]
        ),
        "",
        "## Training Runtime",
        "",
        *_markdown_table(training.head(20)),
        "",
        "## Risk Signal",
        "",
        summary["risk_signal"]["definition"]
        if summary["risk_signal"].get("available")
        else "Risk signal summary unavailable.",
        "",
    ]
    (run_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        rows.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return rows


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return _fmt(value)
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def _tuple_strings(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list | tuple):
        raise ValueError("expected a list of strings")
    return tuple(str(item) for item in value)


def _tuple_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if not isinstance(value, list | tuple):
        raise ValueError("expected a list of integers")
    return tuple(int(item) for item in value)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return cast(dict[str, Any], value)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return None if not np.isfinite(numeric) else numeric
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
