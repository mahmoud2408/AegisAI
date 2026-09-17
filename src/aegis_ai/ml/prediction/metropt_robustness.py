"""Phase 7B MetroPT failure-prediction robustness and target validation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.pipeline import Pipeline

from aegis_ai.data.adapters.metropt import BINARY_STATE_COLUMNS, CONTINUOUS_SENSORS
from aegis_ai.data.dataset_registry import project_root
from aegis_ai.ml.prediction.failure_prediction import (
    _METROPT_NON_FEATURE_COLUMNS,
    FailureEvent,
    FailureModelConfig,
    MetroPTExperimentConfig,
    SHAPConfig,
    _duration_days,
    _events_in_split,
    _numeric_preprocessor,
    _positive_probabilities,
    _positive_weight,
    _run_model_family,
    _save_json,
    _select_final_tree_model,
    _shade_failure_events,
    _tuple_floats,
    _tuple_ints,
    _tuple_strings,
    _validate_threshold_grid,
    _write_dataframe,
    _write_shap_artifacts,
    audit_metropt_frame,
    build_metropt_causal_features,
    build_metropt_failure_target,
    classification_metrics,
    compute_early_warning_metrics,
    load_metropt_frame,
)

DEFAULT_ROBUSTNESS_HORIZONS = (0.5, 2.0, 6.0)
DEFAULT_ROBUSTNESS_THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class MetroPTRobustnessExperimentConfig:
    """Configuration for the MetroPT Phase 7B robustness experiment."""

    random_seed: int = 42
    horizons_hours: tuple[float, ...] = DEFAULT_ROBUSTNESS_HORIZONS
    prediction_stride_rows: int = 6
    train_end: str = "2020-05-31 00:00:00"
    validation_end: str = "2020-06-08 00:00:00"
    lag_steps: tuple[int, ...] = (1, 6, 60)
    rolling_windows: tuple[str, ...] = ("5min", "30min")
    trend_window: str = "30min"
    state_transition_count_window: str | None = "30min"
    min_periods: int = 3
    max_train_rows: int = 60000
    negative_to_positive_ratio: int = 25
    threshold_grid: tuple[float, ...] = DEFAULT_ROBUSTNESS_THRESHOLDS
    failure_events: tuple[FailureEvent, ...] = field(
        default_factory=lambda: MetroPTExperimentConfig().failure_events
    )
    models: FailureModelConfig = field(default_factory=FailureModelConfig)
    shap: SHAPConfig = field(default_factory=SHAPConfig)

    def __post_init__(self) -> None:
        if not self.horizons_hours:
            raise ValueError("horizons_hours must not be empty")
        if any(horizon <= 0 for horizon in self.horizons_hours):
            raise ValueError("horizons_hours must contain positive values")
        if tuple(sorted(self.horizons_hours)) != self.horizons_hours:
            raise ValueError("horizons_hours must be sorted from shortest to longest")
        if self.prediction_stride_rows < 1:
            raise ValueError("prediction_stride_rows must be positive")
        if not self.lag_steps or any(step < 1 for step in self.lag_steps):
            raise ValueError("lag_steps must contain positive integers")
        if not self.rolling_windows:
            raise ValueError("rolling_windows must not be empty")
        if self.min_periods < 1:
            raise ValueError("min_periods must be positive")
        if self.max_train_rows < 1:
            raise ValueError("max_train_rows must be positive")
        if self.negative_to_positive_ratio < 1:
            raise ValueError("negative_to_positive_ratio must be positive")
        if pd.Timestamp(self.train_end) >= pd.Timestamp(self.validation_end):
            raise ValueError("train_end must be before validation_end")
        _validate_threshold_grid(self.threshold_grid)


@dataclass(frozen=True)
class MetroPTRobustnessExperimentResult:
    """Location and summary for a completed Phase 7B run."""

    run_dir: Path
    metrics: dict[str, Any]


def load_metropt_robustness_config(path: Path) -> MetroPTRobustnessExperimentConfig:
    """Load Phase 7B MetroPT robustness config from YAML."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a YAML mapping")
    default = MetroPTRobustnessExperimentConfig()
    events_payload = payload.get("failure_events")
    events = (
        tuple(FailureEvent(**event) for event in events_payload)
        if events_payload is not None
        else default.failure_events
    )
    state_transition_count_window = payload.get(
        "state_transition_count_window",
        default.state_transition_count_window,
    )
    return MetroPTRobustnessExperimentConfig(
        random_seed=int(payload.get("random_seed", default.random_seed)),
        horizons_hours=_tuple_floats(payload.get("horizons_hours"), default.horizons_hours),
        prediction_stride_rows=int(
            payload.get("prediction_stride_rows", default.prediction_stride_rows)
        ),
        train_end=str(payload.get("train_end", default.train_end)),
        validation_end=str(payload.get("validation_end", default.validation_end)),
        lag_steps=_tuple_ints(payload.get("lag_steps"), default.lag_steps),
        rolling_windows=_tuple_strings(payload.get("rolling_windows"), default.rolling_windows),
        trend_window=str(payload.get("trend_window", default.trend_window)),
        state_transition_count_window=(
            None if state_transition_count_window is None else str(state_transition_count_window)
        ),
        min_periods=int(payload.get("min_periods", default.min_periods)),
        max_train_rows=int(payload.get("max_train_rows", default.max_train_rows)),
        negative_to_positive_ratio=int(
            payload.get("negative_to_positive_ratio", default.negative_to_positive_ratio)
        ),
        threshold_grid=_tuple_floats(payload.get("threshold_grid"), default.threshold_grid),
        failure_events=events,
        models=FailureModelConfig(**_mapping(payload.get("models"))),
        shap=SHAPConfig(**_mapping(payload.get("shap"))),
    )


def run_metropt_robustness_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: MetroPTRobustnessExperimentConfig | None = None,
) -> MetroPTRobustnessExperimentResult:
    """Run MetroPT Phase 7B robustness and target-validation experiment."""

    cfg = config or MetroPTRobustnessExperimentConfig()
    project = project_root() if root is None else root
    resolved_run_id = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    base_output = output_root or project / "experiments" / "prediction" / "metropt_robustness"
    run_dir = base_output / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "config.json", asdict(cfg))

    started = time.perf_counter()
    raw = load_metropt_frame(project)
    raw_audit = audit_metropt_frame(raw, failure_events=cfg.failure_events)
    processed_audit = audit_processed_metropt_dataset(project)
    _save_json(run_dir / "dataset_audit.json", raw_audit)
    _save_json(run_dir / "processed_dataset_audit.json", processed_audit)

    event_table = characterize_failure_events(
        raw,
        failure_events=cfg.failure_events,
        train_end=pd.Timestamp(cfg.train_end),
        validation_end=pd.Timestamp(cfg.validation_end),
    )
    event_table.to_csv(run_dir / "failure_events.csv", index=False)
    event_summary = summarize_failure_event_structure(event_table)
    _save_json(run_dir / "failure_event_summary.json", event_summary)
    _plot_failure_event_timeline(event_table, figures_dir / "failure_event_timeline.png")

    prefailure_sensor_summary, prefailure_state_summary, prefailure_effects = (
        analyze_prefailure_buffers(raw, cfg.failure_events)
    )
    prefailure_sensor_summary.to_csv(run_dir / "prefailure_sensor_summary.csv", index=False)
    prefailure_state_summary.to_csv(
        run_dir / "prefailure_state_transition_summary.csv",
        index=False,
    )
    prefailure_effects.to_csv(run_dir / "prefailure_effect_sizes.csv", index=False)
    _plot_prefailure_sensor_summary(
        prefailure_sensor_summary,
        figures_dir / "prefailure_sensor_summary.png",
    )

    largest_horizon = max(cfg.horizons_hours)
    base_labeled = build_metropt_failure_target(
        raw,
        failure_events=cfg.failure_events,
        horizon_hours=largest_horizon,
    )
    base_features = build_metropt_causal_features(
        base_labeled,
        continuous_columns=CONTINUOUS_SENSORS,
        state_columns=BINARY_STATE_COLUMNS,
        lag_steps=cfg.lag_steps,
        rolling_windows=cfg.rolling_windows,
        trend_window=cfg.trend_window,
        min_periods=cfg.min_periods,
        state_transition_count_window=cfg.state_transition_count_window,
    )

    horizon_outputs: list[dict[str, Any]] = []
    for horizon_hours in cfg.horizons_hours:
        horizon_outputs.append(
            _run_horizon_robustness(
                raw=raw,
                base_features=base_features,
                output_dir=run_dir / f"horizon_{_horizon_slug(horizon_hours)}",
                config=cfg,
                horizon_hours=horizon_hours,
            )
        )

    metrics = pd.concat(
        [cast(pd.DataFrame, output["metrics"]) for output in horizon_outputs],
        ignore_index=True,
    )
    thresholds = pd.concat(
        [cast(pd.DataFrame, output["thresholds"]) for output in horizon_outputs],
        ignore_index=True,
    )
    threshold_sensitivity = pd.concat(
        [cast(pd.DataFrame, output["threshold_sensitivity"]) for output in horizon_outputs],
        ignore_index=True,
    )
    event_results = pd.concat(
        [cast(pd.DataFrame, output["event_results"]) for output in horizon_outputs],
        ignore_index=True,
    )
    false_alarm_results = pd.concat(
        [cast(pd.DataFrame, output["false_alarm_results"]) for output in horizon_outputs],
        ignore_index=True,
    )
    target_split_counts = pd.concat(
        [cast(pd.DataFrame, output["split_counts"]) for output in horizon_outputs],
        ignore_index=True,
    )
    target_event_counts = pd.concat(
        [cast(pd.DataFrame, output["event_target_counts"]) for output in horizon_outputs],
        ignore_index=True,
    )
    distribution_shift = pd.concat(
        [cast(pd.DataFrame, output["distribution_shift"]) for output in horizon_outputs],
        ignore_index=True,
    )
    predictions = pd.concat(
        [cast(pd.DataFrame, output["predictions"]) for output in horizon_outputs],
        ignore_index=True,
    )

    metrics.to_csv(run_dir / "metrics.csv", index=False)
    thresholds.to_csv(run_dir / "thresholds.csv", index=False)
    threshold_sensitivity.to_csv(run_dir / "threshold_sensitivity.csv", index=False)
    event_results.to_csv(run_dir / "event_level_results.csv", index=False)
    false_alarm_results.to_csv(run_dir / "false_alarm_analysis.csv", index=False)
    target_split_counts.to_csv(run_dir / "target_split_counts.csv", index=False)
    target_event_counts.to_csv(run_dir / "target_event_counts.csv", index=False)
    distribution_shift.to_csv(run_dir / "distribution_shift.csv", index=False)
    _write_dataframe(predictions, run_dir / "predictions")
    _save_json(
        run_dir / "leakage_audit.json",
        {str(output["horizon_hours"]): output["leakage_audit"] for output in horizon_outputs},
    )

    best = select_best_candidate(metrics)
    best_prediction_frame = predictions[
        predictions["candidate_id"].eq(best["candidate_id"]) & predictions["split"].eq("test")
    ].copy()
    july_event = _held_out_july_event(cfg.failure_events)
    july_analysis = analyze_july_failure(
        raw=raw,
        predictions=best_prediction_frame,
        event=july_event,
        horizon_hours=float(best["horizon_hours"]),
        threshold=float(best["threshold"]),
    )
    _save_json(run_dir / "july_failure_analysis.json", july_analysis)
    _plot_july_failure_analysis(
        raw=raw,
        predictions=best_prediction_frame,
        event=july_event,
        horizon_hours=float(best["horizon_hours"]),
        threshold=float(best["threshold"]),
        output_path=figures_dir / "july_failure_analysis.png",
    )
    best_shift = distribution_shift[
        distribution_shift["candidate_id"].eq(best["candidate_id"])
    ].copy()
    best_shift.to_csv(run_dir / "best_candidate_distribution_shift.csv", index=False)
    _plot_distribution_shift(best_shift, figures_dir / "distribution_shift_top_features.png")
    _plot_horizon_comparison(metrics, figures_dir / "horizon_model_comparison.png")
    _plot_threshold_sensitivity(
        threshold_sensitivity,
        figures_dir / "threshold_sensitivity_validation.png",
    )

    final_tree = _final_tree_from_horizon_outputs(horizon_outputs, best)
    shap_status = _write_shap_artifacts(
        final_tree,
        x_reference=cast(pd.DataFrame, final_tree["x_reference"]),
        output_dir=run_dir / "shap",
        config=cfg.shap,
    )
    _save_json(run_dir / "shap" / "status.json", shap_status)

    summary = {
        "run_id": resolved_run_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "failure_event_summary": event_summary,
        "horizons_hours": list(cfg.horizons_hours),
        "best_candidate": best,
        "july_failure_analysis": july_analysis,
        "final_decision": _scientific_decision(best, event_summary, july_analysis),
        "shap": shap_status,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _save_json(run_dir / "metrics.json", summary)
    _write_run_report(run_dir, summary=summary, metrics=metrics)
    return MetroPTRobustnessExperimentResult(run_dir=run_dir, metrics=summary)


def audit_processed_metropt_dataset(root: Path) -> dict[str, Any]:
    """Read processed MetroPT parquet metadata without loading the full table."""

    directory = root / "data" / "processed" / "metrics" / "metropt"
    parquet_files = sorted(directory.glob("*.parquet"))
    if not parquet_files:
        return {"status": "missing", "path": str(directory)}
    row_count = 0
    row_groups = 0
    schema_names: list[str] = []
    files: list[dict[str, Any]] = []
    for path in parquet_files:
        parquet_file = pq.ParquetFile(path)
        rows = int(parquet_file.metadata.num_rows)
        groups = int(parquet_file.metadata.num_row_groups)
        row_count += rows
        row_groups += groups
        if not schema_names:
            schema_names = list(parquet_file.schema.names)
        files.append(
            {
                "path": str(path),
                "rows": rows,
                "row_groups": groups,
                "bytes": int(path.stat().st_size),
            }
        )
    return {
        "status": "available",
        "path": str(directory),
        "file_count": len(parquet_files),
        "row_count": row_count,
        "row_groups": row_groups,
        "columns": schema_names,
        "files": files,
    }


def characterize_failure_events(
    frame: pd.DataFrame,
    *,
    failure_events: tuple[FailureEvent, ...],
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Create a complete table of curated MetroPT failure intervals."""

    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    records: list[dict[str, Any]] = []
    sorted_events = tuple(sorted(failure_events, key=lambda event: event.start))
    previous: FailureEvent | None = None
    for index, event in enumerate(sorted_events):
        observation_count = int(((timestamps >= event.start) & (timestamps <= event.end)).sum())
        overlaps_previous = previous is not None and event.start <= previous.end
        split_role = _split_role_for_timestamp(
            event.start,
            train_end=train_end,
            validation_end=validation_end,
        )
        records.append(
            {
                "event_id": event.event_id,
                "start_time": event.start.isoformat(),
                "end_time": event.end.isoformat(),
                "duration_hours": _hours_between(event.start, event.end),
                "number_of_observations": observation_count,
                "multiple_rows_same_failure": observation_count > 1,
                "failure_type": event.failure_type,
                "severity": event.severity,
                "split_role": split_role,
                "is_held_out_test_event": split_role == "test",
                "overlaps_previous_event": overlaps_previous,
                "hours_since_previous_start": (
                    None if previous is None else _hours_between(previous.start, event.start)
                ),
                "hours_since_previous_end": (
                    None if previous is None else _hours_between(previous.end, event.start)
                ),
                "source": event.source,
                "event_order": index + 1,
            }
        )
        previous = event
    return pd.DataFrame(records)


def summarize_failure_event_structure(event_table: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether the curated events support supervised learning."""

    event_count = int(event_table.shape[0])
    held_out_count = int(event_table["is_held_out_test_event"].sum()) if event_count else 0
    held_out_july = event_table[
        event_table["is_held_out_test_event"]
        & event_table["start_time"].map(lambda value: pd.Timestamp(value).month == 7)
    ]
    overlap_count = int(event_table["overlaps_previous_event"].sum()) if event_count else 0
    type_count = int(event_table["failure_type"].nunique()) if event_count else 0
    min_gap = event_table["hours_since_previous_start"].dropna()
    return {
        "failure_event_count": event_count,
        "held_out_test_event_count": held_out_count,
        "held_out_july_failure_is_unique": int(held_out_july.shape[0]) == 1,
        "overlapping_failure_intervals": overlap_count,
        "failure_type_count": type_count,
        "all_events_same_failure_type": type_count == 1,
        "minimum_hours_between_failure_starts": (None if min_gap.empty else float(min_gap.min())),
        "supervised_learning_sufficiency": (
            "insufficient_for_reliable_supervised_learning"
            if event_count < 10 or type_count <= 1
            else "potentially_sufficient"
        ),
        "note": (
            "The target uses four report-curated air-leak intervals. This is enough for "
            "a case-study robustness audit, not enough for a high-confidence supervised "
            "predictive-maintenance benchmark."
        ),
    }


def analyze_prefailure_buffers(
    frame: pd.DataFrame,
    failure_events: tuple[FailureEvent, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize sensor behavior before, during, and after each failure."""

    stages = (
        ("pre_6h_to_1h", pd.Timedelta(hours=-6), pd.Timedelta(hours=-1), "start"),
        ("pre_1h_to_10min", pd.Timedelta(hours=-1), pd.Timedelta(minutes=-10), "start"),
        ("pre_10min_to_start", pd.Timedelta(minutes=-10), pd.Timedelta(0), "start"),
        ("during_failure", pd.Timedelta(0), pd.Timedelta(0), "active"),
        ("post_1h", pd.Timedelta(0), pd.Timedelta(hours=1), "end"),
    )
    ordered = frame.sort_values("timestamp").reset_index(drop=True).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    sensor_records: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    for event in failure_events:
        for stage_name, start_offset, end_offset, anchor in stages:
            if anchor == "active":
                start = event.start
                end = event.end
            elif anchor == "end":
                start = event.end + start_offset
                end = event.end + end_offset
            else:
                start = event.start + start_offset
                end = event.start + end_offset
            window = ordered[(ordered["timestamp"] >= start) & (ordered["timestamp"] < end)]
            if anchor == "active":
                window = ordered[(ordered["timestamp"] >= start) & (ordered["timestamp"] <= end)]
            for sensor in CONTINUOUS_SENSORS:
                values = pd.to_numeric(window[sensor], errors="coerce")
                sensor_records.append(
                    {
                        "event_id": event.event_id,
                        "stage": stage_name,
                        "window_start": start.isoformat(),
                        "window_end": end.isoformat(),
                        "sensor": sensor,
                        "rows": int(values.shape[0]),
                        "mean": _nullable_float(values.mean()),
                        "std": _nullable_float(values.std(ddof=0)),
                        "min": _nullable_float(values.min()),
                        "max": _nullable_float(values.max()),
                    }
                )
            for state in BINARY_STATE_COLUMNS:
                values = pd.to_numeric(window[state], errors="coerce")
                transitions = int(values.ne(values.shift(1)).sum() - (0 if values.empty else 1))
                state_records.append(
                    {
                        "event_id": event.event_id,
                        "stage": stage_name,
                        "state": state,
                        "rows": int(values.shape[0]),
                        "mean_state": _nullable_float(values.mean()),
                        "transition_count": max(transitions, 0),
                    }
                )
    sensor_summary = pd.DataFrame(sensor_records)
    state_summary = pd.DataFrame(state_records)
    effects = _prefailure_effect_sizes(sensor_summary)
    return sensor_summary, state_summary, effects


def select_best_candidate(metrics: pd.DataFrame) -> dict[str, Any]:
    """Select the best model/horizon using validation metrics only."""

    ordered = metrics.sort_values(
        [
            "validation_warning_coverage",
            "validation_f1",
            "validation_pr_auc",
            "validation_false_alarms_per_day",
            "validation_false_positive_rate",
        ],
        ascending=[False, False, False, True, True],
    )
    return cast(dict[str, Any], ordered.iloc[0].to_dict())


def analyze_july_failure(
    *,
    raw: pd.DataFrame,
    predictions: pd.DataFrame,
    event: FailureEvent,
    horizon_hours: float,
    threshold: float,
) -> dict[str, Any]:
    """Analyze probability and sensor behavior around the held-out July failure."""

    horizon = pd.Timedelta(hours=horizon_hours)
    prediction_frame = predictions.copy()
    prediction_frame["timestamp"] = pd.to_datetime(prediction_frame["timestamp"], errors="raise")
    warning_window = prediction_frame[
        (prediction_frame["timestamp"] >= event.start - horizon)
        & (prediction_frame["timestamp"] < event.start)
    ]
    baseline_window = prediction_frame[
        (prediction_frame["timestamp"] >= event.start - pd.Timedelta(hours=24))
        & (prediction_frame["timestamp"] < event.start - horizon)
    ]
    baseline_probabilities = baseline_window["probability"].astype(float)
    warning_probabilities = warning_window["probability"].astype(float)
    threshold_crossings = warning_window[warning_probabilities >= threshold]
    increase_threshold = _robust_probability_increase_threshold(baseline_probabilities)
    increases = warning_window[warning_probabilities >= increase_threshold]
    raw_frame = raw.copy()
    raw_frame["timestamp"] = pd.to_datetime(raw_frame["timestamp"], errors="raise")
    sensor_window = raw_frame[
        (raw_frame["timestamp"] >= event.start - pd.Timedelta(hours=24))
        & (raw_frame["timestamp"] <= event.end + pd.Timedelta(hours=1))
    ]
    sensor_ranges = {
        sensor: {
            "min": _nullable_float(sensor_window[sensor].min()),
            "max": _nullable_float(sensor_window[sensor].max()),
            "mean": _nullable_float(sensor_window[sensor].mean()),
        }
        for sensor in CONTINUOUS_SENSORS
    }
    return {
        "event_id": event.event_id,
        "horizon_hours": horizon_hours,
        "threshold": threshold,
        "warning_window_rows": int(warning_window.shape[0]),
        "warnings_in_horizon": int(threshold_crossings.shape[0]),
        "first_threshold_crossing": (
            None
            if threshold_crossings.empty
            else pd.Timestamp(threshold_crossings["timestamp"].min()).isoformat()
        ),
        "first_probability_increase_time": (
            None if increases.empty else pd.Timestamp(increases["timestamp"].min()).isoformat()
        ),
        "baseline_probability_median": _nullable_float(baseline_probabilities.median()),
        "baseline_probability_p95": _nullable_float(baseline_probabilities.quantile(0.95)),
        "warning_probability_max": _nullable_float(warning_probabilities.max()),
        "probability_increase_threshold": increase_threshold,
        "probability_increased_before_failure": not increases.empty,
        "sensor_ranges_24h_before_to_1h_after": sensor_ranges,
    }


def audit_horizon_target_integrity(
    labeled: pd.DataFrame,
    modeling: pd.DataFrame,
    *,
    failure_events: tuple[FailureEvent, ...],
    horizon_hours: float,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> dict[str, Any]:
    """Check target construction and split boundaries for leakage risks."""

    horizon = pd.Timedelta(hours=horizon_hours)
    event_start_by_id = {event.event_id: event.start for event in failure_events}
    invalid_positive_rows = 0
    positive_rows = labeled[labeled["target"].eq(1)].copy()
    for row in positive_rows.itertuples(index=False):
        event_id = str(row.future_failure_event_id)
        timestamp = pd.Timestamp(row.timestamp)
        event_start = event_start_by_id.get(event_id)
        if event_start is None or timestamp >= event_start or timestamp < event_start - horizon:
            invalid_positive_rows += 1
    boundaries = (train_end, validation_end)
    boundary_inside_failure = sum(
        int(any(event.start <= boundary <= event.end for event in failure_events))
        for boundary in boundaries
    )
    target_window_crosses_split_boundary = 0
    for event in failure_events:
        role = _split_role_for_timestamp(
            event.start,
            train_end=train_end,
            validation_end=validation_end,
        )
        window_start = event.start - horizon
        if role == "validation" and window_start < train_end:
            target_window_crosses_split_boundary += 1
        if role == "test" and window_start < validation_end:
            target_window_crosses_split_boundary += 1
    overlap_count = 0
    sorted_events = tuple(sorted(failure_events, key=lambda event: event.start))
    for previous, current in zip(sorted_events, sorted_events[1:], strict=False):
        overlap_count += int(current.start <= previous.end)
    return {
        "horizon_hours": horizon_hours,
        "positive_rows": int(positive_rows.shape[0]),
        "modeling_rows": int(modeling.shape[0]),
        "active_failure_rows_labeled_positive": int(
            (labeled["target"].eq(1) & labeled["is_failure_period"].astype(bool)).sum()
        ),
        "active_failure_rows_in_modeling_frame": int(modeling["is_failure_period"].sum()),
        "invalid_positive_target_rows": invalid_positive_rows,
        "split_boundary_inside_failure_interval_count": boundary_inside_failure,
        "target_window_crosses_split_boundary_count": target_window_crosses_split_boundary,
        "overlapping_failure_interval_count": overlap_count,
        "leakage_detected": any(
            value > 0
            for value in (
                invalid_positive_rows,
                int((labeled["target"].eq(1) & labeled["is_failure_period"].astype(bool)).sum()),
                int(modeling["is_failure_period"].sum()),
                boundary_inside_failure,
                target_window_crosses_split_boundary,
                overlap_count,
            )
        ),
    }


def false_alarm_analysis(
    predictions: pd.DataFrame,
    *,
    failure_events: tuple[FailureEvent, ...],
    horizon_hours: float,
    threshold: float,
) -> dict[str, Any]:
    """Measure false warning volume and persistent warning episodes."""

    frame = predictions.sort_values("timestamp").copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["warning"] = frame["probability"].astype(float) >= threshold
    protected = np.zeros(frame.shape[0], dtype=bool)
    horizon = pd.Timedelta(hours=horizon_hours)
    timestamps = frame["timestamp"]
    for event in failure_events:
        protected |= ((timestamps >= event.start - horizon) & (timestamps < event.start)).to_numpy()
    false_warning_frame = frame[frame["warning"] & ~protected].copy()
    negative_rows = int((frame["target"].astype(int) == 0).sum())
    duration_days = _duration_days(frame["timestamp"])
    episode_count, median_episode_minutes, max_episode_minutes = _persistent_warning_episodes(
        false_warning_frame
    )
    false_warnings = int(false_warning_frame.shape[0])
    return {
        "threshold": threshold,
        "warning_rows": int(frame["warning"].sum()),
        "false_warnings": false_warnings,
        "negative_rows": negative_rows,
        "false_alarm_rate": float(false_warnings / negative_rows) if negative_rows else 0.0,
        "false_alarms_per_hour": (
            float(false_warnings / (duration_days * 24)) if duration_days > 0 else 0.0
        ),
        "false_alarms_per_day": (
            float(false_warnings / duration_days) if duration_days > 0 else 0.0
        ),
        "persistent_warning_episodes": episode_count,
        "median_persistent_warning_minutes": median_episode_minutes,
        "max_persistent_warning_minutes": max_episode_minutes,
    }


def _run_horizon_robustness(
    *,
    raw: pd.DataFrame,
    base_features: pd.DataFrame,
    output_dir: Path,
    config: MetroPTRobustnessExperimentConfig,
    horizon_hours: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    labeled = build_metropt_failure_target(
        raw,
        failure_events=config.failure_events,
        horizon_hours=horizon_hours,
    )
    features = base_features.copy(deep=False)
    for column in (
        "target",
        "future_failure_event_id",
        "is_failure_period",
        "eligible_for_prediction",
    ):
        features[column] = labeled[column].to_numpy()
    metro_config = MetroPTExperimentConfig(
        horizon_hours=horizon_hours,
        prediction_stride_rows=config.prediction_stride_rows,
        train_end=config.train_end,
        validation_end=config.validation_end,
        lag_steps=config.lag_steps,
        rolling_windows=config.rolling_windows,
        trend_window=config.trend_window,
        min_periods=config.min_periods,
        max_train_rows=config.max_train_rows,
        negative_to_positive_ratio=config.negative_to_positive_ratio,
        threshold_grid=config.threshold_grid,
        failure_events=config.failure_events,
    )
    modeling = prepare_horizon_modeling_frame(features, metro_config)
    feature_columns = tuple(
        column for column in modeling.columns if column not in _METROPT_NON_FEATURE_COLUMNS
    )
    (output_dir / "feature_columns.json").write_text(
        json.dumps(list(feature_columns), indent=2),
        encoding="utf-8",
    )
    split_counts = _split_counts(modeling, horizon_hours)
    split_counts.to_csv(output_dir / "target_split_counts.csv", index=False)
    event_target_counts = _event_target_counts(modeling, horizon_hours)
    event_target_counts.to_csv(output_dir / "target_event_counts.csv", index=False)
    leakage_audit = audit_horizon_target_integrity(
        labeled,
        modeling,
        failure_events=config.failure_events,
        horizon_hours=horizon_hours,
        train_end=pd.Timestamp(config.train_end),
        validation_end=pd.Timestamp(config.validation_end),
    )
    _save_json(output_dir / "leakage_audit.json", leakage_audit)

    train_frame = modeling[modeling["split"].eq("train")]
    validation_frame = modeling[modeling["split"].eq("validation")]
    test_frame = modeling[modeling["split"].eq("test")]
    train_sample = _sample_training_for_horizon(train_frame, config)
    positive_weight = _positive_weight(train_sample["target"].to_numpy(dtype=int))
    model_results = _run_model_family(
        dataset="metropt",
        x_train=train_sample.loc[:, list(feature_columns)],
        y_train=train_sample["target"].to_numpy(dtype=int),
        x_validation=validation_frame.loc[:, list(feature_columns)],
        y_validation=validation_frame["target"].to_numpy(dtype=int),
        x_test=test_frame.loc[:, list(feature_columns)],
        y_test=test_frame["target"].to_numpy(dtype=int),
        preprocessor=_numeric_preprocessor(feature_columns),
        model_config=config.models,
        threshold_grid=config.threshold_grid,
        positive_weight=positive_weight,
        models_dir=models_dir,
    )

    metrics_rows: list[dict[str, Any]] = []
    thresholds_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    false_alarm_rows: list[dict[str, Any]] = []
    distribution_frames: list[pd.DataFrame] = []
    for result in model_results:
        candidate_id = _candidate_id(horizon_hours, str(result["model"]))
        threshold = float(result["threshold_selection"]["threshold"])
        all_split_predictions: list[pd.DataFrame] = []
        for split_name, split_frame in (
            ("train", train_frame),
            ("validation", validation_frame),
            ("test", test_frame),
        ):
            split_predictions = _predictions_for_split(
                split_frame,
                feature_columns=feature_columns,
                pipeline=cast(Pipeline, result["pipeline"]),
                threshold=threshold,
                horizon_hours=horizon_hours,
                model=str(result["model"]),
                candidate_id=candidate_id,
            )
            all_split_predictions.append(split_predictions)
            if split_name in {"validation", "test"}:
                threshold_rows.extend(
                    _threshold_sensitivity_rows(
                        split_predictions,
                        split=split_name,
                        model=str(result["model"]),
                        candidate_id=candidate_id,
                        horizon_hours=horizon_hours,
                        threshold_grid=config.threshold_grid,
                        failure_events=_events_for_split(
                            config.failure_events,
                            split=split_name,
                            train_end=pd.Timestamp(config.train_end),
                            validation_end=pd.Timestamp(config.validation_end),
                        ),
                    )
                )
            event_split_frame = _event_level_rows_for_split(
                split_predictions,
                split=split_name,
                model=str(result["model"]),
                candidate_id=candidate_id,
                horizon_hours=horizon_hours,
                threshold=threshold,
                failure_events=_events_for_split(
                    config.failure_events,
                    split=split_name,
                    train_end=pd.Timestamp(config.train_end),
                    validation_end=pd.Timestamp(config.validation_end),
                ),
            )
            if not event_split_frame.empty:
                event_frames.append(event_split_frame)
            false_alarm = false_alarm_analysis(
                split_predictions,
                failure_events=_events_for_split(
                    config.failure_events,
                    split=split_name,
                    train_end=pd.Timestamp(config.train_end),
                    validation_end=pd.Timestamp(config.validation_end),
                ),
                horizon_hours=horizon_hours,
                threshold=threshold,
            )
            false_alarm.update(
                {
                    "candidate_id": candidate_id,
                    "horizon_hours": horizon_hours,
                    "model": result["model"],
                    "split": split_name,
                }
            )
            false_alarm_rows.append(false_alarm)
        combined_predictions = pd.concat(all_split_predictions, ignore_index=True)
        prediction_frames.append(combined_predictions)
        validation_predictions = combined_predictions[
            combined_predictions["split"].eq("validation")
        ]
        test_predictions = combined_predictions[combined_predictions["split"].eq("test")]
        validation_metrics = classification_metrics(
            validation_predictions["target"].to_numpy(dtype=int),
            validation_predictions["y_pred"].to_numpy(dtype=int),
            validation_predictions["probability"].to_numpy(dtype=float),
        ).to_dict()
        test_metrics = classification_metrics(
            test_predictions["target"].to_numpy(dtype=int),
            test_predictions["y_pred"].to_numpy(dtype=int),
            test_predictions["probability"].to_numpy(dtype=float),
        ).to_dict()
        validation_summary, _ = compute_early_warning_metrics(
            validation_predictions,
            failure_events=_events_for_split(
                config.failure_events,
                split="validation",
                train_end=pd.Timestamp(config.train_end),
                validation_end=pd.Timestamp(config.validation_end),
            ),
            horizon_hours=horizon_hours,
            threshold=threshold,
        )
        test_summary, _ = compute_early_warning_metrics(
            test_predictions,
            failure_events=_events_for_split(
                config.failure_events,
                split="test",
                train_end=pd.Timestamp(config.train_end),
                validation_end=pd.Timestamp(config.validation_end),
            ),
            horizon_hours=horizon_hours,
            threshold=threshold,
        )
        validation_false_alarm = false_alarm_analysis(
            validation_predictions,
            failure_events=_events_for_split(
                config.failure_events,
                split="validation",
                train_end=pd.Timestamp(config.train_end),
                validation_end=pd.Timestamp(config.validation_end),
            ),
            horizon_hours=horizon_hours,
            threshold=threshold,
        )
        test_false_alarm = false_alarm_analysis(
            test_predictions,
            failure_events=_events_for_split(
                config.failure_events,
                split="test",
                train_end=pd.Timestamp(config.train_end),
                validation_end=pd.Timestamp(config.validation_end),
            ),
            horizon_hours=horizon_hours,
            threshold=threshold,
        )
        metrics_row = {
            "candidate_id": candidate_id,
            "horizon_hours": horizon_hours,
            "model": result["model"],
            "threshold": threshold,
            "training_seconds": result["test_metrics"]["training_seconds"],
            "inference_seconds": result["test_metrics"]["inference_seconds"],
            "train_rows": int(train_sample.shape[0]),
            "validation_rows": int(validation_frame.shape[0]),
            "test_rows": int(test_frame.shape[0]),
            "validation_warning_coverage": validation_summary.warning_coverage,
            "validation_median_lead_time_hours": validation_summary.median_lead_time_hours,
            "validation_false_alarms_per_day": validation_false_alarm["false_alarms_per_day"],
            "test_warning_coverage": test_summary.warning_coverage,
            "test_median_lead_time_hours": test_summary.median_lead_time_hours,
            "test_false_alarms_per_day": test_false_alarm["false_alarms_per_day"],
        }
        metrics_row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        metrics_row.update({f"test_{key}": value for key, value in test_metrics.items()})
        metrics_rows.append(metrics_row)
        threshold_record = dict(cast(dict[str, Any], result["threshold_selection"]))
        threshold_record.update(
            {
                "candidate_id": candidate_id,
                "horizon_hours": horizon_hours,
                "model": result["model"],
            }
        )
        thresholds_rows.append(threshold_record)
        distribution_frames.append(
            distribution_shift_for_candidate(
                modeling,
                feature_columns=feature_columns,
                candidate_id=candidate_id,
                horizon_hours=horizon_hours,
                model=str(result["model"]),
            )
        )

    metrics = pd.DataFrame(metrics_rows)
    thresholds = pd.DataFrame(thresholds_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    threshold_sensitivity = pd.DataFrame(threshold_rows)
    event_results = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(columns=["candidate_id", "event_id"])
    )
    false_alarm_results = pd.DataFrame(false_alarm_rows)
    distribution_shift = pd.concat(distribution_frames, ignore_index=True)

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    thresholds.to_csv(output_dir / "thresholds.csv", index=False)
    threshold_sensitivity.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    event_results.to_csv(output_dir / "event_level_results.csv", index=False)
    false_alarm_results.to_csv(output_dir / "false_alarm_analysis.csv", index=False)
    distribution_shift.to_csv(output_dir / "distribution_shift.csv", index=False)
    _write_dataframe(predictions, output_dir / "predictions")

    for result in model_results:
        result["x_reference"] = test_frame.loc[:, list(feature_columns)]
    return {
        "horizon_hours": horizon_hours,
        "metrics": metrics,
        "thresholds": thresholds,
        "threshold_sensitivity": threshold_sensitivity,
        "event_results": event_results,
        "false_alarm_results": false_alarm_results,
        "split_counts": split_counts,
        "event_target_counts": event_target_counts,
        "distribution_shift": distribution_shift,
        "predictions": predictions,
        "leakage_audit": leakage_audit,
        "model_results": model_results,
    }


def prepare_horizon_modeling_frame(
    features: pd.DataFrame,
    config: MetroPTExperimentConfig,
) -> pd.DataFrame:
    """Prepare a horizon-specific frame using the Phase 7 temporal policy."""

    from aegis_ai.ml.prediction.failure_prediction import prepare_metropt_modeling_frame

    return prepare_metropt_modeling_frame(features, config)


def distribution_shift_for_candidate(
    modeling: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    candidate_id: str,
    horizon_hours: float,
    model: str,
) -> pd.DataFrame:
    """Compare held-out July pre-failure features against train pre-failure rows."""

    train_positive = modeling[modeling["split"].eq("train") & modeling["target"].eq(1)]
    test_positive = modeling[modeling["split"].eq("test") & modeling["target"].eq(1)]
    records: list[dict[str, Any]] = []
    for feature in feature_columns:
        train_values = pd.to_numeric(train_positive[feature], errors="coerce").dropna()
        test_values = pd.to_numeric(test_positive[feature], errors="coerce").dropna()
        if train_values.empty or test_values.empty:
            continue
        train_std = float(train_values.std(ddof=0))
        pooled = max(train_std, 1e-8)
        train_p01 = float(train_values.quantile(0.01))
        train_p99 = float(train_values.quantile(0.99))
        test_mean = float(test_values.mean())
        records.append(
            {
                "candidate_id": candidate_id,
                "horizon_hours": horizon_hours,
                "model": model,
                "feature": feature,
                "train_positive_mean": float(train_values.mean()),
                "train_positive_std": train_std,
                "test_positive_mean": test_mean,
                "test_positive_std": float(test_values.std(ddof=0)),
                "standardized_mean_difference": float(
                    (test_mean - float(train_values.mean())) / pooled
                ),
                "train_positive_p01": train_p01,
                "train_positive_p99": train_p99,
                "test_mean_outside_train_p01_p99": test_mean < train_p01 or test_mean > train_p99,
            }
        )
    return (
        pd.DataFrame(records)
        .assign(
            abs_standardized_mean_difference=lambda frame: frame[
                "standardized_mean_difference"
            ].abs()
        )
        .sort_values("abs_standardized_mean_difference", ascending=False)
        .reset_index(drop=True)
    )


def _predictions_for_split(
    split_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    pipeline: Pipeline,
    threshold: float,
    horizon_hours: float,
    model: str,
    candidate_id: str,
) -> pd.DataFrame:
    probabilities = _positive_probabilities(pipeline, split_frame.loc[:, list(feature_columns)])
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "horizon_hours": horizon_hours,
            "model": model,
            "split": split_frame["split"].to_numpy(dtype=object),
            "timestamp": split_frame["timestamp"].to_numpy(),
            "sequence_index": split_frame["sequence_index"].to_numpy(dtype=np.int64),
            "future_failure_event_id": split_frame["future_failure_event_id"].to_numpy(
                dtype=object
            ),
            "target": split_frame["target"].to_numpy(dtype=np.int8),
            "y_true": split_frame["target"].to_numpy(dtype=np.int8),
            "probability": probabilities.astype(np.float32),
            "threshold": np.full(probabilities.shape[0], threshold, dtype=np.float32),
            "y_pred": predictions.astype(np.int8),
        }
    )


def _threshold_sensitivity_rows(
    predictions: pd.DataFrame,
    *,
    split: str,
    model: str,
    candidate_id: str,
    horizon_hours: float,
    threshold_grid: tuple[float, ...],
    failure_events: tuple[FailureEvent, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true = predictions["target"].to_numpy(dtype=int)
    probabilities = predictions["probability"].to_numpy(dtype=float)
    for threshold in threshold_grid:
        y_pred = (probabilities >= threshold).astype(int)
        metrics = classification_metrics(y_true, y_pred, probabilities).to_dict()
        lead_summary, _ = compute_early_warning_metrics(
            predictions.assign(y_pred=y_pred),
            failure_events=failure_events,
            horizon_hours=horizon_hours,
            threshold=threshold,
        )
        false_alarm = false_alarm_analysis(
            predictions,
            failure_events=failure_events,
            horizon_hours=horizon_hours,
            threshold=threshold,
        )
        row = {
            "candidate_id": candidate_id,
            "horizon_hours": horizon_hours,
            "model": model,
            "split": split,
            "threshold": threshold,
            "warning_coverage": lead_summary.warning_coverage,
            "median_lead_time_hours": lead_summary.median_lead_time_hours,
            "false_alarms_per_day": false_alarm["false_alarms_per_day"],
            "persistent_warning_episodes": false_alarm["persistent_warning_episodes"],
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _event_level_rows_for_split(
    predictions: pd.DataFrame,
    *,
    split: str,
    model: str,
    candidate_id: str,
    horizon_hours: float,
    threshold: float,
    failure_events: tuple[FailureEvent, ...],
) -> pd.DataFrame:
    if not failure_events:
        return pd.DataFrame()
    summary, episodes = compute_early_warning_metrics(
        predictions,
        failure_events=failure_events,
        horizon_hours=horizon_hours,
        threshold=threshold,
    )
    if episodes.empty:
        return episodes
    episodes = episodes.copy()
    episodes.insert(0, "candidate_id", candidate_id)
    episodes.insert(1, "horizon_hours", horizon_hours)
    episodes.insert(2, "model", model)
    episodes.insert(3, "split", split)
    episodes["event_detection_rate_for_split"] = summary.warning_coverage
    episodes["false_alarm_rate_for_split"] = summary.false_alarm_rate
    episodes["false_alarms_per_day_for_split"] = summary.false_alarms_per_day
    return episodes


def _split_counts(modeling: pd.DataFrame, horizon_hours: float) -> pd.DataFrame:
    counts = (
        modeling.groupby("split", observed=True)["target"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "sample_count",
                "sum": "positive_samples",
                "mean": "positive_ratio",
            }
        )
    )
    counts.insert(0, "horizon_hours", horizon_hours)
    counts["negative_samples"] = counts["sample_count"] - counts["positive_samples"]
    return counts


def _event_target_counts(modeling: pd.DataFrame, horizon_hours: float) -> pd.DataFrame:
    positives = modeling[modeling["target"].eq(1)].copy()
    if positives.empty:
        return pd.DataFrame(
            columns=[
                "horizon_hours",
                "future_failure_event_id",
                "split",
                "positive_samples",
            ]
        )
    counts = (
        positives.groupby(["future_failure_event_id", "split"], observed=True)["target"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "positive_prediction_rows", "sum": "target_sum"})
    )
    counts.insert(0, "horizon_hours", horizon_hours)
    return counts


def _sample_training_for_horizon(
    train_frame: pd.DataFrame,
    config: MetroPTRobustnessExperimentConfig,
) -> pd.DataFrame:
    from aegis_ai.ml.prediction.failure_prediction import sample_temporal_training_rows

    return sample_temporal_training_rows(
        train_frame,
        target_column="target",
        max_rows=config.max_train_rows,
        negative_to_positive_ratio=config.negative_to_positive_ratio,
    )


def _events_for_split(
    events: tuple[FailureEvent, ...],
    *,
    split: str,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[FailureEvent, ...]:
    if split == "train":
        return _events_in_split(events, start=pd.Timestamp.min, end=train_end - pd.Timedelta("1ns"))
    if split == "validation":
        return _events_in_split(events, start=train_end, end=validation_end - pd.Timedelta("1ns"))
    if split == "test":
        return _events_in_split(events, start=validation_end, end=pd.Timestamp.max)
    raise ValueError(f"unknown split: {split}")


def _split_role_for_timestamp(
    timestamp: pd.Timestamp,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> str:
    if timestamp < train_end:
        return "train"
    if timestamp < validation_end:
        return "validation"
    return "test"


def _persistent_warning_episodes(frame: pd.DataFrame) -> tuple[int, float | None, float | None]:
    if frame.empty:
        return 0, None, None
    ordered = frame.sort_values("timestamp").copy()
    timestamps = pd.to_datetime(ordered["timestamp"], errors="raise")
    deltas = timestamps.diff().dt.total_seconds()
    median_delta = deltas.dropna().median()
    max_gap = float(median_delta * 2) if pd.notna(median_delta) and median_delta > 0 else 120.0
    group_id = (deltas.isna() | (deltas > max_gap)).cumsum()
    durations: list[float] = []
    for _, group in ordered.groupby(group_id, observed=True):
        group_timestamps = pd.to_datetime(group["timestamp"], errors="raise")
        if group_timestamps.shape[0] <= 1:
            durations.append(0.0)
        else:
            durations.append(
                float((group_timestamps.max() - group_timestamps.min()).total_seconds() / 60)
            )
    return int(len(durations)), float(np.median(durations)), float(np.max(durations))


def _prefailure_effect_sizes(sensor_summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    baseline = sensor_summary[sensor_summary["stage"].eq("pre_6h_to_1h")]
    immediate = sensor_summary[sensor_summary["stage"].eq("pre_10min_to_start")]
    for _, row in immediate.iterrows():
        baseline_row = baseline[
            baseline["event_id"].eq(row["event_id"]) & baseline["sensor"].eq(row["sensor"])
        ]
        if baseline_row.empty:
            continue
        base = baseline_row.iloc[0]
        pooled_std = max(float(base["std"] or 0.0), 1e-8)
        mean_delta = float(row["mean"] or 0.0) - float(base["mean"] or 0.0)
        records.append(
            {
                "event_id": row["event_id"],
                "sensor": row["sensor"],
                "baseline_stage": "pre_6h_to_1h",
                "comparison_stage": "pre_10min_to_start",
                "mean_delta": mean_delta,
                "standardized_mean_delta": mean_delta / pooled_std,
            }
        )
    return (
        pd.DataFrame(records)
        .assign(abs_standardized_mean_delta=lambda frame: frame["standardized_mean_delta"].abs())
        .sort_values("abs_standardized_mean_delta", ascending=False)
        .reset_index(drop=True)
    )


def _final_tree_from_horizon_outputs(
    horizon_outputs: list[dict[str, Any]],
    best: dict[str, Any],
) -> dict[str, Any]:
    matching_output = next(
        output
        for output in horizon_outputs
        if float(output["horizon_hours"]) == float(best["horizon_hours"])
    )
    model_results = cast(list[dict[str, Any]], matching_output["model_results"])
    tree = _select_final_tree_model(model_results)
    tree["x_reference"] = next(
        result["x_reference"] for result in model_results if result["model"] == tree["model"]
    )
    return tree


def _scientific_decision(
    best: dict[str, Any],
    event_summary: dict[str, Any],
    july_analysis: dict[str, Any],
) -> dict[str, Any]:
    event_rate = float(best["test_warning_coverage"])
    validation_rate = float(best["validation_warning_coverage"])
    event_count = int(event_summary["failure_event_count"])
    false_alarm_day = float(best["test_false_alarms_per_day"])
    if event_rate > 0 and validation_rate > 0 and false_alarm_day < 24 and event_count >= 10:
        decision = "A"
        conclusion = "MetroPT supports useful failure prediction with the corrected target."
    elif event_rate > 0 or validation_rate > 0:
        decision = "B"
        conclusion = (
            "MetroPT supports limited failure prediction but has strong generalization limitations."
        )
    else:
        decision = "C"
        conclusion = (
            "MetroPT does not contain enough consistent predictive signal for reliable "
            "supervised early-warning modeling under the current setup."
        )
    return {
        "decision": decision,
        "conclusion": conclusion,
        "basis": {
            "curated_failure_event_count": event_count,
            "validation_warning_coverage": validation_rate,
            "test_warning_coverage": event_rate,
            "test_false_alarms_per_day": false_alarm_day,
            "july_probability_increased_before_failure": july_analysis[
                "probability_increased_before_failure"
            ],
        },
    }


def _held_out_july_event(events: tuple[FailureEvent, ...]) -> FailureEvent:
    july_events = [event for event in events if event.start.month == 7]
    if not july_events:
        return max(events, key=lambda event: event.start)
    return july_events[0]


def _plot_failure_event_timeline(event_table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 3), constrained_layout=True)
    for index, row in event_table.iterrows():
        start = pd.Timestamp(row["start_time"])
        end = pd.Timestamp(row["end_time"])
        axis.barh(
            y=index,
            width=end - start,
            left=start,
            color="#b8322b" if row["split_role"] == "test" else "#33658a",
            alpha=0.75,
        )
    axis.set_yticks(range(event_table.shape[0]))
    axis.set_yticklabels(event_table["event_id"].astype(str))
    axis.set_title("Curated MetroPT failure events")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore[no-untyped-call]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def _plot_prefailure_sensor_summary(sensor_summary: pd.DataFrame, output_path: Path) -> None:
    sensors = ("TP2", "Oil_temperature", "Motor_current", "Reservoirs")
    stages = (
        "pre_6h_to_1h",
        "pre_1h_to_10min",
        "pre_10min_to_start",
        "during_failure",
        "post_1h",
    )
    frame = sensor_summary[sensor_summary["sensor"].isin(sensors)].copy()
    grouped = frame.groupby(["sensor", "stage"], observed=True)["mean"].mean().reset_index()
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for sensor in sensors:
        group = grouped[grouped["sensor"].eq(sensor)].set_index("stage").reindex(stages)
        axis.plot(stages, group["mean"], marker="o", label=sensor)
    axis.set_title("Mean sensor behavior around curated failures")
    axis.set_ylabel("sensor value")
    axis.tick_params(axis="x", labelrotation=25)
    axis.legend(loc="best")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_horizon_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for model, group in metrics.groupby("model", observed=True):
        axes[0].plot(group["horizon_hours"], group["validation_f1"], marker="o", label=str(model))
        axes[1].plot(
            group["horizon_hours"],
            group["test_warning_coverage"],
            marker="o",
            label=str(model),
        )
    axes[0].set_title("Validation F1 by horizon")
    axes[0].set_xlabel("horizon hours")
    axes[0].set_ylabel("F1")
    axes[1].set_title("Held-out event detection by horizon")
    axes[1].set_xlabel("horizon hours")
    axes[1].set_ylabel("event detection rate")
    for axis in axes:
        axis.legend(loc="best")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_threshold_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    validation = sensitivity[sensitivity["split"].eq("validation")]
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for candidate_id, group in validation.groupby("candidate_id", observed=True):
        axis.plot(group["threshold"], group["f1"], marker="o", linewidth=1, label=str(candidate_id))
    axis.set_xscale("log")
    axis.set_title("Validation threshold sensitivity")
    axis.set_xlabel("threshold")
    axis.set_ylabel("F1")
    axis.legend(loc="upper right", fontsize=7)
    figure.savefig(output_path)
    plt.close(figure)


def _plot_july_failure_analysis(
    *,
    raw: pd.DataFrame,
    predictions: pd.DataFrame,
    event: FailureEvent,
    horizon_hours: float,
    threshold: float,
    output_path: Path,
) -> None:
    horizon = pd.Timedelta(hours=horizon_hours)
    raw_frame = raw.copy()
    raw_frame["timestamp"] = pd.to_datetime(raw_frame["timestamp"], errors="raise")
    sensor_window = raw_frame[
        (raw_frame["timestamp"] >= event.start - pd.Timedelta(hours=12))
        & (raw_frame["timestamp"] <= event.end + pd.Timedelta(hours=2))
    ]
    prediction_frame = predictions.copy()
    prediction_frame["timestamp"] = pd.to_datetime(prediction_frame["timestamp"], errors="raise")
    prediction_window = prediction_frame[
        (prediction_frame["timestamp"] >= event.start - pd.Timedelta(hours=12))
        & (prediction_frame["timestamp"] <= event.end + pd.Timedelta(hours=2))
    ]
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].plot(sensor_window["timestamp"], sensor_window["TP2"], label="TP2")
    axes[0].plot(sensor_window["timestamp"], sensor_window["Reservoirs"], label="Reservoirs")
    axes[0].set_title("Held-out July failure pressure sensors")
    axes[0].legend(loc="upper right")
    axes[1].plot(
        sensor_window["timestamp"],
        sensor_window["Oil_temperature"],
        label="Oil temperature",
        color="#805ad5",
    )
    axes[1].plot(
        sensor_window["timestamp"],
        sensor_window["Motor_current"],
        label="Motor current",
        color="#2f855a",
    )
    axes[1].set_title("Held-out July failure thermal/current sensors")
    axes[1].legend(loc="upper right")
    axes[2].plot(
        prediction_window["timestamp"],
        prediction_window["probability"],
        color="#2454a6",
        label="predicted probability",
    )
    axes[2].axhline(threshold, color="#a83232", linestyle="--", label="selected threshold")
    axes[2].axvspan(
        event.start - horizon,
        event.start,
        color="#f59e0b",
        alpha=0.16,
        linewidth=0,
        label="target horizon",
    )
    axes[2].set_title("Held-out July failure probability")
    axes[2].set_ylabel("probability")
    axes[2].legend(loc="upper right")
    for axis in axes:
        _shade_failure_events(axis, (event,))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))  # type: ignore[no-untyped-call]
    figure.savefig(output_path)
    plt.close(figure)


def _plot_distribution_shift(shift: pd.DataFrame, output_path: Path) -> None:
    if shift.empty:
        return
    top = shift.head(20).sort_values("abs_standardized_mean_difference")
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.barh(
        top["feature"],
        top["standardized_mean_difference"],
        color=np.where(top["standardized_mean_difference"] >= 0, "#33658a", "#b8322b"),
    )
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_title("July pre-failure distribution shift vs train pre-failure")
    axis.set_xlabel("standardized mean difference")
    figure.savefig(output_path)
    plt.close(figure)


def _write_run_report(run_dir: Path, *, summary: dict[str, Any], metrics: pd.DataFrame) -> None:
    best = summary["best_candidate"]
    decision = summary["final_decision"]
    lines = [
        "# MetroPT Robustness Phase 7B",
        "",
        f"- Failure events: {summary['failure_event_summary']['failure_event_count']}",
        f"- Horizons tested: {', '.join(str(value) for value in summary['horizons_hours'])} hours",
        f"- Best validation-selected candidate: {best['candidate_id']}",
        f"- Validation F1: {_fmt(best['validation_f1'])}",
        f"- Test F1: {_fmt(best['test_f1'])}",
        f"- Test PR-AUC: {_fmt(best['test_pr_auc'])}",
        f"- Test event detection rate: {_fmt(best['test_warning_coverage'])}",
        f"- Test median lead time hours: {_fmt(best['test_median_lead_time_hours'])}",
        f"- Test false alarms per day: {_fmt(best['test_false_alarms_per_day'])}",
        f"- Final decision: {decision['decision']} - {decision['conclusion']}",
        "",
        "Top test-ranked rows are stored in metrics.csv. Threshold sensitivity, event-level "
        "results, false alarms, July analysis, and distribution-shift artifacts are stored "
        "beside this report.",
    ]
    selected_columns = [
        "candidate_id",
        "validation_f1",
        "validation_warning_coverage",
        "test_f1",
        "test_pr_auc",
        "test_warning_coverage",
        "test_false_alarms_per_day",
    ]
    lines.extend(["", "## Metrics Preview", ""])
    lines.extend(_markdown_table(metrics.loc[:, selected_columns]))
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping")
    return dict(value)


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _robust_probability_increase_threshold(probabilities: pd.Series) -> float:
    if probabilities.empty:
        return 0.0
    median = float(probabilities.median())
    mad = float((probabilities - median).abs().median())
    p95 = float(probabilities.quantile(0.95))
    return max(p95, median + 3 * 1.4826 * mad)


def _hours_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float((end - start).total_seconds() / 3600)


def _candidate_id(horizon_hours: float, model: str) -> str:
    return f"h{_horizon_slug(horizon_hours)}__{model}"


def _horizon_slug(horizon_hours: float) -> str:
    return str(horizon_hours).replace(".", "p")


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in frame.to_dict(orient="records"):
        values = [_markdown_cell(record[column]) for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _markdown_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except (TypeError, ValueError):
        pass
    return value
