"""Reproducible NAB anomaly-detection baseline experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from aegis_ai.data.dataset_registry import project_root
from aegis_ai.data.preprocessing.splitting import SplitName, chronological_split
from aegis_ai.evaluation.anomaly import (
    compute_binary_metrics,
    compute_detection_delay,
    positive_segments,
)
from aegis_ai.features.timeseries import (
    FEATURE_COLUMNS,
    CausalRollingFeatureConfig,
    build_causal_rolling_features,
    feature_matrix,
    finite_feature_mask,
)
from aegis_ai.ml.anomaly.baselines import (
    RollingZScoreBaseline,
    RollingZScoreBaselineConfig,
    quantile_threshold,
)
from aegis_ai.ml.anomaly.isolation_forest import IsolationForestConfig, IsolationForestDetector

matplotlib.rcParams.update(
    {
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "font.size": 10,
    }
)


@dataclass(frozen=True)
class NabExperimentConfig:
    """Configuration for the first NAB anomaly baseline."""

    train_fraction: float = 0.5
    validation_fraction: float = 0.2
    threshold_quantile: float = 0.99
    feature_config: CausalRollingFeatureConfig = field(default_factory=CausalRollingFeatureConfig)
    zscore_config: RollingZScoreBaselineConfig = field(default_factory=RollingZScoreBaselineConfig)
    isolation_forest_config: IsolationForestConfig = field(default_factory=IsolationForestConfig)


@dataclass(frozen=True)
class NabSeries:
    """One processed NAB time series plus its labels."""

    entity_id: str
    metric_name: str
    frame: pd.DataFrame
    windows: pd.DataFrame
    point_labels: pd.DataFrame


@dataclass(frozen=True)
class NabExperimentResult:
    """Location and summary for an experiment run."""

    run_dir: Path
    metrics: dict[str, Any]
    dataset_summary: dict[str, Any]


def run_nab_baseline_experiment(
    *,
    root: Path | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    config: NabExperimentConfig | None = None,
) -> NabExperimentResult:
    """Run rolling z-score and Isolation Forest baselines on processed NAB data."""

    root = root or project_root()
    cfg = config or NabExperimentConfig()
    run_dir = _create_run_dir(
        output_root or root / "experiments" / "anomaly" / "nab" / "isolation_forest", run_id
    )
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    series_items = load_processed_nab(root)
    per_series_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for series in series_items:
        result = _evaluate_series(series, cfg, run_dir)
        per_series_rows.extend(result["metrics"])
        prediction_frames.append(result["predictions"])

    per_series = pd.DataFrame(per_series_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    aggregate_metrics = _aggregate_metrics(predictions, per_series)
    error_analysis = _build_error_analysis(predictions)
    dataset_summary = _dataset_summary(series_items, predictions, cfg)

    _write_json(run_dir / "config.json", _config_payload(cfg))
    _write_json(run_dir / "dataset_summary.json", dataset_summary)
    _write_json(run_dir / "metrics.json", aggregate_metrics)
    _write_json(run_dir / "error_analysis.json", error_analysis)
    per_series.to_csv(run_dir / "per_series_metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    _plot_all(run_dir, predictions, per_series, aggregate_metrics)
    _write_run_report(run_dir, aggregate_metrics, dataset_summary, error_analysis)

    return NabExperimentResult(
        run_dir=run_dir,
        metrics=aggregate_metrics,
        dataset_summary=dataset_summary,
    )


def load_processed_nab(root: Path) -> list[NabSeries]:
    """Load processed NAB Parquet without relying on sanitized partition ids."""

    metrics_root = root / "data" / "processed" / "metrics" / "nab"
    labels_path = root / "data" / "processed" / "labels" / "nab" / "part-00000.parquet"
    metric_files = sorted(metrics_root.rglob("*.parquet"))
    if not metric_files:
        raise FileNotFoundError(f"no processed NAB metric files found under {metrics_root}")
    if not labels_path.exists():
        raise FileNotFoundError(f"processed NAB labels are missing: {labels_path}")

    labels = pq.read_table(labels_path).to_pandas()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce")
    labels["timestamp_start"] = pd.to_datetime(labels["timestamp_start"], errors="coerce")
    labels["timestamp_end"] = pd.to_datetime(labels["timestamp_end"], errors="coerce")

    windows = labels[labels["label_value"].eq("anomaly_window")].copy()
    point_labels = labels[labels["label_value"].eq("anomaly")].copy()
    series_items: list[NabSeries] = []
    for path in metric_files:
        frame = pq.read_table(
            path,
            columns=[
                "event_id",
                "timestamp",
                "sequence_index",
                "entity_id",
                "metric_name",
                "metric_value",
                "quality_flag",
                "source_file",
                "source_row_id",
            ],
        ).to_pandas()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.sort_values(["timestamp", "sequence_index"], kind="mergesort").reset_index(
            drop=True
        )
        entity_id = str(frame["entity_id"].iloc[0])
        metric_name = str(frame["metric_name"].iloc[0])
        series_items.append(
            NabSeries(
                entity_id=entity_id,
                metric_name=metric_name,
                frame=frame,
                windows=windows[windows["entity_id"].eq(entity_id)].copy(),
                point_labels=point_labels[point_labels["entity_id"].eq(entity_id)].copy(),
            )
        )
    return series_items


def labels_for_series(series: NabSeries) -> np.ndarray:
    """Create point-level labels from official NAB scoring windows."""

    timestamps = series.frame["timestamp"]
    y_true = np.zeros(len(series.frame), dtype=int)
    for _, row in series.windows.iterrows():
        start = row["timestamp_start"]
        end = row["timestamp_end"]
        if pd.isna(start) or pd.isna(end):
            continue
        y_true[((timestamps >= start) & (timestamps <= end)).to_numpy()] = 1

    if y_true.sum() == 0 and not series.point_labels.empty:
        for _, row in series.point_labels.iterrows():
            point = row["timestamp"]
            if pd.isna(point):
                continue
            y_true[timestamps.eq(point).to_numpy()] = 1
    return y_true


def _evaluate_series(
    series: NabSeries,
    cfg: NabExperimentConfig,
    run_dir: Path,
) -> dict[str, Any]:
    feature_frame = build_causal_rolling_features(series.frame, config=cfg.feature_config)
    feature_frame["y_true"] = labels_for_series(series)
    feature_frame["split"] = ""
    bounds = chronological_split(
        len(feature_frame),
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
    )
    for bound in bounds:
        feature_frame.loc[bound.start : bound.end - 1, "split"] = bound.name.value

    usable = finite_feature_mask(feature_frame)
    usable_frame = feature_frame.loc[usable].copy()
    train = usable_frame[usable_frame["split"].eq(SplitName.TRAIN.value)]
    validation = usable_frame[usable_frame["split"].eq(SplitName.VALIDATION.value)]
    test = usable_frame[usable_frame["split"].eq(SplitName.TEST.value)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"series does not have enough usable rows: {series.entity_id}")

    baseline = RollingZScoreBaseline(cfg.zscore_config).fit(train)
    baseline_validation_scores = baseline.score_samples(validation)
    baseline_threshold = quantile_threshold(
        baseline_validation_scores,
        cfg.threshold_quantile,
    )
    baseline_scores = baseline.score_samples(test)
    baseline_predictions = (baseline_scores >= baseline_threshold).astype(int)

    detector = IsolationForestDetector(cfg.isolation_forest_config).fit(
        feature_matrix(train, FEATURE_COLUMNS)
    )
    iforest_validation_scores = detector.score_samples(feature_matrix(validation, FEATURE_COLUMNS))
    iforest_threshold = quantile_threshold(iforest_validation_scores, cfg.threshold_quantile)
    iforest_scores = detector.score_samples(feature_matrix(test, FEATURE_COLUMNS))
    iforest_predictions = (iforest_scores >= iforest_threshold).astype(int)
    detector.save(run_dir / "models" / f"{_safe_filename(series.entity_id)}.joblib")

    prediction_frame = test[
        ["entity_id", "timestamp", "metric_name", "metric_value", "y_true"]
    ].copy()
    prediction_frame["baseline_score"] = baseline_scores
    prediction_frame["baseline_prediction"] = baseline_predictions
    prediction_frame["isolation_forest_score"] = iforest_scores
    prediction_frame["isolation_forest_prediction"] = iforest_predictions
    prediction_frame["baseline_threshold"] = baseline_threshold
    prediction_frame["isolation_forest_threshold"] = iforest_threshold

    metrics = [
        _series_metric_row(
            model="rolling_zscore",
            series=series,
            train_rows=len(train),
            validation_rows=len(validation),
            test_rows=len(test),
            threshold=baseline_threshold,
            y_true=prediction_frame["y_true"].to_numpy(dtype=int),
            y_pred=baseline_predictions,
            scores=baseline_scores,
            timestamps=prediction_frame["timestamp"].tolist(),
        ),
        _series_metric_row(
            model="isolation_forest",
            series=series,
            train_rows=len(train),
            validation_rows=len(validation),
            test_rows=len(test),
            threshold=iforest_threshold,
            y_true=prediction_frame["y_true"].to_numpy(dtype=int),
            y_pred=iforest_predictions,
            scores=iforest_scores,
            timestamps=prediction_frame["timestamp"].tolist(),
        ),
    ]

    return {"metrics": metrics, "predictions": prediction_frame}


def _series_metric_row(
    *,
    model: Literal["rolling_zscore", "isolation_forest"],
    series: NabSeries,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    timestamps: list[Any],
) -> dict[str, Any]:
    metrics = compute_binary_metrics(y_true, y_pred, scores).to_dict()
    delay = compute_detection_delay(y_true, y_pred, timestamps).to_dict()
    return {
        "entity_id": series.entity_id,
        "metric_name": series.metric_name,
        "model": model,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "threshold": threshold,
        **metrics,
        **{f"detection_{key}": value for key, value in delay.items()},
    }


def _aggregate_metrics(
    predictions: pd.DataFrame,
    per_series: pd.DataFrame,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "global": {},
        "per_series_distribution": {},
        "best_series_by_f1": {},
        "worst_series_by_f1": {},
    }
    model_columns = {
        "rolling_zscore": ("baseline_prediction", "baseline_score"),
        "isolation_forest": ("isolation_forest_prediction", "isolation_forest_score"),
    }
    y_true = predictions["y_true"].to_numpy(dtype=int)
    for model, (prediction_column, score_column) in model_columns.items():
        y_pred = predictions[prediction_column].to_numpy(dtype=int)
        scores = predictions[score_column].to_numpy(dtype=float)
        metrics = compute_binary_metrics(y_true, y_pred, scores).to_dict()
        delay = _grouped_detection_delay(predictions, prediction_column)
        aggregate["global"][model] = {
            **metrics,
            "detection_delay": delay,
        }
        model_series = per_series[per_series["model"].eq(model)].copy()
        aggregate["per_series_distribution"][model] = _distribution_summary(model_series)
        aggregate["best_series_by_f1"][model] = _ranked_series(model_series, best=True)
        aggregate["worst_series_by_f1"][model] = _ranked_series(model_series, best=False)
    return aggregate


def _grouped_detection_delay(
    predictions: pd.DataFrame,
    prediction_column: str,
) -> dict[str, Any]:
    """Aggregate detection-delay windows without crossing time-series boundaries."""

    delay_steps: list[int] = []
    delay_seconds: list[float] = []
    anomaly_windows = 0

    for _entity_id, group in predictions.groupby("entity_id", sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        truth = ordered["y_true"].to_numpy(dtype=int)
        pred = ordered[prediction_column].to_numpy(dtype=int)
        timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce")

        for start, end in positive_segments(truth):
            anomaly_windows += 1
            local_hits = np.flatnonzero(pred[start:end] == 1)
            if local_hits.size == 0:
                continue

            first_detection = start + int(local_hits[0])
            delay_steps.append(first_detection - start)
            start_time = timestamps.iloc[start]
            detection_time = timestamps.iloc[first_detection]
            if pd.notna(start_time) and pd.notna(detection_time):
                delay_seconds.append(float((detection_time - start_time).total_seconds()))

    detected_windows = len(delay_steps)
    return {
        "anomaly_windows": anomaly_windows,
        "detected_windows": detected_windows,
        "missed_windows": anomaly_windows - detected_windows,
        "mean_delay_steps": float(np.mean(delay_steps)) if delay_steps else None,
        "median_delay_steps": float(np.median(delay_steps)) if delay_steps else None,
        "mean_delay_seconds": float(np.mean(delay_seconds)) if delay_seconds else None,
        "median_delay_seconds": float(np.median(delay_seconds)) if delay_seconds else None,
    }


def _distribution_summary(model_series: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    fields = ["precision", "recall", "f1", "pr_auc", "false_positive_rate"]
    summary: dict[str, dict[str, float | None]] = {}
    for field_name in fields:
        values = pd.to_numeric(model_series[field_name], errors="coerce").dropna()
        summary[field_name] = {
            "mean": _maybe_float(values.mean()),
            "median": _maybe_float(values.median()),
            "std": _maybe_float(values.std(ddof=0)),
        }
    return summary


def _ranked_series(model_series: pd.DataFrame, *, best: bool) -> list[dict[str, Any]]:
    ranked = model_series[model_series["positives"] > 0].sort_values(
        ["f1", "recall", "precision"],
        ascending=not best,
    )
    fields = ["entity_id", "precision", "recall", "f1", "pr_auc", "positives", "threshold"]
    return cast(list[dict[str, Any]], ranked.loc[:, fields].head(5).to_dict(orient="records"))


def _build_error_analysis(predictions: pd.DataFrame) -> dict[str, Any]:
    return {
        "rolling_zscore": _model_error_analysis(
            predictions,
            prediction_column="baseline_prediction",
            score_column="baseline_score",
        ),
        "isolation_forest": _model_error_analysis(
            predictions,
            prediction_column="isolation_forest_prediction",
            score_column="isolation_forest_score",
        ),
    }


def _model_error_analysis(
    predictions: pd.DataFrame,
    *,
    prediction_column: str,
    score_column: str,
) -> dict[str, Any]:
    false_positives = predictions[
        predictions[prediction_column].eq(1) & predictions["y_true"].eq(0)
    ].sort_values(score_column, ascending=False)
    false_positive_examples = [
        _row_example(row, score_column) for _, row in false_positives.head(10).iterrows()
    ]

    missed_windows: list[dict[str, Any]] = []
    delayed_windows: list[dict[str, Any]] = []
    for entity_id, group in predictions.groupby("entity_id", sort=True):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        truth = ordered["y_true"].to_numpy(dtype=int)
        pred = ordered[prediction_column].to_numpy(dtype=int)
        for start, end in positive_segments(truth):
            window = ordered.iloc[start:end]
            hit_offsets = np.flatnonzero(pred[start:end] == 1)
            summary = {
                "entity_id": entity_id,
                "window_start": str(window["timestamp"].iloc[0]),
                "window_end": str(window["timestamp"].iloc[-1]),
                "points": int(len(window)),
                "max_score": float(window[score_column].max()),
                "min_value": float(window["metric_value"].min()),
                "max_value": float(window["metric_value"].max()),
            }
            if hit_offsets.size == 0:
                missed_windows.append(summary)
            else:
                delay_steps = int(hit_offsets[0])
                if delay_steps > 0:
                    delayed = dict(summary)
                    delayed["delay_steps"] = delay_steps
                    delayed["first_detection"] = str(window["timestamp"].iloc[delay_steps])
                    delayed_windows.append(delayed)

    delayed_windows.sort(key=lambda item: int(item["delay_steps"]), reverse=True)
    missed_windows.sort(key=lambda item: float(item["max_score"]), reverse=True)
    return {
        "false_positive_examples": false_positive_examples,
        "missed_window_examples": missed_windows[:10],
        "delayed_window_examples": delayed_windows[:10],
    }


def _dataset_summary(
    series_items: list[NabSeries],
    predictions: pd.DataFrame,
    cfg: NabExperimentConfig,
) -> dict[str, Any]:
    total_rows = sum(len(item.frame) for item in series_items)
    point_labels = sum(len(item.point_labels) for item in series_items)
    windows = sum(len(item.windows) for item in series_items)
    return {
        "dataset_id": "nab",
        "series_count": len(series_items),
        "observation_count": total_rows,
        "point_labels": point_labels,
        "scoring_windows": windows,
        "evaluated_test_rows": int(len(predictions)),
        "test_positive_points": int(predictions["y_true"].sum()),
        "label_semantics": (
            "Point labels come from combined_labels.json; interval labels come from "
            "combined_windows.json and are used as the generic temporal evaluation target."
        ),
        "split": {
            "strategy": "per-series chronological split",
            "train_fraction": cfg.train_fraction,
            "validation_fraction": cfg.validation_fraction,
            "test_fraction": 1 - cfg.train_fraction - cfg.validation_fraction,
            "threshold_selection": (
                "unsupervised validation-score quantile; labels are evaluation-only"
            ),
        },
    }


def _plot_all(
    run_dir: Path,
    predictions: pd.DataFrame,
    per_series: pd.DataFrame,
    aggregate_metrics: dict[str, Any],
) -> None:
    example_entity = _choose_example_entity(per_series)
    example = predictions[predictions["entity_id"].eq(example_entity)].sort_values("timestamp")
    _plot_ground_truth(run_dir, example)
    _plot_detections(run_dir, example)
    _plot_scores(run_dir, example)
    _plot_metric_comparison(run_dir, aggregate_metrics)
    _plot_per_series_distribution(run_dir, per_series)


def _plot_ground_truth(run_dir: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame["timestamp"], frame["metric_value"], color="#1f4e79", linewidth=1.1)
    positives = frame[frame["y_true"].eq(1)]
    if not positives.empty:
        ax.scatter(
            positives["timestamp"],
            positives["metric_value"],
            s=14,
            color="#d95f02",
            label="Ground truth window",
            zorder=3,
        )
    ax.set_title(f"NAB Ground Truth: {frame['entity_id'].iloc[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Metric value")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "01_timeseries_ground_truth.png")
    plt.close(fig)


def _plot_detections(run_dir: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame["timestamp"], frame["metric_value"], color="#334155", linewidth=1.0)
    positives = frame[frame["y_true"].eq(1)]
    if not positives.empty:
        ax.scatter(
            positives["timestamp"],
            positives["metric_value"],
            s=13,
            color="#f59e0b",
            label="Ground truth",
            zorder=3,
        )
    iforest_hits = frame[frame["isolation_forest_prediction"].eq(1)]
    baseline_hits = frame[frame["baseline_prediction"].eq(1)]
    ax.scatter(
        baseline_hits["timestamp"],
        baseline_hits["metric_value"],
        s=16,
        marker="x",
        color="#7c3aed",
        label="Rolling z-score",
    )
    ax.scatter(
        iforest_hits["timestamp"],
        iforest_hits["metric_value"],
        s=20,
        marker="o",
        facecolors="none",
        edgecolors="#dc2626",
        label="Isolation Forest",
    )
    ax.set_title(f"NAB Detections: {frame['entity_id'].iloc[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Metric value")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "02_timeseries_detected_anomalies.png")
    plt.close(fig)


def _plot_scores(run_dir: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        frame["timestamp"],
        frame["baseline_score"],
        label="Rolling z-score",
        color="#7c3aed",
        linewidth=1.0,
    )
    ax.plot(
        frame["timestamp"],
        frame["isolation_forest_score"],
        label="Isolation Forest",
        color="#dc2626",
        linewidth=1.0,
    )
    ax.axhline(float(frame["baseline_threshold"].iloc[0]), color="#7c3aed", linestyle="--")
    ax.axhline(
        float(frame["isolation_forest_threshold"].iloc[0]),
        color="#dc2626",
        linestyle="--",
    )
    ax.set_title(f"Anomaly Scores: {frame['entity_id'].iloc[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly score")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "03_anomaly_scores.png")
    plt.close(fig)


def _plot_metric_comparison(run_dir: Path, aggregate_metrics: dict[str, Any]) -> None:
    models = ["rolling_zscore", "isolation_forest"]
    metric_names = ["precision", "recall", "f1"]
    values = [
        [aggregate_metrics["global"][model][metric_name] for metric_name in metric_names]
        for model in models
    ]
    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, values[0], width, label="Rolling z-score", color="#7c3aed")
    ax.bar(x + width / 2, values[1], width, label="Isolation Forest", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1)
    ax.set_title("Global Point-Level Metrics")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "04_precision_recall_f1_comparison.png")
    plt.close(fig)


def _plot_per_series_distribution(run_dir: Path, per_series: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [
        per_series[per_series["model"].eq("rolling_zscore")]["f1"].to_numpy(dtype=float),
        per_series[per_series["model"].eq("isolation_forest")]["f1"].to_numpy(dtype=float),
    ]
    ax.boxplot(data, tick_labels=["Rolling z-score", "Isolation Forest"], showmeans=True)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Per-Series F1 Distribution")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "05_per_series_f1_distribution.png")
    plt.close(fig)


def _write_run_report(
    run_dir: Path,
    metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
    error_analysis: dict[str, Any],
) -> None:
    lines = [
        "# NAB Anomaly Baseline Run",
        "",
        f"Run directory: `{run_dir.as_posix()}`",
        "",
        "## Dataset",
        "",
        f"- Series: {dataset_summary['series_count']}",
        f"- Observations: {dataset_summary['observation_count']:,}",
        f"- Test rows evaluated: {dataset_summary['evaluated_test_rows']:,}",
        f"- Test positive points from windows: {dataset_summary['test_positive_points']:,}",
        "",
        "## Global Metrics",
        "",
        "| Model | Precision | Recall | F1 | PR-AUC | FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("rolling_zscore", "isolation_forest"):
        item = metrics["global"][model]
        lines.append(
            f"| {model} | {_fmt(item['precision'])} | {_fmt(item['recall'])} | "
            f"{_fmt(item['f1'])} | {_fmt(item['pr_auc'])} | "
            f"{_fmt(item['false_positive_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Error Analysis Counts",
            "",
            (
                "| Model | False positive examples | Missed window examples | "
                "Delayed window examples |"
            ),
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model in ("rolling_zscore", "isolation_forest"):
        item = error_analysis[model]
        lines.append(
            f"| {model} | {len(item['false_positive_examples'])} | "
            f"{len(item['missed_window_examples'])} | {len(item['delayed_window_examples'])} |"
        )
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _choose_example_entity(per_series: pd.DataFrame) -> str:
    candidates = per_series[
        per_series["model"].eq("isolation_forest") & (per_series["positives"] > 0)
    ].sort_values(["positives", "f1"], ascending=[False, False])
    if candidates.empty:
        return str(per_series["entity_id"].iloc[0])
    return str(candidates["entity_id"].iloc[0])


def _row_example(row: pd.Series, score_column: str) -> dict[str, Any]:
    return {
        "entity_id": row["entity_id"],
        "timestamp": str(row["timestamp"]),
        "metric_value": float(row["metric_value"]),
        "score": float(row[score_column]),
    }


def _config_payload(cfg: NabExperimentConfig) -> dict[str, Any]:
    return {
        "train_fraction": cfg.train_fraction,
        "validation_fraction": cfg.validation_fraction,
        "threshold_quantile": cfg.threshold_quantile,
        "feature_config": asdict(cfg.feature_config),
        "zscore_config": asdict(cfg.zscore_config),
        "isolation_forest_config": asdict(cfg.isolation_forest_config),
    }


def _create_run_dir(output_root: Path, run_id: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = output_root / resolved_run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{resolved_run_id}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _safe_filename(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _maybe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
