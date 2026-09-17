"""Evaluation utilities for anomaly-detection experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class BinaryAnomalyMetrics:
    """Binary point-level anomaly metrics."""

    precision: float
    recall: float
    f1: float
    pr_auc: float | None
    false_positive_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    support: int
    positives: int
    predicted_positives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionDelaySummary:
    """Delay from anomaly-window start to first in-window detection."""

    anomaly_windows: int
    detected_windows: int
    missed_windows: int
    mean_delay_steps: float | None
    median_delay_steps: float | None
    mean_delay_seconds: float | None
    median_delay_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray | None = None,
) -> BinaryAnomalyMetrics:
    """Compute deterministic binary anomaly metrics."""

    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D")
    if truth.size == 0:
        raise ValueError("cannot evaluate empty predictions")

    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fn = int(np.sum((truth == 1) & (pred == 0)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    false_positive_rate = _safe_divide(fp, fp + tn)

    pr_auc: float | None = None
    if scores is not None and len(set(truth.tolist())) == 2:
        score_array = np.asarray(scores, dtype=float)
        if score_array.shape != truth.shape:
            raise ValueError("scores must have the same shape as y_true")
        if np.isfinite(score_array).all():
            pr_auc = float(average_precision_score(truth, score_array))

    return BinaryAnomalyMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        false_positive_rate=false_positive_rate,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        support=int(truth.size),
        positives=int(np.sum(truth == 1)),
        predicted_positives=int(np.sum(pred == 1)),
    )


def compute_detection_delay(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    timestamps: Sequence[Any] | None = None,
) -> DetectionDelaySummary:
    """Compute first-detection delay for contiguous positive regions."""

    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    segments = positive_segments(truth)
    delay_steps: list[int] = []
    delay_seconds: list[float] = []

    timestamp_series = (
        pd.to_datetime(pd.Series(timestamps), errors="coerce") if timestamps is not None else None
    )
    for start, end in segments:
        local_hits = np.flatnonzero(pred[start:end] == 1)
        if local_hits.size == 0:
            continue
        first_detection = start + int(local_hits[0])
        delay_steps.append(first_detection - start)
        if timestamp_series is not None:
            start_time = timestamp_series.iloc[start]
            detection_time = timestamp_series.iloc[first_detection]
            if pd.notna(start_time) and pd.notna(detection_time):
                delay_seconds.append(float((detection_time - start_time).total_seconds()))

    detected = len(delay_steps)
    missed = len(segments) - detected
    return DetectionDelaySummary(
        anomaly_windows=len(segments),
        detected_windows=detected,
        missed_windows=missed,
        mean_delay_steps=_mean_or_none(delay_steps),
        median_delay_steps=_median_or_none(delay_steps),
        mean_delay_seconds=_mean_or_none(delay_seconds),
        median_delay_seconds=_median_or_none(delay_seconds),
    )


def compute_grouped_detection_delay(
    frame: pd.DataFrame,
    *,
    y_true_column: str,
    y_pred_column: str,
    entity_column: str = "entity_id",
    timestamp_column: str = "timestamp",
) -> DetectionDelaySummary:
    """Compute detection delay independently per entity, then aggregate windows."""

    required = {y_true_column, y_pred_column, entity_column, timestamp_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns for grouped delay: {sorted(missing)}")

    delay_steps: list[int] = []
    delay_seconds: list[float] = []
    anomaly_windows = 0
    for _entity_id, group in frame.groupby(entity_column, sort=True):
        ordered = group.sort_values(timestamp_column).reset_index(drop=True)
        truth = ordered[y_true_column].to_numpy(dtype=int)
        pred = ordered[y_pred_column].to_numpy(dtype=int)
        timestamps = pd.to_datetime(ordered[timestamp_column], errors="coerce")
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

    detected = len(delay_steps)
    return DetectionDelaySummary(
        anomaly_windows=anomaly_windows,
        detected_windows=detected,
        missed_windows=anomaly_windows - detected,
        mean_delay_steps=_mean_or_none(delay_steps),
        median_delay_steps=_median_or_none(delay_steps),
        mean_delay_seconds=_mean_or_none(delay_seconds),
        median_delay_seconds=_median_or_none(delay_seconds),
    )


def positive_segments(y_true: Sequence[int] | np.ndarray) -> list[tuple[int, int]]:
    """Return half-open contiguous positive regions."""

    truth = np.asarray(y_true, dtype=int)
    segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0
    for index, value in enumerate(truth):
        if value == 1 and not in_segment:
            start = index
            in_segment = True
        elif value == 0 and in_segment:
            segments.append((start, index))
            in_segment = False
    if in_segment:
        segments.append((start, int(truth.size)))
    return segments


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median_or_none(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None
