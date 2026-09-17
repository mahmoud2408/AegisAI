"""Sequence window generation for temporal anomaly-detection models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceWindowConfig:
    """Configuration for split-local sequence windows."""

    sequence_length: int = 32
    stride: int = 1

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")


@dataclass(frozen=True)
class SequenceWindowSet:
    """Window tensor plus mapping back to source observations."""

    windows: np.ndarray
    start_indices: np.ndarray
    end_indices: np.ndarray
    timestamps: list[Any]
    labels: np.ndarray | None = None
    original_start_indices: np.ndarray | None = None
    original_end_indices: np.ndarray | None = None

    @property
    def is_empty(self) -> bool:
        return bool(self.windows.shape[0] == 0)


def make_sequence_windows(
    values: np.ndarray,
    *,
    timestamps: Sequence[Any] | None = None,
    labels: np.ndarray | Sequence[int] | None = None,
    config: SequenceWindowConfig | None = None,
) -> SequenceWindowSet:
    """Create temporally ordered windows from one contiguous split."""

    cfg = config or SequenceWindowConfig()
    array = _as_2d_float_array(values)
    row_count, feature_count = array.shape
    if timestamps is not None and len(timestamps) != row_count:
        raise ValueError("timestamps length must match number of observations")
    label_array = None if labels is None else np.asarray(labels, dtype=int)
    if label_array is not None and label_array.shape[0] != row_count:
        raise ValueError("labels length must match number of observations")

    if row_count < cfg.sequence_length:
        empty = np.empty((0, cfg.sequence_length, feature_count), dtype=np.float32)
        return SequenceWindowSet(
            windows=empty,
            start_indices=np.empty(0, dtype=int),
            end_indices=np.empty(0, dtype=int),
            timestamps=[],
            labels=np.empty(0, dtype=int) if label_array is not None else None,
            original_start_indices=np.empty(0, dtype=int),
            original_end_indices=np.empty(0, dtype=int),
        )

    starts = np.arange(0, row_count - cfg.sequence_length + 1, cfg.stride, dtype=int)
    ends = starts + cfg.sequence_length - 1
    windows = np.stack(
        [array[start : end + 1] for start, end in zip(starts, ends, strict=True)]
    ).astype(np.float32)
    window_timestamps = (
        [timestamps[int(end)] for end in ends]
        if timestamps is not None
        else [int(end) for end in ends]
    )
    window_labels = None
    if label_array is not None:
        window_labels = np.asarray(
            [
                int(label_array[start : end + 1].max())
                for start, end in zip(starts, ends, strict=True)
            ],
            dtype=int,
        )
    return SequenceWindowSet(
        windows=windows,
        start_indices=starts,
        end_indices=ends,
        timestamps=window_timestamps,
        labels=window_labels,
        original_start_indices=starts.copy(),
        original_end_indices=ends.copy(),
    )


def make_split_sequence_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    split_value: str,
    split_column: str = "split",
    timestamp_column: str = "timestamp",
    label_column: str | None = "y_true",
    config: SequenceWindowConfig | None = None,
) -> SequenceWindowSet:
    """Create windows for one split without crossing split boundaries."""

    if split_column not in frame.columns:
        raise ValueError(f"missing split column: {split_column}")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if timestamp_column not in frame.columns:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    if label_column is not None and label_column not in frame.columns:
        raise ValueError(f"missing label column: {label_column}")

    split_frame = frame[frame[split_column].eq(split_value)].copy()
    values = split_frame.loc[:, list(feature_columns)].to_numpy(dtype=float)
    labels = (
        split_frame[label_column].to_numpy(dtype=int)
        if label_column is not None and not split_frame.empty
        else None
    )
    windows = make_sequence_windows(
        values,
        timestamps=split_frame[timestamp_column].tolist(),
        labels=labels,
        config=config,
    )
    original_indices = split_frame.index.to_numpy(dtype=int)
    if windows.start_indices.size == 0:
        return windows
    return SequenceWindowSet(
        windows=windows.windows,
        start_indices=windows.start_indices,
        end_indices=windows.end_indices,
        timestamps=windows.timestamps,
        labels=windows.labels,
        original_start_indices=original_indices[windows.start_indices],
        original_end_indices=original_indices[windows.end_indices],
    )


def map_window_scores_to_points(
    scores: np.ndarray | Sequence[float],
    end_indices: np.ndarray | Sequence[int],
    *,
    point_count: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Map causal window scores to the window-end observation index."""

    score_array = np.asarray(scores, dtype=float)
    end_array = np.asarray(end_indices, dtype=int)
    if score_array.shape[0] != end_array.shape[0]:
        raise ValueError("scores and end_indices must have the same length")
    if point_count < 0:
        raise ValueError("point_count must be non-negative")
    if end_array.size and (end_array.min() < 0 or end_array.max() >= point_count):
        raise ValueError("end_indices must fall inside point_count")

    point_scores = np.full(point_count, fill_value, dtype=float)
    point_scores[end_array] = score_array
    return point_scores


def _as_2d_float_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("expected a 2D array shaped [observations, features]")
    if not np.isfinite(array).all():
        raise ValueError("window input contains NaN or infinite values")
    return array
