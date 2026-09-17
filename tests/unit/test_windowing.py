from __future__ import annotations

import numpy as np
import pandas as pd

from aegis_ai.features.windowing import (
    SequenceWindowConfig,
    make_sequence_windows,
    make_split_sequence_windows,
    map_window_scores_to_points,
)


def test_sequence_windows_preserve_order_and_end_timestamps() -> None:
    values = np.arange(10, dtype=float).reshape(-1, 1)
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
    labels = np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 1, 0])

    windows = make_sequence_windows(
        values,
        timestamps=timestamps.tolist(),
        labels=labels,
        config=SequenceWindowConfig(sequence_length=4, stride=2),
    )

    assert windows.windows.shape == (4, 4, 1)
    assert windows.windows[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert windows.end_indices.tolist() == [3, 5, 7, 9]
    assert windows.timestamps == [timestamps[3], timestamps[5], timestamps[7], timestamps[9]]
    assert windows.labels.tolist() == [1, 1, 0, 1]


def test_split_sequence_windows_do_not_cross_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "value": np.arange(8, dtype=float),
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="h"),
            "split": ["train", "train", "train", "train", "test", "test", "test", "test"],
            "y_true": [0, 0, 0, 0, 1, 1, 0, 0],
        }
    )

    train_windows = make_split_sequence_windows(
        frame,
        feature_columns=["value"],
        split_value="train",
        config=SequenceWindowConfig(sequence_length=3, stride=1),
    )
    test_windows = make_split_sequence_windows(
        frame,
        feature_columns=["value"],
        split_value="test",
        config=SequenceWindowConfig(sequence_length=3, stride=1),
    )

    assert train_windows.original_end_indices.tolist() == [2, 3]
    assert test_windows.original_start_indices.tolist() == [4, 5]
    assert test_windows.original_end_indices.tolist() == [6, 7]


def test_window_scores_map_to_window_end_points() -> None:
    point_scores = map_window_scores_to_points(
        [0.5, 0.8],
        [2, 4],
        point_count=6,
    )

    assert np.isnan(point_scores[0])
    assert point_scores[2] == 0.5
    assert point_scores[4] == 0.8
