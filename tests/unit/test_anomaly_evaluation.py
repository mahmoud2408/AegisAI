from __future__ import annotations

import pandas as pd

from aegis_ai.evaluation.anomaly import (
    compute_binary_metrics,
    compute_detection_delay,
    positive_segments,
)


def test_binary_metrics_are_computed_from_confusion_counts() -> None:
    metrics = compute_binary_metrics(
        y_true=[0, 1, 1, 0, 0, 1],
        y_pred=[0, 0, 1, 1, 0, 0],
        scores=[0.1, 0.2, 0.9, 0.8, 0.1, 0.3],
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 2
    assert metrics.precision == 0.5
    assert metrics.recall == 1 / 3
    assert metrics.pr_auc is not None


def test_detection_delay_uses_first_detection_inside_positive_segment() -> None:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="min")
    delay = compute_detection_delay(
        y_true=[0, 1, 1, 0, 0, 1],
        y_pred=[0, 0, 1, 1, 0, 0],
        timestamps=timestamps,
    )

    assert positive_segments([0, 1, 1, 0, 0, 1]) == [(1, 3), (5, 6)]
    assert delay.anomaly_windows == 2
    assert delay.detected_windows == 1
    assert delay.missed_windows == 1
    assert delay.mean_delay_steps == 1.0
    assert delay.mean_delay_seconds == 60.0
