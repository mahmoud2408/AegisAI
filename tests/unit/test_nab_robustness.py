from __future__ import annotations

import numpy as np
import pandas as pd

from aegis_ai.ml.anomaly.nab_deep_experiment import MODEL_NAMES
from aegis_ai.ml.anomaly.nab_experiment import NabSeries
from aegis_ai.ml.anomaly.nab_robustness import (
    build_series_error_table,
    classify_series_difficulty,
    empirical_percentile_scores,
    threshold_from_scores,
)


def test_classify_series_difficulty_uses_positive_and_negative_criteria() -> None:
    assert (
        classify_series_difficulty(
            test_positive_points=12,
            best_f1=0.62,
            best_recall=0.45,
            max_false_positive_rate=0.12,
        )
        == "easy"
    )
    assert (
        classify_series_difficulty(
            test_positive_points=12,
            best_f1=0.0,
            best_recall=0.0,
            max_false_positive_rate=0.0,
        )
        == "complete_failure"
    )
    assert (
        classify_series_difficulty(
            test_positive_points=0,
            best_f1=0.0,
            best_recall=0.0,
            max_false_positive_rate=0.02,
        )
        == "moderate"
    )


def test_threshold_from_scores_supports_unsupervised_strategies() -> None:
    train = np.asarray([1.0, 1.1, 1.2, 1.3, 4.0])
    validation = np.asarray([1.0, 2.0, 3.0])

    assert threshold_from_scores(
        "train_p99",
        train_scores=train,
        validation_scores=validation,
        percentile=80,
        mad_multiplier=6,
    ) == np.percentile(train, 80)
    assert threshold_from_scores(
        "validation_p99",
        train_scores=train,
        validation_scores=validation,
        percentile=80,
        mad_multiplier=6,
    ) == np.percentile(validation, 80)
    assert threshold_from_scores(
        "train_mad",
        train_scores=train,
        validation_scores=validation,
        percentile=80,
        mad_multiplier=6,
    ) > np.median(train)


def test_empirical_percentile_scores_are_relative_to_reference() -> None:
    percentiles = empirical_percentile_scores(
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([5.0, 10.0, 25.0, 35.0]),
    )

    np.testing.assert_allclose(percentiles, np.asarray([0.0, 1 / 3, 2 / 3, 1.0]))


def test_build_series_error_table_adds_model_metrics_and_classification() -> None:
    timestamps = pd.date_range("2024-01-01", periods=8, freq="h")
    series = NabSeries(
        entity_id="nab:fixture",
        metric_name="fixture",
        frame=pd.DataFrame(
            {
                "timestamp": timestamps,
                "metric_value": range(8),
                "entity_id": "nab:fixture",
                "sequence_index": range(8),
            }
        ),
        windows=pd.DataFrame(
            {
                "entity_id": ["nab:fixture"],
                "timestamp_start": [timestamps[2]],
                "timestamp_end": [timestamps[4]],
            }
        ),
        point_labels=pd.DataFrame(),
    )
    per_series = pd.DataFrame([_metric_row(model) for model in MODEL_NAMES])
    predictions = pd.DataFrame(
        {
            "entity_id": ["nab:fixture"] * 4,
            "timestamp": timestamps[4:],
            "metric_name": ["fixture"] * 4,
            "metric_value": [4.0, 5.0, 6.0, 7.0],
            "y_true": [1, 0, 0, 0],
        }
    )

    table = build_series_error_table([series], per_series, predictions)

    assert table.loc[0, "observations"] == 8
    assert table.loc[0, "anomaly_windows"] == 1
    assert table.loc[0, "test_positive_points"] == 1
    assert table.loc[0, "best_model"] == "dense_autoencoder"
    assert table.loc[0, "difficulty_class"] == "easy"


def _metric_row(model: str) -> dict[str, float | int | str | None]:
    f1 = 0.7 if model == "dense_autoencoder" else 0.1
    recall = 0.5 if model == "dense_autoencoder" else 0.05
    return {
        "entity_id": "nab:fixture",
        "metric_name": "fixture",
        "model": model,
        "precision": 1.0,
        "recall": recall,
        "f1": f1,
        "pr_auc": 0.8,
        "false_positive_rate": 0.01,
        "positives": 1,
        "detection_detected_windows": 1,
        "detection_missed_windows": 0,
        "detection_median_delay_steps": 0,
    }
