from __future__ import annotations

import pandas as pd

from aegis_ai.ml.anomaly.nab_experiment import NabSeries, labels_for_series


def test_nab_window_labels_map_to_point_level_targets() -> None:
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

    assert labels_for_series(series).tolist() == [0, 0, 1, 1, 1, 0, 0, 0]


def test_nab_point_labels_are_fallback_when_no_windows_exist() -> None:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h")
    series = NabSeries(
        entity_id="nab:fixture",
        metric_name="fixture",
        frame=pd.DataFrame(
            {
                "timestamp": timestamps,
                "metric_value": range(5),
                "entity_id": "nab:fixture",
                "sequence_index": range(5),
            }
        ),
        windows=pd.DataFrame(),
        point_labels=pd.DataFrame({"timestamp": [timestamps[3]]}),
    )

    assert labels_for_series(series).tolist() == [0, 0, 0, 1, 0]
