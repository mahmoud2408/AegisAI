from __future__ import annotations

import pandas as pd

from aegis_ai.ml.anomaly.nab_experiment import NabSeries, labels_for_series
from aegis_ai.ml.anomaly.nab_labels import audit_nab_label_alignment


def test_label_windows_are_closed_intervals() -> None:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h")
    series = NabSeries(
        entity_id="nab:fixture",
        metric_name="value",
        frame=pd.DataFrame(
            {
                "timestamp": timestamps,
                "metric_value": range(5),
                "entity_id": "nab:fixture",
                "sequence_index": range(5),
            }
        ),
        windows=pd.DataFrame(
            {
                "entity_id": ["nab:fixture"],
                "timestamp_start": [timestamps[1]],
                "timestamp_end": [timestamps[3]],
            }
        ),
        point_labels=pd.DataFrame(),
    )

    assert labels_for_series(series).tolist() == [0, 1, 1, 1, 0]
    audit = audit_nab_label_alignment([series])
    assert audit["closed_interval_extra_boundary_points"] == 1
    assert audit["windows_without_observations"] == 0


def test_label_audit_flags_windows_outside_observation_range() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h")
    series = NabSeries(
        entity_id="nab:fixture",
        metric_name="value",
        frame=pd.DataFrame(
            {
                "timestamp": timestamps,
                "metric_value": range(4),
                "entity_id": "nab:fixture",
                "sequence_index": range(4),
            }
        ),
        windows=pd.DataFrame(
            {
                "entity_id": ["nab:fixture"],
                "timestamp_start": [timestamps[-1] + pd.Timedelta(hours=1)],
                "timestamp_end": [timestamps[-1] + pd.Timedelta(hours=2)],
            }
        ),
        point_labels=pd.DataFrame(),
    )

    audit = audit_nab_label_alignment([series])

    assert audit["windows_outside_observation_range"] == 1
    assert audit["windows_without_observations"] == 1


def test_label_audit_handles_multiple_windows() -> None:
    timestamps = pd.date_range("2024-01-01", periods=8, freq="h")
    series = NabSeries(
        entity_id="nab:fixture",
        metric_name="value",
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
                "entity_id": ["nab:fixture", "nab:fixture"],
                "timestamp_start": [timestamps[1], timestamps[5]],
                "timestamp_end": [timestamps[2], timestamps[6]],
            }
        ),
        point_labels=pd.DataFrame(),
    )

    audit = audit_nab_label_alignment([series])

    assert audit["series_with_multiple_windows_count"] == 1
    assert audit["anomaly_window_count"] == 2
