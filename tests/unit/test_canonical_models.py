from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from aegis_ai.data.models import (
    FailureObservation,
    LabelEvent,
    LabelKind,
    LabelSourceType,
    MetricObservation,
    MetricType,
)


def test_metric_observation_requires_time_or_sequence() -> None:
    record = MetricObservation(
        event_id="evt-1",
        sequence_index=1,
        entity_id="machine-1",
        metric_name="cpu",
        metric_value=0.5,
        metric_type=MetricType.CONTINUOUS,
        source_dataset="fixture",
        source_file="fixture.csv",
    )

    assert record.sequence_index == 1

    with pytest.raises(ValidationError):
        MetricObservation(
            event_id="evt-2",
            entity_id="machine-1",
            metric_name="cpu",
            metric_value=0.5,
            source_dataset="fixture",
            source_file="fixture.csv",
        )


def test_failure_observation_preserves_binary_target_contract() -> None:
    with pytest.raises(ValidationError):
        FailureObservation(
            event_id="failure-1",
            sequence_index=0,
            entity_id="asset-1",
            target=2,
            failure_type="machine_failure",
            label_source="fixture",
            source_dataset="fixture",
            source_file="fixture.csv",
        )


def test_label_event_allows_intervals_without_point_timestamp() -> None:
    label = LabelEvent(
        label_id="label-1",
        label_kind=LabelKind.ANOMALY,
        label_source_type=LabelSourceType.REAL,
        label_value="anomaly_window",
        label_source="fixture",
        timestamp_start=datetime(2020, 1, 1),
        timestamp_end=datetime(2020, 1, 2),
        source_dataset="fixture",
        source_file="labels.json",
    )

    assert label.label_value == "anomaly_window"
