"""MetroPT-3 dataset adapter."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from aegis_ai.data.adapters.base import (
    BaseDatasetAdapter,
    DatasetProfile,
    RawDataset,
    Record,
    canonical_event_id,
)
from aegis_ai.data.models.common import MetricType
from aegis_ai.data.preprocessing.cleaning import parse_float
from aegis_ai.data.preprocessing.timestamps import flag_temporal_order, parse_timestamp

CONTINUOUS_SENSORS = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)

BINARY_STATE_COLUMNS = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

UNITS = {
    "Oil_temperature": "celsius",
    "Motor_current": "ampere",
}


class MetroPTAdapter(BaseDatasetAdapter):
    """Convert MetroPT compressor telemetry to long-format metric records."""

    dataset_id = "metropt"
    entity_id = "metropt_air_compressor_01"

    def profile(self, raw: RawDataset | None = None) -> DatasetProfile:
        raw = raw or self.load_raw()
        csv_path = self._csv_path(raw)
        rows = max(count_lines(csv_path) - 1, 0) if csv_path.exists() else 0
        return DatasetProfile(
            dataset_id=self.dataset_id,
            status=self.validate(raw).status,
            row_count=rows,
            column_count=17,
            label_count=0,
            profile={
                "entity_id": self.entity_id,
                "continuous_sensors": list(CONTINUOUS_SENSORS),
                "binary_state_columns": list(BINARY_STATE_COLUMNS),
                "machine_readable_failure_labels": False,
                "timestamp_assumption": (
                    "timezone unavailable in source; preserving naive timestamp"
                ),
            },
        )

    def iter_metric_records(self, raw: RawDataset) -> Iterable[Record]:
        csv_path = self._csv_path(raw)
        source_file = self.relative_path(csv_path)
        previous_timestamp = None
        seen_timestamps: set[str] = set()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row_index, row in enumerate(csv.DictReader(file)):
                parsed = parse_timestamp(row.get("timestamp"))
                duplicate = parsed.original in seen_timestamps if parsed.original else False
                if parsed.original:
                    seen_timestamps.add(parsed.original)
                quality = flag_temporal_order(parsed.timestamp, previous_timestamp, duplicate)
                if parsed.timestamp is not None and not duplicate:
                    previous_timestamp = parsed.timestamp

                for metric_name in (*CONTINUOUS_SENSORS, *BINARY_STATE_COLUMNS):
                    value = parse_float(row.get(metric_name))
                    if value is None:
                        continue
                    metric_type = (
                        MetricType.BINARY_STATE
                        if metric_name in BINARY_STATE_COLUMNS
                        else MetricType.CONTINUOUS
                    )
                    yield {
                        "event_id": canonical_event_id("metropt", row_index, metric_name),
                        "timestamp": parsed.timestamp,
                        "timestamp_original": parsed.original,
                        "timezone_assumption": parsed.timezone_assumption,
                        "sequence_index": row_index,
                        "entity_id": self.entity_id,
                        "service_id": None,
                        "host_id": self.entity_id,
                        "metric_name": metric_name,
                        "metric_value": value,
                        "metric_type": metric_type.value,
                        "unit": UNITS.get(metric_name),
                        "split": None,
                        "source_dataset": self.dataset_id,
                        "source_file": source_file,
                        "source_row_id": row.get("") or str(row_index),
                        "quality_flag": quality.value,
                        "metadata": (
                            f'{{"sensor_family": "{metric_type.value}", '
                            f'"source_sensor_name": "{metric_name}"}}'
                        ),
                    }

    def _csv_path(self, raw: RawDataset) -> Path:
        return raw.path / "MetroPT3(AirCompressor).csv"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file)
