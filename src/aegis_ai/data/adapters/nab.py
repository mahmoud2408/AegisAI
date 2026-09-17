"""NAB dataset adapter."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aegis_ai.data.adapters.base import (
    BaseDatasetAdapter,
    DatasetProfile,
    RawDataset,
    Record,
    canonical_event_id,
)
from aegis_ai.data.models.common import LabelKind, LabelSourceType, MetricType, QualityFlag
from aegis_ai.data.preprocessing.cleaning import parse_float
from aegis_ai.data.preprocessing.timestamps import flag_temporal_order, parse_timestamp


class NABAdapter(BaseDatasetAdapter):
    """Convert NAB time series and labels to canonical metric and label records."""

    dataset_id = "nab"

    def profile(self, raw: RawDataset | None = None) -> DatasetProfile:
        raw = raw or self.load_raw()
        data_root = raw.path / "data"
        series_files = sorted(data_root.rglob("*.csv"))
        labels, windows = self._load_label_payloads(raw.path)
        rows = 0
        categories: Counter[str] = Counter()
        for file_path in series_files:
            categories[file_path.relative_to(data_root).as_posix().split("/", 1)[0]] += 1
            rows += max(count_lines(file_path) - 1, 0)

        return DatasetProfile(
            dataset_id=self.dataset_id,
            status=self.validate(raw).status,
            row_count=rows,
            column_count=2,
            label_count=sum(len(value) for value in labels.values())
            + sum(len(value) for value in windows.values()),
            profile={
                "series_files": len(series_files),
                "category_counts": dict(sorted(categories.items())),
                "point_labels": sum(len(value) for value in labels.values()),
                "scoring_windows": sum(len(value) for value in windows.values()),
                "timestamp_assumption": (
                    "timezone unavailable in source; preserving naive timestamp"
                ),
            },
        )

    def iter_metric_records(self, raw: RawDataset) -> Iterable[Record]:
        data_root = raw.path / "data"
        for file_path in sorted(data_root.rglob("*.csv")):
            relative_series = file_path.relative_to(data_root).as_posix()
            entity_id = f"nab:{relative_series.removesuffix('.csv')}"
            metric_name = Path(relative_series).stem
            source_file = self.relative_path(file_path)
            previous_timestamp = None
            seen_timestamps: set[str] = set()
            with file_path.open("r", encoding="utf-8-sig", newline="") as file:
                for row_index, row in enumerate(csv.DictReader(file)):
                    parsed = parse_timestamp(row.get("timestamp"))
                    value = parse_float(row.get("value"))
                    duplicate = parsed.original in seen_timestamps if parsed.original else False
                    if parsed.original:
                        seen_timestamps.add(parsed.original)
                    quality = parsed.quality_flag
                    if value is None:
                        quality = QualityFlag.INVALID
                    else:
                        quality = flag_temporal_order(
                            parsed.timestamp, previous_timestamp, duplicate
                        )
                    if parsed.timestamp is not None and not duplicate:
                        previous_timestamp = parsed.timestamp
                    if value is None:
                        continue
                    yield {
                        "event_id": canonical_event_id("nab", relative_series, row_index),
                        "timestamp": parsed.timestamp,
                        "timestamp_original": parsed.original,
                        "timezone_assumption": parsed.timezone_assumption,
                        "sequence_index": row_index,
                        "entity_id": entity_id,
                        "service_id": None,
                        "host_id": None,
                        "metric_name": metric_name,
                        "metric_value": value,
                        "metric_type": MetricType.CONTINUOUS.value,
                        "unit": None,
                        "split": None,
                        "source_dataset": self.dataset_id,
                        "source_file": source_file,
                        "source_row_id": str(row_index),
                        "quality_flag": quality.value,
                        "metadata": {"nab_series": relative_series},
                    }

    def get_labels(self, raw: RawDataset) -> Iterable[Record]:
        labels, windows = self._load_label_payloads(raw.path)
        labels_source_file = self.relative_path(raw.path / "labels" / "combined_labels.json")
        windows_source_file = self.relative_path(raw.path / "labels" / "combined_windows.json")
        for series, timestamps in sorted(labels.items()):
            entity_id = f"nab:{series.removesuffix('.csv')}"
            for index, value in enumerate(timestamps):
                parsed = parse_timestamp(value)
                yield {
                    "label_id": canonical_event_id("nab", "label", series, index),
                    "label_kind": LabelKind.ANOMALY.value,
                    "label_source_type": LabelSourceType.REAL.value,
                    "label_value": "anomaly",
                    "label_source": "combined_labels.json",
                    "event_id": None,
                    "timestamp": parsed.timestamp,
                    "timestamp_original": parsed.original,
                    "timezone_assumption": parsed.timezone_assumption,
                    "sequence_index": None,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "start_index": None,
                    "end_index": None,
                    "entity_id": entity_id,
                    "service_id": None,
                    "source_dataset": self.dataset_id,
                    "source_file": labels_source_file,
                    "source_row_id": str(index),
                    "quality_flag": parsed.quality_flag.value,
                    "metadata": {"nab_series": series, "label_representation": "point_timestamp"},
                }

        for series, intervals in sorted(windows.items()):
            entity_id = f"nab:{series.removesuffix('.csv')}"
            for index, interval in enumerate(intervals):
                start = parse_timestamp(interval[0] if interval else None)
                end = parse_timestamp(interval[1] if len(interval) > 1 else None)
                quality = (
                    QualityFlag.VALID
                    if start.timestamp is not None and end.timestamp is not None
                    else QualityFlag.INVALID
                )
                yield {
                    "label_id": canonical_event_id("nab", "window", series, index),
                    "label_kind": LabelKind.ANOMALY.value,
                    "label_source_type": LabelSourceType.REAL.value,
                    "label_value": "anomaly_window",
                    "label_source": "combined_windows.json",
                    "event_id": None,
                    "timestamp": None,
                    "timestamp_original": None,
                    "timezone_assumption": start.timezone_assumption or end.timezone_assumption,
                    "sequence_index": None,
                    "timestamp_start": start.timestamp,
                    "timestamp_end": end.timestamp,
                    "start_index": None,
                    "end_index": None,
                    "entity_id": entity_id,
                    "service_id": None,
                    "source_dataset": self.dataset_id,
                    "source_file": windows_source_file,
                    "source_row_id": str(index),
                    "quality_flag": quality.value,
                    "metadata": {"nab_series": series, "label_representation": "scoring_window"},
                }

    def _load_label_payloads(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        labels_path = path / "labels" / "combined_labels.json"
        windows_path = path / "labels" / "combined_windows.json"
        labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
        windows = (
            json.loads(windows_path.read_text(encoding="utf-8")) if windows_path.exists() else {}
        )
        return labels, windows


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file)
