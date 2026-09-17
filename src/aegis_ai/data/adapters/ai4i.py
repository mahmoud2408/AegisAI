"""AI4I 2020 predictive-maintenance adapter."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from aegis_ai.data.adapters.base import (
    BaseDatasetAdapter,
    DatasetProfile,
    RawDataset,
    Record,
    canonical_event_id,
)
from aegis_ai.data.models.common import LabelKind, LabelSourceType, MetricType, QualityFlag
from aegis_ai.data.preprocessing.cleaning import parse_float, parse_int_label

NUMERIC_FEATURES = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)

UNITS = {
    "Air temperature [K]": "K",
    "Process temperature [K]": "K",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "Nm",
    "Tool wear [min]": "min",
}

FAILURE_COLUMNS = ("Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF")


class AI4IAdapter(BaseDatasetAdapter):
    """Convert AI4I tabular observations to metric and failure records."""

    dataset_id = "ai4i"

    def profile(self, raw: RawDataset | None = None) -> DatasetProfile:
        raw = raw or self.load_raw()
        csv_path = self._csv_path(raw)
        rows = 0
        label_counts: dict[str, Counter[str]] = {column: Counter() for column in FAILURE_COLUMNS}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                rows += 1
                for column in FAILURE_COLUMNS:
                    label_counts[column][row.get(column, "")] += 1

        return DatasetProfile(
            dataset_id=self.dataset_id,
            status=self.validate(raw).status,
            row_count=rows,
            column_count=14,
            label_count=rows,
            profile={
                "numeric_features": list(NUMERIC_FEATURES),
                "failure_columns": list(FAILURE_COLUMNS),
                "identifier_columns": ["UDI", "Product ID"],
                "categorical_features": ["Type"],
                "label_counts": {
                    column: dict(sorted(counts.items())) for column, counts in label_counts.items()
                },
                "temporal_axis": "row_index_only",
            },
        )

    def iter_metric_records(self, raw: RawDataset) -> Iterable[Record]:
        csv_path = self._csv_path(raw)
        source_file = self.relative_path(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row_index, row in enumerate(csv.DictReader(file)):
                entity_id = row.get("Product ID") or f"ai4i_row_{row_index}"
                for feature in NUMERIC_FEATURES:
                    value = parse_float(row.get(feature))
                    if value is None:
                        continue
                    yield {
                        "event_id": canonical_event_id("ai4i", row_index, feature),
                        "timestamp": None,
                        "timestamp_original": None,
                        "timezone_assumption": None,
                        "sequence_index": row_index,
                        "entity_id": entity_id,
                        "service_id": None,
                        "host_id": None,
                        "metric_name": normalize_metric_name(feature),
                        "metric_value": value,
                        "metric_type": MetricType.CONTINUOUS.value,
                        "unit": UNITS.get(feature),
                        "split": None,
                        "source_dataset": self.dataset_id,
                        "source_file": source_file,
                        "source_row_id": row.get("UDI") or str(row_index),
                        "quality_flag": QualityFlag.VALID.value,
                        "metadata": {
                            "source_feature_name": feature,
                            "product_type": row.get("Type"),
                            "udi": row.get("UDI"),
                        },
                    }

    def iter_failure_records(self, raw: RawDataset) -> Iterable[Record]:
        csv_path = self._csv_path(raw)
        source_file = self.relative_path(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row_index, row in enumerate(csv.DictReader(file)):
                entity_id = row.get("Product ID") or f"ai4i_row_{row_index}"
                target = parse_int_label(row.get("Machine failure"))
                if target is None:
                    continue
                yield {
                    "event_id": canonical_event_id("ai4i", "failure", row_index),
                    "timestamp": None,
                    "timestamp_original": None,
                    "timezone_assumption": None,
                    "sequence_index": row_index,
                    "entity_id": entity_id,
                    "target": target,
                    "failure_type": "machine_failure",
                    "label_source": "Machine failure",
                    "label_source_type": LabelSourceType.REAL.value,
                    "source_dataset": self.dataset_id,
                    "source_file": source_file,
                    "source_row_id": row.get("UDI") or str(row_index),
                    "quality_flag": QualityFlag.VALID.value,
                    "metadata": {
                        "product_type": row.get("Type"),
                        "failure_modes": {
                            column: parse_int_label(row.get(column))
                            for column in FAILURE_COLUMNS
                            if column != "Machine failure"
                        },
                    },
                }

    def get_labels(self, raw: RawDataset) -> Iterable[Record]:
        csv_path = self._csv_path(raw)
        source_file = self.relative_path(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row_index, row in enumerate(csv.DictReader(file)):
                entity_id = row.get("Product ID") or f"ai4i_row_{row_index}"
                for column in FAILURE_COLUMNS:
                    target = parse_int_label(row.get(column))
                    if target is None:
                        continue
                    yield {
                        "label_id": canonical_event_id("ai4i", "label", column, row_index),
                        "label_kind": (
                            LabelKind.FAILURE.value if target == 1 else LabelKind.NORMAL.value
                        ),
                        "label_source_type": LabelSourceType.REAL.value,
                        "label_value": "failure" if target == 1 else "normal",
                        "label_source": column,
                        "event_id": None,
                        "timestamp": None,
                        "timestamp_original": None,
                        "timezone_assumption": None,
                        "sequence_index": row_index,
                        "timestamp_start": None,
                        "timestamp_end": None,
                        "start_index": None,
                        "end_index": None,
                        "entity_id": entity_id,
                        "service_id": None,
                        "source_dataset": self.dataset_id,
                        "source_file": source_file,
                        "source_row_id": row.get("UDI") or str(row_index),
                        "quality_flag": QualityFlag.VALID.value,
                        "metadata": {"label_column": column, "raw_label": target},
                    }

    def _csv_path(self, raw: RawDataset) -> Path:
        return raw.path / "ai4i2020.csv"


def normalize_metric_name(value: str) -> str:
    return value.lower().replace(" [", "_").replace("]", "").replace(" ", "_").replace("/", "_")
