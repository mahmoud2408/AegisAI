"""SMAP/MSL label-metadata adapters."""

from __future__ import annotations

import ast
import csv
import re
from collections.abc import Iterable
from pathlib import Path

from aegis_ai.data.adapters.base import (
    BaseDatasetAdapter,
    DatasetProfile,
    RawDataset,
    Record,
    ValidationIssue,
    ValidationReport,
    canonical_event_id,
)
from aegis_ai.data.models.common import LabelKind, LabelSourceType, QualityFlag


class SMAPMSLBaseAdapter(BaseDatasetAdapter):
    """Base adapter for the label-only local SMAP/MSL bundle."""

    raw_dataset_id = "smap-msl"
    spacecraft: str

    def validate(self, raw: RawDataset | None = None) -> ValidationReport:
        raw = raw or self.load_raw()
        labels_path = self._labels_path(raw)
        issues: list[ValidationIssue] = []
        if not labels_path.exists():
            issues.append(
                ValidationIssue("ERROR", "missing labeled_anomalies.csv", str(labels_path))
            )
        if not (raw.path / "data.zip").exists():
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "telemetry bundle is missing; adapter can emit labels only",
                    str(raw.path / "data.zip"),
                )
            )
        status = "LABELS_ONLY" if labels_path.exists() else "INCOMPLETE"
        return ValidationReport(self.dataset_id, status, tuple(issues))

    def profile(self, raw: RawDataset | None = None) -> DatasetProfile:
        raw = raw or self.load_raw()
        rows = 0
        sequence_count = 0
        classes: dict[str, int] = {}
        for row in self._iter_rows(raw):
            rows += 1
            sequences = parse_sequences(row.get("anomaly_sequences", ""))
            sequence_count += len(sequences)
            for anomaly_class in parse_classes(row.get("class", "")):
                classes[anomaly_class] = classes.get(anomaly_class, 0) + 1
        return DatasetProfile(
            dataset_id=self.dataset_id,
            status=self.validate(raw).status,
            row_count=0,
            column_count=None,
            label_count=sequence_count,
            profile={
                "spacecraft": self.spacecraft,
                "channels": rows,
                "anomaly_sequences": sequence_count,
                "anomaly_class_counts": classes,
                "telemetry_bundle_present": (raw.path / "data.zip").exists(),
                "canonical_output": "labels only until telemetry bundle is available",
            },
        )

    def get_labels(self, raw: RawDataset) -> Iterable[Record]:
        labels_path = self._labels_path(raw)
        source_file = self.relative_path(labels_path)
        for row_index, row in enumerate(self._iter_rows(raw)):
            channel_id = row.get("chan_id") or f"{self.spacecraft.lower()}_channel_{row_index}"
            classes = parse_classes(row.get("class", ""))
            sequences = parse_sequences(row.get("anomaly_sequences", ""))
            for sequence_index, interval in enumerate(sequences):
                anomaly_class = (
                    classes[sequence_index] if sequence_index < len(classes) else "unknown"
                )
                yield {
                    "label_id": canonical_event_id(
                        self.dataset_id, "label", channel_id, sequence_index
                    ),
                    "label_kind": LabelKind.ANOMALY.value,
                    "label_source_type": LabelSourceType.REAL.value,
                    "label_value": anomaly_class,
                    "label_source": "labeled_anomalies.csv",
                    "event_id": None,
                    "timestamp": None,
                    "timestamp_original": None,
                    "timezone_assumption": None,
                    "sequence_index": None,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "start_index": interval[0],
                    "end_index": interval[1],
                    "entity_id": channel_id,
                    "service_id": None,
                    "source_dataset": self.dataset_id,
                    "source_file": source_file,
                    "source_row_id": str(row_index),
                    "quality_flag": QualityFlag.VALID.value,
                    "metadata": {
                        "spacecraft": self.spacecraft,
                        "num_values": row.get("num_values"),
                        "telemetry_bundle_present": False,
                        "label_join_status": "unjoinable_until_telemetry_present",
                    },
                }

    def _iter_rows(self, raw: RawDataset) -> Iterable[dict[str, str]]:
        labels_path = self._labels_path(raw)
        if not labels_path.exists():
            return
        with labels_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("spacecraft") == self.spacecraft:
                    yield row

    def _labels_path(self, raw: RawDataset) -> Path:
        return raw.path / "labeled_anomalies.csv"


class SMAPAdapter(SMAPMSLBaseAdapter):
    """SMAP labels from the local SMAP/MSL bundle."""

    dataset_id = "smap"
    spacecraft = "SMAP"


class MSLAdapter(SMAPMSLBaseAdapter):
    """MSL labels from the local SMAP/MSL bundle."""

    dataset_id = "msl"
    spacecraft = "MSL"


def parse_sequences(value: str) -> list[tuple[int, int]]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    sequences = []
    for item in parsed:
        if isinstance(item, list | tuple) and len(item) == 2:
            try:
                sequences.append((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue
    return sequences


def parse_classes(value: str) -> list[str]:
    return re.findall(r"[A-Za-z_]+", value)
