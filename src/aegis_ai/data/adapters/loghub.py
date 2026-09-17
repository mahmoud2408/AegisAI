"""LogHub dataset adapter."""

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
from aegis_ai.data.models.common import LabelKind, LabelSourceType, QualityFlag
from aegis_ai.data.preprocessing.cleaning import blank_to_none, extract_request_id
from aegis_ai.data.preprocessing.timestamps import (
    ParsedTimestamp,
    combine_date_time,
    flag_temporal_order,
    parse_timestamp,
)

LOGHUB_DATASETS = {
    "hdfs": "loghub-hdfs",
    "bgl": "loghub-bgl",
    "openstack": "loghub-openstack",
    "hadoop": "loghub-hadoop",
    "spark": "loghub-spark",
    "zookeeper": "loghub-zookeeper",
}


class LogHubAdapter(BaseDatasetAdapter):
    """Reusable adapter for local LogHub 2k samples."""

    def __init__(self, family: str, root: Path | None = None) -> None:
        if family not in LOGHUB_DATASETS:
            raise KeyError(f"unsupported LogHub family: {family}")
        self.family = family
        self.dataset_id = LOGHUB_DATASETS[family]
        super().__init__(root)

    def profile(self, raw: RawDataset | None = None) -> DatasetProfile:
        raw = raw or self.load_raw()
        structured_path = self._structured_path(raw)
        template_path = self._template_path(raw)
        rows = 0
        levels: Counter[str] = Counter()
        event_ids: Counter[str] = Counter()
        labels: Counter[str] = Counter()
        if structured_path.exists():
            with structured_path.open("r", encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    rows += 1
                    levels[row.get("Level", "")] += 1
                    event_ids[row.get("EventId", "")] += 1
                    if "Label" in row:
                        labels[row.get("Label", "")] += 1
        template_rows = max(count_lines(template_path) - 1, 0) if template_path.exists() else 0
        return DatasetProfile(
            dataset_id=self.dataset_id,
            status=self.validate(raw).status,
            row_count=rows,
            column_count=None,
            label_count=sum(labels.values()),
            profile={
                "family": self.family,
                "structured_file": self.relative_path(structured_path)
                if structured_path.exists()
                else None,
                "template_file": self.relative_path(template_path)
                if template_path.exists()
                else None,
                "template_rows": template_rows,
                "level_counts": dict(sorted(levels.items())),
                "event_id_top_10": dict(event_ids.most_common(10)),
                "label_counts": dict(sorted(labels.items())),
                "timestamp_assumption": (
                    "timezone unavailable in source unless numeric epoch is used"
                ),
            },
        )

    def iter_log_records(self, raw: RawDataset) -> Iterable[Record]:
        structured_path = self._structured_path(raw)
        source_file = self.relative_path(structured_path)
        previous_timestamp = None
        seen_timestamps: set[str] = set()
        with structured_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row_index, row in enumerate(csv.DictReader(file)):
                parsed = self._parse_row_timestamp(row)
                duplicate = parsed.original in seen_timestamps if parsed.original else False
                if parsed.original:
                    seen_timestamps.add(parsed.original)
                quality = flag_temporal_order(parsed.timestamp, previous_timestamp, duplicate)
                if parsed.timestamp is not None and not duplicate:
                    previous_timestamp = parsed.timestamp

                component = blank_to_none(row.get("Component"))
                node = blank_to_none(row.get("Node") or row.get("NodeRepeat"))
                process_id = blank_to_none(row.get("Pid") or row.get("Process") or row.get("Id"))
                request_id = extract_request_id(row.get("ADDR"), row.get("Content"))
                entity_id = node or component or f"{self.dataset_id}:logs"
                yield {
                    "event_id": canonical_event_id(self.dataset_id, row.get("LineId") or row_index),
                    "timestamp": parsed.timestamp,
                    "timestamp_original": parsed.original,
                    "timezone_assumption": parsed.timezone_assumption,
                    "sequence_index": row_index,
                    "entity_id": entity_id,
                    "service_id": component,
                    "host_id": node,
                    "process_id": process_id,
                    "request_id": request_id,
                    "trace_id": None,
                    "log_level": blank_to_none(row.get("Level")),
                    "event_type": blank_to_none(row.get("EventId")),
                    "event_template": blank_to_none(row.get("EventTemplate")),
                    "message": row.get("Content") or "",
                    "source_dataset": self.dataset_id,
                    "source_file": source_file,
                    "source_row_id": row.get("LineId") or str(row_index),
                    "quality_flag": quality.value,
                    "metadata": {
                        "family": self.family,
                        "line_id": row.get("LineId"),
                        "logrecord": row.get("Logrecord"),
                        "addr": row.get("ADDR"),
                        "source_event_id": row.get("EventId"),
                    },
                }

    def get_labels(self, raw: RawDataset) -> Iterable[Record]:
        structured_path = self._structured_path(raw)
        source_file = self.relative_path(structured_path)
        with structured_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "Label" not in reader.fieldnames:
                return
            for row_index, row in enumerate(reader):
                raw_label = row.get("Label", "")
                is_normal = raw_label == "-"
                parsed = self._parse_row_timestamp(row)
                yield {
                    "label_id": canonical_event_id(self.dataset_id, "label", row_index),
                    "label_kind": (
                        LabelKind.NORMAL.value if is_normal else LabelKind.ANOMALY.value
                    ),
                    "label_source_type": LabelSourceType.REAL.value,
                    "label_value": "normal" if is_normal else raw_label,
                    "label_source": "Label",
                    "event_id": canonical_event_id(self.dataset_id, row.get("LineId") or row_index),
                    "timestamp": parsed.timestamp,
                    "timestamp_original": parsed.original,
                    "timezone_assumption": parsed.timezone_assumption,
                    "sequence_index": row_index,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "start_index": None,
                    "end_index": None,
                    "entity_id": row.get("Node") or row.get("Component"),
                    "service_id": row.get("Component"),
                    "source_dataset": self.dataset_id,
                    "source_file": source_file,
                    "source_row_id": row.get("LineId") or str(row_index),
                    "quality_flag": QualityFlag.VALID.value,
                    "metadata": {"raw_label": raw_label, "family": self.family},
                }

    def _parse_row_timestamp(self, row: dict[str, str]) -> ParsedTimestamp:
        if self.family == "bgl" and row.get("Timestamp"):
            return parse_timestamp(row.get("Timestamp"), timezone="UTC")
        return parse_timestamp(combine_date_time(row.get("Date"), row.get("Time")))

    def _structured_path(self, raw: RawDataset) -> Path:
        matches = sorted(raw.path.glob("*_2k.log_structured.csv"))
        return matches[0] if matches else raw.path / f"{self.family}_2k.log_structured.csv"

    def _template_path(self, raw: RawDataset) -> Path:
        matches = sorted(raw.path.glob("*_2k.log_templates.csv"))
        return matches[0] if matches else raw.path / f"{self.family}_2k.log_templates.csv"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file)
