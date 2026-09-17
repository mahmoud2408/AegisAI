"""Server Machine Dataset adapter."""

from __future__ import annotations

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
from aegis_ai.data.models.common import LabelKind, LabelSourceType, MetricType, QualityFlag
from aegis_ai.data.preprocessing.cleaning import parse_float, parse_int_label


class SMDAdapter(BaseDatasetAdapter):
    """Convert SMD matrix files to canonical long-format metric records."""

    dataset_id = "smd"
    metric_count = 38

    def validate(self, raw: RawDataset | None = None) -> ValidationReport:
        raw = raw or self.load_raw()
        base_report = super().validate(raw)
        issues = list(base_report.issues)
        for test_file in sorted((raw.path / "test").glob("*.txt")):
            label_file = raw.path / "test_label" / test_file.name
            if not label_file.exists():
                issues.append(
                    ValidationIssue("ERROR", "missing matching test_label file", str(label_file))
                )
                continue
            test_rows = count_lines(test_file)
            label_rows = count_lines(label_file)
            if test_rows != label_rows:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"test/label row mismatch: test={test_rows}, label={label_rows}",
                        str(test_file),
                    )
                )
        return ValidationReport(
            self.dataset_id,
            "READY" if not issues else "INCOMPLETE",
            tuple(issues),
        )

    def profile(self, raw: RawDataset | None = None) -> DatasetProfile:
        raw = raw or self.load_raw()
        train_files = sorted((raw.path / "train").glob("*.txt"))
        test_files = sorted((raw.path / "test").glob("*.txt"))
        label_files = sorted((raw.path / "test_label").glob("*.txt"))
        train_rows = sum(count_lines(path) for path in train_files)
        test_rows = sum(count_lines(path) for path in test_files)
        label_counts = [count_label_values(path) for path in label_files]
        positive_labels = sum(counts.get("1", 0) for counts in label_counts)
        label_rows = sum(sum(counts.values()) for counts in label_counts)

        return DatasetProfile(
            dataset_id=self.dataset_id,
            status=self.validate(raw).status,
            row_count=train_rows + test_rows,
            column_count=self.metric_count,
            label_count=label_rows,
            profile={
                "machines": len({file.stem for file in train_files + test_files}),
                "train_rows": train_rows,
                "test_rows": test_rows,
                "test_label_rows": label_rows,
                "positive_test_labels": positive_labels,
                "positive_label_rate": positive_labels / label_rows if label_rows else None,
                "temporal_axis": "row_index",
                "timestamp_column": False,
                "splits": ["train", "test"],
            },
        )

    def iter_metric_records(self, raw: RawDataset) -> Iterable[Record]:
        for split in ("train", "test"):
            for file_path in sorted((raw.path / split).glob("*.txt")):
                entity_id = file_path.stem
                source_file = self.relative_path(file_path)
                with file_path.open("r", encoding="utf-8") as file:
                    for row_index, line in enumerate(file):
                        values = line.strip().split(",")
                        if not values or values == [""]:
                            continue
                        for metric_index, raw_value in enumerate(values):
                            value = parse_float(raw_value)
                            if value is None:
                                continue
                            metric_name = f"metric_{metric_index:02d}"
                            yield {
                                "event_id": canonical_event_id(
                                    "smd", split, entity_id, row_index, metric_name
                                ),
                                "timestamp": None,
                                "timestamp_original": None,
                                "timezone_assumption": None,
                                "sequence_index": row_index,
                                "entity_id": entity_id,
                                "service_id": None,
                                "host_id": entity_id,
                                "metric_name": metric_name,
                                "metric_value": value,
                                "metric_type": MetricType.CONTINUOUS.value,
                                "unit": None,
                                "split": split,
                                "source_dataset": self.dataset_id,
                                "source_file": source_file,
                                "source_row_id": str(row_index),
                                "quality_flag": QualityFlag.VALID.value,
                                "metadata": (
                                    f'{{"metric_index": {metric_index}, '
                                    f'"temporal_axis": "row_index"}}'
                                ),
                            }

    def get_labels(self, raw: RawDataset) -> Iterable[Record]:
        for file_path in sorted((raw.path / "test_label").glob("*.txt")):
            entity_id = file_path.stem
            source_file = self.relative_path(file_path)
            with file_path.open("r", encoding="utf-8") as file:
                for row_index, line in enumerate(file):
                    target = parse_int_label(line.strip())
                    if target is None:
                        continue
                    label_kind = LabelKind.ANOMALY if target == 1 else LabelKind.NORMAL
                    yield {
                        "label_id": canonical_event_id("smd", "test_label", entity_id, row_index),
                        "label_kind": label_kind.value,
                        "label_source_type": LabelSourceType.REAL.value,
                        "label_value": "anomaly" if target == 1 else "normal",
                        "label_source": "test_label/*.txt",
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
                        "source_row_id": str(row_index),
                        "quality_flag": QualityFlag.VALID.value,
                        "metadata": {
                            "split": "test",
                            "raw_label": target,
                            "label_representation": "point_wise_binary",
                        },
                    }

        for file_path in sorted((raw.path / "interpretation_label").glob("*.txt")):
            entity_id = file_path.stem
            source_file = self.relative_path(file_path)
            for line_index, line in enumerate(read_non_empty_lines(file_path)):
                interval, affected_metrics = parse_interpretation_line(line)
                if interval is None:
                    continue
                start_index, end_index = interval
                yield {
                    "label_id": canonical_event_id("smd", "interpretation", entity_id, line_index),
                    "label_kind": LabelKind.ANOMALY.value,
                    "label_source_type": LabelSourceType.REAL.value,
                    "label_value": "affected_metric_interval",
                    "label_source": "interpretation_label/*.txt",
                    "event_id": None,
                    "timestamp": None,
                    "timestamp_original": None,
                    "timezone_assumption": None,
                    "sequence_index": None,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "start_index": start_index,
                    "end_index": end_index,
                    "entity_id": entity_id,
                    "service_id": None,
                    "source_dataset": self.dataset_id,
                    "source_file": source_file,
                    "source_row_id": str(line_index),
                    "quality_flag": QualityFlag.VALID.value,
                    "metadata": {
                        "affected_metric_indices": affected_metrics,
                        "label_representation": "interpretation_interval",
                    },
                }


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return sum(1 for line in file if line.strip())


def count_label_values(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in read_non_empty_lines(path):
        counts[line] = counts.get(line, 0) + 1
    return counts


def read_non_empty_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            clean = line.strip()
            if clean:
                yield clean


def parse_interpretation_line(line: str) -> tuple[tuple[int, int] | None, list[int]]:
    if ":" not in line or "-" not in line:
        return None, []
    interval_text, metrics_text = line.split(":", maxsplit=1)
    start_text, end_text = interval_text.split("-", maxsplit=1)
    try:
        interval = (int(start_text), int(end_text))
    except ValueError:
        return None, []
    metrics = []
    for value in metrics_text.split(","):
        value = value.strip()
        if value.isdigit():
            metrics.append(int(value))
    return interval, metrics
