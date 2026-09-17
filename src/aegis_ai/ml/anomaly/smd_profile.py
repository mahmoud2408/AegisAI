"""SMD profiling utilities for multivariate anomaly experiment planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis_ai.data.adapters.smd import count_lines, parse_interpretation_line, read_non_empty_lines
from aegis_ai.evaluation.anomaly import positive_segments


@dataclass(frozen=True)
class SMDProfileConfig:
    """Configuration for SMD dataset profiling."""

    metric_count: int = 38


def profile_smd_dataset(root: Path, config: SMDProfileConfig | None = None) -> dict[str, Any]:
    """Profile raw and processed SMD artifacts without loading the full long table."""

    cfg = config or SMDProfileConfig()
    raw_root = root / "data" / "raw" / "smd"
    processed_root = root / "data" / "processed" / "metrics" / "smd"
    if not raw_root.exists():
        raise FileNotFoundError(f"raw SMD directory not found: {raw_root}")

    machine_rows = []
    for train_file in sorted((raw_root / "train").glob("*.txt")):
        machine_id = train_file.stem
        test_file = raw_root / "test" / train_file.name
        label_file = raw_root / "test_label" / train_file.name
        interpretation_file = raw_root / "interpretation_label" / train_file.name
        if not test_file.exists() or not label_file.exists():
            raise FileNotFoundError(f"missing test or label file for SMD machine: {machine_id}")

        train_rows = count_lines(train_file)
        test_rows = count_lines(test_file)
        labels = np.asarray([int(value) for value in read_non_empty_lines(label_file)], dtype=int)
        if labels.shape[0] != test_rows:
            raise ValueError(f"SMD test/label length mismatch for {machine_id}")
        segments = positive_segments(labels)
        interpretation_records = _interpretation_records(interpretation_file)
        machine_rows.append(
            {
                "machine_id": machine_id,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "label_rows": int(labels.shape[0]),
                "positive_test_labels": int(labels.sum()),
                "positive_label_rate": float(labels.mean()) if labels.size else 0.0,
                "anomaly_segments": len(segments),
                "max_anomaly_segment_length": (
                    max((end - start for start, end in segments), default=0)
                ),
                "interpretation_intervals": len(interpretation_records),
                "affected_metric_mentions": sum(
                    len(record["affected_metric_indices"]) for record in interpretation_records
                ),
                "estimated_wide_float32_mib": _matrix_mib(
                    train_rows + test_rows,
                    cfg.metric_count,
                    bytes_per_value=4,
                ),
                "estimated_wide_float64_mib": _matrix_mib(
                    train_rows + test_rows,
                    cfg.metric_count,
                    bytes_per_value=8,
                ),
            }
        )

    machine_frame = pd.DataFrame(machine_rows).sort_values("machine_id")
    raw_size_bytes = sum(path.stat().st_size for path in raw_root.rglob("*") if path.is_file())
    processed_files = list(processed_root.rglob("*.parquet")) if processed_root.exists() else []
    processed_size_bytes = sum(path.stat().st_size for path in processed_files)
    processed_record_count = _processed_record_count(root)
    summary = {
        "dataset_id": "smd",
        "machine_count": int(machine_frame["machine_id"].nunique()),
        "metric_count": cfg.metric_count,
        "temporal_axis": "sequence_index",
        "timestamp_column": False,
        "train_rows": int(machine_frame["train_rows"].sum()),
        "test_rows": int(machine_frame["test_rows"].sum()),
        "label_rows": int(machine_frame["label_rows"].sum()),
        "positive_test_labels": int(machine_frame["positive_test_labels"].sum()),
        "positive_label_rate": float(
            machine_frame["positive_test_labels"].sum() / machine_frame["label_rows"].sum()
        ),
        "machines_with_anomalies": int((machine_frame["positive_test_labels"] > 0).sum()),
        "anomaly_segments": int(machine_frame["anomaly_segments"].sum()),
        "interpretation_intervals": int(machine_frame["interpretation_intervals"].sum()),
        "train_rows_min": int(machine_frame["train_rows"].min()),
        "train_rows_max": int(machine_frame["train_rows"].max()),
        "test_rows_min": int(machine_frame["test_rows"].min()),
        "test_rows_max": int(machine_frame["test_rows"].max()),
        "raw_file_count": len(list(raw_root.rglob("*.txt"))) + int((raw_root / "LICENSE").exists()),
        "raw_size_mib": raw_size_bytes / (1024 * 1024),
        "processed_metric_parquet_files": len(processed_files),
        "processed_metric_size_mib": processed_size_bytes / (1024 * 1024),
        "processed_metric_records": processed_record_count,
        "estimated_wide_float32_mib": float(machine_frame["estimated_wide_float32_mib"].sum()),
        "estimated_wide_float64_mib": float(machine_frame["estimated_wide_float64_mib"].sum()),
        "machine_profile": machine_rows,
    }
    return summary


def write_smd_profile_artifacts(profile: dict[str, Any], output_dir: Path) -> None:
    """Write SMD profile JSON and machine-level CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smd_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(profile["machine_profile"]).to_csv(
        output_dir / "smd_machine_profile.csv",
        index=False,
    )


def _interpretation_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in read_non_empty_lines(path):
        interval, affected_metrics = parse_interpretation_line(line)
        if interval is None:
            continue
        records.append(
            {
                "start_index": interval[0],
                "end_index": interval[1],
                "affected_metric_indices": affected_metrics,
            }
        )
    return records


def _matrix_mib(rows: int, columns: int, *, bytes_per_value: int) -> float:
    return float(rows * columns * bytes_per_value / (1024 * 1024))


def _processed_record_count(root: Path) -> int | None:
    lineage_path = root / "data" / "manifests" / "lineage" / "smd.json"
    if not lineage_path.exists():
        return None
    payload = json.loads(lineage_path.read_text(encoding="utf-8"))
    value = payload.get("output_records", {}).get("metrics")
    return int(value) if value is not None else None
