from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def load_audit_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "datasets" / "audit_datasets.py"
    spec = importlib.util.spec_from_file_location("audit_datasets", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_csv_file_classifies_binary_states_and_identifiers(tmp_path: Path) -> None:
    audit = load_audit_module()
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["LineId", "timestamp", "value", "state", "message"])
        writer.writerow([1, "2020-01-01 00:00:00", "1.5", "1.0", "ok"])
        writer.writerow([2, "2020-01-01 00:00:10", "2.5", "0.0", "ok"])
        writer.writerow([3, "2020-01-01 00:00:10", "2.5", "0.0", "ok"])

    profile = audit.profile_csv_file(csv_path, timestamp_column="timestamp")

    assert profile["rows"] == 3
    assert profile["missing_values"] == 0
    assert profile["duplicate_timestamps"] == 1
    assert "LineId" in profile["identifier_columns"]
    assert profile["column_types"]["state"] == "bool"
    assert "state" in profile["categorical_columns"]
    assert profile["column_types"]["value"] == "float"
    assert profile["temporal"]["median_delta_seconds"] == 5


def test_row_count_mismatches_reports_only_mismatched_files() -> None:
    audit = load_audit_module()

    mismatches = audit.row_count_mismatches(
        [{"file": "a.txt", "rows": 10}, {"file": "b.txt", "rows": 12}],
        [{"file": "a.txt", "rows": 10}, {"file": "b.txt", "rows": 11}],
    )

    assert mismatches == [{"file": "b.txt", "left_rows": 12, "right_rows": 11}]
