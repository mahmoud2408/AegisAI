from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_ai.data.dataset_registry import load_dataset_catalog  # noqa: E402
from aegis_ai.data.manifest import directory_size, format_bytes  # noqa: E402
from aegis_ai.data.validation import validate_all  # noqa: E402

UNIQUE_VALUE_LIMIT = 100_000
DUPLICATE_HASH_LIMIT = 2_500_000
SAMPLE_LINES = 5

DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S,%f",
    "%Y-%m-%d",
    "%y/%m/%d %H:%M:%S",
    "%y/%m/%d",
    "%Y.%m.%d",
    "%m%d%y %H%M%S",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile local AegisAI datasets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to the detected project root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "dataset_audit.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print a compact human-readable summary after writing JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output

    audit = build_audit(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    if args.pretty:
        print(format_summary(audit))
    return 0


def build_audit(root: Path) -> dict[str, Any]:
    catalog = load_dataset_catalog(root / "config" / "datasets.yaml")
    validation = {result.dataset_id: result for result in validate_all(root)}
    raw_root = root / "data" / "raw"

    datasets: list[dict[str, Any]] = []
    for dataset_id, definition in sorted(catalog.items()):
        result = validation[dataset_id]
        datasets.append(
            {
                "id": dataset_id,
                "name": definition.name,
                "category": definition.category,
                "path": str(definition.destination.relative_to(root)),
                "status": result.status,
                "health_problems": list(result.problems),
                "size_bytes": result.size_bytes,
                "size_human": format_bytes(result.size_bytes),
                "source": definition.official_source,
                "license": definition.license,
                "files": list_files(definition.destination, root),
                "profile": profile_dataset(dataset_id, definition.destination, root),
            }
        )

    data_lake = {
        stage: stage_summary(root / "data" / stage, root)
        for stage in ("raw", "interim", "processed", "synthetic", "manifests")
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "data_lake": data_lake,
        "raw_total_size_bytes": directory_size(raw_root),
        "raw_total_size_human": format_bytes(directory_size(raw_root)),
        "datasets": datasets,
        "methodology": {
            "duplicate_rows": (
                "Counted with streaming BLAKE2b row-content hashes. Files above "
                f"{DUPLICATE_HASH_LIMIT:,} rows are marked capped rather than keeping "
                "unbounded hash state."
            ),
            "types": "Inferred from observed non-missing values without coercing source files.",
            "temporal": "Computed from parseable timestamp/date columns or documented row index.",
            "large_files": "CSV and text matrices are scanned line by line.",
        },
    }


def stage_summary(path: Path, root: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
    return {
        "path": str(path.relative_to(root)),
        "exists": path.exists(),
        "file_count": len(files),
        "size_bytes": directory_size(path),
        "size_human": format_bytes(directory_size(path)),
    }


def list_files(path: Path, root: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files.append(
                {
                    "path": str(item.relative_to(root)),
                    "size_bytes": item.stat().st_size,
                    "size_human": format_bytes(item.stat().st_size),
                    "format": infer_format(item),
                }
            )
    return files


def infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".zip":
        return "zip"
    if suffix in {".md", ".txt", ".log", ""}:
        if path.name.endswith(".log"):
            return "raw_log"
        if path.suffix.lower() == ".txt" and "smd" in path.parts:
            return "numeric_matrix_txt"
        return "text"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".py":
        return "python"
    return suffix.removeprefix(".") or "unknown"


def profile_dataset(dataset_id: str, path: Path, root: Path) -> dict[str, Any]:
    if dataset_id == "nab":
        return profile_nab(path)
    if dataset_id == "smd":
        return profile_smd(path)
    if dataset_id == "smap-msl":
        return profile_smap_msl(path)
    if dataset_id.startswith("loghub-"):
        return profile_loghub(path, dataset_id.removeprefix("loghub-"))
    if dataset_id == "metropt":
        return profile_metropt(path)
    if dataset_id == "ai4i":
        return profile_ai4i(path)
    if dataset_id == "opentelemetry-demo":
        return profile_opentelemetry(path)
    return profile_manual_or_empty(path, root)


def profile_nab(path: Path) -> dict[str, Any]:
    data_root = path / "data"
    csv_files = sorted(data_root.rglob("*.csv")) if data_root.exists() else []
    file_profiles = []
    category_counts: Counter[str] = Counter()
    label_points: dict[str, list[str]] = {}
    label_windows: dict[str, list[list[str]]] = {}

    labels_path = path / "labels" / "combined_labels.json"
    windows_path = path / "labels" / "combined_windows.json"
    if labels_path.exists():
        label_points = json.loads(labels_path.read_text(encoding="utf-8"))
    if windows_path.exists():
        label_windows = json.loads(windows_path.read_text(encoding="utf-8"))

    for csv_file in csv_files:
        rel = csv_file.relative_to(data_root).as_posix()
        category_counts[rel.split("/", 1)[0]] += 1
        profile = profile_csv_file(csv_file, timestamp_column="timestamp")
        profile["relative_series"] = rel
        profile["label_points"] = len(label_points.get(rel, []))
        profile["label_windows"] = len(label_windows.get(rel, []))
        file_profiles.append(profile)

    total_rows = sum(profile.get("rows", 0) for profile in file_profiles)
    total_point_labels = sum(len(value) for value in label_points.values())
    total_windows = sum(len(value) for value in label_windows.values())
    deltas = Counter()
    min_time: str | None = None
    max_time: str | None = None
    regular_files = 0
    constant_series = 0
    missing_values = 0
    duplicate_rows = 0
    duplicate_timestamps = 0

    for profile in file_profiles:
        temporal = profile.get("temporal", {})
        for delta, count in temporal.get("top_deltas_seconds", []):
            deltas[str(delta)] += count
        if temporal.get("is_regular") is True:
            regular_files += 1
        min_time = min_known_datetime(min_time, temporal.get("min"))
        max_time = max_known_datetime(max_time, temporal.get("max"))
        constant_series += len(profile.get("constant_columns", []))
        missing_values += profile.get("missing_values", 0)
        duplicate_rows += profile.get("duplicate_rows", 0)
        duplicate_timestamps += profile.get("duplicate_timestamps", 0)

    return {
        "series_files": len(csv_files),
        "category_counts": dict(sorted(category_counts.items())),
        "rows": total_rows,
        "columns": ["timestamp", "value"],
        "labels": {
            "series_with_point_labels": sum(1 for value in label_points.values() if value),
            "point_labels": total_point_labels,
            "series_with_windows": sum(1 for value in label_windows.values() if value),
            "windows": total_windows,
            "representation": "point timestamps plus interval scoring windows",
        },
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "constant_value_series": constant_series,
        "temporal": {
            "min": min_time,
            "max": max_time,
            "regular_series_files": regular_files,
            "top_deltas_seconds": deltas.most_common(10),
        },
        "series_profiles_sample": file_profiles[:8],
    }


def profile_smd(path: Path) -> dict[str, Any]:
    train_files = sorted((path / "train").glob("*.txt"))
    test_files = sorted((path / "test").glob("*.txt"))
    label_files = sorted((path / "test_label").glob("*.txt"))
    interpretation_files = sorted((path / "interpretation_label").glob("*.txt"))

    train_profiles = [profile_numeric_matrix(file) for file in train_files]
    test_profiles = [profile_numeric_matrix(file) for file in test_files]
    label_profiles = [profile_label_vector(file) for file in label_files]
    interpretation_profiles = [profile_interpretation_label(file) for file in interpretation_files]
    test_label_mismatches = row_count_mismatches(test_profiles, label_profiles)

    anomaly_points = sum(profile["positive_labels"] for profile in label_profiles)
    label_rows = sum(profile["rows"] for profile in label_profiles)
    train_rows = sum(profile["rows"] for profile in train_profiles)
    test_rows = sum(profile["rows"] for profile in test_profiles)
    columns = sorted({profile["columns"] for profile in train_profiles + test_profiles})

    return {
        "machines": len({file.stem for file in train_files + test_files}),
        "train_files": len(train_files),
        "test_files": len(test_files),
        "test_label_files": len(label_files),
        "interpretation_label_files": len(interpretation_files),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "rows": train_rows + test_rows,
        "columns_per_row": columns,
        "labels": {
            "test_label_rows": label_rows,
            "positive_test_labels": anomaly_points,
            "positive_label_rate": safe_ratio(anomaly_points, label_rows),
            "representation": "point-wise binary labels for test split",
            "interpretation_intervals": sum(
                profile["intervals"] for profile in interpretation_profiles
            ),
            "test_label_mismatches": test_label_mismatches,
        },
        "missing_values": sum(
            profile["missing_values"] for profile in train_profiles + test_profiles + label_profiles
        ),
        "constant_columns_by_file_sample": [
            {
                "file": profile["file"],
                "constant_columns": profile["constant_columns"],
            }
            for profile in (train_profiles + test_profiles)[:8]
            if profile["constant_columns"]
        ],
        "duplicate_rows": sum(
            profile["duplicate_rows"] for profile in train_profiles + test_profiles
        ),
        "temporal": {
            "timestamp_column": False,
            "sample_axis": "row_index",
            "sample_rate": "not encoded in files",
        },
        "train_profile_sample": train_profiles[:3],
        "test_profile_sample": test_profiles[:3],
        "label_profile_sample": label_profiles[:3],
    }


def profile_smap_msl(path: Path) -> dict[str, Any]:
    labels = path / "labeled_anomalies.csv"
    profile = profile_csv_file(labels) if labels.exists() else {"rows": 0}
    spacecraft_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    sequence_count = 0
    channels = 0
    if labels.exists():
        with labels.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                channels += 1
                spacecraft_counts[row.get("spacecraft", "")] += 1
                sequences = re.findall(r"\[[0-9]+,\s*[0-9]+\]", row.get("anomaly_sequences", ""))
                sequence_count += len(sequences)
                for value in re.findall(r"[A-Za-z_]+", row.get("class", "")):
                    class_counts[value] += 1

    profile.update(
        {
            "telemetry_bundle_present": (path / "data.zip").exists(),
            "channels": channels,
            "spacecraft_counts": dict(spacecraft_counts),
            "anomaly_sequences": sequence_count,
            "anomaly_class_counts": dict(class_counts),
            "labels": {
                "representation": "interval sequences by telemetry channel",
                "usable_without_telemetry": False,
            },
        }
    )
    return profile


def profile_ai4i(path: Path) -> dict[str, Any]:
    csv_path = path / "ai4i2020.csv"
    profile = profile_csv_file(csv_path)
    target_columns = ("Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF")
    profile["labels"] = {
        "representation": "row-level binary failure target plus binary failure-mode columns",
        "target_column": "Machine failure",
        "failure_mode_columns": list(target_columns[1:]),
        "value_counts": {
            column: column_counts(csv_path, column)
            for column in target_columns
            if csv_path.exists()
        },
    }
    profile["identifiers"] = ["UDI", "Product ID"]
    profile["temporal"]["sample_axis"] = "row_index_only"
    return profile


def profile_metropt(path: Path) -> dict[str, Any]:
    csv_path = path / "MetroPT3(AirCompressor).csv"
    profile = profile_csv_file(csv_path, timestamp_column="timestamp")
    profile["labels"] = {
        "representation": "none found as machine-readable CSV columns",
        "failure_label_ready": False,
    }
    profile["identifiers"] = ["unnamed_index"]
    return profile


def profile_loghub(path: Path, name: str) -> dict[str, Any]:
    structured = next(path.glob("*_2k.log_structured.csv"), None)
    templates = next(path.glob("*_2k.log_templates.csv"), None)
    raw_log = next(path.glob("*_2k.log"), None)
    structured_profile = (
        profile_csv_file(structured, timestamp_column=detect_loghub_timestamp_column(structured))
        if structured
        else {}
    )
    template_profile = profile_csv_file(templates) if templates else {}

    labels: dict[str, Any] = {"representation": "none found in local sample"}
    if structured and "Label" in structured_profile.get("columns", []):
        counts = column_counts(structured, "Label")
        labels = {
            "column": "Label",
            "values": counts,
            "representation": "row-level label in structured sample",
        }

    return {
        "sample_name": name,
        "raw_log_lines": count_lines(raw_log) if raw_log else 0,
        "structured": structured_profile,
        "templates": template_profile,
        "labels": labels,
        "event_id_counts_top": (
            column_counts(structured, "EventId", limit=15) if structured else {}
        ),
        "level_counts": column_counts(structured, "Level", limit=15) if structured else {},
        "ids_observed": identify_log_columns(structured_profile.get("columns", [])),
    }


def profile_opentelemetry(path: Path) -> dict[str, Any]:
    files = list_files(path, ROOT)
    config_files = [
        item["path"]
        for item in files
        if item["format"] in {"yaml", "python"} or item["path"].endswith(".json")
    ]
    return {
        "static_telemetry_rows": 0,
        "reference_files": len(files),
        "config_or_generator_files": config_files,
        "labels": {"representation": "none; reference/generator resources only"},
        "main_use": "future synthetic observability generator and trace/log schema examples",
    }


def profile_manual_or_empty(path: Path, root: Path) -> dict[str, Any]:
    real_files = (
        [
            item
            for item in path.rglob("*")
            if item.is_file() and item.name not in {".gitkeep", "README.md"}
        ]
        if path.exists()
        else []
    )
    return {
        "manual_or_empty": True,
        "data_files_present": len(real_files),
        "files": [str(file.relative_to(root)) for file in real_files],
        "labels": {"representation": "not inspectable until official files are installed"},
    }


def profile_csv_file(path: Path | None, timestamp_column: str | None = None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            return {"exists": True, "file": str(path), "rows": 0, "columns": []}

        columns = [column if column else "unnamed_index" for column in header]
        stats = new_column_stats(columns)
        row_count = 0
        duplicate_rows = 0
        seen_hashes: set[bytes] = set()
        duplicate_hash_capped = False
        samples: list[list[str]] = []
        duplicate_timestamps = 0
        timestamp_seen: set[str] = set()
        temporal_stats = new_temporal_stats()

        for row in reader:
            row_count += 1
            if len(samples) < SAMPLE_LINES:
                samples.append(row)
            normalized = normalize_width(row, len(columns))
            update_column_stats(stats, normalized)

            if not duplicate_hash_capped:
                digest = hashlib.blake2b(
                    "\x1f".join(normalized).encode("utf-8", errors="replace"),
                    digest_size=16,
                ).digest()
                if digest in seen_hashes:
                    duplicate_rows += 1
                elif len(seen_hashes) >= DUPLICATE_HASH_LIMIT:
                    duplicate_hash_capped = True
                    seen_hashes.clear()
                else:
                    seen_hashes.add(digest)

            time_value = extract_time_value(columns, normalized, timestamp_column)
            if time_value:
                if time_value in timestamp_seen:
                    duplicate_timestamps += 1
                else:
                    timestamp_seen.add(time_value)
                update_temporal_stats(temporal_stats, time_value)

    temporal = describe_temporal_stats(temporal_stats)
    numeric_stats = finalize_numeric_stats(stats)
    return {
        "exists": True,
        "file": path.name,
        "rows": row_count,
        "columns_count": len(columns),
        "columns": columns,
        "size_bytes": path.stat().st_size,
        "size_human": format_bytes(path.stat().st_size),
        "missing_values": sum(item["missing"] for item in stats),
        "missing_by_column": {item["name"]: item["missing"] for item in stats if item["missing"]},
        "duplicate_rows": duplicate_rows,
        "duplicate_count_capped": duplicate_hash_capped,
        "duplicate_timestamps": duplicate_timestamps,
        "constant_columns": [item["name"] for item in stats if item["constant"]],
        "near_constant_columns": [
            item["name"]
            for item in stats
            if item["unique_exact"] and item["rows"] > 0 and item["unique_count"] <= 2
        ],
        "categorical_columns": infer_categorical_columns(stats, row_count),
        "identifier_columns": infer_identifier_columns(stats),
        "numerical_columns": [
            item["name"] for item in stats if infer_type(item) in {"int", "float"}
        ],
        "text_columns": [item["name"] for item in stats if infer_type(item) == "text"],
        "column_types": {item["name"]: infer_type(item) for item in stats},
        "unique_counts": {
            item["name"]: item["unique_count"] if item["unique_exact"] else f">{UNIQUE_VALUE_LIMIT}"
            for item in stats
        },
        "numeric_ranges": {
            item["name"]: {"min": item["min"], "max": item["max"], "mean": item["mean"]}
            for item in numeric_stats
        },
        "temporal": temporal,
        "sample_rows": samples,
    }


def profile_numeric_matrix(path: Path) -> dict[str, Any]:
    rows = 0
    columns: int | None = None
    missing_values = 0
    first_values: list[str | None] = []
    changed: list[bool] = []
    duplicate_rows = 0
    seen_hashes: set[bytes] = set()
    duplicate_hash_capped = False

    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()
            if not line:
                continue
            rows += 1
            values = line.split(",")
            if columns is None:
                columns = len(values)
                first_values = [None] * columns
                changed = [False] * columns
            normalized = normalize_width(values, columns)
            missing_values += sum(1 for value in normalized if value == "")

            for index, value in enumerate(normalized):
                if first_values[index] is None and value != "":
                    first_values[index] = value
                elif first_values[index] != value:
                    changed[index] = True

            if not duplicate_hash_capped:
                digest = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
                if digest in seen_hashes:
                    duplicate_rows += 1
                elif len(seen_hashes) >= DUPLICATE_HASH_LIMIT:
                    duplicate_hash_capped = True
                    seen_hashes.clear()
                else:
                    seen_hashes.add(digest)

    return {
        "file": path.name,
        "rows": rows,
        "columns": columns or 0,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_count_capped": duplicate_hash_capped,
        "constant_columns": [index for index, is_changed in enumerate(changed) if not is_changed],
    }


def profile_label_vector(path: Path) -> dict[str, Any]:
    rows = 0
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            label = raw.strip()
            if not label:
                continue
            rows += 1
            counts[label] += 1
    return {
        "file": path.name,
        "rows": rows,
        "columns": 1,
        "missing_values": 0,
        "positive_labels": counts.get("1", 0),
        "value_counts": dict(counts),
    }


def row_count_mismatches(
    left_profiles: list[dict[str, Any]],
    right_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    right_by_file = {profile["file"]: profile for profile in right_profiles}
    mismatches = []
    for left in left_profiles:
        right = right_by_file.get(left["file"])
        if right is None:
            mismatches.append({"file": left["file"], "left_rows": left["rows"], "right_rows": None})
        elif left["rows"] != right["rows"]:
            mismatches.append(
                {"file": left["file"], "left_rows": left["rows"], "right_rows": right["rows"]}
            )
    return mismatches


def profile_interpretation_label(path: Path) -> dict[str, Any]:
    intervals = 0
    channels: set[int] = set()
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            if ":" not in raw:
                continue
            intervals += 1
            _, channel_text = raw.strip().split(":", maxsplit=1)
            for value in channel_text.split(","):
                value = value.strip()
                if value.isdigit():
                    channels.add(int(value))
    return {"file": path.name, "intervals": intervals, "channels": sorted(channels)}


def new_column_stats(columns: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": column,
            "rows": 0,
            "missing": 0,
            "first": None,
            "constant": True,
            "unique": set(),
            "unique_exact": True,
            "unique_count": 0,
            "can_int": True,
            "can_float": True,
            "can_bool": True,
            "can_datetime": is_temporal_name(column),
            "min": math.inf,
            "max": -math.inf,
            "sum": 0.0,
            "numeric_count": 0,
            "mean": None,
        }
        for column in columns
    ]


def update_column_stats(stats: list[dict[str, Any]], row: list[str]) -> None:
    for item, value in zip(stats, row, strict=True):
        item["rows"] += 1
        clean = value.strip()
        if clean == "":
            item["missing"] += 1
            continue

        if item["first"] is None:
            item["first"] = clean
        elif item["first"] != clean:
            item["constant"] = False

        if item["unique_exact"]:
            unique = item["unique"]
            if len(unique) >= UNIQUE_VALUE_LIMIT and clean not in unique:
                item["unique_exact"] = False
                unique.clear()
            else:
                unique.add(clean)

        if item["can_bool"] and not is_bool_like(clean):
            item["can_bool"] = False
        if item["can_int"] and not is_int(clean):
            item["can_int"] = False
        if item["can_float"]:
            try:
                number = float(clean)
            except ValueError:
                item["can_float"] = False
            else:
                item["min"] = min(item["min"], number)
                item["max"] = max(item["max"], number)
                item["sum"] += number
                item["numeric_count"] += 1
        # Timestamp parseability is assessed by the temporal profiler once per row.


def finalize_numeric_stats(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = []
    for item in stats:
        item["unique_count"] = (
            len(item["unique"]) if item["unique_exact"] else UNIQUE_VALUE_LIMIT + 1
        )
        if item["numeric_count"]:
            item["mean"] = item["sum"] / item["numeric_count"]
        if infer_type(item) in {"int", "float"}:
            numeric.append(item)
    return numeric


def infer_type(item: dict[str, Any]) -> str:
    item["unique_count"] = len(item["unique"]) if item["unique_exact"] else UNIQUE_VALUE_LIMIT + 1
    if item["rows"] == item["missing"]:
        return "empty"
    if item["can_bool"]:
        return "bool"
    if item["can_int"]:
        return "int"
    if item["can_float"]:
        return "float"
    if item["can_datetime"]:
        return "datetime"
    return "text"


def infer_categorical_columns(stats: list[dict[str, Any]], row_count: int) -> list[str]:
    categorical = []
    for item in stats:
        inferred = infer_type(item)
        unique_count = item["unique_count"]
        if item["name"] in infer_identifier_columns(stats):
            continue
        if inferred == "bool" and item["unique_exact"]:
            categorical.append(item["name"])
        elif (
            inferred == "text"
            and item["unique_exact"]
            and row_count
            and unique_count <= min(max(50, int(row_count * 0.2)), 5_000)
        ):
            categorical.append(item["name"])
    return categorical


def infer_identifier_columns(stats: list[dict[str, Any]]) -> list[str]:
    identifiers = []
    for item in stats:
        lower = item["name"].lower()
        unique_count = item["unique_count"]
        rows = item["rows"]
        if lower in {"udi", "product id", "lineid", "line_id", "unnamed_index"}:
            identifiers.append(item["name"])
        elif lower.endswith("id") and rows and unique_count / rows > 0.9:
            identifiers.append(item["name"])
    return identifiers


def normalize_width(values: list[str], width: int) -> list[str]:
    if len(values) == width:
        return values
    if len(values) > width:
        return values[: width - 1] + [",".join(values[width - 1 :])]
    return values + [""] * (width - len(values))


def extract_time_value(
    columns: list[str],
    row: list[str],
    timestamp_column: str | None = None,
) -> str | None:
    if timestamp_column and timestamp_column in columns:
        return row[columns.index(timestamp_column)].strip()

    lowered = [column.lower() for column in columns]
    if "timestamp" in lowered:
        return row[lowered.index("timestamp")].strip()
    if "date" in lowered and "time" in lowered:
        return f"{row[lowered.index('date')].strip()} {row[lowered.index('time')].strip()}"
    return None


def parse_datetime_value(value: str) -> datetime | None:
    clean = value.strip().strip('"')
    if not clean:
        return None
    if clean.isdigit() and len(clean) >= 10:
        try:
            return datetime.fromtimestamp(int(clean[:10]), tz=UTC).replace(tzinfo=None)
        except (OverflowError, ValueError, OSError):
            pass
    clean = clean.replace(",", ".")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    return None


def new_temporal_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "min": None,
        "max": None,
        "last": None,
        "deltas": Counter(),
    }


def update_temporal_stats(stats: dict[str, Any], value: str) -> None:
    parsed = parse_datetime_value(value)
    if parsed is None:
        return
    stats["count"] += 1
    if stats["min"] is None or parsed < stats["min"]:
        stats["min"] = parsed
    if stats["max"] is None or parsed > stats["max"]:
        stats["max"] = parsed
    if stats["last"] is not None:
        delta = int((parsed - stats["last"]).total_seconds())
        stats["deltas"][delta] += 1
    stats["last"] = parsed


def describe_temporal_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if not stats["count"]:
        return {
            "has_parseable_time": False,
            "min": None,
            "max": None,
            "duration_seconds": None,
            "top_deltas_seconds": [],
            "is_regular": None,
            "gap_count_gt_1_5x_median": None,
        }
    counts = stats["deltas"]
    median_delta = weighted_median(counts)
    gap_count = (
        sum(count for delta, count in counts.items() if median_delta and delta > median_delta * 1.5)
        if median_delta
        else 0
    )
    min_time: datetime = stats["min"]
    max_time: datetime = stats["max"]
    return {
        "has_parseable_time": True,
        "min": min_time.isoformat(sep=" "),
        "max": max_time.isoformat(sep=" "),
        "duration_seconds": int((max_time - min_time).total_seconds()),
        "top_deltas_seconds": counts.most_common(10),
        "median_delta_seconds": median_delta,
        "is_regular": len(counts) <= 1 if counts else True,
        "gap_count_gt_1_5x_median": gap_count,
    }


def weighted_median(counts: Counter[int]) -> float | None:
    total = sum(counts.values())
    if total == 0:
        return None
    midpoint = (total - 1) / 2
    cumulative = 0
    ordered = sorted(counts.items())
    for index, (value, count) in enumerate(ordered):
        previous = cumulative
        cumulative += count
        if previous <= midpoint < cumulative:
            if total % 2:
                return float(value)
            if midpoint + 1 < cumulative:
                return float(value)
            return (value + ordered[index + 1][0]) / 2
    return float(ordered[-1][0])


def detect_loghub_timestamp_column(path: Path | None) -> str | None:
    if path is None:
        return None
    if "BGL" in path.name:
        return "Timestamp"
    return None


def column_counts(path: Path, column: str, limit: int | None = None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            counts[row.get(column, "")] += 1
    common = counts.most_common(limit)
    return dict(common if limit else sorted(counts.items()))


def identify_log_columns(columns: list[str]) -> list[str]:
    names = []
    for column in columns:
        lower = column.lower()
        if lower in {"pid", "process", "node", "addr", "id"} or lower.endswith("id"):
            names.append(column)
    return names


def count_lines(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def is_bool_like(value: str) -> bool:
    clean = value.strip().lower()
    if clean in {"true", "false"}:
        return True
    try:
        return float(clean) in {0.0, 1.0}
    except ValueError:
        return False


def is_temporal_name(column: str) -> bool:
    lower = column.lower()
    return "time" in lower or "date" in lower or lower == "timestamp"


def min_known_datetime(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def max_known_datetime(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def format_summary(audit: dict[str, Any]) -> str:
    lines = [
        f"Generated: {audit['generated_at']}",
        f"Raw data size: {audit['raw_total_size_human']}",
        "",
    ]
    for dataset in audit["datasets"]:
        profile = dataset["profile"]
        rows = profile.get("rows")
        if rows is None and "structured" in profile:
            rows = profile["structured"].get("rows")
        lines.append(
            f"{dataset['id']:<20} {dataset['status']:<18} "
            f"{dataset['size_human']:>10} rows={rows if rows is not None else 'n/a'}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
