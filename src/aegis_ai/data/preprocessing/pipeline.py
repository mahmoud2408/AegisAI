"""Dataset preprocessing orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from aegis_ai.data.adapters import PRIMARY_DATASETS, create_adapter
from aegis_ai.data.adapters.base import BaseDatasetAdapter, CanonicalStreams, RawDataset, Record
from aegis_ai.data.dataset_registry import project_root
from aegis_ai.data.preprocessing.io import (
    PartitionedParquetWriter,
    reset_generated_dataset_dir,
    write_json,
)
from aegis_ai.data.preprocessing.schemas import SCHEMAS


@dataclass(frozen=True)
class PreprocessResult:
    """Result of one dataset preprocessing run."""

    dataset_id: str
    adapter: str
    status: str
    input_records: int
    output_records: dict[str, int]
    output_files: dict[str, list[str]]
    profile_path: str
    lineage_path: str
    warnings: list[str] = field(default_factory=list)


def preprocess_dataset(
    dataset_id: str,
    *,
    root: Path | None = None,
    batch_size: int = 50_000,
) -> PreprocessResult:
    """Preprocess one configured dataset into canonical Parquet outputs."""

    root = root or project_root()
    adapter = create_adapter(dataset_id, root)
    raw = adapter.load_raw()
    validation = adapter.validate(raw)
    profile = adapter.profile(raw)

    profile_path = root / "data" / "manifests" / "profiles" / f"{adapter.dataset_id}.json"
    write_json(profile_path, asdict(profile))

    warnings = [issue.message for issue in validation.issues if issue.severity == "WARNING"]
    output_records: dict[str, int] = {}
    output_files: dict[str, list[str]] = {}

    if validation.status == "INCOMPLETE":
        lineage_path = write_lineage(
            root,
            adapter,
            raw,
            validation.status,
            profile.row_count,
            output_records,
            output_files,
            warnings=[issue.message for issue in validation.issues],
        )
        return PreprocessResult(
            dataset_id=adapter.dataset_id,
            adapter=adapter.adapter_name,
            status=validation.status,
            input_records=profile.row_count,
            output_records=output_records,
            output_files=output_files,
            profile_path=relative(root, profile_path),
            lineage_path=relative(root, lineage_path),
            warnings=[issue.message for issue in validation.issues],
        )

    streams = adapter.to_canonical(raw)
    for canonical_type, records in stream_items(streams):
        target_root = root / "data" / "processed" / canonical_type / adapter.dataset_id
        reset_generated_dataset_dir(target_root, root / "data" / "processed")
        writer = PartitionedParquetWriter(
            target_root,
            SCHEMAS[canonical_type],
            partition_keys=partition_keys_for(adapter.dataset_id, canonical_type),
            batch_size=batch_size,
        )
        count = writer.write_many(records)
        if count:
            output_records[canonical_type] = count
            output_files[canonical_type] = [relative(root, path) for path in writer.output_files()]

    lineage_path = write_lineage(
        root,
        adapter,
        raw,
        validation.status,
        profile.row_count,
        output_records,
        output_files,
        warnings=warnings,
    )

    return PreprocessResult(
        dataset_id=adapter.dataset_id,
        adapter=adapter.adapter_name,
        status=validation.status,
        input_records=profile.row_count,
        output_records=output_records,
        output_files=output_files,
        profile_path=relative(root, profile_path),
        lineage_path=relative(root, lineage_path),
        warnings=warnings,
    )


def preprocess_many(
    dataset_ids: list[str] | tuple[str, ...] = PRIMARY_DATASETS,
    *,
    root: Path | None = None,
    batch_size: int = 50_000,
) -> list[PreprocessResult]:
    """Preprocess multiple datasets and write a combined summary/report."""

    root = root or project_root()
    results = [
        preprocess_dataset(dataset_id, root=root, batch_size=batch_size)
        for dataset_id in dataset_ids
    ]
    write_preprocessing_summary(root, results)
    write_preprocessing_report(root, results)
    return results


def profile_dataset(dataset_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Generate one adapter profile JSON without writing processed outputs."""

    root = root or project_root()
    adapter = create_adapter(dataset_id, root)
    raw = adapter.load_raw()
    profile = adapter.profile(raw)
    profile_path = root / "data" / "manifests" / "profiles" / f"{adapter.dataset_id}.json"
    write_json(profile_path, asdict(profile))
    return asdict(profile)


def profile_many(
    dataset_ids: list[str] | tuple[str, ...] = PRIMARY_DATASETS,
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate adapter profiles for multiple datasets."""

    root = root or project_root()
    profiles = [profile_dataset(dataset_id, root=root) for dataset_id in dataset_ids]
    write_json(
        root / "data" / "manifests" / "profiles" / "summary.json",
        {"generated_at": datetime.now(UTC).isoformat(), "profiles": profiles},
    )
    return profiles


def validate_processed(
    dataset_ids: list[str] | tuple[str, ...] = PRIMARY_DATASETS,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Validate generated Parquet outputs are readable and row-counted."""

    root = root or project_root()
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": [],
    }
    for dataset_id in dataset_ids:
        adapter = create_adapter(dataset_id, root)
        dataset_result: dict[str, Any] = {"dataset_id": adapter.dataset_id, "types": {}}
        for canonical_type in SCHEMAS:
            output_dir = root / "data" / "processed" / canonical_type / adapter.dataset_id
            files = sorted(output_dir.rglob("*.parquet")) if output_dir.exists() else []
            rows = 0
            schemas = []
            for file_path in files:
                parquet_file = pq.ParquetFile(file_path)
                rows += parquet_file.metadata.num_rows
                schemas.append(str(parquet_file.schema_arrow))
            dataset_result["types"][canonical_type] = {
                "files": [relative(root, path) for path in files],
                "row_count": rows,
                "readable": all(path.exists() for path in files),
                "schema_count": len(set(schemas)),
            }
        summary["datasets"].append(dataset_result)
    write_json(root / "data" / "manifests" / "processed_validation.json", summary)
    return summary


def stream_items(streams: CanonicalStreams) -> Iterator[tuple[str, Iterable[Record]]]:
    """Yield named canonical streams in storage order."""

    yield "metrics", streams.metrics
    yield "logs", streams.logs
    yield "traces", streams.traces
    yield "incidents", streams.incidents
    yield "failures", streams.failures
    yield "labels", streams.labels


def partition_keys_for(dataset_id: str, canonical_type: str) -> tuple[str, ...]:
    """Choose conservative Parquet partition keys."""

    if canonical_type == "metrics" and dataset_id == "smd":
        return ("split", "entity_id")
    if canonical_type == "metrics" and dataset_id == "nab":
        return ("entity_id",)
    if canonical_type == "labels" and dataset_id == "smd":
        return ("entity_id",)
    return ()


def write_lineage(
    root: Path,
    adapter: BaseDatasetAdapter,
    raw: RawDataset,
    status: str,
    input_records: int,
    output_records: dict[str, int],
    output_files: dict[str, list[str]],
    *,
    warnings: list[str],
) -> Path:
    """Write reproducible lineage metadata for one processed dataset."""

    lineage = {
        "dataset_id": adapter.dataset_id,
        "adapter": adapter.adapter_name,
        "adapter_version": adapter.adapter_version,
        "processing_date": datetime.now(UTC).isoformat(),
        "status": status,
        "source_dataset": adapter.definition.id,
        "source_path": relative(root, raw.path),
        "source_files": [relative(root, path) for path in raw.files if path.name != ".gitkeep"],
        "input_records": input_records,
        "output_records": output_records,
        "rejected_records": 0,
        "output_files": output_files,
        "transformations_applied": [
            "schema validation",
            "timestamp normalization without invented timezone",
            "quality flag assignment",
            "canonical long-format conversion",
            "Parquet serialization with source lineage",
        ],
        "warnings": warnings,
    }
    return write_json(
        root / "data" / "manifests" / "lineage" / f"{adapter.dataset_id}.json", lineage
    )


def write_preprocessing_summary(root: Path, results: list[PreprocessResult]) -> Path:
    """Write a machine-readable preprocessing summary."""

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": [asdict(result) for result in results],
    }
    return write_json(root / "data" / "manifests" / "preprocessing_summary.json", payload)


def write_preprocessing_report(root: Path, results: list[PreprocessResult]) -> Path:
    """Write a concise human-readable preprocessing report."""

    lines = [
        "# Preprocessing Report",
        "",
        f"Generated: `{datetime.now(UTC).isoformat()}`",
        "",
        "| Dataset | Adapter | Status | Input Rows | Output Records |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for result in results:
        output = ", ".join(
            f"{name}: {count:,}" for name, count in sorted(result.output_records.items())
        )
        lines.append(
            f"| {result.dataset_id} | {result.adapter} | {result.status} | "
            f"{result.input_records:,} | {output or 'none'} |"
        )
    path = root / "data" / "manifests" / "preprocessing_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
