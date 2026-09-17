"""Streaming Parquet output helpers."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from aegis_ai.data.preprocessing.cleaning import sanitize_partition_value

Record = dict[str, Any]


def ensure_within(child: Path, parent: Path) -> None:
    """Protect generated-output cleanup from escaping the intended directory."""

    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    if parent_resolved != child_resolved and parent_resolved not in child_resolved.parents:
        raise ValueError(f"{child} is outside {parent}")


def reset_generated_dataset_dir(path: Path, allowed_parent: Path) -> None:
    """Remove and recreate one generated dataset output directory."""

    ensure_within(path, allowed_parent)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


class PartitionedParquetWriter:
    """Write dictionaries to partitioned Parquet files without full-data buffering."""

    def __init__(
        self,
        output_root: Path,
        schema: pa.Schema,
        *,
        partition_keys: tuple[str, ...] = (),
        batch_size: int = 50_000,
    ) -> None:
        self.output_root = output_root
        self.schema = schema
        self.partition_keys = partition_keys
        self.batch_size = batch_size
        self.buffers: dict[tuple[str, ...], list[Record]] = defaultdict(list)
        self.writers: dict[tuple[str, ...], pq.ParquetWriter] = {}
        self.paths: dict[tuple[str, ...], Path] = {}
        self.row_count = 0

    def write_many(self, records: Iterable[Record]) -> int:
        """Write all records from an iterable."""

        for record in records:
            self.write(record)
        self.close()
        return self.row_count

    def write(self, record: Record) -> None:
        """Buffer one record and flush its partition when needed."""

        key = self._partition_key(record)
        self.buffers[key].append(serialize_record(record))
        self.row_count += 1
        if len(self.buffers[key]) >= self.batch_size:
            self._flush(key)

    def close(self) -> None:
        """Flush and close every partition writer."""

        for key in list(self.buffers):
            self._flush(key)
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def output_files(self) -> list[Path]:
        """Return every file written by the writer."""

        return sorted(self.paths.values())

    def _partition_key(self, record: Record) -> tuple[str, ...]:
        if not self.partition_keys:
            return ("__single__",)
        return tuple(str(record.get(key, "unknown")) for key in self.partition_keys)

    def _path_for(self, key: tuple[str, ...]) -> Path:
        if key in self.paths:
            return self.paths[key]
        if not self.partition_keys:
            directory = self.output_root
        else:
            parts = [
                f"{name}={sanitize_partition_value(value)}"
                for name, value in zip(self.partition_keys, key, strict=True)
            ]
            directory = self.output_root.joinpath(*parts)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part-00000.parquet"
        self.paths[key] = path
        return path

    def _writer_for(self, key: tuple[str, ...]) -> pq.ParquetWriter:
        if key not in self.writers:
            self.writers[key] = pq.ParquetWriter(
                self._path_for(key),
                self.schema,
                compression="snappy",
                use_dictionary=True,
            )
        return self.writers[key]

    def _flush(self, key: tuple[str, ...]) -> None:
        rows = self.buffers.get(key, [])
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=self.schema)
        self._writer_for(key).write_table(table)
        rows.clear()


def serialize_record(record: Record) -> Record:
    """Convert datetimes and nested metadata into Parquet-friendly values."""

    result: Record = {}
    for key, value in record.items():
        if key == "metadata":
            result["metadata_json"] = (
                value if isinstance(value, str) else json.dumps(value or {}, sort_keys=True)
            )
        elif hasattr(value, "isoformat"):
            result[key] = value.isoformat(sep=" ")
        else:
            result[key] = value
    if "metadata_json" not in result:
        result["metadata_json"] = "{}"
    return result


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a generated JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
