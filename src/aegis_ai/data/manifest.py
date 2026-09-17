"""Dataset manifest creation and storage reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

MANIFEST_RELATIVE_PATH = Path("data") / "manifests" / "datasets_manifest.json"
CHECKSUM_SIZE_LIMIT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class DatasetDownloadResult:
    """Result returned by dataset download handlers."""

    dataset_id: str
    name: str
    status: str
    local_path: Path
    source_urls: tuple[str, ...]
    license: str
    notes: str
    message: str = ""
    files: tuple[Path, ...] = field(default_factory=tuple)


def update_manifest(root: Path, result: DatasetDownloadResult) -> Path:
    """Upsert one dataset result into the JSON manifest."""

    manifest_path = root / MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _read_manifest(manifest_path)
    datasets = {
        str(item["dataset_id"]): item
        for item in payload.get("datasets", [])
        if isinstance(item, dict) and "dataset_id" in item
    }
    file_records = collect_file_records(result.local_path, root)

    datasets[result.dataset_id] = {
        "dataset_id": result.dataset_id,
        "name": result.name,
        "source_url": list(result.source_urls),
        "download_date": datetime.now(UTC).isoformat(),
        "local_path": _relative_path(result.local_path, root),
        "files": file_records,
        "total_size": sum(int(item["size_bytes"]) for item in file_records),
        "status": result.status,
        "license": result.license,
        "notes": result.notes,
        "message": result.message,
    }

    payload["generated_at"] = datetime.now(UTC).isoformat()
    payload["datasets"] = [datasets[key] for key in sorted(datasets)]
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def collect_file_records(path: Path, root: Path) -> list[dict[str, Any]]:
    """Collect file sizes and checksums for a dataset directory."""

    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        if file_path.name == ".gitkeep":
            continue
        size = file_path.stat().st_size
        record: dict[str, Any] = {
            "path": _relative_path(file_path, root),
            "size_bytes": size,
        }
        if size <= CHECKSUM_SIZE_LIMIT_BYTES:
            record["sha256"] = sha256_file(file_path)
        else:
            record["sha256"] = None
            record["checksum_note"] = "Skipped because file exceeds checksum size limit."
        records.append(record)
    return records


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    """Return the total byte size of files under a directory."""

    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_bytes(value: int) -> str:
    """Format bytes in a compact human-readable form."""

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"generated_at": None, "datasets": []}
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
