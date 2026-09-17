"""Dataset installation validation for the AegisAI data lake."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from aegis_ai.data.dataset_bootstrap import MANUAL_DATASET_IDS
from aegis_ai.data.dataset_registry import DatasetDefinition, load_dataset_catalog, project_root
from aegis_ai.data.manifest import directory_size, format_bytes

READY_STATUSES = {"ready", "reference_ready"}


@dataclass(frozen=True)
class DatasetValidationResult:
    """Validation status for one dataset directory."""

    dataset_id: str
    display_name: str
    status: str
    path: Path
    size_bytes: int
    problems: tuple[str, ...] = ()


def validate_all(root: Path | None = None) -> list[DatasetValidationResult]:
    """Validate all configured datasets."""

    root = root or project_root()
    catalog = load_dataset_catalog(root / "config" / "datasets.yaml")
    return [validate_dataset(definition) for _, definition in sorted(catalog.items())]


def validate_dataset(definition: DatasetDefinition) -> DatasetValidationResult:
    """Validate one dataset against its expected files and manual-access policy."""

    path = definition.destination
    problems: list[str] = []
    if not path.exists():
        problems.append("dataset directory is missing")
    else:
        for pattern in definition.expected_files:
            if not list(path.glob(pattern)):
                problems.append(f"missing expected file pattern: {pattern}")

    manual_data_present = (
        _manual_data_present(path) if definition.id in MANUAL_DATASET_IDS else False
    )
    if definition.id in MANUAL_DATASET_IDS and not manual_data_present:
        problems.append("manual dataset files not installed")

    if definition.id == "smap-msl":
        problems.extend(_validate_zip(path / "data.zip"))
    if definition.id in {"metropt", "ai4i"}:
        archives = list(path.glob("*.zip"))
        if not archives:
            problems.append("missing original ZIP archive")
        for archive in archives:
            problems.extend(_validate_zip(archive))

    status = _status_for(definition, problems, manual_data_present)
    return DatasetValidationResult(
        dataset_id=definition.id,
        display_name=_display_name(definition),
        status=status,
        path=path,
        size_bytes=directory_size(path),
        problems=tuple(problems),
    )


def format_validation_report(results: list[DatasetValidationResult]) -> str:
    """Return a console-friendly validation report."""

    width = max(len(result.display_name) for result in results) if results else 10
    lines = []
    for result in results:
        size = format_bytes(result.size_bytes)
        line = f"{result.display_name:<{width}}  {result.status:<16}  {size:>10}"
        if result.problems:
            line = f"{line}  - {'; '.join(result.problems)}"
        lines.append(line)
    return "\n".join(lines)


def _status_for(
    definition: DatasetDefinition, problems: list[str], manual_data_present: bool
) -> str:
    if definition.id in MANUAL_DATASET_IDS:
        if manual_data_present and not problems:
            return "PRESENT_UNVERIFIED"
        if definition.id == "cicids2017":
            return "MANUAL_DOWNLOAD"
        return "MANUAL_ACCESS"
    if not problems:
        if definition.id == "opentelemetry-demo":
            return "REFERENCE_READY"
        return "READY"
    return "INCOMPLETE"


def _display_name(definition: DatasetDefinition) -> str:
    names = {
        "ai4i": "AI4I",
        "cicids2017": "CICIDS2017",
        "loghub-bgl": "LOGHUB-BGL",
        "loghub-hadoop": "LOGHUB-HADOOP",
        "loghub-hdfs": "LOGHUB-HDFS",
        "loghub-openstack": "LOGHUB-OPENSTACK",
        "loghub-spark": "LOGHUB-SPARK",
        "loghub-zookeeper": "LOGHUB-ZOOKEEPER",
        "metropt": "METROPT",
        "nab": "NAB",
        "opentelemetry-demo": "OPENTELEMETRY",
        "smap-msl": "SMAP/MSL",
        "smd": "SMD",
        "swat": "SWAT",
        "wadi": "WADI",
    }
    return names.get(definition.id, definition.id.upper())


def _validate_zip(path: Path) -> list[str]:
    if not path.exists():
        return []
    problems: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            bad_file = archive.testzip()
            if bad_file:
                problems.append(f"corrupted archive member: {bad_file}")
    except zipfile.BadZipFile:
        problems.append(f"corrupted archive: {path.name}")
    return problems


def _manual_data_present(path: Path) -> bool:
    if not path.exists():
        return False
    placeholder_names = {".gitkeep", "README.md"}
    return any(item.is_file() and item.name not in placeholder_names for item in path.rglob("*"))
