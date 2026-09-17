"""Dataset catalog loading for the AegisAI data bootstrap layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root from the installed source tree."""

    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetDefinition:
    """A single dataset entry from ``config/datasets.yaml``."""

    id: str
    name: str
    category: str
    description: str
    official_source: str
    download_method: str
    destination: Path
    expected_files: tuple[str, ...]
    requires_manual_access: bool
    license: str
    notes: str
    download_url: str | None = None
    safe_default: bool = True
    large: bool = False
    aliases: tuple[str, ...] = field(default_factory=tuple)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: dict[str, Any], root: Path) -> DatasetDefinition:
        """Build a typed dataset definition from a YAML mapping."""

        required_keys = {
            "id",
            "name",
            "category",
            "description",
            "official_source",
            "download_method",
            "destination",
            "expected_files",
            "requires_manual_access",
            "license",
            "notes",
        }
        missing = sorted(required_keys.difference(item))
        if missing:
            raise ValueError(f"Dataset entry is missing required keys: {missing}")

        destination = root / str(item["destination"])
        expected_files = tuple(str(value) for value in item.get("expected_files", []))
        aliases = tuple(str(value) for value in item.get("aliases", []))

        return cls(
            id=str(item["id"]),
            name=str(item["name"]),
            category=str(item["category"]),
            description=str(item["description"]),
            official_source=str(item["official_source"]),
            download_method=str(item["download_method"]),
            download_url=item.get("download_url"),
            destination=destination,
            expected_files=expected_files,
            requires_manual_access=bool(item["requires_manual_access"]),
            safe_default=bool(item.get("safe_default", True)),
            large=bool(item.get("large", False)),
            license=str(item["license"]),
            notes=str(item["notes"]),
            aliases=aliases,
            raw=dict(item),
        )


def load_dataset_catalog(config_path: Path | None = None) -> dict[str, DatasetDefinition]:
    """Load the dataset catalog keyed by dataset id."""

    path = config_path or project_root() / "config" / "datasets.yaml"
    root = path.parent.parent if path.parent.name == "config" else project_root()
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("config/datasets.yaml must contain a top-level 'datasets' list.")

    catalog = {definition.id: definition for definition in _build_definitions(datasets, root)}
    if len(catalog) != len(datasets):
        raise ValueError("Dataset ids must be unique.")
    return catalog


def alias_index(catalog: dict[str, DatasetDefinition]) -> dict[str, str]:
    """Return a lookup from id or alias to canonical dataset id."""

    index: dict[str, str] = {}
    for dataset_id, definition in catalog.items():
        keys = (dataset_id, *definition.aliases)
        for key in keys:
            normalized = normalize_dataset_key(key)
            if normalized in index and index[normalized] != dataset_id:
                raise ValueError(f"Dataset alias collision: {key}")
            index[normalized] = dataset_id
    return index


def normalize_dataset_key(value: str) -> str:
    """Normalize a dataset selector for CLI matching."""

    return value.strip().lower().replace("_", "-")


def _build_definitions(datasets: list[dict[str, Any]], root: Path) -> list[DatasetDefinition]:
    return [DatasetDefinition.from_mapping(item, root) for item in datasets]
