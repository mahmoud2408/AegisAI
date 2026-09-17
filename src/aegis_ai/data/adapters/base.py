"""Base classes and shared types for dataset adapters."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis_ai.data.dataset_registry import DatasetDefinition, load_dataset_catalog, project_root

Record = dict[str, Any]


@dataclass(frozen=True)
class RawDataset:
    """A source dataset directory and discovered files."""

    dataset_id: str
    path: Path
    files: tuple[Path, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    """A validation warning or error."""

    severity: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Structured validation result for a dataset adapter."""

    dataset_id: str
    status: str
    issues: tuple[ValidationIssue, ...] = ()
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_valid(self) -> bool:
        return self.status in {"READY", "REFERENCE_READY", "LABELS_ONLY"}


@dataclass(frozen=True)
class DatasetProfile:
    """Machine-readable dataset profile."""

    dataset_id: str
    status: str
    row_count: int
    column_count: int | None
    label_count: int
    profile: dict[str, Any]
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class CanonicalStreams:
    """Streaming canonical record collections produced by an adapter."""

    metrics: Iterable[Record] = ()
    logs: Iterable[Record] = ()
    traces: Iterable[Record] = ()
    incidents: Iterable[Record] = ()
    failures: Iterable[Record] = ()
    labels: Iterable[Record] = ()


class BaseDatasetAdapter(ABC):
    """Common interface implemented by all dataset adapters."""

    dataset_id: str
    adapter_version = "0.1.0"
    raw_dataset_id: str | None = None

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or project_root()
        self.catalog = load_dataset_catalog(self.root / "config" / "datasets.yaml")
        self.definition = self._resolve_definition()

    @property
    def adapter_name(self) -> str:
        return self.__class__.__name__

    @property
    def raw_path(self) -> Path:
        return self.definition.destination

    def load_raw(self) -> RawDataset:
        """Discover local raw files without modifying them."""

        files = tuple(sorted(item for item in self.raw_path.rglob("*") if item.is_file()))
        return RawDataset(self.dataset_id, self.raw_path, files, self.get_metadata())

    def validate(self, raw: RawDataset | None = None) -> ValidationReport:
        """Validate configured expected files for the adapter."""

        raw = raw or self.load_raw()
        issues: list[ValidationIssue] = []
        if not raw.path.exists():
            issues.append(
                ValidationIssue("ERROR", "raw dataset directory is missing", str(raw.path))
            )
        for pattern in self.definition.expected_files:
            if not list(raw.path.glob(pattern)):
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"missing expected file pattern: {pattern}",
                        str(raw.path),
                    )
                )
        return ValidationReport(
            self.dataset_id,
            "READY" if not issues else "INCOMPLETE",
            tuple(issues),
        )

    def profile(self, raw: RawDataset | None = None) -> DatasetProfile:
        """Return a minimal profile. Concrete adapters override this."""

        raw = raw or self.load_raw()
        return DatasetProfile(
            dataset_id=self.dataset_id,
            status=self.validate(raw).status,
            row_count=0,
            column_count=None,
            label_count=0,
            profile={"files": [self.relative_path(file) for file in raw.files]},
        )

    def transform(self, raw: RawDataset | None = None) -> RawDataset:
        """Source-specific interim transformation hook."""

        return raw or self.load_raw()

    def to_canonical(self, raw: RawDataset | None = None) -> CanonicalStreams:
        """Return streaming canonical records."""

        raw = raw or self.load_raw()
        return CanonicalStreams(
            metrics=self.iter_metric_records(raw),
            logs=self.iter_log_records(raw),
            traces=self.iter_trace_records(raw),
            incidents=self.iter_incident_records(raw),
            failures=self.iter_failure_records(raw),
            labels=self.get_labels(raw),
        )

    def iter_metric_records(self, raw: RawDataset) -> Iterable[Record]:
        return ()

    def iter_log_records(self, raw: RawDataset) -> Iterable[Record]:
        return ()

    def iter_trace_records(self, raw: RawDataset) -> Iterable[Record]:
        return ()

    def iter_incident_records(self, raw: RawDataset) -> Iterable[Record]:
        return ()

    def iter_failure_records(self, raw: RawDataset) -> Iterable[Record]:
        return ()

    def get_labels(self, raw: RawDataset) -> Iterable[Record]:
        return ()

    def get_metadata(self) -> dict[str, Any]:
        """Return adapter and source metadata."""

        return {
            "dataset_id": self.dataset_id,
            "adapter": self.adapter_name,
            "adapter_version": self.adapter_version,
            "source_dataset": self.definition.id,
            "source_url": self.definition.official_source,
            "license": self.definition.license,
        }

    def relative_path(self, path: Path) -> str:
        """Return a repository-relative path if possible."""

        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _resolve_definition(self) -> DatasetDefinition:
        definition_id = self.raw_dataset_id or self.dataset_id
        if definition_id not in self.catalog:
            raise KeyError(f"dataset is not configured: {definition_id}")
        return self.catalog[definition_id]


def canonical_event_id(*parts: object) -> str:
    """Create a deterministic readable canonical id from stable source parts."""

    return ":".join(str(part).replace("\\", "/") for part in parts)
