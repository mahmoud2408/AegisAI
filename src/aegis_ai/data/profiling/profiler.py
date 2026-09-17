"""Adapter-backed dataset profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aegis_ai.data.preprocessing.pipeline import profile_dataset, profile_many


def profile(dataset_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Profile one dataset and write its JSON manifest."""

    return profile_dataset(dataset_id, root=root)


def profile_all(
    dataset_ids: list[str] | tuple[str, ...],
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Profile multiple datasets and write a summary manifest."""

    return profile_many(dataset_ids, root=root)
