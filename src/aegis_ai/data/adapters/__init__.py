"""Dataset adapter registry."""

from __future__ import annotations

from pathlib import Path

from aegis_ai.data.adapters.ai4i import AI4IAdapter
from aegis_ai.data.adapters.base import BaseDatasetAdapter
from aegis_ai.data.adapters.loghub import LOGHUB_DATASETS, LogHubAdapter
from aegis_ai.data.adapters.metropt import MetroPTAdapter
from aegis_ai.data.adapters.nab import NABAdapter
from aegis_ai.data.adapters.smap_msl import MSLAdapter, SMAPAdapter
from aegis_ai.data.adapters.smd import SMDAdapter

PRIMARY_DATASETS = ("nab", "smd", "metropt", "ai4i", "smap", "msl", "loghub-openstack")


def available_adapter_ids() -> tuple[str, ...]:
    """Return all adapter ids accepted by the registry."""

    ids = ["nab", "smd", "metropt", "ai4i", "smap", "msl"]
    ids.extend(sorted(LOGHUB_DATASETS.values()))
    ids.extend(sorted(LOGHUB_DATASETS))
    return tuple(ids)


def create_adapter(dataset_id: str, root: Path | None = None) -> BaseDatasetAdapter:
    """Create a dataset adapter by canonical id or supported alias."""

    normalized = dataset_id.strip().lower().replace("_", "-")
    if normalized == "nab":
        return NABAdapter(root)
    if normalized == "smd":
        return SMDAdapter(root)
    if normalized == "metropt":
        return MetroPTAdapter(root)
    if normalized == "ai4i":
        return AI4IAdapter(root)
    if normalized == "smap":
        return SMAPAdapter(root)
    if normalized == "msl":
        return MSLAdapter(root)
    if normalized in LOGHUB_DATASETS:
        return LogHubAdapter(normalized, root)
    for family, configured_id in LOGHUB_DATASETS.items():
        if normalized == configured_id:
            return LogHubAdapter(family, root)
    raise KeyError(f"no adapter registered for dataset: {dataset_id}")


__all__ = [
    "AI4IAdapter",
    "BaseDatasetAdapter",
    "LogHubAdapter",
    "MSLAdapter",
    "MetroPTAdapter",
    "NABAdapter",
    "PRIMARY_DATASETS",
    "SMAPAdapter",
    "SMDAdapter",
    "available_adapter_ids",
    "create_adapter",
]
