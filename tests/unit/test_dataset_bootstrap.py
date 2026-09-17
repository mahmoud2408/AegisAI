from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from aegis_ai.data.dataset_bootstrap import resolve_dataset_selection
from aegis_ai.data.dataset_registry import load_dataset_catalog
from aegis_ai.data.downloads import DownloadError, safe_extract_zip
from aegis_ai.data.validation import validate_dataset

ROOT = Path(__file__).resolve().parents[2]


def test_dataset_catalog_contains_required_entries() -> None:
    catalog = load_dataset_catalog(ROOT / "config" / "datasets.yaml")

    required_ids = {
        "ai4i",
        "cicids2017",
        "loghub-bgl",
        "loghub-hadoop",
        "loghub-hdfs",
        "loghub-openstack",
        "loghub-spark",
        "loghub-zookeeper",
        "metropt",
        "nab",
        "opentelemetry-demo",
        "smap-msl",
        "smd",
        "swat",
        "wadi",
    }

    assert required_ids.issubset(catalog)


def test_public_selection_excludes_manual_large_datasets() -> None:
    selection = set(resolve_dataset_selection("public"))

    assert {"swat", "wadi", "cicids2017"}.isdisjoint(selection)
    assert {"nab", "smd", "metropt", "ai4i"}.issubset(selection)


def test_all_selection_includes_manual_datasets_for_instructional_output() -> None:
    selection = set(resolve_dataset_selection("all"))

    assert {"swat", "wadi", "cicids2017"}.issubset(selection)


def test_manual_dataset_alias_can_be_selected_without_large_flag() -> None:
    assert resolve_dataset_selection("swat") == ("swat",)


def test_loghub_alias_resolves_to_single_dataset() -> None:
    assert resolve_dataset_selection("hdfs") == ("loghub-hdfs",)


def test_safe_extract_zip_blocks_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    with pytest.raises(DownloadError, match="Unsafe ZIP member path"):
        safe_extract_zip(archive_path, tmp_path / "out")


def test_manual_dataset_reports_manual_access_when_only_readme_exists() -> None:
    catalog = load_dataset_catalog(ROOT / "config" / "datasets.yaml")

    result = validate_dataset(catalog["swat"])

    assert result.status == "MANUAL_ACCESS"
