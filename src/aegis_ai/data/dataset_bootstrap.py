"""Dataset download orchestration for the AegisAI bootstrap phase."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import requests

from aegis_ai.data.dataset_registry import (
    DatasetDefinition,
    alias_index,
    load_dataset_catalog,
    normalize_dataset_key,
    project_root,
)
from aegis_ai.data.downloads import (
    DownloadError,
    create_session,
    download_github_files,
    download_github_tree,
    safe_extract_zip,
    stream_download,
)
from aegis_ai.data.manifest import DatasetDownloadResult, update_manifest

LOGHUB_DATASET_MAP = {
    "loghub-hdfs": ("HDFS", "hdfs"),
    "loghub-bgl": ("BGL", "bgl"),
    "loghub-openstack": ("OpenStack", "openstack"),
    "loghub-hadoop": ("Hadoop", "hadoop"),
    "loghub-spark": ("Spark", "spark"),
    "loghub-zookeeper": ("Zookeeper", "zookeeper"),
}
LOGHUB_SMALL_IDS = tuple(LOGHUB_DATASET_MAP)
PUBLIC_GROUP_IDS = (
    "nab",
    "smd",
    "smap-msl",
    *LOGHUB_SMALL_IDS,
    "opentelemetry-demo",
    "metropt",
    "ai4i",
)
MANUAL_DATASET_IDS = ("swat", "wadi", "cicids2017")
OPENTELEMETRY_REFERENCE_FILES = (
    "README.md",
    "LICENSE",
    "src/log-generator/app.py",
    "src/log-generator/requirements.txt",
    "src/otel-collector/otelcol-config.yml",
    "src/otel-collector/otelcol-config-extras.yml",
    "src/prometheus/prometheus-config.yaml",
    "src/grafana/provisioning/dashboards/demo/demo-dashboard.json",
    "src/grafana/provisioning/dashboards/demo/apm-dashboard.json",
    "test/tracetesting/run.bash",
    "test/tracetesting/tracetest-config.yaml",
)


def list_available_datasets() -> list[DatasetDefinition]:
    """Return datasets sorted by id."""

    return [item for _, item in sorted(load_dataset_catalog().items())]


def resolve_dataset_selection(selection: str, *, include_large: bool = False) -> tuple[str, ...]:
    """Resolve a CLI dataset selection to canonical dataset ids."""

    catalog = load_dataset_catalog()
    normalized = normalize_dataset_key(selection)
    if normalized == "public":
        return tuple(
            dataset_id
            for dataset_id in PUBLIC_GROUP_IDS
            if include_large or not catalog[dataset_id].large
        )
    if normalized == "all":
        return tuple(sorted(catalog))
    if normalized in {"loghub", "all-small"}:
        return LOGHUB_SMALL_IDS

    aliases = alias_index(catalog)
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases) | {"public", "all", "loghub", "all-small"}))
        raise ValueError(f"Unknown dataset selection '{selection}'. Valid selections: {valid}")
    dataset_id = aliases[normalized]
    definition = catalog[dataset_id]
    if definition.large and not include_large and not definition.requires_manual_access:
        raise ValueError(
            f"{definition.id} is marked large/manual. Re-run with --large if you really want "
            "to include large/manual dataset handling."
        )
    return (dataset_id,)


def download_dataset(
    dataset_id: str,
    *,
    root: Path | None = None,
    include_large: bool = False,
    force: bool = False,
    retries: int = 3,
    timeout_seconds: int = 60,
) -> DatasetDownloadResult:
    """Download or record one dataset according to the catalog."""

    root = root or project_root()
    catalog = load_dataset_catalog(root / "config" / "datasets.yaml")
    definition = catalog[dataset_id]
    definition.destination.mkdir(parents=True, exist_ok=True)

    if definition.requires_manual_access:
        return _manual_result(definition)
    if definition.large and not include_large:
        return _result(
            definition,
            status="skipped_large",
            message="Skipped because the dataset is marked large and --large was not provided.",
        )

    session = create_session(retries)
    try:
        if dataset_id == "nab":
            return _download_nab(definition, session, timeout_seconds, force)
        if dataset_id == "smd":
            return _download_smd(definition, session, timeout_seconds, force)
        if dataset_id == "smap-msl":
            return _download_smap_msl(definition, session, timeout_seconds, force)
        if dataset_id in LOGHUB_DATASET_MAP:
            return _download_loghub_sample(definition, session, timeout_seconds, force)
        if dataset_id == "opentelemetry-demo":
            return _download_opentelemetry_reference(definition, session, timeout_seconds, force)
        if dataset_id == "metropt":
            return _download_zip_dataset(
                definition,
                archive_name="metropt_dataset.zip",
                session=session,
                timeout_seconds=timeout_seconds,
                force=force,
            )
        if dataset_id == "ai4i":
            return _download_zip_dataset(
                definition,
                archive_name="ai4i_2020_dataset.zip",
                session=session,
                timeout_seconds=timeout_seconds,
                force=force,
            )
    except DownloadError as exc:
        return _result(definition, status="failed", message=str(exc))

    return _result(definition, status="failed", message="No downloader is registered.")


def download_many(
    dataset_ids: Iterable[str],
    *,
    root: Path | None = None,
    include_large: bool = False,
    force: bool = False,
    retries: int = 3,
    timeout_seconds: int = 60,
    update_manifest_file: bool = True,
) -> list[DatasetDownloadResult]:
    """Download multiple datasets and optionally update the manifest after each one."""

    root = root or project_root()
    results: list[DatasetDownloadResult] = []
    seen: set[str] = set()
    for dataset_id in dataset_ids:
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        print(f"\n==> {dataset_id}")
        result = download_dataset(
            dataset_id,
            root=root,
            include_large=include_large,
            force=force,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        if update_manifest_file:
            update_manifest(root, result)
        print(f"{result.dataset_id}: {result.status.upper()} {result.message}".strip())
        results.append(result)
    return results


def _download_nab(
    definition: DatasetDefinition,
    session: requests.Session,
    timeout_seconds: int,
    force: bool,
) -> DatasetDownloadResult:
    files = download_github_tree(
        owner="numenta",
        repository="NAB",
        ref="master",
        include_prefixes=("data", "labels", "README.md", "LICENSE.txt"),
        destination=definition.destination,
        session=session,
        timeout_seconds=timeout_seconds,
        force=force,
    )
    return _result(definition, status="ready", files=files)


def _download_smd(
    definition: DatasetDefinition,
    session: requests.Session,
    timeout_seconds: int,
    force: bool,
) -> DatasetDownloadResult:
    files = download_github_tree(
        owner="das2ar",
        repository="OmniAnomaly",
        ref="master",
        include_prefixes=("ServerMachineDataset",),
        destination=definition.destination,
        session=session,
        timeout_seconds=timeout_seconds,
        strip_prefix="ServerMachineDataset",
        force=force,
    )
    return _result(definition, status="ready", files=files)


def _download_smap_msl(
    definition: DatasetDefinition,
    session: requests.Session,
    timeout_seconds: int,
    force: bool,
) -> DatasetDownloadResult:
    files: list[Path] = []
    labels_url = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
    label_path = stream_download(
        labels_url,
        definition.destination / "labeled_anomalies.csv",
        session=session,
        timeout_seconds=timeout_seconds,
        force=force,
    )
    files.append(label_path)

    data_url = definition.download_url
    if not data_url:
        raise DownloadError("Missing download URL for SMAP/MSL.")
    archive_path = stream_download(
        data_url,
        definition.destination / "data.zip",
        session=session,
        timeout_seconds=timeout_seconds,
        force=force,
    )
    files.append(archive_path)
    files.extend(safe_extract_zip(archive_path, definition.destination, force=force))
    return _result(definition, status="ready", files=tuple(files))


def _download_loghub_sample(
    definition: DatasetDefinition,
    session: requests.Session,
    timeout_seconds: int,
    force: bool,
) -> DatasetDownloadResult:
    upstream_dir, _local_dir = LOGHUB_DATASET_MAP[definition.id]
    files = download_github_tree(
        owner="logpai",
        repository="loghub",
        ref="master",
        include_prefixes=(upstream_dir,),
        destination=definition.destination,
        session=session,
        timeout_seconds=timeout_seconds,
        strip_prefix=upstream_dir,
        force=force,
    )
    return _result(definition, status="ready", files=files)


def _download_opentelemetry_reference(
    definition: DatasetDefinition,
    session: requests.Session,
    timeout_seconds: int,
    force: bool,
) -> DatasetDownloadResult:
    files = download_github_files(
        owner="openobserve",
        repository="opentelemetry-demo-dataset",
        ref="main",
        file_paths=OPENTELEMETRY_REFERENCE_FILES,
        destination=definition.destination,
        session=session,
        timeout_seconds=timeout_seconds,
        force=force,
    )
    message = "Downloaded selected reference files; this repository is not a static telemetry dump."
    return _result(definition, status="reference_ready", files=files, message=message)


def _download_zip_dataset(
    definition: DatasetDefinition,
    *,
    archive_name: str,
    session: requests.Session,
    timeout_seconds: int,
    force: bool,
) -> DatasetDownloadResult:
    if not definition.download_url:
        raise DownloadError(f"Missing download URL for {definition.id}.")
    archive_path = stream_download(
        definition.download_url,
        definition.destination / archive_name,
        session=session,
        timeout_seconds=timeout_seconds,
        force=force,
    )
    files = [archive_path]
    files.extend(safe_extract_zip(archive_path, definition.destination, force=force))
    return _result(definition, status="ready", files=tuple(files))


def _manual_result(definition: DatasetDefinition) -> DatasetDownloadResult:
    message = f"Manual access required. See {definition.destination / 'README.md'}"
    return _result(definition, status="manual_required", message=message)


def _result(
    definition: DatasetDefinition,
    *,
    status: str,
    files: Iterable[Path] = (),
    message: str = "",
) -> DatasetDownloadResult:
    urls = tuple(
        value
        for value in (definition.download_url, definition.official_source)
        if isinstance(value, str) and value
    )
    return DatasetDownloadResult(
        dataset_id=definition.id,
        name=definition.name,
        status=status,
        local_path=definition.destination,
        source_urls=urls,
        license=definition.license,
        notes=definition.notes,
        message=message,
        files=tuple(files),
    )
