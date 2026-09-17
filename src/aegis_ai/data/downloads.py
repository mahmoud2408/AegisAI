"""Robust download helpers for public dataset bootstrap scripts."""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "AegisAI-dataset-bootstrap/0.1"
CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """Raised when a dataset download cannot be completed safely."""


def create_session(retries: int) -> requests.Session:
    """Create a requests session with bounded retries for transient failures."""

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        backoff_factor=1.0,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def stream_download(
    url: str,
    destination: Path,
    *,
    session: requests.Session,
    timeout_seconds: int,
    force: bool = False,
    resume: bool = True,
) -> Path:
    """Download a URL to a file with streaming, retries, and optional resume."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force and destination.stat().st_size > 0:
        print(f"SKIP existing file: {destination}")
        return destination
    if destination.exists() and force:
        destination.unlink()

    partial_path = destination.with_name(f"{destination.name}.part")
    existing_size = partial_path.stat().st_size if partial_path.exists() and resume else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    mode = "ab" if existing_size else "wb"

    try:
        with session.get(
            url,
            stream=True,
            timeout=timeout_seconds,
            headers=headers,
            allow_redirects=True,
        ) as response:
            if response.status_code == 416 and partial_path.exists():
                partial_path.replace(destination)
                return destination
            if response.status_code == 200 and existing_size:
                existing_size = 0
                mode = "wb"
            if response.status_code >= 400:
                raise DownloadError(f"{response.status_code} response while downloading {url}")

            content_length = int(response.headers.get("Content-Length", "0") or 0)
            total = (
                content_length + existing_size if response.status_code == 206 else content_length
            )
            with partial_path.open(mode) as file:
                with tqdm(
                    total=total or None,
                    initial=existing_size,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        file.write(chunk)
                        progress.update(len(chunk))
    except requests.RequestException as exc:
        raise DownloadError(f"Network error while downloading {url}: {exc}") from exc

    if destination.exists() and not force:
        raise DownloadError(f"Refusing to overwrite existing file: {destination}")
    partial_path.replace(destination)
    return destination


def download_github_tree(
    *,
    owner: str,
    repository: str,
    ref: str,
    include_prefixes: Iterable[str],
    destination: Path,
    session: requests.Session,
    timeout_seconds: int,
    strip_prefix: str | None = None,
    max_total_bytes: int | None = None,
    force: bool = False,
) -> list[Path]:
    """Download selected files from a GitHub tree without cloning the whole repository."""

    destination.mkdir(parents=True, exist_ok=True)
    api_url = f"https://api.github.com/repos/{owner}/{repository}/git/trees/{ref}?recursive=1"
    response = session.get(api_url, timeout=timeout_seconds)
    if response.status_code >= 400:
        raise DownloadError(f"{response.status_code} response while reading GitHub tree: {api_url}")

    tree_payload = response.json()
    selected = [
        item
        for item in tree_payload.get("tree", [])
        if item.get("type") == "blob"
        and _matches_prefix(str(item.get("path", "")), tuple(include_prefixes))
    ]
    selected.sort(key=lambda item: str(item["path"]))

    total_size = sum(int(item.get("size") or 0) for item in selected)
    if max_total_bytes is not None and total_size > max_total_bytes:
        raise DownloadError(
            f"Selected GitHub files total {total_size} bytes, above limit {max_total_bytes}."
        )

    downloaded: list[Path] = []
    for item in selected:
        upstream_path = str(item["path"])
        relative_path = _strip_prefix(upstream_path, strip_prefix)
        target = destination / Path(*PurePosixPath(relative_path).parts)
        raw_url = (
            f"https://raw.githubusercontent.com/{owner}/{repository}/{ref}/{quote(upstream_path)}"
        )
        downloaded.append(
            stream_download(
                raw_url,
                target,
                session=session,
                timeout_seconds=timeout_seconds,
                force=force,
            )
        )
    return downloaded


def download_github_files(
    *,
    owner: str,
    repository: str,
    ref: str,
    file_paths: Iterable[str],
    destination: Path,
    session: requests.Session,
    timeout_seconds: int,
    force: bool = False,
) -> list[Path]:
    """Download an explicit set of files from GitHub raw URLs."""

    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for upstream_path in file_paths:
        target = destination / Path(*PurePosixPath(upstream_path).parts)
        raw_url = (
            f"https://raw.githubusercontent.com/{owner}/{repository}/{ref}/{quote(upstream_path)}"
        )
        downloaded.append(
            stream_download(
                raw_url,
                target,
                session=session,
                timeout_seconds=timeout_seconds,
                force=force,
            )
        )
    return downloaded


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    strip_prefix: str | None = None,
    force: bool = False,
) -> list[Path]:
    """Safely extract a ZIP archive and prevent path traversal."""

    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise DownloadError(f"Corrupted ZIP member in {archive_path}: {bad_file}")

        for member in archive.infolist():
            if member.is_dir():
                continue
            relative_name = _safe_zip_member_name(member.filename, strip_prefix)
            if relative_name is None:
                continue
            target = (destination / relative_name).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as exc:
                raise DownloadError(f"Unsafe ZIP member path: {member.filename}") from exc
            if target.exists() and not force:
                extracted.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)
    return extracted


def _matches_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    normalized = path.strip("/")
    for prefix in prefixes:
        clean_prefix = prefix.strip("/")
        if normalized == clean_prefix or normalized.startswith(f"{clean_prefix}/"):
            return True
    return False


def _strip_prefix(path: str, prefix: str | None) -> str:
    normalized = path.strip("/")
    if not prefix:
        return normalized
    clean_prefix = prefix.strip("/")
    if normalized == clean_prefix:
        return PurePosixPath(normalized).name
    if normalized.startswith(f"{clean_prefix}/"):
        return normalized[len(clean_prefix) + 1 :]
    return normalized


def _safe_zip_member_name(filename: str, strip_prefix: str | None) -> Path | None:
    pure_path = PurePosixPath(filename)
    if pure_path.is_absolute() or any(part == ".." for part in pure_path.parts):
        raise DownloadError(f"Unsafe ZIP member path: {filename}")

    relative = _strip_prefix(str(pure_path), strip_prefix)
    if not relative or relative == ".":
        return None
    return Path(*PurePosixPath(relative).parts)
