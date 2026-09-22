"""Document loading and text extraction for Phase 11 knowledge ingestion."""

from __future__ import annotations

import hashlib
import importlib
import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, cast

import yaml

from aegis_ai.rag.models import KnowledgeDocument, Metadata

SUPPORTED_LOADERS = {
    ".md": "markdown",
    ".txt": "text",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


class BaseDocumentLoader(ABC):
    """Base loader interface for untrusted local documents."""

    def __init__(self, path: Path, *, knowledge_root: Path, default_language: str = "en") -> None:
        self.path = path
        self.knowledge_root = knowledge_root
        self.default_language = default_language

    def load(self) -> KnowledgeDocument:
        """Extract content and metadata into a normalized document."""

        text = normalize_text(self.extract_text())
        metadata = self.extract_metadata()
        title = str(metadata.get("title") or self.path.stem.replace("_", " ").title())
        domain = str(metadata.get("domain") or self._domain_from_path())
        source = str(metadata.get("source") or "unknown")
        version = str(metadata.get("version") or "unknown")
        language = str(metadata.get("language") or self.default_language)
        source_url = _optional_string(metadata.get("source_url"))
        document_type = str(metadata.get("document_type") or self.document_type)
        created_at = _created_at(metadata, self.path)
        document_id = str(metadata.get("document_id") or stable_document_id(self.path, title))
        enriched_metadata = dict(metadata)
        enriched_metadata.update(
            {
                "path": self._relative_path(),
                "loader": self.document_type,
                "technology": metadata.get("technology") or self.path.parent.name,
                "provenance": metadata.get("provenance") or source,
            }
        )
        return KnowledgeDocument(
            document_id=document_id,
            title=title,
            source=source,
            source_url=source_url,
            version=version,
            domain=domain,
            document_type=document_type,
            language=language,
            created_at=created_at,
            content=text,
            metadata=_metadata_strings(enriched_metadata),
        )

    @property
    @abstractmethod
    def document_type(self) -> str:
        """Return a stable document type name."""

    @abstractmethod
    def extract_text(self) -> str:
        """Extract text without executing document content."""

    def extract_metadata(self) -> dict[str, Any]:
        """Extract metadata. Subclasses may override."""

        return {}

    def _domain_from_path(self) -> str:
        try:
            first = self.path.relative_to(self.knowledge_root).parts[0]
        except ValueError:
            first = self.path.parent.name
        return DOMAIN_BY_DIRECTORY.get(first, first)

    def _relative_path(self) -> str:
        try:
            return self.path.relative_to(self.knowledge_root).as_posix()
        except ValueError:
            return self.path.as_posix()


class MarkdownDocumentLoader(BaseDocumentLoader):
    """Load Markdown documents with optional YAML front matter."""

    @property
    def document_type(self) -> str:
        return "markdown"

    def extract_text(self) -> str:
        content, _metadata = _split_front_matter(self.path.read_text(encoding="utf-8"))
        return content

    def extract_metadata(self) -> dict[str, Any]:
        _content, metadata = _split_front_matter(self.path.read_text(encoding="utf-8"))
        return metadata


class TextDocumentLoader(BaseDocumentLoader):
    """Load plain text documents."""

    @property
    def document_type(self) -> str:
        return "text"

    def extract_text(self) -> str:
        content, _metadata = _split_front_matter(self.path.read_text(encoding="utf-8"))
        return content

    def extract_metadata(self) -> dict[str, Any]:
        _content, metadata = _split_front_matter(self.path.read_text(encoding="utf-8"))
        return metadata


class HTMLDocumentLoader(BaseDocumentLoader):
    """Load HTML by extracting visible text and selected metadata."""

    @property
    def document_type(self) -> str:
        return "html"

    def extract_text(self) -> str:
        parser = _VisibleHTMLTextParser()
        parser.feed(self.path.read_text(encoding="utf-8"))
        return parser.text()

    def extract_metadata(self) -> dict[str, Any]:
        parser = _HTMLMetadataParser()
        parser.feed(self.path.read_text(encoding="utf-8"))
        metadata = parser.metadata
        if parser.title and "title" not in metadata:
            metadata["title"] = parser.title
        return metadata


class PDFDocumentLoader(BaseDocumentLoader):
    """Load PDF text with optional pypdf if it is installed."""

    @property
    def document_type(self) -> str:
        return "pdf"

    def extract_text(self) -> str:
        pypdf = importlib.import_module("pypdf")
        reader = pypdf.PdfReader(str(self.path))
        parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            parts.append(str(page_text))
        return "\n\n".join(parts)


class _VisibleHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag.lower() in {"p", "br", "div", "section", "h1", "h2", "h3", "li"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag.lower() in {"p", "div", "section", "h1", "h2", "h3", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return normalize_text(" ".join(self._parts))


class _HTMLMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.metadata: dict[str, Any] = {}
        self.title: str | None = None
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
        if lower == "title":
            self._in_title = True
        if lower == "meta":
            name = attrs_dict.get("name") or attrs_dict.get("property")
            content = attrs_dict.get("content")
            allowed = {
                "document_id",
                "domain",
                "language",
                "provenance",
                "source",
                "source_url",
                "technology",
                "title",
                "version",
            }
            if name and content and name in allowed:
                self.metadata[name] = content

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
            title = normalize_text(" ".join(self._title_parts))
            self.title = title or None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)


DOMAIN_BY_DIRECTORY = {
    "openstack": "cloud_infrastructure",
    "linux": "operating_system",
    "docker": "containers",
    "kubernetes": "containers",
    "databases": "databases",
    "networking": "networking",
    "monitoring": "monitoring",
    "incident_runbooks": "incident_response",
    "synthetic": "industrial",
}


def load_documents(
    knowledge_dir: Path,
    *,
    supported_extensions: tuple[str, ...],
    default_language: str = "en",
) -> list[KnowledgeDocument]:
    """Load supported documents under the knowledge directory."""

    documents: list[KnowledgeDocument] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "retrieval_eval.jsonl":
            continue
        if path.suffix.lower() not in supported_extensions:
            continue
        loader = create_document_loader(
            path, knowledge_root=knowledge_dir, default_language=default_language
        )
        documents.append(loader.load())
    return documents


def create_document_loader(
    path: Path,
    *,
    knowledge_root: Path,
    default_language: str = "en",
) -> BaseDocumentLoader:
    """Return a loader for the file extension."""

    suffix = path.suffix.lower()
    if suffix in {".md"}:
        return MarkdownDocumentLoader(
            path, knowledge_root=knowledge_root, default_language=default_language
        )
    if suffix in {".txt"}:
        return TextDocumentLoader(
            path, knowledge_root=knowledge_root, default_language=default_language
        )
    if suffix in {".html", ".htm"}:
        return HTMLDocumentLoader(
            path, knowledge_root=knowledge_root, default_language=default_language
        )
    if suffix == ".pdf":
        return PDFDocumentLoader(
            path, knowledge_root=knowledge_root, default_language=default_language
        )
    raise ValueError(f"Unsupported document extension: {path.suffix}")


def stable_document_id(path: Path, title: str) -> str:
    """Create a stable document id from path and title."""

    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    digest = hashlib.sha1(f"{path.as_posix()}|{title}".encode()).hexdigest()[:8]
    return f"{stem}_{digest}"


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_front_matter(text: str) -> tuple[str, dict[str, Any]]:
    if not text.startswith("---\n"):
        return text, {}
    try:
        _empty, metadata_text, body = text.split("---", 2)
    except ValueError:
        return text, {}
    metadata_raw = yaml.safe_load(metadata_text) or {}
    if not isinstance(metadata_raw, dict):
        return body, {}
    return body.lstrip("\n"), cast(dict[str, Any], metadata_raw)


def _created_at(metadata: dict[str, Any], path: Path) -> datetime:
    value = metadata.get("created_at")
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_strings(metadata: dict[str, Any]) -> Metadata:
    result: Metadata = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result
