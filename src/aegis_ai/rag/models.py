"""Typed document, chunk, retrieval, and evaluation models for Phase 11 RAG."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JSONScalar = str | int | float | bool | None
Metadata = dict[str, JSONScalar]


@dataclass(frozen=True)
class KnowledgeDocument:
    """Normalized document with preserved provenance."""

    document_id: str
    title: str
    source: str
    source_url: str | None
    version: str
    domain: str
    document_type: str
    language: str
    created_at: datetime
    content: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if not self.title:
            raise ValueError("title must not be empty")
        if not self.domain:
            raise ValueError("domain must not be empty")
        if not self.content.strip():
            raise ValueError("content must not be empty")

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly document record."""

        return {
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "source_url": self.source_url,
            "version": self.version,
            "domain": self.domain,
            "document_type": self.document_type,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> KnowledgeDocument:
        """Build a document from a saved JSON record."""

        created_at = datetime.fromisoformat(str(record["created_at"]))
        metadata = _metadata_from_object(record.get("metadata", {}))
        return cls(
            document_id=str(record["document_id"]),
            title=str(record["title"]),
            source=str(record["source"]),
            source_url=None
            if record.get("source_url") in {None, ""}
            else str(record.get("source_url")),
            version=str(record["version"]),
            domain=str(record["domain"]),
            document_type=str(record["document_type"]),
            language=str(record["language"]),
            created_at=created_at,
            content=str(record["content"]),
            metadata=metadata,
        )


@dataclass(frozen=True)
class DocumentChunk:
    """Traceable chunk derived from a KnowledgeDocument."""

    chunk_id: str
    document_id: str
    text: str
    section: str
    source: str
    source_url: str | None
    domain: str
    version: str
    position: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if not self.text.strip():
            raise ValueError("chunk text must not be empty")
        if self.position < 0:
            raise ValueError("position must be non-negative")

    def citation(self, score: float) -> ChunkCitation:
        """Return citation metadata for a retrieval score."""

        return ChunkCitation(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            document=str(self.metadata.get("document_title") or self.document_id),
            section=self.section,
            source=self.source,
            source_url=self.source_url,
            domain=self.domain,
            version=self.version,
            position=self.position,
            score=score,
        )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly chunk record."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "section": self.section,
            "source": self.source,
            "source_url": self.source_url,
            "domain": self.domain,
            "version": self.version,
            "position": self.position,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> DocumentChunk:
        """Build a chunk from a saved JSON record."""

        return cls(
            chunk_id=str(record["chunk_id"]),
            document_id=str(record["document_id"]),
            text=str(record["text"]),
            section=str(record["section"]),
            source=str(record["source"]),
            source_url=None
            if record.get("source_url") in {None, ""}
            else str(record.get("source_url")),
            domain=str(record["domain"]),
            version=str(record["version"]),
            position=int(record["position"]),
            metadata=_metadata_from_object(record.get("metadata", {})),
        )


@dataclass(frozen=True)
class ChunkCitation:
    """Citation returned with every retrieval result."""

    chunk_id: str
    document_id: str
    document: str
    section: str
    source: str
    source_url: str | None
    domain: str
    version: str
    position: int
    score: float

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly citation record."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document": self.document,
            "section": self.section,
            "source": self.source,
            "source_url": self.source_url,
            "domain": self.domain,
            "version": self.version,
            "position": self.position,
            "score": self.score,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """Dense retrieval result with text and citation metadata."""

    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: Metadata
    citation: ChunkCitation

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly retrieval result."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
            "citation": self.citation.to_record(),
        }


@dataclass(frozen=True)
class IncidentRetrievalContext:
    """Structured incident context used to build a retrieval query."""

    component: str | None = None
    event_id: str | None = None
    event_template: str | None = None
    log_level: str | None = None
    candidate_metric: str | None = None
    incident_type: str | None = None
    root_cause_candidate: str | None = None
    severity: str | None = None
    source_dataset: str | None = None
    domain: str | None = None

    def to_query_terms(self) -> list[str]:
        """Return non-empty factual terms without adding invented context."""

        values = [
            self.component,
            self.event_id,
            self.event_template,
            self.log_level,
            self.candidate_metric,
            self.incident_type,
            self.root_cause_candidate,
            self.severity,
            self.source_dataset,
        ]
        return [value for value in values if value]


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for reproducible Phase 11 RAG scripts."""

    knowledge_dir: Path = Path("knowledge")
    artifact_root: Path = Path("experiments/rag")
    run_id: str = "phase11_rag_knowledge_20260922"
    eval_dataset_path: Path = Path("knowledge/retrieval_eval.jsonl")
    supported_extensions: tuple[str, ...] = (".md", ".txt", ".html", ".htm", ".pdf")
    default_language: str = "en"
    chunk_size_tokens: int = 160
    chunk_overlap_tokens: int = 30
    chunk_separators: tuple[str, ...] = ("\n\n", "\n", ". ")
    embedding_model_name: str = "local_hashing_embedding"
    embedding_model_version: str = "1.0"
    embedding_dimensions: int = 384
    vector_store_backend: str = "local_json"
    top_k: int = 5
    score_threshold: float = 0.0
    rerank_top_k: int = 5
    retrieval_configurations: tuple[str, ...] = ("dense", "dense_rerank")

    def __post_init__(self) -> None:
        if self.chunk_size_tokens < 20:
            raise ValueError("chunk_size_tokens must be at least 20")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be non-negative")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        if self.embedding_dimensions < 16:
            raise ValueError("embedding_dimensions must be at least 16")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.rerank_top_k < 1:
            raise ValueError("rerank_top_k must be positive")

    def run_dir(self, root: Path) -> Path:
        """Return the configured run directory."""

        return _resolve_path(root, self.artifact_root) / self.run_id


@dataclass(frozen=True)
class RetrievalEvaluationRecord:
    """Gold retrieval query used by Phase 11 evaluation."""

    query: str
    expected_document_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    domain: str
    query_id: str

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.expected_document_ids and not self.expected_chunk_ids:
            raise ValueError("at least one expected document or chunk id is required")
        if not self.domain:
            raise ValueError("domain must not be empty")

    @classmethod
    def from_record(cls, record: dict[str, Any], index: int) -> RetrievalEvaluationRecord:
        """Build a gold record from JSONL."""

        return cls(
            query=str(record["query"]),
            expected_document_ids=_tuple_strings(record.get("expected_document_ids", ())),
            expected_chunk_ids=_tuple_strings(record.get("expected_chunk_ids", ())),
            domain=str(record["domain"]),
            query_id=str(record.get("query_id") or f"q{index:04d}"),
        )


@dataclass(frozen=True)
class RetrievalEvaluationMetrics:
    """Aggregate metrics for one retrieval configuration."""

    configuration: str
    query_count: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_latency_ms: float
    mean_retrieved_documents: float
    citation_completeness: float

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly metrics record."""

        return {
            "configuration": self.configuration,
            "query_count": self.query_count,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "mean_latency_ms": self.mean_latency_ms,
            "mean_retrieved_documents": self.mean_retrieved_documents,
            "citation_completeness": self.citation_completeness,
        }


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""

    return datetime.now(tz=UTC)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records."""

    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record in {path} must be an object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _metadata_from_object(value: object) -> Metadata:
    if not isinstance(value, dict):
        return {}
    metadata: Metadata = {}
    for key, item in value.items():
        if isinstance(item, str | int | float | bool) or item is None:
            metadata[str(key)] = item
        else:
            metadata[str(key)] = json.dumps(item, sort_keys=True)
    return metadata


def _tuple_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if str(item))
    return ()


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
