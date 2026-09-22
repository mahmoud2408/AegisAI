"""Vector store abstraction and local persisted backend."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegis_ai.rag.embeddings import Vector, cosine_similarity
from aegis_ai.rag.models import DocumentChunk, RetrievalResult


@dataclass(frozen=True)
class VectorRecord:
    """Persisted vector and chunk pair."""

    chunk: DocumentChunk
    vector: Vector

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-friendly vector record."""

        return {"chunk": self.chunk.to_record(), "vector": list(self.vector)}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> VectorRecord:
        """Build from a saved JSON record."""

        vector_raw = record.get("vector", [])
        if not isinstance(vector_raw, list):
            raise ValueError("vector must be a list")
        return cls(
            chunk=DocumentChunk.from_record(record["chunk"]),
            vector=tuple(float(value) for value in vector_raw),
        )


class VectorStore(ABC):
    """Common vector store interface."""

    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records."""

    @abstractmethod
    def search(
        self,
        query_vector: Vector,
        *,
        top_k: int,
        score_threshold: float = 0.0,
        domain: str | None = None,
    ) -> list[RetrievalResult]:
        """Search vectors."""

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> int:
        """Delete records and return deletion count."""

    @abstractmethod
    def count(self) -> int:
        """Return record count."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Return backend health metadata."""


class LocalVectorStore(VectorStore):
    """In-memory vector store with optional JSON persistence."""

    def __init__(self, records: list[VectorRecord] | None = None) -> None:
        self._records: dict[str, VectorRecord] = {}
        if records:
            self.upsert(records)

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records by chunk id."""

        for record in records:
            self._records[record.chunk.chunk_id] = record

    def search(
        self,
        query_vector: Vector,
        *,
        top_k: int,
        score_threshold: float = 0.0,
        domain: str | None = None,
    ) -> list[RetrievalResult]:
        """Return highest-scoring chunks with optional domain filtering."""

        scored: list[tuple[float, VectorRecord]] = []
        for record in self._records.values():
            if domain and record.chunk.domain != domain:
                continue
            score = cosine_similarity(query_vector, record.vector)
            if score >= score_threshold:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for score, record in scored[:top_k]:
            citation = record.chunk.citation(score)
            results.append(
                RetrievalResult(
                    chunk_id=record.chunk.chunk_id,
                    document_id=record.chunk.document_id,
                    score=score,
                    text=record.chunk.text,
                    metadata=record.chunk.metadata,
                    citation=citation,
                )
            )
        return results

    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunk ids from the store."""

        deleted = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._records:
                del self._records[chunk_id]
                deleted += 1
        return deleted

    def count(self) -> int:
        """Return record count."""

        return len(self._records)

    def health_check(self) -> dict[str, Any]:
        """Return local backend health."""

        dimensions = 0
        if self._records:
            first = next(iter(self._records.values()))
            dimensions = len(first.vector)
        return {
            "backend": "local_json",
            "status": "ok",
            "count": self.count(),
            "dimensions": dimensions,
        }

    def save(self, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        """Persist vectors and chunk metadata to JSON."""

        payload = {
            "metadata": metadata or {},
            "records": [record.to_record() for record in self._records.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LocalVectorStore:
        """Load vectors from JSON."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        records_raw = payload.get("records", [])
        if not isinstance(records_raw, list):
            raise ValueError("vector store records must be a list")
        return cls([VectorRecord.from_record(record) for record in records_raw])


def build_vector_records(
    chunks: list[DocumentChunk],
    vectors: list[Vector],
) -> list[VectorRecord]:
    """Build vector records with one-to-one chunk/vector validation."""

    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    return [
        VectorRecord(chunk=chunk, vector=vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
