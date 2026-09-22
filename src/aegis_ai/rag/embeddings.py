"""Local embedding abstractions for Phase 11 retrieval."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from aegis_ai.rag.chunking import tokenize

Vector = tuple[float, ...]


class EmbeddingModel(ABC):
    """Embedding interface used by the retriever and vector store."""

    model_name: str
    model_version: str
    dimensions: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of documents/chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """Embed a query."""


@dataclass(frozen=True)
class HashingEmbeddingModel(EmbeddingModel):
    """Deterministic local hashing embedding.

    This is a lightweight baseline, not a semantic transformer. It is useful for
    reproducible local tests and gives the vector-store/retrieval architecture a
    real embedding path before adding optional sentence-transformer or Qdrant
    deployments.
    """

    model_name: str = "local_hashing_embedding"
    model_version: str = "1.0"
    dimensions: int = 384

    def __post_init__(self) -> None:
        if self.dimensions < 16:
            raise ValueError("dimensions must be at least 16")

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of texts."""

        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        """Embed one query."""

        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        terms = expand_terms(tokenize(text))
        for term in terms:
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return normalize_vector(tuple(vector))


def expand_terms(tokens: list[str]) -> list[str]:
    """Return unigrams plus adjacent bigrams for modest phrase sensitivity."""

    terms = list(tokens)
    terms.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False))
    return terms


def normalize_vector(vector: Vector) -> Vector:
    """L2-normalize a vector."""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return tuple(value / norm for value in vector)


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Cosine similarity for normalized or unnormalized vectors."""

    if len(left) != len(right):
        raise ValueError("vectors must have matching dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
