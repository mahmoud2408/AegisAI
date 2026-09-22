"""Citation-aware retrieval and optional reranking."""

from __future__ import annotations

from dataclasses import dataclass

from aegis_ai.rag.chunking import tokenize
from aegis_ai.rag.embeddings import EmbeddingModel
from aegis_ai.rag.models import IncidentRetrievalContext, RetrievalResult
from aegis_ai.rag.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievalConfig:
    """Runtime retrieval settings."""

    top_k: int = 5
    score_threshold: float = 0.0
    rerank_top_k: int = 5
    rerank: bool = False

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.rerank_top_k < 1:
            raise ValueError("rerank_top_k must be positive")


class DenseRetriever:
    """Dense retriever backed by an embedding model and vector store."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        reranker: LexicalReranker | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        config: RetrievalConfig,
        domain: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve chunks for a user query."""

        query_vector = self.embedding_model.embed_query(query)
        candidate_k = max(config.top_k, config.rerank_top_k) if config.rerank else config.top_k
        results = self.vector_store.search(
            query_vector,
            top_k=candidate_k,
            score_threshold=config.score_threshold,
            domain=domain,
        )
        if config.rerank and self.reranker is not None:
            return self.reranker.rerank(query, results, top_k=config.top_k)
        return results[: config.top_k]

    def retrieve_for_incident(
        self,
        context: IncidentRetrievalContext,
        *,
        config: RetrievalConfig,
    ) -> list[RetrievalResult]:
        """Retrieve chunks from structured incident context."""

        terms = context.to_query_terms()
        query = " ".join(terms)
        if not query:
            return []
        domain = context.domain or infer_domain(context)
        return self.retrieve(query, config=config, domain=domain)


class LexicalReranker:
    """Lightweight optional reranker using token overlap."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Sort dense candidates by lexical overlap plus dense score."""

        query_tokens = set(tokenize(query))

        def score(result: RetrievalResult) -> tuple[float, float]:
            text_tokens = set(tokenize(result.text))
            if not query_tokens or not text_tokens:
                lexical = 0.0
            else:
                lexical = len(query_tokens.intersection(text_tokens)) / len(query_tokens)
            return lexical, result.score

        return sorted(results, key=score, reverse=True)[:top_k]


def infer_domain(context: IncidentRetrievalContext) -> str | None:
    """Conservatively infer a retrieval domain from observed context only."""

    source = (context.source_dataset or "").lower()
    terms = " ".join(context.to_query_terms()).lower()
    if "openstack" in source or "nova" in terms:
        return "cloud_infrastructure"
    if "metropt" in source or any(metric in terms for metric in {"tp2", "tp3", "motor_current"}):
        return "industrial"
    if "docker" in terms or "container" in terms:
        return "containers"
    if "kubernetes" in terms or "pod" in terms:
        return "containers"
    if "postgres" in terms or "database" in terms:
        return "databases"
    if "packet" in terms or "latency" in terms:
        return "networking"
    if "prometheus" in terms or "alert" in terms:
        return "monitoring"
    return None
