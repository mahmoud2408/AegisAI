"""Document ingestion, embeddings, retrieval, citations, and RAG evaluation."""

from aegis_ai.rag.chunking import ChunkingConfig, StructureAwareChunker
from aegis_ai.rag.embeddings import EmbeddingModel, HashingEmbeddingModel
from aegis_ai.rag.evaluation import evaluate_retriever, load_evaluation_records
from aegis_ai.rag.loaders import BaseDocumentLoader, create_document_loader, load_documents
from aegis_ai.rag.models import (
    ChunkCitation,
    DocumentChunk,
    IncidentRetrievalContext,
    KnowledgeDocument,
    RAGConfig,
    RetrievalEvaluationMetrics,
    RetrievalEvaluationRecord,
    RetrievalResult,
)
from aegis_ai.rag.pipeline import (
    build_index,
    evaluate_knowledge_retrieval,
    ingest_knowledge,
    load_rag_config,
)
from aegis_ai.rag.retrieval import DenseRetriever, LexicalReranker, RetrievalConfig
from aegis_ai.rag.vector_store import LocalVectorStore, VectorRecord, VectorStore

__all__ = [
    "BaseDocumentLoader",
    "ChunkCitation",
    "ChunkingConfig",
    "DenseRetriever",
    "DocumentChunk",
    "EmbeddingModel",
    "HashingEmbeddingModel",
    "IncidentRetrievalContext",
    "KnowledgeDocument",
    "LexicalReranker",
    "LocalVectorStore",
    "RAGConfig",
    "RetrievalConfig",
    "RetrievalEvaluationMetrics",
    "RetrievalEvaluationRecord",
    "RetrievalResult",
    "StructureAwareChunker",
    "VectorRecord",
    "VectorStore",
    "build_index",
    "create_document_loader",
    "evaluate_knowledge_retrieval",
    "evaluate_retriever",
    "ingest_knowledge",
    "load_documents",
    "load_evaluation_records",
    "load_rag_config",
]
