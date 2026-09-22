"""Reproducible Phase 11 RAG ingestion, indexing, and evaluation pipeline."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from aegis_ai.data.dataset_registry import project_root
from aegis_ai.rag.chunking import ChunkingConfig, StructureAwareChunker
from aegis_ai.rag.embeddings import HashingEmbeddingModel
from aegis_ai.rag.evaluation import evaluate_retriever, load_evaluation_records
from aegis_ai.rag.loaders import load_documents
from aegis_ai.rag.models import (
    DocumentChunk,
    KnowledgeDocument,
    RAGConfig,
    RetrievalEvaluationMetrics,
    read_jsonl,
    write_jsonl,
)
from aegis_ai.rag.retrieval import DenseRetriever, LexicalReranker, RetrievalConfig
from aegis_ai.rag.vector_store import LocalVectorStore, build_vector_records


def load_rag_config(path: Path) -> RAGConfig:
    """Load RAG YAML config."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("RAG config must be a YAML mapping")
    default = RAGConfig()
    return RAGConfig(
        knowledge_dir=Path(str(payload.get("knowledge_dir", default.knowledge_dir))),
        artifact_root=Path(str(payload.get("artifact_root", default.artifact_root))),
        run_id=str(payload.get("run_id", default.run_id)),
        eval_dataset_path=Path(str(payload.get("eval_dataset_path", default.eval_dataset_path))),
        supported_extensions=_tuple_strings(
            payload.get("supported_extensions"), default.supported_extensions
        ),
        default_language=str(payload.get("default_language", default.default_language)),
        chunk_size_tokens=int(payload.get("chunk_size_tokens", default.chunk_size_tokens)),
        chunk_overlap_tokens=int(payload.get("chunk_overlap_tokens", default.chunk_overlap_tokens)),
        chunk_separators=_tuple_strings(payload.get("chunk_separators"), default.chunk_separators),
        embedding_model_name=str(payload.get("embedding_model_name", default.embedding_model_name)),
        embedding_model_version=str(
            payload.get("embedding_model_version", default.embedding_model_version)
        ),
        embedding_dimensions=int(payload.get("embedding_dimensions", default.embedding_dimensions)),
        vector_store_backend=str(payload.get("vector_store_backend", default.vector_store_backend)),
        top_k=int(payload.get("top_k", default.top_k)),
        score_threshold=float(payload.get("score_threshold", default.score_threshold)),
        rerank_top_k=int(payload.get("rerank_top_k", default.rerank_top_k)),
        retrieval_configurations=_tuple_strings(
            payload.get("retrieval_configurations"), default.retrieval_configurations
        ),
    )


def ingest_knowledge(
    *,
    root: Path | None = None,
    config: RAGConfig | None = None,
) -> tuple[list[KnowledgeDocument], list[DocumentChunk], dict[str, Any]]:
    """Load documents, create chunks, and save ingestion artifacts."""

    project = project_root() if root is None else root
    cfg = config or RAGConfig()
    run_dir = cfg.run_dir(project)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    knowledge_dir = _resolve(project, cfg.knowledge_dir)
    documents = load_documents(
        knowledge_dir,
        supported_extensions=cfg.supported_extensions,
        default_language=cfg.default_language,
    )
    chunker = StructureAwareChunker(
        ChunkingConfig(
            chunk_size_tokens=cfg.chunk_size_tokens,
            chunk_overlap_tokens=cfg.chunk_overlap_tokens,
            separators=cfg.chunk_separators,
        )
    )
    chunks = chunker.chunk_documents(documents)
    write_jsonl(run_dir / "documents.jsonl", [document.to_record() for document in documents])
    write_jsonl(run_dir / "chunks.jsonl", [chunk.to_record() for chunk in chunks])
    stats = ingestion_statistics(documents, chunks)
    stats.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "knowledge_dir": str(knowledge_dir),
            "run_id": cfg.run_id,
        }
    )
    save_json(run_dir / "ingestion_statistics.json", stats)
    save_json(run_dir / "chunk_statistics.json", chunk_statistics(chunks))
    save_json(run_dir / "config.json", config_to_record(cfg))
    return documents, chunks, stats


def build_index(
    *,
    root: Path | None = None,
    config: RAGConfig | None = None,
) -> dict[str, Any]:
    """Build and persist the local vector index."""

    project = project_root() if root is None else root
    cfg = config or RAGConfig()
    run_dir = cfg.run_dir(project)
    chunks = load_chunks(run_dir / "chunks.jsonl")
    if not chunks:
        _documents, chunks, _stats = ingest_knowledge(root=project, config=cfg)
    started = time.perf_counter()
    embedding_model = HashingEmbeddingModel(
        model_name=cfg.embedding_model_name,
        model_version=cfg.embedding_model_version,
        dimensions=cfg.embedding_dimensions,
    )
    vectors = embedding_model.embed_documents([chunk.text for chunk in chunks])
    records = build_vector_records(chunks, vectors)
    store = LocalVectorStore(records)
    index_path = run_dir / "index" / "vector_store.json"
    metadata = {
        "embedding_model": embedding_model.model_name,
        "embedding_model_version": embedding_model.model_version,
        "embedding_dimensions": embedding_model.dimensions,
        "vector_store_backend": cfg.vector_store_backend,
        "chunk_count": len(chunks),
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    store.save(index_path, metadata=metadata)
    metadata.update(
        {
            "index_path": str(index_path),
            "elapsed_seconds": time.perf_counter() - started,
            "health": store.health_check(),
        }
    )
    save_json(run_dir / "embedding_metadata.json", metadata)
    return metadata


def evaluate_knowledge_retrieval(
    *,
    root: Path | None = None,
    config: RAGConfig | None = None,
) -> dict[str, Any]:
    """Evaluate configured retrieval variants and save artifacts."""

    project = project_root() if root is None else root
    cfg = config or RAGConfig()
    run_dir = cfg.run_dir(project)
    index_path = run_dir / "index" / "vector_store.json"
    if not index_path.exists():
        build_index(root=project, config=cfg)
    store = LocalVectorStore.load(index_path)
    embedding_model = HashingEmbeddingModel(
        model_name=cfg.embedding_model_name,
        model_version=cfg.embedding_model_version,
        dimensions=cfg.embedding_dimensions,
    )
    eval_path = _resolve(project, cfg.eval_dataset_path)
    records = load_evaluation_records(eval_path)
    all_metrics: list[RetrievalEvaluationMetrics] = []
    all_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for configuration in cfg.retrieval_configurations:
        rerank = configuration == "dense_rerank"
        retriever = DenseRetriever(
            embedding_model=embedding_model,
            vector_store=store,
            reranker=LexicalReranker() if rerank else None,
        )
        retrieval_config = RetrievalConfig(
            top_k=cfg.top_k,
            score_threshold=cfg.score_threshold,
            rerank_top_k=cfg.rerank_top_k,
            rerank=rerank,
        )
        metrics, rows, failures = evaluate_retriever(
            records,
            retriever,
            configuration=configuration,
            retrieval_config=retrieval_config,
        )
        all_metrics.append(metrics)
        all_rows.extend(rows)
        all_failures.extend(failures)
    metrics_records = [metrics.to_record() for metrics in all_metrics]
    write_jsonl(run_dir / "retrieval_results.jsonl", all_rows)
    save_json(run_dir / "evaluation_results.json", metrics_records)
    save_json(run_dir / "failure_cases.json", all_failures)
    save_json(run_dir / "failure_analysis.json", failure_analysis(all_failures))
    save_json(
        run_dir / "latency_results.json",
        {row["configuration"] + ":" + row["query_id"]: row["latency_ms"] for row in all_rows},
    )
    summary = {
        "run_id": cfg.run_id,
        "query_count": len(records),
        "configurations": metrics_records,
        "failure_case_count": len(all_failures),
        "best_configuration": choose_best_configuration(metrics_records),
        "citation_completeness": {
            record["configuration"]: record["citation_completeness"] for record in metrics_records
        },
    }
    save_json(run_dir / "metrics.json", summary)
    write_run_report(run_dir, summary)
    return summary


def load_documents_artifact(path: Path) -> list[KnowledgeDocument]:
    """Load normalized document artifacts."""

    return [KnowledgeDocument.from_record(record) for record in read_jsonl(path)]


def load_chunks(path: Path) -> list[DocumentChunk]:
    """Load chunk artifacts."""

    return [DocumentChunk.from_record(record) for record in read_jsonl(path)]


def ingestion_statistics(
    documents: list[KnowledgeDocument],
    chunks: list[DocumentChunk],
) -> dict[str, Any]:
    """Build ingestion statistics."""

    docs_by_domain: dict[str, int] = {}
    docs_by_source: dict[str, int] = {}
    for document in documents:
        docs_by_domain[document.domain] = docs_by_domain.get(document.domain, 0) + 1
        docs_by_source[document.source] = docs_by_source.get(document.source, 0) + 1
    chunks_by_domain: dict[str, int] = {}
    for chunk in chunks:
        chunks_by_domain[chunk.domain] = chunks_by_domain.get(chunk.domain, 0) + 1
    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents_by_domain": docs_by_domain,
        "documents_by_source": docs_by_source,
        "chunks_by_domain": chunks_by_domain,
    }


def chunk_statistics(chunks: list[DocumentChunk]) -> dict[str, Any]:
    """Return simple chunk statistics."""

    lengths = [len(chunk.text.split()) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "mean_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }


def choose_best_configuration(metrics_records: list[dict[str, Any]]) -> str | None:
    """Select best configuration by recall, MRR, then latency."""

    if not metrics_records:
        return None
    ordered = sorted(
        metrics_records,
        key=lambda record: (
            float(record["recall_at_k"]),
            float(record["mrr"]),
            -float(record["mean_latency_ms"]),
        ),
        reverse=True,
    )
    return str(ordered[0]["configuration"])


def failure_analysis(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observed and tracked retrieval failure modes."""

    counts: dict[str, int] = {}
    for failure in failures:
        failure_type = str(failure.get("failure_type", "unknown"))
        counts[failure_type] = counts.get(failure_type, 0) + 1
    return {
        "observed_failure_count": len(failures),
        "observed_failure_types": counts,
        "tracked_failure_modes": [
            "correct_document_not_retrieved",
            "correct_section_not_retrieved",
            "irrelevant_chunks_ranked_too_highly",
            "terminology_mismatch",
            "ambiguous_query",
            "cross_domain_confusion",
        ],
        "interpretation": (
            "No observed failures means none were found in this small benchmark; it is not "
            "evidence that retrieval will generalize to larger or official knowledge packs."
        )
        if not failures
        else "Observed failures should be reviewed before promoting the retrieval configuration.",
    }


def write_run_report(run_dir: Path, summary: dict[str, Any]) -> None:
    """Write a compact Markdown run report."""

    lines = [
        "# Phase 11 RAG Retrieval Run",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Query count: {summary['query_count']}",
        f"- Failure cases: {summary['failure_case_count']}",
        f"- Best configuration: `{summary['best_configuration']}`",
        "",
        "## Configurations",
        "",
    ]
    for record in summary["configurations"]:
        lines.append(
            "- "
            f"{record['configuration']}: P@K={record['precision_at_k']:.4f}, "
            f"R@K={record['recall_at_k']:.4f}, MRR={record['mrr']:.4f}, "
            f"nDCG={record['ndcg_at_k']:.4f}, "
            f"latency_ms={record['mean_latency_ms']:.4f}"
        )
    lines.append("")
    run_dir.joinpath("run_report.md").write_text("\n".join(lines), encoding="utf-8")


def config_to_record(config: RAGConfig) -> dict[str, Any]:
    """Serialize config."""

    return {
        "knowledge_dir": str(config.knowledge_dir),
        "artifact_root": str(config.artifact_root),
        "run_id": config.run_id,
        "eval_dataset_path": str(config.eval_dataset_path),
        "supported_extensions": list(config.supported_extensions),
        "default_language": config.default_language,
        "chunk_size_tokens": config.chunk_size_tokens,
        "chunk_overlap_tokens": config.chunk_overlap_tokens,
        "chunk_separators": list(config.chunk_separators),
        "embedding_model_name": config.embedding_model_name,
        "embedding_model_version": config.embedding_model_version,
        "embedding_dimensions": config.embedding_dimensions,
        "vector_store_backend": config.vector_store_backend,
        "top_k": config.top_k,
        "score_threshold": config.score_threshold,
        "rerank_top_k": config.rerank_top_k,
        "retrieval_configurations": list(config.retrieval_configurations),
    }


def save_json(path: Path, payload: Any) -> None:
    """Save JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _tuple_strings(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return default


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
