"""Retrieval evaluation metrics and failure analysis."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from aegis_ai.rag.models import (
    RetrievalEvaluationMetrics,
    RetrievalEvaluationRecord,
    RetrievalResult,
    read_jsonl,
)
from aegis_ai.rag.retrieval import DenseRetriever, RetrievalConfig


def load_evaluation_records(path: Path) -> list[RetrievalEvaluationRecord]:
    """Load retrieval evaluation records from JSONL."""

    return [
        RetrievalEvaluationRecord.from_record(record, index)
        for index, record in enumerate(read_jsonl(path))
    ]


def evaluate_retriever(
    records: list[RetrievalEvaluationRecord],
    retriever: DenseRetriever,
    *,
    configuration: str,
    retrieval_config: RetrievalConfig,
) -> tuple[RetrievalEvaluationMetrics, list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate a retriever configuration."""

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    latency_values: list[float] = []
    retrieved_document_counts: list[float] = []
    citation_values: list[float] = []

    for record in records:
        started = time.perf_counter()
        results = retriever.retrieve(
            record.query,
            config=retrieval_config,
            domain=record.domain,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        metrics = evaluate_single_query(record, results, retrieval_config.top_k)
        precision_values.append(metrics["precision_at_k"])
        recall_values.append(metrics["recall_at_k"])
        reciprocal_ranks.append(metrics["reciprocal_rank"])
        ndcg_values.append(metrics["ndcg_at_k"])
        latency_values.append(latency_ms)
        retrieved_document_counts.append(float(len({result.document_id for result in results})))
        citation_values.append(citation_completeness(results))
        row = {
            "configuration": configuration,
            "query_id": record.query_id,
            "query": record.query,
            "domain": record.domain,
            "latency_ms": latency_ms,
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "reciprocal_rank": metrics["reciprocal_rank"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "retrieved_chunk_ids": [result.chunk_id for result in results],
            "retrieved_document_ids": [result.document_id for result in results],
            "citations": [result.citation.to_record() for result in results],
        }
        rows.append(row)
        if metrics["recall_at_k"] < 1.0:
            failures.append(
                {
                    "configuration": configuration,
                    "query_id": record.query_id,
                    "query": record.query,
                    "domain": record.domain,
                    "failure_type": classify_failure(record, results),
                    "expected_document_ids": list(record.expected_document_ids),
                    "expected_chunk_ids": list(record.expected_chunk_ids),
                    "retrieved_document_ids": [result.document_id for result in results],
                    "retrieved_chunk_ids": [result.chunk_id for result in results],
                }
            )

    aggregate = RetrievalEvaluationMetrics(
        configuration=configuration,
        query_count=len(records),
        precision_at_k=mean(precision_values),
        recall_at_k=mean(recall_values),
        mrr=mean(reciprocal_ranks),
        ndcg_at_k=mean(ndcg_values),
        mean_latency_ms=mean(latency_values),
        mean_retrieved_documents=mean(retrieved_document_counts),
        citation_completeness=mean(citation_values),
    )
    return aggregate, rows, failures


def evaluate_single_query(
    record: RetrievalEvaluationRecord,
    results: list[RetrievalResult],
    top_k: int,
) -> dict[str, float]:
    """Evaluate one query."""

    expected_chunks = set(record.expected_chunk_ids)
    expected_docs = set(record.expected_document_ids)
    if expected_chunks:
        expected_units = expected_chunks
        retrieved_units = [result.chunk_id for result in results[:top_k]]
    else:
        expected_units = expected_docs
        retrieved_units = [result.document_id for result in results[:top_k]]
    seen_relevant: set[str] = set()
    relevant: list[float] = []
    for unit in retrieved_units:
        if unit in expected_units and unit not in seen_relevant:
            relevant.append(1.0)
            seen_relevant.add(unit)
        else:
            relevant.append(0.0)
    hit_count = sum(relevant)
    precision = hit_count / max(top_k, 1)
    recall = hit_count / max(len(expected_units), 1)
    reciprocal_rank = 0.0
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            reciprocal_rank = 1.0 / rank
            break
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevant))
    ideal_relevant = [1.0] * min(len(expected_units), top_k)
    idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal_relevant))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "precision_at_k": precision,
        "recall_at_k": min(recall, 1.0),
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": ndcg,
    }


def citation_completeness(results: list[RetrievalResult]) -> float:
    """Measure required citation metadata completeness."""

    if not results:
        return 0.0
    complete = 0
    for result in results:
        citation = result.citation
        if (
            citation.chunk_id
            and citation.document_id
            and citation.document
            and citation.section
            and citation.source
            and citation.domain
            and math.isfinite(citation.score)
        ):
            complete += 1
    return complete / len(results)


def classify_failure(
    record: RetrievalEvaluationRecord,
    results: list[RetrievalResult],
) -> str:
    """Classify a representative retrieval failure."""

    if not results:
        return "no_results"
    retrieved_domains = {result.citation.domain for result in results}
    if retrieved_domains and retrieved_domains != {record.domain}:
        return "cross_domain_confusion"
    expected_docs = set(record.expected_document_ids)
    retrieved_docs = {result.document_id for result in results}
    if expected_docs.isdisjoint(retrieved_docs):
        return "correct_document_not_retrieved"
    if record.expected_chunk_ids:
        retrieved_chunks = {result.chunk_id for result in results}
        if set(record.expected_chunk_ids).isdisjoint(retrieved_chunks):
            return "correct_section_not_retrieved"
    return "relevant_item_ranked_too_low"


def mean(values: list[float]) -> float:
    """Return arithmetic mean or zero."""

    return sum(values) / len(values) if values else 0.0
