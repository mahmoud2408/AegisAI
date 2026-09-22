from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegis_ai.rag import (
    ChunkingConfig,
    DenseRetriever,
    HashingEmbeddingModel,
    IncidentRetrievalContext,
    LocalVectorStore,
    RetrievalConfig,
    StructureAwareChunker,
    create_document_loader,
)
from aegis_ai.rag.evaluation import citation_completeness, evaluate_single_query
from aegis_ai.rag.models import KnowledgeDocument, RetrievalEvaluationRecord, utc_now
from aegis_ai.rag.vector_store import build_vector_records


def test_markdown_loader_preserves_front_matter_metadata(tmp_path: Path) -> None:
    doc_path = tmp_path / "knowledge" / "openstack" / "nova.md"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(
        "\n".join(
            [
                "---",
                "document_id: openstack_test_doc",
                "title: Nova Test",
                "source: project-generated",
                "version: '1'",
                "domain: cloud_infrastructure",
                "source_url: https://example.invalid/nova",
                "---",
                "# Nova Test",
                "",
                "nova-compute lifecycle triage content",
            ]
        ),
        encoding="utf-8",
    )

    loader = create_document_loader(doc_path, knowledge_root=tmp_path / "knowledge")
    document = loader.load()

    assert document.document_id == "openstack_test_doc"
    assert document.domain == "cloud_infrastructure"
    assert document.source_url == "https://example.invalid/nova"
    assert document.metadata["path"] == "openstack/nova.md"
    assert "nova-compute" in document.content


def test_html_loader_sanitizes_script_text_and_reads_metadata(tmp_path: Path) -> None:
    doc_path = tmp_path / "knowledge" / "monitoring" / "alert.html"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(
        """
        <html><head>
        <meta name="document_id" content="html_alert_doc">
        <meta name="source" content="project-generated">
        <meta name="domain" content="monitoring">
        <meta name="version" content="1">
        <title>Alert Doc</title>
        </head><body>
        <script>secretToken = "do-not-ingest";</script>
        <h1>Alert Doc</h1><p>Prometheus alert labels and scrape health.</p>
        </body></html>
        """,
        encoding="utf-8",
    )

    document = create_document_loader(doc_path, knowledge_root=tmp_path / "knowledge").load()

    assert document.document_id == "html_alert_doc"
    assert document.title == "Alert Doc"
    assert "Prometheus alert" in document.content
    assert "secretToken" not in document.content


def test_chunker_preserves_section_and_provenance() -> None:
    document = _document(
        content="# First\n\nalpha beta gamma\n\n# Second\n\nrestart count exit code logs"
    )

    chunks = StructureAwareChunker(
        ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=4)
    ).chunk_document(document)

    assert {chunk.section for chunk in chunks} == {"First", "Second"}
    assert all(chunk.document_id == "doc1" for chunk in chunks)
    assert all(chunk.metadata["document_title"] == "Doc One" for chunk in chunks)
    assert chunks[0].chunk_id.startswith("doc1_chunk_")


def test_hash_embedding_shape_and_determinism() -> None:
    model = HashingEmbeddingModel(dimensions=64)

    first = model.embed_query("nova compute lifecycle")
    second = model.embed_query("nova compute lifecycle")
    other = model.embed_query("postgres connection pool")

    assert len(first) == 64
    assert first == second
    assert first != other


def test_vector_store_operations_retrieval_filtering_and_citations() -> None:
    docs = [
        _document(
            document_id="cloud_doc",
            domain="cloud_infrastructure",
            content="# Nova\n\nnova compute lifecycle request id",
        ),
        _document(
            document_id="db_doc",
            domain="databases",
            content="# Database\n\npostgres connection saturation lock waits",
        ),
    ]
    chunker = StructureAwareChunker(ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=4))
    chunks = [chunk for document in docs for chunk in chunker.chunk_document(document)]
    model = HashingEmbeddingModel(dimensions=64)
    store = LocalVectorStore(
        build_vector_records(chunks, model.embed_documents([c.text for c in chunks]))
    )

    retriever = DenseRetriever(embedding_model=model, vector_store=store)
    results = retriever.retrieve(
        "postgres lock waits",
        config=RetrievalConfig(top_k=3),
        domain="databases",
    )

    assert store.count() == len(chunks)
    assert store.health_check()["status"] == "ok"
    assert results
    assert all(result.citation.domain == "databases" for result in results)
    assert results[0].citation.chunk_id == results[0].chunk_id
    assert citation_completeness(results) == pytest.approx(1.0)
    assert store.delete([results[0].chunk_id]) == 1


def test_incident_context_retrieval_infers_domain_without_inventing_terms() -> None:
    document = _document(
        document_id="industrial_doc",
        domain="industrial",
        content="# Pressure\n\nTP2 TP3 pressure path forecast deviation compressor",
    )
    chunker = StructureAwareChunker(ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=4))
    chunks = chunker.chunk_document(document)
    model = HashingEmbeddingModel(dimensions=64)
    store = LocalVectorStore(
        build_vector_records(chunks, model.embed_documents([c.text for c in chunks]))
    )
    retriever = DenseRetriever(embedding_model=model, vector_store=store)

    results = retriever.retrieve_for_incident(
        IncidentRetrievalContext(candidate_metric="TP2", source_dataset="metropt"),
        config=RetrievalConfig(top_k=2),
    )

    assert results
    assert results[0].citation.domain == "industrial"


def test_retrieval_metrics_deduplicate_document_hits() -> None:
    record = RetrievalEvaluationRecord(
        query="postgres locks",
        expected_document_ids=("db_doc",),
        expected_chunk_ids=(),
        domain="databases",
        query_id="q1",
    )
    document = _document(document_id="db_doc", domain="databases", content="# DB\n\npostgres locks")
    chunks = StructureAwareChunker(
        ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=4)
    ).chunk_document(document)
    model = HashingEmbeddingModel(dimensions=32)
    store = LocalVectorStore(
        build_vector_records(chunks, model.embed_documents([c.text for c in chunks]))
    )
    retriever = DenseRetriever(embedding_model=model, vector_store=store)
    results = retriever.retrieve(
        "postgres locks", config=RetrievalConfig(top_k=5), domain="databases"
    )
    duplicated_results = results + results

    metrics = evaluate_single_query(record, duplicated_results, top_k=5)

    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["ndcg_at_k"] <= 1.0


def test_local_vector_store_round_trip(tmp_path: Path) -> None:
    document = _document(content="# Round Trip\n\ncontainer restart loop")
    chunk = StructureAwareChunker(
        ChunkingConfig(chunk_size_tokens=20, chunk_overlap_tokens=4)
    ).chunk_document(document)[0]
    model = HashingEmbeddingModel(dimensions=32)
    store = LocalVectorStore(build_vector_records([chunk], model.embed_documents([chunk.text])))
    path = tmp_path / "index.json"

    store.save(path, metadata={"embedding_model": model.model_name})
    loaded = LocalVectorStore.load(path)

    assert loaded.count() == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["embedding_model"] == model.model_name


def _document(
    *,
    document_id: str = "doc1",
    domain: str = "containers",
    content: str,
) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=document_id,
        title="Doc One",
        source="project-generated",
        source_url=None,
        version="1",
        domain=domain,
        document_type="markdown",
        language="en",
        created_at=utc_now(),
        content=content,
        metadata={"document_title": "Doc One"},
    )
