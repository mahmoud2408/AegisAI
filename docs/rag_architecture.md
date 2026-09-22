# RAG Architecture

Phase 11 implements the retrieval foundation for AegisAI. It does not generate answers, call an LLM, run an agent, or perform autonomous investigation.

## Pipeline

```text
Knowledge documents
  -> document loaders
  -> normalized KnowledgeDocument records
  -> structure-aware chunks
  -> local embeddings
  -> vector store
  -> retriever
  -> optional reranker
  -> citation-aware results
  -> retrieval evaluation
```

## Implemented Components

- `KnowledgeDocument` and `DocumentChunk` typed models with provenance.
- Markdown, plain text, HTML, and optional PDF loaders.
- Structure-aware chunking by section and paragraph.
- Local deterministic hashing embeddings.
- Local persisted JSON vector store behind a vector-store interface.
- Dense retrieval with domain filtering.
- Optional lexical reranking.
- Incident-context retrieval using factual fields only.
- Retrieval evaluation with Precision@K, Recall@K, MRR, nDCG, latency, document count, and citation completeness.

## Domain Isolation

Retrieval supports domain filtering. This is required because AegisAI datasets describe different systems. OpenStack documentation should not be retrieved for a MetroPT compressor incident merely because token overlap exists.

Current domains:

- `cloud_infrastructure`
- `operating_system`
- `containers`
- `databases`
- `networking`
- `monitoring`
- `incident_response`
- `industrial`

## Current Backend

The current backend is `local_json`. This is intentional for Phase 11 because it keeps tests and experiments reproducible without a running service. Qdrant remains the preferred production vector database once infrastructure is introduced.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\rag\ingest_knowledge.py
.\.venv\Scripts\python.exe scripts\rag\build_index.py
.\.venv\Scripts\python.exe scripts\rag\evaluate_retrieval.py
```

Artifacts are written to:

```text
experiments/rag/phase11_rag_knowledge_20260922
```

## Limitations

- The current corpus is small and mostly project-generated.
- The local hashing embedding is a deterministic baseline, not a semantic transformer.
- Retrieval quality is measured only on a small gold set.
- No answer generation or prompt-level hallucination mitigation is implemented in Phase 11.
