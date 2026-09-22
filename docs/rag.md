# RAG

Phase status: Phase 11 retrieval foundation implemented.

## Pipeline

The RAG subsystem now:

1. Load documents.
2. Extract text.
3. Normalize text.
4. Split text into chunks.
5. Generate embeddings.
6. Store embeddings and metadata.
7. Retrieve relevant chunks.
8. Optionally rerank results.
9. Evaluates retrieval quality and citation completeness.

Generation is intentionally not implemented in Phase 11.

## Metadata Contract

Each chunk should store:

- Document name
- Source
- Section
- Chunk ID
- Timestamp
- Version
- Content hash
- Token count

## Retrieval Evaluation

Implemented metrics:

- Precision@K
- Recall@K
- MRR
- nDCG@K
- retrieval latency
- retrieved document count
- citation completeness

Measured run:

```text
experiments/rag/phase11_rag_knowledge_20260922
```

Best configuration: `dense_rerank`.

Detailed docs:

- `docs/rag_architecture.md`
- `docs/document_ingestion.md`
- `docs/chunking.md`
- `docs/embedding_strategy.md`
- `docs/retrieval.md`
- `docs/retrieval_evaluation.md`
- `docs/knowledge_packs.md`
