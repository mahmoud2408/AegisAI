# RAG

Phase status: design proposal only.

## Pipeline

The RAG subsystem will:

1. Load documents.
2. Extract text.
3. Normalize text.
4. Split text into chunks.
5. Generate embeddings.
6. Store embeddings and metadata.
7. Retrieve relevant chunks.
8. Optionally rerank results.
9. Generate grounded answers with citations.

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

Planned metrics:

- Recall@k
- MRR
- Citation precision
- Citation coverage
- Grounded answer rate

Generated answers must reference retrieved chunks. If retrieval fails, the system should say that it lacks supporting documentation instead of inventing a citation.
