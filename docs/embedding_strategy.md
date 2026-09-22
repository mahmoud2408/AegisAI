# Embedding Strategy

Phase 11 uses a local deterministic embedding baseline.

## Current Model

| Field | Value |
| --- | --- |
| Model | `local_hashing_embedding` |
| Version | `1.0` |
| Dimensions | 384 |
| External API calls | none |

The model tokenizes text, adds adjacent bigrams, hashes terms into a fixed-dimensional vector, and L2-normalizes the result.

## Why This Model

This baseline was selected because it is:

- local;
- deterministic;
- fast;
- dependency-light;
- safe for private documents;
- enough to validate the ingestion, indexing, retrieval, filtering, citation, and evaluation contracts.

It is not a semantic transformer. It should be replaced or compared against a locally runnable sentence-transformer once the optional RAG dependencies and model cache are explicitly configured.

## Versioning

The vector index records:

- embedding model name;
- model version;
- dimensions;
- chunking configuration;
- run ID;
- creation timestamp.

This makes retrieval runs reproducible.
