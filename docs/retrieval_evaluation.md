# Retrieval Evaluation

Phase 11 evaluates retrieval without an LLM.

## Gold Dataset

Gold queries live in:

```text
knowledge/retrieval_eval.jsonl
```

Each record contains:

- `query`
- `expected_document_ids`
- `expected_chunk_ids`
- `domain`
- `query_id`

The measured run contains 18 queries across 8 domains.

## Metrics

Implemented metrics:

- Precision@K
- Recall@K
- MRR
- nDCG@K
- mean retrieval latency
- mean number of retrieved documents
- citation completeness

Document-level relevance is deduplicated, so retrieving multiple chunks from the same expected document cannot inflate nDCG beyond 1.

## Measured Results

Run:

```text
experiments/rag/phase11_rag_knowledge_20260922
```

| Configuration | Precision@K | Recall@K | MRR | nDCG@K | Mean latency ms | Citation completeness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.2000 | 1.0000 | 0.9722 | 0.9795 | 1.9606 | 1.0000 |
| `dense_rerank` | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 2.1500 | 1.0000 |

The best measured configuration is `dense_rerank`.

## Failure Analysis

No retrieval failures were observed in the current small benchmark after correcting HTML document ID metadata.

Tracked failure modes remain:

- correct document not retrieved;
- correct section not retrieved;
- irrelevant chunks ranked too highly;
- terminology mismatch;
- ambiguous query;
- cross-domain confusion.

The absence of observed failures should not be overinterpreted. The corpus is small, domain filtering is strong, and most domains currently contain one document.
