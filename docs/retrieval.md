# Retrieval

Phase 11 implements citation-aware retrieval only. It does not generate final answers.

## Modes

### User Query

Example:

```text
What should I check when nova-compute reports repeated VM lifecycle problems?
```

The retriever embeds the query, searches the vector store, applies optional domain filtering, and returns cited chunks.

### Incident Context Query

Structured incident fields can be passed through `IncidentRetrievalContext`:

- `component`
- `event_id`
- `event_template`
- `log_level`
- `candidate_metric`
- `incident_type`
- `root_cause_candidate`
- `severity`
- `source_dataset`
- `domain`

The retriever builds a query from non-empty factual fields only. It does not invent missing facts.

## Domain Filtering

Domain filtering is supported at search time. Conservative inference maps:

- OpenStack/Nova terms to `cloud_infrastructure`;
- MetroPT or `TP2`/`TP3`/`Motor_current` terms to `industrial`;
- Docker/Kubernetes/container terms to `containers`;
- database/PostgreSQL terms to `databases`;
- latency/packet terms to `networking`;
- Prometheus/alert terms to `monitoring`.

## Citations

Every result includes:

- `chunk_id`
- `document_id`
- document title
- section
- source
- source URL when available
- domain
- version
- chunk position
- score

The system never creates a citation without a retrieved chunk.

## Current Configurations

The benchmark compares:

- `dense`
- `dense_rerank`

`dense_rerank` applies a lightweight lexical reranker to dense candidates. No heavy reranker is introduced in Phase 11.
