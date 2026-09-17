# Incident Engine

Phase 9 introduces the first deterministic incident decision layer for AegisAI. It consumes Phase 8B evidence and risk artifacts, groups related risk buckets into incidents, and attaches ranked root-cause candidates.

This is not the final production incident engine. It is the deterministic foundation that future API, dashboard, RCA, RAG, and agent layers can call.

## Responsibilities

Implemented:

- incident-window detection;
- deduplication and merging;
- evidence grouping by entity, metric, evidence type, and timestamp;
- severity propagation from Phase 8B risk severity;
- incident timeline generation;
- metric/component/entity candidate ranking;
- evidence lineage for every candidate;
- deterministic summaries;
- case studies and deterministic quality metrics.

Not implemented:

- LLM investigation;
- RAG;
- AI agent;
- recommendations;
- FastAPI;
- React dashboard;
- database persistence;
- service dependency inference;
- causal inference.

## Severity

Incident severity reuses Phase 8B severity values:

```text
INFO < LOW < MEDIUM < HIGH < CRITICAL
```

The incident window severity is the maximum severity among risk buckets in the window. Phase 9 does not define a second incompatible severity system.

## Incident Timeline

Each report includes a timeline row per bucket:

- timestamp;
- risk score;
- severity;
- evidence count;
- evidence types;
- metric names.

This timeline is saved to:

```text
experiments/rca/phase9_rca_20260917/incident_timeline.csv
```

## Lineage

Candidate lineage is saved to:

```text
experiments/rca/phase9_rca_20260917/evidence_mappings.csv
```

Each mapping row includes:

- `incident_id`
- `candidate_id`
- `signal_id`
- `timestamp`
- `metric_name`
- `signal_type`
- `source_dataset`
- `source_model`

This makes every RCA candidate reviewable back to the original evidence record.

## Scientific Limitation

The incident engine performs:

```text
evidence-based incident grouping and candidate ranking
```

It does not perform:

```text
causal inference or guaranteed root-cause identification
```

Future phases may add logs, traces, service dependencies, RAG, and agentic investigation, but those layers must preserve this evidence/inference distinction.

## Next Phase

The next recommended phase is deterministic remediation and report packaging, or a service/dependency model if real service topology is introduced. RAG and LLM agents should wait until the deterministic incident/RCA artifacts are stable enough to cite.
