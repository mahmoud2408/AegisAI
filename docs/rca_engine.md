# RCA Engine

Phase 9 implements deterministic incident grouping and root-cause candidate ranking on top of Phase 8B evidence artifacts.

The pipeline is:

```text
EvidenceSignals
  -> Incident Window Detection
  -> Evidence Grouping
  -> Candidate Ranking
  -> Root Cause Candidate Report
```

No LLM, RAG, AI agent, API, dashboard, database backend, MLflow, Prometheus, or Grafana is implemented in this phase.

## Output Model

### IncidentCandidate

Each incident report contains:

- `incident_id`
- `start_time`
- `end_time`
- `severity`
- `risk_score`
- `affected_entities`
- `affected_services`
- `top_metric_signals`
- `root_cause_candidates`
- `supporting_evidence`
- `uncertainty`
- `data_sources`
- `timeline`
- `summary`

`affected_services` is currently empty for MetroPT because there is no service graph in the dataset.

### RootCauseCandidate

Each ranked candidate contains:

- `candidate_id`
- `incident_id`
- `entity`
- `metric_component`
- `candidate_type`
- `score`
- `rank`
- `supporting_signal_ids`
- `observed_evidence`
- `inference`
- `uncertainty`
- `limitations`
- `data_sources`
- `source_models`
- `first_evidence_time`
- `last_evidence_time`
- `persistence_minutes`
- `feature_scores`

Every candidate preserves `supporting_signal_ids`, source datasets, source models, and evidence timestamps through `evidence_mappings.csv`.

## Evidence vs Inference

The RCA output separates:

- observed evidence: original evidence descriptions and signal IDs;
- model inference: deterministic feature scores and candidate ranking;
- candidate explanation: why the candidate was ranked highly;
- uncertainty and limitations: why the candidate is not proof of causality.

Example wording:

```text
metric candidate 'TP2' is ranked from supporting signals...
This is evidence-based candidate ranking, not causal proof.
```

## Artifacts

Run command:

```bash
python scripts/experiments/run_rca.py --run-id phase9_rca_20260917
```

Generated artifacts:

- `incident_windows.csv`
- `preliminary_incident_windows.csv`
- `root_cause_candidates.csv`
- `candidate_rankings.csv`
- `evidence_mappings.csv`
- `incident_timeline.csv`
- `incident_reports.json`
- `case_studies.json`
- `metrics.json`
- `metadata.json`
- `run_report.md`
- figures under `figures/`

Figures:

- `incident_timeline.png`
- `risk_score.png`
- `evidence_signals.png`
- `top_candidate_metrics.png`
- `candidate_ranking.png`
- `metric_trajectories.png`
- `anomaly_forecast_overlap.png`

## Measured Phase 9 Run

Run:

```text
experiments/rca/phase9_rca_20260917
```

| Metric | Value |
| --- | ---: |
| Preliminary windows | 148 |
| Merged/deduplicated windows | 142 |
| Average evidence count per incident | 11.0352 |
| Candidate count | 1,122 |
| Duplicate incident rate | 0.0405 |
| Evidence coverage | 0.1514 |
| High-risk bucket coverage | 0.6375 |
| Provenance completeness | 1.0 |
| Temporal consistency | 1.0 |
| Rule consistency | 1.0 |

All 142 deduplicated windows in this default run are `CRITICAL` because Phase 9 intentionally uses an evidence-rich critical threshold.

## Case Studies

The run writes five case studies:

| Case | Result |
| --- | --- |
| Normal period | No incident created; below configured threshold. |
| Isolated anomaly | Shortest deduplicated incident that passed criteria. |
| Persistent anomaly | Longest deduplicated incident window. |
| Pre-failure period | Window overlapping the July MetroPT pre-failure context. |
| Difficult/noisy case | Highest uncertainty score from mixed validity and close rankings. |

Measured pre-failure/persistent case:

- Incident: `inc_30cfe846c6fc845c`
- Window: `2020-07-15T13:50:00` to `2020-07-15T14:25:00`
- Severity: `CRITICAL`
- Risk score: `0.9756`
- Top candidate: `TP2`

This is a candidate contributing signal, not proven causality.

## Evaluation

This phase does not claim RCA accuracy. It evaluates deterministic properties:

- evidence coverage;
- high-risk bucket coverage;
- duplicate incident rate;
- provenance completeness;
- temporal consistency;
- rule consistency;
- candidate counts.

Ground-truth RCA labels are not available for MetroPT, so diagnosis accuracy is not measured.
