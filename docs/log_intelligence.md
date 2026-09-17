# Log Intelligence

Phase 10 adds deterministic log analysis on top of the canonical LogHub adapters. The goal is to turn structured log behavior into auditable investigation evidence without using LLMs, RAG, agents, or dashboard/API paths.

## Scope

Implemented:

- loading actual local LogHub structured samples through existing dataset adapters;
- canonical log normalization for timestamp, dataset, source file, event ID/template, component, process ID, request ID, level, and message;
- dataset-specific temporal aggregation based on observed timestamp coverage;
- event sequences and adjacent transition statistics;
- deterministic rare-event, frequency-deviation, log-level, sequence-rarity, and component-activity detectors;
- low-reliability log evidence signals compatible with the Phase 8B evidence engine;
- BGL row-label evaluation when labels are available;
- structural metrics for unlabeled datasets;
- reproducible run artifacts under `experiments/logs/`.

Not implemented:

- natural language log embeddings;
- LLM log summarization;
- RAG over log text;
- AI-agent investigation;
- FastAPI or React integration;
- incident-diagnosis improvement claims.

## Data Analyzed

Run: `experiments/logs/phase10_log_intelligence_20260917`

| Dataset | Rows | Event IDs | Components | Log levels | Bucket | Labels |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `loghub-openstack` | 2,000 | 43 | 10 | 2 | `10s` | 0 |
| `loghub-hdfs` | 2,000 | 14 | 6 | 2 | `7D` | 0 |
| `loghub-bgl` | 2,000 | 120 | 5 | 5 | `7D` | 2,000 |
| `loghub-hadoop` | 2,000 | 114 | 31 | 4 | `10s` | 0 |
| `loghub-spark` | 2,000 | 36 | 18 | 1 | `1s` | 0 |
| `loghub-zookeeper` | 2,000 | 50 | 70 | 3 | `1D` | 0 |

Only BGL has real row-level labels in the local samples. The other datasets are evaluated structurally only.

## Architecture

The Phase 10 path is:

1. `load_loghub_events` reads a LogHub family with the existing adapter.
2. `normalize_log_records` converts adapter rows into a stable canonical DataFrame.
3. `select_bucket_frequency` chooses a temporal bucket from observed coverage, unless overridden in config.
4. `aggregate_log_events` builds bucket, event, level, component, and transition tables.
5. Detector functions produce `LogEvidenceSignal` objects.
6. `LogEvidenceSignal.to_evidence_signal` converts them into shared `EvidenceSignal` records.
7. The existing evidence engine aggregates low-weight log evidence into `RiskSignal` buckets.

Primary files:

- `src/aegis_ai/logs/intelligence.py`
- `configs/logs/log_intelligence.yaml`
- `scripts/experiments/run_log_intelligence.py`
- `tests/unit/test_log_intelligence.py`

## Run

```bash
python scripts/experiments/run_log_intelligence.py --run-id phase10_log_intelligence_20260917
```

Generated artifacts include:

- `profiles.csv` and `profiles.json`
- `event_statistics.csv`
- `bucket_summary.csv`
- `event_frequency_by_bucket.csv`
- `log_level_frequency_by_bucket.csv`
- `component_activity_by_bucket.csv`
- `transition_statistics.csv`
- `event_sequences.csv`
- `log_anomaly_signals.csv`
- `evidence_signals.csv` and `evidence_signals.parquet`
- `risk_signals.csv`
- `case_studies.json`
- `comparison.json`
- `metrics.json`
- `run_report.md`
- figures under `figures/`

## Measured Summary

The Phase 10 run generated 2,139 log evidence signals and 694 low-weight log risk buckets.

| Signal type | Count |
| --- | ---: |
| `LOG_SEQUENCE_ANOMALY` | 1,107 |
| `LOG_RARE_EVENT` | 497 |
| `LOG_FREQUENCY_ANOMALY` | 269 |
| `LOG_COMPONENT_ANOMALY` | 230 |
| `LOG_LEVEL_ANOMALY` | 36 |

Structural evaluation:

- datasets with signals: all six LogHub samples;
- provenance completeness: `1.0`;
- deterministic reproducibility: `true`;
- mean normalized signal value: `0.8292`;
- max normalized signal value: `1.0`.

## BGL Label Evaluation

BGL has 143 anomalous rows and 1,857 normal rows in the local sample. Evaluation is row-level and uses detector source event IDs.

| Metric | Value |
| --- | ---: |
| Precision | 0.0713 |
| Recall | 0.9510 |
| F1 | 0.1327 |
| False alarm rate | 0.9537 |
| True positives | 136 |
| False positives | 1,771 |
| False negatives | 7 |
| True negatives | 86 |

This is not a production detector result. The measured behavior shows that broad deterministic log heuristics have high recall but extremely low precision on the BGL row labels.

## Limitations

- LogHub 2k samples are small benchmark slices, not full production streams.
- Event rarity and transition rarity can overflag normal but infrequent operational behavior.
- WARN-heavy datasets can produce level evidence even when WARN is normal for that sample.
- Correlation between log behavior and incidents is not causal proof.
- Cross-dataset comparison with MetroPT telemetry does not measure diagnosis improvement.
