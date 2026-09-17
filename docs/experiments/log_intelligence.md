# Experiment: Phase 10 Log Intelligence

## Research Question

Can deterministic structured-log behavior provide additional evidence for incident investigation beyond telemetry-only evidence?

This phase measures log evidence coverage and BGL row-label behavior. It does not measure end-to-end incident diagnosis improvement because the LogHub samples and MetroPT telemetry are different systems.

## Command

```bash
python scripts/experiments/run_log_intelligence.py --run-id phase10_log_intelligence_20260917
```

Output directory:

```text
experiments/logs/phase10_log_intelligence_20260917
```

## Inputs

Local LogHub structured samples:

- `data/raw/loghub/openstack/OpenStack_2k.log_structured.csv`
- `data/raw/loghub/hdfs/HDFS_2k.log_structured.csv`
- `data/raw/loghub/bgl/BGL_2k.log_structured.csv`
- `data/raw/loghub/hadoop/Hadoop_2k.log_structured.csv`
- `data/raw/loghub/spark/Spark_2k.log_structured.csv`
- `data/raw/loghub/zookeeper/Zookeeper_2k.log_structured.csv`

The run also reads the Phase 8B evidence artifacts for a modality-level comparison:

- `experiments/evidence/phase8b_evidence_engine_20260916/evidence_signals.csv`
- `experiments/evidence/phase8b_evidence_engine_20260916/risk_signals.csv`

## Artifacts

Core tables:

- `profiles.csv`
- `event_statistics.csv`
- `bucket_summary.csv`
- `event_frequency_by_bucket.csv`
- `log_level_frequency_by_bucket.csv`
- `component_activity_by_bucket.csv`
- `transition_statistics.csv`
- `event_sequences.csv`
- `log_anomaly_signals.csv`
- `evidence_signals.csv`
- `risk_signals.csv`

Reports:

- `metrics.json`
- `case_studies.json`
- `comparison.json`
- `run_report.md`

Figures:

- `figures/event_frequency_over_time.png`
- `figures/event_type_distribution.png`
- `figures/log_level_distribution.png`
- `figures/anomaly_events_timeline.png`
- `figures/component_activity.png`
- `figures/sequence_anomaly_examples.png`
- `figures/telemetry_log_evidence_comparison.png`

## Dataset Profile

| Dataset | Rows | Top event examples |
| --- | ---: | --- |
| `loghub-openstack` | 2,000 | `E25` 931, `E27` 82, `E34` 82 |
| `loghub-hdfs` | 2,000 | `E6` 314, `E10` 311, `E11` 292 |
| `loghub-bgl` | 2,000 | `E67` 721, `E70` 208, `E4` 121 |
| `loghub-hadoop` | 2,000 | `E10` 476, `E44` 326, `E80` 289 |
| `loghub-spark` | 2,000 | `E35` 375, `E11` 305, `E24` 305 |
| `loghub-zookeeper` | 2,000 | `E24` 314, `E40` 299, `E11` 291 |

## Results

Total log evidence signals: 2,139.

| Signal type | Count |
| --- | ---: |
| `LOG_COMPONENT_ANOMALY` | 230 |
| `LOG_FREQUENCY_ANOMALY` | 269 |
| `LOG_LEVEL_ANOMALY` | 36 |
| `LOG_RARE_EVENT` | 497 |
| `LOG_SEQUENCE_ANOMALY` | 1,107 |

Total log risk buckets: 694.

| Severity | Buckets |
| --- | ---: |
| `INFO` | 204 |
| `LOW` | 361 |
| `MEDIUM` | 71 |
| `HIGH` | 31 |
| `CRITICAL` | 27 |

## BGL Label Evaluation

Only BGL has real labels in the local samples.

| Metric | Value |
| --- | ---: |
| Anomaly rows | 143 |
| Predicted event rows | 1,907 |
| Precision | 0.0713 |
| Recall | 0.9510 |
| F1 | 0.1327 |
| False alarm rate | 0.9537 |
| True positives | 136 |
| False positives | 1,771 |
| False negatives | 7 |
| True negatives | 86 |

Interpretation: the deterministic log heuristics recover most labeled BGL anomaly rows but overflag heavily. They are useful as investigation evidence and unsuitable as standalone anomaly detectors without calibration and stronger context.

## Case Studies

The run selected six case-study records:

| Case | Dataset | Timestamp | Signal |
| --- | --- | --- | --- |
| Normal log period | `loghub-bgl` | `2005-12-29T00:00:00` | none |
| Rare event | `loghub-openstack` | `2017-05-16T00:00:59.733000` | `LOG_RARE_EVENT` |
| Event frequency spike | `loghub-hdfs` | `2009-08-06T00:00:00` | `LOG_FREQUENCY_ANOMALY` |
| Error-rate spike | `loghub-bgl` | `2005-06-09T00:00:00` | `LOG_LEVEL_ANOMALY` |
| Sequence anomaly | `loghub-openstack` | `2017-05-16T00:14:05.577000` | `LOG_SEQUENCE_ANOMALY` |
| Noisy/difficult case | `loghub-spark` | `2017-06-09T20:11:11` | `LOG_FREQUENCY_ANOMALY` |

Raw-event previews and detected evidence records are stored in `case_studies.json`.

## Comparison

Phase 8B telemetry evidence:

- evidence signals: 10,341;
- risk buckets: 3,699;
- signal types: current anomaly, forecast deviation, trend, failure signal, state transition, historical context.

Phase 10 log evidence:

- evidence signals: 2,139;
- risk buckets: 694;
- signal types: rare event, frequency anomaly, level anomaly, sequence anomaly, component anomaly.

Claim: logs add complementary evidence types and entities. No cross-dataset incident-diagnosis improvement is claimed.

## Limitations

- The LogHub 2k files are small benchmark samples.
- Only BGL has labels in the local LogHub set.
- Sequence rarity and event rarity are broad heuristics and can overflag.
- WARN-heavy datasets can produce level evidence that is normal for that system.
- Log evidence is low-reliability and low-weight in risk aggregation.
- This phase does not perform LLM reasoning, RAG retrieval, or root-cause inference.
