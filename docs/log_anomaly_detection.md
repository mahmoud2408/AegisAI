# Log Anomaly Detection

Phase 10 uses deterministic, explainable log detectors. These detectors are designed to create investigation evidence, not final incident labels.

## Canonical Inputs

Every detector consumes normalized log rows with:

- `timestamp`
- `source_dataset`
- `source_file`
- `canonical_log_event_id`
- `sequence_index`
- `entity_id`
- `component`
- `process_id`
- `request_id`
- `log_level`
- `event_id`
- `event_template`
- `message`

Missing optional values are preserved. When component is missing, the normalizer falls back to the canonical entity ID so downstream evidence still has an attributable entity.

## Temporal Buckets

Bucket sizes are selected from observed dataset coverage:

| Coverage | Bucket |
| --- | --- |
| up to 120 seconds | `1s` |
| up to 1 hour | `10s` |
| up to 7 days | `1h` |
| up to 45 days | `1D` |
| more than 45 days | `7D` |

The measured run selected:

| Dataset | Bucket |
| --- | --- |
| `loghub-openstack` | `10s` |
| `loghub-hdfs` | `7D` |
| `loghub-bgl` | `7D` |
| `loghub-hadoop` | `10s` |
| `loghub-spark` | `1s` |
| `loghub-zookeeper` | `1D` |

## Detectors

### Rare Event

Signal: `LOG_RARE_EVENT`

The rare-event rule flags event IDs whose total sample frequency is below `rare_event_max_frequency` or whose count is at most `rare_event_max_count`. It emits only the first few occurrences per rare event ID.

Limitation: rare events can be routine but uncommon operations.

### Frequency Deviation

Signal: `LOG_FREQUENCY_ANOMALY`

For each event ID, the detector builds a per-bucket count series and compares observed counts against the event's median using a robust median absolute deviation scale. It flags buckets where count and robust z-score exceed configured thresholds.

Limitation: high volume can reflect workload, retries, or sample boundary effects, not necessarily failure.

### Log-Level Anomaly

Signal: `LOG_LEVEL_ANOMALY`

The level detector tracks the bucket rate of `WARN`, `WARNING`, `ERROR`, `FATAL`, `SEVERE`, and `CRITICAL`. It flags buckets using either:

- robust deviation from that dataset's own baseline; or
- an absolute non-INFO level rate at or above `level_rate_absolute_threshold`.

Limitation: some LogHub samples, especially Hadoop and Zookeeper, are naturally WARN-heavy. Level evidence is therefore low reliability and must be interpreted with surrounding context.

### Sequence Rarity

Signal: `LOG_SEQUENCE_ANOMALY`

The sequence detector sorts rows chronologically and builds adjacent event transitions. Rare transitions are flagged when transition count or transition frequency is below configured thresholds.

Limitation: adjacent event rarity is an n-gram pattern, not causal sequence reasoning.

### Component Activity

Signal: `LOG_COMPONENT_ANOMALY`

The component detector aggregates event counts by component and bucket, then applies the same robust median/MAD spike rule used by frequency deviation.

Limitation: component activity spikes identify noisy components, but do not infer dependency impact by themselves.

## Measured Signal Counts

Run: `experiments/logs/phase10_log_intelligence_20260917`

| Dataset | Component | Frequency | Level | Rare | Sequence |
| --- | ---: | ---: | ---: | ---: | ---: |
| `loghub-bgl` | 18 | 48 | 3 | 212 | 250 |
| `loghub-hadoop` | 47 | 51 | 32 | 166 | 250 |
| `loghub-hdfs` | 12 | 25 | 0 | 10 | 95 |
| `loghub-openstack` | 41 | 0 | 0 | 22 | 138 |
| `loghub-spark` | 82 | 106 | 0 | 25 | 124 |
| `loghub-zookeeper` | 30 | 39 | 1 | 62 | 250 |

The sequence detector hits the per-dataset cap for BGL, Hadoop, and Zookeeper. That is expected for a broad rare-transition heuristic and is one reason the evidence is low weight.

## Evaluation Policy

- BGL uses real row-level labels and reports precision, recall, F1, and false alarm rate.
- Other datasets report structural metrics only.
- The run never fabricates missing labels.
- The experiment does not report PR-AUC, ROC-AUC, NAB score, or incident diagnosis accuracy because those labels are not available for this phase.
