# Log Evidence

Phase 10 extends the shared evidence schema with log-derived signal types. The goal is to make logs available to the incident and RCA layers as factual, low-weight evidence.

## Signal Types

The evidence enum now includes:

- `LOG_RARE_EVENT`
- `LOG_FREQUENCY_ANOMALY`
- `LOG_LEVEL_ANOMALY`
- `LOG_SEQUENCE_ANOMALY`
- `LOG_COMPONENT_ANOMALY`

Each log signal is created as `LogEvidenceSignal` and converted to the shared `EvidenceSignal` schema before risk aggregation.

## Provenance

Log-specific records keep provenance that the shared evidence schema does not yet expose directly:

- `source_dataset`
- `source_file`
- `source_event_ids`
- `detector_name`
- `event_id`
- `component`
- detector metadata such as reference frequency, robust z-score, bucket size, and thresholds

The run writes both:

- `log_anomaly_signals.csv`, with log-specific provenance;
- `evidence_signals.csv`, with the shared evidence schema.

## Reliability and Weights

Log evidence is intentionally low influence. Default signal weights are:

| Signal type | Weight |
| --- | ---: |
| `LOG_RARE_EVENT` | 0.25 |
| `LOG_FREQUENCY_ANOMALY` | 0.45 |
| `LOG_LEVEL_ANOMALY` | 0.40 |
| `LOG_SEQUENCE_ANOMALY` | 0.35 |
| `LOG_COMPONENT_ANOMALY` | 0.40 |

All LogHub detector sources are configured as `LOW` reliability. The evidence engine multiplies normalized signal value by signal-type weight and reliability weight during risk aggregation.

## Risk Impact

Measured run: `experiments/logs/phase10_log_intelligence_20260917`

| Metric | Value |
| --- | ---: |
| Log evidence signals | 2,139 |
| Log risk buckets | 694 |
| Mean log risk score | 0.2521 |
| Max log risk score | 1.0000 |

Log risk severity distribution:

| Severity | Buckets |
| --- | ---: |
| `INFO` | 204 |
| `LOW` | 361 |
| `MEDIUM` | 71 |
| `HIGH` | 31 |
| `CRITICAL` | 27 |

High or critical log risk buckets should be interpreted as concentrated log evidence, not as incident probabilities.

## Complementary Evidence

Comparison with the Phase 8B telemetry evidence run:

| Source | Evidence signals | Risk buckets | Signal types |
| --- | ---: | ---: | --- |
| MetroPT telemetry evidence | 10,341 | 3,699 | current anomaly, forecast deviation, trend, failure signal, state transition, historical context |
| LogHub log evidence | 2,139 | 694 | rare event, frequency anomaly, level anomaly, sequence anomaly, component anomaly |

The comparison shows modality coverage, not model superiority. MetroPT and LogHub are different systems, so this phase does not claim incident diagnosis improvement from adding logs.

## Downstream Use

Incident and RCA layers should display log evidence under a separate evidence category with wording such as:

- detected evidence: event frequency spike, rare transition, WARN/ERROR-rate bucket;
- model inference: low-reliability deterministic heuristic;
- probable cause: only if supported by telemetry, topology, history, and log context together;
- recommendation: operational check or investigation step, not automatic remediation.

This distinction prevents log correlation from being presented as causal proof.
