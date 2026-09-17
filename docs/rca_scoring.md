# RCA Scoring

Phase 9 ranks root-cause candidates using deterministic evidence features. The score is a ranking score only. It is not a probability of causation, not incident probability, and not a remediation guarantee.

## Candidate Types

The engine emits candidates for:

- metric-level signals, such as `TP2` or `Oil_temperature`;
- component groups derived from observed MetroPT sensor names, such as `compressor_pressure_path`;
- the affected entity, currently `metropt_compressor`.

No service dependency graph is fabricated. `affected_services` remains empty for MetroPT because the dataset does not provide service relationships.

## Formula

Each feature is normalized to `[0, 1]`.

```text
candidate_score =
  candidate_type_weight
  * signal_role_multiplier
  * weighted_mean(feature_i)
```

Default feature weights:

| Feature | Weight | Meaning |
| --- | ---: | --- |
| `magnitude` | 0.24 | Mean/max normalized evidence strength. |
| `persistence` | 0.16 | Duration of supporting abnormal evidence, capped by `persistence_cap_minutes`. |
| `signal_count` | 0.12 | Log-scaled supporting evidence count. |
| `multi_signal_support` | 0.14 | Number of distinct evidence types supporting the candidate. |
| `forecast_deviation` | 0.10 | Strongest forecast-residual evidence. |
| `trend_strength` | 0.08 | Strongest trend evidence. |
| `state_transition` | 0.04 | Strongest operating-state transition evidence. |
| `temporal_precedence` | 0.06 | Whether candidate evidence appears before the incident reference point. |
| `historical_recurrence` | 0.04 | Strongest historical-context evidence. |
| `confidence` | 0.02 | Mean evidence confidence inherited from Phase 8B. |

Default candidate-type weights:

| Candidate Type | Weight |
| --- | ---: |
| `metric` | 1.00 |
| `component` | 0.95 |
| `entity` | 0.75 |

Entity candidates provide affected-machine context, but the reduced type weight prevents broad entity aggregation from automatically outranking specific metric/component evidence.

## Signal Role Weights

The signal role multiplier comes from the average evidence role weight, plus a small base offset. Defaults:

| Evidence Type | Role Weight |
| --- | ---: |
| `CURRENT_ANOMALY` | 1.00 |
| `FORECAST_DEVIATION` | 0.95 |
| `TREND` | 0.75 |
| `HISTORICAL_CONTEXT` | 0.55 |
| `STATE_TRANSITION` | 0.45 |
| `FAILURE_SIGNAL` | 0.20 |

`FAILURE_SIGNAL` remains low weight because Phase 7B marked MetroPT failure prediction as research-only.

## Persistence

Candidate persistence is:

```text
(last_supporting_evidence_time - first_supporting_evidence_time) + bucket_duration
```

The normalized feature is capped by `persistence_cap_minutes`, currently `60`.

## Temporal Precedence

Temporal precedence rewards candidates whose first evidence occurs earlier than the configured incident reference point. The default reference is the peak-risk bucket in the incident window.

This is only a weak ranking signal. Earlier evidence does not prove causality.

## Multi-Signal Support

A candidate supported by multiple evidence types receives a higher `multi_signal_support` score. Example:

```text
CURRENT_ANOMALY + TREND + FORECAST_DEVIATION
```

This is described as multi-signal support, not statistical independence.

## Measured Run

Run:

```text
experiments/rca/phase9_rca_20260917
```

Measured output:

| Metric | Value |
| --- | ---: |
| Root-cause candidates | 1,122 |
| Average candidates per incident | 7.9014 |
| Provenance completeness | 1.0 |
| Rule consistency | 1.0 |
| Temporal consistency | 1.0 |

## Limitations

- No causal graph is used.
- No intervention data is available.
- No RCA ground-truth labels are available.
- Scores rank evidence-supported candidates only.
- Close scores should be treated as uncertainty, not as decisive ordering.
