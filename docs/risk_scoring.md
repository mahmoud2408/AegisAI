# Risk Scoring

Phase 8B risk scoring is deterministic and auditable. It converts normalized evidence into a bounded risk signal for an entity and time bucket. It must not be described as failure probability, incident probability, or causation probability.

## Aggregation Formula

For each evidence item:

```text
contribution = normalized_value * signal_type_weight * reliability_weight
```

For each entity/time bucket:

```text
risk_score = 1 - product(1 - contribution_i)
```

The complement-product form is monotonic and bounded in `[0, 1]`. It lets multiple independent evidence items accumulate without calling the result a probability.

## Severity Thresholds

Default risk severity thresholds:

| Severity | Lower Bound |
| --- | ---: |
| `LOW` | 0.15 |
| `MEDIUM` | 0.35 |
| `HIGH` | 0.60 |
| `CRITICAL` | 0.80 |

Values below `LOW` are labeled `INFO`.

Signal-specific thresholds are configured in [configs/risk/evidence_scoring.yaml](../configs/risk/evidence_scoring.yaml):

- current anomaly: train-normal absolute z-score;
- forecast residual: absolute residual divided by train standard deviation;
- trend: rolling window delta divided by train standard deviation;
- failure signal: Phase 7B score threshold, retained as research-only;
- historical context: count of prior signals inside the lookback window.

## Signal Weights

| Signal Type | Weight |
| --- | ---: |
| `CURRENT_ANOMALY` | 1.00 |
| `FORECAST_DEVIATION` | 0.90 |
| `TREND` | 0.65 |
| `HISTORICAL_CONTEXT` | 0.45 |
| `STATE_TRANSITION` | 0.35 |
| `FAILURE_SIGNAL` | 0.30 |

These weights reflect current engineering confidence. They are policy parameters, not learned values.

## Reliability Tiers

| Tier | Weight | Rule |
| --- | ---: | --- |
| `HIGH` | 1.00 | Validated on matching data with stable metrics and production-quality monitoring. |
| `MEDIUM` | 0.75 | Measured on matching data, but not calibrated enough to act alone. |
| `LOW` | 0.45 | Rule-based or descriptive signal that supports investigation context. |
| `RESEARCH_ONLY` | 0.15 | Known target/validation limitations; cannot dominate aggregation. |

Current source assignments:

| Source | Tier |
| --- | --- |
| `metropt:train_normal_range_rule` | `MEDIUM` |
| `metropt:lstm_multivariate` | `MEDIUM` |
| `metropt:moving_average` | `MEDIUM` |
| `metropt:rolling_trend_rule` | `LOW` |
| `metropt:state_transition_rule` | `LOW` |
| `metropt:historical_context_rule` | `LOW` |
| `metropt:metropt_failure_prediction` | `RESEARCH_ONLY` |

## Failure-Prediction Handling

The Phase 7B MetroPT failure-prediction artifact is allowed into the evidence layer only as `FAILURE_SIGNAL` with validity `RESEARCH_ONLY`. This means:

- it receives the low reliability weight `0.15`;
- it also receives the `FAILURE_SIGNAL` type weight `0.30`;
- a maximally normalized failure signal contributes at most `0.045` by itself;
- it can provide context but cannot dominate risk.

This policy is intentional because Phase 7B documented target fragility, imbalance, and limited failure-event validation.

## Confidence

Signal confidence is derived from source reliability and normalized signal strength:

```text
signal_confidence = reliability_weight * (0.5 + 0.5 * normalized_value)
```

Bucket confidence is the mean signal confidence scaled by evidence count. It is an evidence-confidence indicator, not model calibration.

## Interpretation

The risk signal answers:

> How much weighted evidence suggests this entity deserves investigation at this timestamp?

It does not answer:

> What is the probability that an incident will occur?

That distinction must remain visible in UI copy, reports, future API fields, and portfolio discussion.
