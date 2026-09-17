# Incident Windowing

Phase 9 groups Phase 8B `RiskSignal` buckets into incident windows. It does not treat every high-risk bucket as a separate incident.

## Inputs

Default input run:

```text
experiments/evidence/phase8b_evidence_engine_20260916
```

Required artifacts:

- `risk_signals.csv` or `risk_signals.parquet`
- `evidence_signals.csv` or `evidence_signals.parquet`

The window detector keeps the original evidence records. It never rewrites signal IDs, source models, source datasets, or timestamps.

## Configuration

Config file:

```text
configs/rca/rca_scoring.yaml
```

Default incident-window thresholds:

| Parameter | Value |
| --- | ---: |
| `min_risk_threshold` | 0.80 |
| `min_bucket_evidence_count` | 6 |
| `min_window_evidence_count` | 8 |
| `min_persistence_minutes` | 5 |
| `critical_single_bucket_threshold` | 0.80 |
| `max_gap_minutes` | 15 |
| `merge_gap_minutes` | 20 |
| `merge_pattern_jaccard_threshold` | 0.25 |

These defaults intentionally focus Phase 9 on evidence-rich critical windows. Lower thresholds produce many short windows and noisier RCA reports.

## Algorithm

1. Sort risk buckets by `entity_id` and `timestamp`.
2. Keep buckets where:
   - `risk_score >= min_risk_threshold`;
   - `evidence_count >= min_bucket_evidence_count`.
3. Start a new preliminary window when the time gap from the previous kept bucket is greater than `max_gap_minutes`.
4. Attach all evidence whose bucket timestamp falls inside the window.
5. Keep the window only if it has at least `min_window_evidence_count` evidence records and either:
   - persists for at least `min_persistence_minutes`; or
   - contains a bucket with `risk_score >= critical_single_bucket_threshold`.
6. Merge adjacent windows when:
   - they belong to the same entity;
   - their gap is at or below `merge_gap_minutes`;
   - their metric/evidence-type pattern Jaccard overlap is at or above `merge_pattern_jaccard_threshold`.

## Deduplication

Persistent evidence is deduplicated into one incident window instead of one incident per bucket. The measured Phase 9 run produced:

| Metric | Value |
| --- | ---: |
| Preliminary windows | 148 |
| Merged windows | 142 |
| Duplicate incident rate | 0.0405 |

## Time Semantics

Risk timestamps are bucket timestamps. Evidence records inside that bucket can occur after the bucket start. The incident window stores bucket start/end times and computes persistence as:

```text
persistence_minutes = (last_bucket - first_bucket) + aggregation_bucket_duration
```

The temporal-consistency audit uses the effective bucket end time so evidence within the final bucket is counted correctly.

## Limitations

- Windowing is deterministic thresholding, not incident ground truth.
- The method can miss lower-risk but operationally meaningful events.
- The method can split related activity if the evidence gap exceeds the configured threshold.
- Merge overlap is based on observed metric/evidence patterns, not causal topology.
