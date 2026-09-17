# MetroPT Failure Events

Phase 7B characterizes the curated MetroPT-3 failure intervals before treating them as supervised labels. The raw telemetry CSV has no machine-readable failure column; these events come from the local source report and are used as documented case-study labels.

## Current Artifact

- Run: `experiments/prediction/metropt_robustness/phase7b_metropt_robustness_20260914`
- Event table: `failure_events.csv`
- Event summary: `failure_event_summary.json`
- Processed dataset audit: `processed_dataset_audit.json`

## Processed Data Check

The processed MetroPT long-format metric table is available locally:

| Item | Value |
| --- | ---: |
| Processed parquet files | 1 |
| Processed metric rows | 22754220 |
| Row groups | 911 |
| File size bytes | 228891089 |

The processed table stores metric records, not failure labels.

## Curated Event Table

| Event | Start | End | Duration hours | Observations | Split | Gap since previous end hours |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `metropt_air_leak_2020_04_18` | 2020-04-18 00:00 | 2020-04-18 23:59 | 23.9833 | 8657 | Train | n/a |
| `metropt_air_leak_2020_05_29` | 2020-05-29 23:30 | 2020-05-30 06:00 | 6.5000 | 2360 | Train | 983.5167 |
| `metropt_air_leak_2020_06_05` | 2020-06-05 10:00 | 2020-06-07 14:30 | 52.5000 | 17315 | Validation | 148.0000 |
| `metropt_air_leak_2020_07_15` | 2020-07-15 14:30 | 2020-07-15 19:00 | 4.5000 | 1622 | Test | 912.0000 |

## Event Structure Findings

| Finding | Value |
| --- | --- |
| Failure event count | 4 |
| Failure types | 1 |
| All events same type | Yes, air leak |
| Multiple telemetry rows per event | Yes |
| Overlapping failure intervals | 0 |
| Held-out July failure unique | Yes |
| Minimum gap between failure starts | 154.5 hours |
| Supervised learning sufficiency | Insufficient for reliable supervised learning |

## Interpretation

The event table supports a small case-study target, but not a high-confidence supervised predictive-maintenance benchmark. There are only four events, all from the same failure family, and the final test set contains exactly one held-out failure. This makes event-level metrics essential: row-level precision and recall alone can hide whether the model actually warns before a failure.

## Label Boundary

The experiment treats each interval as a failure episode. Rows inside active failure intervals are excluded from early-warning training and evaluation. Pre-failure rows are labeled positive only if a failure begins within the configured future horizon.
