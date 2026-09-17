# Evidence Timeline

Phase 8B writes a time-aligned evidence timeline for the MetroPT compressor case study. The timeline is designed for future incident pages, root-cause analysis, and AI-agent tools.

## Artifact

Run directory:

```text
experiments/evidence/phase8b_evidence_engine_20260916
```

Key timeline files:

- `evidence_signals.csv`: one row per evidence item;
- `risk_signals.csv`: one row per entity/time bucket;
- `evidence_timeline.csv`: joined risk buckets with per-type evidence counts;
- `case_studies.json`: normal, anomalous, difficult, and pre-failure examples;
- `figures/risk_score_timeline.png`;
- `figures/evidence_timeline.png`;
- `figures/forecast_deviation_vs_risk.png`;
- `figures/current_anomaly_vs_risk.png`;
- `figures/top_evidence_types.png`;
- `figures/per_metric_evidence.png`.

The timeline buckets evidence using the same configured aggregation frequency as risk scoring (`5min` by default).

## Case Studies

The case-study selector intentionally includes easy and difficult contexts. It does not cherry-pick only successes.

| Case | Timestamp | Risk Score | Severity | Evidence Count | Selection Rule |
| --- | --- | ---: | --- | ---: | --- |
| Normal | `2020-06-13T18:00:00` | 0.0045 | `INFO` | 1 | Lowest aggregate evidence bucket. |
| Anomalous | `2020-07-17T07:25:00` | 0.9991 | `CRITICAL` | 11 | Highest aggregate evidence bucket. |
| Difficult | `2020-07-17T07:25:00` | 0.9991 | `CRITICAL` | 11 | Largest normalized forecast-residual bucket. |
| Pre-failure | `2020-07-15T14:10:00` | 0.9756 | `CRITICAL` | 9 | Highest bucket inside the configured pre-failure window for `metropt_air_leak_2020_07_15`. |

### Normal Example

The lowest-risk bucket still contains one Phase 7B failure-prediction signal. Because that source is `RESEARCH_ONLY`, its risk contribution is small. This is useful: the case demonstrates that research-only signals can appear in the timeline without dominating the score.

### Anomalous/Difficult Example

The highest-risk and largest forecast-residual bucket occur at the same time in this run: `2020-07-17T07:25:00`. Top evidence includes critical forecast residuals for `Reservoirs` and `TP3`.

Example evidence:

- `Reservoirs` forecast residual: `-2.786`, equivalent to `4.42` train-standard-deviation units.
- `TP3` forecast residual: `-2.760`, equivalent to `4.37` train-standard-deviation units.

This is a difficult case because large residuals can indicate model mismatch, unusual operating state, unmodeled dynamics, or true degradation. The evidence engine records the signal and limitation; it does not claim causation.

### Pre-Failure Example

The pre-failure case is selected from the six-hour context window before the curated July MetroPT failure event. Top evidence includes current anomaly signals:

- `Oil_temperature` observed value `80.875`, train-normal absolute z-score `3.30`;
- `TP2` observed value `10.508`, train-normal absolute z-score `3.16`.

These are evidence items, not proof that the July event was caused by either metric.

## Leakage Audit

The run checked:

- selected forecast rows: `18,000`;
- forecast residual signals: `2,571`;
- forecast alignment errors: `0`;
- residual signals timestamped before input end: `0`;
- leakage detected: `false`.

The key policy is that residual evidence is timestamped at `target_timestamp`. It is not available at forecast `input_end_timestamp`, because it depends on the future observed value.

## Dashboard Handoff

Future UI/API consumers should present four separate layers:

- detected evidence: raw values, residuals, trends, state transitions;
- model inference: source model/rule, reliability, confidence;
- probable cause: future RCA engine output only;
- recommendation: future action engine output only.

The evidence timeline should remain factual and cite `signal_id` values so reports can trace every statement back to measured artifacts.
