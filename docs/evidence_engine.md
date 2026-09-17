# Evidence Engine

Phase 8B adds a deterministic evidence layer between measured model outputs and the future incident engine. It does not introduce a new model. Its purpose is to transform heterogeneous telemetry, forecast, operating-state, and research-model outputs into auditable `EvidenceSignal` records that can be inspected, aggregated, plotted, and later served through APIs.

## Scope

Implemented:

- current anomaly evidence from observed MetroPT values compared with train-only normal ranges;
- forecast residual evidence from the Phase 8 validation-selected LSTM forecaster;
- rolling trend evidence from prior/current observations only;
- operating-state transition evidence from MetroPT binary state columns;
- research-only failure-prediction evidence from the Phase 7B robustness artifact when available;
- historical context evidence from repeated prior signals;
- deterministic `RiskSignal` aggregation;
- evidence timeline and diagnostic plots under `experiments/evidence/`.

Not implemented in this phase:

- LogHub NLP;
- RAG;
- LLM or AI agent reasoning;
- FastAPI or dashboard endpoints;
- MLflow/Prometheus/Grafana integration;
- final incident decision engine.

## Signal Schema

Every evidence record uses the same fields:

| Field | Meaning |
| --- | --- |
| `signal_id` | Deterministic hash of timestamp, entity, type, metric, source, and value. |
| `timestamp` | Time when the evidence is available. Forecast residuals use target time, not input time. |
| `entity_id` | Entity being evaluated, currently `metropt_compressor`. |
| `signal_type` | One of `CURRENT_ANOMALY`, `FORECAST_DEVIATION`, `TREND`, `FAILURE_SIGNAL`, `STATE_TRANSITION`, `HISTORICAL_CONTEXT`. |
| `metric_name` | Sensor, state column, or derived metric name. |
| `value` | Raw signal value, such as residual, probability-like score, trend delta, or state. |
| `normalized_value` | Bounded `[0, 1]` strength after threshold normalization. |
| `severity` | `INFO`, `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`. |
| `confidence` | Bounded source-confidence score derived from reliability tier and signal strength. |
| `source_model` | Producing model or rule, for example `lstm_multivariate` or `train_normal_range_rule`. |
| `source_dataset` | Dataset provenance, currently `metropt`. |
| `evidence_description` | Human-readable factual evidence statement. |
| `validity` | `VALID`, `LIMITED`, or `RESEARCH_ONLY`. |
| `limitations` | Explicit caveat for downstream presentation. |

## Inputs

The default config is [configs/risk/evidence_scoring.yaml](../configs/risk/evidence_scoring.yaml).

The measured Phase 8B run consumed:

- `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915/forecasts.parquet`
- `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915/normal_ranges.csv`
- `experiments/prediction/metropt_robustness/phase7b_metropt_robustness_20260914/predictions.parquet`
- raw MetroPT data, if available locally, for binary state transitions

The engine does not mix NAB/SMD anomaly outputs into the MetroPT timeline because those outputs describe different entities/datasets. Current anomaly evidence for MetroPT is therefore generated as a train-normal-range rule and labeled accordingly.

## Run

```bash
python scripts/experiments/run_evidence_engine.py --run-id phase8b_evidence_engine_20260916
```

The run writes:

- `evidence_signals.csv` and `evidence_signals.parquet`
- `risk_signals.csv` and `risk_signals.parquet`
- `evidence_timeline.csv`
- `per_metric_evidence.csv`
- `model_reliability.csv`
- `case_studies.json`
- `metrics.json`
- `run_report.md`
- figures under `figures/`

## Measured Run Summary

Run: `experiments/evidence/phase8b_evidence_engine_20260916`

| Metric | Value |
| --- | ---: |
| Evidence signals | 10,341 |
| Risk buckets | 3,699 |
| Mean risk score | 0.2950 |
| Max risk score | 0.9991 |
| Mean confidence | 0.2329 |
| Leakage detected | false |

Evidence counts:

| Signal Type | Count |
| --- | ---: |
| `CURRENT_ANOMALY` | 1,011 |
| `FORECAST_DEVIATION` | 2,571 |
| `TREND` | 1,829 |
| `FAILURE_SIGNAL` | 1,500 |
| `STATE_TRANSITION` | 2,000 |
| `HISTORICAL_CONTEXT` | 1,430 |

Risk severity distribution:

| Severity | Buckets |
| --- | ---: |
| `INFO` | 1,712 |
| `LOW` | 909 |
| `MEDIUM` | 398 |
| `HIGH` | 274 |
| `CRITICAL` | 406 |

## Temporal Validity

The leakage audit checked 18,000 selected forecast rows and 2,571 forecast residual evidence records. It found:

- forecast alignment errors: `0`;
- residual signals timestamped before input end: `0`;
- leakage detected: `false`.

Forecast residual evidence is available only after the target timestamp has been observed. Forecast-input timestamps remain in source artifacts for audit, but the evidence engine never timestamps residual evidence at input time.

## Limitations

- Risk score is an evidence aggregation score, not a calibrated incident probability.
- Current anomaly evidence uses empirical train-only ranges, not physics-informed safety thresholds.
- Forecast residual evidence is retrospective at the target timestamp.
- MetroPT failure prediction remains `RESEARCH_ONLY`.
- State transitions and historical context support investigation but do not prove causality.
