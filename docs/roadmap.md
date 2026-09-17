# Development Roadmap

This roadmap follows the audited data dependencies. It deliberately delays model training until canonical data handling and leakage-safe splits exist.

## Development Order

| Phase | Milestone | Why This Comes Here | Exit Criteria |
| --- | --- | --- | --- |
| 1 | Dataset audit and strategy | Establish source truth before modeling. | Audit docs and `dataset_audit.json` generated. |
| 2 | Canonical schemas and adapter protocol | Prevent one-off parsing and label misuse. | Typed schemas, adapter interface, tests. |
| 3 | NAB adapter and validation | Smallest labeled time-series benchmark. | Canonical metrics/labels with per-series validation. |
| 4 | Leakage-safe split utilities | Required before features/models. | Chronological split metadata and window-overlap tests. |
| 5 | NAB statistical anomaly baseline | First measurable benchmark. | Reproducible metrics, no fabricated results. |
| 6 | SMD adapter and multivariate feature views | Core server telemetry benchmark. | 28 machines, labels, and interpretation metadata canonicalized. |
| 7 | SMD anomaly baselines | Classical and deep multivariate experiments. | Baseline results, delay metrics, MLflow-ready artifacts. |
| 8 | AI4I adapter and supervised prediction | Clean compact failure target and XAI path. | Feature/label split, calibration-ready evaluation plan. |
| 9 | Incident prediction on SMD/AI4I | Bridges anomaly detection and risk prediction. | Future-window labels documented; no temporal leakage. |
| 10 | MetroPT adapter and forecasting views | High-volume timestamped telemetry. | Resampling policy, binary state handling, forecast splits. |
| 11 | Forecasting baselines | Adds future behavior estimation. | Horizon-specific MAE/RMSE and residuals. |
| 12 | LogHub adapter | Adds log parsing/NLP modality. | Raw/structured/template log events canonicalized. |
| 13 | Log-event modeling | Event features and BGL classification. | Parser/evaluation scripts and BGL metrics. |
| 14 | Synthetic observability generator | Needed for real metric-log-trace correlation. | Scenario configs, trace ids, service graph, labels. |
| 15 | RCA evidence engine | Uses canonical metrics/logs/synthetic incidents. | Evidence-ranked probable causes with clear uncertainty. |
| 16 | RAG document pipeline | Adds grounded technical context. | Chunk metadata, citations, retrieval evaluation. |
| 17 | Investigation agent | Orchestrates tools over existing components. | Structured reports, tool budgets, error handling. |
| 18 | FastAPI integration | Exposes validated capabilities. | Versioned endpoints and WebSocket alerts. |
| 19 | React dashboard | Portfolio-grade product interface. | Dashboard, incidents, models, benchmarks, RAG, agent pages. |
| 20 | MLflow/Prometheus/Grafana | MLOps and monitoring polish. | Tracked experiments and operational dashboards. |
| 21 | Docker Compose and CI/CD | Reproducible delivery. | Running stack, lint/type/test/build workflow. |
| 22 | Benchmark and ablation study | Final research contribution. | Reproducible measured comparison across ablations. |
| 23 | Portfolio polish | Interview readiness. | Clear docs, model cards, limitations, demo path. |

## Next Recommended Milestone

Implement **forecast residual and degradation evidence scoring**.

Scope:

- Convert Phase 8 forecasts into explicit evidence fields: current abnormality, forecasted abnormality, residual magnitude, trend direction, and operating-state transition context.
- Keep the output deterministic and inspectable before adding any LLM or agent layer.
- Use train-only normal ranges and residual distributions.
- Evaluate near curated MetroPT events without calling the signal a failure predictor.
- Keep MetroPT supervised failure prediction in research status until more event labels or stronger event-aware validation are available.
- Do not add RAG, agents, API, dashboard, MLflow, or monitoring until the evidence layer is ready.

Rationale:

- Phase 7B found only four curated MetroPT failure events, one held-out test event, and strong distribution shift.
- The validation-selected Random Forest missed the July failure and produced an impractical false-alarm burden.
- Phase 8 found that MetroPT is useful for continuous telemetry forecasting: the validation-selected 5-minute multivariate LSTM reached test RMSE `2.1687`, while moving average remained strong at 15 and 30 minutes.
- The forecast-based risk signal is useful as evidence but weak as a standalone signal, with sampled mean precision `0.3296`, recall `0.0698`, and F1 `0.1030`.
- It should not be presented as a deployable supervised early-warning classifier under the current labels.

## Priority Matrix

| Feature | Priority | Reason |
| --- | --- | --- |
| Raw data immutability and manifests | P0 | Required for reproducibility. |
| Dataset audit script | P0 | Required to avoid fabricated statistics. |
| Canonical schemas | P0 | Required before any multi-dataset work. |
| Adapter protocol | P0 | Required for maintainable ingestion. |
| NAB adapter | P0 | First labeled benchmark. |
| Leakage-safe split utilities | P0 | Required for credible ML. |
| Statistical anomaly baseline | P0 | First honest performance floor. |
| SMD adapter | P0 | Core multivariate server benchmark. |
| SMD anomaly models | P0 | Central ML benchmark. |
| AI4I supervised classification | P0 | Clean failure-prediction/XAI task. |
| MetroPT forecasting | P0 | Strong high-volume forecasting track. |
| LogHub adapter | P1 | Adds NLP/log analytics. |
| BGL labeled log anomaly task | P1 | Adds labeled log benchmark. |
| Synthetic observability generator | P1 | Needed for metric-log-trace-agent correlation. |
| RCA evidence engine | P1 | Core product value. |
| RAG pipeline | P1 | Required for grounded recommendations. |
| Investigation agent | P1 | Signature AegisAI capability. |
| FastAPI | P1 | Required product interface. |
| React dashboard | P1 | Required portfolio demonstration. |
| MLflow tracking | P1 | MLOps credibility. |
| Prometheus/Grafana | P1 | Operational credibility. |
| Docker Compose | P1 | Reproducible local stack. |
| GitHub Actions | P1 | CI credibility. |
| SHAP explanations | P1/P2 | High value, but only after supervised models exist. |
| LSTM autoencoder | P2 | Advanced anomaly model after baselines. |
| Transformer forecasting | P2 | Use only if baselines justify complexity. |
| Reranker for RAG | P2 | Useful after baseline retrieval evaluation. |
| SWaT/WADI integration | P2/P3 | Strong future benchmark but access-gated. |
| CIC-IDS2017 integration | P3 | Valuable security extension but outside core incident scope. |
| Cross-dataset transfer studies | P2 | Important research component after stable baselines. |

## What Not To Do

Datasets that should not be mixed as one incident source:

- NAB with SMD, MetroPT, AI4I, or LogHub as if they describe the same services.
- MetroPT and AI4I as if they describe the same equipment.
- SMD and LogHub as if log lines correspond to SMD machine anomalies.
- OpenTelemetry reference files as if they are measured incidents.

Datasets unsuitable for specific tasks:

- AI4I is unsuitable for forecasting and temporal incident prediction claims because it has no timestamp.
- MetroPT-3 is unsuitable for native supervised failure prediction from the CSV alone; use only documented report-curated failure intervals for case-study targets.
- SMAP/MSL is unsuitable for any benchmark while telemetry files are missing.
- Unlabeled LogHub samples are unsuitable for supervised anomaly claims.
- Spark 2k is too narrow for robust anomaly evaluation because all local rows are `INFO` and cover only 31 seconds.

Columns that should not be used as predictive features:

- AI4I `UDI` and `Product ID`.
- Any source row index used only as ordering/provenance, unless explicitly evaluating sequence position effects.
- Label columns or failure-mode columns as predictors of the target they define.
- NAB scoring-window boundaries as model inputs.
- LogHub `EventTemplate` as an input when evaluating a parser that is supposed to infer templates.

Transformations that could introduce leakage:

- Random train/test splits on NAB, SMD, MetroPT, or LogHub event sequences.
- Fitting scalers or imputers on full data before splitting.
- Computing rolling statistics with centered windows.
- Creating SMD future-incident labels before preserving a strict feature cutoff.
- Allowing sliding windows to overlap across train/test boundaries without a documented warm-up policy.
- Using downstream labels, failure modes, or template files as features.

Misleading conclusions to avoid:

- Claiming root cause from correlation alone.
- Claiming RAG improves diagnosis without citation and faithfulness evaluation.
- Claiming the agent is better than ML without a controlled synthetic/curated benchmark.
- Reporting model metrics before reproducible scripts produce them.
- Treating synthetic results as real-world performance.

## Storage Recommendations

Raw:

- Keep all downloaded source artifacts in `data/raw`.
- Never edit raw CSV/TXT/JSON/PDF/ZIP files.
- Do not commit raw datasets.

Interim:

- Store canonicalized per-source parquet/CSV/JSONL outputs.
- Include adapter version, source path, source row index, and label type.
- Keep large interim files ignored by Git.

Processed:

- Store benchmark-ready features, labels, and split definitions.
- Persist split metadata next to features.
- Store small schema fixtures in tests only when legally safe and size-appropriate.

Synthetic:

- Store generated scenarios separately from real datasets.
- Persist generation config, random seed, scenario id, incident id, and label provenance.

Routine development:

- Use NAB and AI4I by default because they are compact.
- Use LogHub 2k samples for parser tests.
- Use a small SMD machine subset for rapid local checks.
- Use sampled MetroPT windows for rapid checks.

Benchmark runs:

- Use full NAB.
- Use full SMD.
- Use full MetroPT for forecasting after the adapter exists.
- Use full LogHub samples currently installed; add full releases only as an explicit large-data milestone.

Excluded from routine experiments until installed/completed:

- SMAP/MSL telemetry.
- SWaT.
- WADI.
- CIC-IDS2017.

## Current Status

Completed:

- Local raw data inspected.
- Dataset audit script added.
- `data/manifests/dataset_audit.json` generated.
- Data strategy, audit, model, experiment, and roadmap docs created.
- Canonical Pydantic models and PyArrow schemas.
- Dataset adapters and preprocessing for NAB, SMD, MetroPT, AI4I, selected LogHub samples, and labels-only SMAP/MSL metadata.
- NAB statistical, Isolation Forest, dense autoencoder, and LSTM autoencoder anomaly experiments.
- NAB robustness/error analysis and SMD transition planning.
- SMD multivariate anomaly detection across all 28 machines.
- AI4I supervised failure-prediction baselines.
- MetroPT event-derived predictive-maintenance baselines.
- MetroPT Phase 7B robustness audit, target validation, July failure analysis, threshold sensitivity, false-alarm analysis, distribution-shift audit, and SHAP feature contribution artifacts.
- MetroPT Phase 8 forecasting baselines, LSTM models, per-sensor/per-horizon metrics, forecast artifacts, plots, and forecast-based degradation/risk signal analysis.

Not started:

- RAG.
- Agent.
- API.
- Frontend.
- MLflow.
- Monitoring.
- Docker Compose.
- CI/CD.
