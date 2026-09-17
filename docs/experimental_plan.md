# Experimental Plan

This plan turns the current datasets into a coherent research program for AegisAI. It intentionally avoids model implementation details beyond experiment design.

Research question:

> Does combining predictive models, telemetry analysis, retrieval-augmented generation, and agentic reasoning improve automated incident diagnosis compared with isolated machine-learning approaches?

## Dataset to Capability Mapping

| Dataset | Detection | Prediction | Forecasting | NLP | RCA | RAG | Agent | Benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAB | ✅ | 🟡 | ✅ | ❌ | ❌ | ❌ | 🟡 | ✅ |
| SMD | ✅ | ✅ | 🟡 | ❌ | 🟡 | ❌ | ✅ | ✅ |
| MetroPT-3 | ✅ | 🟡 | ✅ | ❌ | 🟡 | ❌ | ✅ | 🟡 |
| AI4I 2020 | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | 🟡 | 🟡 |
| LogHub BGL | ✅ | 🟡 | ❌ | ✅ | 🟡 | ❌ | ✅ | 🟡 |
| Other LogHub samples | 🟡 | ❌ | ❌ | ✅ | 🟡 | ❌ | ✅ | 🟡 |
| OpenTelemetry resources | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ✅ | ❌ |
| SMAP/MSL | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SWaT/WADI | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| CIC-IDS2017 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Legend:

- ✅ strongly recommended now.
- 🟡 possible with caveats or after adapter work.
- ❌ inappropriate with current local files.

Non-obvious decisions:

- MetroPT-3 is strong for forecasting and unsupervised anomaly detection, but not supervised incident prediction until failure labels are curated.
- AI4I is strong for supervised classification and explainability, but it is synthetic and non-temporal, so it cannot prove temporal incident prediction.
- LogHub templates support parsing and event representations. They are not anomaly labels.
- OpenTelemetry files are useful for generator design only; they are not measured telemetry.
- SMAP/MSL labels are present but unusable until the telemetry bundle is available.

## Label Strategy

| Dataset | Real Labels | Derived Labels | Synthetic Labels | Allowed Use |
| --- | --- | --- | --- | --- |
| NAB | Anomaly timestamps and scoring windows | Point/interval evaluation labels from official files | No | Anomaly detection evaluation and NAB scoring. |
| SMD | Point-wise test anomaly labels; interpretation intervals | Future-window incident labels from point labels | No | Multivariate detection and incident-window prediction. |
| MetroPT-3 | None in inspected CSV | Report-curated failure-window labels for documented case studies | Possible overlay | Forecasting, unsupervised detection, and carefully scoped early-warning prediction. |
| AI4I 2020 | Machine failure and mode columns | Class grouping if documented | No | Supervised failure prediction and XAI. |
| LogHub BGL | Row-level labels | Sequence labels from real row labels | No | Labeled log anomaly/classification. |
| Other LogHub | None local | Event frequency/window labels only for unsupervised signals | No | Parsing and representation learning. |
| OpenTelemetry | None | None | Yes | Synthetic cross-modal incidents. |
| SMAP/MSL | Interval label metadata | Not usable without telemetry | No | Blocked. |

## Leakage-Safe Splits

NAB:

- Process each series independently.
- For model selection, use chronological splits within each series.
- Fit normalization only on train slices.
- When using official NAB windows, use windows for evaluation/scoring only.
- Do not allow sliding windows from the end of train to overlap validation/test labels unless the split metadata records warm-up context separately.

SMD:

- Respect the official train/test separation.
- Use the final chronological portion of each train file as validation if a validation set is needed.
- Fit scalers on train only, per machine or train population depending on experiment.
- For incident prediction labels, derive `incident_within_horizon` from future test labels only after feature windows are built from current/past rows.
- Keep machine id in split metadata to avoid accidental entity leakage.

MetroPT-3:

- Use chronological splits only.
- Fit resampling/interpolation/scaling on train only.
- Forecasting targets must be shifted future values, never centered rolling statistics.
- Supervised failure-prediction claims require documented incident-label curation; Phase 7 uses report-derived failure intervals and keeps active failure rows out of the prediction task.

AI4I 2020:

- No temporal claim is allowed.
- Use stratified train/validation/test splits for classification.
- Drop `UDI` and `Product ID` from model features.
- Treat failure-mode columns carefully: they may be explanatory targets, but should not be used as predictors for `Machine failure` if the goal is early failure prediction.

LogHub:

- Split by line/time order for sequence experiments.
- For parser evaluation, source template files are labels/evaluation references, not training features.
- For BGL anomaly classification, preserve label imbalance and report PR-AUC.
- Avoid cross-file leakage when testing generalization across log systems.

OpenTelemetry/synthetic:

- Synthetic scenarios must persist scenario ids, root-cause labels, trace ids, service dependencies, and generation config.
- Train/test splits must group by scenario instance to avoid near-duplicate windows leaking across splits.

## Experiment Progression

| Experiment | Dataset | Task | Inputs | Outputs | Baseline | Candidate Models | Metrics | Contribution | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. NAB statistical baseline | NAB | Univariate anomaly detection | timestamp/value series | anomaly score and binary alerts | rolling z-score/MAD | EWMA, seasonal residual score | Precision, Recall, F1, PR-AUC, FPR, delay, NAB score | Establish honest benchmark floor | Mixed frequencies and scoring-window conventions |
| 2. SMD classical baseline | SMD | Multivariate anomaly detection | 38-metric windows | point anomaly score | per-feature z-score aggregation | Isolation Forest later | Precision, Recall, F1, PR-AUC, FPR, delay | Move from univariate to server telemetry | Anonymous metrics limit explanation |
| 3. AI4I failure baseline | AI4I | Failure classification | tabular sensor/state features | failure probability | Logistic Regression | Random Forest/XGBoost later | Precision, Recall, F1, ROC-AUC, PR-AUC, calibration | Supervised prediction and XAI sandbox | Synthetic, non-temporal |
| 4. MetroPT forecasting baseline | MetroPT-3 | Metric forecasting | timestamped multivariate sensors | future sensor values | persistence/seasonal naive | Statistical and LSTM later | MAE, RMSE, MAPE where valid | Real high-volume forecasting | No incident labels |
| 5. SMD incident-window prediction | SMD | Predict anomaly within future row window | past/current windows plus anomaly scores | future incident risk | Logistic Regression | Random Forest/XGBoost later | Precision, Recall, F1, ROC-AUC, PR-AUC, calibration | Bridges detection to prediction | Leakage from window overlap/future labels |
| 6. LogHub parser/sequence modeling | LogHub | Log representation | raw/structured log lines | templates, event sequences, anomaly signals | source EventId frequency | Drain-style parser later, sequence model later | Parser accuracy where templates exist, F1/PR-AUC for BGL | Adds NLP/log dimension | Small 2k samples |
| 7. Telemetry + log correlation | SMD + LogHub + synthetic | Correlation/RCA prototype | canonical metrics/logs/events | evidence graph | deterministic time/entity grouping | Graph/ranking model later | Evidence precision, latency, investigation success on synthetic | Tests unified schema | Real datasets lack shared IDs |
| 8. RAG-assisted diagnosis | Docs + synthetic incidents | Grounded remediation | incident summary + docs | cited answer | lexical retrieval | embeddings/reranker later | retrieval precision/recall, faithfulness, citation correctness | Adds technical knowledge layer | Must avoid unsupported recommendations |
| 9. Full AegisAI agent | Synthetic + selected real evidence | Incident investigation | tools over metrics/logs/docs/models | structured report | fixed workflow | tool-using LLM agent later | success rate, tool selection, RCA accuracy, recommendation quality, latency | End-to-end portfolio demo | Agent evaluation can become subjective |

## Benchmark Design

Anomaly detection:

- Precision, Recall, F1.
- PR-AUC for score-based detectors.
- False positive rate.
- Detection delay in timestamps for NAB/MetroPT and row indices for SMD.
- NAB score where official windows apply.

Failure prediction:

- Precision, Recall, F1.
- ROC-AUC.
- PR-AUC.
- Calibration: Brier score, reliability curve, expected calibration error.
- Report class imbalance and threshold policy.

Forecasting:

- MAE and RMSE.
- MAPE only when targets are strictly away from zero and interpretation is valid.
- Horizon-specific metrics, never one blended number without horizon documentation.

Log analysis:

- Template parsing accuracy where source templates are available.
- Event-level classification metrics for BGL.
- Sequence anomaly metrics when labels are available or synthetic.
- Event coverage and unknown-template rate.

RAG:

- Retrieval precision at k.
- Retrieval recall at k against curated question/chunk pairs.
- Context relevance.
- Answer relevance.
- Faithfulness.
- Citation correctness.

Agent:

- Investigation success rate.
- Correct tool selection.
- Root-cause accuracy when real/curated/synthetic labels exist.
- Recommendation quality by rubric.
- Latency.
- Tool failure/retry rate.

## Cross-Dataset Generalization

Candidate transfer experiments:

| Train | Test | Current Feasibility | Domain Shift | Measurement |
| --- | --- | --- | --- | --- |
| NAB category subset | Held-out NAB category | Feasible | Different univariate domains/cadences | Compare F1, PR-AUC, delay by category. |
| SMD machines subset | Held-out SMD machines | Feasible | Machine/entity shift with same metric space | Entity-held-out anomaly metrics. |
| SMD | SMAP/MSL | Blocked | Server metrics vs spacecraft telemetry | Revisit only when telemetry bundle is present. |
| MetroPT-3 | Synthetic industrial telemetry | Feasible after generator | Real compressor vs generated scenarios | Forecasting/detection degradation and calibration. |
| LogHub OpenStack | LogHub Hadoop/Spark/Zookeeper | Feasible for parsing representations | Different log templates and systems | Unknown-template rate and event-feature transfer. |
| BGL | Other LogHub samples | Weak for anomaly labels | BGL labels do not map cleanly to unlabeled systems | Use unsupervised score distribution only. |

Do not expect cross-dataset transfer to work automatically. These experiments are valuable precisely because they measure failure under domain shift.

## Synthetic Data Strategy

Synthetic data supplements real datasets. It should produce coherent metrics, logs, traces, service dependencies, incidents, root causes, and severity labels.

| Scenario | Affected Variables | Temporal Pattern | Expected Signature | Root Cause Label | Severity |
| --- | --- | --- | --- | --- | --- |
| CPU spike | CPU, latency, request queue | Sudden rise, short duration | CPU high before latency | overloaded service | medium/high |
| Memory leak | memory, GC, latency, restarts | Slow monotonic drift | memory increases until restart/error | memory leak | high |
| Request burst | traffic, CPU, latency, errors | Step or burst | traffic leads resource pressure | demand spike | medium |
| Network saturation | network in/out, latency, timeout logs | Gradual or burst | latency/errors with network utilization | network bottleneck | high |
| Disk bottleneck | disk IO, queue depth, DB latency | Sustained saturation | DB/API latency follows IO wait | disk saturation | high |
| Database latency | DB response, API latency, error rate | Slow degradation | downstream services affected | database bottleneck | critical |
| Service outage | availability, error logs, trace failures | Abrupt failure | one service unavailable | service crash | critical |
| Cascading failure | multiple service SLIs | Propagating wave | upstream/downstream sequence | dependency cascade | critical |
| Sensor drift | one sensor | Slow bias shift | residual grows without state change | sensor drift | low/medium |
| Sensor failure | one sensor | Flatline/dropout/spikes | impossible constant or missing behavior | sensor failure | medium |
| Pressure instability | pressure sensors, compressor states | Oscillation | TP/H1/Reservoirs instability | pressure-control issue | high |
| Thermal anomaly | temperature, current | Gradual overheating | oil temperature rises with current/load | cooling issue | high |

Synthetic generator requirements:

- Persist scenario configuration and random seed.
- Emit canonical metrics, logs, traces, labels, incidents, dependencies, and documentation references.
- Clearly mark all rows as synthetic.
- Keep synthetic benchmark results separate from real-data metrics.

## Data Fusion Strategy

Fusion should happen through canonical events and evidence graphs, not by forcing unrelated public datasets into the same incident.

Potential correlation keys:

- Timestamp or row-index window.
- Entity/machine/service id.
- Host/node/process id.
- Request id.
- Trace id.
- Incident id.
- Source dataset and scenario id.

Observed compatible identifiers:

- SMD: machine id from filename; no timestamps, request ids, or service graph.
- NAB: series path and timestamps; no logs/traces/entities beyond series.
- MetroPT-3: timestamp and compressor sensors/states; no logs/traces.
- AI4I: row/product identifiers; no timestamp/log/trace.
- LogHub BGL: node, component, event id, label.
- LogHub OpenStack: component, process id, `ADDR` field with request-like information.
- LogHub HDFS: process id and block ids embedded in content.
- OpenTelemetry resources: no rows, but future generated data can include service/request/trace ids.

Recommended fusion architecture:

```text
Raw datasets
  -> Dataset adapters
  -> Canonical metric/log/trace/label events
  -> Time/entity/request indexed feature views
  -> Evidence graph
  -> Incident engine
  -> RCA ranker
  -> RAG retriever
  -> Investigation agent
```

Current real-data fusion limits:

- Do not claim that an SMD anomaly and a LogHub event describe the same incident.
- Do not join MetroPT and AI4I as if they represent the same equipment.
- Use synthetic OpenTelemetry-style scenarios for true metric-log-trace correlation until a real correlated dataset is added.

## Ablation Study

| Condition | Components | Evaluation Data | Key Metrics |
| --- | --- | --- | --- |
| Baseline | Rules/statistics only | NAB, SMD, MetroPT, BGL | Detection metrics, latency |
| ML | Classical ML models | SMD, AI4I, BGL | Detection/prediction metrics, calibration |
| ML + Forecasting | ML plus forecast residuals | MetroPT, NAB, SMD | Forecast error, residual anomaly metrics |
| ML + RAG | ML evidence plus retrieved docs | Synthetic incidents with doc corpus | Faithfulness, citation correctness, recommendation quality |
| Full Agent | Tools, models, RCA, RAG | Synthetic correlated incidents plus real evidence snippets | Investigation success, RCA accuracy, tool selection, latency |

No ablation result should be shown until the corresponding scripts have generated measured outputs.
