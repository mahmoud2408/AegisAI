# AegisAI Data Strategy

Latest local audit: `2026-09-03T12:47:49Z`.

Measured raw lake size: `895.0 MiB`.

The current repository has enough data to build a serious first version of AegisAI, but not every planned dataset is usable. The strongest strategy is to treat the available datasets as complementary evidence sources, not as one artificially merged incident universe.

## Strategic Recommendation

The scientifically strongest and technically most impressive use of the current data is:

1. Use **NAB** for the first univariate anomaly-detection benchmark because it has official anomaly timestamps and scoring windows.
2. Use **SMD** as the core multivariate server-telemetry benchmark because it has 28 machine entities, 38 telemetry dimensions, clean train/test structure, point-wise anomaly labels, and interpretation labels.
3. Use **MetroPT-3** for high-volume multivariate forecasting, operating-state modeling, unsupervised anomaly discovery, and carefully documented predictive-maintenance case studies. The CSV has no machine-readable failure label column; Phase 7 uses curated event intervals from the source report and keeps that provenance explicit.
4. Use **AI4I 2020** for supervised failure prediction and explainability only. It is clean and labeled, but synthetic and non-temporal.
5. Use **LogHub** samples for log parsing, event-template features, and log-sequence experiments. Use BGL for row-level log anomaly classification; treat the other local LogHub samples as unlabeled parsing/representation datasets.
6. Use **OpenTelemetry reference resources** to design a synthetic observability generator that can produce correlated metrics, logs, and traces. The local OpenTelemetry files are not a static benchmark dataset.
7. Keep **SMAP/MSL, SWaT, WADI, and CIC-IDS2017** out of benchmark claims until the actual telemetry/flow files are legally present and profiled.

This gives AegisAI a credible progression: clean anomaly detection, multivariate telemetry modeling, supervised failure prediction, forecasting, log analytics, synthetic cross-modal investigation, then RAG/agent evaluation.

## Dataset Roles

| Dataset | Primary Role | Secondary Role | Current Decision |
| --- | --- | --- | --- |
| NAB | Univariate anomaly detection benchmark | Forecasting baseline per series | P0 benchmark |
| SMD | Multivariate anomaly detection benchmark | Incident-window prediction from real anomaly labels | P0 benchmark |
| MetroPT-3 | Forecasting and unsupervised predictive maintenance | Curated early-warning case study | P0 data asset, Phase 7 supervised case study with documented report-derived labels |
| AI4I 2020 | Supervised failure prediction and XAI | Calibration demo | P0 compact supervised task |
| LogHub BGL | Labeled log anomaly/event classification | NLP/log-template features | P1 labeled log task |
| Other LogHub samples | Log parsing and event sequence modeling | Correlation features | P1 representation tasks |
| OpenTelemetry resources | Synthetic observability scenario design | Trace/log/metric schema examples | P1 generator input |
| SMAP/MSL | Future spacecraft anomaly benchmark | Cross-domain generalization | P3 blocked until telemetry bundle is present |
| SWaT/WADI | Future industrial-control anomaly benchmark | RCA case studies | P2/P3 manual access |
| CIC-IDS2017 | Future network security extension | Network incident class | P3 manual download |

## Real Labels, Derived Labels, Synthetic Labels

Real labels currently usable:

- NAB: 120 anomaly timestamps and 116 scoring windows across 52 labeled series.
- SMD: 708,420 point-wise test labels with 29,444 positive anomaly labels; interpretation files contain 327 anomaly intervals with affected metric indices.
- AI4I: row-level `Machine failure` target with 339 failures, plus failure-mode columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.
- LogHub BGL: row-level `Label`; `-` appears 1,857 times and non-normal labels appear 143 times in the local sample.

Real labels present but not currently usable:

- SMAP/MSL: 82 channel label rows and 105 anomaly intervals are present, but the telemetry archive is absent. These labels cannot be joined to observations yet.

Derived labels allowed with strict documentation:

- SMD incident-prediction labels may be derived by asking whether a test anomaly occurs within a future row-index window. Features must use only current/past rows.
- MetroPT early-warning labels may be derived only from documented source-report failure intervals. The source CSV still has no dense label column, and report-derived targets must remain separated from native labels.
- NAB interval labels may be derived from official windows for evaluation, but model features cannot use future points or scoring-window boundaries.
- LogHub sequence labels may be derived only where real labels exist, currently BGL in the local sample.

Synthetic labels allowed:

- Synthetic telemetry scenarios may generate explicit incident labels, root-cause labels, severity labels, and service-dependency metadata. These must be marked as synthetic/demo data and kept separate from real benchmark results.

Do not invent labels for unlabeled LogHub samples, OpenTelemetry resources, or manual datasets that have not been installed. For MetroPT-3, use only documented report-derived failure intervals and preserve provenance in experiment artifacts.

## Quality Score Methodology

The dataset quality score is not a model-performance metric. It is a readiness score for AegisAI experiments:

- Completeness and local availability: 25 points
- Schema/readability: 15 points
- Label usability for stated tasks: 20 points
- Integrity from measured missing/duplicate checks: 15 points
- Temporal usability: 15 points
- Practical routine-development cost: 10 points

| Dataset | Score | Reason |
| --- | ---: | --- |
| NAB | 91 | Complete, labeled, small, benchmark-standard; mixed frequencies and 49 duplicated timestamps require per-series handling. |
| SMD | 88 | Complete multivariate server benchmark with aligned labels; no real timestamps and unnamed dimensions limit production realism. |
| MetroPT-3 | 79 | Large clean timestamped telemetry with no missing/duplicate rows; no machine-readable failure labels in the CSV. |
| AI4I 2020 | 85 | Clean supervised labels and no missing/duplicate rows; synthetic and non-temporal. |
| LogHub BGL | 84 | Structured logs, templates, and row labels; only a 2k sample with irregular event time. |
| LogHub HDFS | 75 | Clean structured sample and templates; no anomaly labels in local files. |
| LogHub OpenStack | 77 | Clean structured sample with request-like `ADDR`; no labels in local files. |
| LogHub Hadoop | 74 | Clean structured sample; heavy duplicate timestamps and no labels. |
| LogHub Zookeeper | 73 | Clean structured sample with node/id fields; no labels and irregular event time. |
| LogHub Spark | 66 | Clean sample, but only 31 seconds of events, all `INFO`, and no labels. |
| OpenTelemetry resources | 48 | Useful reference/generator files, but not measured telemetry. |
| SMAP/MSL | 30 | Label metadata is present, telemetry bundle is missing. |
| SWaT | 10 | Manual-access placeholder only. |
| WADI | 10 | Manual-access placeholder only. |
| CIC-IDS2017 | 10 | Manual-download placeholder only. |

## Development Implications

The next coding milestone should be **canonical data model plus adapter interface**, followed by a **NAB adapter** and a **SMD adapter**. That order is better than jumping to models because it prevents leakage, label misuse, and one-off parsing logic.

Immediate P0 implementation sequence:

1. Define canonical schemas for metric observations, log events, trace spans, labels, anomaly windows, and incident events.
2. Define a dataset adapter protocol with `load_raw`, `validate`, `profile`, `transform`, `to_canonical`, and `get_labels`.
3. Implement and test `NABAdapter`.
4. Implement and test `SMDAdapter`.
5. Build leakage-safe split utilities and persist split metadata.
6. Only then start the first statistical anomaly baseline.

## Storage Strategy

- Keep all `data/raw` files immutable and ignored by Git.
- Keep generated manifests in `data/manifests`; they are local audit artifacts, not committed source truth.
- Write source-specific cleaned tables to `data/interim`.
- Write benchmark-ready windows/features/splits to `data/processed`.
- Write generated observability scenarios to `data/synthetic`, with explicit scenario and label metadata.
- Use sampled derivatives of SMD and MetroPT-3 for routine tests; reserve full scans for benchmark runs.
