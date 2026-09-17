# AegisAI

Autonomous AI platform for predictive incident detection and investigation.

This repository is being built as a production-oriented AI engineering portfolio project. The current state is **Phase 10 deterministic log intelligence and log evidence**. It contains the architecture and data layer from earlier phases plus measured NAB/SMD anomaly-detection experiments, AI4I/MetroPT failure-prediction baselines, a MetroPT robustness audit, MetroPT forecasting baselines, deterministic evidence/risk aggregation, root-cause candidate ranking, and LogHub log-evidence generation. API endpoints, RAG, agents, monitoring dashboards, Docker Compose, and CI/CD remain later milestones.

## Objective

AegisAI will ingest telemetry, logs, traces, and technical documentation; detect anomalies; forecast risky system behavior; predict incidents; investigate probable causes; retrieve supporting documentation; and produce evidence-based incident reports through APIs and a React dashboard.

The project is designed to demonstrate:

- Machine learning, deep learning, anomaly detection, forecasting, and explainable AI
- Data engineering and reproducible experiment pipelines
- LLM provider abstraction, RAG, and agentic investigation workflows
- FastAPI, PostgreSQL, Redis, React, TypeScript, and Tailwind CSS
- MLOps with MLflow, Prometheus, Grafana, Docker, testing, and CI/CD
- Research methodology with honest measured results and ablation studies

## Proposed Architecture

```text
Telemetry / Logs / Traces / Docs
  -> Data ingestion
  -> Validation
  -> Feature engineering
  -> ML models
  -> Decision engine
  -> Root-cause analysis
  -> RAG retriever
  -> Investigation agent
  -> FastAPI
  -> React dashboard
```

Major boundaries:

- `src/aegis_ai/data`: dataset import, validation, preprocessing, and synthetic telemetry generation
- `src/aegis_ai/features`: leakage-safe feature engineering for time-series and log-derived signals
- `src/aegis_ai/ml`: anomaly detection, incident prediction, model evaluation, and model registry adapters
- `src/aegis_ai/forecasting`: metric forecasting baselines and sequence models
- `src/aegis_ai/logs`: deterministic structured-log profiling, anomaly signals, and log evidence
- `src/aegis_ai/decision`: incident severity, risk scoring, and decision policies
- `src/aegis_ai/rca`: deterministic and model-assisted root-cause analysis
- `src/aegis_ai/rag`: document ingestion, chunking, embeddings, retrieval, reranking, citations, and evaluation
- `src/aegis_ai/agents`: investigation tools, orchestration, validation, retries, timeouts, and report schemas
- `src/aegis_ai/api`: versioned FastAPI routes and WebSocket alert streams
- `src/aegis_ai/db`: SQLAlchemy models, repositories, and migrations
- `src/aegis_ai/observability`: logs, traces, metrics, and Prometheus instrumentation

See [docs/architecture.md](docs/architecture.md) for the full architecture proposal.

## Proposed Directory Tree

```text
.
|-- data/
|   |-- external/
|   |-- interim/
|   |-- processed/
|   `-- raw/
|-- docs/
|-- frontend/
|   `-- src/
|-- infrastructure/
|   |-- docker/
|   |-- mlflow/
|   |-- monitoring/
|   `-- postgres/
|-- notebooks/
|-- scripts/
|-- src/
|   `-- aegis_ai/
|       |-- agents/
|       |-- api/
|       |-- data/
|       |-- db/
|       |-- decision/
|       |-- domain/
|       |-- features/
|       |-- forecasting/
|       |-- llm/
|       |-- ml/
|       |-- observability/
|       |-- rag/
|       |-- rca/
|       `-- shared/
`-- tests/
    |-- agent/
    |-- api/
    |-- integration/
    |-- ml/
    |-- rag/
    `-- unit/
```

## Technology Decisions

- Backend runtime: Python 3.12 target for ML library compatibility.
- API: FastAPI with Pydantic response/request schemas.
- Persistence: PostgreSQL for relational state, Redis for cache/queues/rate-limiting support.
- ORM and migrations: SQLAlchemy 2.x and Alembic.
- ML stack: NumPy, Pandas, scikit-learn, XGBoost, LightGBM, PyTorch, statsmodels, SHAP.
- Experiment tracking: MLflow.
- RAG: Qdrant as the default vector database, FAISS as a local optional backend.
- LLMs: provider abstraction with Ollama and OpenAI-compatible external APIs as initial targets.
- Frontend: React, TypeScript, Tailwind CSS, Recharts, React Router, TanStack Query.
- Observability: structured logs, Prometheus metrics, Grafana dashboards.
- Testing: pytest for backend/ML/RAG/agent tests; Vitest and Testing Library for frontend tests.

## Dependency Layout

Python dependencies are declared in [pyproject.toml](pyproject.toml).

- Core dependencies are kept installable without the heaviest ML/RAG extras.
- Optional dependency groups separate `ml`, `rag`, `llm`, `mlops`, and `dev`.
- Exact lock files are intentionally not generated in Phase 1; they should be produced once the local Python 3.12 environment is selected.

Frontend dependencies are declared in [frontend/package.json](frontend/package.json).

## Data Layer

Phase 2 adds canonical dataset adapters and generated Parquet outputs for the local datasets available on this machine. Generated raw data, processed data, profiles, and lineage manifests are intentionally ignored by Git.

Core commands:

```powershell
python scripts/data/profile_all.py
python scripts/data/preprocess_all.py --batch-size 25000
python scripts/data/validate_processed.py
```

Measured local preprocessing outputs from the latest Phase 2 run:

| Dataset | Status | Output |
| --- | --- | --- |
| NAB | READY | 365,558 metrics; 236 labels |
| SMD | READY | 53,839,350 metrics; 708,747 labels |
| MetroPT-3 | READY | 22,754,220 metrics |
| AI4I | READY | 50,000 metrics; 10,000 failures; 60,000 labels |
| SMAP | LABELS_ONLY | 69 labels |
| MSL | LABELS_ONLY | 36 labels |
| LogHub OpenStack | READY | 2,000 logs |

These are row counts from preprocessing validation, not model-performance results.

Data docs:

- [docs/data_pipeline.md](docs/data_pipeline.md)
- [docs/dataset_adapters.md](docs/dataset_adapters.md)
- [docs/data_quality.md](docs/data_quality.md)
- [docs/data_splits.md](docs/data_splits.md)
- [docs/data_model.md](docs/data_model.md)

## Datasets

Dataset downloads are configured in [config/datasets.yaml](config/datasets.yaml). Raw datasets are ignored by Git and should live under `data/raw`.

| Dataset | Purpose | Status | Access |
| --- | --- | --- | --- |
| NAB | Univariate streaming anomaly detection | SCRIPTED | Public GitHub |
| SMD | Multivariate server anomaly detection | SCRIPTED | Public GitHub |
| SMAP/MSL | Spacecraft telemetry anomaly detection | SCRIPTED, source availability checked at runtime | Public Telemanom URLs |
| LogHub HDFS/BGL/OpenStack/Hadoop/Spark/Zookeeper | Log parsing and log-derived anomaly signals | SCRIPTED, small samples by default | Public GitHub |
| OpenTelemetry demo resources | Observability reference data and future telemetry generation | SCRIPTED_REFERENCE | Public GitHub |
| MetroPT-3 | Predictive maintenance and anomaly explanation | SCRIPTED | Public UCI |
| AI4I 2020 | Failure prediction and XAI practice | SCRIPTED | Public UCI |
| SWaT | Industrial-control anomaly detection | MANUAL | iTrust access request |
| WADI | Industrial-control anomaly detection | MANUAL | iTrust access request |
| CIC-IDS2017 | Cybersecurity anomaly detection | MANUAL | Official CIC download |

Dataset commands:

```powershell
python scripts/datasets/download_all.py --list
python scripts/datasets/download_all.py --dataset public
python scripts/datasets/download_all.py --dataset nab
python scripts/datasets/download_all.py --dataset loghub
python scripts/datasets/download_all.py --dataset loghub --large
python scripts/datasets/validate_datasets.py
```

The downloader writes `data/manifests/datasets_manifest.json` after each attempted dataset. That manifest is ignored by Git because it describes local data files.

Audit and strategy docs:

- [docs/dataset_audit.md](docs/dataset_audit.md)
- [docs/data_strategy.md](docs/data_strategy.md)
- [docs/data_model.md](docs/data_model.md)
- [docs/data_pipeline.md](docs/data_pipeline.md)
- [docs/dataset_adapters.md](docs/dataset_adapters.md)
- [docs/data_quality.md](docs/data_quality.md)
- [docs/data_splits.md](docs/data_splits.md)
- [docs/experimental_plan.md](docs/experimental_plan.md)
- [docs/roadmap.md](docs/roadmap.md)

## Environment Variables

The initial environment contract is documented in [.env.example](.env.example). Important groups:

- Database and cache: `AEGIS_DATABASE_URL`, `AEGIS_REDIS_URL`
- MLflow: `AEGIS_MLFLOW_TRACKING_URI`, `AEGIS_MLFLOW_EXPERIMENT`
- RAG: `AEGIS_VECTOR_STORE`, `AEGIS_QDRANT_URL`, `AEGIS_EMBEDDING_MODEL`
- LLM: `AEGIS_LLM_PROVIDER`, `AEGIS_LLM_BASE_URL`, `AEGIS_LLM_MODEL`, `AEGIS_EXTERNAL_LLM_API_KEY`
- Agent controls: `AEGIS_AGENT_MAX_TOOL_CALLS`, `AEGIS_AGENT_TOOL_TIMEOUT_SECONDS`
- Security and operations: `AEGIS_ALLOWED_ORIGINS`, `AEGIS_EXPENSIVE_ENDPOINT_RATE_LIMIT`

Do not commit real secrets. Use `.env` locally.

## Current Milestone

Completed foundations:

- Source package boundaries
- Dependency manifests
- Environment configuration
- Dataset storage policy
- Required documentation skeleton
- Basic scaffold tests
- Canonical Pydantic models and PyArrow schemas
- Dataset adapters for ready datasets and labels-only SMAP/MSL handling
- Streaming preprocessing into partitioned Parquet
- Profiles, lineage manifests, and processed-output validation
- NAB statistical, Isolation Forest, dense autoencoder, and LSTM autoencoder experiments
- NAB robustness/error analysis and SMD transition planning
- SMD multivariate anomaly detection across 28 machines
- AI4I supervised failure prediction and MetroPT event-derived predictive-maintenance baselines
- MetroPT robustness/target validation with event-level evaluation, threshold sensitivity, false-alarm analysis, July holdout analysis, distribution-shift diagnostics, and SHAP feature contribution artifacts
- MetroPT one-minute telemetry forecasting with persistence, moving average, rolling linear trend, ridge autoregression, univariate LSTM, multivariate LSTM, and forecast-based degradation/risk analysis

Latest measured Phase 7 run: `experiments/prediction/failure_prediction/phase7_failure_prediction_20260912`.

- AI4I best test F1: Random Forest with F1 `0.6327`, precision `0.6596`, recall `0.6078`, ROC-AUC `0.9674`, and PR-AUC `0.6757`.
- MetroPT result: no deployable winner; all classical baselines missed the held-out July failure under validation-selected thresholds. XGBoost had F1 `0.0000`, recall `0.0000`, PR-AUC `0.0035`, and FPR `0.0126`.

Latest measured Phase 7B run: `experiments/prediction/metropt_robustness/phase7b_metropt_robustness_20260914`.

- MetroPT labels: four report-curated air-leak events, one held-out July test failure, and no overlapping failure intervals.
- Best validation-selected candidate: 6-hour Random Forest at threshold `0.005`.
- Validation: F1 `0.1193`, PR-AUC `0.0567`, event detection `1.0000`, median lead time `5.9517` hours.
- Test: precision `0.0000`, recall `0.0000`, F1 `0.0000`, PR-AUC `0.0036`, event detection `0.0000`, false-positive rate `0.4376`, and `377.2708` false alarms per day.
- Conclusion: Decision **B**. MetroPT remains useful for forecasting, anomaly explanation, robustness, and future RCA evidence, but the current supervised early-warning classifier is not deployable.

Latest measured Phase 8 run: `experiments/forecasting/metropt/phase8_metropt_forecasting_20260915`.

- Forecast cadence: one-minute resample from raw median cadence `10.0` seconds.
- Selected sensors: `TP2`, `TP3`, `H1`, `Reservoirs`, `Oil_temperature`, and `Motor_current`.
- Horizons: 5, 15, and 30 minutes.
- Validation-selected global model: 5-minute multivariate LSTM.
- Test result for validation-selected model: MAE `1.2845`, RMSE `2.1687`, MAPE `10.6241`.
- Longer horizon winners: moving average at 15 minutes and 30 minutes.
- Forecast-based risk signal: mean precision `0.3296`, mean recall `0.0698`, mean F1 `0.1030`; useful as evidence, not as a standalone failure classifier.

See [docs/experiments/ai4i_failure_prediction.md](docs/experiments/ai4i_failure_prediction.md), [docs/experiments/metropt_failure_prediction.md](docs/experiments/metropt_failure_prediction.md), [docs/experiments/lead_time_analysis.md](docs/experiments/lead_time_analysis.md), [docs/experiments/metropt_failure_events.md](docs/experiments/metropt_failure_events.md), [docs/experiments/metropt_target_analysis.md](docs/experiments/metropt_target_analysis.md), [docs/experiments/metropt_robustness.md](docs/experiments/metropt_robustness.md), [docs/experiments/metropt_forecasting.md](docs/experiments/metropt_forecasting.md), [docs/experiments/forecast_risk_signal.md](docs/experiments/forecast_risk_signal.md), and [docs/models/forecasting_baselines.md](docs/models/forecasting_baselines.md).

## Known Risks

- Public datasets differ in label quality, licensing constraints, and benchmark conventions.
- Incident prediction is vulnerable to temporal leakage unless splits and rolling features are handled carefully.
- Root-cause analysis must distinguish evidence, correlation, model inference, and recommendations.
- LLM responses must remain grounded in retrieved citations and structured tool outputs.
- Deep learning experiments can become expensive; baselines and ablations must stay reproducible.
- Windows local development may need a dedicated Python 3.12 installation because the current shell resolves `python` to a broken WindowsApps stub.

## Local Commands

After installing Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

Frontend setup:

```powershell
cd frontend
npm install
npm run typecheck
npm test
```

Additional phase-specific commands will be added as later features are implemented.

Run the Phase 6 SMD multivariate experiment:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_smd_multivariate.py --run-id phase6_smd_multivariate_20260911_eps1e3
```

Run the Phase 7 failure-prediction experiment:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_failure_prediction.py --run-id phase7_failure_prediction_20260912
```

Run the Phase 7B MetroPT robustness experiment:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_metropt_robustness.py --run-id phase7b_metropt_robustness_20260914
```

Run the Phase 8 MetroPT forecasting experiment:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_metropt_forecasting.py --run-id phase8_metropt_forecasting_20260915
```
