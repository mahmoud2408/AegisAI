# AegisAI

Autonomous AI platform for predictive incident detection and investigation.

This repository is being built as a production-oriented AI engineering portfolio project. The current state is **Phase 1: repository architecture**. It intentionally contains architecture, package boundaries, dependency manifests, configuration contracts, and documentation scaffolding only. Data ingestion, model training, API endpoints, RAG, agents, monitoring dashboards, Docker Compose, and CI/CD will be added in later milestones.

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

## Environment Variables

The initial environment contract is documented in [.env.example](.env.example). Important groups:

- Database and cache: `AEGIS_DATABASE_URL`, `AEGIS_REDIS_URL`
- MLflow: `AEGIS_MLFLOW_TRACKING_URI`, `AEGIS_MLFLOW_EXPERIMENT`
- RAG: `AEGIS_VECTOR_STORE`, `AEGIS_QDRANT_URL`, `AEGIS_EMBEDDING_MODEL`
- LLM: `AEGIS_LLM_PROVIDER`, `AEGIS_LLM_BASE_URL`, `AEGIS_LLM_MODEL`, `AEGIS_EXTERNAL_LLM_API_KEY`
- Agent controls: `AEGIS_AGENT_MAX_TOOL_CALLS`, `AEGIS_AGENT_TOOL_TIMEOUT_SECONDS`
- Security and operations: `AEGIS_ALLOWED_ORIGINS`, `AEGIS_EXPENSIVE_ENDPOINT_RATE_LIMIT`

Do not commit real secrets. Use `.env` locally.

## First Milestone

Phase 1 establishes a clean repository contract:

- Source package boundaries
- Dependency manifests
- Environment configuration
- Dataset storage policy
- Required documentation skeleton
- Basic scaffold tests

No performance claims are made because no experiments have been run.

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

Phase-specific commands will be added as features are implemented.
