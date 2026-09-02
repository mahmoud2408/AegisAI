# Architecture

Phase status: repository architecture only.

## Design Goals

AegisAI is a modular incident intelligence platform. The design separates data engineering, model development, serving, root-cause analysis, RAG, agent orchestration, API delivery, and user experience so that each layer can be tested and evaluated independently.

The system must avoid:

- Training code inside API routes
- Database code inside business logic
- Unlabeled synthetic data in production paths
- Fabricated metrics, explanations, citations, or recommendations
- Temporal leakage in feature engineering and supervised prediction
- LLM answers that are not grounded in retrieved evidence or structured tool outputs

## Runtime Flow

```text
Telemetry / logs / traces / docs
  -> ingestion jobs
  -> validation
  -> preprocessing
  -> feature engineering
  -> anomaly detection / forecasting / prediction
  -> decision engine
  -> root-cause analysis
  -> RAG retrieval
  -> investigation agent
  -> incident report
  -> API and dashboard
```

## Package Boundaries

| Package | Responsibility | Must not contain |
| --- | --- | --- |
| `aegis_ai.api` | FastAPI routers, request/response schemas, WebSocket handlers | Training loops, SQL query construction |
| `aegis_ai.data` | Dataset downloads, importers, validators, preprocessors, synthetic generator | API routes, model-specific training logic |
| `aegis_ai.features` | Rolling, lagged, trend, SLI, log, and trace features | Future-aware feature calculations |
| `aegis_ai.ml` | Anomaly detection, incident prediction, evaluation, explainability | FastAPI route handlers |
| `aegis_ai.forecasting` | Forecasting datasets, baselines, LSTM/Transformer models | Incident report generation |
| `aegis_ai.decision` | Risk scoring, severity policies, alert thresholds | Raw database access |
| `aegis_ai.rca` | Evidence aggregation, correlations, dependency-aware RCA | Claims that correlation proves causation |
| `aegis_ai.rag` | Document ingestion, chunking, embedding, retrieval, citations | Hidden or uncited source material |
| `aegis_ai.agents` | Tool schemas, agent execution, validation, retries, reports | Hidden chain-of-thought exposure |
| `aegis_ai.db` | SQLAlchemy models, sessions, migrations, repositories | Model training and LLM calls |
| `aegis_ai.observability` | Structured logging and Prometheus instrumentation | Business decisions |
| `aegis_ai.llm` | Provider abstraction for Ollama and external APIs | Provider-specific logic leaking into agents |
| `aegis_ai.domain` | Shared domain objects and enums | Infrastructure imports |
| `aegis_ai.shared` | Small generic utilities | Domain ownership |

## Storage Design

PostgreSQL is the system of record for:

- Services and machines
- Metrics, logs, traces
- Anomalies and forecasts
- Predictions and incidents
- Investigation reports
- Recommendations
- Model metadata and model runs
- Documents and embedding metadata

Redis will be used for:

- Short-lived cache entries
- Expensive endpoint rate limiting
- Background job coordination where practical

Vector storage will start with Qdrant and keep FAISS as an optional local fallback.

## Data Contracts

All data pipelines should convert source-specific formats into canonical internal schemas:

- Metric sample: timestamp, service, machine, metric name, value, unit, source
- Log event: timestamp, service, severity, template, message, attributes
- Trace span: trace ID, span ID, parent ID, service, operation, duration, status
- Incident label: incident window, affected service, severity, source, confidence

## Evaluation Boundaries

Each major component needs a reproducible evaluator:

- Anomaly detection: precision, recall, F1, PR-AUC, false positive rate, detection delay, NAB score when applicable
- Incident prediction: precision, recall, F1, ROC-AUC, PR-AUC, calibration
- Forecasting: MAE, RMSE, MAPE when appropriate
- RAG: recall@k, MRR, citation accuracy, grounded-answer checks
- Agent: investigation success rate, evidence quality, latency, tool failure recovery

No metric should be displayed unless produced by a reproducible command and saved artifact.

## Proposed Deployment Topology

Local development will eventually use Docker Compose for:

- FastAPI service
- PostgreSQL
- Redis
- Qdrant
- MLflow
- Prometheus
- Grafana
- React frontend

Phase 1 does not implement Docker Compose yet.
