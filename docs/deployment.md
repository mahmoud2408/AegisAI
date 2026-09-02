# Deployment

Phase status: deployment design only. Docker Compose is planned for a later phase.

## Local Services

The eventual local stack will include:

- FastAPI backend
- React frontend
- PostgreSQL
- Redis
- Qdrant or FAISS
- MLflow
- Prometheus
- Grafana

## Configuration

Configuration is environment-driven. Start from `.env.example` and create a local `.env` file that is never committed.

## Secrets

No secrets should be committed. External LLM credentials, database passwords, and service tokens must come from environment variables or a secret manager.

## Future Deployment Work

- Dockerfiles for backend and frontend
- Docker Compose for local development
- Database migrations
- Health checks
- Prometheus scrape config
- Grafana dashboards
- GitHub Actions build and test pipeline
