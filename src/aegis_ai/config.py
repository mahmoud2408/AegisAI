"""Typed application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvironmentName = Literal["local", "test", "staging", "production"]
VectorStoreName = Literal["qdrant", "faiss"]
LlmProviderName = Literal["ollama", "openai_compatible", "none"]


class Settings(BaseSettings):  # type: ignore[misc]
    """Runtime configuration shared across API, workers, and pipelines."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AEGIS_",
        extra="ignore",
    )

    environment: EnvironmentName = "local"
    log_level: str = "INFO"
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    database_url: str = "postgresql+psycopg://aegis:aegis@localhost:5432/aegis_ai"
    redis_url: str = "redis://localhost:6379/0"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "aegis-ai-local"

    vector_store: VectorStoreName = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "aegis_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: LlmProviderName = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1"
    external_llm_api_key: SecretStr | None = None

    agent_max_tool_calls: int = Field(default=8, ge=1, le=30)
    agent_tool_timeout_seconds: int = Field(default=15, ge=1, le=120)
    expensive_endpoint_rate_limit: str = "10/minute"

    prometheus_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
