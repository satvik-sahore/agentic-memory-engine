from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM & Embedding Settings
    provider: str = "gemini"  # "gemini" or "openai"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 3072
    extraction_model: str = "gemini-3.5-flash-lite"
    reconciliation_model: str = "gemini-3.5-flash-lite"

    @property
    def active_api_key(self) -> Optional[str]:
        """Returns the active API key based on provider or fallback."""
        return self.gemini_api_key or self.openai_api_key

    # Qdrant Vector Store Settings
    qdrant_host: Optional[str] = "localhost"
    qdrant_port: Optional[int] = 6333
    qdrant_grpc_port: Optional[int] = 6334
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "agent_memories"
    qdrant_storage_path: Optional[str] = "./qdrant_data"  # Embedded local disk mode
    qdrant_prefer_embedded: bool = False  # Set True to force embedded disk mode

    # Memory Engine Pipeline Thresholds
    similarity_threshold: float = 0.60
    max_search_limit: int = 5
    enable_temporal_decay: bool = True
    recency_weight: float = 0.20
    decay_half_life_days: float = 30.0

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


settings = Settings()
