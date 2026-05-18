from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenRouter API — no default, fails fast if missing
    openrouter_api_key: str

    # LLM models
    llm_text_model: str = "qwen/qwen3-30b-a3b"
    llm_vlm_model: str = "qwen/qwen2.5-vl-72b-instruct"

    # Embedding
    embed_model: str = "qwen/qwen3-embedding-8b"
    embed_dim: int = 768

    # Milvus
    milvus_db_path: str = "./milvus_lite.db"
    milvus_collection: str = "rag_multimodal_collection"

    # HTTP / retry
    timeout: int = 120
    max_retries: int = 3

    # FastAPI
    allowed_origins: list[str] = Field(default=["http://localhost:8501"])

    # RAG
    rag_working_dir: str = "rag_workdir"

    # MinerU parser — "cuda", "cpu", or "mps" (auto-detects if not set)
    mineru_device: str = "cuda"

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
