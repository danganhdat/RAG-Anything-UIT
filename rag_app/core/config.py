from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenRouter API — no default, fails fast if missing
    openrouter_api_key: str

    # LLM models
    llm_text_model: str = "qwen/qwen3-30b-a3b"
    llm_vlm_model: str = "qwen/qwen2.5-vl-72b-instruct"

    # Embedding
    embed_model: str = "qwen/qwen3-embedding-8b"
    embed_dim: int = 4096

    # Milvus
    milvus_db_path: str = "./milvus_lite.db"
    milvus_db_name: str = ""
    milvus_collection: str = "rag_multimodal_collection"

    # HTTP / retry
    timeout: int = 120
    max_retries: int = 3

    # FastAPI
    allowed_origins: list[str] = Field(default=["http://localhost:8501"])

    # RAG
    rag_working_dir: str = "rag_workdir"

    # Reranker (uses OpenRouter Cohere reranker — reuses openrouter_api_key)
    reranker_enabled: bool = True
    reranker_model: str = "cohere/rerank-v3.5"

    # Language
    summary_language: str = "Vietnamese"
    query_user_prompt: str = "Always respond in Vietnamese (Tiếng Việt). Use Vietnamese terminology."

    # RAGAnything processing
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True
    rag_parser: str = "mineru"
    parse_method: str = "auto"
    context_window: int = 1
    context_mode: str = "page"
    max_context_tokens: int = 2000

    # MinerU parser — "cuda", "cpu", or "mps" (auto-detects if not set)
    mineru_device: str = "cuda"

    # Logging
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """Resolve relative paths against the project root so they work
        regardless of the current working directory or OS (WSL / Windows)."""
        self.rag_working_dir = str((_PROJECT_ROOT / self.rag_working_dir).resolve())
        self.milvus_db_path = str((_PROJECT_ROOT / self.milvus_db_path).resolve())
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
