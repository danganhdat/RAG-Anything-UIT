from __future__ import annotations

import logging

from raganything import RAGAnything

from rag_app.adapters.embeddings import EmbeddingAdapter
from rag_app.adapters.llm import LLMAdapter
from rag_app.core.config import Settings
from rag_app.services.ingest import IngestionService
from rag_app.services.rag import RAGService

log = logging.getLogger(__name__)


class ServiceContainer:
    """Owns all services. Manages their lifecycle (startup / shutdown)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm: LLMAdapter | None = None
        self.emb: EmbeddingAdapter | None = None
        self.rag: RAGAnything | None = None

    async def startup(self) -> None:
        log.info("Starting services...")
        self.llm = LLMAdapter(self.settings)
        self.emb = EmbeddingAdapter(self.settings)
        self.rag = await RAGService.create(self.settings, self.llm, self.emb)
        log.info("All services started")

    async def shutdown(self) -> None:
        log.info("Shutting down services...")
        if self.llm:
            await self.llm.close()
        if self.emb:
            await self.emb.close()
        log.info("All services stopped")

    def get_ingestion_service(self) -> IngestionService:
        if self.rag is None:
            raise RuntimeError("ServiceContainer not started — call startup() first")
        return IngestionService(self.rag, self.settings)
