from __future__ import annotations

import logging
from pathlib import Path

from raganything import RAGAnything

from rag_app.core.config import Settings
from rag_app.core.exceptions import IngestionError

log = logging.getLogger(__name__)


class IngestionService:
    """Handles PDF ingestion via RAGAnything's MinerU pipeline."""

    def __init__(self, rag: RAGAnything, settings: Settings) -> None:
        self._rag = rag
        self._settings = settings

    async def ingest_pdf(
        self,
        path: Path,
        *,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        kwargs: dict[str, int] = {}
        if start_page is not None:
            kwargs["start_page"] = start_page
        if end_page is not None:
            kwargs["end_page"] = end_page

        page_range = f"{kwargs.get('start_page', 0)}-{kwargs.get('end_page', 'end')}"
        log.info("Ingesting %s (pages %s)...", path.name, page_range)

        try:
            await self._rag.process_document_complete(
                str(path), device=self._settings.mineru_device, **kwargs
            )
        except Exception as exc:
            raise IngestionError(f"Failed to ingest {path.name}: {exc}") from exc

        log.info("Ingestion completed: %s", path.name)
