from __future__ import annotations

import logging

from rag_app.adapters.base import OPENROUTER_BASE_URL, BaseOpenRouterClient
from rag_app.core.config import Settings

log = logging.getLogger(__name__)


class EmbeddingAdapter(BaseOpenRouterClient):
    """OpenRouter text embeddings (single + batch)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._embed_url = f"{OPENROUTER_BASE_URL}/embeddings"

    async def embed_text(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self._settings.embed_model, "input": texts}
        data = await self._post_with_retry(self._embed_url, payload)
        return [item["embedding"] for item in data["data"]]
