from __future__ import annotations

import logging

import numpy as np
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything
from raganything.config import RAGAnythingConfig

from rag_app.adapters.embeddings import EmbeddingAdapter
from rag_app.adapters.llm import LLMAdapter
from rag_app.core.config import Settings

log = logging.getLogger(__name__)


class RAGService:
    """Creates and manages a RAGAnything instance."""

    @staticmethod
    async def create(
        settings: Settings,
        llm: LLMAdapter,
        emb: EmbeddingAdapter,
    ) -> RAGAnything:
        async def llm_func(prompt: str, **kwargs: object) -> str:
            return await llm.chat(prompt)

        async def embed_func(texts: list[str]) -> np.ndarray:
            result = await emb.embed_texts(texts)
            return np.array(result)

        config = RAGAnythingConfig(working_dir=settings.rag_working_dir)
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=llm.chat_with_image,
            embedding_func=EmbeddingFunc(
                embedding_dim=settings.embed_dim,
                max_token_size=8192,
                func=embed_func,
            ),
        )

        log.info("RAGAnything initialized (working_dir=%s)", settings.rag_working_dir)
        return rag
