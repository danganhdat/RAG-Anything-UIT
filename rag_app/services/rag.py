from __future__ import annotations

from configparser import ConfigParser
import logging
import os
from pathlib import Path

import numpy as np
from lightrag.kg import STORAGE_ENV_REQUIREMENTS
from lightrag.rerank import generic_rerank_api
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything
from raganything.config import RAGAnythingConfig

from raganything import register_prompt_language, set_prompt_language

from rag_app.adapters.embeddings import EmbeddingAdapter
from rag_app.adapters.llm import LLMAdapter
from rag_app.core.config import Settings

log = logging.getLogger(__name__)


def _write_lightrag_milvus_config(settings: Settings) -> Path:
    """Write config.ini so LightRAG's Milvus code reads the local DB path.

    We avoid setting MILVUS_URI as an env var because pymilvus reads it at
    import time and may reject local file paths.  LightRAG falls back to
    config.ini when the env var is absent.
    """
    config_path = Path.cwd() / "config.ini"
    parser = ConfigParser()
    if config_path.exists():
        parser.read(config_path, encoding="utf-8")
    if not parser.has_section("milvus"):
        parser.add_section("milvus")
    parser.set("milvus", "uri", settings.milvus_db_path)
    if settings.milvus_db_name.strip():
        parser.set("milvus", "db_name", settings.milvus_db_name.strip())
    elif parser.has_option("milvus", "db_name"):
        parser.remove_option("milvus", "db_name")
    with config_path.open("w", encoding="utf-8") as fh:
        parser.write(fh)
    return config_path


def _patch_raganything_vlm_query_guard() -> None:
    """Guard RAGAnything's VLM prompt processing against None prompts.

    Some LightRAG query paths can return None for only_need_prompt=True. The
    upstream VLM-enhanced flow assumes a string and crashes in regex matching.
    Returning ("", 0) makes the caller fall back to the normal text query path.
    """
    import raganything.query as rag_query

    query_cls = rag_query.QueryMixin
    if getattr(query_cls, "_rag_anything_uit_vlm_guard_patched", False):
        return

    original_process_image_paths = query_cls._process_image_paths_for_vlm

    async def _patched_process_image_paths_for_vlm(self, prompt, extra_safe_dirs=None):
        if not isinstance(prompt, str):
            self.logger.warning(
                "VLM-enhanced query received a non-string prompt (%s); falling back to normal query",
                type(prompt).__name__,
            )
            return "", 0
        return await original_process_image_paths(
            self, prompt, extra_safe_dirs=extra_safe_dirs
        )

    query_cls._process_image_paths_for_vlm = _patched_process_image_paths_for_vlm
    query_cls._rag_anything_uit_vlm_guard_patched = True


class RAGService:
    """Creates and manages a RAGAnything instance."""

    @staticmethod
    async def create(
        settings: Settings,
        llm: LLMAdapter,
        emb: EmbeddingAdapter,
    ) -> RAGAnything:
        os.environ.setdefault("SUMMARY_LANGUAGE", settings.summary_language)

        if settings.summary_language.lower() == "vietnamese":
            from rag_app.prompts_vi import PROMPTS_VI

            register_prompt_language("vi", PROMPTS_VI)
            set_prompt_language("vi")
            log.info("Vietnamese prompts activated")

        async def llm_func(prompt: str, **kwargs: object) -> str:
            return await llm.chat(
                prompt,
                system_prompt=kwargs.get("system_prompt"),
                history_messages=kwargs.get("history_messages"),
            )

        async def embed_func(texts: list[str]) -> np.ndarray:
            result = await emb.embed_texts(texts)
            return np.array(result)

        if settings.vector_storage == "MilvusVectorDBStorage":
            STORAGE_ENV_REQUIREMENTS["MilvusVectorDBStorage"] = ["MILVUS_DB_NAME"]
            os.environ.pop("MILVUS_URI", None)
            os.environ["MILVUS_DB_NAME"] = settings.milvus_db_name.strip()
            config_path = _write_lightrag_milvus_config(settings)
            log.info(
                "Vector storage: MilvusVectorDBStorage (uri=%s, config=%s)",
                settings.milvus_db_path,
                config_path,
            )
        else:
            log.info(
                "Vector storage: %s (working_dir=%s)",
                settings.vector_storage,
                settings.rag_working_dir,
            )

        _patch_raganything_vlm_query_guard()

        lightrag_kwargs: dict = {
            "vector_storage": settings.vector_storage,
        }

        if settings.reranker_enabled:
            api_key = settings.openrouter_api_key
            model = settings.reranker_model

            async def rerank_func(query, documents, top_n=None, **kwargs):
                return await generic_rerank_api(
                    query=query,
                    documents=documents,
                    model=model,
                    base_url="https://openrouter.ai/api/v1/rerank",
                    api_key=api_key,
                    top_n=top_n,
                )

            lightrag_kwargs["rerank_model_func"] = rerank_func
            log.info("Reranker enabled: model=%s", model)

        config = RAGAnythingConfig(
            working_dir=settings.rag_working_dir,
            enable_image_processing=settings.enable_image_processing,
            enable_table_processing=settings.enable_table_processing,
            enable_equation_processing=settings.enable_equation_processing,
            parser=settings.rag_parser,
            parse_method=settings.parse_method,
            context_window=settings.context_window,
            context_mode=settings.context_mode,
            max_context_tokens=settings.max_context_tokens,
        )
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=llm.chat_with_image,
            embedding_func=EmbeddingFunc(
                embedding_dim=settings.embed_dim,
                max_token_size=8192,
                func=embed_func,
            ),
            lightrag_kwargs=lightrag_kwargs,
        )

        # Skip the subprocess parser check — it fails when the venv bin/
        # directory is not on PATH (e.g. running without `source activate`).
        # The parser is only needed for full document ingestion, not reingest
        # or query, and its availability is validated at parse time anyway.
        rag._parser_installation_checked = True
        await rag._ensure_lightrag_initialized()
        log.info("RAGAnything initialized (working_dir=%s)", settings.rag_working_dir)
        return rag
