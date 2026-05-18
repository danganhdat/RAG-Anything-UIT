from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_app.api.routes import router
from rag_app.core.config import Settings, get_settings
from rag_app.services.container import ServiceContainer

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    container = ServiceContainer(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await container.startup()
        yield
        await container.shutdown()

    app = FastAPI(
        title="RAG-Anything API",
        description="Multimodal RAG powered by Qwen and Milvus",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.container = container

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app
