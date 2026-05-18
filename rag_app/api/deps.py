from __future__ import annotations

from fastapi import HTTPException, Request
from raganything import RAGAnything

from rag_app.services.container import ServiceContainer


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_rag(request: Request) -> RAGAnything:
    container: ServiceContainer = request.app.state.container
    if container.rag is None:
        raise HTTPException(status_code=503, detail="RAG system is not initialized yet")
    return container.rag
