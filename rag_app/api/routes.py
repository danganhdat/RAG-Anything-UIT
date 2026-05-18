from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from raganything import RAGAnything, QueryParam

from rag_app.api.deps import get_container, get_rag
from rag_app.services.container import ServiceContainer

log = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    elapsed_seconds: float


@router.get("/health")
async def health_check(container: ServiceContainer = Depends(get_container)) -> dict:
    return {
        "status": "healthy" if container.rag else "initializing",
        "rag_loaded": container.rag is not None,
    }


@router.post("/chat", response_model=QueryResponse)
async def chat(
    request: QueryRequest,
    rag: RAGAnything = Depends(get_rag),
) -> QueryResponse:
    start = time.monotonic()
    log.info("Query: %s", request.query[:200])

    try:
        answer = await rag.aquery(request.query, param=QueryParam(mode="hybrid"))
    except Exception as exc:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.monotonic() - start, 2)
    log.info("Response generated in %.2fs", elapsed)
    return QueryResponse(answer=answer, elapsed_seconds=elapsed)
