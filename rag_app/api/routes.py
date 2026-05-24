from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from raganything import RAGAnything

from rag_app.api.deps import get_container, get_rag
from rag_app.services.container import ServiceContainer

log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "hybrid"
    top_k: int | None = None
    response_type: str = "Multiple Paragraphs"
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    only_need_context: bool = False


class QueryResponse(BaseModel):
    answer: str
    elapsed_seconds: float
    mode: str


class IngestRequest(BaseModel):
    file_path: str
    start_page: int | None = None
    end_page: int | None = None


class IngestResponse(BaseModel):
    status: str
    file_name: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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
    container: ServiceContainer = Depends(get_container),
) -> QueryResponse:
    start = time.monotonic()
    log.info("Query [%s]: %s", request.mode, request.query[:200])

    kwargs: dict = {"response_type": request.response_type}
    if request.top_k is not None:
        kwargs["top_k"] = request.top_k
    if request.conversation_history:
        kwargs["conversation_history"] = request.conversation_history
    if request.only_need_context:
        kwargs["only_need_context"] = request.only_need_context

    try:
        answer = await rag.aquery(
            request.query,
            mode=request.mode,
            user_prompt=container.settings.query_user_prompt or None,
            **kwargs,
        )
    except Exception as exc:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.monotonic() - start, 2)
    log.info("Response generated in %.2fs", elapsed)
    return QueryResponse(answer=answer, elapsed_seconds=elapsed, mode=request.mode)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    container: ServiceContainer = Depends(get_container),
) -> IngestResponse:
    svc = container.get_ingestion_service()
    path = Path(request.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    start = time.monotonic()
    log.info("Ingesting: %s", path.name)

    try:
        await svc.ingest_document(
            path, start_page=request.start_page, end_page=request.end_page,
        )
    except Exception as exc:
        log.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.monotonic() - start, 2)
    log.info("Ingested %s in %.2fs", path.name, elapsed)
    return IngestResponse(status="ok", file_name=path.name, elapsed_seconds=elapsed)


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    start_page: int | None = Form(None),
    end_page: int | None = Form(None),
    container: ServiceContainer = Depends(get_container),
) -> IngestResponse:
    suffix = Path(file.filename or "upload").suffix or ".pdf"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    start = time.monotonic()
    log.info("Ingesting upload: %s (%d bytes)", file.filename, len(content))

    try:
        svc = container.get_ingestion_service()
        await svc.ingest_document(
            tmp_path, start_page=start_page, end_page=end_page,
        )
    except Exception as exc:
        log.exception("Upload ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    elapsed = round(time.monotonic() - start, 2)
    return IngestResponse(
        status="ok", file_name=file.filename or "upload", elapsed_seconds=elapsed,
    )


@router.get("/system/info")
async def system_info(rag: RAGAnything = Depends(get_rag)) -> dict:
    return {
        "config": rag.get_config_info(),
        "processors": rag.get_processor_info(),
    }
