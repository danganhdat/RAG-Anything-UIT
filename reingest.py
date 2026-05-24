"""Re-ingest from pre-parsed content_list.json (no GPU required).

Clears rag_workdir before ingesting to avoid duplicate document errors.
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path

from rag_app.core.config import get_settings
from rag_app.core.logging import setup_logging
from rag_app.services.container import ServiceContainer

log = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent

CONTENT_LIST = _SCRIPT_DIR / Path(
    "output/cam_nang_sau_dai_hoc_2025_0_725c0097/"
    "cam_nang_sau_dai_hoc_2025_0/hybrid_auto/"
    "cam_nang_sau_dai_hoc_2025_0_content_list.json"
)
IMAGES_DIR = CONTENT_LIST.parent / "images"
FILE_REF = "cam_nang_sau_dai_hoc_2025_0.pdf"


_CACHE_FILES = {"kv_store_llm_response_cache.json"}


def clear_workdir(workdir: str, keep_cache: bool = True) -> None:
    """Remove files in rag_workdir so LightRAG starts fresh.

    When keep_cache is True, preserves the LLM response cache so
    re-ingestion with the same model and data skips cached LLM calls.
    """
    p = Path(workdir)
    if not p.exists():
        return
    for item in p.iterdir():
        if keep_cache and item.name in _CACHE_FILES:
            log.info("Keeping cache: %s", item.name)
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    log.info("Cleared %s (keep_cache=%s)", workdir, keep_cache)


def clear_milvus_storage(db_path: str) -> None:
    """Remove Milvus Lite storage (file or directory)."""
    p = Path(db_path)
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    log.info("Removed Milvus storage: %s", p)


def prepare_content_list(raw: list[dict]) -> list[dict]:
    """Fix relative img_path to absolute and filter unsupported types."""
    supported = {"text", "image", "table", "equation"}
    items = []
    for entry in raw:
        t = entry.get("type", "")
        if t not in supported:
            continue
        if "img_path" in entry and entry["img_path"]:
            entry["img_path"] = str(
                (IMAGES_DIR / Path(entry["img_path"]).name).resolve()
            )
        items.append(entry)
    return items


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    if not CONTENT_LIST.exists():
        log.error("Content list not found: %s", CONTENT_LIST)
        return

    raw = json.loads(CONTENT_LIST.read_text(encoding="utf-8"))
    content_list = prepare_content_list(raw)

    types: dict[str, int] = {}
    for item in content_list:
        t = item["type"]
        types[t] = types.get(t, 0) + 1
    log.info("Content list: %d items — %s", len(content_list), types)

    clear_workdir(settings.rag_working_dir)
    clear_milvus_storage(settings.milvus_db_path)

    container = ServiceContainer(settings)
    await container.startup()

    # insert_content_list writes to doc_status *before* calling
    # lightrag.ainsert, which then sees the doc as a duplicate.
    # Bypass the dedup check entirely — we just wiped the workdir,
    # so everything is new by definition.
    _orig_filter = container.rag.lightrag.doc_status.filter_keys

    async def _no_dedup(keys):
        return keys

    container.rag.lightrag.doc_status.filter_keys = _no_dedup

    try:
        await container.rag.insert_content_list(
            content_list=content_list,
            file_path=FILE_REF,
            display_stats=True,
        )
        log.info("Ingestion complete!")
    finally:
        await container.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
