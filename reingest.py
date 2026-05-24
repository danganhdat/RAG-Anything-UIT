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

CONTENT_LIST = Path(
    "output/cam_nang_sau_dai_hoc_2025_0_725c0097/"
    "cam_nang_sau_dai_hoc_2025_0/hybrid_auto/"
    "cam_nang_sau_dai_hoc_2025_0_content_list.json"
)
IMAGES_DIR = CONTENT_LIST.parent / "images"
FILE_REF = "cam_nang_sau_dai_hoc_2025_0.pdf"


def clear_workdir(workdir: str) -> None:
    """Remove all files in rag_workdir so LightRAG starts fresh."""
    p = Path(workdir)
    if p.exists():
        for item in p.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        log.info("Cleared %s", workdir)


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

    container = ServiceContainer(settings)
    await container.startup()

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
