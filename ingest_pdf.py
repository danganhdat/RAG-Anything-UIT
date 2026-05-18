import argparse
import asyncio
import logging
from pathlib import Path

from rag_app.core.config import get_settings
from rag_app.core.logging import setup_logging
from rag_app.services.container import ServiceContainer

log = logging.getLogger(__name__)

DEFAULT_PDF = Path("sample/docs/cam_nang_sau_dai_hoc_2025_0.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a PDF into RAG-Anything")
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF, help="Path to PDF")
    parser.add_argument("--start-page", type=int, default=None, help="Start page (0-based)")
    parser.add_argument("--end-page", type=int, default=None, help="End page (0-based, exclusive)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    container = ServiceContainer(settings)
    await container.startup()

    try:
        svc = container.get_ingestion_service()
        await svc.ingest_pdf(args.pdf, start_page=args.start_page, end_page=args.end_page)
    except Exception:
        log.exception("Ingestion failed")
    finally:
        await container.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
