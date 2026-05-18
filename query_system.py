import asyncio
import logging
import sys

# from lightrag.base import QueryParam

from rag_app.core.config import get_settings
from rag_app.core.logging import setup_logging
from rag_app.services.container import ServiceContainer

log = logging.getLogger(__name__)


async def main(query_text: str) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    container = ServiceContainer(settings)
    await container.startup()

    try:
        log.info("Query: %s", query_text)
        # answer = await container.rag.aquery(query_text, param=QueryParam(mode="hybrid"))
        answer = await container.rag.aquery(query_text, mode="hybrid") 
        print(f"\n{answer}")
    finally:
        await container.shutdown()


if __name__ == "__main__":
    user_query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Điều kiện tuyển sinh sau đại học năm 2025 là gì?"
    )
    asyncio.run(main(user_query))
