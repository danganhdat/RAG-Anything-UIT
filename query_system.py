import argparse
import asyncio
import logging

from rag_app.core.config import get_settings
from rag_app.core.logging import setup_logging
from rag_app.services.container import ServiceContainer

log = logging.getLogger(__name__)


async def main(args: argparse.Namespace) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    container = ServiceContainer(settings)
    await container.startup()

    try:
        kwargs: dict = {"response_type": args.response_type}
        if args.top_k is not None:
            kwargs["top_k"] = args.top_k
        if args.context_only:
            kwargs["only_need_context"] = True

        log.info("Query [%s]: %s", args.mode, args.query)
        answer = await container.rag.aquery(
            args.query,
            mode=args.mode,
            user_prompt=settings.query_user_prompt or None,
            **kwargs,
        )
        print(f"\n{answer}")
    finally:
        await container.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-Anything CLI query")
    parser.add_argument(
        "query",
        nargs="?",
        default="Điều kiện tuyển sinh sau đại học năm 2025 là gì?",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--response-type",
        default="Multiple Paragraphs",
        choices=["Multiple Paragraphs", "Single Paragraph", "Bullet Points"],
    )
    parser.add_argument("--context-only", action="store_true")

    asyncio.run(main(parser.parse_args()))
