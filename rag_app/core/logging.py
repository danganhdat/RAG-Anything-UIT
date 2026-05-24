from __future__ import annotations

import logging
import sys


class _MilvusLiteGrpcNoiseFilter(logging.Filter):
    """Suppress known false-positive gRPC noise from Milvus Lite."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "grpc._server":
            return True
        if record.getMessage() != "Exception calling application: Method not implemented!":
            return True
        if not record.exc_info:
            return False
        exc_type, exc, _ = record.exc_info
        return not (
            exc_type is NotImplementedError
            and str(exc) == "Method not implemented!"
        )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_MilvusLiteGrpcNoiseFilter())
