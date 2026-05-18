from __future__ import annotations


class RAGAppError(Exception):
    """Base exception for all RAG application errors."""


class ConfigurationError(RAGAppError):
    """Invalid or missing configuration."""


class AdapterError(RAGAppError):
    """Error communicating with an external API."""


class RetryExhaustedError(AdapterError):
    """All retry attempts failed."""

    def __init__(self, attempts: int, last_error: Exception | None = None):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"All {attempts} retry attempts exhausted"
            + (f": {last_error}" if last_error else "")
        )


class VectorStoreError(RAGAppError):
    """Error in vector store operations."""


class IngestionError(RAGAppError):
    """Error during document ingestion."""
