from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from rag_app.core.config import Settings
from rag_app.core.exceptions import RetryExhaustedError

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class BaseOpenRouterClient:
    """Shared HTTP client with session reuse, retry, and exponential backoff."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session: aiohttp.ClientSession | None = None

    # -- Session lifecycle ---------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._settings.timeout),
                headers={
                    "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "RAG-Anything",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> BaseOpenRouterClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -- Retry logic ---------------------------------------------------------

    async def _post_with_retry(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = await self._get_session()
        max_retries = self._settings.max_retries
        last_err: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status in RETRYABLE_STATUS:
                        body = await resp.text()
                        log.warning(
                            "HTTP %d (attempt %d/%d): %s",
                            resp.status, attempt, max_retries, body[:200],
                        )
                        last_err = aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=body[:200],
                        )
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue

                    resp.raise_for_status()
                    return await resp.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.warning(
                    "Request failed (attempt %d/%d): %s", attempt, max_retries, exc,
                )
                last_err = exc
                if attempt < max_retries:
                    await asyncio.sleep(min(2 ** attempt, 30))

        raise RetryExhaustedError(attempts=max_retries, last_error=last_err)
