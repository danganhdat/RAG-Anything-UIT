from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

from rag_app.adapters.base import OPENROUTER_BASE_URL, BaseOpenRouterClient
from rag_app.core.config import Settings

log = logging.getLogger(__name__)


def _detect_mime(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


class LLMAdapter(BaseOpenRouterClient):
    """OpenRouter chat completions (text + vision)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._chat_url = f"{OPENROUTER_BASE_URL}/chat/completions"

    async def chat(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.5,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._settings.llm_text_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = await self._post_with_retry(self._chat_url, payload)
        return data["choices"][0]["message"]["content"]

    async def chat_with_image(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        *,
        image_data: str | None = None,
        system_prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        if messages:
            built_messages = messages
        elif image_data:
            built_messages = self._build_messages(prompt, image_data, "image/png", system_prompt)
        elif image_path:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            built_messages = self._build_messages(prompt, b64, _detect_mime(image_path), system_prompt)
        else:
            raise ValueError("Either image_path, image_data, or messages must be provided")

        payload: dict[str, Any] = {
            "model": self._settings.llm_vlm_model,
            "messages": built_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = await self._post_with_retry(self._chat_url, payload)
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _build_messages(
        prompt: str, b64: str, mime: str, system_prompt: str | None
    ) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        })
        return msgs
