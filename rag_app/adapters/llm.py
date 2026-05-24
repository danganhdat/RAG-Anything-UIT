from __future__ import annotations

import base64
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

from rag_app.adapters.base import OPENROUTER_BASE_URL, BaseOpenRouterClient
from rag_app.core.config import Settings

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _detect_mime(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def _extract_content(data: dict) -> str:
    """Extract text content from an OpenRouter response, handling thinking models."""
    msg = data["choices"][0]["message"]
    content = msg.get("content")
    if not content:
        for fallback_key in ("reasoning_content", "reasoning"):
            if msg.get(fallback_key):
                content = msg[fallback_key]
                break
    if not content:
        return ""
    return _THINK_RE.sub("", content).strip()


class LLMAdapter(BaseOpenRouterClient):
    """OpenRouter chat completions (text + vision)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._chat_url = f"{OPENROUTER_BASE_URL}/chat/completions"

    async def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.5,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._settings.llm_text_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = await self._post_with_retry(self._chat_url, payload)
        return _extract_content(data)

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
            # No image — fall back to text-only chat (used for table/equation analysis)
            return await self.chat(
                prompt, system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature,
            )

        payload: dict[str, Any] = {
            "model": self._settings.llm_vlm_model,
            "messages": built_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = await self._post_with_retry(self._chat_url, payload)
        return _extract_content(data)

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
