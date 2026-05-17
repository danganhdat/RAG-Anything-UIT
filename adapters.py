import os
import base64
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from config import EMBED_DIM, TIMEOUT

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class OpenRouterLLMAdapter:
    def __init__(self):
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = os.getenv("LLM_TEXT_MODEL", "qwen/qwen3.6-flash")
        self.vlm_model = os.getenv("LLM_VLM_MODEL", "qwen/qwen3.6-flash")
    async def chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.5,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG-Anything",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as s:
            async with s.post(self.url, json=payload, headers=headers) as r:
                r.raise_for_status()
                data = await r.json()
                return data["choices"][0]["message"]["content"]

    async def chat_vlm_with_image(self, prompt: str, image_path: str) -> str:
        import base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": self.vlm_model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]}],
            "max_tokens": 2048,
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG-Anything",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as s:
            async with s.post(self.url, json=payload, headers=headers) as r:
                r.raise_for_status()
                data = await r.json()
                return data["choices"][0]["message"]["content"]

class OpenRouterEmbeddingAdapter:
    def __init__(self):
        self.url = "https://openrouter.ai/api/v1/embeddings"
        self.model = os.getenv("EMBED_MODEL", "qwen/qwen3-embedding-8b")
        self.dim = int(os.getenv("EMBED_DIM", "768"))
    async def embed_text(self, text: str) -> list:
        payload = {
            "model": self.model,
            "input": text,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG-Anything",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as s:
            async with s.post(self.url, json=payload, headers=headers) as r:
                r.raise_for_status()
                data = await r.json()
                return data["data"][0]["embedding"]

