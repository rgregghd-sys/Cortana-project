"""
OpenRouter provider — single API key, many free models.
Free models end with `:free` suffix (e.g. meta-llama/llama-3.1-8b-instruct:free).
Uses the OpenAI-compatible REST API at https://openrouter.ai/api/v1
Install: pip install openai   (openai SDK works with OpenRouter)
"""
from __future__ import annotations
from typing import Callable, List, Optional

from cortana import config
from cortana.providers.base import BaseProvider


class OpenRouterProvider(BaseProvider):
    name = "openrouter"

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self) -> None:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set in .env")
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=self._BASE_URL,
            )
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

    def think(
        self,
        messages: List[dict],
        system: str,
        max_tokens: int = 4096,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        or_msgs = [{"role": "system", "content": system}] + self._normalize(messages)
        stream = self._client.chat.completions.create(
            model=config.OPENROUTER_MAIN_MODEL,
            messages=or_msgs,
            max_tokens=max_tokens,
            stream=True,
            extra_headers={"HTTP-Referer": "https://cortana-ai.local"},
        )
        full = ""
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            full += text
            if on_chunk and text:
                on_chunk(text)
        return full

    def think_simple(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=config.OPENROUTER_SUB_MODEL,
            messages=msgs,
            max_tokens=max_tokens,
            stream=False,
            extra_headers={"HTTP-Referer": "https://cortana-ai.local"},
        )
        return resp.choices[0].message.content or ""

    def think_vision(
        self,
        image_b64: str,
        question: str,
        system: str = "",
        max_tokens: int = 512,
    ) -> str:
        """Vision call using OpenRouter's free vision model."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": question},
            ],
        })
        resp = self._client.chat.completions.create(
            model=config.OPENROUTER_VISION_MODEL,
            messages=msgs,
            max_tokens=max_tokens,
            stream=False,
            extra_headers={"HTTP-Referer": "https://cortana-ai.local"},
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _normalize(messages: List[dict]) -> List[dict]:
        role_map = {"model": "assistant", "user": "user", "assistant": "assistant"}
        return [{"role": role_map.get(m["role"], m["role"]), "content": m["content"]} for m in messages]
