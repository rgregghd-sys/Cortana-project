"""
Groq provider — uses the OpenAI-compatible Groq REST API.
Free tier: ~14,400 requests/day, 500k tokens/min.
Install: pip install groq
"""
from __future__ import annotations
from typing import Callable, List, Optional

from cortana import config
from cortana.providers.base import BaseProvider


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self) -> None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        try:
            from groq import Groq
            self._client = Groq(api_key=config.GROQ_API_KEY)
        except ImportError:
            raise RuntimeError("groq package not installed. Run: pip install groq")

    def think(
        self,
        messages: List[dict],
        system: str,
        max_tokens: int = 4096,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        groq_msgs = [{"role": "system", "content": system}] + self._normalize(messages)
        stream = self._client.chat.completions.create(
            model=config.GROQ_MAIN_MODEL,
            messages=groq_msgs,
            max_tokens=max_tokens,
            stream=True,
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
            model=config.GROQ_SUB_MODEL,
            messages=msgs,
            max_tokens=max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _normalize(messages: List[dict]) -> List[dict]:
        """Normalize Gemini 'model' role → 'assistant' for OpenAI-compatible APIs."""
        role_map = {"model": "assistant", "user": "user", "assistant": "assistant"}
        return [{"role": role_map.get(m["role"], m["role"]), "content": m["content"]} for m in messages]
