"""
Gemini provider — wraps google.generativeai.
Extracted from the original layer4_reasoning.py.
"""
from __future__ import annotations
from typing import Callable, List, Optional

import google.generativeai as genai

from cortana import config
from cortana.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self) -> None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=config.GEMINI_API_KEY)

    def think(
        self,
        messages: List[dict],
        system: str,
        max_tokens: int = 4096,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        model = genai.GenerativeModel(
            model_name=config.MAIN_MODEL,
            system_instruction=system,
        )
        gemini_msgs = self._to_gemini(messages)
        response = model.generate_content(
            contents=gemini_msgs,
            stream=True,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
        )
        full = ""
        for chunk in response:
            text = chunk.text
            full += text
            if on_chunk:
                on_chunk(text)
        return full

    def think_simple(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        model = genai.GenerativeModel(
            model_name=config.SUB_AGENT_MODEL,
            system_instruction=system or "You are a helpful AI assistant.",
        )
        resp = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
        )
        return resp.text

    @staticmethod
    def _to_gemini(messages: List[dict]) -> List[dict]:
        role_map = {"user": "user", "assistant": "model", "model": "model"}
        return [
            {"role": role_map.get(m["role"], m["role"]), "parts": [{"text": m["content"]}]}
            for m in messages
        ]
