"""
Abstract base class for all LLM providers.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List, Optional


class BaseProvider(ABC):
    """
    Common interface every provider must implement.
    Layer 4 talks to this interface — never to a concrete provider directly.
    """

    name: str = "base"

    @abstractmethod
    def think(
        self,
        messages: List[dict],
        system: str,
        max_tokens: int = 4096,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Streaming-capable multi-turn call.
        messages: [{"role": "user"|"model"|"assistant", "content": "..."}]
        system: system / identity prompt
        on_chunk: if provided, called with each streamed token
        Returns the full response string.
        """

    @abstractmethod
    def think_simple(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """
        Lightweight single-turn call (no streaming).
        Used by sub-agents, reflection, security layers.
        """
