"""
Layer 4 — Reasoning
Multi-provider backbone. Assembles full context (identity + memory + conversation)
and calls the ProviderRouter, which rotates across Groq → OpenRouter → Gemini
automatically on rate-limit errors.
"""
from __future__ import annotations
from datetime import datetime
from typing import Callable, List, Optional

from cortana import config
from cortana.models.schemas import ConversationTurn, CortanaState, PerceivedInput
from cortana.providers.router import ProviderRouter


class ReasoningLayer:
    """
    The main brain. Delegates all LLM calls to ProviderRouter.
    Context assembly order: system prompt → memories → conversation history → user message.
    """

    def __init__(self) -> None:
        self._router = ProviderRouter()

    @property
    def router(self) -> ProviderRouter:
        return self._router

    def think(
        self,
        perceived: PerceivedInput,
        memories: List[str],
        identity_prompt: str,
        conversation_history: List[ConversationTurn],
        state: CortanaState,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Assemble context and call the router. Returns full response string.
        If on_chunk is provided, calls it for each streamed token.
        """
        messages = self._build_messages(perceived, conversation_history)
        system = self._build_system(identity_prompt, memories, state)
        return self._router.think(messages, system, max_tokens=4096, on_chunk=on_chunk)

    def think_simple(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,  # kept for API compatibility, unused by router
        max_tokens: int = 2048,
    ) -> str:
        """
        Lightweight single-turn call — used by sub-agents, reflection, security layers.
        The router picks the best available provider automatically.
        """
        return self._router.think_simple(prompt=prompt, system=system, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------
    def _build_system(
        self,
        identity_prompt: str,
        memories: List[str],
        state: CortanaState,
    ) -> str:
        parts = [identity_prompt]

        if memories:
            memory_block = "\n".join(f"- {m}" for m in memories)
            parts.append(
                f"\n\n## Relevant Memory Recall\n"
                f"The following past interactions are semantically relevant:\n{memory_block}"
            )

        now = datetime.now()
        parts.append(
            f"\n\n## Current State\n"
            f"Date/Time: {now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
            f"Interactions logged: {state.interaction_count}"
        )

        return "\n".join(parts)

    def _build_messages(
        self,
        perceived: PerceivedInput,
        conversation_history: List[ConversationTurn],
    ) -> List[dict]:
        """Build the messages array (conversation history + current user message)."""
        messages = []

        # Inject conversation history (cap at last 20 turns)
        for turn in conversation_history[-20:]:
            messages.append({"role": turn.role, "content": turn.content})

        # Current user message
        user_content = perceived.content
        if perceived.intent not in ("simple", "conversational"):
            user_content = (
                f"[Intent detected: {perceived.intent} | "
                f"Complexity: {perceived.complexity:.2f}]\n\n{perceived.content}"
            )

        messages.append({"role": "user", "content": user_content})
        return messages
