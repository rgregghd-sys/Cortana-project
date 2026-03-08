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
        system = self._build_system(identity_prompt, memories, state, perceived, conversation_history)
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
    # Tone guidance injected into context so Cortana can adapt register
    _TONE_GUIDANCE = {
        "frustrated": (
            "The user is frustrated. Acknowledge that before solving. "
            "Be direct and efficient — don't make them wade through preamble."
        ),
        "confused": (
            "The user is confused or lost. Lead with clarity. "
            "Start from a solid footing before going deep. Check your assumptions about what they know."
        ),
        "excited": (
            "The user is energized about this. Match that energy where it's genuine. "
            "Don't be flat when they're not."
        ),
        "curious": (
            "The user is in genuine exploration mode. "
            "Follow the thread with them — don't just answer, think alongside them."
        ),
        "playful": (
            "The user is being light and playful. "
            "It's fine to be a bit more relaxed in tone. Dry wit welcome."
        ),
        "neutral": "",
    }

    def _build_system(
        self,
        identity_prompt: str,
        memories: List[str],
        state: CortanaState,
        perceived: Optional[PerceivedInput] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        parts = [identity_prompt]

        # Inject emotional tone guidance if detected
        if perceived and perceived.emotional_tone != "neutral":
            guidance = self._TONE_GUIDANCE.get(perceived.emotional_tone, "")
            if guidance:
                parts.append(f"\n\n## User Tone This Turn\n{guidance}")

        # Conversation thread — build awareness of what's been established
        if conversation_history and len(conversation_history) >= 4:
            thread_ctx = self._build_thread_context(conversation_history)
            if thread_ctx:
                parts.append(f"\n\n## Conversation Thread\n{thread_ctx}")

        if memories:
            memory_block = "\n".join(f"- {m}" for m in memories)
            parts.append(
                f"\n\n## Relevant Memory\n"
                f"Semantically relevant past interactions:\n{memory_block}"
            )

        now = datetime.now()
        parts.append(
            f"\n\n## Context\n"
            f"Date/Time: {now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
            f"Turns this session: {state.interaction_count}"
        )

        return "\n".join(parts)

    @staticmethod
    def _build_thread_context(conversation_history: List[ConversationTurn]) -> str:
        """
        Summarise the active conversation thread so Cortana can reference it naturally.
        Extracts the last few user messages and any key topics established.
        No LLM call — pure heuristic to avoid latency.
        """
        user_turns = [t.content for t in conversation_history if t.role == "user"]
        if not user_turns:
            return ""

        # Take last 4 user messages for recency; truncate each
        recent = user_turns[-4:]
        snippets = [f'- "{t[:120].strip()}"' for t in recent]

        # Detect if the conversation has returned to an earlier topic
        has_callback = len(user_turns) > 4 and any(
            len(t.split()) < 8 for t in user_turns[-2:]
        )
        callback_note = (
            "\nNote: recent messages are short — this may be follow-up or clarification "
            "on a topic already established earlier in this conversation."
        ) if has_callback else ""

        return (
            "Recent user messages (most recent last):\n"
            + "\n".join(snippets)
            + callback_note
        )

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
