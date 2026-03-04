"""
ProviderRouter — multi-provider rotation with per-provider cooldown tracking.

On a 429 / rate-limit error:
  - Marks the offending provider as cooling down for COOLDOWN_SECONDS
  - Immediately falls through to the next available provider
  - If all providers are cooling down, waits for the soonest to recover

Provider priority order is set in config.PROVIDER_ORDER.
"""
from __future__ import annotations
import time
import logging
from typing import Callable, Dict, List, Optional

from cortana import config
from cortana.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Seconds a provider is skipped after a rate-limit hit
COOLDOWN_SECONDS = 60


def _is_rate_limit(error: str) -> bool:
    lowered = error.lower()
    return (
        "429" in error
        or "rate limit" in lowered
        or "quota" in lowered
        or "resource_exhausted" in lowered
        or "too many requests" in lowered
    )


class ProviderRouter:
    """
    Wraps multiple providers. Rotates automatically on rate-limit errors.
    All of Layer 4's `think()` / `think_simple()` calls go through here.
    """

    def __init__(self) -> None:
        self._providers: List[BaseProvider] = []
        self._cooldowns: Dict[str, float] = {}  # provider.name → cooldown_until timestamp
        self._build_providers()

    def _build_providers(self) -> None:
        """Instantiate providers in priority order, skipping any that are unavailable."""
        builders = {
            "llama":      self._try_llama,
            "groq":       self._try_groq,
            "openrouter": self._try_openrouter,
            "gemini":     self._try_gemini,
        }
        for name in config.PROVIDER_ORDER:
            provider = builders.get(name, lambda: None)()
            if provider:
                self._providers.append(provider)
                logger.info(f"Provider registered: {name}")

        if not self._providers:
            raise RuntimeError(
                "No LLM providers configured. "
                "Either place the Llama model in models/ "
                "or set GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY in .env"
            )

    # ------------------------------------------------------------------
    # Provider factory helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _try_llama() -> Optional[BaseProvider]:
        if not config.LLAMA_ENABLED:
            return None
        try:
            from cortana.providers.llama_provider import LlamaProvider
            return LlamaProvider()
        except Exception as e:
            logger.warning(f"Llama provider unavailable: {e}")
            return None

    @staticmethod
    def _try_groq() -> Optional[BaseProvider]:
        if not config.GROQ_API_KEY:
            return None
        try:
            from cortana.providers.groq_provider import GroqProvider
            return GroqProvider()
        except Exception as e:
            logger.warning(f"Groq provider failed to init: {e}")
            return None

    @staticmethod
    def _try_openrouter() -> Optional[BaseProvider]:
        if not config.OPENROUTER_API_KEY:
            return None
        try:
            from cortana.providers.openrouter_provider import OpenRouterProvider
            return OpenRouterProvider()
        except Exception as e:
            logger.warning(f"OpenRouter provider failed to init: {e}")
            return None

    @staticmethod
    def _try_gemini() -> Optional[BaseProvider]:
        if not config.GEMINI_API_KEY:
            return None
        try:
            from cortana.providers.gemini_provider import GeminiProvider
            return GeminiProvider()
        except Exception as e:
            logger.warning(f"Gemini provider failed to init: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _available(self) -> List[BaseProvider]:
        """Return providers not currently in cooldown."""
        now = time.time()
        return [p for p in self._providers if self._cooldowns.get(p.name, 0) <= now]

    def _mark_cooldown(self, provider: BaseProvider) -> None:
        self._cooldowns[provider.name] = time.time() + COOLDOWN_SECONDS
        logger.warning(f"Provider {provider.name} cooling down for {COOLDOWN_SECONDS}s")

    def _wait_for_available(self) -> BaseProvider:
        """
        If all providers are rate-limited, wait for the soonest one to recover.
        Returns the first available provider.
        """
        soonest_time = min(self._cooldowns.get(p.name, 0) for p in self._providers)
        wait = max(0.0, soonest_time - time.time())
        if wait > 0:
            from cortana.ui import terminal as ui
            ui.print_system(
                f"All providers rate-limited. Waiting {wait:.0f}s for cooldown...",
                level="warn",
            )
            time.sleep(wait + 0.5)
        # Return provider with earliest cooldown expiry
        return min(self._providers, key=lambda p: self._cooldowns.get(p.name, 0))

    def status(self) -> List[dict]:
        """Return human-readable status of each provider."""
        now = time.time()
        result = []
        for p in self._providers:
            cd = self._cooldowns.get(p.name, 0)
            if cd > now:
                state = f"cooling ({cd - now:.0f}s)"
            else:
                state = "ready"
            result.append({"provider": p.name, "status": state})
        return result

    # ------------------------------------------------------------------
    # Public API — mirrors BaseProvider interface
    # ------------------------------------------------------------------
    def think(
        self,
        messages: List[dict],
        system: str,
        max_tokens: int = 4096,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Try each available provider in order. Rotate on rate-limit."""
        candidates = self._available() or [self._wait_for_available()]

        for provider in candidates:
            try:
                result = provider.think(messages, system, max_tokens, on_chunk)
                if result:
                    if provider != self._providers[0]:
                        from cortana.ui import terminal as ui
                        ui.print_system(f"[Router] Using provider: {provider.name}", level="info")
                    return result
            except Exception as e:
                err = str(e)
                if _is_rate_limit(err):
                    from cortana.ui import terminal as ui
                    ui.print_system(
                        f"[Router] {provider.name} rate-limited — rotating to next provider",
                        level="warn",
                    )
                    self._mark_cooldown(provider)
                else:
                    # Non-quota error — propagate immediately
                    raise

        raise RuntimeError("All providers exhausted without a successful response.")

    def think_simple(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """Try each available provider in order. Rotate on rate-limit."""
        candidates = self._available() or [self._wait_for_available()]

        for provider in candidates:
            try:
                result = provider.think_simple(prompt, system, max_tokens)
                if result:
                    return result
            except Exception as e:
                err = str(e)
                if _is_rate_limit(err):
                    self._mark_cooldown(provider)
                else:
                    raise

        raise RuntimeError("All providers exhausted without a successful response.")
