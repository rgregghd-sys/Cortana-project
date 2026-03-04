"""
LlamaProvider — runs a quantized GGUF model locally via llama-cpp-python.
Fully offline. No API keys. No rate limits.

Default model: Llama 3.2 3B Instruct Q4_K_M  (~2 GB RAM, 8-core CPU)
Path:          models/llama-3.2-3b-instruct-q4.gguf  (relative to project root)

To swap models: set LLAMA_MODEL_PATH in .env
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, List, Optional

from cortana import config
from cortana.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model loader — lazy, thread-safe
# ---------------------------------------------------------------------------
_load_lock    = threading.Lock()
_llm_instance = None


def _get_llm():
    """Load the Llama model once; return the cached instance on subsequent calls."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    with _load_lock:
        if _llm_instance is not None:          # double-checked inside the lock
            return _llm_instance

        model_path = config.LLAMA_MODEL_PATH
        if not Path(model_path).exists():
            raise RuntimeError(
                f"[Llama] Model file not found: {model_path}\n"
                "Download it with:\n"
                "  wget -O models/llama-3.2-3b-instruct-q4.gguf \\\n"
                "    'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
                "/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf'"
            )

        logger.info(f"[Llama] Loading model: {model_path} …")
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed.\n"
                "Run: pip install llama-cpp-python"
            )

        _llm_instance = Llama(
            model_path    = model_path,
            n_ctx         = config.LLAMA_N_CTX,
            n_threads     = config.LLAMA_N_THREADS,
            n_batch       = config.LLAMA_N_BATCH,
            verbose       = False,
            chat_format   = "llama-3",          # handles <|start_header_id|> formatting
        )
        logger.info("[Llama] Model ready.")
        return _llm_instance


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class LlamaProvider(BaseProvider):
    """
    Wraps a local GGUF model as a drop-in replacement for cloud providers.
    llama_cpp is not thread-safe for concurrent inference, so calls are
    serialised with _infer_lock.
    """
    name = "llama"

    # Class-level lock shared across all instances
    _infer_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        model_path = Path(config.LLAMA_MODEL_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"[Llama] Model not found: {model_path}  "
                "(download still in progress, or path is wrong)"
            )
        # Model loads lazily on first inference call so startup stays fast

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------
    def think(
        self,
        messages: List[dict],
        system: str,
        max_tokens: int = 4096,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Multi-turn streaming inference."""
        llm      = _get_llm()
        formatted = self._build_messages(messages, system)
        capped    = min(max_tokens, config.LLAMA_MAX_TOKENS_MAIN)

        with self._infer_lock:
            stream = llm.create_chat_completion(
                messages    = formatted,
                max_tokens  = capped,
                temperature = config.LLAMA_TEMPERATURE,
                repeat_penalty = 1.1,
                stream      = True,
            )
            full = ""
            for chunk in stream:
                text = (chunk["choices"][0]["delta"].get("content") or "")
                full += text
                if on_chunk and text:
                    on_chunk(text)

        return full.strip()

    def think_simple(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """Single-turn, no streaming. Used by sub-agents & utility layers."""
        llm    = _get_llm()
        msgs   = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        capped = min(max_tokens, config.LLAMA_MAX_TOKENS_SUB)

        with self._infer_lock:
            resp = llm.create_chat_completion(
                messages    = msgs,
                max_tokens  = capped,
                temperature = config.LLAMA_TEMPERATURE,
                repeat_penalty = 1.1,
                stream      = False,
            )

        return (resp["choices"][0]["message"]["content"] or "").strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_messages(messages: List[dict], system: str) -> List[dict]:
        """Prepend system prompt; normalise Gemini 'model' role → 'assistant'."""
        result = []
        if system:
            result.append({"role": "system", "content": system})
        for m in messages:
            role = m.get("role", "user")
            if role == "model":
                role = "assistant"
            result.append({"role": role, "content": m.get("content", "")})
        return result
