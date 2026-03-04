"""
Search event signaling via ContextVar.
Allows layer8_tools.web_search() to notify the UI of live search activity
without threading callbacks through every layer.
Usage:
  - Set callback before pipeline runs: set_search_callback(cb)
  - web_search emits: emit("search_start", query=...) / emit("search_done", ...)
"""
from __future__ import annotations
import contextvars
from typing import Any, Callable, Optional

_search_callback: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar(
    "cortana_search_callback", default=None
)


def set_search_callback(cb: Optional[Callable]) -> None:
    """Bind a search event callback to the current async context."""
    _search_callback.set(cb)


def emit(event_type: str, **data: Any) -> None:
    """Fire a search event if a callback is registered in this context."""
    cb = _search_callback.get(None)
    if cb:
        try:
            cb(event_type, data)
        except Exception:
            pass
