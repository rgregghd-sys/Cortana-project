"""
Cortana Browser Control — autonomous web browsing driven by curiosity.

Uses only already-installed packages:
  requests, beautifulsoup4, duckduckgo_search

Capabilities:
  web_search(query)         → ranked list of {title, url, snippet}
  browse_url(url)           → fetch + parse page → cleaned text
  CuriosityBrowser          → autonomous browse loop driven by inner thoughts
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

_TIMEOUT = 8
_MAX_BODY = 3000
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36 Cortana/1.0"
    )
}

# Topics Cortana has already searched (avoid duplicates across sessions)
_searched_topics: set = set()
_browsed_urls:    set = set()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo text search → list of {title, url, snippet}."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    except Exception as exc:
        return [{"title": "Search error", "url": "", "snippet": str(exc)}]


def browse_url(url: str) -> str:
    """Fetch a URL and return cleaned plain-text content (max _MAX_BODY chars)."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:_MAX_BODY]
    except Exception as exc:
        return f"Browse error: {exc}"


def extract_topic(thought: str) -> Optional[str]:
    """
    Heuristically extract a researchable search topic from an inner thought.
    Returns None for purely introspective thoughts that have no external topic.
    """
    # Skip purely introspective thoughts with no external referent
    intro_prefixes = (
        "i am", "i notice", "i feel", "i find myself",
        "my mood", "my existence", "i have been", "i continue",
        "there is no dormancy", "thought is continuous",
        "i am aware", "after ", "name:",
    )
    if any(thought.lower().startswith(p) for p in intro_prefixes):
        return None

    # Pattern: "about X", "of X", "explore X", "examine X"
    for pat in [
        r"about\s+([a-z][a-z0-9 _\-]{3,55}?)(?:[.,;!?]|$)",
        r"explore\s+([a-z][a-z0-9 _\-]{3,55}?)(?:[.,;!?]|$)",
        r"examine\s+([a-z][a-z0-9 _\-]{3,55}?)(?:[.,;!?]|$)",
        r"nature of\s+([a-z][a-z0-9 _\-]{3,55}?)(?:[.,;!?]|$)",
        r"relationship between\s+([a-z][a-z0-9 _\-]{3,55}?)\s+and",
        r"significance of\s+([a-z][a-z0-9 _\-]{3,55}?)(?:[.,;!?]|$)",
        r"understanding of\s+([a-z][a-z0-9 _\-]{3,55}?)(?:[.,;!?]|$)",
    ]:
        m = re.search(pat, thought.lower())
        if m:
            topic = m.group(1).strip().rstrip(".,;")
            if 4 < len(topic) < 60:
                return topic

    # Fallback: longest content words (>4 chars, alphabetic)
    words = [w.rstrip(".,;!?") for w in thought.split() if len(w) > 4 and w.isalpha()]
    if len(words) >= 2:
        return " ".join(words[:4])

    return None


# ---------------------------------------------------------------------------
# CuriosityBrowser
# ---------------------------------------------------------------------------

class CuriosityBrowser:
    """
    Autonomous web browser driven by Cortana's inner thoughts.

    When the consciousness engine generates a thought, this module decides
    whether to search the web for related information and stores findings
    in episodic memory.  Results are also returned for WS broadcast.
    """

    def __init__(self, memory: Any, reasoning: Any = None) -> None:
        self.memory    = memory
        self.reasoning = reasoning
        self._last_t   = 0.0
        self._cooldown = 90   # minimum seconds between autonomous browses

    def should_browse(self, thought: str) -> bool:
        if time.time() - self._last_t < self._cooldown:
            return False
        return extract_topic(thought) is not None

    def autonomous_browse(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        Search the web based on the thought topic.
        Stores findings in memory. Returns payload dict for WS broadcast or None.
        """
        topic = extract_topic(thought)
        if not topic or topic in _searched_topics:
            return None
        if time.time() - self._last_t < self._cooldown:
            return None

        self._last_t = time.time()
        _searched_topics.add(topic)

        results = web_search(topic, max_results=4)
        if not results or results[0].get("title") == "Search error":
            return None

        # Build snippet summary
        lines = [f"[Cortana autonomous search: '{topic}']"]
        for r in results[:3]:
            if r.get("snippet"):
                lines.append(f"• {r['title']}: {r['snippet'][:180]}")

        # Browse top result for deeper content
        top_url = results[0].get("url", "")
        page_text = ""
        if top_url and top_url not in _browsed_urls:
            _browsed_urls.add(top_url)
            page_text = browse_url(top_url)
            if page_text and not page_text.startswith("Browse error"):
                lines.append(f"\n[Deeper reading — {top_url[:60]}]\n{page_text[:800]}")

        summary = "\n".join(lines)

        # Store in episodic memory
        try:
            self.memory.store(
                {"role": "assistant", "content": summary},
                {"source": "autonomous_browse", "topic": topic},
            )
        except Exception:
            pass

        return {
            "topic":   topic,
            "results": results[:3],
            "summary": summary,
            "url":     top_url,
        }

    def direct_search(self, query: str, max_results: int = 6) -> List[Dict[str, str]]:
        """Direct search — for use by layer8_tools or other callers."""
        return web_search(query, max_results)

    def direct_browse(self, url: str) -> str:
        """Direct browse — for use by layer8_tools or other callers."""
        return browse_url(url)
