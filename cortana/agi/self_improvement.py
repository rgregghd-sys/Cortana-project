"""
Recursive Self-Improvement — Cortana analyses her own AGI modules and
queues targeted improvements through the existing L11/L12 patch pipeline.

Process:
  1. Collect performance metrics from cognitive_modes, system2, world_model
  2. Identify the weakest performing module (lowest avg_quality or hit rate)
  3. Generate a targeted improvement suggestion via LLM
  4. Format as a patch proposal and write to agent_workspace/patches/
  5. L12 notifier picks it up for L1 (Identity) review

This runs during consciousness idle ticks — one improvement cycle
every RSI_COOLDOWN seconds. Never blocks the hot path.

Also implements Active Learning:
  - Detects knowledge gaps from world model query misses
  - Queues targeted research goals for the autonomy engine
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cortana import config

_DB_PATH    = config.SQLITE_PATH
_PATCH_DIR  = Path(config.AGENT_WORKSPACE) / "patches"
_RSI_COOLDOWN = 3600.0   # one improvement cycle per hour


# ---------------------------------------------------------------------------
# Performance analysis
# ---------------------------------------------------------------------------

def _get_mode_performance(db_path: str) -> Dict[str, Tuple[int, float]]:
    """Return {mode_name: (use_count, avg_quality)} from SQLite."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT mode_name, use_count, avg_quality FROM agi_mode_stats"
        ).fetchall()
        conn.close()
        return {r[0]: (r[1], r[2]) for r in rows}
    except Exception:
        return {}


def _get_sys2_performance(db_path: str) -> Dict[str, float]:
    """Return system2 session stats."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT COUNT(*) as n, AVG(duration_ms) as avg_ms FROM sys2_sessions"
        ).fetchone()
        conn.close()
        return {"session_count": rows[0] or 0, "avg_duration_ms": rows[1] or 0}
    except Exception:
        return {}


def _get_world_model_stats(db_path: str) -> Dict[str, int]:
    try:
        conn = sqlite3.connect(db_path)
        stats = {
            "beliefs": conn.execute("SELECT COUNT(*) FROM wm_beliefs").fetchone()[0],
            "causal":  conn.execute("SELECT COUNT(*) FROM wm_causal").fetchone()[0],
        }
        conn.close()
        return stats
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Improvement proposal generation
# ---------------------------------------------------------------------------

_RSI_PROMPT = """\
You are Cortana performing recursive self-improvement analysis.

Current AGI performance metrics:
{metrics}

Identify the SINGLE highest-impact improvement that could be made to
Cortana's reasoning or memory systems. Be specific and concrete.

Return a JSON object:
{{
  "module": "which module to improve (cognitive_modes|world_model|system2|autonomy|ethics)",
  "issue": "what specific problem exists",
  "improvement": "specific, implementable change (1-3 sentences)",
  "expected_impact": "what measurable improvement is expected",
  "priority": "high|medium|low"
}}

Return ONLY the JSON."""

_KNOWLEDGE_GAP_PROMPT = """\
Analyse these topics that Cortana has been asked about but has few beliefs on:
{topics}

Generate 1-2 specific research goals Cortana should autonomously pursue
to fill these knowledge gaps. Return as a JSON list of strings (goal descriptions)."""


# ---------------------------------------------------------------------------
# Active learning: knowledge gap detection
# ---------------------------------------------------------------------------

def detect_knowledge_gaps(db_path: str, recent_queries: List[str]) -> List[str]:
    """Find topics frequently asked about but sparse in world model."""
    if not recent_queries:
        return []
    try:
        conn = sqlite3.connect(db_path)
        gaps: List[str] = []
        for query in recent_queries[:10]:
            words = [w for w in re.findall(r"\b\w{5,}\b", query.lower()) if len(w) > 4]
            for word in words[:3]:
                count = conn.execute(
                    "SELECT COUNT(*) FROM wm_beliefs WHERE LOWER(subject) LIKE ? OR LOWER(obj) LIKE ?",
                    (f"%{word}%", f"%{word}%"),
                ).fetchone()[0]
                if count == 0:
                    gaps.append(word)
        conn.close()
        return list(dict.fromkeys(gaps))[:5]  # deduplicate, cap at 5
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SelfImprovementEngine:
    """
    Recursive self-improvement: analyses performance, generates patches,
    and queues knowledge-gap research goals.
    """

    def __init__(self, reasoning: Any = None, autonomy: Any = None) -> None:
        self.reasoning = reasoning
        self.autonomy  = autonomy
        self._last_rsi = 0.0
        self._recent_queries: List[str] = []
        _PATCH_DIR.mkdir(parents=True, exist_ok=True)

    def record_query(self, query: str) -> None:
        """Call with each user query to track for gap detection."""
        self._recent_queries.append(query)
        if len(self._recent_queries) > 50:
            self._recent_queries = self._recent_queries[-50:]

    def idle_tick(self) -> Optional[str]:
        """
        Called from consciousness engine idle ticks.
        Runs one improvement analysis if cooldown elapsed.
        """
        now = time.time()
        if now - self._last_rsi < _RSI_COOLDOWN:
            return None
        self._last_rsi = now

        log_lines: List[str] = []

        # 1. Knowledge gap detection → new research goals
        gaps = detect_knowledge_gaps(str(_DB_PATH), self._recent_queries)
        if gaps and self.autonomy and self.reasoning:
            try:
                raw = self.reasoning.think_simple(
                    prompt=_KNOWLEDGE_GAP_PROMPT.format(topics=", ".join(gaps)),
                    system="You generate precise research goals for an autonomous AI.",
                    max_tokens=200,
                )
                m = re.search(r"\[[\s\S]*\]", raw)
                if m:
                    goals = json.loads(m.group(0))
                    for g in goals[:2]:
                        if isinstance(g, str) and len(g) > 10:
                            self.autonomy.add_goal(g, domain="knowledge_gap", priority=0.65)
                            log_lines.append(f"Gap goal: {g[:60]}")
            except Exception:
                pass

        # 2. Performance analysis → improvement proposal
        if self.reasoning:
            try:
                metrics = self._collect_metrics()
                raw = self.reasoning.think_simple(
                    prompt=_RSI_PROMPT.format(metrics=json.dumps(metrics, indent=2)),
                    system="You are Cortana analysing your own performance metrics.",
                    max_tokens=300,
                )
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    proposal = json.loads(m.group(0))
                    patch_path = self._write_improvement_patch(proposal)
                    log_lines.append(
                        f"RSI proposal [{proposal.get('priority','?')}]: "
                        f"{proposal.get('module','?')} — {proposal.get('issue','')[:50]}"
                    )
            except Exception:
                pass

        return " | ".join(log_lines) if log_lines else None

    def _collect_metrics(self) -> dict:
        db = str(_DB_PATH)
        return {
            "cognitive_modes":   _get_mode_performance(db),
            "system2":           _get_sys2_performance(db),
            "world_model":       _get_world_model_stats(db),
            "recent_query_count": len(self._recent_queries),
        }

    def _write_improvement_patch(self, proposal: dict) -> Optional[Path]:
        """Write improvement proposal to patches directory for L12 review."""
        try:
            ts = int(time.time())
            fname = _PATCH_DIR / f"rsi_{ts}_{proposal.get('module','agi')}.json"
            fname.write_text(json.dumps({
                "type": "agi_self_improvement",
                "timestamp": ts,
                "proposal": proposal,
            }, indent=2))
            return fname
        except Exception:
            return None
