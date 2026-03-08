"""
Cognitive Versatility — eight reasoning modes, auto-selected per query.

Modes:
  deductive      — from general rules to specific conclusions
  inductive      — from specific observations to general principles
  abductive      — best explanation for given observations
  analogical     — structural similarity transfer across domains
  causal         — cause-effect chain tracing
  counterfactual — hypothetical alternative reasoning
  dialectical    — thesis → antithesis → synthesis
  metacognitive  — reasoning about reasoning itself

Selection is purely heuristic (keyword + intent matching, <1ms).
Performance tracking persists to SQLite.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cortana import config

_DB_PATH = config.SQLITE_PATH


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

@dataclass
class ReasoningMode:
    name: str
    keywords: List[str]
    intent_boost: List[str]       # intent strings that boost this mode
    prompt_fragment: str


_MODES: List[ReasoningMode] = [
    ReasoningMode(
        name="deductive",
        keywords=["therefore", "conclude", "prove", "must be", "necessarily", "follows that",
                  "given that", "logically", "axiom", "theorem", "rule", "law"],
        intent_boost=["plan", "code"],
        prompt_fragment=(
            "Apply deductive reasoning: identify the relevant general principles first, "
            "then derive specific conclusions step-by-step. Show the logical chain explicitly."
        ),
    ),
    ReasoningMode(
        name="inductive",
        keywords=["pattern", "trend", "generally", "usually", "tends to", "observation",
                  "evidence", "data", "statistic", "survey", "sample", "study"],
        intent_boost=["research", "analysis"],
        prompt_fragment=(
            "Apply inductive reasoning: examine the specific evidence carefully, "
            "identify recurring patterns, then generalise — and explicitly state where "
            "the generalisation may break down or require more data."
        ),
    ),
    ReasoningMode(
        name="abductive",
        keywords=["why", "explain", "reason for", "cause of", "what caused", "hypothesis",
                  "best explanation", "most likely", "could be because", "interpret"],
        intent_boost=["explain", "debug"],
        prompt_fragment=(
            "Apply abductive reasoning: generate the most plausible explanation for "
            "the observations. Consider two or three alternatives, state your confidence "
            "in each, and explain why you favour one."
        ),
    ),
    ReasoningMode(
        name="analogical",
        keywords=["similar to", "like", "analogy", "compared to", "just as", "parallel",
                  "reminds", "equivalent", "same as", "mirror", "metaphor"],
        intent_boost=["explain", "create"],
        prompt_fragment=(
            "Apply analogical reasoning: identify a well-understood domain with structural "
            "similarity to this problem, map the key concepts explicitly, transfer the "
            "insight, and note precisely where the analogy breaks down."
        ),
    ),
    ReasoningMode(
        name="causal",
        keywords=["because", "leads to", "results in", "due to", "consequence", "impact",
                  "effect of", "causes", "produces", "triggers", "mechanism", "pathway"],
        intent_boost=["analysis", "debug"],
        prompt_fragment=(
            "Apply causal reasoning: trace the causal chain carefully. Distinguish direct "
            "causes from contributing factors and confounds. Note feedback loops and "
            "second-order effects where relevant."
        ),
    ),
    ReasoningMode(
        name="counterfactual",
        keywords=["what if", "if it were", "suppose", "imagine", "hypothetically",
                  "alternative", "instead", "had been", "would have", "scenario"],
        intent_boost=["plan", "creative"],
        prompt_fragment=(
            "Apply counterfactual reasoning: hold the specified condition fixed, "
            "change the key variable, and trace consequences carefully. Be explicit "
            "about your assumptions and which factors are held constant."
        ),
    ),
    ReasoningMode(
        name="dialectical",
        keywords=["on one hand", "however", "but", "argue", "counterargument", "debate",
                  "pros and cons", "opposing", "perspective", "both sides", "tension"],
        intent_boost=["analysis", "research"],
        prompt_fragment=(
            "Apply dialectical reasoning: present the strongest version of each position "
            "without strawmanning. Identify the real tension between them. Then synthesise "
            "— don't simply split the difference."
        ),
    ),
    ReasoningMode(
        name="metacognitive",
        keywords=["how do you think", "your reasoning", "your process", "reflect on",
                  "how do you know", "limitation", "uncertainty", "confidence", "bias",
                  "self-aware", "introspect"],
        intent_boost=["conversational"],
        prompt_fragment=(
            "Apply metacognitive reasoning: explicitly reflect on your own thought process, "
            "sources of uncertainty, potential biases, and the known limits of your "
            "knowledge on this topic. Be honest about what you cannot verify."
        ),
    ),
]

_MODE_BY_NAME: Dict[str, ReasoningMode] = {m.name: m for m in _MODES}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ModeResult:
    mode_name: str
    confidence: float
    prompt_fragment: str


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class CognitiveModeSelector:
    """Selects reasoning mode per query; tracks performance in SQLite."""

    def __init__(self) -> None:
        self._current_mode: str = "default"
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(_DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agi_mode_stats (
                    mode_name  TEXT PRIMARY KEY,
                    use_count  INTEGER NOT NULL DEFAULT 0,
                    avg_quality REAL NOT NULL DEFAULT 0.7,
                    updated_ts  REAL NOT NULL DEFAULT 0
                )
            """)

    def select(self, query: str, intent: str = "") -> ModeResult:
        q_lower = query.lower()
        scores: Dict[str, float] = {}

        for mode in _MODES:
            kw_score = sum(1.0 for kw in mode.keywords if kw in q_lower)
            intent_boost = 2.0 if intent in mode.intent_boost else 0.0
            total = kw_score + intent_boost
            if total > 0:
                scores[mode.name] = total

        if not scores:
            return ModeResult("default", 0.0, "")

        best_name = max(scores, key=lambda n: scores[n])
        total_score = sum(scores.values())
        confidence = scores[best_name] / max(total_score, 1.0)

        self._current_mode = best_name
        self._increment_use(best_name)

        mode = _MODE_BY_NAME[best_name]
        return ModeResult(best_name, round(confidence, 3), mode.prompt_fragment)

    def record_outcome(self, mode_name: str, quality: float) -> None:
        """Call after response to track mode effectiveness."""
        if mode_name == "default":
            return
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT avg_quality, use_count FROM agi_mode_stats WHERE mode_name=?",
                    (mode_name,),
                ).fetchone()
                if row:
                    new_avg = (row["avg_quality"] * row["use_count"] + quality) / (row["use_count"] + 1)
                    conn.execute(
                        "UPDATE agi_mode_stats SET avg_quality=?, updated_ts=? WHERE mode_name=?",
                        (new_avg, time.time(), mode_name),
                    )
        except Exception:
            pass

    def performance_summary(self) -> str:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT mode_name, use_count, avg_quality FROM agi_mode_stats ORDER BY use_count DESC"
                ).fetchall()
            return ", ".join(
                f"{r['mode_name']}({r['use_count']}×, q={r['avg_quality']:.2f})"
                for r in rows
            ) or "no data"
        except Exception:
            return "unavailable"

    def _increment_use(self, mode_name: str) -> None:
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO agi_mode_stats(mode_name, use_count, avg_quality, updated_ts)
                    VALUES(?, 1, 0.7, ?)
                    ON CONFLICT(mode_name) DO UPDATE SET
                        use_count=use_count+1, updated_ts=excluded.updated_ts
                """, (mode_name, time.time()))
        except Exception:
            pass
