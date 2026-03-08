"""
System 2 Reasoning — slow, deliberate, multi-step reasoning engine.

Contrasts with the fast System 1 (immediate L4 response).

System 2 activates when:
  - complexity score > 0.7
  - intent in (research, analysis, plan, code) AND query length > 120 chars
  - explicit trigger words ("think carefully", "step by step", "reason through")

Process:
  1. Decompose — break problem into sub-questions
  2. Deliberate — answer each sub-question sequentially (chain-of-thought)
  3. Critique   — identify flaws, gaps, or contradictions in own reasoning
  4. Synthesise — assemble final answer from validated sub-answers

Each stage is a separate LLM call with explicit instructions. Results
are stored in SQLite for inspection and memory injection.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from cortana import config

_DB_PATH = config.SQLITE_PATH

_TRIGGER_PHRASES = [
    "think carefully", "think through", "step by step", "reason through",
    "walk me through", "break it down", "think step", "reason carefully",
    "thorough analysis", "deep dive", "analyse thoroughly",
]

_TRIGGER_INTENTS = {"research", "analysis", "plan", "code", "debug"}
_COMPLEXITY_THRESHOLD = 0.72
_LENGTH_THRESHOLD     = 120


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class System2Result:
    triggered: bool
    sub_questions: List[str]      = field(default_factory=list)
    sub_answers:   List[str]      = field(default_factory=list)
    critique:      str            = ""
    synthesis:     str            = ""
    duration_ms:   float          = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class System2Engine:
    """
    Slow deliberate reasoning. Called from main.py process_session
    when System 2 should activate; returns a synthesis to replace or
    augment the L4 prompt.
    """

    def __init__(self, reasoning: Any = None) -> None:
        self.reasoning = reasoning
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(_DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sys2_sessions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    query        TEXT NOT NULL,
                    sub_questions TEXT NOT NULL DEFAULT '[]',
                    synthesis    TEXT NOT NULL DEFAULT '',
                    duration_ms  REAL NOT NULL DEFAULT 0,
                    ts           REAL NOT NULL
                )
            """)

    def should_activate(self, query: str, intent: str,
                        complexity: float) -> bool:
        """Decide whether System 2 should engage for this query."""
        q_lower = query.lower()
        if any(phrase in q_lower for phrase in _TRIGGER_PHRASES):
            return True
        if (complexity >= _COMPLEXITY_THRESHOLD
                and intent in _TRIGGER_INTENTS
                and len(query) >= _LENGTH_THRESHOLD):
            return True
        return False

    def reason(self, query: str, context: str = "") -> System2Result:
        """
        Full System 2 pipeline: decompose → deliberate → critique → synthesise.
        Returns a System2Result; synthesis is the key output.
        """
        if self.reasoning is None:
            return System2Result(triggered=False)

        t0 = time.time()

        # ── Stage 1: Decompose ──────────────────────────────────────────
        decomp_prompt = (
            f"Question: {query}\n\n"
            "Break this into 2-4 specific sub-questions that, when answered, "
            "will fully address the original question. "
            "Return ONLY a numbered list, one sub-question per line."
        )
        try:
            decomp_raw = self.reasoning.think_simple(
                prompt=decomp_prompt,
                system=(
                    "You are a careful analytical reasoner. "
                    "Decompose problems into precise, answerable sub-questions."
                ),
                max_tokens=200,
            )
            sub_qs = self._parse_list(decomp_raw)
        except Exception:
            return System2Result(triggered=True)

        if not sub_qs:
            return System2Result(triggered=True)

        # ── Stage 2: Deliberate ─────────────────────────────────────────
        sub_answers: List[str] = []
        for sq in sub_qs[:4]:  # cap at 4
            try:
                ctx_block = f"Context: {context[:400]}\n\n" if context else ""
                ans = self.reasoning.think_simple(
                    prompt=f"{ctx_block}Sub-question: {sq}\n\nAnswer carefully and concisely.",
                    system=(
                        "You are a precise reasoner. "
                        "Answer the specific sub-question accurately and briefly."
                    ),
                    max_tokens=250,
                )
                sub_answers.append(ans.strip())
            except Exception:
                sub_answers.append("[reasoning error]")

        # ── Stage 3: Critique ───────────────────────────────────────────
        qa_pairs = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(sub_qs, sub_answers)
        )
        critique = ""
        try:
            critique = self.reasoning.think_simple(
                prompt=(
                    f"Original question: {query}\n\n"
                    f"Reasoning so far:\n{qa_pairs}\n\n"
                    "Identify any gaps, contradictions, or errors in this reasoning. "
                    "Be concise — only flag genuine issues."
                ),
                system=(
                    "You are a critical reviewer. "
                    "Find real flaws in reasoning; don't manufacture issues."
                ),
                max_tokens=150,
            ).strip()
        except Exception:
            pass

        # ── Stage 4: Synthesise ─────────────────────────────────────────
        critique_block = f"\nCritique of above reasoning: {critique}\n" if critique else ""
        synthesis = ""
        try:
            synthesis = self.reasoning.think_simple(
                prompt=(
                    f"Original question: {query}\n\n"
                    f"Reasoning:\n{qa_pairs}{critique_block}\n\n"
                    "Synthesise a complete, accurate answer to the original question. "
                    "Incorporate corrections from the critique. Be clear and direct."
                ),
                system=(
                    "You are a precise synthesiser. "
                    "Produce the final answer — coherent, accurate, well-reasoned."
                ),
                max_tokens=600,
            ).strip()
        except Exception:
            synthesis = sub_answers[-1] if sub_answers else ""

        duration_ms = (time.time() - t0) * 1000

        # Store for inspection
        try:
            import json
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO sys2_sessions(query, sub_questions, synthesis, duration_ms, ts)
                    VALUES(?, ?, ?, ?, ?)
                """, (query[:300], json.dumps(sub_qs), synthesis[:2000],
                      duration_ms, time.time()))
        except Exception:
            pass

        return System2Result(
            triggered=True,
            sub_questions=sub_qs,
            sub_answers=sub_answers,
            critique=critique,
            synthesis=synthesis,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = text.strip().splitlines()
        items: List[str] = []
        for line in lines:
            line = re.sub(r"^\s*[\d\-\*\.]+\s*", "", line).strip()
            if len(line) > 10:
                items.append(line)
        return items[:5]
