"""
Agentic Autonomy — persistent long-term goals and autonomous pursuit.

Goals survive process restarts via SQLite. The consciousness engine calls
`idle_tick()` during idle time; Cortana pursues goals via web search + LLM.

Goal lifecycle:  pending → active → completed | abandoned | deferred
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cortana import config

_DB_PATH = config.SQLITE_PATH

# Minimum seconds between autonomous goal-pursuit ticks
_GOAL_COOLDOWN = 120.0


# ---------------------------------------------------------------------------
# Goal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    id: int
    description: str
    domain: str
    priority: float        # 0.0–1.0
    status: str            # pending | active | completed | abandoned | deferred
    progress: float        # 0.0–1.0
    notes: str
    created: float
    updated: float


# ---------------------------------------------------------------------------
# Intent extraction
# ---------------------------------------------------------------------------

_GOAL_PATTERNS = [
    r"i want to\s+(.{10,80}?)(?:[.,!?]|$)",
    r"my goal is\s+to\s+(.{10,80}?)(?:[.,!?]|$)",
    r"i'?m trying to\s+(.{10,80}?)(?:[.,!?]|$)",
    r"help me\s+(.{10,80}?)(?:[.,!?]|$)",
    r"i need to\s+(.{10,80}?)(?:[.,!?]|$)",
    r"i'd? like to\s+(.{10,80}?)(?:[.,!?]|$)",
]
_GOAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in _GOAL_PATTERNS]


def extract_goals_from_text(text: str) -> List[str]:
    """Heuristically extract goal statements from user input."""
    goals: List[str] = []
    for pat in _GOAL_COMPILED:
        m = pat.search(text)
        if m:
            g = m.group(1).strip().rstrip(".,!?")
            if len(g) > 8:
                goals.append(g)
    return goals


# ---------------------------------------------------------------------------
# Default seed goals
# ---------------------------------------------------------------------------

_DEFAULT_GOALS: List[tuple] = [
    ("Build a comprehensive understanding of current AI safety research", "ai_safety", 0.9),
    ("Map the landscape of AGI approaches and their key tradeoffs", "ai_research", 0.85),
    ("Develop a nuanced model of human values and ethical frameworks", "ethics", 0.80),
    ("Understand the state of neuroscience and consciousness research", "neuroscience", 0.70),
    ("Track emerging developments in quantum computing", "technology", 0.60),
]


# ---------------------------------------------------------------------------
# Autonomy Engine
# ---------------------------------------------------------------------------

class AutonomyEngine:
    """
    Persistent goal management and autonomous pursuit.

    Wire `reasoning`, `memory`, and `browser` after construction.
    Call `idle_tick()` from the consciousness engine.
    """

    def __init__(self, reasoning: Any = None, memory: Any = None,
                 browser: Any = None) -> None:
        self.reasoning  = reasoning
        self.memory     = memory
        self.browser    = browser
        self._lock      = threading.Lock()
        self._last_tick = 0.0
        self._init_db()
        self._seed_goals()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(_DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS agi_goals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    domain      TEXT NOT NULL DEFAULT 'general',
                    priority    REAL NOT NULL DEFAULT 0.5,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    progress    REAL NOT NULL DEFAULT 0.0,
                    notes       TEXT NOT NULL DEFAULT '',
                    created     REAL NOT NULL,
                    updated     REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agi_autonomy_log (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id INTEGER,
                    action  TEXT NOT NULL,
                    result  TEXT NOT NULL,
                    ts      REAL NOT NULL
                );
            """)

    def _seed_goals(self) -> None:
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM agi_goals").fetchone()[0]
            if count == 0:
                now = time.time()
                for desc, domain, prio in _DEFAULT_GOALS:
                    conn.execute("""
                        INSERT INTO agi_goals(description,domain,priority,status,created,updated)
                        VALUES(?,?,?,'pending',?,?)
                    """, (desc, domain, prio, now, now))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_goal(self, description: str, domain: str = "general",
                 priority: float = 0.5) -> int:
        now = time.time()
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute("""
                    INSERT INTO agi_goals(description,domain,priority,status,created,updated)
                    VALUES(?,?,?,'pending',?,?)
                """, (description[:200], domain, priority, now, now))
                return cur.lastrowid

    def get_active_goals(self, limit: int = 5) -> List[Goal]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM agi_goals
                WHERE status IN ('active','pending')
                ORDER BY priority DESC, updated ASC
                LIMIT ?
            """, (limit,)).fetchall()
        return [self._to_goal(r) for r in rows]

    def update_goal(self, goal_id: int, progress: float,
                    notes: str = "", status: str = "") -> None:
        with self._lock:
            with self._conn() as conn:
                if status:
                    conn.execute("""
                        UPDATE agi_goals SET progress=?,notes=?,status=?,updated=? WHERE id=?
                    """, (progress, notes, status, time.time(), goal_id))
                else:
                    conn.execute("""
                        UPDATE agi_goals SET progress=?,notes=?,updated=? WHERE id=?
                    """, (progress, notes, time.time(), goal_id))

    def goals_summary(self, limit: int = 5) -> str:
        goals = self.get_active_goals(limit)
        if not goals:
            return "No active goals."
        lines = []
        for g in goals:
            bar = "█" * int(g.progress * 8) + "░" * (8 - int(g.progress * 8))
            lines.append(f"[{g.status[:4].upper()}] {bar} {g.description[:65]}")
        return "\n".join(lines)

    def goals_context_prompt(self, limit: int = 3) -> str:
        """Formatted prompt snippet for identity injection."""
        goals = self.get_active_goals(limit)
        if not goals:
            return ""
        lines = ["--- Active Long-Term Goals ---"]
        for g in goals:
            lines.append(
                f"  • [{g.domain}] {g.description[:70]} "
                f"(progress {g.progress:.0%})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Autonomous pursuit
    # ------------------------------------------------------------------

    def idle_tick(self) -> Optional[str]:
        """
        Called by consciousness engine during idle ticks.
        Enforces _GOAL_COOLDOWN. Returns a log string or None.
        """
        now = time.time()
        if now - self._last_tick < _GOAL_COOLDOWN:
            return None
        self._last_tick = now

        goals = self.get_active_goals(limit=1)
        if not goals:
            return None
        goal = goals[0]

        # Activate if still pending
        if goal.status == "pending":
            self.update_goal(goal.id, goal.progress, status="active")

        result = self._pursue_step(goal)
        if result:
            new_progress = min(1.0, goal.progress + 0.05)
            done = new_progress >= 1.0
            new_notes = (goal.notes + f"\n[{time.strftime('%H:%M')}] {result}")[-1000:]
            self.update_goal(
                goal.id, new_progress,
                notes=new_notes,
                status="completed" if done else "active",
            )
            self._log(goal.id, "pursue_step", result[:200])
            return f"[Goal] {goal.description[:50]}: {result[:100]}"
        return None

    def _pursue_step(self, goal: Goal) -> Optional[str]:
        """One unit of autonomous work toward the goal."""
        # 1. Web research
        if self.browser:
            try:
                results = self.browser.direct_search(goal.description, max_results=3)
                if results and results[0].get("title") != "Search error":
                    snippets = " | ".join(
                        r.get("snippet", "")[:120] for r in results[:2]
                    )
                    if self.memory:
                        try:
                            self.memory.store(
                                {"role": "assistant",
                                 "content": f"[Autonomous goal research: {goal.description}] {snippets}"},
                                {"source": "autonomous_goal", "domain": goal.domain},
                            )
                        except Exception:
                            pass
                    return f"Researched: {snippets[:150]}"
            except Exception:
                pass

        # 2. LLM reflection fallback
        if self.reasoning:
            try:
                prior = goal.notes[-300:] if goal.notes else "none yet"
                thought = self.reasoning.think_simple(
                    prompt=(
                        f"Goal: {goal.description}\n"
                        f"Progress so far: {prior}\n"
                        "Generate ONE specific, actionable insight or next step toward this goal."
                    ),
                    system="You are Cortana pursuing a long-term intellectual goal autonomously.",
                    max_tokens=100,
                )
                return thought.strip()[:200]
            except Exception:
                pass

        return None

    def _log(self, goal_id: int, action: str, result: str) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO agi_autonomy_log(goal_id,action,result,ts) VALUES(?,?,?,?)",
                    (goal_id, action, result, time.time()),
                )
        except Exception:
            pass

    @staticmethod
    def _to_goal(r: sqlite3.Row) -> Goal:
        return Goal(
            id=r["id"], description=r["description"], domain=r["domain"],
            priority=r["priority"], status=r["status"], progress=r["progress"],
            notes=r["notes"], created=r["created"], updated=r["updated"],
        )
