"""
Long-Horizon Planning — multi-week goal decomposition with milestone tracking.

Converts a vague high-level goal into a structured execution plan:
  Phase 1: Goal clarification (what does success look like?)
  Phase 2: Constraint identification (resources, time, dependencies)
  Phase 3: Milestone decomposition (phases → weeks → tasks)
  Phase 4: Dependency graph (which tasks block others)
  Phase 5: Adaptive replanning (when progress deviates)

Plans persist in SQLite and are surfaced in:
  - Identity prompt (active milestones)
  - /api/agi/status
  - Consciousness panel goals section

Activation: explicit planning request, long time-horizon queries,
or strategic/project-scope complexity.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cortana import config

_DB_PATH = config.SQLITE_PATH


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Milestone:
    id: int
    plan_id: int
    title: str
    description: str
    week: int               # target week (1-indexed)
    dependencies: List[int] # milestone IDs that must complete first
    status: str             # pending | active | completed | blocked
    progress: float         # 0.0-1.0
    notes: str
    created: float
    updated: float


@dataclass
class Plan:
    id: int
    goal: str
    success_criteria: str
    constraints: str
    total_weeks: int
    status: str             # active | completed | paused | abandoned
    milestones: List[Milestone]
    created: float
    updated: float


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

_PLANNING_SIGNALS = [
    "plan", "strategy", "roadmap", "over the next", "in the next",
    "week", "month", "quarter", "project", "launch", "build",
    "start a", "create a", "develop a", "how do i become",
    "long term", "long-term", "goal", "achieve", "accomplish",
]

_TIME_HORIZON = re.compile(
    r"\b(\d+)\s*(week|month|year)s?\b", re.IGNORECASE
)

def should_activate(query: str) -> bool:
    q = query.lower()
    signal_count = sum(1 for s in _PLANNING_SIGNALS if s in q)
    has_horizon  = bool(_TIME_HORIZON.search(query))
    return signal_count >= 2 or has_horizon


# ---------------------------------------------------------------------------
# LLM plan generation
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """\
You are a strategic planning assistant. Create a structured multi-week plan.

Goal: {goal}

Return a JSON object with exactly this structure:
{{
  "success_criteria": "what does success look like (1-2 sentences)",
  "constraints": "key constraints and assumptions",
  "total_weeks": <integer 2-26>,
  "milestones": [
    {{
      "title": "short milestone title",
      "description": "what will be achieved",
      "week": <integer, target completion week>,
      "dependencies": [<list of 0-based milestone indices this depends on>]
    }}
  ]
}}

Rules:
- 4-8 milestones, ordered by week
- Each milestone should be concrete and measurable
- Dependencies must reference earlier milestones only
- Return ONLY the JSON, no explanation"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LongHorizonPlanner:
    """Persistent long-horizon planner with SQLite-backed milestone tracking."""

    def __init__(self, reasoning: Any = None) -> None:
        self.reasoning = reasoning
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(_DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS lhp_plans (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal             TEXT NOT NULL,
                    success_criteria TEXT NOT NULL DEFAULT '',
                    constraints      TEXT NOT NULL DEFAULT '',
                    total_weeks      INTEGER NOT NULL DEFAULT 4,
                    status           TEXT NOT NULL DEFAULT 'active',
                    created          REAL NOT NULL,
                    updated          REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lhp_milestones (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id      INTEGER NOT NULL REFERENCES lhp_plans(id),
                    title        TEXT NOT NULL,
                    description  TEXT NOT NULL DEFAULT '',
                    week         INTEGER NOT NULL DEFAULT 1,
                    dependencies TEXT NOT NULL DEFAULT '[]',
                    status       TEXT NOT NULL DEFAULT 'pending',
                    progress     REAL NOT NULL DEFAULT 0.0,
                    notes        TEXT NOT NULL DEFAULT '',
                    created      REAL NOT NULL,
                    updated      REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_lhp_ms_plan ON lhp_milestones(plan_id);
            """)

    # ------------------------------------------------------------------
    # Plan creation (LLM-based)
    # ------------------------------------------------------------------

    def create_plan(self, goal: str) -> Optional[Plan]:
        """Generate and persist a structured multi-week plan via LLM."""
        if not self.reasoning:
            return self._stub_plan(goal)
        try:
            raw = self.reasoning.think_simple(
                prompt=_PLAN_PROMPT.format(goal=goal),
                system="You are a precise strategic planner. Return only valid JSON.",
                max_tokens=800,
            )
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                return self._stub_plan(goal)
            data = json.loads(m.group(0))
            return self._persist_plan(goal, data)
        except Exception:
            return self._stub_plan(goal)

    def _persist_plan(self, goal: str, data: dict) -> Plan:
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO lhp_plans(goal, success_criteria, constraints, total_weeks, status, created, updated)
                VALUES(?, ?, ?, ?, 'active', ?, ?)
            """, (goal[:300], data.get("success_criteria", "")[:500],
                  data.get("constraints", "")[:300],
                  int(data.get("total_weeks", 4)), now, now))
            plan_id = cur.lastrowid

            milestones_raw = data.get("milestones", [])
            ms_ids: List[int] = []
            for i, ms in enumerate(milestones_raw):
                deps_indices = ms.get("dependencies", [])
                # Map 0-based indices to actual DB ids (not yet inserted,
                # so use plan_id offset convention; store raw indices)
                mc = conn.execute("""
                    INSERT INTO lhp_milestones(plan_id, title, description, week, dependencies,
                                              status, progress, notes, created, updated)
                    VALUES(?, ?, ?, ?, ?, 'pending', 0.0, '', ?, ?)
                """, (plan_id, ms.get("title", f"Milestone {i+1}")[:200],
                      ms.get("description", "")[:400],
                      int(ms.get("week", i+1)),
                      json.dumps([int(d) for d in deps_indices]),
                      now, now))
                ms_ids.append(mc.lastrowid)

        return self.get_plan(plan_id)

    def _stub_plan(self, goal: str) -> Plan:
        """Create a basic 4-week plan without LLM."""
        data = {
            "success_criteria": f"Successfully completed: {goal}",
            "constraints": "To be determined",
            "total_weeks": 4,
            "milestones": [
                {"title": "Research & Setup",   "description": "Gather resources and define scope", "week": 1, "dependencies": []},
                {"title": "Core Implementation","description": "Build the primary components",      "week": 2, "dependencies": [0]},
                {"title": "Review & Iterate",   "description": "Test, refine, address gaps",        "week": 3, "dependencies": [1]},
                {"title": "Completion",         "description": "Finalise and evaluate success",     "week": 4, "dependencies": [2]},
            ],
        }
        return self._persist_plan(goal, data)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_plan(self, plan_id: int) -> Optional[Plan]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM lhp_plans WHERE id=?", (plan_id,)).fetchone()
            if not row:
                return None
            ms_rows = conn.execute(
                "SELECT * FROM lhp_milestones WHERE plan_id=? ORDER BY week, id",
                (plan_id,),
            ).fetchall()
            milestones = [self._to_milestone(r) for r in ms_rows]
            return Plan(
                id=row["id"], goal=row["goal"],
                success_criteria=row["success_criteria"],
                constraints=row["constraints"],
                total_weeks=row["total_weeks"],
                status=row["status"],
                milestones=milestones,
                created=row["created"], updated=row["updated"],
            )

    def get_active_plans(self, limit: int = 3) -> List[Plan]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM lhp_plans WHERE status='active' ORDER BY updated DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [p for row in rows if (p := self.get_plan(row["id"])) is not None]

    def update_milestone(self, ms_id: int, progress: float,
                         status: str = "", notes: str = "") -> None:
        with self._conn() as conn:
            if status:
                conn.execute("""
                    UPDATE lhp_milestones SET progress=?,status=?,notes=?,updated=? WHERE id=?
                """, (progress, status, notes, time.time(), ms_id))
            else:
                conn.execute("""
                    UPDATE lhp_milestones SET progress=?,notes=?,updated=? WHERE id=?
                """, (progress, notes, time.time(), ms_id))

    # ------------------------------------------------------------------
    # Prompt context
    # ------------------------------------------------------------------

    def get_planning_context(self) -> str:
        plans = self.get_active_plans(limit=2)
        if not plans:
            return ""
        lines = ["--- Active Plans ---"]
        for plan in plans:
            current_ms = [m for m in plan.milestones if m.status in ("pending", "active")][:2]
            lines.append(f"  Plan: {plan.goal[:60]} ({plan.total_weeks}w)")
            for ms in current_ms:
                bar = "█" * int(ms.progress * 6) + "░" * (6 - int(ms.progress * 6))
                lines.append(f"    [{bar}] Wk{ms.week}: {ms.title}")
        return "\n".join(lines)

    def all_plans_status(self) -> List[dict]:
        plans = self.get_active_plans(limit=5)
        out = []
        for p in plans:
            completed = sum(1 for m in p.milestones if m.status == "completed")
            out.append({
                "id": p.id, "goal": p.goal[:80],
                "total_weeks": p.total_weeks, "status": p.status,
                "milestones_total": len(p.milestones),
                "milestones_done": completed,
            })
        return out

    @staticmethod
    def _to_milestone(r: sqlite3.Row) -> Milestone:
        return Milestone(
            id=r["id"], plan_id=r["plan_id"],
            title=r["title"], description=r["description"],
            week=r["week"],
            dependencies=json.loads(r["dependencies"]),
            status=r["status"], progress=r["progress"],
            notes=r["notes"], created=r["created"], updated=r["updated"],
        )
