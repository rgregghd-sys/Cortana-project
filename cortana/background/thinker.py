"""
Background Thinker — persistent iterative reasoning engine.

When Cortana is asked to work on something "in the background", this module
runs an iterative chain-of-thought loop in a daemon thread, saving progress
to SQLite. When done, it notifies all WebSocket clients via the main event loop.

Each thinking step either:
  THINKING: <current analysis + next angle to explore>
  FINAL:    <complete, well-formed answer ready to present>
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Callable, List, Optional, Tuple


_THINK_SYSTEM = """You are Cortana's background deep reasoning engine.
You work through complex problems methodically, building on each step.

Each response MUST start with exactly one of:
  THINKING: — if more analysis is needed
  FINAL:    — if you have a complete, well-formed answer

Under THINKING: summarise what you've established so far, then state exactly
what angle you will explore next.

Under FINAL: give the complete answer, ready to present to the user directly.

Rules:
- Never repeat reasoning already covered — always advance
- Challenge your own assumptions
- Be thorough but precise — avoid filler
"""


class BackgroundThinker:
    """
    Manages background reasoning tasks.
    - Tasks run in daemon threads (LLM calls are synchronous)
    - Progress and results are persisted to SQLite
    - Completion events are pushed to WebSocket clients via the FastAPI event loop
    """

    MAX_ITERATIONS = 8

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._reasoning = None          # set via set_reasoning()
        self._broadcast_fn: Optional[Callable] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Setup (called after dependent layers are ready)
    # ------------------------------------------------------------------

    def set_reasoning(self, reasoning_layer) -> None:
        self._reasoning = reasoning_layer

    def set_broadcast(
        self,
        fn: Callable,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Called by ChatLayer at startup so completed tasks can push WebSocket
        events to all connected browsers.
        """
        self._broadcast_fn = fn
        self._loop = loop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_task(self, query: str) -> Tuple[str, str]:
        """
        Launch a background reasoning task.
        Returns (task_id, short_name).
        """
        task_id = str(uuid.uuid4())[:8]
        name = _make_name(query)
        self._insert_task(task_id, name, query)
        t = threading.Thread(
            target=self._think_loop,
            args=(task_id, query),
            daemon=True,
            name=f"cortana-bg-{task_id}",
        )
        t.start()
        return task_id, name

    def get_latest_task(self) -> Optional[dict]:
        """Return the most recently started task (any status)."""
        rows = self._query(
            "SELECT * FROM background_tasks ORDER BY created_at DESC LIMIT 1"
        )
        return rows[0] if rows else None

    def get_active_tasks(self) -> List[dict]:
        """Return tasks currently running."""
        return self._query(
            "SELECT * FROM background_tasks WHERE status='running' ORDER BY created_at DESC"
        )

    def get_all_tasks(self, limit: int = 10) -> List[dict]:
        return self._query(
            "SELECT * FROM background_tasks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    def has_any_tasks(self) -> bool:
        return bool(self._query("SELECT id FROM background_tasks LIMIT 1"))

    def has_active_tasks(self) -> bool:
        return bool(
            self._query(
                "SELECT id FROM background_tasks WHERE status='running' LIMIT 1"
            )
        )

    def format_status(self, task: dict) -> str:
        """Format a task record into a human-readable status response."""
        name = task["name"]
        status = task["status"]
        iterations = task.get("iterations", 0)
        thoughts: List[str] = json.loads(task.get("thoughts", "[]"))

        if status == "running":
            last = ""
            if thoughts:
                last = thoughts[-1].replace("THINKING:", "").strip()[:300]
            return (
                f"Still working on '{name}' — {iterations} reasoning "
                f"step(s) done so far.\n\n"
                f"Current thread of thought:\n{last}…" if last else
                f"Still working on '{name}' — {iterations} step(s) so far, "
                f"just getting started."
            )

        if status == "done":
            result = task.get("result") or "No result recorded."
            return (
                f"Finished '{name}' after {iterations} reasoning step(s).\n\n"
                f"{result}"
            )

        return (
            f"Task '{name}' ran into a problem: "
            f"{task.get('result', 'Unknown error')}"
        )

    # ------------------------------------------------------------------
    # Iterative reasoning loop — runs in daemon thread
    # ------------------------------------------------------------------

    def _think_loop(self, task_id: str, query: str) -> None:
        if self._reasoning is None:
            self._update_task(
                task_id, status="failed", result="Reasoning layer not initialised."
            )
            return

        thoughts: List[str] = []
        try:
            for i in range(self.MAX_ITERATIONS):
                prompt = _build_prompt(query, thoughts, i, self.MAX_ITERATIONS)
                response = self._reasoning.think_simple(
                    prompt=prompt,
                    system=_THINK_SYSTEM,
                    max_tokens=1024,
                )
                thoughts.append(response)
                self._update_task(task_id, thoughts=thoughts, iterations=i + 1)

                # Broadcast progress every 2 steps so the UI can reflect it
                if (i + 1) % 2 == 0:
                    preview = response.replace("THINKING:", "").strip()[:150]
                    self._notify({
                        "type": "background_progress",
                        "task_id": task_id,
                        "name": self._get_name(task_id),
                        "iteration": i + 1,
                        "preview": preview,
                    })

                if response.strip().startswith("FINAL:"):
                    result = response.strip()[6:].strip()
                    self._update_task(
                        task_id, status="done", result=result, thoughts=thoughts
                    )
                    self._notify({
                        "type": "background_done",
                        "task_id": task_id,
                        "name": self._get_name(task_id),
                        "preview": result[:250],
                    })
                    return

            # Max iterations reached — treat last thought as best result
            last = thoughts[-1].replace("THINKING:", "").strip() if thoughts else ""
            self._update_task(task_id, status="done", result=last, thoughts=thoughts)
            self._notify({
                "type": "background_done",
                "task_id": task_id,
                "name": self._get_name(task_id),
                "preview": last[:250],
            })

        except Exception as exc:
            self._update_task(task_id, status="failed", result=f"Error: {exc}")
            self._notify({
                "type": "background_done",
                "task_id": task_id,
                "name": self._get_name(task_id),
                "preview": f"Task failed: {exc}",
            })

    # ------------------------------------------------------------------
    # WebSocket notification (thread-safe → main event loop)
    # ------------------------------------------------------------------

    def _notify(self, data: dict) -> None:
        if self._broadcast_fn and self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_fn(data), self._loop
            )

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS background_tasks (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    query       TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'running',
                    thoughts    TEXT NOT NULL DEFAULT '[]',
                    result      TEXT,
                    iterations  INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def _insert_task(self, task_id: str, name: str, query: str) -> None:
        now = datetime.now().isoformat()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO background_tasks "
                "(id, name, query, status, thoughts, iterations, created_at, updated_at) "
                "VALUES (?, ?, ?, 'running', '[]', 0, ?, ?)",
                (task_id, name, query, now, now),
            )

    def _update_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        thoughts: Optional[List[str]] = None,
        result: Optional[str] = None,
        iterations: Optional[int] = None,
    ) -> None:
        sets, vals = [], []
        if status is not None:
            sets.append("status=?")
            vals.append(status)
        if thoughts is not None:
            sets.append("thoughts=?")
            vals.append(json.dumps(thoughts))
        if result is not None:
            sets.append("result=?")
            vals.append(result)
        if iterations is not None:
            sets.append("iterations=?")
            vals.append(iterations)
        sets.append("updated_at=?")
        vals.append(datetime.now().isoformat())
        vals.append(task_id)
        with self._lock, self._conn() as conn:
            conn.execute(
                f"UPDATE background_tasks SET {', '.join(sets)} WHERE id=?", vals
            )

    def _query(self, sql: str, params: tuple = ()) -> List[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def _get_name(self, task_id: str) -> str:
        rows = self._query(
            "SELECT name FROM background_tasks WHERE id=?", (task_id,)
        )
        return rows[0]["name"] if rows else task_id


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _make_name(query: str) -> str:
    words = query.split()[:7]
    name = " ".join(words)
    return (name[:52] + "…") if len(query) > 52 else name


def _build_prompt(
    query: str,
    thoughts: List[str],
    iteration: int,
    max_iter: int,
) -> str:
    if not thoughts:
        return f"Problem to solve:\n{query}\n\nBegin your analysis."
    history = "\n\n".join(
        f"Step {j + 1}:\n{t}" for j, t in enumerate(thoughts)
    )
    return (
        f"Problem:\n{query}\n\n"
        f"Reasoning so far:\n{history}\n\n"
        f"This is step {iteration + 1} of maximum {max_iter}. "
        f"Continue — or conclude with FINAL: if you have a complete answer."
    )
