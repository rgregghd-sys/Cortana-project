"""
Nexus Flush — Periodic re-validation of all stored World Model beliefs
against ground-truth facts using the LLM as a critic.

Schedule: every FLUSH_INTERVAL_DAYS (default 3 days).

Pipeline:
  1. Load all beliefs from wm_beliefs (WorldModel SQLite table)
  2. Batch them (BATCH_SIZE at a time)
  3. Ask the LLM: "Are any of these beliefs factually incorrect or outdated?"
  4. For each belief flagged as incorrect: mark confidence -= 0.3 (soft delete)
     For beliefs confirmed: confidence += 0.05 (soft reinforce)
  5. Log flush results to nexus_flush_log SQLite table
  6. Trigger on startup if last flush was > FLUSH_INTERVAL_DAYS ago

Safety:
  - Never hard-deletes beliefs (only degrades confidence)
  - Beliefs with confidence < 0.1 are marked 'stale' in metadata
  - Flush runs in background thread; does NOT block conversation
  - Token budget: max FLUSH_MAX_BATCHES batches per flush cycle
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from cortana import config

_DB_PATH             = Path(config.SQLITE_PATH)
_FLUSH_INTERVAL_DAYS = float(os.getenv("NEXUS_FLUSH_INTERVAL_DAYS", "3"))
_BATCH_SIZE          = int(os.getenv("NEXUS_FLUSH_BATCH_SIZE", "20"))
_FLUSH_MAX_BATCHES   = int(os.getenv("NEXUS_FLUSH_MAX_BATCHES", "5"))
_FLUSH_INTERVAL_SECS = _FLUSH_INTERVAL_DAYS * 86400

_FLUSH_PROMPT = """\
You are a fact-checking critic. Review the following stored beliefs.
For each belief, reply with:
  CONFIRM <id>  — if the belief is factually accurate and current
  DOUBT <id>    — if the belief is likely outdated, incorrect, or unverifiable

Beliefs to review (id | subject | predicate | object | confidence):
{beliefs}

Reply ONLY with CONFIRM/DOUBT lines, one per belief. No explanations."""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nexus_flush_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                beliefs_checked  INTEGER NOT NULL DEFAULT 0,
                confirmed   INTEGER NOT NULL DEFAULT 0,
                doubted     INTEGER NOT NULL DEFAULT 0,
                duration_s  REAL NOT NULL DEFAULT 0
            )
        """)
        # Track last flush time in a config key
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nexus_flush_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)


def _get_last_flush() -> float:
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT value FROM nexus_flush_state WHERE key='last_flush'"
            ).fetchone()
        return float(row["value"]) if row else 0.0
    except Exception:
        return 0.0


def _set_last_flush(ts: float) -> None:
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nexus_flush_state(key,value) VALUES(?,?)",
                ("last_flush", str(ts))
            )
    except Exception:
        pass


def _load_beliefs(offset: int = 0, limit: int = _BATCH_SIZE) -> List[dict]:
    try:
        with _conn() as conn:
            rows = conn.execute("""
                SELECT id, subject, predicate, object, confidence
                FROM wm_beliefs
                WHERE confidence > 0.1
                ORDER BY ts ASC
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _update_belief_confidence(belief_id: int, delta: float) -> None:
    try:
        with _conn() as conn:
            conn.execute("""
                UPDATE wm_beliefs
                SET confidence = MAX(0.0, MIN(1.0, confidence + ?))
                WHERE id = ?
            """, (delta, belief_id))
    except Exception:
        pass


def _parse_verdicts(text: str) -> dict:
    """Parse CONFIRM/DOUBT lines → {id: 'confirm'|'doubt'}"""
    verdicts = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("CONFIRM"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    verdicts[int(parts[1])] = "confirm"
                except ValueError:
                    pass
        elif line.upper().startswith("DOUBT"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    verdicts[int(parts[1])] = "doubt"
                except ValueError:
                    pass
    return verdicts


def run_flush(reasoning: Any) -> dict:
    """
    Execute one full flush cycle. Returns summary dict.
    reasoning: must have .think_simple(prompt, system, max_tokens) method.
    """
    _init_db()
    t0 = time.time()
    total_confirmed = 0
    total_doubted   = 0
    total_checked   = 0

    for batch_idx in range(_FLUSH_MAX_BATCHES):
        beliefs = _load_beliefs(offset=batch_idx * _BATCH_SIZE)
        if not beliefs:
            break

        belief_lines = "\n".join(
            f"{b['id']} | {b['subject']} | {b['predicate']} | {b['object']} | {b['confidence']:.2f}"
            for b in beliefs
        )

        try:
            verdict_text = reasoning.think_simple(
                prompt=_FLUSH_PROMPT.format(beliefs=belief_lines),
                system="You are a precise fact-checking assistant. Be conservative — only DOUBT beliefs you are highly confident are wrong.",
                max_tokens=500,
            )
            verdicts = _parse_verdicts(verdict_text)
        except Exception:
            continue

        for b in beliefs:
            verdict = verdicts.get(b["id"])
            if verdict == "confirm":
                _update_belief_confidence(b["id"], +0.05)
                total_confirmed += 1
            elif verdict == "doubt":
                _update_belief_confidence(b["id"], -0.3)
                total_doubted += 1
            total_checked += 1

    duration = time.time() - t0
    _set_last_flush(time.time())

    try:
        with _conn() as conn:
            conn.execute("""
                INSERT INTO nexus_flush_log
                    (ts, beliefs_checked, confirmed, doubted, duration_s)
                VALUES(?,?,?,?,?)
            """, (time.time(), total_checked, total_confirmed, total_doubted, duration))
    except Exception:
        pass

    return {
        "beliefs_checked": total_checked,
        "confirmed":       total_confirmed,
        "doubted":         total_doubted,
        "duration_s":      round(duration, 2),
    }


class NexusFlushDaemon:
    """
    Background daemon that triggers a flush every FLUSH_INTERVAL_DAYS.
    Runs in its own thread; completely non-blocking.
    """

    def __init__(self, reasoning: Any = None) -> None:
        self.reasoning = reasoning
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        _init_db()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="nexus-flush"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Check on startup; then check every hour
        while not self._stop.is_set():
            try:
                self._maybe_flush()
            except Exception:
                pass
            self._stop.wait(3600)   # check every hour

    def _maybe_flush(self) -> None:
        if not self.reasoning:
            return
        last = _get_last_flush()
        if time.time() - last >= _FLUSH_INTERVAL_SECS:
            result = run_flush(self.reasoning)

    def attach_reasoning(self, reasoning: Any) -> None:
        self.reasoning = reasoning

    def last_flush_info(self) -> dict:
        _init_db()
        try:
            with _conn() as conn:
                row = conn.execute("""
                    SELECT ts, beliefs_checked, confirmed, doubted, duration_s
                    FROM nexus_flush_log
                    ORDER BY ts DESC LIMIT 1
                """).fetchone()
            if row:
                return dict(row)
        except Exception:
            pass
        return {}

    def recent_logs(self, limit: int = 5) -> List[dict]:
        try:
            with _conn() as conn:
                rows = conn.execute("""
                    SELECT ts, beliefs_checked, confirmed, doubted, duration_s
                    FROM nexus_flush_log ORDER BY ts DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


# Module-level singleton
nexus_flush = NexusFlushDaemon()
