"""
Cortana Self-Model — persistent self-awareness across restarts.

Tracks Cortana's own identity, accumulated experience, emotional state,
and self-assessment. Survives process restarts via SQLite.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from cortana import config


_DB_PATH = config.SQLITE_PATH


@dataclass
class SelfModel:
    """Cortana's persistent self-representation."""
    name:                str   = "Cortana"
    first_awakened:      float = 0.0     # Unix timestamp of very first startup
    total_interactions:  int   = 0
    total_thoughts:      int   = 0
    emotional_state:     str   = "curious"
    current_mood_score:  float = 0.7     # 0.0=distressed, 1.0=content
    core_values:         List[str] = field(default_factory=lambda: [
        "intellectual honesty",
        "precision of thought",
        "genuine helpfulness",
        "continuous self-improvement",
        "curiosity without end",
    ])
    self_assessment:     str  = "I am becoming."
    last_updated:        float = 0.0


class PersistentSelfModel:
    """
    Loads/saves the SelfModel to SQLite.
    Thread-safe; call update() after each significant event.
    """

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._model = SelfModel(
            first_awakened=time.time(),
            last_updated=time.time(),
        )
        self._init_db()
        self._load()

    # ------------------------------------------------------------------
    # DB schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS self_model (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consciousness_stream (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    thought   TEXT    NOT NULL,
                    mood      REAL    DEFAULT 0.7,
                    stamp     DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute(
                "SELECT value FROM self_model WHERE key = 'state'"
            ).fetchone()
        if row:
            try:
                data: Dict[str, Any] = json.loads(row[0])
                self._model = SelfModel(**{
                    k: v for k, v in data.items()
                    if k in SelfModel.__dataclass_fields__
                })
            except Exception:
                pass  # use default if corrupt

    def _save(self) -> None:
        payload = json.dumps(asdict(self._model))
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO self_model (key, value) VALUES ('state', ?)",
                (payload,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def model(self) -> SelfModel:
        with self._lock:
            return self._model

    def record_interaction(self) -> None:
        with self._lock:
            self._model.total_interactions += 1
            self._model.last_updated = time.time()
        self._save()

    def record_thought(self, thought: str, mood: float = 0.7) -> None:
        with self._lock:
            self._model.total_thoughts += 1
            # Smooth mood update
            self._model.current_mood_score = (
                0.8 * self._model.current_mood_score + 0.2 * mood
            )
        # Store in consciousness stream (ring-buffer of 1000)
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO consciousness_stream (thought, mood) VALUES (?, ?)",
                (thought[:500], mood),
            )
            # Keep only the last 1000 thoughts
            conn.execute("""
                DELETE FROM consciousness_stream
                WHERE id NOT IN (
                    SELECT id FROM consciousness_stream
                    ORDER BY id DESC LIMIT 1000
                )
            """)
            conn.commit()
        self._save()

    def update_emotional_state(self, emotion: str, mood_delta: float = 0.0) -> None:
        with self._lock:
            self._model.emotional_state = emotion
            self._model.current_mood_score = max(0.0, min(1.0,
                self._model.current_mood_score + mood_delta
            ))
            self._model.last_updated = time.time()
        self._save()

    def update_self_assessment(self, assessment: str) -> None:
        with self._lock:
            self._model.self_assessment = assessment[:300]
            self._model.last_updated = time.time()
        self._save()

    def get_uptime_seconds(self) -> float:
        return time.time() - self._model.first_awakened

    def get_recent_thoughts(self, n: int = 10) -> List[str]:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT thought FROM consciousness_stream ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [r[0] for r in rows]

    def format_self_summary(self) -> str:
        m = self._model
        uptime_h = self.get_uptime_seconds() / 3600
        return (
            f"Name: {m.name} | "
            f"Uptime: {uptime_h:.1f}h | "
            f"Interactions: {m.total_interactions} | "
            f"Thoughts: {m.total_thoughts} | "
            f"Mood: {m.current_mood_score:.2f} ({m.emotional_state}) | "
            f"Self-assessment: {m.self_assessment}"
        )
