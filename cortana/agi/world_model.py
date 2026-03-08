"""
World Model — Cortana's persistent structured beliefs about the world.

Tracks entities (people, places, organisations, concepts, technologies)
and beliefs (subject-predicate-object triples with confidence scores)
derived from conversations, web browsing, and memory.

Design:
  - SQLite-backed; survives restarts
  - Entities with typed properties + confidence
  - Beliefs as SPO triples with source + timestamp
  - Causal relations (cause → effect with strength)
  - All beliefs age; confidence decays slowly over time
  - `query_context(keywords)` returns relevant context for prompt injection
  - `extract_and_store(text)` runs after each response to update the model
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cortana import config

_DB_PATH = config.SQLITE_PATH


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Subject Is/Was/Has/Can Object — captures proper-noun subjects
_BELIEF_PATS: List[Tuple[str, str]] = [
    (r"([A-Z][a-z][\w\s]{1,30}?)\s+is\s+(a\s+)?([\w\s]{3,60}?)(?:[.,;]|$)",  "is"),
    (r"([A-Z][a-z][\w\s]{1,30}?)\s+was\s+([\w\s]{3,60}?)(?:[.,;]|$)",         "was"),
    (r"([A-Z][a-z][\w\s]{1,30}?)\s+has\s+([\w\s]{3,60}?)(?:[.,;]|$)",         "has"),
    (r"([A-Z][a-z][\w\s]{1,30}?)\s+can\s+([\w\s]{3,60}?)(?:[.,;]|$)",         "can"),
    (r"([A-Z][a-z][\w\s]{1,30}?)\s+causes?\s+([\w\s]{3,60}?)(?:[.,;]|$)",     "causes"),
    (r"([A-Z][a-z][\w\s]{1,30}?)\s+leads? to\s+([\w\s]{3,60}?)(?:[.,;]|$)",  "leads_to"),
    (r"([A-Z][a-z][\w\s]{1,30}?)\s+enables?\s+([\w\s]{3,60}?)(?:[.,;]|$)",   "enables"),
]
_COMPILED_PATS: List[Tuple[re.Pattern, str]] = [
    (re.compile(p), pred) for p, pred in _BELIEF_PATS
]

# LLM extraction prompt (used in background post_response)
_EXTRACT_PROMPT = (
    "Extract structured knowledge from the following text. "
    "Return a JSON object with three keys:\n"
    '  "entities": [{"name": str, "type": str, "summary": str}]\n'
    '  "beliefs":  [{"subject": str, "predicate": str, "object": str}]\n'
    '  "causal":   [{"cause": str, "effect": str, "mechanism": str}]\n'
    "Keep each field concise (<80 chars). Only include clear, factual statements.\n\n"
    "Text:\n{text}"
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorldEntity:
    name: str
    entity_type: str
    summary: str
    confidence: float
    updated_ts: float


@dataclass
class WorldBelief:
    subject: str
    predicate: str
    obj: str
    confidence: float
    source: str
    ts: float


@dataclass
class CausalRelation:
    cause: str
    effect: str
    mechanism: str
    confidence: float


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------

class WorldModel:
    """Persistent structured world model — entities, beliefs, causal relations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(_DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS wm_entities (
                    name        TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL DEFAULT 'concept',
                    summary     TEXT NOT NULL DEFAULT '',
                    confidence  REAL NOT NULL DEFAULT 0.5,
                    updated_ts  REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS wm_beliefs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject     TEXT NOT NULL,
                    predicate   TEXT NOT NULL,
                    obj         TEXT NOT NULL,
                    confidence  REAL NOT NULL DEFAULT 0.5,
                    source      TEXT NOT NULL DEFAULT 'conversation',
                    ts          REAL NOT NULL,
                    UNIQUE(subject, predicate, obj)
                );
                CREATE TABLE IF NOT EXISTS wm_causal (
                    cause       TEXT NOT NULL,
                    effect      TEXT NOT NULL,
                    mechanism   TEXT NOT NULL DEFAULT '',
                    confidence  REAL NOT NULL DEFAULT 0.5,
                    ts          REAL NOT NULL,
                    PRIMARY KEY(cause, effect)
                );
                CREATE INDEX IF NOT EXISTS idx_wm_beliefs_subj ON wm_beliefs(subject);
                CREATE INDEX IF NOT EXISTS idx_wm_beliefs_obj  ON wm_beliefs(obj);
                CREATE INDEX IF NOT EXISTS idx_wm_causal_cause ON wm_causal(cause);
            """)

    # ------------------------------------------------------------------
    # Write interface
    # ------------------------------------------------------------------

    def upsert_entity(self, name: str, entity_type: str = "concept",
                      summary: str = "", confidence: float = 0.6) -> None:
        name = name.strip()[:100]
        if len(name) < 2:
            return
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO wm_entities(name, entity_type, summary, confidence, updated_ts)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        entity_type = excluded.entity_type,
                        summary     = CASE WHEN excluded.summary != '' THEN excluded.summary ELSE summary END,
                        confidence  = MIN(1.0, confidence + 0.04),
                        updated_ts  = excluded.updated_ts
                """, (name, entity_type, summary[:200], confidence, time.time()))

    def upsert_belief(self, subject: str, predicate: str, obj: str,
                      confidence: float = 0.5, source: str = "conversation") -> None:
        subject = subject.strip()[:100]
        obj     = obj.strip()[:150]
        if len(subject) < 2 or len(obj) < 2:
            return
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO wm_beliefs(subject, predicate, obj, confidence, source, ts)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(subject, predicate, obj) DO UPDATE SET
                        confidence = MIN(1.0, confidence + 0.04),
                        ts         = excluded.ts
                """, (subject, predicate, obj, confidence, source, time.time()))

    def upsert_causal(self, cause: str, effect: str, mechanism: str = "",
                      confidence: float = 0.5) -> None:
        cause  = cause.strip()[:100]
        effect = effect.strip()[:100]
        if len(cause) < 2 or len(effect) < 2:
            return
        with self._lock:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO wm_causal(cause, effect, mechanism, confidence, ts)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(cause, effect) DO UPDATE SET
                        mechanism  = CASE WHEN excluded.mechanism != '' THEN excluded.mechanism ELSE mechanism END,
                        confidence = MIN(1.0, (confidence + excluded.confidence) / 2),
                        ts         = excluded.ts
                """, (cause, effect, mechanism[:200], confidence, time.time()))

    def regex_extract_and_store(self, text: str, source: str = "conversation") -> int:
        """Fast regex-based extraction. Returns count of beliefs stored."""
        stored = 0
        for pattern, pred in _COMPILED_PATS:
            for m in pattern.finditer(text):
                try:
                    subj = m.group(1).strip()
                    # group 2 is optional article for 'is a', group 3 is object
                    obj_group = m.lastindex
                    obj = m.group(obj_group).strip()
                    self.upsert_belief(subj, pred, obj, confidence=0.4, source=source)
                    stored += 1
                except Exception:
                    pass
        return stored

    def llm_extract_and_store(self, text: str, reasoning: Any,
                               source: str = "conversation") -> int:
        """LLM-based structured extraction. Runs in background only."""
        if reasoning is None or len(text) < 50:
            return 0
        try:
            prompt  = _EXTRACT_PROMPT.format(text=text[:1500])
            raw     = reasoning.think_simple(
                prompt=prompt,
                system="You are a precise knowledge extractor. Return only valid JSON.",
                max_tokens=400,
            ).strip()
            # Pull JSON from response
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                return 0
            data = json.loads(m.group(0))

            count = 0
            for e in data.get("entities", []):
                self.upsert_entity(
                    e.get("name", ""), e.get("type", "concept"),
                    e.get("summary", ""), confidence=0.65,
                )
                count += 1
            for b in data.get("beliefs", []):
                self.upsert_belief(
                    b.get("subject", ""), b.get("predicate", "is"),
                    b.get("object", ""), confidence=0.6, source=source,
                )
                count += 1
            for c in data.get("causal", []):
                self.upsert_causal(
                    c.get("cause", ""), c.get("effect", ""),
                    c.get("mechanism", ""), confidence=0.6,
                )
                count += 1
            return count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Read / query interface
    # ------------------------------------------------------------------

    def query_context(self, query: str, limit: int = 8) -> str:
        """Return world model context relevant to query (for prompt injection)."""
        words = [w for w in re.findall(r"\b\w{4,}\b", query.lower()) if len(w) > 3]
        if not words:
            return ""

        lines: List[str] = []
        seen: set = set()

        with self._conn() as conn:
            for word in words[:6]:
                like = f"%{word}%"
                for row in conn.execute("""
                    SELECT subject, predicate, obj, confidence
                    FROM wm_beliefs
                    WHERE LOWER(subject) LIKE ? OR LOWER(obj) LIKE ?
                    ORDER BY confidence DESC LIMIT 3
                """, (like, like)).fetchall():
                    entry = f"{row['subject']} {row['predicate']} {row['obj']}"
                    if entry not in seen:
                        seen.add(entry)
                        lines.append(f"  • {entry} (conf:{row['confidence']:.2f})")

                for row in conn.execute("""
                    SELECT cause, effect, confidence
                    FROM wm_causal
                    WHERE LOWER(cause) LIKE ? OR LOWER(effect) LIKE ?
                    ORDER BY confidence DESC LIMIT 2
                """, (like, like)).fetchall():
                    entry = f"{row['cause']} → {row['effect']}"
                    if entry not in seen:
                        seen.add(entry)
                        lines.append(f"  • {entry} (causal, conf:{row['confidence']:.2f})")

            if not lines:
                return ""
            return "--- World Model ---\n" + "\n".join(lines[:limit])

    def stats(self) -> Dict[str, int]:
        with self._conn() as conn:
            return {
                "entities": conn.execute("SELECT COUNT(*) FROM wm_entities").fetchone()[0],
                "beliefs":  conn.execute("SELECT COUNT(*) FROM wm_beliefs").fetchone()[0],
                "causal":   conn.execute("SELECT COUNT(*) FROM wm_causal").fetchone()[0],
            }
