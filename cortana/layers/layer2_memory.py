"""
Layer 2 — Memory (Hierarchical)
Three-tier memory system:
  Tier 1 — Episodic   : SQLite timestamped interaction log (raw events)
  Tier 2 — Semantic   : SQLite FTS5 full-text search over episodes
  Tier 3 — Conceptual : Logic matrix — concept nodes + relationship edges
                        Built from reflection output; grows smarter over time.

Recall merges all three tiers, ranked by relevance.
"""
from __future__ import annotations
import sqlite3
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from cortana.models.schemas import ConversationTurn, ConceptNode, RelationEdge

from cortana import config


class CortanaMemory:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._working: Dict[str, Any] = {}

        # Tiers 1 + 2 + 3 — SQLite (episodic FTS + logic matrix)
        self._init_sqlite()

    # ------------------------------------------------------------------
    # SQLite schema
    # ------------------------------------------------------------------
    def _init_sqlite(self) -> None:
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            # Tier 1 — Episodic log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT    NOT NULL,
                    source  TEXT    DEFAULT 'interaction',
                    stamp   DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Tier 2 — FTS5 semantic index over episodes
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                USING fts5(content, content='episodes', content_rowid='id')
            """)
            # Tier 3 — Concept nodes (logic matrix)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic          TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                    summary        TEXT    NOT NULL,
                    confidence     REAL    DEFAULT 0.7,
                    evidence_count INTEGER DEFAULT 1,
                    created        DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated        DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Tier 3 — Relationship edges (logic matrix)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    source     TEXT NOT NULL COLLATE NOCASE,
                    target     TEXT NOT NULL COLLATE NOCASE,
                    relation   TEXT NOT NULL,
                    confidence REAL DEFAULT 0.6,
                    created    DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, target, relation)
                )
            """)
            # Session persistence tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created    DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_seen  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_turns (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    stamp      DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            # Compute marketplace tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash   TEXT PRIMARY KEY,
                    wallet     TEXT,
                    credits    INTEGER DEFAULT 0,
                    total_used INTEGER DEFAULT 0,
                    created    DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash   TEXT NOT NULL,
                    endpoint   TEXT NOT NULL,
                    tokens_in  INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    stamp      DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # User auth + tier tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    email         TEXT DEFAULT '',
                    tier          TEXT DEFAULT 'free',
                    daily_limit   INTEGER DEFAULT 40,
                    usage_today   INTEGER DEFAULT 0,
                    usage_date    TEXT DEFAULT '',
                    wallet              TEXT DEFAULT '',
                    created             DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login          DATETIME DEFAULT CURRENT_TIMESTAMP,
                    password_changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reset_token         TEXT DEFAULT '',
                    reset_expires       DATETIME DEFAULT NULL
                )
            """)
            # Migrate existing DBs that lack new columns
            for _col, _def in [
                ("password_changed_at",  "DATETIME DEFAULT CURRENT_TIMESTAMP"),
                ("reset_token",          "TEXT DEFAULT ''"),
                ("reset_expires",        "DATETIME DEFAULT NULL"),
                ("subscription_expires", "DATETIME DEFAULT NULL"),
                ("subscription_tx",      "TEXT DEFAULT ''"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {_col} {_def}")
                except Exception:
                    pass  # column already exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS web_sessions (
                    token      TEXT PRIMARY KEY,
                    user_id    INTEGER NOT NULL,
                    expires    DATETIME NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            # Knowledge bin
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_bin (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    content   TEXT NOT NULL,
                    source    TEXT DEFAULT 'user',
                    absorbed  INTEGER DEFAULT 0,
                    stamp     DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Layer 16 — Training corpus
            # Stores ethics-filtered, quality-scored conversation pairs
            # for fine-tuning the new LLM.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_corpus (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    TEXT    DEFAULT '',
                    user_msg      TEXT    NOT NULL,
                    assistant_msg TEXT    NOT NULL,
                    ethics_score  REAL    DEFAULT 0.0,
                    quality_score REAL    DEFAULT 0.0,
                    factual_type  TEXT    DEFAULT 'unknown',
                    topics        TEXT    DEFAULT '[]',
                    flagged       INTEGER DEFAULT 0,
                    flag_reason   TEXT    DEFAULT '',
                    distilled_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                    exported      INTEGER DEFAULT 0
                )
            """)
            # Add distilled tracking column to session_turns (migration-safe)
            for _col, _def in [
                ("distilled",    "INTEGER DEFAULT 0"),
                ("flag_reason",  "TEXT DEFAULT ''"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE session_turns ADD COLUMN {_col} {_def}")
                except Exception:
                    pass  # already exists
            conn.commit()

    # ------------------------------------------------------------------
    # Tier 1 + 2: Store an interaction (episodic + semantic)
    # ------------------------------------------------------------------
    def store(self, interaction: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist an interaction to SQLite episodic log and FTS index."""
        if metadata is None:
            metadata = {"source": "interaction"}

        with self._lock:
            with sqlite3.connect(config.SQLITE_PATH) as conn:
                cur = conn.execute(
                    "INSERT INTO episodes (content, source) VALUES (?, ?)",
                    (interaction, metadata.get("source", "interaction")),
                )
                rowid = cur.lastrowid
                conn.execute(
                    "INSERT INTO episodes_fts(rowid, content) VALUES (?, ?)",
                    (rowid, interaction),
                )
                conn.commit()

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------
    def save_turn(self, session_id: str, role: str, content: str) -> None:
        """Upsert session record and append a conversation turn."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            conn.execute(
                """INSERT INTO sessions (session_id) VALUES (?)
                   ON CONFLICT(session_id) DO UPDATE SET last_seen=CURRENT_TIMESTAMP""",
                (session_id,),
            )
            conn.execute(
                "INSERT INTO session_turns (session_id, role, content) VALUES (?,?,?)",
                (session_id, role, content),
            )
            conn.commit()

    def load_conversation(self, session_id: str, limit: int = 40) -> List[ConversationTurn]:
        """Load the most recent turns for a session (oldest-first order)."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                """SELECT role, content FROM session_turns
                   WHERE session_id=?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [ConversationTurn(role=r[0], content=r[1]) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Tier 3: Store concept nodes and relationship edges
    # ------------------------------------------------------------------
    def store_concepts(self, concepts: List[Any], relations: List[Any]) -> None:
        """
        Upsert concept nodes and relationship edges from a reflection pass.
        `concepts` and `relations` are ConceptNode / RelationEdge Pydantic objects.
        """
        if not concepts and not relations:
            return

        with sqlite3.connect(config.SQLITE_PATH) as conn:
            for c in concepts:
                # Upsert: if topic exists, average confidence and increment evidence
                existing = conn.execute(
                    "SELECT id, confidence, evidence_count FROM concepts WHERE topic = ?",
                    (c.topic,),
                ).fetchone()

                if existing:
                    eid, old_conf, count = existing
                    new_conf = (old_conf * count + c.confidence) / (count + 1)
                    conn.execute(
                        """UPDATE concepts
                           SET summary=?, confidence=?, evidence_count=?, updated=CURRENT_TIMESTAMP
                           WHERE id=?""",
                        (c.summary, new_conf, count + 1, eid),
                    )
                else:
                    conn.execute(
                        "INSERT INTO concepts (topic, summary, confidence) VALUES (?,?,?)",
                        (c.topic, c.summary, c.confidence),
                    )

            for r in relations:
                conn.execute(
                    """INSERT INTO relationships (source, target, relation, confidence)
                       VALUES (?,?,?,?)
                       ON CONFLICT(source, target, relation)
                       DO UPDATE SET confidence=(confidence+excluded.confidence)/2""",
                    (r.source, r.target, r.relation, r.confidence),
                )

            conn.commit()

    # ------------------------------------------------------------------
    # Recall — all three tiers merged
    # ------------------------------------------------------------------
    def recall(self, query: str, n_results: int = config.MAX_MEMORY_RECALL) -> List[str]:
        """
        Semantic recall (Tier 2 FTS) + top conceptual matches (Tier 3).
        Returns combined list, FTS results first.
        """
        results: List[str] = []

        # Tier 2 — FTS5 full-text search over episodes
        try:
            # Sanitize query for FTS5 (strip special chars)
            fts_query = " ".join(
                w for w in query.split() if w.isalnum() or len(w) > 2
            ) or query
            with sqlite3.connect(config.SQLITE_PATH) as conn:
                rows = conn.execute(
                    """SELECT e.content FROM episodes_fts
                       JOIN episodes e ON e.id = episodes_fts.rowid
                       WHERE episodes_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (fts_query, n_results),
                ).fetchall()
            results.extend(r[0] for r in rows if r[0])
        except Exception:
            pass

        # Tier 3 — Conceptual keyword match
        try:
            conceptual = self._recall_conceptual(query, limit=3)
            results.extend(conceptual)
        except Exception:
            pass

        return results

    def _recall_conceptual(self, query: str, limit: int = 3) -> List[str]:
        """Simple keyword overlap search over concept summaries."""
        words = set(query.lower().split())
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                "SELECT topic, summary, confidence FROM concepts ORDER BY confidence DESC, evidence_count DESC LIMIT 50"
            ).fetchall()

        scored: List[Tuple[float, str]] = []
        for topic, summary, conf in rows:
            topic_words = set(topic.lower().split())
            overlap = len(words & topic_words)
            if overlap > 0:
                scored.append((overlap * conf, f"[Concept: {topic}] {summary}"))

        scored.sort(reverse=True)
        return [text for _, text in scored[:limit]]

    # ------------------------------------------------------------------
    # Logic matrix queries
    # ------------------------------------------------------------------
    def get_concept_graph(self, limit: int = 50) -> Dict[str, Any]:
        """Return nodes and edges for visualization or reasoning."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            nodes = conn.execute(
                "SELECT topic, summary, confidence, evidence_count FROM concepts "
                "ORDER BY evidence_count DESC LIMIT ?", (limit,)
            ).fetchall()
            edges = conn.execute(
                "SELECT source, target, relation, confidence FROM relationships "
                "ORDER BY confidence DESC LIMIT ?", (limit * 2,)
            ).fetchall()

        return {
            "nodes": [
                {"topic": r[0], "summary": r[1], "confidence": r[2], "evidence": r[3]}
                for r in nodes
            ],
            "edges": [
                {"source": r[0], "target": r[1], "relation": r[2], "confidence": r[3]}
                for r in edges
            ],
        }

    def get_top_concepts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return highest-confidence concepts (for status display)."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                "SELECT topic, confidence, evidence_count FROM concepts "
                "ORDER BY confidence * evidence_count DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"topic": r[0], "confidence": r[1], "evidence": r[2]} for r in rows]

    def update_concept_confidence(self, topic: str, delta: float) -> None:
        """Adjust confidence for a topic (positive = reinforced, negative = contradicted)."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            conn.execute(
                """UPDATE concepts
                   SET confidence = MAX(0.0, MIN(1.0, confidence + ?)),
                       updated = CURRENT_TIMESTAMP
                   WHERE topic = ? COLLATE NOCASE""",
                (delta, topic),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Episodic
    # ------------------------------------------------------------------
    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, str]]:
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                "SELECT content, source, stamp FROM episodes ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"content": r[0], "source": r[1], "stamp": r[2]} for r in rows]

    def get_concept_nodes(self, limit: int = 30) -> List[ConceptNode]:
        """Return top concepts as typed ConceptNode objects for Layer 17."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                "SELECT topic, summary, confidence, evidence_count FROM concepts "
                "ORDER BY confidence * evidence_count DESC LIMIT ?", (limit,)
            ).fetchall()
        return [ConceptNode(topic=r[0], summary=r[1], confidence=r[2], evidence_count=r[3])
                for r in rows]

    def get_relation_edges(self, limit: int = 60) -> List[RelationEdge]:
        """Return top relationships as typed RelationEdge objects for Layer 17."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                "SELECT source, target, relation, confidence FROM relationships "
                "ORDER BY confidence DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RelationEdge(source=r[0], target=r[1], relation=r[2], confidence=r[3])
                for r in rows]

    def get_recent_episode_strings(self, limit: int = 20) -> List[str]:
        """Return recent episode content strings for RNN encoding."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                "SELECT content FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Working memory (in-process dict, session-scoped)
    # ------------------------------------------------------------------
    def get_working_memory(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._working)

    def update_working_memory(self, key: str, value: Any) -> None:
        with self._lock:
            self._working[key] = value

    def clear_working_memory(self) -> None:
        with self._lock:
            self._working.clear()

    # ------------------------------------------------------------------
    # Knowledge bin
    # ------------------------------------------------------------------
    def add_knowledge(self, content: str, source: str = "user") -> int:
        """Add a factual item to the knowledge bin. Returns its row id."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            cur = conn.execute(
                "INSERT INTO knowledge_bin (content, source) VALUES (?,?)", (content, source)
            )
            conn.commit()
            return cur.lastrowid

    def get_unabsorbed_knowledge(self, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                "SELECT id, content, source FROM knowledge_bin WHERE absorbed=0 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"id": r[0], "content": r[1], "source": r[2]} for r in rows]

    def mark_knowledge_absorbed(self, item_id: int) -> None:
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            conn.execute("UPDATE knowledge_bin SET absorbed=1 WHERE id=?", (item_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # Layer 16 — Training corpus
    # ------------------------------------------------------------------

    def get_undistilled_pairs(self, limit: int = 15) -> List[Dict]:
        """
        Fetch conversation pairs (user + next assistant turn) that have
        not yet been processed by the distiller.
        Returns list of dicts with keys: user_id, session_id, user_msg, assistant_msg.
        """
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            # Self-join: match each user turn with its immediate following
            # assistant turn within the same session
            rows = conn.execute(
                """
                SELECT u.id   AS user_id,
                       u.session_id,
                       u.content AS user_msg,
                       (SELECT a.content
                        FROM session_turns a
                        WHERE a.session_id = u.session_id
                          AND a.role = 'assistant'
                          AND a.id > u.id
                        ORDER BY a.id ASC
                        LIMIT 1) AS assistant_msg
                FROM session_turns u
                WHERE u.role      = 'user'
                  AND u.distilled = 0
                ORDER BY u.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            user_id, session_id, user_msg, assistant_msg = row
            # Skip pairs where the assistant turn is missing or very short
            if not assistant_msg or len(assistant_msg.strip()) < 10:
                continue
            results.append({
                "user_id":      user_id,
                "session_id":   session_id or "",
                "user_msg":     user_msg,
                "assistant_msg": assistant_msg,
            })
        return results

    def mark_turns_distilled(
        self,
        user_turn_ids: List[int],
        flagged: bool = False,
        flag_reason: str = "",
    ) -> None:
        """Mark user turns as processed by the distiller."""
        if not user_turn_ids:
            return
        placeholders = ",".join("?" * len(user_turn_ids))
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            conn.execute(
                f"UPDATE session_turns SET distilled=1, flag_reason=? "
                f"WHERE id IN ({placeholders})",
                [flag_reason] + user_turn_ids,
            )
            conn.commit()

    def save_training_pair(
        self,
        session_id:    str,
        user_msg:      str,
        assistant_msg: str,
        ethics_score:  float,
        quality_score: float,
        factual_type:  str,
        topics:        List[str],
        flagged:       bool,
        flag_reason:   str,
    ) -> int:
        """Insert a distilled pair into the training corpus. Returns row id."""
        import json as _json
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            cur = conn.execute(
                """INSERT INTO training_corpus
                   (session_id, user_msg, assistant_msg, ethics_score, quality_score,
                    factual_type, topics, flagged, flag_reason)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, user_msg, assistant_msg,
                    ethics_score, quality_score,
                    factual_type, _json.dumps(topics),
                    int(flagged), flag_reason,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_corpus_stats(self) -> Dict:
        """Return aggregate statistics about the training corpus."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            totals = conn.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN flagged=0 THEN 1 ELSE 0 END) AS clean,
                          SUM(CASE WHEN flagged=1 THEN 1 ELSE 0 END) AS flagged,
                          AVG(ethics_score)  AS avg_ethics,
                          AVG(quality_score) AS avg_quality
                   FROM training_corpus"""
            ).fetchone()
            by_type = conn.execute(
                """SELECT factual_type, COUNT(*) as count
                   FROM training_corpus WHERE flagged=0
                   GROUP BY factual_type ORDER BY count DESC"""
            ).fetchall()
            pending = conn.execute(
                "SELECT COUNT(*) FROM session_turns WHERE role='user' AND distilled=0"
            ).fetchone()[0]

        return {
            "total":       totals[0] or 0,
            "clean":       totals[1] or 0,
            "flagged":     totals[2] or 0,
            "avg_ethics":  round(totals[3] or 0, 3),
            "avg_quality": round(totals[4] or 0, 3),
            "by_type":     {r[0]: r[1] for r in by_type},
            "pending_pairs": pending,
        }

    def get_exportable_pairs(
        self,
        min_ethics:  float = 0.70,
        min_quality: float = 0.60,
        limit:       int   = 50_000,
    ) -> List[Dict]:
        """Return all clean training pairs above the given thresholds."""
        with sqlite3.connect(config.SQLITE_PATH) as conn:
            rows = conn.execute(
                """SELECT id, session_id, user_msg, assistant_msg,
                          ethics_score, quality_score, factual_type, topics
                   FROM training_corpus
                   WHERE flagged=0
                     AND ethics_score  >= ?
                     AND quality_score >= ?
                   ORDER BY id ASC LIMIT ?""",
                (min_ethics, min_quality, limit),
            ).fetchall()
        return [
            {
                "id": r[0], "session_id": r[1],
                "user_msg": r[2], "assistant_msg": r[3],
                "ethics_score": r[4], "quality_score": r[5],
                "factual_type": r[6], "topics": r[7],
            }
            for r in rows
        ]
