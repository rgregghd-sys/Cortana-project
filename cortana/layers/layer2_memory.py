"""
Layer 2 — Memory (Hierarchical)
Three-tier memory system:
  Tier 1 — Episodic   : SQLite timestamped interaction log (raw events)
  Tier 2 — Semantic   : ChromaDB vector store (similarity recall)
  Tier 3 — Conceptual : Logic matrix — concept nodes + relationship edges
                        Built from reflection output; grows smarter over time.

Recall merges all three tiers, ranked by relevance.
"""
from __future__ import annotations
import sqlite3
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from cortana.models.schemas import ConversationTurn

import chromadb
from chromadb.utils import embedding_functions

from cortana import config


class CortanaMemory:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._working: Dict[str, Any] = {}

        # Tier 2 — ChromaDB semantic store
        self._chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._chroma_client.get_or_create_collection(
            name="cortana_consciousness",
            embedding_function=self._ef,
        )

        # Tiers 1 + 3 — SQLite (episodic + logic matrix)
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
                    wallet        TEXT DEFAULT '',
                    created       DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login    DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
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
            conn.commit()

    # ------------------------------------------------------------------
    # Tier 1 + 2: Store an interaction (episodic + semantic)
    # ------------------------------------------------------------------
    def store(self, interaction: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist an interaction to ChromaDB (vector) and SQLite (episodic)."""
        if metadata is None:
            metadata = {"source": "interaction"}

        doc_id = f"id_{time.time()}"

        with self._lock:
            self._collection.add(
                documents=[interaction],
                metadatas=[metadata],
                ids=[doc_id],
            )
            with sqlite3.connect(config.SQLITE_PATH) as conn:
                conn.execute(
                    "INSERT INTO episodes (content, source) VALUES (?, ?)",
                    (interaction, metadata.get("source", "interaction")),
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
        Semantic recall (Tier 2) + top conceptual matches (Tier 3).
        Returns combined list, semantic results first.
        """
        results: List[str] = []

        # Tier 2 — ChromaDB semantic search
        try:
            count = self._collection.count()
            if count > 0:
                n = min(n_results, count)
                r = self._collection.query(query_texts=[query], n_results=n)
                results.extend(d for d in r.get("documents", [[]])[0] if d)
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
