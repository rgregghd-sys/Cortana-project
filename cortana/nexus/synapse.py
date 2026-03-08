"""
Neural Nexus — Synapses with Hebbian Plasticity.

Synapses connect neurons with a weighted directed edge.
Weight starts at 1.0 and is updated via Hebbian learning:

  Δw = LEARNING_RATE × pre_confidence × post_confidence × reward

Where reward comes from L9 quality_score feedback.
Weights are persisted in SQLite so connections strengthen over sessions.

Inhibitory synapses (weight < 0) suppress downstream neurons.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cortana import config

_DB_PATH       = Path(config.SQLITE_PATH)
_LEARNING_RATE = 0.05
_WEIGHT_MIN    = -1.0
_WEIGHT_MAX    =  2.0
_DECAY_RATE    = 0.001   # slow weight decay toward 1.0 each session


@dataclass
class Synapse:
    """
    Directed weighted connection: source_id → target_id.
    weight > 0: excitatory (amplifies signal)
    weight < 0: inhibitory (reduces signal)
    weight = 1.0: neutral pass-through
    """
    source_id:  str
    target_id:  str
    weight:     float = 1.0
    fire_count: int   = 0
    last_update: float = 0.0

    @property
    def effective_strength(self) -> float:
        """Clamp weight to valid range."""
        return max(_WEIGHT_MIN, min(_WEIGHT_MAX, self.weight))


class SynapseRegistry:
    """
    Manages all synapses in the nexus. Persists to SQLite.
    Thread-safe reads; writes are infrequent (end-of-turn Hebbian update).
    """

    def __init__(self) -> None:
        self._synapses: Dict[Tuple[str, str], Synapse] = {}
        self._init_db()
        self._load()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nexus_synapses (
                    source_id   TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    weight      REAL NOT NULL DEFAULT 1.0,
                    fire_count  INTEGER NOT NULL DEFAULT 0,
                    last_update REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (source_id, target_id)
                )
            """)

    def _load(self) -> None:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT source_id, target_id, weight, fire_count, last_update "
                    "FROM nexus_synapses"
                ).fetchall()
            for r in rows:
                key = (r["source_id"], r["target_id"])
                self._synapses[key] = Synapse(
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    weight=r["weight"],
                    fire_count=r["fire_count"],
                    last_update=r["last_update"],
                )
        except Exception:
            pass

    def get_weight(self, source_id: str, target_id: str) -> float:
        """Return the synaptic weight, defaulting to 1.0 if unseen."""
        syn = self._synapses.get((source_id, target_id))
        return syn.effective_strength if syn else 1.0

    def register(self, source_id: str, target_id: str,
                 initial_weight: float = 1.0) -> None:
        """Ensure a synapse exists between two neurons."""
        key = (source_id, target_id)
        if key not in self._synapses:
            self._synapses[key] = Synapse(
                source_id=source_id,
                target_id=target_id,
                weight=initial_weight,
            )

    def hebbian_update(self, source_id: str, target_id: str,
                       pre_confidence: float, post_confidence: float,
                       reward: float) -> None:
        """
        Hebbian plasticity: strengthen connections that produce good results.
        reward = quality_score from L9 (0.0–1.0)
        """
        key = (source_id, target_id)
        if key not in self._synapses:
            self._synapses[key] = Synapse(source_id=source_id, target_id=target_id)

        syn = self._synapses[key]
        delta = _LEARNING_RATE * pre_confidence * post_confidence * (reward - 0.5)
        syn.weight = max(_WEIGHT_MIN, min(_WEIGHT_MAX, syn.weight + delta))
        syn.fire_count += 1
        syn.last_update = time.time()

        self._persist(syn)

    def decay_all(self) -> None:
        """Slowly decay weights back toward 1.0 (homeostatic plasticity)."""
        for syn in self._synapses.values():
            syn.weight += _DECAY_RATE * (1.0 - syn.weight)
        # Batch persist
        try:
            with self._conn() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO nexus_synapses "
                    "(source_id, target_id, weight, fire_count, last_update) "
                    "VALUES(?,?,?,?,?)",
                    [(s.source_id, s.target_id, s.weight, s.fire_count, s.last_update)
                     for s in self._synapses.values()]
                )
        except Exception:
            pass

    def _persist(self, syn: Synapse) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO nexus_synapses "
                    "(source_id, target_id, weight, fire_count, last_update) "
                    "VALUES(?,?,?,?,?)",
                    (syn.source_id, syn.target_id, syn.weight,
                     syn.fire_count, syn.last_update)
                )
        except Exception:
            pass

    def all_synapses(self) -> List[dict]:
        return [
            {"source": s.source_id, "target": s.target_id,
             "weight": round(s.weight, 4), "fires": s.fire_count}
            for s in self._synapses.values()
        ]


# Module-level singleton
synapse_registry = SynapseRegistry()
