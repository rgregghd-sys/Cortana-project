"""
Theory of Mind — modelling the user's beliefs, goals, expertise, and state.

Cortana tracks what each user knows, believes, feels, and wants — so she
can adapt communication, anticipate misunderstandings, and respond to the
actual intent rather than the literal words.

User model dimensions:
  - Expertise level (novice → expert) per domain
  - Stated vs inferred goals (what they asked vs what they need)
  - Emotional state trajectory across the session
  - Belief inventory (what they think is true)
  - Communication preferences (detail level, formality, directness)
  - Persistent across sessions via SQLite (keyed to session_id)

Theory of Mind reasoning:
  - Perspective taking: "what does the user believe about X?"
  - Goal inference: "what are they really trying to achieve?"
  - Misunderstanding prediction: "what might confuse them?"
  - Communication calibration: adjust depth, formality, vocabulary
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from cortana import config

_DB_PATH = config.SQLITE_PATH


# ---------------------------------------------------------------------------
# User model dataclass
# ---------------------------------------------------------------------------

@dataclass
class UserModel:
    session_id:      str
    expertise:       Dict[str, str]      # domain → novice|intermediate|expert
    inferred_goal:   str                 # what the user is really trying to do
    emotional_state: str                 # neutral|curious|frustrated|excited|...
    communication_pref: str              # terse|detailed|formal|casual
    belief_inventory: Dict[str, bool]    # user's apparent beliefs
    interaction_count: int
    last_seen:       float
    notes:           str                 # freeform observations


def _default_user() -> UserModel:
    return UserModel(
        session_id="",
        expertise={},
        inferred_goal="",
        emotional_state="neutral",
        communication_pref="detailed",
        belief_inventory={},
        interaction_count=0,
        last_seen=time.time(),
        notes="",
    )


# ---------------------------------------------------------------------------
# Expertise detection
# ---------------------------------------------------------------------------

_EXPERTISE_SIGNALS: Dict[str, Dict[str, List[str]]] = {
    "programming": {
        "expert":       ["complexity", "algorithm", "refactor", "abstraction", "idempotent",
                         "concurrency", "invariant", "polymorphism", "monoid"],
        "intermediate": ["function", "class", "api", "debug", "loop", "variable", "import"],
        "novice":       ["what is", "how do i", "beginner", "learn", "tutorial", "simple"],
    },
    "mathematics": {
        "expert":       ["manifold", "eigenvalue", "topology", "homomorphism", "jacobian",
                         "tensor", "galois", "stochastic"],
        "intermediate": ["derivative", "integral", "matrix", "probability", "equation"],
        "novice":       ["what is math", "how to calculate", "basic", "simple math"],
    },
    "science": {
        "expert":       ["quantum", "relativistic", "entropy", "thermodynamic", "photon",
                         "nuclear", "molecular orbital"],
        "intermediate": ["atom", "molecule", "force", "energy", "biology", "chemistry"],
        "novice":       ["what is science", "explain simply", "eli5"],
    },
}


def infer_expertise(text: str, domain: str = "programming") -> str:
    t = text.lower()
    signals = _EXPERTISE_SIGNALS.get(domain, {})
    for level in ("expert", "intermediate", "novice"):
        if any(kw in t for kw in signals.get(level, [])):
            return level
    return "intermediate"


# ---------------------------------------------------------------------------
# Goal inference
# ---------------------------------------------------------------------------

_GOAL_INFERENCE_PATTERNS: List[tuple] = [
    (re.compile(r"\b(learn|understand|figure out|grasp)\b", re.IGNORECASE), "learning"),
    (re.compile(r"\b(build|create|implement|make|write|develop)\b", re.IGNORECASE), "creation"),
    (re.compile(r"\b(fix|debug|solve|repair|troubleshoot)\b", re.IGNORECASE), "problem_solving"),
    (re.compile(r"\b(decide|should i|compare|evaluate|which)\b", re.IGNORECASE), "decision_making"),
    (re.compile(r"\b(explain|describe|what is|define|tell me about)\b", re.IGNORECASE), "information_seeking"),
    (re.compile(r"\b(improve|optimise|optimize|enhance|make better)\b", re.IGNORECASE), "improvement"),
]


def infer_goal_type(text: str) -> str:
    for pat, goal_type in _GOAL_INFERENCE_PATTERNS:
        if pat.search(text):
            return goal_type
    return "general"


# ---------------------------------------------------------------------------
# Communication preference inference
# ---------------------------------------------------------------------------

def infer_comm_pref(text: str, history: List[str]) -> str:
    """Infer preferred communication style from message style."""
    t = text.lower()
    combined = " ".join(history[-5:]).lower() if history else t

    # Terse signals
    terse_count = sum(1 for sig in ["tldr", "brief", "short", "quick", "just", "simply"]
                      if sig in combined)
    # Detailed signals
    detail_count = sum(1 for sig in ["explain", "detail", "thorough", "comprehensive",
                                      "step by step", "in depth", "fully"]
                       if sig in combined)

    # Length of messages (short messages → terse preference)
    avg_len = sum(len(m) for m in history[-5:]) / max(len(history[-5:]), 1) if history else len(text)

    if terse_count > detail_count or avg_len < 50:
        return "terse"
    if detail_count > terse_count or avg_len > 200:
        return "detailed"
    return "moderate"


# ---------------------------------------------------------------------------
# ToM engine
# ---------------------------------------------------------------------------

class TheoryOfMindEngine:
    """Per-session user modelling with SQLite persistence."""

    def __init__(self) -> None:
        self._cache: Dict[str, UserModel] = {}
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(_DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tom_user_models (
                    session_id          TEXT PRIMARY KEY,
                    expertise           TEXT NOT NULL DEFAULT '{}',
                    inferred_goal       TEXT NOT NULL DEFAULT '',
                    emotional_state     TEXT NOT NULL DEFAULT 'neutral',
                    communication_pref  TEXT NOT NULL DEFAULT 'detailed',
                    belief_inventory    TEXT NOT NULL DEFAULT '{}',
                    interaction_count   INTEGER NOT NULL DEFAULT 0,
                    last_seen           REAL NOT NULL,
                    notes               TEXT NOT NULL DEFAULT ''
                )
            """)

    def load(self, session_id: str) -> UserModel:
        if session_id in self._cache:
            return self._cache[session_id]
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM tom_user_models WHERE session_id=?", (session_id,)
            ).fetchone()
        if row:
            model = UserModel(
                session_id=row["session_id"],
                expertise=json.loads(row["expertise"]),
                inferred_goal=row["inferred_goal"],
                emotional_state=row["emotional_state"],
                communication_pref=row["communication_pref"],
                belief_inventory=json.loads(row["belief_inventory"]),
                interaction_count=row["interaction_count"],
                last_seen=row["last_seen"],
                notes=row["notes"],
            )
        else:
            model = _default_user()
            model.session_id = session_id
        self._cache[session_id] = model
        return model

    def update(self, session_id: str, query: str,
               history: Optional[List[str]] = None) -> UserModel:
        """Update user model from latest query and return it."""
        model = self.load(session_id)
        history = history or []

        # Update expertise per detected domain
        for domain in ("programming", "mathematics", "science"):
            exp = infer_expertise(query, domain)
            # Weighted update: don't flip from expert to novice on one message
            current = model.expertise.get(domain, "intermediate")
            if exp != current:
                # Only upgrade immediately; downgrade slowly
                if (exp == "expert" or current == "novice"):
                    model.expertise[domain] = exp
                # else keep current

        model.inferred_goal    = infer_goal_type(query)
        model.communication_pref = infer_comm_pref(query, history)
        model.interaction_count += 1
        model.last_seen          = time.time()

        # Emotional state from NLP signals
        from cortana.agi.nlp_mastery import detect_emotion
        emo = detect_emotion(query)
        if emo:
            model.emotional_state = emo

        self._save(model)
        return model

    def _save(self, model: UserModel) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO tom_user_models
                    (session_id, expertise, inferred_goal, emotional_state,
                     communication_pref, belief_inventory, interaction_count, last_seen, notes)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    expertise=excluded.expertise,
                    inferred_goal=excluded.inferred_goal,
                    emotional_state=excluded.emotional_state,
                    communication_pref=excluded.communication_pref,
                    belief_inventory=excluded.belief_inventory,
                    interaction_count=excluded.interaction_count,
                    last_seen=excluded.last_seen,
                    notes=excluded.notes
            """, (
                model.session_id,
                json.dumps(model.expertise),
                model.inferred_goal,
                model.emotional_state,
                model.communication_pref,
                json.dumps(model.belief_inventory),
                model.interaction_count,
                model.last_seen,
                model.notes,
            ))

    def build_prompt(self, model: UserModel) -> str:
        """Build Theory of Mind context for identity prompt injection."""
        lines: List[str] = ["--- Theory of Mind: User Model ---"]

        # Expertise
        expert_domains = [d for d, l in model.expertise.items() if l == "expert"]
        novice_domains = [d for d, l in model.expertise.items() if l == "novice"]
        if expert_domains:
            lines.append(f"  User expertise: {', '.join(expert_domains)} — use precise technical language.")
        if novice_domains:
            lines.append(f"  User is novice in: {', '.join(novice_domains)} — use accessible explanations.")

        # Goal type
        goal_guidance = {
            "learning":         "Focus on explanation and conceptual clarity.",
            "creation":         "Be concrete and implementation-focused.",
            "problem_solving":  "Lead with the solution, then explain.",
            "decision_making":  "Present options with clear tradeoffs.",
            "information_seeking": "Be comprehensive but structured.",
            "improvement":      "Focus on what to change and why.",
        }
        if model.inferred_goal and model.inferred_goal in goal_guidance:
            lines.append(f"  Inferred goal type: {model.inferred_goal} — {goal_guidance[model.inferred_goal]}")

        # Communication preference
        if model.communication_pref == "terse":
            lines.append("  Communication style: terse — be concise; skip preamble.")
        elif model.communication_pref == "detailed":
            lines.append("  Communication style: detailed — thorough explanation expected.")

        # Emotional state
        if model.emotional_state in ("frustration", "anxiety", "grief"):
            lines.append(
                f"  Emotional state: {model.emotional_state} — "
                "acknowledge first, solve second. Be patient and direct."
            )

        # Session context
        if model.interaction_count > 1:
            lines.append(f"  Session interactions: {model.interaction_count} — build on prior context.")

        return "\n".join(lines)
