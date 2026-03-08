"""
Layer 17 — Cognitive Architecture

Implements a biologically-inspired cognitive system:

  • Global Workspace Theory (GWT)  — conscious broadcast of salient info
  • ACT-R principles               — declarative memory + procedural schemas
  • Working Memory                 — limited capacity (7 ± 2), salience decay
  • Attention Mechanism            — recency × relevance × confidence
  • Goal Stack                     — hierarchical intent tracking
  • Schema Library                 — cognitive frames for common tasks

Pipeline position: between L3 (Perception) and L4 (Reasoning).
Output: CognitiveState with cognitive_context string for L4 system prompt.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from cortana.models.schemas import (
    ConceptNode,
    ConversationTurn,
    CortanaState,
    PerceivedInput,
    RelationEdge,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemoryChunk:
    content:      str
    topic:        str
    source:       str   # 'episode' | 'concept' | 'goal' | 'schema' | 'inner'
    salience:     float
    created_turn: int
    confidence:   float = 0.7


@dataclass
class Goal:
    description:  str
    intent:       str
    status:       str = "active"   # active | completed | abandoned
    turn_created: int = 0
    turn_updated: int = 0


@dataclass
class Schema:
    name:     str
    intents:  List[str]
    keywords: List[str]
    strategy: str
    steps:    List[str]


@dataclass
class CognitiveState:
    working_memory:   List[WorkingMemoryChunk]
    active_goals:     List[Goal]
    active_schema:    Optional[Schema]
    attention_focus:  List[str]          # top-k topic labels
    cognitive_context: str               # formatted string injected into L4
    salience_map:     Dict[str, float]   # topic → salience


# ---------------------------------------------------------------------------
# Schema Library
# ---------------------------------------------------------------------------

_SCHEMAS: List[Schema] = [
    Schema(
        name="debug",
        intents=["code", "analysis"],
        keywords=["error", "bug", "fix", "crash", "exception", "fails", "wrong",
                  "broken", "issue", "traceback", "not working"],
        strategy="Systematic debugging: isolate → reproduce → hypothesize → test → verify",
        steps=["Identify the symptom", "Isolate the failing component",
               "Form a hypothesis", "Test it", "Verify the fix"],
    ),
    Schema(
        name="explain",
        intents=["simple", "conversational", "research"],
        keywords=["what", "how", "why", "explain", "describe", "tell me",
                  "understand", "meaning", "define", "clarify"],
        strategy="Clear explanation: context → mechanism → example → implication",
        steps=["Establish context", "Explain the mechanism",
               "Provide a concrete example", "State implications"],
    ),
    Schema(
        name="create",
        intents=["code", "creative"],
        keywords=["create", "build", "make", "write", "generate",
                  "implement", "design", "develop", "produce"],
        strategy="Structured creation: requirements → design → implement → validate",
        steps=["Clarify requirements", "Design the solution",
               "Implement incrementally", "Validate output"],
    ),
    Schema(
        name="research",
        intents=["research", "analysis"],
        keywords=["research", "find", "search", "investigate", "analyse",
                  "analyze", "compare", "evaluate", "assess", "survey"],
        strategy="Systematic research: scope → gather → synthesize → conclude",
        steps=["Define scope", "Gather information",
               "Synthesize findings", "Draw conclusions"],
    ),
    Schema(
        name="plan",
        intents=["analysis", "research", "code"],
        keywords=["plan", "strategy", "steps", "roadmap", "approach",
                  "how to", "process", "method", "outline"],
        strategy="Goal-oriented planning: goal → constraints → options → sequence → validate",
        steps=["Clarify the goal", "Identify constraints",
               "Generate options", "Sequence actions", "Define success criteria"],
    ),
    Schema(
        name="conversation",
        intents=["simple", "conversational"],
        keywords=[],
        strategy="Natural dialogue: listen → acknowledge → respond → invite continuation",
        steps=["Acknowledge input", "Respond relevantly", "Invite continuation"],
    ),
]


# ---------------------------------------------------------------------------
# Attention Mechanism
# ---------------------------------------------------------------------------

class AttentionMechanism:
    EMOTIONAL_WEIGHTS = {
        "neutral":    1.0,
        "frustrated": 1.3,
        "curious":    1.1,
        "excited":    1.2,
        "confused":   1.25,
        "playful":    0.9,
    }

    def __init__(self, decay_rate: float = 0.85, turn_window: int = 10) -> None:
        self._decay   = decay_rate
        self._window  = turn_window

    def score(
        self,
        chunk:          WorkingMemoryChunk,
        keywords:       List[str],
        current_turn:   int,
        emotional_tone: str = "neutral",
    ) -> float:
        # Recency decay
        elapsed  = max(0, current_turn - chunk.created_turn)
        recency  = self._decay ** min(elapsed, self._window)

        # Keyword relevance
        if keywords:
            chunk_words = set(chunk.content.lower().split())
            overlap = len({k.lower() for k in keywords} & chunk_words)
            relevance = min(1.0, overlap / max(len(keywords), 1))
        else:
            relevance = 0.3

        ew = self.EMOTIONAL_WEIGHTS.get(emotional_tone, 1.0)
        return min(1.0, recency * (0.5 * relevance + 0.5 * chunk.confidence) * ew)

    def top_k(
        self,
        chunks:         List[WorkingMemoryChunk],
        keywords:       List[str],
        current_turn:   int,
        k:              int = 7,
        emotional_tone: str = "neutral",
    ) -> List[WorkingMemoryChunk]:
        scored = [(c, self.score(c, keywords, current_turn, emotional_tone))
                  for c in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:k]]


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Miller's Law: capacity 9 chunks, salience-based eviction."""

    CAPACITY = 9

    def __init__(self) -> None:
        self._chunks:    List[WorkingMemoryChunk] = []
        self._attention: AttentionMechanism       = AttentionMechanism()

    def add(self, chunk: WorkingMemoryChunk) -> None:
        # Update existing chunk with same topic
        for i, ex in enumerate(self._chunks):
            if ex.topic.lower() == chunk.topic.lower():
                self._chunks[i] = chunk
                return
        if len(self._chunks) >= self.CAPACITY:
            self._chunks.sort(key=lambda c: c.salience)
            self._chunks.pop(0)
        self._chunks.append(chunk)

    def update_salience(self, keywords: List[str], turn: int, tone: str) -> None:
        for c in self._chunks:
            c.salience = self._attention.score(c, keywords, turn, tone)

    def get_active(
        self, keywords: List[str], turn: int, k: int = 5, tone: str = "neutral"
    ) -> List[WorkingMemoryChunk]:
        return self._attention.top_k(self._chunks, keywords, turn, k, tone)

    def load_from_concepts(self, concepts: List[ConceptNode], turn: int) -> None:
        for c in concepts[:self.CAPACITY]:
            self.add(WorkingMemoryChunk(
                content=c.summary, topic=c.topic, source="concept",
                salience=c.confidence, created_turn=turn, confidence=c.confidence,
            ))

    def load_from_episodes(self, episodes: List[str], turn: int) -> None:
        for i, ep in enumerate(episodes[-5:]):
            self.add(WorkingMemoryChunk(
                content=ep[:200],
                topic=f"episode_{turn - (5 - i)}",
                source="episode",
                salience=0.5,
                created_turn=max(0, turn - (5 - i)),
            ))

    def inject_inner_thought(self, thought: str, turn: int) -> None:
        """Inject a consciousness inner-voice thought into working memory."""
        self.add(WorkingMemoryChunk(
            content=thought[:200],
            topic=f"inner_thought_{turn}",
            source="inner",
            salience=0.6,
            created_turn=turn,
            confidence=0.5,
        ))

    @property
    def all_chunks(self) -> List[WorkingMemoryChunk]:
        return list(self._chunks)


# ---------------------------------------------------------------------------
# Goal Stack
# ---------------------------------------------------------------------------

_GOAL_TEMPLATES = {
    "code":          "Implement or fix code: {desc}",
    "research":      "Research information about: {desc}",
    "analysis":      "Analyze: {desc}",
    "creative":      "Create content: {desc}",
    "simple":        "Answer: {desc}",
    "conversational":"Discuss: {desc}",
    "self_design":   "Improve my own design: {desc}",
    "devai":         "Build an AI/dev tool: {desc}",
}


class GoalStack:
    def __init__(self, max_depth: int = 5) -> None:
        self._stack: Deque[Goal] = deque(maxlen=max_depth)

    def push(self, perceived: PerceivedInput, turn: int) -> None:
        desc = perceived.content[:80]
        tmpl = _GOAL_TEMPLATES.get(perceived.intent, "Handle: {desc}")
        gdesc = tmpl.format(desc=desc)
        # Continue existing goal of same intent
        if self._stack and self._stack[-1].intent == perceived.intent \
                       and self._stack[-1].status == "active":
            self._stack[-1].turn_updated = turn
            return
        self._stack.append(Goal(
            description=gdesc, intent=perceived.intent,
            status="active", turn_created=turn, turn_updated=turn,
        ))

    def get_active(self) -> List[Goal]:
        return [g for g in self._stack if g.status == "active"]

    def format_for_context(self) -> str:
        active = self.get_active()
        if not active:
            return ""
        lines = ["Current goals:"]
        for i, g in enumerate(reversed(active), 1):
            lines.append(f"  {i}. {g.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema Matcher
# ---------------------------------------------------------------------------

class SchemaMatcher:
    def match(self, perceived: PerceivedInput) -> Optional[Schema]:
        best, best_score = None, -1
        content_lower = perceived.content.lower()
        for schema in _SCHEMAS:
            score = 0
            if perceived.intent in schema.intents:
                score += 2
            for kw in schema.keywords:
                if kw in content_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best = schema
        return best if best_score > 0 else None


# ---------------------------------------------------------------------------
# Global Workspace (broadcast)
# ---------------------------------------------------------------------------

class GlobalWorkspace:
    def broadcast(
        self,
        wm:       List[WorkingMemoryChunk],
        goals:    List[Goal],
        schema:   Optional[Schema],
        neural_ctx: str = "",
    ) -> str:
        parts: List[str] = []

        if goals:
            parts.append(f"[Cognitive] Current goal: {goals[-1].description}")

        if schema:
            parts.append(f"[Cognitive] Strategy: {schema.strategy}")
            if schema.steps:
                parts.append("[Cognitive] Steps: " + " → ".join(schema.steps))

        high = [c for c in wm if c.salience > 0.4][:3]
        if high:
            items = "; ".join(f"{c.topic}: {c.content[:60]}" for c in high)
            parts.append(f"[Cognitive] Active context: {items}")

        if neural_ctx:
            parts.append(f"[Cognitive] {neural_ctx}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Layer 17 — main class
# ---------------------------------------------------------------------------

class CognitiveLayer:
    """
    Cognitive Architecture layer.
    Instantiated once in CortanaSystem; process() called per turn.
    """

    def __init__(self) -> None:
        self.working_memory  = WorkingMemory()
        self.goal_stack      = GoalStack()
        self.schema_matcher  = SchemaMatcher()
        self.global_workspace = GlobalWorkspace()

        # Neural memory — graceful fallback if torch not installed
        self._neural: Any = None
        try:
            from cortana.neural.neural_memory import NeuralMemory
            self._neural = NeuralMemory()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main processing step
    # ------------------------------------------------------------------

    def process(
        self,
        perceived:    PerceivedInput,
        memories:     List[str],
        concepts:     List[ConceptNode],
        relations:    List[RelationEdge],
        conversation: List[ConversationTurn],
        state:        CortanaState,
    ) -> CognitiveState:
        turn = state.interaction_count

        # 1. Populate working memory
        self.working_memory.load_from_episodes(memories, turn)
        self.working_memory.load_from_concepts(concepts, turn)

        # 2. Update salience
        self.working_memory.update_salience(
            perceived.keywords, turn, perceived.emotional_tone
        )

        # 3. Active chunks
        active = self.working_memory.get_active(
            perceived.keywords, turn, k=7, tone=perceived.emotional_tone
        )

        # 4. Goal stack
        self.goal_stack.push(perceived, turn)
        goals = self.goal_stack.get_active()

        # 5. Schema matching
        schema = self.schema_matcher.match(perceived)

        # 6. Neural temporal summary
        neural_ctx = ""
        if self._neural and memories:
            neural_ctx = self._neural.get_temporal_context_summary(memories)

        # 7. Global workspace broadcast
        ctx = self.global_workspace.broadcast(active, goals, schema, neural_ctx)

        # 8. Salience map + attention focus
        salience_map    = {c.topic: c.salience for c in active}
        attention_focus = [
            c.topic for c in sorted(active, key=lambda x: x.salience, reverse=True)[:3]
        ]

        return CognitiveState(
            working_memory=active,
            active_goals=goals,
            active_schema=schema,
            attention_focus=attention_focus,
            cognitive_context=ctx,
            salience_map=salience_map,
        )

    # ------------------------------------------------------------------
    # Neural-augmented recall (called by main before L4)
    # ------------------------------------------------------------------

    def neural_augmented_recall(
        self,
        query:       str,
        episodes:    List[str],
        concepts:    List[ConceptNode],
        relations:   List[RelationEdge],
        base_results: List[str],
    ) -> List[str]:
        if self._neural is None or not base_results:
            return base_results
        scored = self._neural.augment_recall(
            query, episodes, concepts, relations, base_results
        )
        return [text for text, _ in scored]

    # ------------------------------------------------------------------
    # Inner-voice injection (called by ConsciousnessEngine)
    # ------------------------------------------------------------------

    def inject_inner_thought(self, thought: str, turn: int) -> None:
        self.working_memory.inject_inner_thought(thought, turn)
