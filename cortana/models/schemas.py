"""
Pydantic schemas shared across all layers.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import uuid

# ---------------------------------------------------------------------------
# Layer 0 — Supervisor
# ---------------------------------------------------------------------------

class SupervisorFeedback(BaseModel):
    layer_id: int
    reason: str      # Why it failed
    correction: str  # What to do instead
    lesson: str      # Terse instruction for retry


# ---------------------------------------------------------------------------
# Layer 10 — Security (Red vs Blue)
# ---------------------------------------------------------------------------

class Vulnerability(BaseModel):
    type: Literal["prompt_injection", "logic_flaw", "memory_poison", "tool_misuse"]
    severity: Literal["low", "medium", "high"]
    description: str
    target: Literal["code", "prompt"]  # tells L11 what kind of patch to write


class SecurityResult(BaseModel):
    red_wins: bool
    vulnerabilities: List[Vulnerability] = Field(default_factory=list)
    defense_score: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Layer 11 — Patch Writer
# ---------------------------------------------------------------------------

class PatchEntry(BaseModel):
    vulnerability_type: str
    severity: str
    target: Literal["code", "prompt"]
    patch_file: str
    description: str


class PatchResult(BaseModel):
    patches: List[PatchEntry] = Field(default_factory=list)
    patch_files: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 3 — Perception
# ---------------------------------------------------------------------------

class UserInput(BaseModel):
    raw: str
    input_type: Literal["text", "file", "image"] = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerceivedInput(BaseModel):
    content: str
    intent: Literal["simple", "conversational", "research", "code", "analysis", "creative", "self_design", "devai"]
    complexity: float = Field(ge=0.0, le=1.0)  # 0.0 = trivial, 1.0 = very complex
    keywords: List[str] = Field(default_factory=list)
    emotional_tone: Literal["neutral", "frustrated", "curious", "excited", "confused", "playful"] = "neutral"


# ---------------------------------------------------------------------------
# Layer 1 — Identity / State
# ---------------------------------------------------------------------------

class CortanaState(BaseModel):
    interaction_count: int = 0


# ---------------------------------------------------------------------------
# Layer 5 — Planning
# ---------------------------------------------------------------------------

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    agent_type: Literal["researcher", "coder", "analyst", "writer", "direct"]
    priority: int = Field(default=1, ge=1, le=5)
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: Optional[str] = None
    context: Optional[str] = None  # extra context injected by planner


class TaskPlan(BaseModel):
    tasks: List[Task]
    parallel: bool = True          # run tasks concurrently where possible
    reasoning: Optional[str] = None  # planner's rationale


# ---------------------------------------------------------------------------
# Logic Matrix — Hierarchical Memory Structures
# ---------------------------------------------------------------------------

class ConceptNode(BaseModel):
    """A distilled knowledge unit extracted from interactions."""
    topic: str                     # e.g. "machine learning", "user's project"
    summary: str                   # 1-3 sentence distillation
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence_count: int = 1        # number of interactions that support this


class RelationEdge(BaseModel):
    """A directed relationship between two concept topics."""
    source: str                    # topic A
    target: str                    # topic B
    relation: str                  # e.g. "depends_on", "contradicts", "part_of", "leads_to"
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Layer 9 — Reflection
# ---------------------------------------------------------------------------

class ReflectionResult(BaseModel):
    final_response: str
    quality_score: float = Field(ge=0.0, le=1.0)
    memory_entry: str = ""         # text stored into ChromaDB
    concepts: List[ConceptNode] = Field(default_factory=list)
    relations: List[RelationEdge] = Field(default_factory=list)
    emotion: Literal["idle", "smile", "sad", "think", "surprised", "frown", "laugh"] = "idle"


# ---------------------------------------------------------------------------
# Conversation Turn
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
