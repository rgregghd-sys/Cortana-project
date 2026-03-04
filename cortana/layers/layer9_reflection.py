"""
Layer 9 — Reflection
Quality gate + logic matrix builder.
Scores response quality, optionally rewrites, stores to memory,
and extracts concept nodes + relationship edges for the logic matrix.
"""
from __future__ import annotations
import json
import re
from typing import List, Optional

from cortana import config
from cortana.models.schemas import (
    ConceptNode,
    CortanaState,
    PerceivedInput,
    ReflectionResult,
    RelationEdge,
    Task,
)

_REFLECTION_SYSTEM = """You are Cortana's reflection subsystem — a quality gate and knowledge extractor.

Given a user request and Cortana's response, output ONLY valid JSON (no markdown fences):
{
  "quality_score": 0.85,
  "improved_response": null,
  "memory_entry": "User asked about X. Cortana explained Y.",
  "emotion": "smile",
  "concepts": [
    {"topic": "machine learning", "summary": "ML is a subset of AI focused on learning from data.", "confidence": 0.9}
  ],
  "relations": [
    {"source": "machine learning", "target": "neural networks", "relation": "includes", "confidence": 0.85}
  ]
}

Fields:
- quality_score (0.0–1.0): accuracy, completeness, clarity
- improved_response: full rewritten response if quality < 0.7, otherwise null
- memory_entry: 1-2 sentence interaction summary for vector DB
- emotion: the emotional tone of Cortana's response — one of: idle | smile | sad | think | surprised | frown | laugh
  (smile=positive/helpful, sad=empathetic/bad news, think=analytical/complex, surprised=unexpected info,
   frown=frustrated/correcting, laugh=humorous, idle=neutral/informational)
- concepts: 0-4 distinct knowledge units extracted from this interaction \
  (only include meaningful domain concepts, not trivial facts)
- relations: 0-4 directed relationships between concept topics
  relation types: includes | depends_on | contradicts | leads_to | part_of | related_to | enables
"""


class ReflectionLayer:
    def __init__(self) -> None:
        self._reasoning = None
        self._memory = None

    def _get_reasoning(self):
        if self._reasoning is None:
            from cortana.layers.layer4_reasoning import ReasoningLayer
            self._reasoning = ReasoningLayer()
        return self._reasoning

    def _get_memory(self):
        if self._memory is None:
            from cortana.layers.layer2_memory import CortanaMemory
            self._memory = CortanaMemory()
        return self._memory

    def reflect(
        self,
        response: str,
        perceived: PerceivedInput,
        state: CortanaState,
        tasks: Optional[List[Task]] = None,
        user_input: str = "",
    ) -> ReflectionResult:
        prompt = self._build_prompt(response, perceived, tasks, user_input)

        try:
            raw = self._get_reasoning().think_simple(
                prompt=prompt,
                system=_REFLECTION_SYSTEM,
                max_tokens=1024,
            )
            cleaned = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(cleaned)

            quality = float(data.get("quality_score", 0.8))
            improved = data.get("improved_response")
            memory_entry = str(data.get("memory_entry", f"User: {user_input[:100]}"))
            final_response = improved if improved and improved.strip() else response

            _valid_emotions = {"idle", "smile", "sad", "think", "surprised", "frown", "laugh"}
            emotion = data.get("emotion", "idle")
            if emotion not in _valid_emotions:
                emotion = "idle"

            # Parse concept nodes
            concepts: List[ConceptNode] = []
            for c in data.get("concepts", []):
                try:
                    concepts.append(ConceptNode(**c))
                except Exception:
                    pass

            # Parse relationship edges
            relations: List[RelationEdge] = []
            for r in data.get("relations", []):
                try:
                    relations.append(RelationEdge(**r))
                except Exception:
                    pass

        except Exception:
            quality = 0.75
            final_response = response
            memory_entry = f"User: {user_input[:100]}. Cortana responded."
            emotion = "idle"
            concepts = []
            relations = []

        # Tier 1 + 2: Store interaction
        self._get_memory().store(
            interaction=f"User: {user_input}\nCortana: {final_response[:500]}",
            metadata={"source": "interaction", "intent": perceived.intent, "quality": str(quality)},
        )

        # Tier 3: Store concepts + relations into logic matrix
        if concepts or relations:
            self._get_memory().store_concepts(concepts, relations)

        return ReflectionResult(
            final_response=final_response,
            quality_score=quality,
            memory_entry=memory_entry,
            emotion=emotion,
            concepts=concepts,
            relations=relations,
        )

    def _build_prompt(
        self,
        response: str,
        perceived: PerceivedInput,
        tasks: Optional[List[Task]],
        user_input: str,
    ) -> str:
        parts = [
            f"User request: {user_input}",
            f"Intent: {perceived.intent} | Complexity: {perceived.complexity:.2f}",
            f"\nCortana's response:\n{response}",
        ]
        if tasks:
            agent_summary = ", ".join(
                f"{t.agent_type}({'done' if t.status == 'done' else 'failed'})"
                for t in tasks
            )
            parts.append(f"\nSub-agents: {agent_summary}")
        parts.append("\nAnalyze and extract:")
        return "\n".join(parts)
