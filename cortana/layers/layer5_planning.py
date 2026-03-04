"""
Layer 5 — Planning
Decomposes complex requests into a TaskPlan via Claude.
Only invoked when PerceivedInput.complexity >= COMPLEXITY_THRESHOLD.
"""
from __future__ import annotations
import json
import re
from typing import Optional

from cortana import config
from cortana.models.schemas import CortanaState, PerceivedInput, Task, TaskPlan

_PLANNING_SYSTEM = """You are Cortana's strategic planning subsystem.
Given a user request, decompose it into a set of discrete sub-tasks.
Each task must be assigned to exactly one agent type:
  - "researcher" : web search, fact-finding, external information
  - "coder"      : write/debug/execute Python code
  - "analyst"    : compare, evaluate, break down data
  - "writer"     : produce prose, reports, documentation
  - "direct"     : simple response, no specialized agent needed

Output ONLY valid JSON matching this schema (no markdown fences, no explanation):
{
  "reasoning": "brief explanation of decomposition strategy",
  "parallel": true,
  "tasks": [
    {
      "description": "precise task description",
      "agent_type": "researcher|coder|analyst|writer|direct",
      "priority": 1,
      "context": "optional extra context for this task"
    }
  ]
}

Rules:
- 1 to 4 tasks maximum
- Prefer parallel=true unless tasks depend on each other's output
- Set parallel=false only when task N+1 clearly needs task N's result
- "direct" tasks indicate the main brain should answer without a sub-agent
- Priority 1 = highest, 5 = lowest
"""


class PlanningLayer:
    """
    Uses a lightweight Claude call to decompose complex PerceivedInput
    into an actionable TaskPlan.
    """

    def __init__(self) -> None:
        # Lazy import to avoid circular import at module load
        self._reasoning = None

    def _get_reasoning(self):
        if self._reasoning is None:
            from cortana.layers.layer4_reasoning import ReasoningLayer
            self._reasoning = ReasoningLayer()
        return self._reasoning

    def plan(
        self,
        perceived: PerceivedInput,
        state: CortanaState,
        conversation_context: Optional[str] = None,
    ) -> TaskPlan:
        """
        Decompose the perceived input into a TaskPlan.
        Falls back to a single 'direct' task if parsing fails.
        """
        prompt = self._build_prompt(perceived, conversation_context)
        raw = self._get_reasoning().think_simple(
            prompt=prompt,
            system=_PLANNING_SYSTEM,
            model=config.SUB_AGENT_MODEL,
            max_tokens=1024,
        )

        return self._parse_plan(raw, perceived)

    def _build_prompt(self, perceived: PerceivedInput, context: Optional[str]) -> str:
        parts = [
            f"User request: {perceived.content}",
            f"Detected intent: {perceived.intent}",
            f"Complexity score: {perceived.complexity:.2f}",
            f"Keywords: {', '.join(perceived.keywords) or 'none'}",
        ]
        if context:
            parts.append(f"Recent conversation context: {context}")
        parts.append("\nDecompose this into a TaskPlan:")
        return "\n".join(parts)

    def _parse_plan(self, raw: str, perceived: PerceivedInput) -> TaskPlan:
        """Parse Claude's JSON output into a TaskPlan. Falls back gracefully."""
        # Strip any accidental markdown fences
        cleaned = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(cleaned)
            tasks = []
            for t in data.get("tasks", []):
                tasks.append(Task(
                    description=t.get("description", perceived.content),
                    agent_type=t.get("agent_type", "direct"),
                    priority=int(t.get("priority", 1)),
                    context=t.get("context"),
                ))
            if not tasks:
                raise ValueError("Empty task list")
            return TaskPlan(
                tasks=tasks,
                parallel=bool(data.get("parallel", True)),
                reasoning=data.get("reasoning"),
            )
        except Exception:
            # Fallback: single direct task
            return TaskPlan(
                tasks=[Task(
                    description=perceived.content,
                    agent_type="direct",
                    priority=1,
                )],
                parallel=False,
                reasoning="Fallback: planning parse failed, routing to direct response.",
            )

    def needs_planning(self, perceived: PerceivedInput) -> bool:
        """True when the request is complex enough to warrant sub-agent dispatch."""
        return (
            perceived.complexity >= config.COMPLEXITY_THRESHOLD
            and perceived.intent not in ("simple", "conversational")
        )
