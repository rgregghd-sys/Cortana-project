"""
Goal Neuron — Injects active autonomy goals and long-horizon plan context.
Threshold: 0.3 (only fires when signal is meaningful, not trivial chit-chat).
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class GoalNeuron(Neuron):
    def __init__(self, agi_layer: Any = None) -> None:
        super().__init__("goal", NeuronType.GOAL, threshold=0.3)
        self.agi_layer = agi_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.agi_layer:
            return ""
        parts = []
        try:
            ctx = self.agi_layer.autonomy.goals_context_prompt(limit=3)
            if ctx:
                parts.append(ctx)
        except Exception:
            pass
        try:
            plans = self.agi_layer.long_horizon.get_planning_context()
            if plans:
                parts.append(plans)
        except Exception:
            pass
        return "\n\n".join(parts)

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 0.7 if content else 0.0
