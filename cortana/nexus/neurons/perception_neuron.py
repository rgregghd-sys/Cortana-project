"""
Perception Neuron — Classifies intent and complexity from query.
Wraps L3 (PerceptionLayer). Sequential: fires first in chain.
Enriches signal payload with intent/complexity for downstream neurons.
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class PerceptionNeuron(Neuron):
    def __init__(self, perception_layer: Any = None) -> None:
        super().__init__("perception", NeuronType.PERCEPTION, threshold=0.0)
        self.perception_layer = perception_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.perception_layer:
            return ""
        try:
            state = signal.payload.get("state")
            result = self.perception_layer.perceive(signal.query, state)
            # Inject back into workspace metadata
            workspace = signal.payload.get("_workspace")
            if workspace is not None:
                workspace.metadata["intent"]     = getattr(result, "intent", "")
                workspace.metadata["complexity"] = getattr(result, "complexity", 0.5)
                workspace.metadata["perception"] = result
            intent = getattr(result, "intent", "unknown")
            complexity = getattr(result, "complexity", 0.5)
            return f"intent={intent} complexity={complexity:.2f}"
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 0.9 if content else 0.0
