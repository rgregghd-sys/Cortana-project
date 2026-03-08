"""
World Model Neuron — Queries the persistent entity/belief/causal world model
and injects relevant context. Fires in parallel with other context neurons.
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class WorldModelNeuron(Neuron):
    def __init__(self, agi_layer: Any = None) -> None:
        super().__init__("world_model", NeuronType.WORLD_MODEL, threshold=0.0)
        self.agi_layer = agi_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.agi_layer:
            return ""
        try:
            ctx = self.agi_layer.world_model.query_context(signal.query)
            return ctx or ""
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 0.75 if content else 0.0
