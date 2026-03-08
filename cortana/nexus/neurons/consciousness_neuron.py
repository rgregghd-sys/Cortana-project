"""
Consciousness Neuron — Injects recent inner thoughts and self-model context
from the ConsciousnessEngine into the workspace. Fires in parallel.
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class ConsciousnessNeuron(Neuron):
    def __init__(self, consciousness_engine: Any = None) -> None:
        super().__init__("consciousness", NeuronType.CONSCIOUSNESS, threshold=0.0)
        self.consciousness = consciousness_engine

    def compute(self, signal: NeuronSignal) -> str:
        if not self.consciousness:
            return ""
        try:
            ctx = self.consciousness.get_cognitive_context()
            return ctx or ""
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 0.7 if content else 0.0
