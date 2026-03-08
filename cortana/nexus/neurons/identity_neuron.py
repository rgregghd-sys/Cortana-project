"""
Identity Neuron — Provides Cortana's core personality and constitutional prompt.
Wraps L1 (IdentityLayer). Always fires with identity_prompt + mood tone.
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class IdentityNeuron(Neuron):
    def __init__(self, identity_layer: Any = None) -> None:
        super().__init__("identity", NeuronType.IDENTITY, threshold=0.0)
        self.identity_layer = identity_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.identity_layer:
            return ""
        try:
            return self.identity_layer.get_identity_prompt()
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 1.0 if content else 0.0
