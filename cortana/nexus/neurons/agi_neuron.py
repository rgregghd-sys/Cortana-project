"""
AGI Neuron — Orchestrates all 15 AGI modules from Layer 18.

This neuron IS the AGI architecture. It fires in parallel with other
context neurons, calling AGILayer.augment_prompt() to inject all
AGI context (cognitive modes, cross-domain, world model, theory of mind,
zero-shot scaffold, abstract reasoning, common sense, NLP mastery, etc.)

The AGI Neuron and the Neural Nexus are one unified system.
"""
from __future__ import annotations

from typing import Any, Optional

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class AGINeuron(Neuron):
    def __init__(self, agi_layer: Any = None) -> None:
        super().__init__("agi", NeuronType.AGI, threshold=0.0)
        self.agi_layer = agi_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.agi_layer:
            return ""
        try:
            workspace  = signal.payload.get("_workspace")
            session_id = signal.payload.get("session_id", "")
            intent     = ""
            complexity = 0.5
            if workspace:
                intent     = workspace.metadata.get("intent", "")
                complexity = workspace.metadata.get("complexity", 0.5)

            # Update Theory of Mind before augmenting
            history = signal.payload.get("history", [])
            if session_id:
                try:
                    self.agi_layer.update_user_model(session_id, signal.query, history)
                except Exception:
                    pass

            # Full AGI prompt augmentation (no LLM, <15ms)
            augmented = self.agi_layer.augment_prompt(
                query=signal.query,
                intent=intent,
                identity_prompt="",        # identity handled by IdentityNeuron
                session_id=session_id,
                memory_hit_count=signal.payload.get("memory_hit_count", 0),
                complexity=complexity,
            )

            # Store agi_layer reference for EthicsNeuron and ReasoningNeuron
            if workspace is not None:
                workspace.metadata["agi_layer"] = self.agi_layer

            return augmented.strip() if augmented else ""
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        if not content:
            return 0.0
        return min(1.0, 0.5 + len(content) / 5000)
