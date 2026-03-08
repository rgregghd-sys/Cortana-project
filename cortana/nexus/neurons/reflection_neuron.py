"""
Reflection Neuron — L9 quality gate. Sequential; fires after EthicsNeuron.
Scores response quality, stores to memory, emits quality_score for Hebbian update.
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class ReflectionNeuron(Neuron):
    def __init__(self, reflection_layer: Any = None,
                 memory_layer: Any = None) -> None:
        super().__init__("reflection", NeuronType.REFLECTION, threshold=0.0)
        self.reflection_layer = reflection_layer
        self.memory_layer     = memory_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.reflection_layer:
            return ""

        workspace = signal.payload.get("_workspace")
        if not workspace:
            return ""

        from cortana.nexus.neuron import NeuronType as NT
        reasoning_out = workspace.get_first(NT.REASONING)
        if not reasoning_out or not reasoning_out.content:
            return ""

        try:
            history = signal.payload.get("history", [])
            state   = signal.payload.get("state")

            result = self.reflection_layer.reflect(
                query=signal.query,
                response=reasoning_out.content,
                history=history,
                state=state,
            )

            quality_score = getattr(result, "quality_score", 0.7)
            workspace.metadata["quality_score"] = quality_score
            workspace.metadata["reflection_result"] = result

            # Emotion update for consciousness
            emotion = getattr(result, "emotion", "neutral")
            workspace.metadata["emotion"] = emotion

            # Store to memory
            if self.memory_layer:
                try:
                    mem_entry = getattr(result, "memory_entry", "")
                    if mem_entry:
                        self.memory_layer.store(
                            content=mem_entry,
                            metadata={"query": signal.query[:200]},
                        )
                except Exception:
                    pass

            return ""   # reflection doesn't modify visible output
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 1.0
