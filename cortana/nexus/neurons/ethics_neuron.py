"""
Ethics Neuron — Constitutional safety gate. Sequential; fires after
ReasoningNeuron. If violations found, replaces response with a principled
reasoned refusal (not a blunt block).
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class EthicsNeuron(Neuron):
    def __init__(self, agi_layer: Any = None) -> None:
        super().__init__("ethics", NeuronType.ETHICS, threshold=0.0)
        self.agi_layer = agi_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.agi_layer:
            return ""

        workspace = signal.payload.get("_workspace")
        if not workspace:
            return ""

        # Get the reasoning output to check
        from cortana.nexus.neuron import NeuronType as NT
        reasoning_out = workspace.get_first(NT.REASONING)
        if not reasoning_out or not reasoning_out.content:
            return ""

        try:
            result = self.agi_layer.check_response(
                response=reasoning_out.content,
                context=signal.query,
                query=signal.query,
            )
            workspace.metadata["ethics_result"] = result

            if not result.approved:
                modified = getattr(result, "modified_response", "")
                if modified:
                    # Replace reasoning output with ethical refusal
                    reasoning_out.content = modified
                    return f"[Ethics gate applied]"
            return ""
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 1.0
