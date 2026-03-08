"""
Synthesis Neuron — Final output assembler. Sequential; fires last.
Reads the Global Workspace and returns the final response string.

The SynthesisNeuron doesn't call an LLM — it reads what the ReasoningNeuron
produced (which may have been modified by EthicsNeuron) and returns it as
the definitive output. It is the "consciousness" of the nexus.
"""
from __future__ import annotations

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class SynthesisNeuron(Neuron):
    def __init__(self) -> None:
        super().__init__("synthesis", NeuronType.SYNTHESIS, threshold=0.0)

    def compute(self, signal: NeuronSignal) -> str:
        workspace = signal.payload.get("_workspace")
        if not workspace:
            return ""

        from cortana.nexus.neuron import NeuronType as NT

        # Primary: use reasoning output (may have been patched by ethics)
        reasoning_out = workspace.get_first(NT.REASONING)
        if reasoning_out and reasoning_out.content:
            return reasoning_out.content

        # Fallback: concatenate highest-confidence non-infrastructure outputs
        excluded = {NT.IDENTITY, NT.ETHICS, NT.REFLECTION, NT.SECURITY,
                    NT.PERCEPTION, NT.EMOTION, NT.SYNTHESIS}
        candidates = [
            o for o in workspace.all_content_by_confidence()
            if o.neuron_type not in excluded and o.content
        ]
        if candidates:
            return candidates[0].content

        return "I wasn't able to generate a response. Please try again."

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 1.0 if content else 0.0
