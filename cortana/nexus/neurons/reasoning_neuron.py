"""
Reasoning Neuron — Core LLM inference (L4). Sequential; fires after all
parallel neurons have populated the workspace.

Assembles system prompt from workspace outputs, calls L4 reasoning with
the full PerceivedInput + memories interface (compatible with existing layer).
Optionally runs System 2 deliberate reasoning for complex queries.
"""
from __future__ import annotations

from typing import Any, Optional

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class ReasoningNeuron(Neuron):
    def __init__(self, reasoning_layer: Any = None) -> None:
        super().__init__("reasoning", NeuronType.REASONING, threshold=0.0)
        self.reasoning_layer = reasoning_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.reasoning_layer:
            return ""

        workspace  = signal.payload.get("_workspace")
        history    = signal.payload.get("history", [])
        state      = signal.payload.get("state")
        on_chunk   = signal.payload.get("on_chunk")
        memories   = signal.payload.get("memories", [])
        perceived  = signal.payload.get("perceived")

        # Build system prompt from workspace
        system_prompt = self._build_system_prompt(workspace)

        try:
            agi_layer  = workspace.metadata.get("agi_layer") if workspace else None
            intent     = workspace.metadata.get("intent", "") if workspace else ""
            complexity = workspace.metadata.get("complexity", 0.5) if workspace else 0.5

            # System 2 check
            if agi_layer and agi_layer.should_use_system2(signal.query, intent, complexity):
                try:
                    s2 = agi_layer.run_system2(signal.query, context=system_prompt[:600])
                    if s2 and getattr(s2, "triggered", False) and getattr(s2, "synthesis", ""):
                        if workspace:
                            workspace.metadata["system2_result"] = s2
                        return s2.synthesis
                except Exception:
                    pass

            # Use full think() interface if PerceivedInput available
            if perceived is not None:
                from cortana.models.schemas import PerceivedInput
                # Build enhanced perceived with web/research context already in signal
                enhanced_content = signal.query
                enhanced = PerceivedInput(
                    content=enhanced_content,
                    intent=getattr(perceived, "intent", intent) or intent,
                    complexity=getattr(perceived, "complexity", complexity),
                    keywords=getattr(perceived, "keywords", []),
                )
                response = self.reasoning_layer.think(
                    perceived=enhanced,
                    memories=memories if isinstance(memories, list) else [],
                    identity_prompt=system_prompt,
                    conversation_history=history,
                    state=state,
                    on_chunk=on_chunk,
                )
            else:
                # Fallback: think_simple with assembled system
                response = self.reasoning_layer.think_simple(
                    prompt=signal.query,
                    system=system_prompt,
                )

            return response or ""
        except Exception as e:
            return f"I encountered an issue processing that request. ({e})"

    def _build_system_prompt(self, workspace) -> str:
        if not workspace:
            return ""
        from cortana.nexus.neuron import NeuronType as NT
        sections = []

        for ntype in (NT.IDENTITY, NT.AGI, NT.EMOTION, NT.MEMORY,
                      NT.WORLD_MODEL, NT.CONSCIOUSNESS, NT.GOAL, NT.SECURITY):
            for out in workspace.get(ntype):
                if out.content:
                    sections.append(out.content)

        return "\n\n".join(s for s in sections if s)

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        if not content:
            return 0.0
        return min(1.0, 0.5 + len(content) / 2000)
