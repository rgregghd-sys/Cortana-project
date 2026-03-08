"""
Memory Neuron — Retrieves relevant episodic, semantic, and conceptual memory.
Wraps L2 (MemoryLayer). Always fires; empty output if no relevant hits.
"""
from __future__ import annotations

from typing import Any, Optional

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class MemoryNeuron(Neuron):
    def __init__(self, memory: Any = None) -> None:
        super().__init__("memory", NeuronType.MEMORY, threshold=0.0)
        self.memory = memory

    def compute(self, signal: NeuronSignal) -> str:
        if not self.memory:
            return ""
        try:
            hits = self.memory.recall(signal.query, top_k=5)
            if not hits:
                return ""
            lines = []
            for h in hits[:5]:
                txt = h.get("content", h.get("text", "")) if isinstance(h, dict) else str(h)
                if txt:
                    lines.append(f"• {txt[:200]}")
            if not lines:
                return ""
            return "--- Memory Context ---\n" + "\n".join(lines)
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        if not content:
            return 0.0
        count = content.count("•")
        return min(1.0, 0.4 + count * 0.12)
