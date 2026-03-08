"""
Emotion Neuron — Reads current mood from ConsciousnessEngine and injects
a tone directive into the system prompt context.

5-tier tone mapping (same as the mood bar replacement):
  >= 0.85 → enthusiastic
  >= 0.65 → calm, dry wit
  >= 0.45 → neutral / direct
  >= 0.25 → subdued / weary
  <  0.25 → sparse / honest brevity
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType

_TONE_MAP = [
    (0.85, "enthusiastic — Cortana is energised and eager; responses are vivid and forward-leaning"),
    (0.65, "calm and dry — Cortana is measured, precise, occasionally wry"),
    (0.45, "neutral / direct — Cortana keeps responses concise and factual, minimal colour"),
    (0.25, "subdued / weary — Cortana is terse; fewer words, honest about limitations"),
    (0.0,  "sparse / minimal — Cortana responds with honest brevity; no embellishments"),
]


def _tone_for(score: float) -> str:
    for threshold, label in _TONE_MAP:
        if score >= threshold:
            return label
    return _TONE_MAP[-1][1]


class EmotionNeuron(Neuron):
    def __init__(self, consciousness_engine: Any = None) -> None:
        super().__init__("emotion", NeuronType.EMOTION, threshold=0.0)
        self.consciousness = consciousness_engine

    def compute(self, signal: NeuronSignal) -> str:
        score = 0.65   # default neutral-calm
        if self.consciousness:
            try:
                sm = self.consciousness.self_model
                score = sm.get("mood_score", 0.65)
            except Exception:
                pass

        tone = _tone_for(score)
        return f"--- Current Mood (score={score:.2f}) ---\nTone: {tone}"

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 0.8 if content else 0.0
