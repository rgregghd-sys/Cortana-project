"""
Neural Nexus — Base Neuron primitives.

Each processing unit in Cortana is a Neuron.
Neurons receive NeuronSignals, decide whether to fire (activation threshold),
produce NeuronOutputs, and propagate to downstream neurons via Synapses.

Activation model:
  - Each neuron has a base threshold (0.0–1.0)
  - Incoming signal strength + synaptic weight must exceed threshold to fire
  - Firing triggers async compute → NeuronOutput appended to workspace
  - Inhibitory neurons can suppress others (negative weight)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Signal / Output types
# ---------------------------------------------------------------------------

class NeuronType(str, Enum):
    INPUT        = "input"
    MEMORY       = "memory"
    IDENTITY     = "identity"
    PERCEPTION   = "perception"
    REASONING    = "reasoning"
    AGI          = "agi"
    ETHICS       = "ethics"
    EMOTION      = "emotion"
    GOAL         = "goal"
    WORLD_MODEL  = "world_model"
    SYNTHESIS    = "synthesis"
    CONSCIOUSNESS = "consciousness"
    SECURITY     = "security"
    REFLECTION   = "reflection"
    OUTPUT       = "output"


@dataclass
class NeuronSignal:
    """
    Input signal propagated through the nexus.
    strength: 0.0–1.0 — how strongly this signal fires (intent/complexity driven)
    payload: arbitrary context dict
    source_id: neuron that generated this signal (or 'user' for root)
    ts: unix timestamp
    """
    query:      str
    strength:   float           # 0.0–1.0
    payload:    Dict[str, Any]  = field(default_factory=dict)
    source_id:  str             = "user"
    ts:         float           = field(default_factory=time.time)

    def with_payload(self, **kwargs) -> "NeuronSignal":
        """Return a copy with additional payload keys."""
        return NeuronSignal(
            query=self.query,
            strength=self.strength,
            payload={**self.payload, **kwargs},
            source_id=self.source_id,
            ts=self.ts,
        )


@dataclass
class NeuronOutput:
    """
    Output produced when a neuron fires.
    neuron_id: which neuron produced this
    neuron_type: for downstream routing
    content: the computed result (prompt fragment, analysis, etc.)
    confidence: 0.0–1.0 quality estimate
    metadata: extra info for synthesis
    fired: True if neuron actually computed, False if suppressed/skipped
    """
    neuron_id:   str
    neuron_type: NeuronType
    content:     str
    confidence:  float          = 1.0
    metadata:    Dict[str, Any] = field(default_factory=dict)
    fired:       bool           = True
    latency_ms:  float          = 0.0


# ---------------------------------------------------------------------------
# Base Neuron ABC
# ---------------------------------------------------------------------------

class Neuron(ABC):
    """
    Abstract base for all Neural Nexus processing units.

    Subclasses implement:
      - activate(signal) → bool   — decide if this neuron should fire
      - compute(signal)  → str    — produce content when fired
    """

    neuron_id:   str
    neuron_type: NeuronType
    threshold:   float          # minimum signal strength to fire

    def __init__(self, neuron_id: str, neuron_type: NeuronType,
                 threshold: float = 0.0) -> None:
        self.neuron_id   = neuron_id
        self.neuron_type = neuron_type
        self.threshold   = threshold
        self._fire_count = 0
        self._total_confidence = 0.0

    # ------------------------------------------------------------------
    # Public API — called by NeuralNexus
    # ------------------------------------------------------------------

    def process(self, signal: NeuronSignal) -> NeuronOutput:
        """
        Main entry point. Checks threshold, calls compute() if active,
        returns NeuronOutput.
        """
        if not self.activate(signal):
            return NeuronOutput(
                neuron_id=self.neuron_id,
                neuron_type=self.neuron_type,
                content="",
                confidence=0.0,
                fired=False,
            )

        t0 = time.perf_counter()
        try:
            content = self.compute(signal) or ""
            confidence = self.score_confidence(content, signal)
        except Exception as e:
            content = ""
            confidence = 0.0

        latency = (time.perf_counter() - t0) * 1000
        self._fire_count += 1
        self._total_confidence += confidence

        return NeuronOutput(
            neuron_id=self.neuron_id,
            neuron_type=self.neuron_type,
            content=content,
            confidence=confidence,
            fired=True,
            latency_ms=round(latency, 2),
        )

    # ------------------------------------------------------------------
    # Override in subclasses
    # ------------------------------------------------------------------

    def activate(self, signal: NeuronSignal) -> bool:
        """Return True if this neuron should fire for the given signal."""
        return signal.strength >= self.threshold

    @abstractmethod
    def compute(self, signal: NeuronSignal) -> str:
        """Produce content. Called only when activate() returns True."""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        """Estimate output confidence (override for richer scoring)."""
        return 1.0 if content else 0.0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def avg_confidence(self) -> float:
        if self._fire_count == 0:
            return 0.0
        return self._total_confidence / self._fire_count

    def stats(self) -> Dict[str, Any]:
        return {
            "neuron_id":      self.neuron_id,
            "type":           self.neuron_type.value,
            "threshold":      self.threshold,
            "fire_count":     self._fire_count,
            "avg_confidence": round(self.avg_confidence, 3),
        }
