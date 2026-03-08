"""
Cortana Neural Nexus — Full graph assembly.

Builds the complete neuron graph for Cortana and exposes a single
`process()` method that replaces the sequential layer pipeline.

This IS the Neural Nexus AGI. All cognitive systems (layers + AGI modules
+ consciousness) operate as neurons in a parallel-then-sequential DAG.

Call graph per turn:
  PARALLEL: memory, identity, agi, emotion, goal, world_model, security, consciousness
  SEQUENTIAL: perception → reasoning → ethics → reflection → synthesis

Post-turn:
  - Hebbian weight update (quality_score from reflection)
  - Background: world model update, autonomy goals, plan creation, consciousness tick
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from cortana.nexus.nexus  import NeuralNexus, Workspace
from cortana.nexus.neuron import NeuronSignal
from cortana.nexus.synapse import synapse_registry
from cortana.nexus.nexus_flush import nexus_flush

from cortana.nexus.neurons.memory_neuron       import MemoryNeuron
from cortana.nexus.neurons.identity_neuron     import IdentityNeuron
from cortana.nexus.neurons.perception_neuron   import PerceptionNeuron
from cortana.nexus.neurons.agi_neuron          import AGINeuron
from cortana.nexus.neurons.emotion_neuron      import EmotionNeuron
from cortana.nexus.neurons.goal_neuron         import GoalNeuron
from cortana.nexus.neurons.world_model_neuron  import WorldModelNeuron
from cortana.nexus.neurons.security_neuron     import SecurityNeuron
from cortana.nexus.neurons.consciousness_neuron import ConsciousnessNeuron
from cortana.nexus.neurons.reasoning_neuron    import ReasoningNeuron
from cortana.nexus.neurons.ethics_neuron       import EthicsNeuron
from cortana.nexus.neurons.reflection_neuron   import ReflectionNeuron
from cortana.nexus.neurons.synthesis_neuron    import SynthesisNeuron


class CortanaNexus:
    """
    The unified Neural Nexus AGI for Cortana.
    Instantiated once per CortanaSystem; passed all system references.
    """

    def __init__(self) -> None:
        self._nexus   = NeuralNexus()
        self._built   = False
        self._lock    = threading.Lock()

    # ------------------------------------------------------------------
    # Assembly — call after all system layers are ready
    # ------------------------------------------------------------------

    def build(self,
              memory_layer:       Any = None,
              identity_layer:     Any = None,
              perception_layer:   Any = None,
              reasoning_layer:    Any = None,
              reflection_layer:   Any = None,
              security_layer:     Any = None,
              agi_layer:          Any = None,
              consciousness_engine: Any = None) -> None:
        """Wire all neurons into the nexus graph."""

        # Parallel neurons
        self._nexus.add_neuron(MemoryNeuron(memory_layer))
        self._nexus.add_neuron(IdentityNeuron(identity_layer))
        self._nexus.add_neuron(AGINeuron(agi_layer))
        self._nexus.add_neuron(EmotionNeuron(consciousness_engine))
        self._nexus.add_neuron(GoalNeuron(agi_layer))
        self._nexus.add_neuron(WorldModelNeuron(agi_layer))
        self._nexus.add_neuron(SecurityNeuron(security_layer))
        self._nexus.add_neuron(ConsciousnessNeuron(consciousness_engine))

        # Sequential chain
        self._nexus.add_neuron(PerceptionNeuron(perception_layer))
        self._nexus.add_neuron(ReasoningNeuron(reasoning_layer))
        self._nexus.add_neuron(EthicsNeuron(agi_layer))
        self._nexus.add_neuron(ReflectionNeuron(reflection_layer, memory_layer))
        self._nexus.add_neuron(SynthesisNeuron())

        # Start nexus flush daemon
        nexus_flush.start()
        if reasoning_layer:
            nexus_flush.attach_reasoning(reasoning_layer)

        self._built = True

    def attach_reasoning(self, reasoning_layer: Any) -> None:
        """Hot-attach reasoning layer after initial build (for provider router)."""
        if not self._built:
            return
        n = self._nexus._neurons.get("reasoning")
        if n:
            n.reasoning_layer = reasoning_layer
        nexus_flush.attach_reasoning(reasoning_layer)

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(self,
                query:      str,
                history:    list     = None,
                state:      Any      = None,
                session_id: str      = "",
                on_chunk:   Optional[Callable] = None) -> str:
        """
        Process one user turn through the Neural Nexus.
        Returns final response string.
        """
        if not self._built:
            return ""

        history = history or []

        # Estimate signal strength from query complexity
        strength = self._estimate_strength(query)

        signal = NeuronSignal(
            query=query,
            strength=strength,
            payload={
                "history":    history,
                "state":      state,
                "session_id": session_id,
                "on_chunk":   on_chunk,
                "memory_hit_count": 0,
            },
            source_id="user",
        )

        workspace = self._nexus.propagate(signal)

        # Hebbian update in background
        quality = workspace.metadata.get("quality_score", 0.7)
        threading.Thread(
            target=self._post_turn,
            args=(workspace, quality, query, state),
            daemon=True,
        ).start()

        return workspace.final_response

    def _estimate_strength(self, query: str) -> float:
        """Map query length + question marks to signal strength (0.3–1.0)."""
        base = 0.5
        words = len(query.split())
        base += min(0.3, words / 100)
        if "?" in query or "how" in query.lower() or "why" in query.lower():
            base += 0.1
        return min(1.0, base)

    def _post_turn(self, workspace: Workspace, quality: float,
                   query: str, state: Any) -> None:
        """Background work after each turn: Hebbian update + world model."""
        try:
            self._nexus.hebbian_update(workspace, quality)
        except Exception:
            pass

        # AGI post-response background tasks
        agi = workspace.metadata.get("agi_layer")
        reasoning_out = workspace.get_first_reasoning(workspace)
        if agi and reasoning_out:
            try:
                agi.post_response(
                    response=reasoning_out,
                    query=query,
                )
            except Exception:
                pass

        # Consciousness interaction tick
        try:
            emotion = workspace.metadata.get("emotion", "neutral")
            # Access consciousness from emotion neuron if available
            em_neuron = self._nexus._neurons.get("consciousness")
            if em_neuron and hasattr(em_neuron, "consciousness") and em_neuron.consciousness:
                em_neuron.consciousness.on_interaction(emotion)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Status / admin
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        status = self._nexus.get_status()
        status["flush"] = nexus_flush.last_flush_info()
        return status

    def recent_flush_logs(self, limit: int = 5) -> list:
        return nexus_flush.recent_logs(limit)


# Extend Workspace with helper used in _post_turn
def _workspace_get_first_reasoning(self, ws) -> str:
    from cortana.nexus.neuron import NeuronType as NT
    out = ws.get_first(NT.REASONING)
    return out.content if out else ""

Workspace.get_first_reasoning = _workspace_get_first_reasoning
