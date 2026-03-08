"""
Neural Nexus — Core graph engine.

Architecture:
  1. Signal enters via InputNeuron
  2. All non-sequential neurons fire in parallel (ThreadPoolExecutor)
  3. Synaptic weights modulate effective signal strength to each neuron
  4. ReasoningNeuron fires last (sequential) — uses all parallel outputs as context
  5. EthicsNeuron gates the output
  6. SynthesisNeuron assembles the final response from the Global Workspace
  7. Hebbian update applied using L9 quality_score

Global Workspace Theory:
  All fired NeuronOutputs are collected in a shared workspace.
  SynthesisNeuron reads the full workspace to produce coherent final output.

Propagation order (DAG):
  PARALLEL (concurrent):
    memory, identity, agi, emotion, goal, world_model, security, consciousness
  SEQUENTIAL (in order):
    perception  →  reasoning  →  ethics  →  reflection  →  synthesis
"""
from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from cortana.nexus.neuron  import Neuron, NeuronOutput, NeuronSignal, NeuronType
from cortana.nexus.synapse import synapse_registry


# Neurons that fire concurrently before the sequential chain
_PARALLEL_TYPES = frozenset([
    NeuronType.MEMORY,
    NeuronType.IDENTITY,
    NeuronType.AGI,
    NeuronType.EMOTION,
    NeuronType.GOAL,
    NeuronType.WORLD_MODEL,
    NeuronType.SECURITY,
    NeuronType.CONSCIOUSNESS,
])

# Sequential chain (fired in this order after parallel stage)
_SEQUENTIAL_ORDER = [
    NeuronType.PERCEPTION,
    NeuronType.REASONING,
    NeuronType.ETHICS,
    NeuronType.REFLECTION,
    NeuronType.SYNTHESIS,
]

_PARALLEL_WORKERS = 8


class Workspace:
    """Global Workspace — shared blackboard for all neuron outputs."""

    def __init__(self) -> None:
        self.outputs: List[NeuronOutput] = []
        self._by_type: Dict[NeuronType, List[NeuronOutput]] = {}
        self.metadata: Dict[str, Any] = {}

    def add(self, output: NeuronOutput) -> None:
        if output.fired and output.content:
            self.outputs.append(output)
            self._by_type.setdefault(output.neuron_type, []).append(output)

    def get(self, neuron_type: NeuronType) -> List[NeuronOutput]:
        return self._by_type.get(neuron_type, [])

    def get_first(self, neuron_type: NeuronType) -> Optional[NeuronOutput]:
        lst = self._by_type.get(neuron_type, [])
        return lst[0] if lst else None

    def aggregate_content(self, neuron_type: NeuronType) -> str:
        return "\n".join(o.content for o in self.get(neuron_type) if o.content)

    def all_content_by_confidence(self) -> List[NeuronOutput]:
        return sorted(
            [o for o in self.outputs if o.content],
            key=lambda o: o.confidence, reverse=True
        )

    @property
    def final_response(self) -> str:
        syn = self.get_first(NeuronType.SYNTHESIS)
        return syn.content if syn else ""


class NeuralNexus:
    """
    The Neural Nexus — Cortana's unified AGI processing graph.

    All cognitive functions (layers, AGI modules, consciousness) are neurons
    connected by weighted synapses. This IS the AGI architecture.
    """

    def __init__(self) -> None:
        self._neurons:   Dict[str, Neuron]   = {}
        self._callbacks: List[Callable]      = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_neuron(self, neuron: Neuron) -> None:
        self._neurons[neuron.neuron_id] = neuron
        # Ensure synapses exist for all cross-connections
        for other_id in list(self._neurons.keys()):
            if other_id != neuron.neuron_id:
                synapse_registry.register(other_id, neuron.neuron_id)
                synapse_registry.register(neuron.neuron_id, other_id)

    def connect(self, source_id: str, target_id: str,
                weight: float = 1.0) -> None:
        synapse_registry.register(source_id, target_id, weight)

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def propagate(self, signal: NeuronSignal) -> Workspace:
        """
        Run one full propagation cycle through the nexus.
        Returns populated Workspace for the SynthesisNeuron to read.
        """
        workspace = Workspace()

        # --- Stage 1: Parallel neurons ---
        parallel_neurons = [
            n for n in self._neurons.values()
            if n.neuron_type in _PARALLEL_TYPES
        ]
        if parallel_neurons:
            with concurrent.futures.ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as ex:
                futures = {
                    ex.submit(self._fire, n, signal): n
                    for n in parallel_neurons
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        output = fut.result(timeout=5.0)
                        workspace.add(output)
                    except Exception:
                        pass

        # --- Stage 2: Sequential chain ---
        enriched_signal = self._enrich_signal(signal, workspace)
        for ntype in _SEQUENTIAL_ORDER:
            for neuron in self._get_by_type(ntype):
                output = self._fire(neuron, enriched_signal, workspace)
                workspace.add(output)
                # Re-enrich after each sequential step
                enriched_signal = self._enrich_signal(enriched_signal, workspace)

        return workspace

    def _fire(self, neuron: Neuron, signal: NeuronSignal,
              workspace: Optional[Workspace] = None) -> NeuronOutput:
        """
        Fire a single neuron. Modulates signal strength by synaptic weight.
        Passes workspace reference for sequential neurons that need context.
        """
        # Compute effective strength via incoming synaptic weights
        incoming_weight = 1.0
        for src_id in list(self._neurons.keys()):
            if src_id != neuron.neuron_id:
                w = synapse_registry.get_weight(src_id, neuron.neuron_id)
                incoming_weight = max(incoming_weight, w)

        modulated = NeuronSignal(
            query=signal.query,
            strength=min(1.0, signal.strength * max(0.0, incoming_weight)),
            payload={**signal.payload, "_workspace": workspace},
            source_id=signal.source_id,
            ts=signal.ts,
        )
        return neuron.process(modulated)

    def _enrich_signal(self, signal: NeuronSignal,
                       workspace: Workspace) -> NeuronSignal:
        """Inject current workspace into signal payload."""
        return NeuronSignal(
            query=signal.query,
            strength=signal.strength,
            payload={**signal.payload, "_workspace": workspace},
            source_id=signal.source_id,
            ts=signal.ts,
        )

    def _get_by_type(self, ntype: NeuronType) -> List[Neuron]:
        return [n for n in self._neurons.values() if n.neuron_type == ntype]

    # ------------------------------------------------------------------
    # Hebbian learning — call after each turn with L9 quality_score
    # ------------------------------------------------------------------

    def hebbian_update(self, workspace: Workspace, quality_score: float) -> None:
        """
        Update all active synapse weights based on turn quality.
        Neurons that contributed to a high-quality response get strengthened.
        """
        fired = [o for o in workspace.outputs if o.fired]
        for i, src_out in enumerate(fired):
            for tgt_out in fired[i+1:]:
                synapse_registry.hebbian_update(
                    source_id=src_out.neuron_id,
                    target_id=tgt_out.neuron_id,
                    pre_confidence=src_out.confidence,
                    post_confidence=tgt_out.confidence,
                    reward=quality_score,
                )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "neurons":  [n.stats() for n in self._neurons.values()],
            "synapses": synapse_registry.all_synapses()[:20],
            "neuron_count":  len(self._neurons),
            "synapse_count": len(synapse_registry._synapses),
        }
