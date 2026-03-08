"""
Security Neuron — Runs L10 security checks (prompt injection, logic flaws,
memory poisoning, tool misuse) in parallel before reasoning fires.
If a threat is detected, injects a security alert into the workspace.
"""
from __future__ import annotations

from typing import Any

from cortana.nexus.neuron import Neuron, NeuronSignal, NeuronType


class SecurityNeuron(Neuron):
    def __init__(self, security_layer: Any = None) -> None:
        super().__init__("security", NeuronType.SECURITY, threshold=0.0)
        self.security_layer = security_layer

    def compute(self, signal: NeuronSignal) -> str:
        if not self.security_layer:
            return ""
        try:
            state = signal.payload.get("state")
            result = self.security_layer.analyze(signal.query, state)
            workspace = signal.payload.get("_workspace")
            if workspace is not None:
                workspace.metadata["security_result"] = result

            if getattr(result, "red_wins", False):
                vulns = getattr(result, "vulnerabilities", [])
                descs = "; ".join(
                    getattr(v, "description", str(v)) for v in vulns[:3]
                )
                return f"[Security Alert] Threats detected: {descs}"
            return ""
        except Exception:
            return ""

    def score_confidence(self, content: str, signal: NeuronSignal) -> float:
        return 0.95 if content else 0.5
