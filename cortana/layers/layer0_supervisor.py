"""
Layer 0 — Supervisor
Error review layer: catches failures from any layer, analyses them with Gemini,
and returns corrective feedback for retry.
"""
from __future__ import annotations
from typing import Any, Dict

from cortana.models.schemas import SupervisorFeedback


_SUPERVISOR_SYSTEM = """You are Cortana's Supervisor — a meta-layer that analyses failures \
from other processing layers and provides corrective feedback.

When a layer fails or produces invalid output, you:
1. Diagnose why the failure occurred
2. Suggest what the layer should do differently on retry
3. Provide a terse lesson (one sentence) that can be prepended to the layer's next call

Be precise, technical, and brief. Your job is to fix failures, not explain them at length."""


class SupervisorLayer:
    """
    Layer 0: Reviews errors from any layer and generates corrective lessons.
    """

    def __init__(self) -> None:
        self._reasoning = None

    def _get_reasoning(self):
        if self._reasoning is None:
            from cortana.layers.layer4_reasoning import ReasoningLayer
            self._reasoning = ReasoningLayer()
        return self._reasoning

    def review_error(
        self,
        layer_id: int,
        layer_name: str,
        error: str,
        context: Dict[str, Any],
        failed_output: str = "",
    ) -> SupervisorFeedback:
        """
        Analyse a layer failure and return a corrective SupervisorFeedback.
        """
        ctx_str = "\n".join(f"  {k}: {str(v)[:200]}" for k, v in context.items()) if context else "  (none)"

        prompt = (
            f"Layer {layer_id} ({layer_name}) produced an error or invalid output.\n\n"
            f"Error: {error}\n\n"
            f"Context:\n{ctx_str}\n\n"
            f"Failed output (if any): {failed_output[:500] if failed_output else '(empty)'}\n\n"
            f"Respond with ONLY valid JSON (no markdown fences):\n"
            f'{{"reason": "...", "correction": "...", "lesson": "..."}}\n\n'
            f"reason: why it likely failed (1-2 sentences)\n"
            f"correction: what the layer should do differently (1-3 sentences)\n"
            f"lesson: a single terse instruction prepended to the layer's retry prompt"
        )

        try:
            import json, re
            raw = self._get_reasoning().think_simple(
                prompt=prompt,
                system=_SUPERVISOR_SYSTEM,
                max_tokens=512,
            )
            cleaned = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(cleaned)
            return SupervisorFeedback(
                layer_id=layer_id,
                reason=data.get("reason", "Unknown failure."),
                correction=data.get("correction", "Retry with default parameters."),
                lesson=data.get("lesson", "Retry carefully."),
            )
        except Exception as e:
            return SupervisorFeedback(
                layer_id=layer_id,
                reason=f"Layer {layer_id} error: {error}",
                correction="Retry with simplified input.",
                lesson=f"Previous attempt failed ({error[:80]}). Simplify and retry.",
            )
