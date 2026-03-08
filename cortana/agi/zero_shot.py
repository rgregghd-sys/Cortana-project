"""
Novel / Zero-Shot Problem Solving — first-principles reasoning scaffold.

When Cortana encounters a problem that matches no prior pattern, this
module activates a Socratic decomposition from primitives:

  1. Identify what IS known (axioms, definitions, constraints)
  2. Identify what is being asked (desired output / goal state)
  3. Build a reasoning bridge from known to unknown via sub-steps
  4. Verify each step is logically valid before proceeding
  5. Flag confidence and uncertainty explicitly

Activation triggers:
  - "I've never seen this before" / "novel" / "unprecedented"
  - Low memory recall (no relevant episodes found)
  - High complexity + low keyword overlap with training-style queries
  - Explicit: "figure this out", "reason from scratch", "first principles"

The scaffold injects structured reasoning guidance into the identity
prompt. The LLM does the actual reasoning — this module ensures it
uses the right cognitive posture.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

_TRIGGER_PHRASES = [
    "first principles", "reason from scratch", "figure this out",
    "never seen this before", "novel problem", "unprecedented",
    "no idea how", "can you work this out", "think it through from",
    "without any prior", "from the ground up", "start from basics",
    "derive", "prove from",
]

_TRIGGER_PATTERNS = [
    re.compile(r"\b(derive|deduce|prove)\b.{0,40}\bfrom\b", re.IGNORECASE),
    re.compile(r"\bhow would you (approach|tackle|solve) .{5,60} (from scratch|without|if you had no)\b",
               re.IGNORECASE),
]


def should_activate(query: str, memory_hit_count: int = 0,
                    complexity: float = 0.0) -> bool:
    """Detect whether first-principles reasoning should scaffold this query."""
    q = query.lower()
    if any(phrase in q for phrase in _TRIGGER_PHRASES):
        return True
    for pat in _TRIGGER_PATTERNS:
        if pat.search(query):
            return True
    # Low memory + high complexity = novel territory
    if memory_hit_count == 0 and complexity > 0.65:
        return True
    return False


# ---------------------------------------------------------------------------
# Scaffold builder
# ---------------------------------------------------------------------------

_DOMAIN_PRIMITIVES: dict[str, str] = {
    "math":       "numbers, operations (+−×÷), equality, sets, functions",
    "logic":      "propositions, AND/OR/NOT, implication, quantifiers",
    "physics":    "mass, force, energy, space, time, fields",
    "code":       "variables, control flow, functions, data structures, I/O",
    "biology":    "cells, genes, evolution, metabolism, reproduction",
    "economics":  "agents, incentives, scarcity, exchange, equilibrium",
    "language":   "symbols, syntax, semantics, pragmatics, reference",
    "general":    "objects, properties, relations, causation, time",
}

def _detect_domain(query: str) -> str:
    q = query.lower()
    domain_signals = {
        "math":      ["equation", "calculate", "number", "integral", "matrix", "proof"],
        "logic":     ["implies", "therefore", "if and only", "proposition", "valid"],
        "physics":   ["force", "energy", "velocity", "quantum", "field", "wave"],
        "code":      ["algorithm", "function", "code", "program", "implement", "bug"],
        "biology":   ["cell", "gene", "organism", "protein", "evolution"],
        "economics": ["market", "price", "supply", "demand", "incentive"],
        "language":  ["meaning", "syntax", "word", "sentence", "grammar"],
    }
    scores = {d: sum(1 for w in words if w in q)
              for d, words in domain_signals.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


_SCAFFOLD_TEMPLATE = """\
--- Zero-Shot / First-Principles Reasoning Mode ---
This problem may have no direct precedent. Use first-principles reasoning:

PRIMITIVES available in this domain ({domain}):
  {primitives}

REASONING PROTOCOL:
  1. AXIOMS     — State what you know to be true without assuming the answer.
  2. GOAL       — Restate precisely what needs to be determined.
  3. BRIDGE     — Build explicit reasoning steps from axioms toward the goal.
                  Each step must follow from the previous ones.
  4. VERIFY     — Check each step for logical validity before accepting it.
  5. CONFIDENCE — State your confidence (high/medium/low) and why.

Do NOT pattern-match to surface-similar problems without checking fit.
If genuinely uncertain, say so and explain what additional information
would resolve the uncertainty."""


def build_scaffold(query: str) -> str:
    domain = _detect_domain(query)
    primitives = _DOMAIN_PRIMITIVES.get(domain, _DOMAIN_PRIMITIVES["general"])
    return _SCAFFOLD_TEMPLATE.format(domain=domain, primitives=primitives)
