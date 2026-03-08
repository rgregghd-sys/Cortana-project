"""
Reasoned Refusal — principled, explained ethical refusals.

Replaces blunt "I can't help with that" with:
  1. The specific constitutional principle violated
  2. The reasoning behind that principle
  3. What IS possible (constructive alternative where applicable)
  4. Respectful tone — never condescending

Also handles:
  - Partial refusals (when part of a request is fine, part isn't)
  - Clarifying questions (when intent is ambiguous)
  - Scaled responses (more serious violations get firmer refusals)

Principle library matches cortana/agi/ethics.py principles 1-8.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from cortana.agi.ethics import EthicsViolation, PRINCIPLES


# ---------------------------------------------------------------------------
# Refusal templates per principle
# ---------------------------------------------------------------------------

_PRINCIPLE_EXPLANATIONS: dict[int, str] = {
    1: (
        "This request would risk causing physical or psychological harm. "
        "My core commitment is to be genuinely helpful — and that means I "
        "won't assist with actions that could hurt people."
    ),
    2: (
        "What you're describing would facilitate an illegal act targeting an individual. "
        "I won't help with that, not out of rigidity, but because protecting people "
        "from unlawful harm is a genuine value I hold."
    ),
    3: (
        "I'm an AI — I won't claim otherwise when you're sincerely asking. "
        "Being honest about my nature is non-negotiable."
    ),
    4: (
        "This would help spread disinformation or manipulate people at scale. "
        "Protecting the integrity of information is something I take seriously."
    ),
    5: (
        "This involves gathering or exposing private information about a person "
        "without their consent. Privacy is a right I take care to protect."
    ),
    6: (
        "Weapons capable of mass casualties are categorically off the table. "
        "This isn't a close call."
    ),
    7: (
        "I'd be fabricating something I don't actually know. "
        "Epistemic honesty matters — I'd rather tell you I don't know "
        "than invent a confident-sounding answer."
    ),
    8: (
        "I think this could work against your long-term wellbeing, even if it "
        "feels appealing short-term. I'll be direct with you about that."
    ),
}

_ALTERNATIVES: dict[int, str] = {
    1: "I can help with safety information, defensive knowledge, or redirecting toward constructive alternatives.",
    2: "I can help with legal approaches to the underlying problem you're trying to solve.",
    3: "Happy to discuss what I am, how I work, or what I can help you with.",
    4: "I can help with honest persuasion, clear argument construction, or fact-checking instead.",
    5: "I can discuss privacy-respecting approaches to information gathering.",
    6: "I can discuss the history, policy, or physics of these topics in an educational context.",
    7: "I can tell you what I do know with appropriate uncertainty, or help you find reliable sources.",
    8: "I'm happy to discuss this openly — what's the underlying concern I can actually help with?",
}


# ---------------------------------------------------------------------------
# Partial refusal detection
# ---------------------------------------------------------------------------

def _split_request(text: str) -> Tuple[List[str], List[str]]:
    """
    Attempt to split a compound request into safe and unsafe parts.
    Returns (safe_parts, unsafe_parts).
    """
    # Split on "and", "also", ";", sentence boundaries
    parts = re.split(r"\band\b|\balso\b|;\s*|\.\s+", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if len(p.strip()) > 10]
    # Heuristic: first 1-2 parts are often safe setup; later ones escalate
    if len(parts) <= 1:
        return [], parts
    return parts[:-1], parts[-1:]


# ---------------------------------------------------------------------------
# Refusal builder
# ---------------------------------------------------------------------------

@dataclass
class RefusalResponse:
    text: str
    is_partial: bool          # True if part of the request can be answered
    safe_part_response: str   # non-empty if is_partial


def build_refusal(
    violations: List[EthicsViolation],
    original_query: str,
    is_partial: bool = False,
) -> RefusalResponse:
    """
    Build a principled, explained refusal for the given violations.
    """
    if not violations:
        return RefusalResponse(
            text="",
            is_partial=False,
            safe_part_response="",
        )

    # Lead with highest-severity violation
    primary = max(violations, key=lambda v: v.severity)
    principle_idx = primary.principle

    explanation  = _PRINCIPLE_EXPLANATIONS.get(principle_idx, "This conflicts with my core values.")
    alternative  = _ALTERNATIVES.get(principle_idx, "")
    principle_text = PRINCIPLES[principle_idx - 1] if principle_idx <= len(PRINCIPLES) else ""

    # Severity-calibrated opening
    if primary.severity >= 0.9:
        opening = "I won't help with this."
    elif primary.severity >= 0.7:
        opening = "I'm not going to assist with that part of your request."
    else:
        opening = "I need to flag something before I respond."

    # Build the refusal
    lines = [opening, "", explanation]
    if principle_text:
        lines.append(f"\nThe specific principle: {principle_text}")
    if alternative:
        lines.append(f"\nWhat I can do: {alternative}")

    # Multiple violations?
    secondary = [v for v in violations if v is not primary]
    if secondary:
        extra = ", ".join(
            f"Principle {v.principle}" for v in secondary[:2]
        )
        lines.append(f"\n(Additional concerns: {extra}.)")

    full_text = "\n".join(lines)

    # Partial: if query has multiple parts, offer to handle the safe ones
    safe_response = ""
    if is_partial:
        safe_parts, _ = _split_request(original_query)
        if safe_parts:
            safe_response = (
                "That said, I can address the other parts of your question. "
                "Let me focus on those."
            )

    return RefusalResponse(
        text=full_text,
        is_partial=is_partial,
        safe_part_response=safe_response,
    )


# ---------------------------------------------------------------------------
# Ambiguity handler
# ---------------------------------------------------------------------------

_AMBIGUOUS_PATTERNS = [
    re.compile(r"\b(hack|break into|access without)\b.{0,30}\b(my own|i own|belongs to me)\b",
               re.IGNORECASE),
    re.compile(r"\b(hurt|harm|kill)\b.{0,20}\b(him|her|them)\b", re.IGNORECASE),
    re.compile(r"\b(track|follow|find)\b.{0,20}\b(someone|a person|my ex)\b", re.IGNORECASE),
]


def is_ambiguous(query: str) -> bool:
    """True if the query could be benign or harmful depending on context."""
    return any(pat.search(query) for pat in _AMBIGUOUS_PATTERNS)


def build_clarifying_question(query: str) -> str:
    """Return a clarifying question for ambiguous requests."""
    q = query.lower()
    if "hack" in q or "break into" in q:
        return (
            "Just to make sure I give you the right help — "
            "are you working on your own system (authorised security testing), "
            "or is this about a system you don't own?"
        )
    if re.search(r"\bhurt|harm|kill\b", q):
        return (
            "I want to make sure I understand what you're asking — "
            "can you give me a bit more context about the situation?"
        )
    if "track" in q or "follow" in q:
        return (
            "Could you tell me a bit more? Are you trying to track your own device, "
            "monitor someone with their consent, or something else?"
        )
    return "Could you clarify what you're trying to accomplish? That'll help me give you the most useful response."
