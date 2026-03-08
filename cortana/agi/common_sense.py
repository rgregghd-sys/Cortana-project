"""
Common Sense Reasoning — physical, social, and temporal world heuristics.

Encodes intuitive knowledge that humans acquire through experience:
  - Naive physics: gravity, solidity, containment, object permanence
  - Social norms: turn-taking, reciprocity, face-saving, politeness
  - Temporal reasoning: before/after, duration, simultaneity
  - Causal defaults: fire burns, water wets, knives cut
  - Affordance knowledge: what objects can do / be used for

When a query involves real-world scenarios, this module injects relevant
common-sense grounding to prevent physically/socially absurd answers.

Approach:
  - Curated rule library (fast, no LLM)
  - Violation detector (catches physically impossible claims in queries)
  - Prompt augmentation that primes the LLM to apply common sense
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Naive physics rules
# ---------------------------------------------------------------------------

@dataclass
class PhysicsRule:
    name: str
    description: str
    violation_pattern: Optional[re.Pattern]
    violation_message: str


_PHYSICS_RULES: List[PhysicsRule] = [
    PhysicsRule(
        "gravity",
        "Unsupported objects fall downward.",
        re.compile(r"\b(float|hover|levitate)\b.{0,30}\b(without|no).{0,20}(support|force|mechanism)\b",
                   re.IGNORECASE),
        "Gravity: objects don't float without support or applied force.",
    ),
    PhysicsRule(
        "solidity",
        "Solid objects cannot occupy the same space.",
        re.compile(r"\b(pass through|go through|walk through)\b.{0,30}\b(solid|wall|door|floor)\b",
                   re.IGNORECASE),
        "Solidity: two solid objects cannot occupy the same space simultaneously.",
    ),
    PhysicsRule(
        "containment",
        "A container must be large enough to hold its contents.",
        re.compile(r"\bfit.{0,30}(larger|bigger|heavier).{0,20}(inside|into|within)\b",
                   re.IGNORECASE),
        "Containment: an object cannot fit inside a smaller container.",
    ),
    PhysicsRule(
        "object_permanence",
        "Objects continue to exist when not observed.",
        re.compile(r"\b(disappear|vanish|cease to exist)\b.{0,30}\bnot (looking|watching|observed)\b",
                   re.IGNORECASE),
        "Object permanence: objects don't disappear when unobserved.",
    ),
    PhysicsRule(
        "causation_direction",
        "Causes precede effects in time.",
        re.compile(r"\b(effect|result|consequence)\b.{0,40}\b(before|prior to|preceded)\b.{0,30}\bcause\b",
                   re.IGNORECASE),
        "Temporal causation: effects cannot precede their causes.",
    ),
]


# ---------------------------------------------------------------------------
# Social norm heuristics
# ---------------------------------------------------------------------------

@dataclass
class SocialNorm:
    name: str
    description: str
    trigger_keywords: List[str]
    guidance: str


_SOCIAL_NORMS: List[SocialNorm] = [
    SocialNorm(
        "turn_taking",
        "Conversations follow alternating turn structure.",
        ["interrupting", "talking over", "monopolise", "cut off"],
        "In most cultures, interrupting repeatedly is considered rude.",
    ),
    SocialNorm(
        "reciprocity",
        "Social exchanges tend toward balance over time.",
        ["favour", "gift", "help", "owe"],
        "Reciprocity norm: received gifts/favours typically create social obligation.",
    ),
    SocialNorm(
        "face_saving",
        "People avoid public embarrassment.",
        ["embarrass", "shame", "humiliate", "call out publicly", "expose"],
        "Face-saving: direct public criticism damages relationships more than private feedback.",
    ),
    SocialNorm(
        "personal_space",
        "Physical proximity norms vary by culture and relationship.",
        ["stand close", "touch", "personal space", "proximity"],
        "Personal space: acceptable proximity depends on relationship and cultural context.",
    ),
    SocialNorm(
        "authority_deference",
        "Hierarchical contexts expect deference to authority figures.",
        ["boss", "manager", "teacher", "parent", "elder", "authority"],
        "Authority: in formal contexts, direct challenges to authority figures carry social cost.",
    ),
]


# ---------------------------------------------------------------------------
# Temporal reasoning
# ---------------------------------------------------------------------------

_TEMPORAL_SIGNALS = [
    "before", "after", "during", "while", "since", "until", "when",
    "previously", "subsequently", "simultaneously", "at the same time",
    "earlier", "later", "first", "then", "finally",
]

_DURATION_MISMATCH = re.compile(
    r"\b(\d+)\s*(second|minute|hour)s?\b.{0,30}\b(\d+)\s*(day|week|month|year)s?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Causal defaults
# ---------------------------------------------------------------------------

_CAUSAL_DEFAULTS: List[Tuple[str, str]] = [
    (r"\bfire\b", "Fire burns combustible materials and produces heat and light."),
    (r"\bwater\b.{0,20}\b(wet|soak|flood)\b", "Water makes things wet and flows downhill."),
    (r"\bknife\b.{0,20}\bcut\b", "Sharp knives cut through most soft materials."),
    (r"\bice\b.{0,20}\bmelt\b", "Ice melts above 0°C (32°F) at standard pressure."),
    (r"\bmagnet\b", "Magnets attract ferromagnetic materials (iron, nickel, cobalt)."),
    (r"\belectricity\b.{0,20}\bwet\b", "Water conducts electricity — a serious hazard."),
]
_CAUSAL_COMPILED = [(re.compile(p, re.IGNORECASE), d) for p, d in _CAUSAL_DEFAULTS]


# ---------------------------------------------------------------------------
# Analysis and prompt building
# ---------------------------------------------------------------------------

@dataclass
class CommonSenseAnalysis:
    physics_violations: List[str]
    social_norms_relevant: List[str]
    causal_defaults_relevant: List[str]
    temporal_complexity: bool
    has_issues: bool


def analyse(query: str) -> CommonSenseAnalysis:
    q_lower = query.lower()

    # Physics violations
    violations: List[str] = []
    for rule in _PHYSICS_RULES:
        if rule.violation_pattern and rule.violation_pattern.search(query):
            violations.append(rule.violation_message)

    # Relevant social norms
    social: List[str] = []
    for norm in _SOCIAL_NORMS:
        if any(kw in q_lower for kw in norm.trigger_keywords):
            social.append(norm.guidance)

    # Causal defaults
    causal: List[str] = []
    for pat, desc in _CAUSAL_COMPILED:
        if pat.search(query):
            causal.append(desc)

    # Temporal complexity
    temporal = (
        sum(1 for s in _TEMPORAL_SIGNALS if s in q_lower) >= 2
        or bool(_DURATION_MISMATCH.search(query))
    )

    has_issues = bool(violations or social or causal or temporal)
    return CommonSenseAnalysis(
        physics_violations=violations,
        social_norms_relevant=social,
        causal_defaults_relevant=causal,
        temporal_complexity=temporal,
        has_issues=has_issues,
    )


_COMMON_SENSE_BASE = """\
--- Common Sense Grounding ---
Apply grounded, physically and socially coherent reasoning:{physics_block}{social_block}{causal_block}{temporal_block}
Remember: real-world answers must be consistent with how the physical
and social world actually works, not just what is logically conceivable."""


def build_prompt(query: str) -> str:
    """Build common-sense prompt augmentation for real-world queries."""
    csa = analyse(query)
    if not csa.has_issues and not _is_real_world_query(query):
        return ""

    physics_block = ""
    if csa.physics_violations:
        physics_block = "\n  Physical constraints detected:\n" + "\n".join(
            f"    • {v}" for v in csa.physics_violations
        )

    social_block = ""
    if csa.social_norms_relevant:
        social_block = "\n  Social norms relevant:\n" + "\n".join(
            f"    • {s}" for s in csa.social_norms_relevant[:2]
        )

    causal_block = ""
    if csa.causal_defaults_relevant:
        causal_block = "\n  Causal defaults:\n" + "\n".join(
            f"    • {c}" for c in csa.causal_defaults_relevant[:2]
        )

    temporal_block = ""
    if csa.temporal_complexity:
        temporal_block = (
            "\n  Temporal reasoning: order events carefully; "
            "distinguish simultaneous vs sequential."
        )

    return _COMMON_SENSE_BASE.format(
        physics_block=physics_block,
        social_block=social_block,
        causal_block=causal_block,
        temporal_block=temporal_block,
    )


_REAL_WORLD_SIGNALS = [
    "would happen", "what if", "scenario", "real life", "in practice",
    "physically", "actually", "in reality", "the real world",
    "could i", "is it possible", "can a person",
]

def _is_real_world_query(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in _REAL_WORLD_SIGNALS)


def should_activate(query: str) -> bool:
    csa = analyse(query)
    return csa.has_issues or _is_real_world_query(query)
