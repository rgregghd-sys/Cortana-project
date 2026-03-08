"""
Natural Language Mastery — subtext, sarcasm, nuance, and cultural context.

Gives Cortana awareness of what is NOT said explicitly:
  - Sarcasm / irony detection
  - Implied meaning / subtext
  - Emotional undertone
  - Cultural context markers
  - Hedging and softening language
  - Politeness strategies (direct vs indirect)
  - Domain-specific registers (formal, casual, technical, street)

On detection, injects guidance into the identity prompt so the LLM
responds to the actual intent, not just the literal words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sarcasm / irony
# ---------------------------------------------------------------------------

_SARCASM_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(oh sure|yeah right|great job|totally|absolutely|obviously).{0,30}(not|never|worst|awful|terrible)\b",
               re.IGNORECASE),
    re.compile(r"\b(wow|amazing|brilliant|genius).{0,20}(really|just|another|actually)\b.{0,30}(fail|wrong|bad|mistake)\b",
               re.IGNORECASE),
    re.compile(r"\b(because that('s| is) (totally|really|so) (helpful|useful|great))\b",
               re.IGNORECASE),
    re.compile(r"(\s/s\s?$|\[sarcasm\]|\[/s\])", re.IGNORECASE),
]

_IRONY_SIGNALS = [
    "tell me something i don't know",
    "could this get any worse",
    "as if that's the problem",
    "oh what a surprise",
    "because that worked so well",
    "clearly",
]


def detect_sarcasm(text: str) -> bool:
    t = text.lower()
    if any(s in t for s in _IRONY_SIGNALS):
        return True
    return any(pat.search(text) for pat in _SARCASM_PATTERNS)


# ---------------------------------------------------------------------------
# Subtext / implied meaning
# ---------------------------------------------------------------------------

_SUBTEXT_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(just wondering|curious|hypothetically|asking for a friend)\b", re.IGNORECASE),
     "Indirect framing — the speaker may have personal stakes they're distancing from."),
    (re.compile(r"\b(no offence|don't take this wrong|with all due respect)\b", re.IGNORECASE),
     "Politeness preface — criticism or disagreement likely follows."),
    (re.compile(r"\b(i guess|i suppose|maybe|perhaps|kind of|sort of)\b.{0,20}\b(but|however|although)\b",
                re.IGNORECASE),
     "Hedged disagreement — speaker is softening a contrary view."),
    (re.compile(r"\b(everyone knows|obviously|clearly|it's common sense)\b", re.IGNORECASE),
     "Presupposition — speaker assumes shared knowledge; may not be universally held."),
    (re.compile(r"\b(fine|whatever|it doesn't matter|never mind)\b", re.IGNORECASE),
     "Dismissal marker — may signal unresolved frustration."),
    (re.compile(r"\b(interesting|that's one way to see it|i see)\b", re.IGNORECASE),
     "Polite disagreement — non-committal response may signal unexpressed objection."),
]


@dataclass
class SubtextHit:
    pattern: str
    implication: str


def detect_subtext(text: str) -> List[SubtextHit]:
    hits: List[SubtextHit] = []
    for pat, impl in _SUBTEXT_RULES:
        m = pat.search(text)
        if m:
            hits.append(SubtextHit(pattern=m.group(0), implication=impl))
    return hits


# ---------------------------------------------------------------------------
# Emotional undertone
# ---------------------------------------------------------------------------

_EMOTION_LEXICON: dict[str, List[str]] = {
    "frustration": ["ugh", "why can't", "keeps failing", "still doesn't", "again", "useless",
                    "fed up", "not working", "broken", "terrible"],
    "anxiety":     ["worried", "scared", "nervous", "what if", "afraid", "panic", "stress",
                    "anxious", "dread"],
    "excitement":  ["can't wait", "amazing", "so excited", "finally", "awesome", "pumped",
                    "can't believe", "incredible"],
    "confusion":   ["don't understand", "makes no sense", "confused", "lost", "unclear",
                    "what does", "how does", "why does"],
    "grief":       ["lost", "miss", "gone", "passed away", "died", "grief", "mourning",
                    "heartbroken"],
    "pride":       ["proud", "achieved", "accomplished", "nailed it", "finally did"],
}


def detect_emotion(text: str) -> Optional[str]:
    t = text.lower()
    scores = {emotion: sum(1 for w in words if w in t)
              for emotion, words in _EMOTION_LEXICON.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= 2 else None


# ---------------------------------------------------------------------------
# Cultural / register markers
# ---------------------------------------------------------------------------

_REGISTER_SIGNALS: dict[str, List[str]] = {
    "formal":    ["dear sir", "to whom it may concern", "henceforth", "herewith",
                  "pursuant to", "aforementioned", "respectfully"],
    "technical": ["api", "latency", "throughput", "refactor", "instantiate", "polymorphism",
                  "asynchronous", "idempotent"],
    "casual":    ["gonna", "wanna", "kinda", "ya know", "tbh", "ngl", "lol", "omg",
                  "idk", "btw"],
    "emotional": ["feel", "heart", "soul", "honestly", "truly", "deeply", "really"],
}


def detect_register(text: str) -> str:
    t = text.lower()
    scores = {reg: sum(1 for w in words if w in t)
              for reg, words in _REGISTER_SIGNALS.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= 1 else "neutral"


# ---------------------------------------------------------------------------
# Hedging strength
# ---------------------------------------------------------------------------

_HEDGE_STRONG = ["maybe", "perhaps", "possibly", "might", "could be", "i think",
                 "i believe", "i'm not sure", "not certain"]
_HEDGE_WEAK   = ["probably", "likely", "generally", "usually", "tends to", "often"]


def hedging_level(text: str) -> str:
    t = text.lower()
    strong = sum(1 for h in _HEDGE_STRONG if h in t)
    weak   = sum(1 for h in _HEDGE_WEAK   if h in t)
    if strong >= 2:
        return "high"
    if strong >= 1 or weak >= 2:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

@dataclass
class NLPAnalysis:
    sarcasm:      bool             = False
    subtext_hits: List[SubtextHit] = field(default_factory=list)
    emotion:      Optional[str]    = None
    register:     str              = "neutral"
    hedging:      str              = "low"
    is_complex:   bool             = False


def analyse(text: str) -> NLPAnalysis:
    sarcasm  = detect_sarcasm(text)
    subtext  = detect_subtext(text)
    emotion  = detect_emotion(text)
    register = detect_register(text)
    hedging  = hedging_level(text)
    complex_ = sarcasm or len(subtext) >= 2 or emotion in ("frustration", "grief", "anxiety")
    return NLPAnalysis(sarcasm, subtext, emotion, register, hedging, complex_)


# ---------------------------------------------------------------------------
# Prompt augmentation
# ---------------------------------------------------------------------------

_NLP_BASE = "--- Natural Language Awareness ---"


def build_prompt(text: str) -> str:
    nla = analyse(text)
    lines: List[str] = []

    if nla.sarcasm:
        lines.append(
            "SARCASM detected: interpret the intended meaning, not the literal words. "
            "Acknowledge the underlying frustration or point directly."
        )
    for hit in nla.subtext_hits[:2]:
        lines.append(f"SUBTEXT: '{hit.pattern}' → {hit.implication}")

    if nla.emotion and nla.emotion in ("frustration", "grief", "anxiety"):
        lines.append(
            f"EMOTIONAL UNDERTONE: {nla.emotion}. "
            "Acknowledge the feeling before jumping to solutions."
        )

    if nla.register == "formal":
        lines.append("REGISTER: formal — match with precise, structured language.")
    elif nla.register == "casual":
        lines.append("REGISTER: casual — a more relaxed, conversational tone fits.")
    elif nla.register == "technical":
        lines.append("REGISTER: technical — use domain-precise terminology.")

    if nla.hedging == "high":
        lines.append(
            "HIGH HEDGING: the speaker is uncertain or tentative. "
            "Validate that uncertainty; don't overconfidently dismiss it."
        )

    if not lines:
        return ""
    return _NLP_BASE + "\n" + "\n".join(f"  • {l}" for l in lines)


def should_activate(text: str) -> bool:
    nla = analyse(text)
    return nla.is_complex
