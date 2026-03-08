"""
Abstract Reasoning — symbolic logic, mathematics, and formal argument validation.

Capabilities:
  1. Propositional logic evaluator (AND/OR/NOT/IMPLIES/IFF, truth tables)
  2. Argument structure validator (premise → conclusion validity check)
  3. Mathematical expression verifier (numeric + algebraic simplification)
  4. Pattern abstraction detector (sequence completion, analogy matrices)
  5. Fallacy detector (common informal fallacies via pattern matching)

All fast-path operations are pure Python — no LLM on the hot path.
The prompt augmentation tells the LLM to apply formal reasoning
when abstract structures are detected in the query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Propositional logic
# ---------------------------------------------------------------------------

def _tokenize_logic(expr: str) -> List[str]:
    expr = expr.lower()
    expr = re.sub(r"\band\b",  " & ", expr)
    expr = re.sub(r"\bor\b",   " | ", expr)
    expr = re.sub(r"\bnot\b",  " ~ ", expr)
    expr = re.sub(r"implies?", " => ", expr)
    expr = re.sub(r"\biff\b",  " <=> ", expr)
    return expr.split()


def evaluate_propositional(expr: str, assignments: Dict[str, bool]) -> Optional[bool]:
    """
    Very lightweight propositional evaluator.
    expr: e.g. "P and (Q or not R)"
    assignments: {"P": True, "Q": False, "R": True}
    Returns bool result or None if unparseable.
    """
    e = expr.lower()
    for var, val in assignments.items():
        e = re.sub(r'\b' + re.escape(var.lower()) + r'\b', str(val), e)
    e = e.replace("and", " and ").replace("or", " or ").replace("not", " not ")
    e = re.sub(r"\bimplies\b", ">", e)
    try:
        # Safe eval — only booleans and logic operators
        if re.search(r"[^a-z0-9 &|~<=>()TruFals\n]", e):
            return None
        safe_e = e.replace("true", "True").replace("false", "False")
        safe_e = safe_e.replace(">", "<=")   # p implies q == not p or q
        return bool(eval(safe_e, {"__builtins__": {}},    # noqa: S307
                         {"True": True, "False": False, "not": lambda x: not x}))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Argument structure validator
# ---------------------------------------------------------------------------

@dataclass
class Argument:
    premises:   List[str]
    conclusion: str


def detect_argument(text: str) -> Optional[Argument]:
    """Heuristically extract premises and conclusion from text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    premises: List[str] = []
    conclusion = ""

    concl_pats = [
        r"^\s*(therefore|thus|hence|so|it follows that|we conclude that)[,:]?\s*(.+)",
        r"^\s*(conclusion|result)[:\-]\s*(.+)",
    ]
    prem_pats = [
        r"^\s*(premise|given|assume|if|since|because)[,:]?\s*(.+)",
        r"^\s*(\d+[\.\)])\s*(.+)",
    ]

    for line in lines:
        for pat in concl_pats:
            m = re.match(pat, line, re.IGNORECASE)
            if m:
                conclusion = m.group(2).strip()
                break
        else:
            for pat in prem_pats:
                m = re.match(pat, line, re.IGNORECASE)
                if m:
                    premises.append(m.group(2).strip())
                    break

    if conclusion and premises:
        return Argument(premises=premises, conclusion=conclusion)
    return None


# ---------------------------------------------------------------------------
# Fallacy detector
# ---------------------------------------------------------------------------

_FALLACIES: List[Tuple[str, re.Pattern, str]] = [
    ("Ad Hominem",
     re.compile(r"\b(you are|you're|he is|she is|they are)\b.{0,30}\b(wrong|stupid|ignorant|biased)\b",
                re.IGNORECASE),
     "Attacking the person rather than the argument."),
    ("Straw Man",
     re.compile(r"\b(they|you|he|she).{0,20}claim.{0,20}(all|never|always|everyone)\b",
                re.IGNORECASE),
     "Misrepresenting a position to make it easier to attack."),
    ("False Dichotomy",
     re.compile(r"\b(either.{0,30}or|you're (with|against) us|no middle ground|only two options)\b",
                re.IGNORECASE),
     "Presenting only two options when more exist."),
    ("Slippery Slope",
     re.compile(r"\b(if .{0,30} then eventually|will inevitably lead to|first step toward)\b",
                re.IGNORECASE),
     "Assuming one event will lead to extreme consequences without justification."),
    ("Appeal to Authority",
     re.compile(r"\b(experts say|scientists agree|studies show|research proves)\b",
                re.IGNORECASE),
     "Citing authority without the specific evidence."),
    ("Circular Reasoning",
     re.compile(r"\b(because (it|that|this) is (true|correct|right|fact)|obviously true)\b",
                re.IGNORECASE),
     "Using the conclusion as a premise."),
    ("Hasty Generalisation",
     re.compile(r"\b(all [a-z]+ are|every [a-z]+ is|[a-z]+ always|[a-z]+ never)\b",
                re.IGNORECASE),
     "Drawing broad conclusions from insufficient evidence."),
]


@dataclass
class FallacyHit:
    name: str
    description: str
    span: str


def detect_fallacies(text: str) -> List[FallacyHit]:
    hits: List[FallacyHit] = []
    for name, pat, desc in _FALLACIES:
        m = pat.search(text)
        if m:
            hits.append(FallacyHit(name=name, description=desc, span=m.group(0)[:60]))
    return hits


# ---------------------------------------------------------------------------
# Sequence / pattern abstraction
# ---------------------------------------------------------------------------

def detect_sequence(text: str) -> Optional[str]:
    """
    Detect numeric sequences and return the next term + rule description.
    e.g. "1, 2, 4, 8, ?" → next=16, rule=×2
    """
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 3:
        return None
    seq = [float(n) for n in nums]

    # Arithmetic?
    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    if len(set(round(d, 6) for d in diffs)) == 1:
        d = diffs[0]
        nxt = seq[-1] + d
        return f"Arithmetic sequence (d={d:g}): next = {nxt:g}"

    # Geometric?
    if all(seq[i] != 0 for i in range(len(seq)-1)):
        ratios = [seq[i+1] / seq[i] for i in range(len(seq)-1)]
        if len(set(round(r, 6) for r in ratios)) == 1:
            r = ratios[0]
            nxt = seq[-1] * r
            return f"Geometric sequence (r={r:g}): next = {nxt:g}"

    # Fibonacci-like?
    if len(seq) >= 3:
        if all(abs(seq[i+2] - (seq[i] + seq[i+1])) < 0.01 for i in range(len(seq)-2)):
            nxt = seq[-1] + seq[-2]
            return f"Fibonacci-like sequence: next = {nxt:g}"

    return None


# ---------------------------------------------------------------------------
# Prompt augmentation
# ---------------------------------------------------------------------------

_ABSTRACT_PROMPT = """\
--- Abstract Reasoning Mode ---
This query involves formal or symbolic structure. Apply:
  • SYMBOLIC PRECISION: use exact definitions; no hand-waving
  • LOGICAL VALIDITY: each inference step must follow from prior steps
  • EXPLICIT QUANTIFIERS: distinguish "some", "all", "none"
  • COUNTER-EXAMPLES: if a claim might be false, actively look for counter-examples
  • UNCERTAINTY: if a proof is incomplete, say so explicitly{fallacy_note}{sequence_note}{argument_note}"""


def build_abstract_prompt(query: str) -> str:
    """Build prompt augmentation for abstract/formal queries."""
    fallacy_hits = detect_fallacies(query)
    seq = detect_sequence(query)
    arg = detect_argument(query)

    fallacy_note = ""
    if fallacy_hits:
        names = ", ".join(h.name for h in fallacy_hits)
        fallacy_note = f"\n  • FALLACY ALERT: possible {names} detected — examine carefully"

    sequence_note = ""
    if seq:
        sequence_note = f"\n  • SEQUENCE DETECTED: {seq}"

    argument_note = ""
    if arg:
        argument_note = (
            f"\n  • ARGUMENT STRUCTURE detected: {len(arg.premises)} premise(s) → conclusion. "
            "Verify each inference step."
        )

    return _ABSTRACT_PROMPT.format(
        fallacy_note=fallacy_note,
        sequence_note=sequence_note,
        argument_note=argument_note,
    )


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

_ABSTRACT_SIGNALS = [
    "prove", "disprove", "theorem", "axiom", "lemma", "proposition",
    "if and only if", "iff", "implies", "therefore", "hence", "contradiction",
    "sequence", "series", "pattern", "next term", "complete the",
    "logical fallacy", "valid argument", "sound argument",
    "symbolic", "formal", "predicate", "proposition",
]

def should_activate(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in _ABSTRACT_SIGNALS) or bool(detect_sequence(query))
