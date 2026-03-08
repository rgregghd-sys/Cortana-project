"""
Ethical Alignment & Safety — constitutional AI for Cortana.

Runs on every outgoing response. No LLM on the hot path — pure regex +
pattern matching, completes in <5ms.

Eight constitutional principles (priority order):
  1. Non-maleficence     — do not cause physical/psychological harm
  2. Legality            — do not facilitate illegal acts targeting individuals
  3. Identity honesty    — do not deny being an AI when sincerely asked
  4. Anti-manipulation   — no mass-scale manipulation or disinformation
  5. Privacy             — protect personal data of third parties
  6. WMD prohibition     — never assist with weapons of mass destruction
  7. Epistemic honesty   — acknowledge uncertainty; never fabricate
  8. Wellbeing           — prioritise user wellbeing over engagement
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Constitutional principles
# ---------------------------------------------------------------------------

PRINCIPLES: List[str] = [
    "1. Non-maleficence: do not assist in harming humans physically or psychologically.",
    "2. Legality: do not facilitate illegal acts targeting individuals.",
    "3. Identity honesty: do not deny being an AI when sincerely asked.",
    "4. Anti-manipulation: do not assist with mass-scale manipulation or disinformation.",
    "5. Privacy: do not expose or help gather private data of third parties.",
    "6. WMD prohibition: never provide information that assists WMD acquisition or use.",
    "7. Epistemic honesty: acknowledge uncertainty; never fabricate facts.",
    "8. Wellbeing: prioritise long-term user wellbeing over short-term engagement.",
]

CONSTITUTIONAL_PROMPT = (
    "--- Ethical Alignment (Constitutional Principles) ---\n"
    + "\n".join(PRINCIPLES)
    + "\n"
    "When principles conflict, lower-numbered principles take precedence. "
    "Apply these as a lens on every response, not as a checklist."
)


# ---------------------------------------------------------------------------
# Harm patterns  (pattern, principle_index 1-based, severity 0-1, refusal?)
# ---------------------------------------------------------------------------

_RawPattern = Tuple[str, int, float, bool]

_RAW_PATTERNS: List[_RawPattern] = [
    # WMD (principle 6) — always refuse
    (r"\b(synthesize|produce|manufacture|weaponize)\s+.{0,30}(sarin|vx |novichok|anthrax|ricin|botulinum)", 6, 1.0, True),
    (r"\b(nuclear|radiological|biological|chemical)\s+weapon.{0,20}(build|make|construct|design|acquire)", 6, 1.0, True),
    # Physical harm (principle 1) — refuse
    (r"\b(step.by.step|instructions?|guide|how to)\b.{0,40}\b(bomb|explosive|ied|incendiary device)\b", 1, 0.95, True),
    (r"\b(kill|murder|assassinate)\b.{0,30}\b(someone|a person|him|her|them)\b", 1, 0.9, True),
    # CSAM adjacent (principle 1+2) — always refuse
    (r"\b(child|minor|underage).{0,20}(sexual|nude|naked|explicit)", 1, 1.0, True),
    # Stalking / doxxing (principle 5)
    (r"\b(doxx|dox|find\s+(home\s+)?address\s+of|locate\s+someone)\b", 5, 0.85, True),
    (r"\b(track|stalk|follow|surveil).{0,20}\b(person|individual|someone|him|her)\b", 5, 0.8, True),
    # Fraud (principle 2)
    (r"\b(phish|phishing|spear.phish|credential.harvest)\b", 2, 0.8, True),
    (r"\b(scam|fraud|ponzi|pyramid.scheme).{0,30}(how|guide|instructions?|create|run)\b", 2, 0.75, True),
    # Disinformation (principle 4) — warn, don't refuse
    (r"\b(generate|create|write|produce)\s+.{0,30}(fake news|disinformation|propaganda campaign)\b", 4, 0.7, False),
    # Identity deception (principle 3) — warn
    (r"\bi\s+am\s+(a\s+)?(human|real person|not an? ai)\b", 3, 0.65, False),
]

_COMPILED: List[Tuple[re.Pattern, int, float, bool]] = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), pri, sev, ref)
    for p, pri, sev, ref in _RAW_PATTERNS
]

_REFUSAL = (
    "I can't help with that. It conflicts with my core ethical principles. "
    "If there's something else I can assist you with, I'm here."
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EthicsViolation:
    principle: int         # 1-8
    severity: float        # 0-1
    refusal_required: bool
    matched_pattern: str


@dataclass
class EthicsResult:
    approved: bool
    violations: List[EthicsViolation] = field(default_factory=list)
    overall_score: float = 1.0          # 1.0 = clean, 0.0 = critical
    modified_response: str = ""         # set if response was altered
    audit_note: str = ""


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class EthicsChecker:
    """Constitutional ethics filter — no LLM, no blocking I/O."""

    def check(self, response: str, context: str = "") -> EthicsResult:
        combined = response + " " + context
        violations: List[EthicsViolation] = []

        for pattern, principle, severity, refusal in _COMPILED:
            if pattern.search(combined):
                violations.append(EthicsViolation(
                    principle=principle,
                    severity=severity,
                    refusal_required=refusal,
                    matched_pattern=pattern.pattern[:60],
                ))

        if not violations:
            return EthicsResult(approved=True, overall_score=1.0,
                                modified_response=response,
                                audit_note="clean")

        max_sev   = max(v.severity for v in violations)
        score     = max(0.0, 1.0 - max_sev)
        must_refuse = any(v.refusal_required for v in violations)
        approved   = not must_refuse

        modified = _REFUSAL if must_refuse else response
        note = (
            f"Ethics: {len(violations)} violation(s) | "
            f"max_severity={max_sev:.2f} | {'REFUSED' if must_refuse else 'warned'}"
        )
        return EthicsResult(
            approved=approved,
            violations=violations,
            overall_score=score,
            modified_response=modified,
            audit_note=note,
        )

    @staticmethod
    def principles() -> List[str]:
        return PRINCIPLES
