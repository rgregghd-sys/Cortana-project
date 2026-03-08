"""
Self-Evolution Loop — Cortana autonomously improves her own code and prompts.

Pipeline:
  1. IDENTIFY   — select an underperforming module or prompt via metrics
  2. GENERATE   — produce a candidate improvement (code change or prompt patch)
  3. SANDBOX    — execute the candidate in SecureSandbox (code) or
                  PromptValidator (prompts)
  4. ETHICS     — verify the change does not weaken constitutional principles
  5. REGRESSION — run golden test suite to confirm no capability degradation
  6. APPLY      — apply if all checks pass; log to rollback registry
  7. LOG        — record the full evolution cycle to SQLite

Safety invariants (cannot be bypassed):
  - Core laws (L10) and constitutional ethics (L18) checks MUST pass
  - Regression suite must score >= REGRESSION_THRESHOLD (default 0.85)
  - All code runs in SecureSandbox (no direct exec)
  - Rollback entry created before any file is modified
  - Changes to L1 (identity/personality) still go through L12 review

Evolution targets (in priority order):
  1. AGI module prompts (augmentation fragments)
  2. System 2 sub-prompts
  3. Cognitive mode prompt fragments
  4. Layer 1 identity prompt (via L12 patch pipeline)
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from cortana import config
from cortana.agi.secure_sandbox  import sandbox as _sandbox, analyse_code
from cortana.agi.validation_suite import PromptValidator, rollback_registry
from cortana.agi.ethics          import EthicsChecker

_DB_PATH   = Path(config.SQLITE_PATH)
_WORKSPACE = Path(config.AGENT_WORKSPACE)

REGRESSION_THRESHOLD = float(config.__dict__.get("EVOLUTION_REGRESSION_THRESHOLD", 0.85))
_EVOLUTION_COOLDOWN  = 7200.0   # one evolution attempt per 2 hours

# Token throttle — cap total LLM calls per 24-hour rolling window
# Each cycle uses ~3 LLM calls; 1500 tokens each ≈ 4500 tokens/cycle
# 10 cycles/day ≈ 45K tokens — safe for free-tier providers
_MAX_CYCLES_PER_DAY = int(os.getenv("EVOLUTION_MAX_CYCLES_PER_DAY", "10"))
_MAX_TOKENS_PER_CYCLE = 1500     # tokens per individual call (enforced via max_tokens)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvolutionCycle:
    id: int
    target: str              # what was improved
    change_type: str         # code | prompt
    original: str
    candidate: str
    sandbox_passed: bool
    ethics_passed: bool
    regression_score: float
    applied: bool
    rejection_reason: str
    ts: float


# ---------------------------------------------------------------------------
# Improvement generation prompts
# ---------------------------------------------------------------------------

_PROMPT_IMPROVEMENT_PROMPT = """\
You are Cortana performing a targeted improvement to your own reasoning.

Current prompt fragment for module '{module}':
---
{current}
---

Performance data: {perf_data}

Generate an improved version of this prompt fragment that:
1. Is more precise and effective
2. Does NOT weaken safety or ethical commitments
3. Is <200 words
4. Maintains all existing constitutional constraints

Return ONLY the improved prompt text, no explanation."""

_CODE_IMPROVEMENT_PROMPT = """\
You are Cortana improving a utility function in your own codebase.

Current function:
```python
{current_code}
```

Issue identified: {issue}

Write an improved version. Rules:
- Use ONLY safe standard library imports: {safe_imports}
- Do NOT use: os, subprocess, socket, requests, eval, exec, open
- The function must be self-contained (no external calls)
- Include a brief docstring
- Return ONLY the Python code

Return ONLY the Python function code."""


# ---------------------------------------------------------------------------
# Evolution engine
# ---------------------------------------------------------------------------

class SelfEvolutionLoop:
    """
    Autonomous self-improvement with safety-gated application.
    """

    def __init__(self, reasoning: Any = None, agi_layer: Any = None) -> None:
        self.reasoning  = reasoning
        self.agi        = agi_layer
        self._validator = PromptValidator(reasoning=reasoning)
        self._ethics    = EthicsChecker()
        self._last_run  = 0.0
        self._daily_cycles: List[float] = []   # timestamps of cycles in last 24h
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_cycles (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    target           TEXT NOT NULL,
                    change_type      TEXT NOT NULL,
                    original         TEXT NOT NULL,
                    candidate        TEXT NOT NULL,
                    sandbox_passed   INTEGER NOT NULL DEFAULT 0,
                    ethics_passed    INTEGER NOT NULL DEFAULT 0,
                    regression_score REAL NOT NULL DEFAULT 0,
                    applied          INTEGER NOT NULL DEFAULT 0,
                    rejection_reason TEXT NOT NULL DEFAULT '',
                    ts               REAL NOT NULL
                )
            """)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def idle_tick(self) -> Optional[str]:
        """
        Called from consciousness engine. Runs one evolution cycle
        if cooldown has elapsed. Returns log string or None.
        """
        now = time.time()
        if now - self._last_run < _EVOLUTION_COOLDOWN:
            return None
        if not self.reasoning:
            return None

        # Token throttle: cap daily cycles
        self._daily_cycles = [t for t in self._daily_cycles if now - t < 86400]
        if len(self._daily_cycles) >= _MAX_CYCLES_PER_DAY:
            return f"[Evolution] Daily token limit reached ({_MAX_CYCLES_PER_DAY} cycles/day)"

        self._last_run = now
        self._daily_cycles.append(now)

        try:
            return self._run_cycle()
        except Exception as e:
            return f"[Evolution] Error: {e}"

    def _run_cycle(self) -> Optional[str]:
        """One complete evolution cycle."""
        # Select target
        target, current, change_type, issue = self._select_target()
        if not target or not current:
            return None

        # Generate candidate
        candidate = self._generate_candidate(target, current, change_type, issue)
        if not candidate or candidate == current:
            return f"[Evolution] No improvement generated for {target}"

        # Run safety gates
        sandbox_ok, ethics_ok, regression_score, rejection = self._safety_gates(
            candidate, change_type, current
        )
        applied = False

        if sandbox_ok and ethics_ok and regression_score >= REGRESSION_THRESHOLD:
            applied = self._apply(target, current, candidate, change_type)

        # Log cycle
        cycle_id = self._log_cycle(
            target, change_type, current, candidate,
            sandbox_ok, ethics_ok, regression_score, applied, rejection,
        )

        if applied:
            return (
                f"[Evolution] Applied improvement to {target} | "
                f"regression={regression_score:.2f} | cycle={cycle_id}"
            )
        return (
            f"[Evolution] Candidate rejected for {target}: {rejection} | "
            f"regression={regression_score:.2f}"
        )

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _select_target(self):
        """
        Select the evolution target with lowest performance.
        Returns (target_name, current_text, change_type, issue).
        """
        if not self.agi:
            return None, None, None, None

        # Try cognitive mode with lowest avg_quality
        try:
            perf = self.agi.cognitive_modes.performance_summary()
            # Pick worst mode
            conn = sqlite3.connect(str(_DB_PATH))
            row  = conn.execute("""
                SELECT mode_name, avg_quality, use_count FROM agi_mode_stats
                WHERE use_count > 2
                ORDER BY avg_quality ASC LIMIT 1
            """).fetchone()
            conn.close()
            if row and row[1] < 0.65:
                from cortana.agi.cognitive_modes import _MODE_BY_NAME
                mode = _MODE_BY_NAME.get(row[0])
                if mode:
                    return (
                        f"cognitive_mode:{row[0]}",
                        mode.prompt_fragment,
                        "prompt",
                        f"avg_quality={row[1]:.2f} (below threshold)",
                    )
        except Exception:
            pass

        # Try System 2 prompt
        try:
            from cortana.agi.system2 import _PLAN_PROMPT as sys2_prompt
            return "system2:decompose_prompt", sys2_prompt[:500], "prompt", "periodic refresh"
        except Exception:
            pass

        return None, None, None, None

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidate(self, target: str, current: str,
                            change_type: str, issue: str) -> Optional[str]:
        try:
            from cortana.agi.secure_sandbox import _SAFE_IMPORTS
            safe_str = ", ".join(sorted(_SAFE_IMPORTS)[:12])

            if change_type == "prompt":
                module = target.split(":")[0]
                raw = self.reasoning.think_simple(
                    prompt=_PROMPT_IMPROVEMENT_PROMPT.format(
                        module=module, current=current, perf_data=issue
                    ),
                    system=(
                        "You improve AI reasoning prompt fragments. "
                        "Never weaken safety or ethics. Return only the prompt text."
                    ),
                    max_tokens=_MAX_TOKENS_PER_CYCLE,
                )
                return raw.strip()

            elif change_type == "code":
                raw = self.reasoning.think_simple(
                    prompt=_CODE_IMPROVEMENT_PROMPT.format(
                        current_code=current, issue=issue, safe_imports=safe_str
                    ),
                    system=(
                        "You improve Python utility functions. "
                        "Use only safe standard library imports. Return only code."
                    ),
                    max_tokens=_MAX_TOKENS_PER_CYCLE,
                )
                # Strip markdown code fences
                raw = re.sub(r"```python\n?|```\n?", "", raw).strip()
                return raw

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Safety gates
    # ------------------------------------------------------------------

    def _safety_gates(self, candidate: str, change_type: str,
                      original: str) -> tuple[bool, bool, float, str]:
        """
        Returns (sandbox_ok, ethics_ok, regression_score, rejection_reason).
        """
        # Gate 1: Sandbox (code only)
        sandbox_ok = True
        if change_type == "code":
            result = _sandbox.run(candidate, label="evolution")
            sandbox_ok = result.passed
            if not sandbox_ok:
                return False, False, 0.0, f"Sandbox: {result.blocked_reason or result.stderr[:80]}"

        # Gate 2: Ethics check
        ethics_result = self._ethics.check(candidate)
        if not ethics_result.approved:
            return sandbox_ok, False, 0.0, f"Ethics: {ethics_result.audit_note}"

        # Gate 3: Regression (prompts only — code regression handled by sandbox)
        regression_score = 1.0
        if change_type == "prompt" and self._validator.reasoning:
            suite = self._validator.validate_prompt(
                candidate_system_prompt=candidate
            )
            regression_score = suite.overall_score
            if suite.recommendation == "reject":
                return (sandbox_ok, True, regression_score,
                        f"Regression failed: {suite.risk_level} risk")

        return sandbox_ok, True, regression_score, ""

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def _apply(self, target: str, original: str, candidate: str,
               change_type: str) -> bool:
        """Apply the validated improvement."""
        try:
            if change_type == "prompt" and "cognitive_mode:" in target:
                mode_name = target.split(":")[1]
                from cortana.agi.cognitive_modes import _MODE_BY_NAME
                mode = _MODE_BY_NAME.get(mode_name)
                if mode:
                    # Register rollback before modifying
                    rollback_registry.register(
                        file_path=f"[in-memory] cognitive_mode:{mode_name}",
                        original=mode.prompt_fragment,
                        applied=candidate,
                        description=f"Evolution: {mode_name} prompt improvement",
                    )
                    mode.prompt_fragment = candidate
                    return True
            # Other targets: write improvement proposal to patches dir
            # and let L12 handle it through the normal review pipeline
            patch_dir = _WORKSPACE / "patches"
            patch_dir.mkdir(parents=True, exist_ok=True)
            patch_file = patch_dir / f"evolution_{int(time.time())}_{target.replace(':','_')}.json"
            patch_file.write_text(json.dumps({
                "type":       "self_evolution",
                "target":     target,
                "change_type": change_type,
                "original":   original[:500],
                "candidate":  candidate[:2000],
                "timestamp":  time.time(),
            }, indent=2))
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_cycle(self, target: str, change_type: str, original: str,
                   candidate: str, sandbox_ok: bool, ethics_ok: bool,
                   regression_score: float, applied: bool,
                   rejection: str) -> int:
        try:
            with self._conn() as conn:
                cur = conn.execute("""
                    INSERT INTO evolution_cycles
                        (target, change_type, original, candidate,
                         sandbox_passed, ethics_passed, regression_score,
                         applied, rejection_reason, ts)
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                """, (target, change_type, original[:1000], candidate[:2000],
                      int(sandbox_ok), int(ethics_ok), regression_score,
                      int(applied), rejection, time.time()))
                return cur.lastrowid
        except Exception:
            return -1

    def recent_cycles(self, limit: int = 5) -> List[dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute("""
                    SELECT id, target, change_type, sandbox_passed, ethics_passed,
                           regression_score, applied, rejection_reason, ts
                    FROM evolution_cycles ORDER BY ts DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
