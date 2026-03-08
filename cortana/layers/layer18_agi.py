"""
Layer 18 — AGI Orchestration Layer.

Integrates all five AGI pillars into a single interface consumed by main.py:
  1. Cognitive Modes      — per-query reasoning mode selection
  2. Cross-Domain Engine  — structural analogy and domain transfer
  3. World Model          — persistent entity/belief/causal store
  4. Autonomy Engine      — persistent goal pursuit
  5. Ethics Checker       — constitutional safety filter

Public API:
  augment_prompt(query, intent, identity_prompt)  → augmented prompt (fast, no LLM)
  check_response(response, context)               → EthicsResult
  post_response(response, query, reasoning)       → None (background update)
  idle_tick()                                     → Optional[str] (log line)
  get_status()                                    → dict (for /api/agi/status)
  attach(reasoning, memory, browser)              → wire live references
"""
from __future__ import annotations

from typing import Any, Optional

from cortana.agi.cognitive_modes import CognitiveModeSelector
from cortana.agi.cross_domain import CrossDomainEngine
from cortana.agi.world_model import WorldModel
from cortana.agi.autonomy import AutonomyEngine, extract_goals_from_text
from cortana.agi.ethics import EthicsChecker, CONSTITUTIONAL_PROMPT, EthicsResult
from cortana.agi.system2 import System2Engine, System2Result


class AGILayer:
    """AGI orchestration — called once per pipeline turn."""

    def __init__(self, reasoning: Any = None, memory: Any = None,
                 browser: Any = None) -> None:
        self.cognitive_modes = CognitiveModeSelector()
        self.cross_domain    = CrossDomainEngine(reasoning=reasoning)
        self.world_model     = WorldModel()
        self.autonomy        = AutonomyEngine(reasoning=reasoning, memory=memory, browser=browser)
        self.ethics          = EthicsChecker()
        self.system2         = System2Engine(reasoning=reasoning)
        self._reasoning      = reasoning
        self._current_mode   = "default"

    def attach(self, reasoning: Any = None, memory: Any = None,
               browser: Any = None) -> None:
        """Wire live references (called after construction once layers are ready)."""
        if reasoning:
            self._reasoning              = reasoning
            self.cross_domain.reasoning  = reasoning
            self.autonomy.reasoning      = reasoning
            self.system2.reasoning       = reasoning
        if memory:
            self.autonomy.memory         = memory
        if browser:
            self.autonomy.browser        = browser

    # ------------------------------------------------------------------
    # Per-turn prompt augmentation  (<10ms, no LLM)
    # ------------------------------------------------------------------

    def augment_prompt(self, query: str, intent: str = "",
                       identity_prompt: str = "") -> str:
        """
        Return the identity_prompt enriched with AGI context sections.
        All steps are synchronous and LLM-free.
        """
        sections: list[str] = [identity_prompt.rstrip()]

        # 1. Constitutional ethics always present
        sections.append(CONSTITUTIONAL_PROMPT)

        # 2. Reasoning mode
        mode_result = self.cognitive_modes.select(query, intent)
        self._current_mode = mode_result.mode_name
        if mode_result.prompt_fragment:
            sections.append(
                f"--- Reasoning Mode: {mode_result.mode_name} "
                f"(confidence {mode_result.confidence:.2f}) ---\n"
                + mode_result.prompt_fragment
            )

        # 3. Cross-domain context
        cd_ctx = self.cross_domain.build_transfer_prompt(query)
        if cd_ctx:
            sections.append(cd_ctx)

        # 4. World model context
        wm_ctx = self.world_model.query_context(query)
        if wm_ctx:
            sections.append(wm_ctx)

        # 5. Active goals context
        goals_ctx = self.autonomy.goals_context_prompt(limit=3)
        if goals_ctx:
            sections.append(goals_ctx)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # System 2 deliberate reasoning
    # ------------------------------------------------------------------

    def should_use_system2(self, query: str, intent: str, complexity: float) -> bool:
        return self.system2.should_activate(query, intent, complexity)

    def run_system2(self, query: str, context: str = "") -> System2Result:
        """Full 4-stage System 2 pipeline. May take several seconds."""
        return self.system2.reason(query, context)

    # ------------------------------------------------------------------
    # Ethics check on outgoing response
    # ------------------------------------------------------------------

    def check_response(self, response: str, context: str = "") -> EthicsResult:
        result = self.ethics.check(response, context)
        if not result.approved:
            self.cognitive_modes.record_outcome(self._current_mode, 0.0)
        return result

    # ------------------------------------------------------------------
    # Post-response background update
    # ------------------------------------------------------------------

    def post_response(self, response: str, query: str,
                      reasoning: Any = None) -> None:
        """
        Update world model and extract goals from this exchange.
        Called in background thread — may make one LLM call.
        """
        r = reasoning or self._reasoning
        try:
            # Fast regex extraction always
            self.world_model.regex_extract_and_store(response, source="cortana_response")
            self.world_model.regex_extract_and_store(query, source="user_query")
            # LLM extraction if reasoning available (richer but slower)
            if r:
                self.world_model.llm_extract_and_store(response, r, source="cortana_llm")
        except Exception:
            pass

        # Extract user goals from query
        try:
            for goal_desc in extract_goals_from_text(query):
                self.autonomy.add_goal(goal_desc, domain="user_stated", priority=0.75)
        except Exception:
            pass

        # Record mode outcome
        try:
            self.cognitive_modes.record_outcome(self._current_mode, 0.7)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Idle-time goal pursuit (called by consciousness engine)
    # ------------------------------------------------------------------

    def idle_tick(self) -> Optional[str]:
        """Called during consciousness idle ticks. Never raises."""
        try:
            return self.autonomy.idle_tick()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Status (for API endpoint)
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        wm      = self.world_model.stats()
        goals   = self.autonomy.get_active_goals(limit=5)
        return {
            "world_model":      wm,
            "active_goals":     [
                {
                    "id":          g.id,
                    "description": g.description,
                    "domain":      g.domain,
                    "priority":    g.priority,
                    "status":      g.status,
                    "progress":    round(g.progress, 3),
                }
                for g in goals
            ],
            "mode_performance": self.cognitive_modes.performance_summary(),
            "principles":       self.ethics.principles(),
        }
