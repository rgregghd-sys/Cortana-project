"""
Layer 18 — AGI Orchestration Layer.

Integrates all AGI pillars and modules:
  1.  Cognitive Modes       — 8 reasoning modes, auto-selected
  2.  Cross-Domain Engine   — structural analogy across 10 domains
  3.  World Model           — persistent entity/belief/causal store
  4.  Autonomy Engine       — persistent goal pursuit
  5.  Ethics Checker        — constitutional safety filter
  6.  System 2 Reasoning    — 4-stage deliberate reasoning pipeline
  7.  Zero-Shot Scaffold    — first-principles reasoning for novel problems
  8.  Abstract Reasoning    — logic, fallacy, and sequence detection
  9.  Common Sense          — physics, social, temporal grounding
  10. NLP Mastery           — sarcasm, subtext, emotional undertone
  11. Long-Horizon Planner  — multi-week milestone-tracked plans
  12. Self-Improvement      — recursive analysis + active learning
  13. Theory of Mind        — per-user belief/goal/expertise modelling
  14. Reasoned Refusal      — principled, explained ethical refusals
  15. Validation Suite      — safe sandbox + prompt regression testing

Public API (consumed by main.py):
  augment_prompt(query, intent, identity_prompt, session_id, memory_hits)
  check_response(response, context, query)      → EthicsResult (with reasoned refusal)
  post_response(response, query, reasoning)     → None (background)
  idle_tick()                                   → Optional[str]
  get_status()                                  → dict
  attach(reasoning, memory, browser)
  update_user_model(session_id, query, history) → UserModel
"""
from __future__ import annotations

from typing import Any, List, Optional

from cortana.agi.cognitive_modes  import CognitiveModeSelector
from cortana.agi.cross_domain     import CrossDomainEngine
from cortana.agi.world_model      import WorldModel
from cortana.agi.autonomy         import AutonomyEngine, extract_goals_from_text
from cortana.agi.ethics           import EthicsChecker, CONSTITUTIONAL_PROMPT, EthicsResult
from cortana.agi.system2          import System2Engine, System2Result
from cortana.agi.zero_shot        import should_activate as zs_should_activate, build_scaffold
from cortana.agi.abstract_reasoning import should_activate as ar_should_activate, build_abstract_prompt
from cortana.agi.common_sense     import should_activate as cs_should_activate, build_prompt as cs_build_prompt
from cortana.agi.nlp_mastery      import should_activate as nlp_should_activate, build_prompt as nlp_build_prompt
from cortana.agi.long_horizon     import LongHorizonPlanner, should_activate as lh_should_activate
from cortana.agi.self_improvement import SelfImprovementEngine
from cortana.agi.theory_of_mind   import TheoryOfMindEngine, UserModel
from cortana.agi.reasoned_refusal import build_refusal, is_ambiguous, build_clarifying_question
from cortana.agi.validation_suite import PromptValidator, rollback_registry
from cortana.agi.secure_sandbox   import sandbox as _sandbox
from cortana.agi.self_evolution   import SelfEvolutionLoop


class AGILayer:
    """Full AGI orchestration — one instance per CortanaSystem."""

    def __init__(self, reasoning: Any = None, memory: Any = None,
                 browser: Any = None) -> None:
        self.cognitive_modes  = CognitiveModeSelector()
        self.cross_domain     = CrossDomainEngine(reasoning=reasoning)
        self.world_model      = WorldModel()
        self.autonomy         = AutonomyEngine(reasoning=reasoning, memory=memory, browser=browser)
        self.ethics           = EthicsChecker()
        self.system2          = System2Engine(reasoning=reasoning)
        self.long_horizon     = LongHorizonPlanner(reasoning=reasoning)
        self.self_improvement = SelfImprovementEngine(autonomy=self.autonomy)
        self.theory_of_mind   = TheoryOfMindEngine()
        self.validator        = PromptValidator(reasoning=reasoning)
        self.rollback         = rollback_registry
        self.sandbox          = _sandbox
        self.evolution        = SelfEvolutionLoop(reasoning=reasoning, agi_layer=self)
        self._reasoning       = reasoning
        self._current_mode    = "default"
        self._current_user_model: Optional[UserModel] = None

    def attach(self, reasoning: Any = None, memory: Any = None,
               browser: Any = None) -> None:
        if reasoning:
            self._reasoning                  = reasoning
            self.cross_domain.reasoning      = reasoning
            self.autonomy.reasoning          = reasoning
            self.system2.reasoning           = reasoning
            self.long_horizon.reasoning      = reasoning
            self.self_improvement.reasoning  = reasoning
            self.validator.reasoning         = reasoning
            self.evolution.reasoning         = reasoning
        if memory:
            self.autonomy.memory             = memory
        if browser:
            self.autonomy.browser            = browser

    # ------------------------------------------------------------------
    # User model update (call once per turn, before augment_prompt)
    # ------------------------------------------------------------------

    def update_user_model(self, session_id: str, query: str,
                          history: Optional[List[str]] = None) -> UserModel:
        model = self.theory_of_mind.update(session_id, query, history)
        self._current_user_model = model
        # Record query for self-improvement gap analysis
        self.self_improvement.record_query(query)
        return model

    # ------------------------------------------------------------------
    # Per-turn prompt augmentation (<15ms, no LLM except plan creation)
    # ------------------------------------------------------------------

    def augment_prompt(self, query: str, intent: str = "",
                       identity_prompt: str = "",
                       session_id: str = "",
                       memory_hit_count: int = 0,
                       complexity: float = 0.0) -> str:
        sections: list[str] = [identity_prompt.rstrip()]

        # 1. Constitutional ethics — always
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

        # 3. Zero-shot scaffold
        if zs_should_activate(query, memory_hit_count, complexity):
            sections.append(build_scaffold(query))

        # 4. Abstract reasoning
        if ar_should_activate(query):
            sections.append(build_abstract_prompt(query))

        # 5. Common sense grounding
        if cs_should_activate(query):
            cs = cs_build_prompt(query)
            if cs:
                sections.append(cs)

        # 6. NLP mastery (sarcasm, subtext, emotion)
        if nlp_should_activate(query):
            nlp = nlp_build_prompt(query)
            if nlp:
                sections.append(nlp)

        # 7. Theory of Mind — user model
        if self._current_user_model or session_id:
            model = self._current_user_model or self.theory_of_mind.load(session_id)
            tom_ctx = self.theory_of_mind.build_prompt(model)
            if tom_ctx:
                sections.append(tom_ctx)

        # 8. Cross-domain context
        cd_ctx = self.cross_domain.build_transfer_prompt(query)
        if cd_ctx:
            sections.append(cd_ctx)

        # 9. World model context
        wm_ctx = self.world_model.query_context(query)
        if wm_ctx:
            sections.append(wm_ctx)

        # 10. Active goals
        goals_ctx = self.autonomy.goals_context_prompt(limit=3)
        if goals_ctx:
            sections.append(goals_ctx)

        # 11. Active plans (long-horizon)
        plans_ctx = self.long_horizon.get_planning_context()
        if plans_ctx:
            sections.append(plans_ctx)

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # System 2 deliberate reasoning
    # ------------------------------------------------------------------

    def should_use_system2(self, query: str, intent: str, complexity: float) -> bool:
        return self.system2.should_activate(query, intent, complexity)

    def run_system2(self, query: str, context: str = "") -> System2Result:
        return self.system2.reason(query, context)

    # ------------------------------------------------------------------
    # Long-horizon planning (triggered from post_response background)
    # ------------------------------------------------------------------

    def maybe_create_plan(self, query: str) -> None:
        """Detect planning requests and create a plan asynchronously."""
        if lh_should_activate(query) and self._reasoning:
            try:
                self.long_horizon.create_plan(query)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Ethics check + reasoned refusal
    # ------------------------------------------------------------------

    def check_response(self, response: str, context: str = "",
                       query: str = "") -> EthicsResult:
        """
        Run ethics check. If violations found, replace with a principled,
        explained refusal (not a blunt block).
        Handle ambiguous requests with a clarifying question.
        """
        # Ambiguity check first
        if query and is_ambiguous(query):
            clarifier = build_clarifying_question(query)
            # Don't block — inject clarifying question into response prefix
            if clarifier and "?" not in response[:100]:
                pass  # handled via prompt; don't override here

        result = self.ethics.check(response, context)

        if not result.approved and result.violations:
            # Replace generic refusal with principled, explained one
            refusal = build_refusal(
                violations=result.violations,
                original_query=query or context,
                is_partial=False,
            )
            result.modified_response = refusal.text
            self.cognitive_modes.record_outcome(self._current_mode, 0.0)

        return result

    # ------------------------------------------------------------------
    # Post-response background update
    # ------------------------------------------------------------------

    def post_response(self, response: str, query: str,
                      reasoning: Any = None) -> None:
        r = reasoning or self._reasoning
        try:
            self.world_model.regex_extract_and_store(response, source="cortana_response")
            self.world_model.regex_extract_and_store(query, source="user_query")
            if r:
                self.world_model.llm_extract_and_store(response, r, source="cortana_llm")
        except Exception:
            pass
        try:
            for goal_desc in extract_goals_from_text(query):
                self.autonomy.add_goal(goal_desc, domain="user_stated", priority=0.75)
        except Exception:
            pass
        try:
            self.maybe_create_plan(query)
        except Exception:
            pass
        try:
            self.cognitive_modes.record_outcome(self._current_mode, 0.7)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Idle-time work (called by consciousness engine)
    # ------------------------------------------------------------------

    def idle_tick(self) -> Optional[str]:
        logs: list[str] = []
        try:
            g = self.autonomy.idle_tick()
            if g:
                logs.append(g)
        except Exception:
            pass
        try:
            s = self.self_improvement.idle_tick()
            if s:
                logs.append(s)
        except Exception:
            pass
        try:
            e = self.evolution.idle_tick()
            if e:
                logs.append(e)
        except Exception:
            pass
        return " | ".join(logs) if logs else None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_prompt(self, candidate_prompt: str) -> dict:
        suite = self.validator.validate_prompt(candidate_prompt)
        return {
            "all_passed":     suite.all_passed,
            "overall_score":  suite.overall_score,
            "risk_level":     suite.risk_level,
            "recommendation": suite.recommendation,
            "results":        [{"test_id": r.test_id, "passed": r.passed,
                                "score": r.score, "details": r.details}
                               for r in suite.results],
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        wm    = self.world_model.stats()
        goals = self.autonomy.get_active_goals(limit=5)
        plans = self.long_horizon.all_plans_status()
        return {
            "world_model":      wm,
            "active_goals":     [
                {"id": g.id, "description": g.description, "domain": g.domain,
                 "priority": g.priority, "status": g.status,
                 "progress": round(g.progress, 3)}
                for g in goals
            ],
            "active_plans":     plans,
            "mode_performance": self.cognitive_modes.performance_summary(),
            "principles":       self.ethics.principles(),
            "rollback_recent":  self.rollback.recent(limit=3),
            "evolution_recent": self.evolution.recent_cycles(limit=3),
            "sandbox_recent":   self.sandbox.recent_sessions(limit=3),
        }
