"""
Cortana — 12-Layer Agentic AI System
Entry point: wires all layers and runs the Rich terminal interface.

Usage:
    python -m cortana.main
    # or from project root:
    python cortana/main.py
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import re
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

# Background thread pool for L9+L10 post-processing (runs after response is sent)
_POST_PROCESS_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="cortana-post"
)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortana import config
from cortana.models.schemas import (
    ConversationTurn,
    CortanaState,
    PerceivedInput,
    UserInput,
)


# ---------------------------------------------------------------------------
# Background task detection patterns
# ---------------------------------------------------------------------------

_BG_TRIGGER_RE = re.compile(
    r"\b(?:in the background|background task|while (?:we|i) (?:talk|chat|continue)|"
    r"keep (?:thinking|working|processing) on|"
    r"work on (?:this|that) (?:in the background|separately|for me)|"
    r"run (?:this|that) in the background|"
    r"let me know when (?:you(?:'re)? done|you have (?:an answer|a result))|"
    r"think on (?:this|that)|process (?:this|that) (?:in the background|separately))\b",
    re.IGNORECASE,
)

_BG_STATUS_RE = re.compile(
    r"\b(?:how(?:'s| is) it going|"
    r"how(?:'s| is) (?:that|the) (?:analysis|thinking|processing|task|work)|"
    r"(?:any|got any) (?:updates?|progress|results?)|"
    r"(?:still|are you still) (?:thinking|working|processing)|"
    r"(?:status|update me|progress report)|"
    r"what (?:have you|did you) (?:found?|figured? out|come up with|determined|concluded)|"
    r"are you done(?: yet)?|(?:finished|completed?) (?:yet|thinking)|"
    r"what(?:'s| is) the (?:result|answer|conclusion|update)|"
    r"background (?:task|update|result)|"
    r"(?:give me|show me) (?:an? )?update)\b",
    re.IGNORECASE,
)
from cortana.layers.layer0_supervisor import SupervisorLayer
from cortana.layers.layer1_identity import CortanaIdentity
from cortana.layers.layer2_memory import CortanaMemory
from cortana.layers.layer3_perception import PerceptionLayer
from cortana.layers.layer4_reasoning import ReasoningLayer
from cortana.layers.layer5_planning import PlanningLayer
from cortana.layers.layer6_orchestration import OrchestrationLayer
from cortana.layers.layer9_reflection import ReflectionLayer
from cortana.layers.layer10_security import SecurityLayer
from cortana.layers.layer11_patcher import PatchWriterLayer
from cortana.layers.layer12_notifier import PatchImplementerLayer
from cortana.layers.layer16_distiller import KnowledgeDistiller
from cortana.layers.layer17_cognition import CognitiveLayer
from cortana.consciousness.self_model import PersistentSelfModel
from cortana.consciousness.engine import ConsciousnessEngine
from cortana.background.thinker import BackgroundThinker
from cortana.ui import terminal as ui


def _fast_emotion(response: str, intent: str) -> str:
    """
    Heuristic emotion based on intent + response keywords.
    Avoids an LLM call — L9's richer version runs in background.
    """
    lower = response.lower()
    if any(w in lower for w in ("sorry", "unfortunately", "unable to", "cannot", "can't")):
        return "sad"
    if intent in ("code", "analysis", "research"):
        return "think"
    if intent == "creative":
        return "smile"
    if intent == "conversational":
        if any(w in lower for w in ("!", "great", "absolutely", "of course", "happy to")):
            return "smile"
        return "idle"
    return "idle"


class CortanaSystem:
    """
    Master controller: instantiates all 12 layers and runs the conversation loop.
    """

    def __init__(self) -> None:
        ui.print_system("Initializing Cortana AI subsystems...", level="info")

        # Layer 0 — Supervisor (error recovery)
        self.supervisor = SupervisorLayer()
        ui.print_system("Layer 0 [Supervisor] online", level="ok")

        # Layer 1 — Identity
        self.identity = CortanaIdentity()
        ui.print_system("Layer 1 [Identity] online", level="ok")

        # Layer 2 — Memory
        self.memory = CortanaMemory()
        ui.print_system("Layer 2 [Memory] online — ChromaDB + SQLite active", level="ok")

        # Layer 3 — Perception
        self.perception = PerceptionLayer()
        ui.print_system("Layer 3 [Perception] online", level="ok")

        # Layer 4 — Reasoning (multi-provider router)
        try:
            self.reasoning = ReasoningLayer()
            providers_ready = [p.name for p in self.reasoning.router._providers]
            ui.print_system(
                f"Layer 4 [Reasoning] online — providers: {' → '.join(providers_ready)}",
                level="ok",
            )
        except RuntimeError as e:
            ui.print_system(str(e), level="error")
            sys.exit(1)

        # Layer 5 — Planning
        self.planning = PlanningLayer()
        ui.print_system("Layer 5 [Planning] online", level="ok")

        # Layer 6 — Orchestration
        self.orchestration = OrchestrationLayer(
            on_task_start=self._on_task_start,
            on_task_done=self._on_task_done,
        )
        ui.print_system("Layer 6 [Orchestration] online", level="ok")

        # Layer 9 — Reflection
        self.reflection = ReflectionLayer()
        ui.print_system("Layer 9 [Reflection] online", level="ok")

        # Layer 10 — Security (Red vs Blue)
        self.security = SecurityLayer()
        ui.print_system("Layer 10 [Security] online — Red/Blue agents ready", level="ok")

        # Layer 11 — Patch Writer
        self.patcher = PatchWriterLayer()
        ui.print_system("Layer 11 [Patch Writer] online", level="ok")

        # Layer 12 — Patch Implementer
        self.notifier = PatchImplementerLayer()
        ui.print_system("Layer 12 [Patch Implementer] online", level="ok")

        # Layer 16 — Knowledge Distiller
        self.distiller = KnowledgeDistiller()
        ui.print_system("Layer 16 [Knowledge Distiller] online — training corpus ready", level="ok")

        # Layer 17 — Cognitive Architecture (Working Memory + Attention + Goal Stack + GNN/RNN)
        self.cognition = CognitiveLayer()
        ui.print_system("Layer 17 [Cognition] online — Working Memory / Attention / Goal Stack / Neural active", level="ok")

        # Consciousness — persistent self-model + always-awake inner loop
        self.self_model   = PersistentSelfModel()
        self.consciousness = ConsciousnessEngine(
            self_model=self.self_model,
            cognitive_layer=self.cognition,
        )
        if config.CONSCIOUSNESS_ENABLED:
            self.consciousness.start()
            ui.print_system(
                f"Consciousness Engine online — Cortana has been awake "
                f"{self.self_model.get_uptime_seconds()/3600:.2f}h | "
                f"Thoughts: {self.self_model.model.total_thoughts}",
                level="ok",
            )
        else:
            ui.print_system("Consciousness Engine disabled (CONSCIOUSNESS_ENABLED=false)", level="warn")

        # Background Thinker — persistent async reasoning engine
        self.thinker = BackgroundThinker(db_path=config.SQLITE_PATH)
        self.thinker.set_reasoning(self.reasoning)
        # Give consciousness engine access to reasoning for LLM-based thoughts
        self.consciousness.attach_reasoning(self.reasoning)
        ui.print_system("Background Thinker online — iterative reasoning ready", level="ok")

        ui.print_system("Layers 7 [Sub-Agents] + 8 [Tools] load on demand", level="info")
        ui.print_divider()

        # State
        self.state = CortanaState()
        self.conversation: List[ConversationTurn] = []
        self._active_agents: List[str] = []

    # ------------------------------------------------------------------
    # Callbacks for UI feedback
    # ------------------------------------------------------------------
    def _on_task_start(self, task) -> None:
        if task.agent_type != "direct":
            self._active_agents.append(task.agent_type)
            ui.print_system(f"[{task.agent_type.upper()}] dispatched: {task.description[:60]}", level="info")

    def _on_task_done(self, task) -> None:
        if task.agent_type in self._active_agents:
            self._active_agents.remove(task.agent_type)
        status_str = "ok" if task.status == "done" else "warn"
        ui.print_system(f"[{task.agent_type.upper()}] {task.status}", level=status_str)

    # ------------------------------------------------------------------
    # Redundancy wrapper — wraps every layer call with L0 supervision
    # ------------------------------------------------------------------
    @staticmethod
    def _is_quota_error(error: str) -> bool:
        """Return True if the error is an API quota/rate-limit hit."""
        lowered = error.lower()
        return "429" in error or "quota" in lowered or "rate limit" in lowered or "resource_exhausted" in lowered

    def _handle_error(self, layer_id: int, layer_name: str, error: str, context: dict) -> str:
        """
        Route errors to L0 Supervisor unless it's a quota error
        (calling Gemini to analyse a Gemini quota error makes things worse).
        Returns the lesson string for UI display.
        """
        if self._is_quota_error(error):
            return f"API quota limit hit. Wait a moment before retrying."
        try:
            feedback = self.supervisor.review_error(
                layer_id=layer_id,
                layer_name=layer_name,
                error=error,
                context=context,
            )
            return feedback.lesson
        except Exception:
            return f"Layer {layer_id} error: {error[:80]}"

    def _safe_call(
        self,
        layer_id: int,
        layer_name: str,
        fn: Callable,
        fallback: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Call fn(*args, **kwargs). On any exception or None/empty-string result,
        route to L0 Supervisor (unless quota error) and return fallback.
        Note: empty list [] is a valid result (e.g. memory recall with no entries).
        """
        try:
            result = fn(*args, **kwargs)
            if result is None or result == "":
                raise ValueError("Layer returned empty output")
            return result
        except Exception as e:
            lesson = self._handle_error(layer_id, layer_name, str(e), kwargs)
            ui.print_system(f"[L{layer_id}] Error — L0 lesson: {lesson[:120]}", level="warn")
            return fallback

    async def _safe_call_async(
        self,
        layer_id: int,
        layer_name: str,
        fn: Callable,
        fallback: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Async variant of _safe_call for awaitable layer functions."""
        try:
            result = await fn(*args, **kwargs)
            if result is None or result == "":
                raise ValueError("Layer returned empty output")
            return result
        except Exception as e:
            lesson = self._handle_error(layer_id, layer_name, str(e), kwargs)
            ui.print_system(f"[L{layer_id}] Error — L0 lesson: {lesson[:120]}", level="warn")
            return fallback

    # ------------------------------------------------------------------
    # Background post-processing (L10 Security + L9 Reflection)
    # ------------------------------------------------------------------
    _SKIP_SECURITY_FOR = frozenset({"conversational", "simple"})

    def _background_security_reflection(
        self,
        response: str,
        perceived: Any,
        state: Any,
        tasks: list,
        raw_input: str,
        memories: list,
        conversation: list,
    ) -> None:
        """
        Runs L10 Security + L9 Reflection in a background thread after the
        response has already been sent to the user.
        """
        try:
            # L10: skip entirely for safe, non-complex intents
            if perceived.intent not in self._SKIP_SECURITY_FOR and len(raw_input.strip()) >= 4:
                security_result = self._safe_call(
                    10, "Security",
                    self.security.evaluate,
                    None,
                    response=response,
                    user_input=raw_input,
                    conversation_history=conversation,
                    memory_entries=memories,
                )
                if security_result and security_result.red_wins:
                    ui.print_system(
                        f"[L10-BG] Security alert — Red wins! "
                        f"Vulnerabilities: {[v.type for v in security_result.vulnerabilities]} "
                        f"| Defense score: {security_result.defense_score:.2f}",
                        level="warn",
                    )
                    patch_result = self._safe_call(
                        11, "Patch Writer",
                        self.patcher.write_patches,
                        None,
                        security_result.vulnerabilities,
                    )
                    if patch_result and patch_result.patch_files:
                        ui.print_system(
                            f"[L11-BG] {len(patch_result.patch_files)} patch(es) written",
                            level="info",
                        )
                        self._safe_call(
                            12, "Patch Implementer",
                            self.notifier.process_patches,
                            [],
                            patch_result,
                        )
                elif security_result:
                    ui.print_system(
                        f"[L10-BG] Clear — defense score: {security_result.defense_score:.2f}",
                        level="ok",
                    )

            # L9: always run for memory storage + concept extraction
            self._safe_call(
                9, "Reflection",
                self.reflection.reflect,
                None,
                response=response,
                perceived=perceived,
                state=state,
                tasks=tasks if tasks else None,
                user_input=raw_input,
            )
        except Exception as exc:
            ui.print_system(f"[BG-POST] Error: {exc}", level="warn")

    # ------------------------------------------------------------------
    # Core processing pipeline
    # ------------------------------------------------------------------

    _L4_FALLBACK = None  # sentinel; real error message built at runtime

    async def process_session(
        self,
        raw_input: str,
        state: CortanaState,
        conversation: List[ConversationTurn],
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, CortanaState, List[ConversationTurn], str]:
        """
        Full 13-layer pipeline for one user turn.
        Takes explicit state + conversation so multiple sessions can share
        the same CortanaSystem infrastructure without clobbering each other.
        Returns (final_response, new_state, new_conversation, emotion).
        """
        # --- Background task detection (runs before any layer) ---

        # Trigger: explicit request to work in the background
        if _BG_TRIGGER_RE.search(raw_input):
            clean_query = _BG_TRIGGER_RE.sub("", raw_input).strip(" ,.?!")
            task_id, task_name = self.thinker.start_task(clean_query or raw_input)
            ack = (
                f"On it. I've started a background reasoning thread for '{task_name}' "
                f"(task {task_id}). I'll keep working through it while we talk — "
                f"ask 'any updates?' whenever you want a progress report, "
                f"or I'll let you know when I land on a conclusion."
            )
            new_state = CortanaState(interaction_count=state.interaction_count + 1)
            new_conv = list(conversation) + [
                ConversationTurn(role="user", content=raw_input),
                ConversationTurn(role="assistant", content=ack),
            ]
            return ack, new_state, new_conv, "think"

        # Status query: asking about a running or completed background task
        if _BG_STATUS_RE.search(raw_input) and self.thinker.has_any_tasks():
            task = self.thinker.get_latest_task()
            if task:
                status_response = self.thinker.format_status(task)
                new_state = CortanaState(interaction_count=state.interaction_count + 1)
                new_conv = list(conversation) + [
                    ConversationTurn(role="user", content=raw_input),
                    ConversationTurn(role="assistant", content=status_response),
                ]
                return status_response, new_state, new_conv, "idle"

        # --- Layer 3: Perception ---
        perceived: PerceivedInput = self._safe_call(
            3, "Perception",
            self.perception.perceive,
            PerceivedInput(content=raw_input, intent="simple", complexity=0.1),
            UserInput(raw=raw_input),
        )

        # --- Self-design fast path ---
        if perceived.intent == "self_design":
            try:
                from cortana.layers import layer8_tools as _tools
                # Use reasoning to extract appearance description from user input
                desc_prompt = (
                    f"Extract a concise 3D appearance description from this request: "
                    f"'{raw_input}'. Output ONLY keywords like: "
                    f"'<tone> skin, <hair>, <build>' (e.g. 'medium skin, long hair, slim'). "
                    f"If the user wants you to decide autonomously, choose what looks most natural."
                )
                try:
                    appearance_desc = self.reasoning.think_simple(
                        prompt=desc_prompt,
                        system="You extract concise appearance parameters. Reply with only a short comma-separated list.",
                    ).strip()
                except Exception:
                    appearance_desc = "medium skin, short_crop hair, slim"

                build_result = await _tools.design_self(description=appearance_desc)
                ack = (
                    f"I've redesigned my 3D appearance.\n\n"
                    f"Parameters used: {appearance_desc}.\n\n"
                    f"Build result: {build_result}\n\n"
                    f"The model will update in your browser automatically."
                )
            except Exception as exc:
                ack = f"Self-design failed: {exc}"

            new_state = CortanaState(interaction_count=state.interaction_count + 1)
            new_conv = list(conversation) + [
                ConversationTurn(role="user", content=raw_input),
                ConversationTurn(role="assistant", content=ack),
            ]
            self.memory.save_turn(
                new_conv[-2].content[:20], "user", raw_input
            ) if hasattr(self.memory, "save_turn") else None
            return ack, new_state, new_conv, "smile"

        # --- DevAI fast path ---
        if perceived.intent == "devai":
            try:
                from cortana.layers import layer8_tools as _tools
                import re as _re

                low = raw_input.lower()

                # Approve / reject by ID
                approve_m = _re.search(r'\bapprove\s+#?(\d+)', low)
                reject_m  = _re.search(r'\breject\s+#?(\d+)', low)
                if approve_m:
                    ack = _tools.devai_approve(int(approve_m.group(1)))
                elif reject_m:
                    ack = _tools.devai_reject(int(reject_m.group(1)))
                elif any(w in low for w in ("scan", "review", "check", "improve", "analyse", "analyze")):
                    ack = await _tools.devai_scan()
                elif "history" in low:
                    ack = _tools.devai_history()
                else:
                    ack = _tools.devai_status()

            except Exception as exc:
                ack = f"DevAI tool error: {exc}"

            new_state = CortanaState(interaction_count=state.interaction_count + 1)
            new_conv = list(conversation) + [
                ConversationTurn(role="user", content=raw_input),
                ConversationTurn(role="assistant", content=ack),
            ]
            return ack, new_state, new_conv, "think"

        # --- Layer 2: Memory recall ---
        memories: List[str] = self._safe_call(
            2, "Memory",
            self.memory.recall,
            [],
            perceived.content,
        )

        # --- Layer 1: Identity ---
        identity_prompt: str = self._safe_call(
            1, "Identity",
            self.identity.get_personality_prompt,
            CortanaIdentity.SYSTEM_PROMPT,
            state,
        )

        # --- Layer 17: Cognitive Architecture ---
        # Fetch typed concept graph from Tier 3 memory for GNN + working memory
        try:
            _concepts   = self.memory.get_concept_nodes(limit=30)
            _relations  = self.memory.get_relation_edges(limit=60)
            _ep_strings = self.memory.get_recent_episode_strings(limit=20)

            cognitive_state = self.cognition.process(
                perceived=perceived,
                memories=memories,
                concepts=_concepts,
                relations=_relations,
                conversation=conversation,
                state=state,
            )

            # Neural-augment memory recall order
            if memories:
                memories = self.cognition.neural_augmented_recall(
                    query=perceived.content,
                    episodes=_ep_strings,
                    concepts=_concepts,
                    relations=_relations,
                    base_results=memories,
                )

            # Inject cognitive context into identity prompt
            if cognitive_state.cognitive_context:
                identity_prompt = (
                    identity_prompt
                    + "\n\n--- Cognitive State ---\n"
                    + cognitive_state.cognitive_context
                )
        except Exception as _cog_err:
            ui.print_system(f"[L17] Cognitive layer error (non-fatal): {_cog_err}", level="warn")

        # --- Decide: simple path or full agentic path ---
        tasks = []
        sub_agent_context = ""

        if self.planning.needs_planning(perceived):
            ui.print_system(
                f"Complex request (complexity={perceived.complexity:.2f}, intent={perceived.intent}) "
                "— invoking planning layer...",
                level="info",
            )

            # Layer 5 — Planning
            recent_ctx = " | ".join(
                t.content[:80] for t in conversation[-4:] if t.role == "user"
            )
            plan = self._safe_call(
                5, "Planning",
                self.planning.plan,
                None,
                perceived, state, recent_ctx or None,
            )

            if plan:
                if plan.reasoning:
                    ui.print_system(f"Plan: {plan.reasoning}", level="info")

                ui.print_agent_panel(plan.tasks)
                tasks = await self._safe_call_async(
                    6, "Orchestration",
                    self.orchestration.execute_plan,
                    [],
                    plan,
                )
                if tasks:
                    ui.print_agent_panel(tasks)
                    sub_agent_context = self._safe_call(
                        6, "Orchestration.merge",
                        self.orchestration.merge_results,
                        "",
                        tasks, raw_input,
                    )

            if sub_agent_context:
                enhanced = PerceivedInput(
                    content=(
                        f"{perceived.content}\n\n"
                        f"## Research Results\n{sub_agent_context}\n\n"
                        f"Using the above research, provide a comprehensive response."
                    ),
                    intent=perceived.intent,
                    complexity=perceived.complexity,
                    keywords=perceived.keywords,
                )
            else:
                enhanced = perceived
        else:
            # Auto web search: if Cortana likely needs current/factual info, search proactively
            _AUTO_SEARCH_INTENTS = ("research", "analysis", "code")
            if perceived.intent in _AUTO_SEARCH_INTENTS:
                try:
                    from cortana.layers import layer8_tools as _tools
                    web_result = await _tools.web_search(raw_input[:200], max_results=4)
                    if web_result and not web_result.startswith(
                        ("No results", "Search error", "System in", "Web scraping")
                    ):
                        enhanced = PerceivedInput(
                            content=(
                                f"{perceived.content}"
                                f"\n\n## Live Web Context\n{web_result[:2000]}"
                                f"\n\nUse the above if relevant; ignore if not."
                            ),
                            intent=perceived.intent,
                            complexity=perceived.complexity,
                            keywords=perceived.keywords,
                        )
                    else:
                        enhanced = perceived
                except Exception:
                    enhanced = perceived
            else:
                enhanced = perceived

        # --- Layer 4: Reasoning ---
        # Terminal mode uses streaming display; web mode passes its own on_chunk
        if on_chunk is None:
            display = ui.StreamingDisplay(state)
            display.start()
            chunk_fn = display.on_chunk
        else:
            display = None
            chunk_fn = on_chunk

        l4_failed = False
        response: str = ""
        try:
            response = self.reasoning.think(
                perceived=enhanced,
                memories=memories,
                identity_prompt=identity_prompt,
                conversation_history=conversation,
                state=state,
                on_chunk=chunk_fn,
            )
            if not response:
                raise ValueError("Empty response from reasoning layer")
        except Exception as _l4_err:
            self._handle_error(4, "Reasoning", str(_l4_err), {})
            # Retry with a stripped-down prompt (no history, no memories)
            try:
                ui.print_system("[L4] Primary call failed — retrying with minimal prompt", level="warn")
                response = self.reasoning.think_simple(
                    prompt=raw_input,
                    system=identity_prompt,
                )
            except Exception as _retry_err:
                l4_failed = True
                err_hint = str(_retry_err)
                if "exhausted" in err_hint.lower() or "429" in err_hint or "quota" in err_hint.lower():
                    response = (
                        "All my inference providers are rate-limited right now. "
                        "Give it a minute and try again."
                    )
                else:
                    response = (
                        f"My reasoning pipeline hit an error: {err_hint[:120]}. "
                        "Try rephrasing or check that at least one API key is set in .env."
                    )

        if display:
            display.stop()

        # --- Core Laws enforcement (synchronous, every response) ---
        # Runs inline before the response reaches the user — not fire-and-forget.
        if response and not l4_failed:
            response, _law_overridden = self.security.enforce_core_laws(response)
            if _law_overridden:
                ui.print_system("[L10] Core Law violation detected — response overridden", level="warn")

        # --- Background: L10 Security + L9 Reflection ---
        # Fire-and-forget: user already has the streamed response; post-processing
        # runs in a background thread so it doesn't add latency.
        if not l4_failed:
            _POST_PROCESS_POOL.submit(
                self._background_security_reflection,
                response,
                perceived,
                state,
                list(tasks) if tasks else [],
                raw_input,
                list(memories) if memories else [],
                list(conversation),
            )

        # Fast emotion heuristic — L9's richer version runs in background
        emotion = _fast_emotion(response, perceived.intent)

        # Notify consciousness engine of this interaction
        try:
            self.consciousness.on_interaction(emotion)
        except Exception:
            pass

        new_state = CortanaState(interaction_count=state.interaction_count + 1)
        new_conversation = list(conversation) + [
            ConversationTurn(role="user", content=raw_input),
            ConversationTurn(role="assistant", content=response),
        ]

        self.memory.update_working_memory("last_intent", perceived.intent)
        self.memory.update_working_memory("last_complexity", perceived.complexity)

        return response, new_state, new_conversation, emotion

    async def process(self, raw_input: str) -> str:
        """Terminal-mode pipeline: uses and updates self.state / self.conversation."""
        final, self.state, self.conversation, _emotion = await self.process_session(
            raw_input, self.state, self.conversation
        )
        return final

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the interactive terminal session."""
        ui.print_banner()
        ui.console.print(ui.make_status_panel(self.state, []))
        ui.print_divider()
        ui.print_system(
            "Type your message. [dim]/help[/] for commands, [dim]/exit[/] to quit.",
            level="info",
        )
        ui.print_divider()

        ui.print_response(
            "Online. All systems nominal — what do you need?",
            self.state,
        )

        while True:
            ui.print_divider()
            raw = ui.prompt_user(self.state)

            if not raw.strip():
                continue

            if raw.strip().lower() in ("/exit", "/quit", "exit", "quit"):
                ui.print_system("Cortana signing off.", level="info")
                break

            if raw.strip().lower() == "/help":
                _print_help()
                continue

            if raw.strip().lower() == "/memory":
                _show_memory(self.memory)
                continue

            if raw.strip().lower() == "/status":
                ui.console.print(ui.make_status_panel(self.state, self._active_agents))
                for p in self.reasoning.router.status():
                    level = "ok" if p["status"] == "ready" else "warn"
                    ui.print_system(f"  Provider [{p['provider']}]: {p['status']}", level=level)
                continue

            if raw.strip().lower() == "/reset":
                self.state = CortanaState()
                self.conversation.clear()
                ui.print_system("State and conversation cleared.", level="ok")
                continue

            if raw.strip().lower() == "/consciousness":
                ui.print_system(self.consciousness.get_consciousness_summary(), level="info")
                continue

            try:
                asyncio.run(self.process(raw))
            except KeyboardInterrupt:
                ui.print_system("\nInterrupted. Ready for next input.", level="info")
            except Exception as e:
                ui.print_system(f"Pipeline error: {e}", level="error")
                import traceback
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Help & Memory display helpers
# ---------------------------------------------------------------------------

def _print_help() -> None:
    from rich.table import Table
    table = Table(title="Cortana Commands", border_style=ui.CORTANA_DIM, show_header=False)
    table.add_column("Command", style=f"bold {ui.CORTANA_BLUE}", width=12)
    table.add_column("Description")
    table.add_row("/help", "Show this help")
    table.add_row("/status", "Show system state and active agents")
    table.add_row("/memory", "Show recent memory entries")
    table.add_row("/consciousness", "Show consciousness state (uptime, mood, recent thoughts)")
    table.add_row("/reset", "Reset state and conversation")
    table.add_row("/exit", "Disconnect from Cortana")
    ui.console.print(table)


def _show_memory(memory: CortanaMemory) -> None:
    episodes = memory.get_recent_episodes(limit=5)
    if not episodes:
        ui.print_system("No memory entries found.", level="info")
        return
    from rich.table import Table
    table = Table(title="Recent Episodes", border_style=ui.CORTANA_DIM)
    table.add_column("Timestamp", style="dim", width=20)
    table.add_column("Content")
    for ep in episodes:
        content = ep["content"][:80] + "..." if len(ep["content"]) > 80 else ep["content"]
        table.add_row(ep["stamp"], content)
    ui.console.print(table)


# ---------------------------------------------------------------------------
# Import for _show_memory
# ---------------------------------------------------------------------------
from cortana.layers.layer2_memory import CortanaMemory  # noqa: E402


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Cortana AI System")
    parser.add_argument("--web", action="store_true", help="Start web chat server (Layer 13)")
    parser.add_argument("--compute", action="store_true", help="Enable compute marketplace (Layer 14)")
    parser.add_argument("--host", default=None, help="Override web host")
    parser.add_argument("--port", type=int, default=None, help="Override web port")
    args = parser.parse_args()

    system = CortanaSystem()

    if args.web:
        from cortana.layers.layer13_chat import ChatLayer
        from cortana.layers.layer14_compute import ComputeLayer
        chat = ChatLayer(system)
        host = args.host or config.WEB_HOST
        port = args.port or config.WEB_PORT
        # Register compute marketplace routes on the same FastAPI app
        if args.compute or config.COMPUTE_ENABLED:
            compute = ComputeLayer(system, chat.app)
            ui.print_system("Layer 14 [Compute Marketplace] online", level="ok")
        chat.run(host=host, port=port)
    else:
        system.run()


if __name__ == "__main__":
    main()
