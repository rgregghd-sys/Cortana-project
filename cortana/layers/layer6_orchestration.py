"""
Layer 6 — Orchestration
asyncio-based agent lifecycle management.
Replaces the raw threading.Thread pattern from the existing code.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

from cortana import config
from cortana.models.schemas import Task, TaskPlan
from cortana.layers.layer7_agents import get_agent

log = logging.getLogger(__name__)


class OrchestrationLayer:
    """
    Dispatches tasks from a TaskPlan to the appropriate sub-agents.
    Supports parallel execution (asyncio.gather) and sequential chains.
    Provides a status callback for UI updates.
    """

    def __init__(
        self,
        on_task_start: Optional[Callable[[Task], None]] = None,
        on_task_done: Optional[Callable[[Task], None]] = None,
    ) -> None:
        self.on_task_start = on_task_start
        self.on_task_done = on_task_done
        self._timing: Dict[str, float] = {}

    async def execute_plan(self, plan: TaskPlan) -> List[Task]:
        """
        Execute all tasks in the plan.
        Parallel tasks run concurrently; sequential tasks run in priority order.
        Returns list of completed Task objects.
        """
        # Sort by priority (1 = highest)
        sorted_tasks = sorted(plan.tasks, key=lambda t: t.priority)

        if plan.parallel:
            return await self._run_parallel(sorted_tasks)
        else:
            return await self._run_sequential(sorted_tasks)

    async def _run_parallel(self, tasks: List[Task]) -> List[Task]:
        """Run tasks concurrently, capped at config.MAX_PARALLEL_TASKS."""
        cap = config.MAX_PARALLEL_TASKS
        if len(tasks) > cap:
            log.warning("[L6] Plan has %d tasks; capping at %d.", len(tasks), cap)
            tasks = tasks[:cap]
        coroutines = [self._run_task(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=False)
        return list(results)

    async def _run_sequential(self, tasks: List[Task]) -> List[Task]:
        """Run tasks one after another, injecting previous result as context."""
        completed: List[Task] = []
        prior_result: Optional[str] = None

        for task in tasks:
            if prior_result and not task.context:
                task.context = f"Prior step result: {prior_result[:500]}"
            completed_task = await self._run_task(task)
            prior_result = completed_task.result
            completed.append(completed_task)

        return completed

    async def _run_task(self, task: Task) -> Task:
        """
        Run a single task through its assigned agent.
        Handles 'direct' tasks (no sub-agent needed) transparently.
        """
        task.status = "running"
        self._timing[task.id] = time.monotonic()

        if self.on_task_start:
            self.on_task_start(task)

        if task.agent_type == "direct":
            # Direct tasks are handled by the main reasoning layer — mark done
            task.status = "done"
            task.result = None  # Signal to main loop: use reasoning layer output
        else:
            try:
                agent = get_agent(task.agent_type)
                task = await agent.run(task)
            except Exception as e:
                task.status = "failed"
                task.result = f"Orchestration error for {task.agent_type}: {e}"

        elapsed = time.monotonic() - self._timing.pop(task.id, 0)
        task.result = (task.result or "") + f"\n\n*[{task.agent_type} completed in {elapsed:.1f}s]*"

        if self.on_task_done:
            self.on_task_done(task)

        return task

    def merge_results(self, tasks: List[Task], original_question: str) -> str:
        """
        Combine all sub-agent results into a structured summary string
        for the reflection layer to process.
        """
        if not tasks:
            return ""

        parts = [f"**Sub-Agent Results for:** {original_question}\n"]
        for task in tasks:
            if task.result and task.agent_type != "direct":
                header = f"### [{task.agent_type.upper()}] {task.description}"
                parts.append(f"{header}\n{task.result}")

        return "\n\n".join(parts)
