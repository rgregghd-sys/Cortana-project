"""
Layer 7 — Sub-Agents
Specialized async agents: Researcher, Coder, Analyst, Writer.
Patterns from: jarvis_system/developer_wing.py (Agent 3/4 code gen + debug loop).
Each agent uses claude-haiku-4-5-20251001 for efficiency.
"""
from __future__ import annotations
import asyncio
import re
from typing import Optional

from cortana import config
from cortana.layers import layer8_tools as tools
from cortana.models.schemas import Task

# We import ReasoningLayer lazily to avoid circular imports at module load
_reasoning: Optional[object] = None


def _get_reasoning():
    global _reasoning
    if _reasoning is None:
        from cortana.layers.layer4_reasoning import ReasoningLayer
        _reasoning = ReasoningLayer()
    return _reasoning


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """All sub-agents share this interface."""

    agent_type: str = "base"

    async def run(self, task: Task) -> Task:
        raise NotImplementedError

    def _llm(self, prompt: str, system: str = "") -> str:
        return _get_reasoning().think_simple(
            prompt=prompt,
            system=system or f"You are Cortana's {self.agent_type} sub-processor. Be concise and precise.",
            model=config.SUB_AGENT_MODEL,
        )


# ---------------------------------------------------------------------------
# Researcher Agent
# ---------------------------------------------------------------------------

class ResearcherAgent(BaseAgent):
    """
    Searches DuckDuckGo, optionally scrapes URLs, and synthesizes findings.
    """
    agent_type = "researcher"

    async def run(self, task: Task) -> Task:
        task.status = "running"
        try:
            # 1. Run web search
            search_results = await tools.web_search(task.description)

            # 2. Synthesize with LLM
            system = (
                "You are Cortana's Researcher sub-processor. "
                "Given raw search results, synthesize a clear, factual summary. "
                "Be precise. Flag any conflicting information. "
                "Output should be 2-4 concise paragraphs."
            )
            prompt = (
                f"Task: {task.description}\n\n"
                f"Search Results:\n{search_results}\n\n"
                f"Additional context: {task.context or 'None'}\n\n"
                f"Synthesize the findings:"
            )

            summary = await asyncio.get_event_loop().run_in_executor(
                None, self._llm, prompt, system
            )

            task.result = summary
            task.status = "done"

        except Exception as e:
            task.result = f"Research failed: {e}"
            task.status = "failed"

        return task


# ---------------------------------------------------------------------------
# Coder Agent  (mirrors developer_wing.py Agent 3/4 pattern)
# ---------------------------------------------------------------------------

class CoderAgent(BaseAgent):
    """
    Generates Python code, validates in sandbox, auto-debugs on failure.
    Implements the Agent 3 (generate) + Agent 4 (debug) pattern from developer_wing.py.
    """
    agent_type = "coder"
    MAX_DEBUG_ATTEMPTS = 2

    async def run(self, task: Task) -> Task:
        task.status = "running"
        try:
            code = await self._generate(task.description, task.context)

            for attempt in range(self.MAX_DEBUG_ATTEMPTS + 1):
                exec_output = await tools.execute_code(code)

                if not exec_output.startswith(("Syntax Error", "Error:", "Execution error", "Execution timed")):
                    # Success
                    task.result = (
                        f"```python\n{code}\n```\n\n"
                        f"**Execution Output:**\n```\n{exec_output}\n```"
                    )
                    task.status = "done"
                    return task

                if attempt < self.MAX_DEBUG_ATTEMPTS:
                    code = await self._debug(code, exec_output)
                else:
                    # Return best effort with error
                    task.result = (
                        f"Code generated but execution failed after {attempt + 1} attempts.\n\n"
                        f"```python\n{code}\n```\n\n"
                        f"**Last Error:**\n```\n{exec_output}\n```"
                    )
                    task.status = "failed"

        except Exception as e:
            task.result = f"Coder agent error: {e}"
            task.status = "failed"

        return task

    async def _generate(self, description: str, context: Optional[str]) -> str:
        system = (
            "You are Cortana's Coder sub-processor. "
            "Write clean, working Python code. "
            "Output ONLY the raw Python code with no markdown fences or explanation."
        )
        prompt = f"Write Python code for: {description}"
        if context:
            prompt += f"\n\nContext: {context}"

        raw = await asyncio.get_event_loop().run_in_executor(
            None, self._llm, prompt, system
        )
        return re.sub(r"```python|```", "", raw).strip()

    async def _debug(self, code: str, error: str) -> str:
        system = (
            "You are Cortana's Debug sub-processor. "
            "Fix the given Python code. "
            "Output ONLY the corrected raw Python code with no markdown fences."
        )
        prompt = f"Fix this Python code:\n```python\n{code}\n```\n\nError:\n{error}"
        raw = await asyncio.get_event_loop().run_in_executor(
            None, self._llm, prompt, system
        )
        return re.sub(r"```python|```", "", raw).strip()


# ---------------------------------------------------------------------------
# Analyst Agent
# ---------------------------------------------------------------------------

class AnalystAgent(BaseAgent):
    """Breaks down problems, compares options, produces structured analysis."""
    agent_type = "analyst"

    async def run(self, task: Task) -> Task:
        task.status = "running"
        try:
            system = (
                "You are Cortana's Analyst sub-processor. "
                "Produce rigorous, structured analysis. "
                "Use headers, bullet points, and clear conclusions. "
                "Be objective. Flag assumptions explicitly."
            )
            prompt = (
                f"Analyze the following:\n{task.description}\n\n"
                f"Context: {task.context or 'None'}"
            )
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._llm, prompt, system
            )
            task.result = result
            task.status = "done"
        except Exception as e:
            task.result = f"Analysis failed: {e}"
            task.status = "failed"
        return task


# ---------------------------------------------------------------------------
# Writer Agent
# ---------------------------------------------------------------------------

class WriterAgent(BaseAgent):
    """Produces polished prose, reports, documentation, creative content."""
    agent_type = "writer"

    async def run(self, task: Task) -> Task:
        task.status = "running"
        try:
            system = (
                "You are Cortana's Writer sub-processor. "
                "Produce clear, well-structured, engaging written content. "
                "Adapt tone to the context (technical, creative, formal). "
                "No filler. Every sentence earns its place."
            )
            prompt = (
                f"Writing task: {task.description}\n\n"
                f"Context: {task.context or 'None'}"
            )
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._llm, prompt, system
            )
            task.result = result
            task.status = "done"
        except Exception as e:
            task.result = f"Writing failed: {e}"
            task.status = "failed"
        return task


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict = {
    "researcher": ResearcherAgent,
    "coder":      CoderAgent,
    "analyst":    AnalystAgent,
    "writer":     WriterAgent,
}


def get_agent(agent_type: str) -> BaseAgent:
    """Factory: return the right agent instance for the given type."""
    cls = AGENT_REGISTRY.get(agent_type)
    if cls is None:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return cls()
