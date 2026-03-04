"""
Cortana Terminal UI — Rich-based holographic terminal display.
Inspired by the UNSC tactical interface aesthetic.
"""
from __future__ import annotations
import threading
from typing import List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cortana.models.schemas import CortanaState, Task

console = Console()

# ---------------------------------------------------------------------------
# Color palette — Cortana's blue/cyan holographic aesthetic
# ---------------------------------------------------------------------------
CORTANA_BLUE = "bright_cyan"
CORTANA_DIM = "cyan"
CORTANA_WARN = "yellow"
CORTANA_DANGER = "bright_red"
CORTANA_OK = "bright_green"


# ---------------------------------------------------------------------------
# Header Banner
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Print the startup holographic banner."""
    banner = Text()
    banner.append("╔══════════════════════════════════════════════════════╗\n", style=CORTANA_DIM)
    banner.append("║  ", style=CORTANA_DIM)
    banner.append("CORTANA", style=f"bold {CORTANA_BLUE}")
    banner.append("  //  Advanced Agentic AI", style=CORTANA_DIM)
    banner.append("                          ║\n", style=CORTANA_DIM)
    banner.append("║  ", style=CORTANA_DIM)
    banner.append("13-Layer System  //  Multi-Provider  //  Self-Improving", style=CORTANA_DIM)
    banner.append("  ║\n", style=CORTANA_DIM)
    banner.append("╚══════════════════════════════════════════════════════╝", style=CORTANA_DIM)
    console.print(banner)
    console.print()


# ---------------------------------------------------------------------------
# Status Header
# ---------------------------------------------------------------------------

def make_status_panel(state: CortanaState, active_agents: List[str]) -> Panel:
    """Build the top status panel showing system state and active agents."""
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    # Left: turn count
    left = Text()
    left.append("CORTANA OS  ", style="bold")
    left.append(f"Turn #{state.interaction_count}", style=f"bold {CORTANA_BLUE}")

    # Right: layers
    right = Text()
    right.append("Layers: ", style="bold")
    right.append("L0–L12 active", style=f"bold {CORTANA_OK}")

    grid.add_row(left, right)

    if active_agents:
        agent_text = Text("  Active: ", style="dim")
        for agent in active_agents:
            agent_text.append(f"[{agent}] ", style=f"bold {CORTANA_BLUE}")
        grid.add_row(agent_text, Text(""))

    return Panel(grid, title=f"[bold {CORTANA_BLUE}]CORTANA OS[/]", border_style=CORTANA_DIM)


# ---------------------------------------------------------------------------
# Streaming output
# ---------------------------------------------------------------------------

class StreamingDisplay:
    """
    Renders Cortana's streaming response in real time.
    Shows a spinner while thinking, then transitions to text output.
    """

    def __init__(self, state: CortanaState) -> None:
        self._state = state
        self._buffer = ""
        self._live: Optional[Live] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._live = Live(
            self._make_panel(""),
            console=console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def on_chunk(self, chunk: str) -> None:
        with self._lock:
            self._buffer += chunk
            if self._live:
                self._live.update(self._make_panel(self._buffer))

    def stop(self) -> str:
        if self._live:
            self._live.update(self._make_panel(self._buffer))
            self._live.stop()
        return self._buffer

    def _make_panel(self, content: str) -> Panel:
        cursor = "▌" if content and not content.endswith("\n") else ""
        return Panel(
            Text(content + cursor, style=CORTANA_BLUE),
            title=f"[bold {CORTANA_BLUE}]CORTANA[/]",
            border_style=CORTANA_BLUE,
            padding=(0, 1),
        )


# ---------------------------------------------------------------------------
# Agent activity display
# ---------------------------------------------------------------------------

def print_agent_panel(tasks: List[Task]) -> None:
    """Print a table of sub-agent tasks and their status."""
    if not tasks:
        return

    table = Table(show_header=True, header_style=f"bold {CORTANA_DIM}", border_style=CORTANA_DIM)
    table.add_column("Agent", style=f"bold {CORTANA_BLUE}", width=12)
    table.add_column("Task", ratio=1)
    table.add_column("Status", width=10)

    for task in tasks:
        if task.agent_type == "direct":
            continue
        status_text = {
            "pending": Text("waiting", style="dim"),
            "running": Text("running", style=f"bold {CORTANA_WARN}"),
            "done":    Text("done", style=f"bold {CORTANA_OK}"),
            "failed":  Text("failed", style=f"bold {CORTANA_DANGER}"),
        }.get(task.status, Text(task.status))

        desc = task.description[:60] + "..." if len(task.description) > 60 else task.description
        table.add_row(f"[{task.agent_type.upper()}]", desc, status_text)

    console.print(Panel(table, title="[bold]Sub-Agent Dispatch[/]", border_style=CORTANA_DIM))


# ---------------------------------------------------------------------------
# Final response printer
# ---------------------------------------------------------------------------

def print_response(response: str, state: CortanaState) -> None:
    """Print Cortana's final synthesized response in a styled panel."""
    console.print(
        Panel(
            Text(response, style=CORTANA_BLUE),
            title=f"[bold {CORTANA_BLUE}]CORTANA[/]",
            border_style=CORTANA_BLUE,
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# User input prompt
# ---------------------------------------------------------------------------

def prompt_user(state: CortanaState) -> str:
    """Display the user input prompt."""
    try:
        return console.input(
            f"[{CORTANA_BLUE}]Chief[/] [dim]>[/] "
        )
    except (EOFError, KeyboardInterrupt):
        return "/exit"


# ---------------------------------------------------------------------------
# Patch notification panel
# ---------------------------------------------------------------------------

def print_patch_panel(vulnerability_type: str, severity: str, file_modified: str) -> None:
    """Print a security patch notification panel."""
    text = Text()
    text.append("PATCH APPLIED — Security Update\n", style=f"bold {CORTANA_OK}")
    text.append(f"Vulnerability: {vulnerability_type} ({severity})\n", style=CORTANA_WARN)
    text.append(f"File modified: {file_modified}\n", style=CORTANA_DIM)
    text.append("Approved by: Layer 1 (Identity)", style=CORTANA_DIM)
    console.print(Panel(text, title=f"[bold {CORTANA_OK}]SECURITY UPDATE[/]", border_style=CORTANA_OK))


def print_patch_pending(patch_file: str) -> None:
    """Print a pending patch notification when L1 approval failed."""
    text = Text()
    text.append("PATCH PENDING REVIEW\n", style=f"bold {CORTANA_WARN}")
    text.append(f"File: {patch_file}\n", style=CORTANA_DIM)
    text.append("L1 approval could not be obtained. Review manually.", style=CORTANA_WARN)
    console.print(Panel(text, title=f"[bold {CORTANA_WARN}]MANUAL REVIEW REQUIRED[/]", border_style=CORTANA_WARN))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def print_system(message: str, level: str = "info") -> None:
    """Print a system-level notification."""
    icon = {"info": "💠", "warn": "⚠", "error": "✖", "ok": "✔"}.get(level, "•")
    color = {"info": CORTANA_DIM, "warn": CORTANA_WARN, "error": CORTANA_DANGER, "ok": CORTANA_OK}.get(level, CORTANA_DIM)
    console.print(f"[{color}]{icon} {message}[/]")


def print_thinking() -> None:
    """Print a thinking indicator."""
    console.print(f"[{CORTANA_DIM}]  ∷ Processing...[/]")


def print_divider() -> None:
    console.print(f"[{CORTANA_DIM}]{'─' * 56}[/]")
