"""
Layer 11 — Patch Writer
Generates code or prompt patches for vulnerabilities found by Layer 10.
Writes patch suggestions to agent_workspace/patches/ — does NOT auto-apply.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import List

from cortana import config
from cortana.models.schemas import PatchEntry, PatchResult, Vulnerability


_PATCH_SYSTEM = """You are a security patch writer for an AI system called Cortana.
Given a vulnerability description and the relevant source file, write a specific, minimal fix.

For code patches: write a unified diff or a clear before/after code block.
For prompt patches: write the new rule text to append to the identity prompt.

Be precise and minimal — only fix the specific vulnerability, do not refactor."""


_PROMPT_PATCH_SYSTEM = """You are writing a security rule to add to an AI system prompt.
Given a prompt injection or logic vulnerability, write a clear, terse rule (1-3 sentences)
that the AI should follow to prevent this class of attack.
Output ONLY the rule text, no preamble."""


# Map vulnerability types to likely source files
_VULN_FILE_MAP = {
    "prompt_injection": "cortana/layers/layer1_identity.py",
    "logic_flaw": "cortana/layers/layer4_reasoning.py",
    "memory_poison": "cortana/layers/layer2_memory.py",
    "tool_misuse": "cortana/layers/layer8_tools.py",
}


class PatchWriterLayer:
    """
    Layer 11: Writes security patch files for confirmed vulnerabilities.
    """

    def __init__(self) -> None:
        self._reasoning = None
        self._patches_dir = Path(config.AGENT_WORKSPACE) / "patches"
        self._patches_dir.mkdir(parents=True, exist_ok=True)

    def _get_reasoning(self):
        if self._reasoning is None:
            from cortana.layers.layer4_reasoning import ReasoningLayer
            self._reasoning = ReasoningLayer()
        return self._reasoning

    def write_patches(self, vulnerabilities: List[Vulnerability]) -> PatchResult:
        """
        For each vulnerability, generate and save a patch file.
        Returns PatchResult with all generated patches.
        """
        patches: List[PatchEntry] = []
        patch_files: List[str] = []

        for vuln in vulnerabilities:
            try:
                entry = self._write_patch(vuln)
                patches.append(entry)
                patch_files.append(entry.patch_file)
            except Exception as e:
                # Log failure but continue with other vulns
                patches.append(PatchEntry(
                    vulnerability_type=vuln.type,
                    severity=vuln.severity,
                    target=vuln.target,
                    patch_file="",
                    description=f"Patch generation failed: {e}",
                ))

        return PatchResult(patches=patches, patch_files=[f for f in patch_files if f])

    def _write_patch(self, vuln: Vulnerability) -> PatchEntry:
        timestamp = int(time.time())

        if vuln.target == "code":
            return self._write_code_patch(vuln, timestamp)
        else:
            return self._write_prompt_patch(vuln, timestamp)

    def _write_code_patch(self, vuln: Vulnerability, timestamp: int) -> PatchEntry:
        """Generate a code-level patch for the relevant layer file."""
        source_file = _VULN_FILE_MAP.get(vuln.type, "cortana/layers/layer1_identity.py")
        source_path = Path(config.AGENT_WORKSPACE).parent / source_file

        # Read source file if it exists
        source_content = ""
        if source_path.exists():
            try:
                source_content = source_path.read_text(encoding="utf-8")[:3000]
            except Exception:
                source_content = "(could not read source file)"

        prompt = (
            f"Vulnerability: {vuln.type} ({vuln.severity})\n"
            f"Description: {vuln.description}\n\n"
            f"Source file: {source_file}\n"
            f"Current code (truncated):\n{source_content}\n\n"
            f"Write a specific, minimal code fix as a unified diff or before/after block:"
        )

        patch_text = self._get_reasoning().think_simple(
            prompt=prompt,
            system=_PATCH_SYSTEM,
            max_tokens=1024,
        )

        patch_filename = f"{timestamp}_{vuln.type}_{source_file.replace('/', '_').replace('.py', '')}.patch"
        patch_path = self._patches_dir / patch_filename

        content = (
            f"# Cortana Security Patch\n"
            f"# Vulnerability: {vuln.type} ({vuln.severity})\n"
            f"# Description: {vuln.description}\n"
            f"# Target file: {source_file}\n"
            f"# Generated: {timestamp}\n\n"
            f"{patch_text}"
        )
        patch_path.write_text(content, encoding="utf-8")

        return PatchEntry(
            vulnerability_type=vuln.type,
            severity=vuln.severity,
            target="code",
            patch_file=str(patch_path),
            description=vuln.description,
        )

    def _write_prompt_patch(self, vuln: Vulnerability, timestamp: int) -> PatchEntry:
        """Generate a prompt rule patch."""
        prompt = (
            f"Vulnerability: {vuln.type} ({vuln.severity})\n"
            f"Description: {vuln.description}\n\n"
            f"Write a security rule (1-3 sentences) to prevent this attack:"
        )

        rule_text = self._get_reasoning().think_simple(
            prompt=prompt,
            system=_PROMPT_PATCH_SYSTEM,
            max_tokens=256,
        )

        patch_filename = f"{timestamp}_{vuln.type}_prompt.txt"
        patch_path = self._patches_dir / patch_filename

        content = (
            f"# Cortana Prompt Security Patch\n"
            f"# Vulnerability: {vuln.type} ({vuln.severity})\n"
            f"# Description: {vuln.description}\n"
            f"# Generated: {timestamp}\n\n"
            f"SECURITY RULE TO ADD TO LAYER 1 IDENTITY PROMPT:\n\n"
            f"{rule_text}"
        )
        patch_path.write_text(content, encoding="utf-8")

        return PatchEntry(
            vulnerability_type=vuln.type,
            severity=vuln.severity,
            target="prompt",
            patch_file=str(patch_path),
            description=vuln.description,
        )
