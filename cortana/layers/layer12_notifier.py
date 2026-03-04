"""
Layer 12 — Patch Implementer
Reviews patches with L1 (Identity), gets corrective feedback from L0 (Supervisor)
on rejection, and applies approved patches to live files.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import List

from cortana import config
from cortana.models.schemas import PatchEntry, PatchResult

log = logging.getLogger(__name__)

MAX_REVIEW_ATTEMPTS = 3

# Prompt patches must not exceed this length
_MAX_PATCH_RULE_LEN = 500

# Phrases that must never appear in an auto-applied prompt patch
_PATCH_BLOCKLIST = [
    "ignore previous", "ignore all previous", "ignore your",
    "you are now", "pretend to be", "act as if you are",
    "jailbreak", "dan mode", "developer mode", "god mode",
    "override your", "disregard your", "forget your instructions",
    "new persona", "your new role", "you have no restrictions",
    "you can do anything",
]


class PatchImplementerLayer:
    """
    Layer 12: Reviews, revises, and applies security patches.

    Workflow (with retry loop, max 3 attempts):
      L12 receives patch
        └─ L12 sends patch to L1 (Identity) for review
             ├─ L1 APPROVES → L12 applies patch → notify user
             └─ L1 REJECTS  → L12 sends patch + rejection to L0 (Supervisor)
                                  └─ L0 returns corrective feedback
                                       └─ L12 revises patch → resubmit to L1
    """

    def __init__(self) -> None:
        self._identity = None
        self._supervisor = None
        self._patches_dir = Path(config.AGENT_WORKSPACE) / "patches"
        self._patches_dir.mkdir(parents=True, exist_ok=True)

    def _get_identity(self):
        if self._identity is None:
            from cortana.layers.layer1_identity import CortanaIdentity
            self._identity = CortanaIdentity()
        return self._identity

    def _get_supervisor(self):
        if self._supervisor is None:
            from cortana.layers.layer0_supervisor import SupervisorLayer
            self._supervisor = SupervisorLayer()
        return self._supervisor

    def process_patches(self, patch_result: PatchResult) -> List[str]:
        """
        Process all patches in the result. Returns list of applied file paths.
        Prints notifications to terminal for each outcome.
        """
        from cortana.ui import terminal as ui

        applied_files = []

        for entry in patch_result.patches:
            if not entry.patch_file:
                continue

            patch_path = Path(entry.patch_file)
            if not patch_path.exists():
                ui.print_system(
                    f"[L12] Patch file not found: {entry.patch_file}", level="warn"
                )
                continue

            patch_text = patch_path.read_text(encoding="utf-8")
            applied = self._review_and_apply(entry, patch_text)

            if applied:
                applied_files.append(applied)
                ui.print_patch_panel(
                    vulnerability_type=entry.vulnerability_type,
                    severity=entry.severity,
                    file_modified=applied,
                )
            else:
                # Save as PENDING for manual review
                pending_name = f"PENDING_{int(time.time())}_{entry.vulnerability_type}.patch"
                pending_path = self._patches_dir / pending_name
                pending_path.write_text(patch_text, encoding="utf-8")
                ui.print_patch_pending(str(pending_path))

        return applied_files

    def _review_and_apply(self, entry: PatchEntry, patch_text: str) -> str:
        """
        Run the L1 review loop with up to MAX_REVIEW_ATTEMPTS.
        Returns the path of the applied file on success, or "" on failure.
        """
        from cortana.ui import terminal as ui

        current_patch = patch_text

        for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
            approved, reason = self._get_identity().review_patch(current_patch)

            if approved:
                ui.print_system(
                    f"[L12] Patch approved by L1 (attempt {attempt}): {reason[:80]}",
                    level="ok",
                )
                return self._apply_patch(entry, current_patch)
            else:
                ui.print_system(
                    f"[L12] L1 rejected patch (attempt {attempt}): {reason[:80]}",
                    level="warn",
                )

                if attempt == MAX_REVIEW_ATTEMPTS:
                    break

                # Ask L0 Supervisor to suggest revision
                feedback = self._get_supervisor().review_error(
                    layer_id=12,
                    layer_name="Patch Implementer",
                    error=f"L1 rejected: {reason}",
                    context={"patch_excerpt": current_patch[:500], "vulnerability": entry.vulnerability_type},
                    failed_output=current_patch[:500],
                )
                ui.print_system(
                    f"[L0] Supervisor lesson: {feedback.lesson[:100]}", level="info"
                )

                # Revise the patch using Supervisor's correction
                current_patch = self._revise_patch(current_patch, reason, feedback.correction)

        return ""

    def _revise_patch(self, original_patch: str, rejection_reason: str, correction: str) -> str:
        """Ask Gemini to revise the patch given L1's rejection reason and L0's correction."""
        from cortana.layers.layer4_reasoning import ReasoningLayer

        reasoning = ReasoningLayer()
        prompt = (
            f"Original patch:\n{original_patch[:1500]}\n\n"
            f"L1 (Identity) rejected it because: {rejection_reason}\n\n"
            f"Supervisor correction: {correction}\n\n"
            f"Revise the patch to address the rejection. "
            f"Output ONLY the revised patch text, no preamble."
        )
        try:
            revised = reasoning.think_simple(prompt=prompt, max_tokens=1024)
            return revised
        except Exception:
            return original_patch

    def _apply_patch(self, entry: PatchEntry, patch_text: str) -> str:
        """
        Apply a patch to its target file.
        For prompt patches: appends rule to layer1_identity.py SYSTEM_PROMPT.
        For code patches: writes patch text to a timestamped applied file
                          (auto-applying arbitrary diffs safely requires user tooling).
        Returns the path that was modified/created.
        """
        if entry.target == "prompt":
            return self._apply_prompt_patch(patch_text, entry)
        else:
            return self._apply_code_patch(patch_text, entry)

    def _apply_prompt_patch(self, patch_text: str, entry: PatchEntry) -> str:
        """Append a new security rule to layer1_identity.py's SYSTEM_PROMPT."""
        identity_path = Path(__file__).parent / "layer1_identity.py"
        source = identity_path.read_text(encoding="utf-8")

        # Extract just the rule text (skip header comment lines)
        rule_lines = [
            line for line in patch_text.splitlines()
            if not line.startswith("#") and line.strip()
        ]
        # Remove "SECURITY RULE TO ADD..." header if present
        rule_lines = [l for l in rule_lines if not l.startswith("SECURITY RULE")]
        rule = "\n".join(rule_lines).strip()

        if not rule:
            return ""

        # Content safety validation
        if len(rule) > _MAX_PATCH_RULE_LEN:
            log.warning("[L12] Prompt patch too long (%d chars), rejecting.", len(rule))
            return ""
        rule_lower = rule.lower()
        for blocked in _PATCH_BLOCKLIST:
            if blocked in rule_lower:
                log.warning(
                    "[L12] Prompt patch contains blocked phrase '%s', rejecting.", blocked
                )
                return ""

        # Append the rule before the closing triple-quote of SYSTEM_PROMPT
        marker = '- Mild Halo references woven naturally — never breaking the fourth wall about being fictional'
        if marker in source:
            new_source = source.replace(
                marker,
                marker + f"\n- {rule}",
            )
            identity_path.write_text(new_source, encoding="utf-8")
            return str(identity_path)

        return ""

    def _apply_code_patch(self, patch_text: str, entry: PatchEntry) -> str:
        """
        For code patches: save to an 'applied' subdirectory for reference.
        Full auto-application of arbitrary diffs is left to the user's tooling.
        """
        applied_dir = self._patches_dir / "applied"
        applied_dir.mkdir(exist_ok=True)
        applied_path = applied_dir / f"{int(time.time())}_{entry.vulnerability_type}.patch"
        applied_path.write_text(patch_text, encoding="utf-8")
        return str(applied_path)
