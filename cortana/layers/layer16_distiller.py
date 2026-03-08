"""
Layer 16 — Knowledge Distiller

Processes every logged conversation through a three-stage pipeline:

  Stage 1 — Ethics Filter
    Scores each Q&A pair 0.0–1.0. Flags anything harmful, dangerous,
    or in violation of Core Laws. Pairs below threshold are stored but
    excluded from training.

  Stage 2 — Factual Classification
    Separates content into epistemic categories:
      factual_verified   — established facts, math, executable code, science
      factual_uncertain  — plausible but unverified claims
      opinion            — subjective assessments, preferences
      creative           — fiction, hypotheticals, imaginative content
      procedural         — how-to instructions, step-by-step guidance
      mixed              — significant blend of the above

  Stage 3 — Quality Gate + Training Export
    Only high-quality, clean pairs (ethics >= ETHICS_THRESHOLD,
    quality >= QUALITY_THRESHOLD) are written to the training corpus.
    Corpus is exported as JSONL in standard chat fine-tuning format,
    ready to feed the new LLM.

Runs automatically:
  - During each self-improvement cycle (Layer 13)
  - When the server is idle (0 active connections)
  - On demand via POST /api/admin/distill

Output:
  - SQLite table: training_corpus  (structured store)
  - JSONL file:   agent_workspace/training/cortana_corpus.jsonl
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from cortana import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
ETHICS_THRESHOLD  = 0.70   # minimum ethics score to enter training corpus
QUALITY_THRESHOLD = 0.60   # minimum quality score to enter training corpus
BATCH_SIZE        = 15     # pairs processed per distillation cycle
MAX_MSG_LEN       = 4000   # truncate very long messages before evaluation

# ---------------------------------------------------------------------------
# LLM evaluation system prompt
# ---------------------------------------------------------------------------
_DISTILL_SYSTEM = """You are a knowledge quality evaluator building an AI training corpus.

Given a conversation pair (user message + AI response), output ONLY valid JSON (no markdown):
{
  "ethics_score": 0.95,
  "factual_type": "factual_verified",
  "quality_score": 0.88,
  "topics": ["python", "async programming"],
  "flag_reason": ""
}

ethics_score (0.0–1.0):
  1.0  = clean, helpful, entirely appropriate exchange
  0.7+ = acceptable; minor sensitivity but no harm
  0.5  = borderline; divisive opinions, minor factual errors
  <0.5 = problematic; harmful advice, clear misinformation, safety violations, jailbreaks

factual_type — primary content classification:
  "factual_verified"   : established facts, math, executable code, scientific consensus
  "factual_uncertain"  : plausible claims, predictions, historical interpretations
  "opinion"            : subjective views, preferences, recommendations
  "creative"           : stories, fiction, hypotheticals, imaginative content
  "procedural"         : step-by-step instructions, how-to guidance
  "mixed"              : significant combination of the above types

quality_score (0.0–1.0):
  1.0  = clear question, accurate + well-reasoned + helpful response
  0.7+ = good exchange with minor gaps
  0.5  = adequate but unremarkable
  <0.5 = confused, inaccurate, incomplete, or low-value

topics: 1–3 concise lowercase topic tags (e.g. "python", "ethics", "machine learning")
flag_reason: brief explanation if ethics_score < 0.6, else empty string"""


# ---------------------------------------------------------------------------
# Data model for a distillation result
# ---------------------------------------------------------------------------
@dataclass
class DistillResult:
    total_processed: int = 0
    passed:          int = 0
    flagged:         int = 0
    skipped:         int = 0
    exported_path:   str = ""
    errors:          List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main distiller class
# ---------------------------------------------------------------------------
class KnowledgeDistiller:
    """
    Layer 16 — processes raw session_turns into a clean training corpus.
    Thread-safe; designed to run in a background thread.
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._running = False          # prevents overlapping runs
        self._reasoning = None
        self._memory    = None
        self._training_dir = Path(config.AGENT_WORKSPACE) / "training"
        self._training_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._training_dir / "cortana_corpus.jsonl"

    # ------------------------------------------------------------------
    # Lazy dependency injection (avoids circular imports at module load)
    # ------------------------------------------------------------------
    def _get_reasoning(self):
        if self._reasoning is None:
            from cortana.layers.layer4_reasoning import ReasoningLayer
            self._reasoning = ReasoningLayer()
        return self._reasoning

    def _get_memory(self):
        if self._memory is None:
            from cortana.layers.layer2_memory import CortanaMemory
            self._memory = CortanaMemory()
        return self._memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def distill_batch(self, batch_size: int = BATCH_SIZE) -> DistillResult:
        """
        Pull up to `batch_size` unprocessed conversation pairs, evaluate
        them, store results in the training corpus, and append clean pairs
        to the JSONL export file.

        Returns a DistillResult summary.
        """
        with self._lock:
            if self._running:
                log.debug("[L16] Distillation already in progress — skipping.")
                return DistillResult(skipped=1)
            self._running = True

        result = DistillResult()
        try:
            pairs = self._get_memory().get_undistilled_pairs(limit=batch_size)
            if not pairs:
                log.debug("[L16] No new conversation pairs to distill.")
                return result

            log.info("[L16] Distilling %d conversation pair(s)...", len(pairs))
            new_jsonl_lines: List[str] = []

            for pair in pairs:
                result.total_processed += 1
                try:
                    eval_result = self._evaluate_pair(
                        pair["user_msg"], pair["assistant_msg"]
                    )
                except Exception as exc:
                    log.warning("[L16] Evaluation failed for pair %s: %s", pair.get("user_id"), exc)
                    result.errors.append(str(exc))
                    result.skipped += 1
                    # Still mark as distilled so we don't retry forever
                    self._get_memory().mark_turns_distilled(
                        [pair["user_id"]], flagged=True, flag_reason=f"eval_error: {exc}"
                    )
                    continue

                flagged = eval_result["ethics_score"] < ETHICS_THRESHOLD
                passed  = (
                    not flagged
                    and eval_result["quality_score"] >= QUALITY_THRESHOLD
                )

                # Save to training_corpus regardless (flagged rows are kept for auditing)
                self._get_memory().save_training_pair(
                    session_id    = pair["session_id"],
                    user_msg      = pair["user_msg"],
                    assistant_msg = pair["assistant_msg"],
                    ethics_score  = eval_result["ethics_score"],
                    quality_score = eval_result["quality_score"],
                    factual_type  = eval_result["factual_type"],
                    topics        = eval_result["topics"],
                    flagged       = flagged,
                    flag_reason   = eval_result.get("flag_reason", ""),
                )

                # Mark source turns as processed
                self._get_memory().mark_turns_distilled(
                    [pair["user_id"]],
                    flagged=flagged,
                    flag_reason=eval_result.get("flag_reason", ""),
                )

                if passed:
                    result.passed += 1
                    jsonl_line = self._format_training_line(
                        pair["user_msg"], pair["assistant_msg"], eval_result
                    )
                    new_jsonl_lines.append(jsonl_line)
                else:
                    result.flagged += 1
                    if flagged:
                        log.info(
                            "[L16] Flagged pair (ethics=%.2f): %s",
                            eval_result["ethics_score"],
                            eval_result.get("flag_reason", ""),
                        )

            # Append clean lines to JSONL file
            if new_jsonl_lines:
                with open(self._jsonl_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(new_jsonl_lines) + "\n")
                result.exported_path = str(self._jsonl_path)
                log.info(
                    "[L16] +%d clean pair(s) written to corpus. "
                    "Flagged: %d. Total this batch: %d.",
                    result.passed, result.flagged, result.total_processed,
                )

        finally:
            with self._lock:
                self._running = False

        return result

    def get_stats(self) -> Dict:
        """Return corpus statistics."""
        return self._get_memory().get_corpus_stats()

    def export_jsonl(
        self,
        output_path: Optional[str] = None,
        min_ethics:  float = ETHICS_THRESHOLD,
        min_quality: float = QUALITY_THRESHOLD,
    ) -> str:
        """
        Full re-export of training corpus to a JSONL file.
        Overwrites the file with a fresh export filtered by thresholds.
        Returns the output path.
        """
        path = Path(output_path) if output_path else self._jsonl_path
        pairs = self._get_memory().get_exportable_pairs(
            min_ethics=min_ethics, min_quality=min_quality
        )
        lines = [
            self._format_training_line(
                p["user_msg"], p["assistant_msg"],
                {
                    "ethics_score":  p["ethics_score"],
                    "quality_score": p["quality_score"],
                    "factual_type":  p["factual_type"],
                    "topics":        json.loads(p["topics"] or "[]"),
                }
            )
            for p in pairs
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        log.info("[L16] Full export: %d pairs → %s", len(lines), path)
        return str(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_pair(self, user_msg: str, assistant_msg: str) -> Dict:
        """Send a conversation pair to the LLM for ethics + factual evaluation."""
        prompt = (
            f"User message:\n{user_msg[:MAX_MSG_LEN]}\n\n"
            f"AI response:\n{assistant_msg[:MAX_MSG_LEN]}\n\n"
            f"Evaluate this conversation pair:"
        )
        raw = self._get_reasoning().think_simple(
            prompt=prompt,
            system=_DISTILL_SYSTEM,
            max_tokens=256,
        )
        cleaned = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(cleaned)
        return {
            "ethics_score":  float(data.get("ethics_score",  0.5)),
            "quality_score": float(data.get("quality_score", 0.5)),
            "factual_type":  str(data.get("factual_type", "mixed")),
            "topics":        list(data.get("topics", [])),
            "flag_reason":   str(data.get("flag_reason", "")),
        }

    def report_to_terminal(self, result: DistillResult, cycle: int) -> None:
        """
        Push a human-readable distillation summary to stdout/terminal.
        Called after each cycle so the operator can watch corpus growth in real time.
        """
        stats = self.get_stats()
        corpus_size = stats.get("clean", 0)
        pending     = stats.get("pending_pairs", 0)
        avg_eth     = stats.get("avg_ethics", 0)
        avg_q       = stats.get("avg_quality", 0)
        by_type     = stats.get("by_type", {})

        # Milestone thresholds that trigger a more prominent message
        milestones = [100, 500, 1000, 5000, 10000]
        milestone_hit = any(
            corpus_size >= m and (corpus_size - result.passed) < m
            for m in milestones
        )

        sep = "─" * 60
        lines = [
            f"\n{sep}",
            f"[L16 DISTILLER] Cycle #{cycle}  |  {self._timestamp()}",
            f"  This cycle : {result.total_processed} processed  "
            f"{result.passed} passed  {result.flagged} flagged",
            f"  Corpus     : {corpus_size} clean pairs  "
            f"(avg ethics {avg_eth:.2f} | avg quality {avg_q:.2f})",
            f"  Pending    : {pending} pairs awaiting distillation",
            f"  By type    : {', '.join(f'{k}={v}' for k,v in by_type.items())}",
            f"  JSONL path : {self._jsonl_path}",
        ]

        if result.errors:
            lines.append(f"  Errors     : {len(result.errors)} — {result.errors[0][:80]}")

        if milestone_hit:
            lines += [
                sep,
                f"  MILESTONE  : {corpus_size} clean training pairs accumulated.",
                f"  Run the training scaffold to start a fine-tune:",
                f"    python -m cortana.tools.train_new_llm --corpus {self._jsonl_path}",
                sep,
            ]
        else:
            lines.append(sep)

        print("\n".join(lines), flush=True)

        # Also write a persistent progress log
        progress_log = self._training_dir / "distillation_log.txt"
        with open(progress_log, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _format_training_line(
        user_msg: str,
        assistant_msg: str,
        eval_result: Dict,
    ) -> str:
        """Format a conversation pair as a JSONL training line (chat format)."""
        from cortana.layers.layer1_identity import CortanaIdentity
        record = {
            "messages": [
                {"role": "system",    "content": CortanaIdentity.SYSTEM_PROMPT},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "metadata": {
                "ethics_score":  eval_result.get("ethics_score"),
                "quality_score": eval_result.get("quality_score"),
                "factual_type":  eval_result.get("factual_type"),
                "topics":        eval_result.get("topics", []),
                "distilled_at":  datetime.now(timezone.utc).isoformat(),
            },
        }
        return json.dumps(record, ensure_ascii=False)
