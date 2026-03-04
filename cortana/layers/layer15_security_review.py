"""
Layer 15 — AI Security Code Review

Scans the Cortana codebase for vulnerabilities using static analysis,
then uses the LLM to do a deeper review of the top suspicious patterns.

Endpoints (mounted by layer13_chat.py):
  GET  /api/security/report  — cached latest report
  POST /api/security/scan    — trigger fresh scan

Usage:
  from cortana.layers.layer15_security_review import get_security_router
  app.include_router(get_security_router())
"""
from __future__ import annotations

import ast
import logging
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SCAN_ROOT    = _PROJECT_ROOT / "cortana"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SecurityFinding:
    file: str
    line: int
    severity: str          # critical | high | medium | low | info
    category: str
    description: str
    recommendation: str


@dataclass
class SecurityReport:
    findings: List[SecurityFinding] = field(default_factory=list)
    score: int = 100
    summary: str = ""
    generated_at: str = ""
    llm_review: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Static analysis rules
# ---------------------------------------------------------------------------

# (pattern, category, severity, description, recommendation)
_REGEX_RULES = [
    (
        re.compile(r'(?i)(api_key|secret_key|password|token|passwd)\s*=\s*["\'][^"\']{6,}["\']'),
        "hardcoded_secret", "critical",
        "Possible hardcoded secret/credential",
        "Move credentials to environment variables or a secrets manager.",
    ),
    (
        re.compile(r'subprocess\.(run|call|Popen|check_output)\s*\([^)]*shell\s*=\s*True'),
        "command_injection", "high",
        "subprocess called with shell=True — enables shell injection",
        "Pass commands as a list and avoid shell=True.",
    ),
    (
        re.compile(r'\bos\.system\s*\('),
        "command_injection", "high",
        "os.system() passes command string to shell — injection risk",
        "Use subprocess with a list argument instead.",
    ),
    (
        re.compile(r'\bos\.popen\s*\('),
        "command_injection", "medium",
        "os.popen() executes shell command",
        "Use subprocess.run() with a list argument.",
    ),
    (
        re.compile(r'\bpickle\.loads?\s*\('),
        "insecure_deserialization", "high",
        "pickle.load/loads deserializes arbitrary data — RCE risk with untrusted input",
        "Use JSON or a safe serialization format for untrusted data.",
    ),
    (
        re.compile(r'\byaml\.load\s*\([^,)]*\)(?!\s*,\s*Loader)'),
        "insecure_deserialization", "high",
        "yaml.load() without Loader= can execute arbitrary Python",
        "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader).",
    ),
    (
        re.compile(r'hashlib\.(md5|sha1)\s*\('),
        "weak_crypto", "medium",
        "MD5/SHA1 are cryptographically weak; not suitable for password hashing",
        "Use bcrypt, argon2, or hashlib.sha256+PBKDF2 for password hashing.",
    ),
    (
        re.compile(r'open\s*\(\s*[^)]*\+[^)]*\)'),
        "path_traversal", "medium",
        "open() with string concatenation may allow path traversal",
        "Validate and sanitize file paths; use pathlib.Path.resolve().",
    ),
    (
        re.compile(r'(?i)@app\.(get|post|put|delete)\s*\(\s*["\'].*?(debug|test|dev)[^"\']*["\']'),
        "debug_endpoint", "low",
        "Possible debug/test endpoint exposed in production",
        "Remove or protect debug routes behind auth/feature flags in production.",
    ),
    (
        re.compile(r'random\.(random|randint|choice|uniform)\s*\('),
        "weak_random", "low",
        "random module is not cryptographically secure",
        "Use secrets module for security-sensitive randomness (tokens, nonces).",
    ),
    (
        re.compile(r'eval\s*\('),
        "code_execution", "critical",
        "eval() executes arbitrary code",
        "Never eval() untrusted input. Use ast.literal_eval() for safe literals.",
    ),
    (
        re.compile(r'exec\s*\('),
        "code_execution", "high",
        "exec() executes arbitrary code",
        "Avoid exec() with any user-controlled or external data.",
    ),
]

_SEVERITY_WEIGHT = {"critical": 25, "high": 15, "medium": 8, "low": 3, "info": 1}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class SecurityScanner:
    def __init__(self) -> None:
        self._last_report: Optional[SecurityReport] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def scan(self) -> SecurityReport:
        findings: List[SecurityFinding] = []
        py_files = list(_SCAN_ROOT.rglob("*.py"))

        for path in py_files:
            rel = str(path.relative_to(_PROJECT_ROOT))
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Regex scan line-by-line
            for lineno, line in enumerate(src.splitlines(), start=1):
                for pattern, cat, sev, desc, rec in _REGEX_RULES:
                    if pattern.search(line):
                        findings.append(SecurityFinding(
                            file=rel, line=lineno, severity=sev,
                            category=cat, description=desc,
                            recommendation=rec,
                        ))

            # AST-based: SQL injection via string formatting in cursor.execute
            try:
                tree = ast.parse(src, filename=str(path))
                findings.extend(self._ast_sql_check(tree, rel))
            except SyntaxError:
                pass

        # De-duplicate (same file+line+category)
        seen: set = set()
        unique: List[SecurityFinding] = []
        for f in findings:
            key = (f.file, f.line, f.category)
            if key not in seen:
                seen.add(key)
                unique.append(f)

        # Score: deduct points per finding
        deduction = sum(_SEVERITY_WEIGHT.get(f.severity, 0) for f in unique)
        score = max(0, 100 - deduction)

        # LLM deep review of top 3 most severe findings
        llm_review = self._llm_review(unique[:3]) if unique else "No significant findings."

        # Summary
        counts = {}
        for f in unique:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        count_str = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        summary = (
            f"Scanned {len(py_files)} Python files. "
            f"Found {len(unique)} issue(s): {count_str or 'none'}. "
            f"Security score: {score}/100."
        )

        report = SecurityReport(
            findings=unique,
            score=score,
            summary=summary,
            generated_at=datetime.now(timezone.utc).isoformat(),
            llm_review=llm_review,
        )
        with self._lock:
            self._last_report = report
        logger.info("[L15] Scan complete: %d findings, score=%d", len(unique), score)
        return report

    def get_last_report(self) -> Optional[SecurityReport]:
        with self._lock:
            return self._last_report

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ast_sql_check(tree: ast.AST, rel_path: str) -> List[SecurityFinding]:
        """Find cursor.execute() calls that use % or .format() string building."""
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "execute"):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            # String % formatting:  "SELECT ... %s" % values
            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                findings.append(SecurityFinding(
                    file=rel_path, line=node.lineno, severity="high",
                    category="sql_injection",
                    description="cursor.execute() with % string formatting — SQL injection risk",
                    recommendation="Use parameterized queries: cursor.execute(sql, (values,))",
                ))
            # .format() on a string literal
            if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                    and arg.func.attr == "format"):
                findings.append(SecurityFinding(
                    file=rel_path, line=node.lineno, severity="high",
                    category="sql_injection",
                    description="cursor.execute() with .format() — SQL injection risk",
                    recommendation="Use parameterized queries: cursor.execute(sql, (values,))",
                ))
        return findings

    @staticmethod
    def _llm_review(top_findings: List[SecurityFinding]) -> str:
        """Send top findings to the LLM for deeper security analysis."""
        if not top_findings:
            return "No significant findings to review."
        try:
            from cortana.providers.router import ProviderRouter
            router = ProviderRouter()
            snippet = "\n".join(
                f"- [{f.severity.upper()}] {f.category} in {f.file}:{f.line} — {f.description}"
                for f in top_findings
            )
            prompt = (
                "You are a cybersecurity expert reviewing an AI assistant codebase.\n"
                "The static scanner found these potential issues:\n\n"
                f"{snippet}\n\n"
                "Provide a concise (3-5 sentence) expert assessment: "
                "Are these true positives? What is the actual risk in context? "
                "What is the single most important fix to make first?"
            )
            return router.think_simple(prompt, max_tokens=512)
        except Exception as e:
            return f"LLM review unavailable: {e}"


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

def get_security_router() -> APIRouter:
    router = APIRouter(prefix="/api/security", tags=["security"])

    @router.get("/report")
    async def get_report():
        report = scanner.get_last_report()
        if report is None:
            return JSONResponse(status_code=202, content={"status": "scan_pending"})
        return report.as_dict()

    @router.post("/scan")
    async def trigger_scan():
        import asyncio
        report = await asyncio.to_thread(scanner.scan)
        return report.as_dict()

    return router


# ---------------------------------------------------------------------------
# Module-level scanner — starts background scan on import
# ---------------------------------------------------------------------------

scanner = SecurityScanner()

def _bg_scan():
    time.sleep(5)   # let the server finish starting up first
    try:
        scanner.scan()
    except Exception as e:
        logger.warning("[L15] Background scan failed: %s", e)

threading.Thread(target=_bg_scan, daemon=True, name="security-scan").start()
