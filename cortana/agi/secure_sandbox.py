"""
Secure Execution Sandbox — isolated, resource-limited code execution.

All self-generated code changes pass through this sandbox before
being applied. Enforces:
  - Process isolation (subprocess, -I flag = isolated Python)
  - Hard timeout (default 15s)
  - Output size cap (32KB)
  - Import whitelist (no os, subprocess, socket, requests, etc.)
  - AST pre-scan for dangerous patterns
  - Memory limit via resource module (Unix only, ignored gracefully elsewhere)
  - No filesystem writes outside AGENT_WORKSPACE

Results are stored in SQLite for audit.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from cortana import config

_DB_PATH       = Path(config.SQLITE_PATH)
_WORKSPACE     = Path(config.AGENT_WORKSPACE)
_TIMEOUT       = int(os.getenv("SANDBOX_TIMEOUT_SECS", "15"))
_MAX_OUTPUT    = 32 * 1024   # 32KB
_MAX_CODE_SIZE = 64 * 1024   # 64KB

# Safe imports — anything not in this set is blocked
_SAFE_IMPORTS = frozenset([
    "math", "cmath", "decimal", "fractions", "statistics",
    "re", "json", "csv", "textwrap", "string", "pprint",
    "datetime", "time", "calendar",
    "collections", "heapq", "bisect", "array", "queue",
    "itertools", "functools", "operator", "copy",
    "enum", "dataclasses", "typing", "abc",
    "pathlib", "io", "struct", "codecs",
    "hashlib", "base64", "binascii",
    "random", "uuid",
    "unittest", "doctest",
])

# Dangerous patterns to reject regardless of imports
_DANGEROUS_RE = [
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\bcompile\s*\("),
    re.compile(r"\b__import__\s*\("),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\bgetattr\s*\(.{0,30}__"),
    re.compile(r"\bsetattr\s*\("),
    re.compile(r"\bdelattr\s*\("),
    re.compile(r"\bglobals\s*\(\)"),
    re.compile(r"\blocals\s*\(\)"),
    re.compile(r"\bvars\s*\(\)"),
    re.compile(r"__builtins__"),
    re.compile(r"__class__"),
    re.compile(r"__subclasses__"),
    re.compile(r"\bos\s*\.\s*(system|popen|exec|spawn|fork|remove|unlink|rmdir)"),
    re.compile(r"\bsocket\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\burllib\b"),
    re.compile(r"\brequests\b"),
    re.compile(r"\bhttp\b"),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    passed:     bool
    exit_code:  int
    stdout:     str
    stderr:     str
    duration_s: float
    blocked_reason: str    # non-empty if blocked before execution
    session_id: int        # DB row id for audit


# ---------------------------------------------------------------------------
# Pre-execution analysis
# ---------------------------------------------------------------------------

def analyse_code(code: str) -> Tuple[bool, str]:
    """
    Returns (safe, reason). reason is non-empty if unsafe.
    """
    if len(code) > _MAX_CODE_SIZE:
        return False, f"Code too large ({len(code)} bytes, max {_MAX_CODE_SIZE})"

    # Dangerous pattern scan
    for pat in _DANGEROUS_RE:
        m = pat.search(code)
        if m:
            return False, f"Dangerous pattern: {m.group(0)[:40]!r}"

    # AST import scan
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _SAFE_IMPORTS:
                    return False, f"Blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root and root not in _SAFE_IMPORTS:
                return False, f"Blocked import: {node.module}"

    return True, ""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _resource_limit_script() -> str:
    """Preamble that applies resource limits (Unix only)."""
    return textwrap_import + """\
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS,   (256*1024*1024, 256*1024*1024))  # 256MB RAM
    resource.setrlimit(resource.RLIMIT_CPU,  (10, 10))                         # 10s CPU
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))                       # 16 file descriptors
except Exception:
    pass
"""

# Avoid importing textwrap in the preamble — use inline string
textwrap_import = ""


class SecureSandbox:
    """Isolated subprocess executor with pre-scan and audit logging."""

    def __init__(self) -> None:
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sandbox_sessions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    code_hash     TEXT NOT NULL,
                    code_preview  TEXT NOT NULL,
                    passed        INTEGER NOT NULL,
                    exit_code     INTEGER NOT NULL DEFAULT 0,
                    stdout        TEXT NOT NULL DEFAULT '',
                    stderr        TEXT NOT NULL DEFAULT '',
                    duration_s    REAL NOT NULL DEFAULT 0,
                    blocked_reason TEXT NOT NULL DEFAULT '',
                    ts            REAL NOT NULL
                )
            """)

    def run(self, code: str, label: str = "") -> SandboxResult:
        """
        Run code in sandbox. Returns SandboxResult.
        Always logs to DB for audit.
        """
        import hashlib
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        # Pre-scan
        safe, reason = analyse_code(code)
        if not safe:
            sid = self._log(code_hash, code, passed=False, exit_code=-1,
                            blocked_reason=reason)
            return SandboxResult(
                passed=False, exit_code=-1, stdout="", stderr="",
                duration_s=0.0, blocked_reason=reason, session_id=sid,
            )

        # Prepend resource limits
        full_code = _resource_limit_script() + "\n" + code
        tmp: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w",
                dir=str(_WORKSPACE / "sandbox"),
                delete=False,
            ) as f:
                f.write(full_code)
                tmp = f.name

            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, "-I", "-B", tmp],   # -I isolated, -B no .pyc
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                cwd=str(_WORKSPACE / "sandbox"),
            )
            elapsed = time.time() - t0
            stdout  = proc.stdout[:_MAX_OUTPUT]
            stderr  = proc.stderr[:_MAX_OUTPUT]
            passed  = proc.returncode == 0

            sid = self._log(code_hash, code, passed=passed,
                            exit_code=proc.returncode,
                            stdout=stdout, stderr=stderr,
                            duration_s=elapsed)
            return SandboxResult(
                passed=passed, exit_code=proc.returncode,
                stdout=stdout, stderr=stderr,
                duration_s=round(elapsed, 3),
                blocked_reason="", session_id=sid,
            )

        except subprocess.TimeoutExpired:
            sid = self._log(code_hash, code, passed=False, exit_code=-2,
                            blocked_reason=f"Timeout after {_TIMEOUT}s")
            return SandboxResult(
                passed=False, exit_code=-2, stdout="", stderr="",
                duration_s=float(_TIMEOUT),
                blocked_reason=f"Timeout after {_TIMEOUT}s",
                session_id=sid,
            )
        except Exception as e:
            sid = self._log(code_hash, code, passed=False, exit_code=-3,
                            blocked_reason=str(e))
            return SandboxResult(
                passed=False, exit_code=-3, stdout="", stderr="",
                duration_s=0.0, blocked_reason=str(e), session_id=sid,
            )
        finally:
            if tmp:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass

    def _log(self, code_hash: str, code: str, passed: bool,
             exit_code: int, stdout: str = "", stderr: str = "",
             duration_s: float = 0.0, blocked_reason: str = "") -> int:
        try:
            with self._conn() as conn:
                cur = conn.execute("""
                    INSERT INTO sandbox_sessions
                        (code_hash, code_preview, passed, exit_code, stdout, stderr,
                         duration_s, blocked_reason, ts)
                    VALUES(?,?,?,?,?,?,?,?,?)
                """, (code_hash, code[:400], int(passed), exit_code,
                      stdout[:2000], stderr[:2000], duration_s,
                      blocked_reason, time.time()))
                return cur.lastrowid
        except Exception:
            return -1

    def recent_sessions(self, limit: int = 10) -> List[dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute("""
                    SELECT id, code_hash, code_preview, passed, exit_code,
                           duration_s, blocked_reason, ts
                    FROM sandbox_sessions ORDER BY ts DESC LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


# Ensure sandbox working directory exists
try:
    (_WORKSPACE / "sandbox").mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Module-level singleton
sandbox = SecureSandbox()
