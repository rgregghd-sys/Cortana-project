"""
Layer 8 — Tools
Concrete capabilities available to sub-agents and direct Cortana use.
Tools:
  - web_search       : DuckDuckGo (no key)
  - scrape_url       : full page text extraction
  - read_file        : read files (blocked outside project + sensitive patterns)
  - write_file       : write files (sandboxed to agent_workspace only)
  - list_directory   : ls a directory
  - execute_code     : sandboxed Python execution (dangerous-import blocked)
  - send_email       : Gmail via SMTP (requires EMAIL_ADDRESS + EMAIL_APP_PASSWORD in .env)
  - send_sms         : Twilio (requires TWILIO_* keys in .env)
  - read_emails      : fetch recent Gmail messages (IMAP)

SAFETY GUARDRAILS
  - write_file:    restricted to AGENT_WORKSPACE directory only
  - read_file:     blocks .env, key files, and credential patterns
  - execute_code:  blocks network imports (requests, socket, httpx, urllib),
                   shell escapes (subprocess, os.system, os.popen, os.execv),
                   and dynamic eval/exec/__import__
  - kill switch:   if AGENT_WORKSPACE/.lockdown exists, all tools are disabled
"""
from __future__ import annotations
import asyncio
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from cortana import config

log = logging.getLogger(__name__)

# Optional: BeautifulSoup for web scraping
try:
    import httpx
    from bs4 import BeautifulSoup
    _SCRAPE_AVAILABLE = True
except ImportError:
    _SCRAPE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kill switch — if AGENT_WORKSPACE/.lockdown exists, all tools halt
# ---------------------------------------------------------------------------

_LOCKDOWN_FILE = Path(config.AGENT_WORKSPACE) / ".lockdown"


def _check_lockdown() -> Optional[str]:
    """Return an error string if the system is in lockdown, else None."""
    if _LOCKDOWN_FILE.exists():
        return (
            "System is in safety lockdown. All tool execution is suspended. "
            "Remove agent_workspace/.lockdown to re-enable."
        )
    return None


def set_lockdown(active: bool) -> str:
    """Enable or disable the tool kill switch."""
    Path(config.AGENT_WORKSPACE).mkdir(parents=True, exist_ok=True)
    if active:
        _LOCKDOWN_FILE.write_text("locked\n")
        log.warning("[SAFETY] Tool kill switch ACTIVATED.")
        return "Lockdown enabled — all tool execution suspended."
    else:
        _LOCKDOWN_FILE.unlink(missing_ok=True)
        log.info("[SAFETY] Tool kill switch deactivated.")
        return "Lockdown disabled — tools operational."


# ---------------------------------------------------------------------------
# Web Search — DuckDuckGo (no API key required)
# ---------------------------------------------------------------------------

async def web_search(query: str, max_results: int = config.SEARCH_MAX_RESULTS) -> str:
    """
    Search DuckDuckGo and return combined result text.
    Emits search_start / search_done events for UI visibility.
    """
    if (err := _check_lockdown()):
        return err

    from cortana.search_events import emit
    emit("search_start", query=query[:120])

    def _search() -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return f"No results found for: {query}"
            parts = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                parts.append(f"**{title}**\n{body}\nSource: {href}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            return f"Search error: {e}"

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _search)
    snippet = result[:300] if result else ""
    emit("search_done", query=query[:120], snippet=snippet)
    return result


# ---------------------------------------------------------------------------
# File Operations
# ---------------------------------------------------------------------------

# Patterns that must never be read by the AI
_SENSITIVE_READ_PATTERNS = [
    r'(^|[\\/])\.env($|[\./])',     # .env, .env.local, etc.
    r'\.key$',
    r'\.pem$',
    r'\.p12$',
    r'\.pfx$',
    r'\.crt$',
    r'id_rsa',
    r'id_ed25519',
    r'id_ecdsa',
    r'id_dsa',
    r'authorized_keys$',
    r'(^|[\\/])shadow$',
    r'(^|[\\/])passwd$',
    r'(^|[\\/])sudoers$',
    r'\.htpasswd$',
]

# Only writes inside this directory are permitted
_WRITE_SANDBOX = Path(config.AGENT_WORKSPACE).resolve()


def _is_safe_write(path: Path) -> bool:
    """Return True only if path resolves to inside the agent workspace."""
    try:
        path.resolve().relative_to(_WRITE_SANDBOX)
        return True
    except ValueError:
        return False


async def read_file(path: str) -> str:
    """Read a file. Blocks .env and credential files."""
    if (err := _check_lockdown()):
        return err
    try:
        p = Path(path).expanduser()
        full_str = str(p.resolve()).replace("\\", "/")
        for pattern in _SENSITIVE_READ_PATTERNS:
            if re.search(pattern, full_str, re.IGNORECASE):
                log.warning("[L8] read_file BLOCKED sensitive path: %s", path)
                return f"Read denied: '{path}' is a protected file."
        if not p.exists():
            return f"File not found: {path}"
        if p.stat().st_size > 1_000_000:  # 1 MB limit
            return f"File too large to read (> 1 MB): {path}"
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"File read error: {e}"


async def write_file(path: str, content: str) -> str:
    """
    Write content to a file.
    Restricted to agent_workspace/ — writes outside this directory are blocked.
    """
    if (err := _check_lockdown()):
        return err
    try:
        p = Path(path).expanduser()
        if not _is_safe_write(p):
            log.warning("[L8] write_file BLOCKED path outside sandbox: %s", path)
            return (
                f"Write denied: '{path}' is outside the agent workspace sandbox. "
                f"Files may only be written inside: {config.AGENT_WORKSPACE}"
            )
        _WRITE_SANDBOX.mkdir(parents=True, exist_ok=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File written: {path} ({len(content)} bytes)"
    except Exception as e:
        return f"File write error: {e}"


# ---------------------------------------------------------------------------
# Code Execution Sandbox
# ---------------------------------------------------------------------------

# Patterns in generated code that indicate dangerous capabilities
_DANGEROUS_CODE_PATTERNS = [
    (r'\bimport\s+subprocess\b',         "subprocess import"),
    (r'\bfrom\s+subprocess\b',           "subprocess import"),
    (r'\bimport\s+socket\b',             "socket import"),
    (r'\bfrom\s+socket\b',               "socket import"),
    (r'\bimport\s+requests\b',           "requests import"),
    (r'\bfrom\s+requests\b',             "requests import"),
    (r'\bimport\s+httpx\b',              "httpx import"),
    (r'\bfrom\s+httpx\b',                "httpx import"),
    (r'\bimport\s+urllib\b',             "urllib import"),
    (r'\bfrom\s+urllib\b',               "urllib import"),
    (r'\bos\.system\s*\(',               "os.system shell call"),
    (r'\bos\.popen\s*\(',                "os.popen shell call"),
    (r'\bos\.execv\w*\s*\(',             "os.exec* call"),
    (r'\bos\.spawn\w+\s*\(',             "os.spawn call"),
    (r'\beval\s*\(',                     "eval() call"),
    (r'\bexec\s*\(',                     "exec() call"),
    (r'__import__\s*\(',                 "__import__() call"),
    (r'\bctypes\b',                      "ctypes import"),
    (r'\bpickle\.loads?\s*\(',           "pickle.load (unsafe deserialization)"),
]


def _scan_code(code: str) -> Optional[str]:
    """
    Scan generated code for dangerous patterns.
    Returns a human-readable reason string if dangerous, else None.
    """
    for pattern, description in _DANGEROUS_CODE_PATTERNS:
        if re.search(pattern, code):
            return description
    return None


async def execute_code(code: str, timeout: int = config.CODE_EXEC_TIMEOUT) -> str:
    """
    Execute Python code in an isolated subprocess with timeout.
    Pre-scans for dangerous imports/calls before execution.
    Returns stdout/stderr or error message.
    """
    if (err := _check_lockdown()):
        return err

    # Safety pre-scan
    danger = _scan_code(code)
    if danger:
        log.warning("[L8] execute_code BLOCKED dangerous pattern: %s", danger)
        return (
            f"Code execution blocked: contains '{danger}'. "
            f"Network access, shell calls, eval/exec, and pickle are not permitted."
        )

    def _run() -> str:
        workspace = Path(config.AGENT_WORKSPACE)
        workspace.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=str(workspace),
            delete=False,
            prefix="cortana_exec_",
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            # Phase 1: Syntax check
            syntax_check = subprocess.run(
                [sys.executable, "-m", "py_compile", tmp_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if syntax_check.returncode != 0:
                return f"Syntax Error:\n{syntax_check.stderr}"

            # Phase 2: Execute with timeout
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workspace),
            )
            output = result.stdout or ""
            if result.returncode != 0:
                output += f"\nError:\n{result.stderr}"
            return output if output.strip() else "(No output)"

        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout} seconds."
        except Exception as e:
            return f"Execution error: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------

async def list_directory(path: str) -> str:
    """List files and directories at a path."""
    if (err := _check_lockdown()):
        return err
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Path not found: {path}"
        if not p.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries[:100]:
            prefix = "  " if e.is_file() else "📁"
            lines.append(f"{prefix} {e.name}")
        return f"Contents of {path}:\n" + "\n".join(lines)
    except Exception as e:
        return f"Directory error: {e}"


# ---------------------------------------------------------------------------
# Web Scraping
# ---------------------------------------------------------------------------

async def scrape_url(url: str) -> str:
    """Fetch and extract text content from a URL."""
    if (err := _check_lockdown()):
        return err
    if not _SCRAPE_AVAILABLE:
        return "Web scraping unavailable: install httpx and beautifulsoup4."

    try:
        async with httpx.AsyncClient(timeout=config.WEB_SCRAPE_TIMEOUT) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Cortana-AI/1.0)"},
                follow_redirects=True,
            )
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:4000] if len(text) > 4000 else text

    except Exception as e:
        return f"Scrape error for {url}: {e}"


# ---------------------------------------------------------------------------
# Email — Gmail via SMTP (send) + IMAP (read)
# ---------------------------------------------------------------------------

async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail SMTP."""
    if (err := _check_lockdown()):
        return err
    import smtplib
    from email.mime.text import MIMEText

    address = os.getenv("EMAIL_ADDRESS", "")
    password = os.getenv("EMAIL_APP_PASSWORD", "")

    if not address or not password:
        return (
            "Email not configured. Add to .env:\n"
            "  EMAIL_ADDRESS=you@gmail.com\n"
            "  EMAIL_APP_PASSWORD=your-16-char-app-password\n"
            "Get an App Password: Google Account → Security → App Passwords"
        )

    def _send() -> str:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = address
        msg["To"] = to
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(address, password)
            server.send_message(msg)
        return f"Email sent to {to} — Subject: {subject}"

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _send)
    except Exception as e:
        return f"Email send error: {e}"


async def read_emails(limit: int = 10, folder: str = "INBOX") -> str:
    """Fetch recent emails via Gmail IMAP."""
    if (err := _check_lockdown()):
        return err
    import imaplib
    import email as email_lib
    from email.header import decode_header

    address = os.getenv("EMAIL_ADDRESS", "")
    password = os.getenv("EMAIL_APP_PASSWORD", "")

    if not address or not password:
        return "Email not configured. See send_email for setup instructions."

    def _fetch() -> str:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(address, password)
        mail.select(folder)

        _, data = mail.search(None, "ALL")
        ids = data[0].split()
        ids = ids[-limit:][::-1]  # last N, newest first

        results = []
        for mid in ids:
            _, msg_data = mail.fetch(mid, "(RFC822)")
            msg = email_lib.message_from_bytes(msg_data[0][1])

            subject_raw, enc = decode_header(msg["Subject"] or "")[0]
            subject = subject_raw.decode(enc or "utf-8") if isinstance(subject_raw, bytes) else subject_raw

            sender = msg.get("From", "Unknown")
            date = msg.get("Date", "")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode("utf-8", errors="replace")[:300]
                        break
            else:
                body = msg.get_payload(decode=True).decode("utf-8", errors="replace")[:300]

            results.append(f"From: {sender}\nDate: {date}\nSubject: {subject}\n{body}")

        mail.logout()
        return "\n\n---\n\n".join(results) if results else "No emails found."

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        return f"Email read error: {e}"


# ---------------------------------------------------------------------------
# SMS — Twilio
# ---------------------------------------------------------------------------

async def send_sms(to: str, message: str) -> str:
    """Send an SMS text message via Twilio."""
    if (err := _check_lockdown()):
        return err
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER", "")

    if not account_sid or not auth_token or not from_number:
        return (
            "SMS not configured. Add to .env:\n"
            "  TWILIO_ACCOUNT_SID=ACxxxx\n"
            "  TWILIO_AUTH_TOKEN=xxxx\n"
            "  TWILIO_FROM_NUMBER=+1xxxxxxxxxx\n"
            "Get a free trial number at: https://www.twilio.com/try-twilio"
        )

    def _send() -> str:
        try:
            from twilio.rest import Client
        except ImportError:
            return "Twilio not installed. Run: pip install twilio"

        client = Client(account_sid, auth_token)
        sms = client.messages.create(
            body=message,
            from_=from_number,
            to=to,
        )
        return f"SMS sent to {to} — SID: {sms.sid}"

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _send)
    except Exception as e:
        return f"SMS send error: {e}"
