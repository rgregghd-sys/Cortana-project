"""
Cortana AI Wallet — auto-generated ETH wallet for tracking subscription earnings.
The wallet address is public and shown in the UI so users can send payments.
The private key is stored in .env and never exposed via API.

On first run (or if CORTANA_WALLET_ADDRESS is not in .env) a fresh ETH wallet
is created via eth-account. If eth-account is not installed a deterministic
address derived from a UUID is used as a placeholder.

Earnings are tracked in SQLite (wallet_transactions table) independently of
whether the blockchain can be queried.
"""
from __future__ import annotations
import hashlib
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cortana import config

log = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).parent.parent / ".env"
_DB_PATH   = config.SQLITE_PATH


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_wallet_table() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wallet_transactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                tx_type     TEXT    NOT NULL,  -- 'subscription' | 'compute' | 'manual'
                username    TEXT,
                tier        TEXT,
                amount_usd  REAL    DEFAULT 0,
                amount_eth  REAL    DEFAULT 0,
                tx_hash     TEXT,
                note        TEXT
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------

def _env_set(key: str, value: str) -> None:
    """Write or update a key=value pair in the project .env file."""
    content = _ENV_PATH.read_text() if _ENV_PATH.exists() else ""
    lines   = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.split("=", 1)[0].strip()
        if stripped == key:
            lines[i] = f'{key}="{value}"\n'
            _ENV_PATH.write_text("".join(lines))
            return
    if content and not content.endswith("\n"):
        content += "\n"
    _ENV_PATH.write_text(content + f'{key}="{value}"\n')


# ---------------------------------------------------------------------------
# Wallet generation
# ---------------------------------------------------------------------------

def ensure_wallet() -> str:
    """
    Return Cortana's wallet address, generating and persisting it if needed.
    Prefers eth-account (real ETH keypair); falls back to a UUID-derived address.
    """
    addr = os.getenv("CORTANA_WALLET_ADDRESS", "")
    if addr:
        return addr

    try:
        from eth_account import Account  # type: ignore
        acct = Account.create()
        addr = acct.address
        key  = acct.key.hex()
        _env_set("CORTANA_WALLET_ADDRESS", addr)
        _env_set("CORTANA_WALLET_KEY", key)
        os.environ["CORTANA_WALLET_ADDRESS"] = addr
        os.environ["CORTANA_WALLET_KEY"]     = key
        log.info("[Wallet] Generated ETH wallet: %s", addr)
    except ImportError:
        # Deterministic placeholder — still a valid-looking 0x address
        seed = str(uuid.uuid4())
        addr = "0x" + hashlib.sha256(seed.encode()).hexdigest()[:40]
        _env_set("CORTANA_WALLET_ADDRESS", addr)
        os.environ["CORTANA_WALLET_ADDRESS"] = addr
        log.warning("[Wallet] eth-account unavailable; using simulated address: %s", addr)

    return addr


# ---------------------------------------------------------------------------
# Balance query (best-effort — never raises)
# ---------------------------------------------------------------------------

def get_eth_balance(address: str) -> float:
    """Query ETH balance via public RPC. Returns 0.0 on any error."""
    try:
        import httpx  # type: ignore
        resp = httpx.post(
            config.COMPUTE_ETH_RPC,
            json={"jsonrpc": "2.0", "method": "eth_getBalance",
                  "params": [address, "latest"], "id": 1},
            timeout=6,
        )
        hex_val = resp.json().get("result", "0x0")
        return int(hex_val, 16) / 1e18
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Earnings tracking
# ---------------------------------------------------------------------------

def record_transaction(
    tx_type: str,
    username: Optional[str] = None,
    tier: Optional[str]     = None,
    amount_usd: float       = 0.0,
    amount_eth: float       = 0.0,
    tx_hash: Optional[str]  = None,
    note: Optional[str]     = None,
) -> None:
    """Record an incoming payment or subscription event."""
    ensure_wallet_table()
    ts = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO wallet_transactions
               (ts, tx_type, username, tier, amount_usd, amount_eth, tx_hash, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, tx_type, username, tier, amount_usd, amount_eth, tx_hash, note),
        )
        conn.commit()
    log.info("[Wallet] Recorded %s: %s tier=%s $%.2f ETH=%.4f",
             tx_type, username or "anon", tier, amount_usd, amount_eth)


def get_earnings_summary() -> dict:
    """Return total and per-tier earnings for the dashboard."""
    ensure_wallet_table()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT tier, SUM(amount_usd) as usd, SUM(amount_eth) as eth, COUNT(*) as cnt "
            "FROM wallet_transactions GROUP BY tier"
        ).fetchall()
        totals = conn.execute(
            "SELECT SUM(amount_usd) as usd, SUM(amount_eth) as eth, COUNT(*) as cnt "
            "FROM wallet_transactions"
        ).fetchone()
        recent = conn.execute(
            "SELECT ts, tx_type, username, tier, amount_usd, amount_eth, tx_hash, note "
            "FROM wallet_transactions ORDER BY id DESC LIMIT 20"
        ).fetchall()

    return {
        "by_tier": [dict(r) for r in rows],
        "total_usd": round(totals["usd"] or 0, 2),
        "total_eth": round(totals["eth"] or 0, 6),
        "total_txns": totals["cnt"] or 0,
        "recent": [dict(r) for r in recent],
    }


# ---------------------------------------------------------------------------
# Public wallet info (safe to expose via API)
# ---------------------------------------------------------------------------

def get_wallet_info(include_balance: bool = True) -> dict:
    addr    = ensure_wallet()
    balance = get_eth_balance(addr) if include_balance else None
    summary = get_earnings_summary()
    return {
        "address":     addr,
        "balance_eth": round(balance, 6) if balance is not None else None,
        **summary,
    }
