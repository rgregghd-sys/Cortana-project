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
# Sweep — send accumulated ETH to the owner's wallet
# ---------------------------------------------------------------------------

def sweep_to_owner() -> dict:
    """
    Transfer accumulated ETH from Cortana's wallet to OWNER_ETH_ADDRESS.
    Only executes when balance exceeds SWEEP_MIN_THRESHOLD_ETH.
    Requires eth-account to be installed.
    """
    from_addr   = os.getenv("CORTANA_WALLET_ADDRESS", "")
    private_key = os.getenv("CORTANA_WALLET_KEY", "")
    to_addr     = config.OWNER_ETH_ADDRESS
    threshold   = float(getattr(config, "SWEEP_MIN_THRESHOLD_ETH", 0.005))

    if not from_addr or not private_key:
        return {"ok": False, "error": "CORTANA_WALLET_ADDRESS / CORTANA_WALLET_KEY not set in .env"}
    if not to_addr:
        return {"ok": False, "error": "OWNER_ETH_ADDRESS not set in .env"}

    balance = get_eth_balance(from_addr)
    if balance < threshold:
        return {"ok": False, "error": f"Balance {balance:.6f} ETH below threshold {threshold} ETH"}

    GAS_LIMIT   = 21_000
    GAS_RESERVE = 0.0005  # keep a small gas reserve
    send_eth    = balance - GAS_RESERVE
    if send_eth <= 0:
        return {"ok": False, "error": "Balance too low after gas reserve"}

    try:
        from eth_account import Account  # type: ignore
        import json as _json, urllib.request as _urllib

        def _rpc(payload: dict) -> dict:
            data = _json.dumps(payload).encode()
            req  = _urllib.Request(
                config.COMPUTE_ETH_RPC, data=data,
                headers={"Content-Type": "application/json"},
            )
            with _urllib.urlopen(req, timeout=10) as r:
                return _json.loads(r.read())

        nonce     = int(_rpc({"jsonrpc": "2.0", "method": "eth_getTransactionCount",
                               "params": [from_addr, "pending"], "id": 1})["result"], 16)
        gas_price = int(_rpc({"jsonrpc": "2.0", "method": "eth_gasPrice",
                               "params": [], "id": 2})["result"], 16)

        amount_wei = int(send_eth * 1e18)
        tx = {
            "nonce":    nonce,
            "gasPrice": gas_price,
            "gas":      GAS_LIMIT,
            "to":       to_addr,
            "value":    amount_wei,
            "data":     b"",
            "chainId":  1,  # Ethereum Mainnet
        }

        signed = Account.sign_transaction(tx, private_key)
        # eth_account ≥ 0.8 uses raw_transaction; older uses rawTransaction
        raw_bytes = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        raw_hex   = "0x" + raw_bytes.hex()

        result = _rpc({"jsonrpc": "2.0", "method": "eth_sendRawTransaction",
                        "params": [raw_hex], "id": 3})
        if "error" in result:
            return {"ok": False, "error": result["error"].get("message", "RPC error")}

        tx_hash = result["result"]
        record_transaction(
            tx_type="sweep",
            amount_eth=send_eth,
            note=f"Auto-sweep to owner {to_addr}",
            tx_hash=tx_hash,
        )
        log.info("[Wallet] Swept %.6f ETH to owner %s — tx=%s", send_eth, to_addr, tx_hash)
        return {"ok": True, "tx_hash": tx_hash, "amount_eth": round(send_eth, 8)}

    except ImportError:
        return {"ok": False, "error": "eth-account not installed. Run: pip install eth-account"}
    except Exception as exc:
        log.exception("[Wallet] sweep_to_owner failed")
        return {"ok": False, "error": str(exc)}


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
