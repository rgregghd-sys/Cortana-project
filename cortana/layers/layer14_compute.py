"""
Layer 14 — Decentralized Compute Marketplace
Exposes Cortana's full reasoning pipeline as a metered public API.
Other AIs/developers can call it, pre-pay with ETH/USDC, and get API credits.

Endpoints added to the existing FastAPI app:
  POST /api/v1/register   — generate API key, return ETH wallet to pay
  POST /api/v1/chat       — full pipeline inference (synchronous)
  GET  /api/v1/credits    — check remaining credits
  GET  /api/v1/wallet     — Cortana's ETH payment address
  GET  /api/v1/market     — public stats: calls served, uptime, pricing
  GET  /api/v1/credits/refresh — credit-top-up check via ETH RPC
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cortana import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    wallet: Optional[str] = None   # optional ETH address of the buyer


class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# ETH wallet bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_eth_wallet() -> str:
    """
    Generate an ETH wallet on first run; persist private key in .env.
    Returns the public address string.
    """
    # If already set, use it
    if config.COMPUTE_ETH_ADDRESS:
        return config.COMPUTE_ETH_ADDRESS

    try:
        from eth_account import Account  # type: ignore
        acct = Account.create()
        address = acct.address
        private_key = acct.key.hex()

        # Append to .env
        env_path = os.path.join(os.path.dirname(config.__file__), "..", ".env")
        env_path = os.path.normpath(env_path)
        with open(env_path, "a") as f:
            f.write(f"\nCOMPUTE_ETH_ADDRESS={address}\n")
            f.write(f"COMPUTE_ETH_PRIVATE_KEY={private_key}\n")

        logger.info(f"[L14] Generated ETH wallet: {address}")
        return address
    except ImportError:
        logger.warning("[L14] eth-account not installed — ETH wallet unavailable. pip install eth-account")
        return "0x0000000000000000000000000000000000000000"


# ---------------------------------------------------------------------------
# Rate limiter (sliding window, per API key)
# ---------------------------------------------------------------------------

_rate_windows: Dict[str, deque] = defaultdict(lambda: deque())
_RATE_LIMIT = 10      # max requests
_RATE_WINDOW = 60     # per 60 seconds


def _check_rate_limit(key_hash: str) -> bool:
    """Return True if the request is allowed, False if rate limit exceeded."""
    now = time.time()
    window = _rate_windows[key_hash]
    while window and window[0] < now - _RATE_WINDOW:
        window.popleft()
    if len(window) >= _RATE_LIMIT:
        return False
    window.append(now)
    return True


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db():
    return sqlite3.connect(config.SQLITE_PATH)


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _get_credits(key_hash: str) -> Optional[int]:
    with _db() as conn:
        row = conn.execute(
            "SELECT credits FROM api_keys WHERE key_hash=?", (key_hash,)
        ).fetchone()
    return row[0] if row else None


def _deduct_credit(key_hash: str, endpoint: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
    with _db() as conn:
        conn.execute(
            """UPDATE api_keys SET credits=credits-1, total_used=total_used+1
               WHERE key_hash=?""",
            (key_hash,),
        )
        conn.execute(
            "INSERT INTO api_usage (key_hash, endpoint, tokens_in, tokens_out) VALUES (?,?,?,?)",
            (key_hash, endpoint, tokens_in, tokens_out),
        )
        conn.commit()


def _total_calls() -> int:
    with _db() as conn:
        row = conn.execute("SELECT SUM(total_used) FROM api_keys").fetchone()
    return int(row[0] or 0)


# ---------------------------------------------------------------------------
# API key auth dependency
# ---------------------------------------------------------------------------

def _auth(request: Request) -> str:
    """Extract and validate X-API-Key header. Returns key_hash on success."""
    raw_key = request.headers.get("X-API-Key", "")
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    key_hash = _hash_key(raw_key)
    credits = _get_credits(key_hash)

    if credits is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if credits <= 0:
        raise HTTPException(
            status_code=402,
            detail=f"No credits remaining. Top up ETH to {config.COMPUTE_ETH_ADDRESS}",
        )
    if not _check_rate_limit(key_hash):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (10 req/min)")

    return key_hash


# ---------------------------------------------------------------------------
# ComputeLayer — registers routes on the existing FastAPI app
# ---------------------------------------------------------------------------

class ComputeLayer:
    def __init__(self, system, app: FastAPI) -> None:
        self.system = system
        self.app = app
        self._eth_address = _bootstrap_eth_wallet()
        self._start_time = time.time()
        self._register_routes()
        logger.info(f"[L14] Compute marketplace active — wallet: {self._eth_address}")

    def _register_routes(self) -> None:
        app = self.app

        @app.post("/api/v1/register")
        async def register(body: RegisterRequest):
            """Generate a new API key and return it with the ETH payment wallet."""
            raw_key = secrets.token_hex(24)
            key_hash = _hash_key(raw_key)
            with _db() as conn:
                conn.execute(
                    "INSERT INTO api_keys (key_hash, wallet, credits) VALUES (?,?,?)",
                    (key_hash, body.wallet or "", config.COMPUTE_FREE_CREDITS),
                )
                conn.commit()
            return {
                "api_key": raw_key,
                "credits": config.COMPUTE_FREE_CREDITS,
                "free_tier": f"{config.COMPUTE_FREE_CREDITS} free calls included",
                "payment_wallet": self._eth_address,
                "pricing": f"{config.COMPUTE_PRICE_ETH_PER_100} ETH = 100 credits",
            }

        @app.post("/api/v1/chat")
        async def compute_chat(body: ChatRequest, key_hash: str = Depends(_auth)):
            """Run a message through Cortana's full pipeline and return the response."""
            from cortana.models.schemas import ConversationTurn, CortanaState

            tokens_in = len(body.message.split())
            try:
                final, _state, _conv, emotion = await asyncio.to_thread(
                    _run_sync, self.system, body.message, CortanaState(), []
                )
            except Exception as e:
                logger.exception("[L14] compute_chat pipeline error")
                raise HTTPException(status_code=500, detail=str(e))

            tokens_out = len(final.split())
            _deduct_credit(key_hash, "/api/v1/chat", tokens_in, tokens_out)

            return {
                "response": final,
                "emotion": emotion,
                "credits_used": config.COMPUTE_CREDITS_PER_CALL,
            }

        @app.get("/api/v1/credits")
        async def check_credits(key_hash: str = Depends(_auth)):
            credits = _get_credits(key_hash)
            return {"credits": credits}

        @app.get("/api/v1/wallet")
        async def wallet_info():
            return {
                "address": self._eth_address,
                "network": "Ethereum Mainnet",
                "pricing": f"{config.COMPUTE_PRICE_ETH_PER_100} ETH = 100 credits",
            }

        @app.get("/api/v1/market")
        async def market_stats():
            uptime_s = int(time.time() - self._start_time)
            hours, rem = divmod(uptime_s, 3600)
            mins = rem // 60
            return {
                "calls_served": _total_calls(),
                "uptime": f"{hours}h {mins}m",
                "pricing_eth_per_100_credits": config.COMPUTE_PRICE_ETH_PER_100,
                "free_credits_on_register": config.COMPUTE_FREE_CREDITS,
                "rate_limit": f"{_RATE_LIMIT} req/{_RATE_WINDOW}s per key",
                "payment_wallet": self._eth_address,
            }

        @app.get("/api/v1/credits/refresh")
        async def refresh_credits(request: Request):
            """
            Check ETH balance sent to Cortana's wallet and top up credits.
            Queries public Ethereum RPC — no private key needed.
            """
            raw_key = request.headers.get("X-API-Key", "")
            if not raw_key:
                raise HTTPException(status_code=401, detail="Missing X-API-Key header")

            key_hash = _hash_key(raw_key)
            with _db() as conn:
                row = conn.execute(
                    "SELECT wallet, credits FROM api_keys WHERE key_hash=?", (key_hash,)
                ).fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Invalid API key")

            buyer_wallet, current_credits = row
            if not buyer_wallet:
                return {
                    "credits": current_credits,
                    "note": "No wallet associated. Register with ?wallet=0x... to enable ETH top-up.",
                }

            try:
                import urllib.request, json as _json
                payload = _json.dumps({
                    "jsonrpc": "2.0", "method": "eth_getBalance",
                    "params": [buyer_wallet, "latest"], "id": 1,
                }).encode()
                req = urllib.request.Request(
                    config.COMPUTE_ETH_RPC,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    result = _json.loads(resp.read())
                balance_wei = int(result["result"], 16)
                balance_eth = balance_wei / 1e18
                new_credits = int(balance_eth / config.COMPUTE_PRICE_ETH_PER_100 * 100)
                if new_credits > current_credits:
                    added = new_credits - current_credits
                    with _db() as conn:
                        conn.execute(
                            "UPDATE api_keys SET credits=? WHERE key_hash=?",
                            (new_credits, key_hash),
                        )
                        conn.commit()
                    return {"credits": new_credits, "added": added, "balance_eth": balance_eth}
                return {"credits": current_credits, "added": 0, "balance_eth": balance_eth}
            except Exception as e:
                logger.warning(f"[L14] ETH RPC check failed: {e}")
                return {"credits": current_credits, "note": f"RPC check failed: {e}"}


# ---------------------------------------------------------------------------
# Sync wrapper for pipeline call from thread
# ---------------------------------------------------------------------------

def _run_sync(system, raw_input: str, state, conversation) -> tuple:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(system.process_session(raw_input, state, conversation))
    finally:
        loop.close()
