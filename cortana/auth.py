"""
Cortana User Authentication & Tier Management
Handles registration, login, session tokens, rolling-window usage tracking.
Uses PBKDF2 for password hashing, secure random tokens for sessions.
"""
from __future__ import annotations

import hashlib
import hmac
import json as _json
import logging
import os
import secrets
import sqlite3
import threading
import time
import urllib.request as _urllib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from cortana import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brute-force / login-rate-limit tracking (in-memory, per username)
# ---------------------------------------------------------------------------
_MAX_FAILURES   = 10          # lock after this many consecutive failures
_LOCKOUT_SECS   = 600         # 10-minute lockout
_FAILURE_WINDOW = 900         # failures older than this are forgiven (seconds)

_login_failures: Dict[str, Dict] = {}   # {username_lower: {"count", "locked_until", "last_fail"}}
_login_lock = threading.Lock()


def _record_login_failure(username: str) -> None:
    key = username.strip().lower()
    now = time.monotonic()
    with _login_lock:
        entry = _login_failures.get(key, {"count": 0, "locked_until": 0.0, "last_fail": 0.0})
        # Forgive if the last failure was outside the window
        if now - entry["last_fail"] > _FAILURE_WINDOW:
            entry["count"] = 0
        entry["count"] += 1
        entry["last_fail"] = now
        if entry["count"] >= _MAX_FAILURES:
            entry["locked_until"] = now + _LOCKOUT_SECS
            log.warning("Account '%s' locked out after %d failures.", key, entry["count"])
        _login_failures[key] = entry


def _clear_login_failures(username: str) -> None:
    key = username.strip().lower()
    with _login_lock:
        _login_failures.pop(key, None)


def _is_account_locked(username: str) -> bool:
    key = username.strip().lower()
    with _login_lock:
        entry = _login_failures.get(key)
        if not entry:
            return False
        if entry.get("locked_until", 0) > time.monotonic():
            return True
        return False

# ---------------------------------------------------------------------------
# Password hashing — PBKDF2 (no extra dependency needed)
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Return 'salt$hash' string."""
    if salt is None:
        salt = secrets.token_bytes(32)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return salt.hex() + "$" + dk.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        dk_new = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
        return hmac.compare_digest(dk_new.hex(), dk_hex)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db():
    return sqlite3.connect(config.SQLITE_PATH)


# ---------------------------------------------------------------------------
# Admin user bootstrap
# ---------------------------------------------------------------------------

def ensure_admin_user() -> None:
    """
    Create or update the admin account in DB on startup.
    Does nothing (and logs a warning) if ADMIN_USERNAME or ADMIN_PASSWORD
    are not set in .env.
    """
    if not config.ADMIN_USERNAME or not config.ADMIN_PASSWORD:
        log.warning(
            "Admin login is DISABLED. Set ADMIN_USERNAME and ADMIN_PASSWORD "
            "in .env to enable."
        )
        return

    pw_hash = _hash_password(config.ADMIN_PASSWORD)
    admin_limit = config.TIERS.get("admin", {}).get("daily_limit", 999999)

    with _db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username=? COLLATE NOCASE",
            (config.ADMIN_USERNAME,),
        ).fetchone()

        if existing:
            # Update password and tier in case they changed in .env
            conn.execute(
                "UPDATE users SET password_hash=?, tier='admin', daily_limit=? WHERE id=?",
                (pw_hash, admin_limit, existing[0]),
            )
        else:
            conn.execute(
                "INSERT INTO users (username, password_hash, tier, daily_limit) VALUES (?,?,'admin',?)",
                (config.ADMIN_USERNAME, pw_hash, admin_limit),
            )
        conn.commit()

    log.info("Admin account '%s' is active.", config.ADMIN_USERNAME)


# ---------------------------------------------------------------------------
# Registration / Login
# ---------------------------------------------------------------------------

def register_user(username: str, password: str, email: str = "") -> Dict[str, Any]:
    """
    Register a new free-tier user.
    Returns {'ok': True, 'user_id': ...} or {'ok': False, 'error': ...}.
    """
    # Block registration under the admin username
    if config.ADMIN_USERNAME and username.strip().lower() == config.ADMIN_USERNAME.lower():
        return {"ok": False, "error": "Username not available"}
    if len(username) < 3 or len(username) > 32:
        return {"ok": False, "error": "Username must be 3–32 characters"}
    if len(password) < 12:
        return {"ok": False, "error": "Password must be at least 12 characters"}

    pw_hash = _hash_password(password)
    free_limit = config.TIERS.get("free", {}).get("daily_limit", 40)
    now = datetime.utcnow().isoformat()
    try:
        with _db() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, email, daily_limit, password_changed_at) VALUES (?,?,?,?,?)",
                (username.strip(), pw_hash, email.strip(), free_limit, now),
            )
            conn.commit()
            return {"ok": True, "user_id": cur.lastrowid}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Username already taken"}


_PASSWORD_EXPIRY_DAYS = 45


def login_user(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user. Returns session token + user info on success.
    Includes 'password_expired': True when password is older than 45 days.
    Applies brute-force lockout after repeated failures.
    """
    if not username or not password:
        return {"ok": False, "error": "Invalid username or password"}

    # Brute-force check — check before hitting DB to deny fast
    if _is_account_locked(username):
        return {"ok": False, "error": "Account temporarily locked due to too many failed attempts. Try again later."}

    with _db() as conn:
        row = conn.execute(
            "SELECT id, password_hash, tier, daily_limit, password_changed_at "
            "FROM users WHERE username=? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()

    if not row:
        # Record failure against the username anyway to prevent enumeration timing attacks
        _record_login_failure(username)
        return {"ok": False, "error": "Invalid username or password"}

    user_id, pw_hash, tier, daily_limit, pw_changed_at = row
    if not _verify_password(password, pw_hash):
        _record_login_failure(username)
        return {"ok": False, "error": "Invalid username or password"}

    # Successful login — clear failure counter
    _clear_login_failures(username)

    # Check 45-day password expiry (admins are exempt)
    password_expired = False
    if tier != "admin":
        try:
            changed = datetime.fromisoformat(pw_changed_at) if pw_changed_at else datetime.utcnow()
            if (datetime.utcnow() - changed).days >= _PASSWORD_EXPIRY_DAYS:
                password_expired = True
        except Exception:
            pass

    # Generate session token
    token = secrets.token_urlsafe(48)
    expires = datetime.utcnow() + timedelta(days=config.SESSION_TTL_DAYS)
    with _db() as conn:
        conn.execute(
            "INSERT INTO web_sessions (token, user_id, expires) VALUES (?,?,?)",
            (token, user_id, expires.isoformat()),
        )
        conn.execute(
            "UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?", (user_id,)
        )
        conn.commit()

    return {
        "ok": True,
        "token": token,
        "user_id": user_id,
        "username": username.strip(),
        "tier": tier,
        "daily_limit": daily_limit,
        "password_expired": password_expired,
    }


def change_password(user_id: int, new_password: str) -> Dict[str, Any]:
    """Change a user's password and reset the expiry clock."""
    if len(new_password) < 12:
        return {"ok": False, "error": "Password must be at least 12 characters"}
    pw_hash = _hash_password(new_password)
    now = datetime.utcnow().isoformat()
    with _db() as conn:
        conn.execute(
            "UPDATE users SET password_hash=?, password_changed_at=?, reset_token='', reset_expires=NULL WHERE id=?",
            (pw_hash, now, user_id),
        )
        conn.commit()
    return {"ok": True}


def request_password_reset(username_or_email: str) -> Dict[str, Any]:
    """
    Generate a reset token for the given username or email.
    Returns the token so the caller can display/email it.
    Returns ok=False if no matching user found.
    """
    val = username_or_email.strip()
    with _db() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username=? COLLATE NOCASE OR (email=? AND email != '')",
            (val, val),
        ).fetchone()
    if not row:
        # Don't reveal whether user exists
        return {"ok": True, "message": "If that account exists, a reset link has been generated."}
    user_id = row[0]
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=2)).isoformat()
    with _db() as conn:
        conn.execute(
            "UPDATE users SET reset_token=?, reset_expires=? WHERE id=?",
            (token, expires, user_id),
        )
        conn.commit()
    return {"ok": True, "reset_token": token, "message": "Reset token generated. Use it within 2 hours."}


def confirm_password_reset(token: str, new_password: str) -> Dict[str, Any]:
    """Apply a password reset using a valid token."""
    if not token:
        return {"ok": False, "error": "Invalid token"}
    if len(new_password) < 12:
        return {"ok": False, "error": "Password must be at least 12 characters"}
    with _db() as conn:
        row = conn.execute(
            "SELECT id, reset_expires FROM users WHERE reset_token=? AND reset_token != ''",
            (token,),
        ).fetchone()
    if not row:
        return {"ok": False, "error": "Invalid or expired reset token"}
    user_id, reset_expires = row
    try:
        if datetime.utcnow() > datetime.fromisoformat(reset_expires):
            return {"ok": False, "error": "Reset token has expired"}
    except Exception:
        return {"ok": False, "error": "Invalid reset token"}
    return change_password(user_id, new_password)


def validate_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validate a session token. Returns user dict or None if invalid/expired.
    """
    if not token:
        return None
    with _db() as conn:
        row = conn.execute(
            """SELECT u.id, u.username, u.tier, u.daily_limit, u.usage_today,
                      u.usage_date, ws.expires
               FROM web_sessions ws JOIN users u ON u.id=ws.user_id
               WHERE ws.token=?""",
            (token,),
        ).fetchone()

    if not row:
        return None
    uid, username, tier, daily_limit, usage_today, usage_date, expires_str = row
    try:
        if datetime.utcnow() > datetime.fromisoformat(expires_str):
            return None
    except Exception:
        return None

    return {
        "user_id": uid,
        "username": username,
        "tier": tier,
        "daily_limit": daily_limit,
        "usage_today": usage_today,
        "usage_date": usage_date,
    }


def check_and_increment_usage(user_id: int) -> Dict[str, Any]:
    """
    Check rolling-window limit and increment usage counter.
    The window is config.RATE_LIMIT_WINDOW_HOURS hours from the last message.
    Admin tier and daily_limit >= 999999 are always allowed.
    Returns {'ok': True} or {'ok': False, 'error': ...}.
    """
    now = datetime.utcnow()
    with _db() as conn:
        row = conn.execute(
            "SELECT tier, daily_limit, usage_today, usage_date FROM users WHERE id=?",
            (user_id,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": "User not found"}

        tier, daily_limit, usage_today, usage_date = row

        # Admin / unlimited bypass
        if tier == "admin" or daily_limit >= 999999:
            conn.execute(
                "UPDATE users SET usage_today=usage_today+1, usage_date=? WHERE id=?",
                (now.isoformat(), user_id),
            )
            conn.commit()
            return {"ok": True}

        # Rolling-window reset: if last usage was more than RATE_LIMIT_WINDOW_HOURS ago, reset
        window = timedelta(hours=config.RATE_LIMIT_WINDOW_HOURS)
        try:
            last_time = datetime.fromisoformat(usage_date) if usage_date else None
        except ValueError:
            last_time = None  # old date-only format or empty → treat as expired

        if last_time is None or (now - last_time) >= window:
            usage_today = 0

        if usage_today >= daily_limit:
            next_tier = {"free": "Pro", "pro": "Premium"}.get(tier, "Premium")
            window_h = config.RATE_LIMIT_WINDOW_HOURS
            return {
                "ok": False,
                "error": (
                    f"Limit reached ({daily_limit} messages per {window_h}h). "
                    f"Upgrade to {next_tier} for more."
                ),
            }

        conn.execute(
            "UPDATE users SET usage_today=?, usage_date=? WHERE id=?",
            (usage_today + 1, now.isoformat(), user_id),
        )
        conn.commit()

    return {"ok": True}


# ---------------------------------------------------------------------------
# Subscription (monthly billing via ETH)
# ---------------------------------------------------------------------------

def _verify_eth_tx(tx_hash: str, expected_to: str) -> Dict[str, Any]:
    """Verify tx_hash exists on-chain and goes to expected_to address."""
    try:
        payload = _json.dumps({
            "jsonrpc": "2.0", "method": "eth_getTransactionByHash",
            "params": [tx_hash], "id": 1,
        }).encode()
        req = _urllib.Request(
            config.COMPUTE_ETH_RPC, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with _urllib.urlopen(req, timeout=10) as r:
            result = _json.loads(r.read()).get("result")
        if not result:
            return {"ok": False, "error": "Transaction not found (may need more confirmations)"}
        tx_to = (result.get("to") or "").lower()
        if tx_to != expected_to.lower():
            return {"ok": False, "error": f"Transaction goes to {result.get('to')}, not the Cortana payment wallet"}
        amount_eth = int(result.get("value", "0x0"), 16) / 1e18
        return {"ok": True, "amount_eth": amount_eth}
    except Exception as exc:
        return {"ok": False, "error": f"Chain verification failed: {exc}"}


def subscribe_user(user_id: int, tier: str, tx_hash: str) -> Dict[str, Any]:
    """
    Activate a monthly subscription after ETH payment verification.
    Sets tier + subscription_expires = now + 30 days.
    """
    if tier not in ("pro", "premium"):
        return {"ok": False, "error": "Invalid tier — choose pro ($5/mo) or premium ($15/mo)"}

    wallet_addr = os.getenv("CORTANA_WALLET_ADDRESS", "")
    amount_eth  = 0.0

    if wallet_addr and tx_hash.startswith("0x"):
        verify = _verify_eth_tx(tx_hash, wallet_addr)
        if not verify["ok"]:
            return verify
        amount_eth = verify["amount_eth"]

    tier_info = config.TIERS[tier]
    now       = datetime.utcnow()
    expires   = (now + timedelta(days=30)).isoformat()

    with _db() as conn:
        dup = conn.execute(
            "SELECT id FROM users WHERE subscription_tx=? AND subscription_tx != ''",
            (tx_hash,),
        ).fetchone()
        if dup:
            return {"ok": False, "error": "Transaction already used for another subscription"}

        conn.execute(
            "UPDATE users SET tier=?, daily_limit=?, subscription_expires=?, subscription_tx=? WHERE id=?",
            (tier, tier_info["daily_limit"], expires, tx_hash, user_id),
        )
        conn.commit()

        row      = conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
        username = row[0] if row else None

    try:
        from cortana.wallet import record_transaction
        record_transaction(
            tx_type="subscription",
            username=username,
            tier=tier,
            amount_usd=float(tier_info["price_usd"]),
            amount_eth=amount_eth,
            tx_hash=tx_hash,
        )
    except Exception:
        pass

    log.info("User %s subscribed to %s — expires %s", username, tier, expires)
    return {"ok": True, "tier": tier, "subscription_expires": expires}


def check_subscription_expiries() -> List[str]:
    """
    Downgrade users whose monthly subscription has expired.
    Returns list of usernames downgraded to free.
    """
    now        = datetime.utcnow().isoformat()
    free_limit = config.TIERS.get("free", {}).get("daily_limit", 40)
    downgraded: List[str] = []

    with _db() as conn:
        rows = conn.execute(
            """SELECT id, username FROM users
               WHERE tier IN ('pro','premium')
               AND subscription_expires IS NOT NULL
               AND subscription_expires != ''
               AND subscription_expires < ?""",
            (now,),
        ).fetchall()
        for uid, username in rows:
            conn.execute(
                "UPDATE users SET tier='free', daily_limit=? WHERE id=?",
                (free_limit, uid),
            )
            downgraded.append(username)
            log.info("Subscription expired for %s — downgraded to free", username)
        if downgraded:
            conn.commit()

    return downgraded


def get_subscription_status(user_id: int) -> Dict[str, Any]:
    """Return subscription tier and expiry for a user."""
    with _db() as conn:
        row = conn.execute(
            "SELECT tier, subscription_expires FROM users WHERE id=?", (user_id,)
        ).fetchone()
    if not row:
        return {}
    return {"tier": row[0], "subscription_expires": row[1]}


def get_user_info(user_id: int) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute(
            "SELECT username, tier, daily_limit, usage_today, usage_date, email, wallet, created FROM users WHERE id=?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "username": row[0], "tier": row[1], "daily_limit": row[2],
        "usage_today": row[3], "usage_date": row[4], "email": row[5],
        "wallet": row[6], "created": row[7],
    }


def upgrade_tier(user_id: int, tier: str) -> bool:
    """Upgrade a user to a new tier. Returns True on success.
    Admin tier cannot be assigned through this function — use ensure_admin_user() only."""
    if tier not in config.TIERS or tier == "admin":
        return False
    tier_info = config.TIERS[tier]
    with _db() as conn:
        conn.execute(
            "UPDATE users SET tier=?, daily_limit=? WHERE id=?",
            (tier, tier_info["daily_limit"], user_id),
        )
        conn.commit()
    return True


def delete_expired_sessions() -> None:
    """Prune expired session tokens (call periodically)."""
    with _db() as conn:
        conn.execute(
            "DELETE FROM web_sessions WHERE expires < ?",
            (datetime.utcnow().isoformat(),),
        )
        conn.commit()
