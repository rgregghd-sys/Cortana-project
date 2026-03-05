"""
Cortana User Authentication & Tier Management
Handles registration, login, session tokens, rolling-window usage tracking.
Uses PBKDF2 for password hashing, secure random tokens for sessions.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from cortana import config

log = logging.getLogger(__name__)

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
    if len(password) < 8:
        return {"ok": False, "error": "Password must be at least 8 characters"}

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
    """
    if not username or not password:
        return {"ok": False, "error": "Invalid username or password"}

    with _db() as conn:
        row = conn.execute(
            "SELECT id, password_hash, tier, daily_limit, password_changed_at "
            "FROM users WHERE username=? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()

    if not row:
        return {"ok": False, "error": "Invalid username or password"}

    user_id, pw_hash, tier, daily_limit, pw_changed_at = row
    if not _verify_password(password, pw_hash):
        return {"ok": False, "error": "Invalid username or password"}

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
    if len(new_password) < 8:
        return {"ok": False, "error": "Password must be at least 8 characters"}
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
    if len(new_password) < 8:
        return {"ok": False, "error": "Password must be at least 8 characters"}
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
    """Upgrade a user to a new tier. Returns True on success."""
    if tier not in config.TIERS:
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
