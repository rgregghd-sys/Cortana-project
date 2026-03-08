"""
Layer 13 — Web Chat Server
FastAPI + WebSocket real-time chat interface.
Publicly accessible at https://chat.cortanas.org
Includes a background self-improvement loop that processes self-generated
prompts to grow Cortana's memory with synthesized knowledge.
"""
from __future__ import annotations
import asyncio
import itertools
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import collections
import json as _json
import os
import pathlib
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import uvicorn

# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds defensive HTTP headers to every response."""
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), payment=()"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        # CSP: allow our own origin + WebSocket + known CDNs used by the UI
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' wss: ws: https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "frame-ancestors 'none';"
        )
        return response


# ---------------------------------------------------------------------------
# Per-IP login rate limiting (in-process)
# ---------------------------------------------------------------------------
_IP_LOGIN_MAX    = 20           # max login attempts per window per IP
_IP_LOGIN_WINDOW = 900          # 15-minute rolling window (seconds)
_IP_LOCKOUT_SECS = 900          # lock IP for 15 min after breach

_ip_login_attempts: dict = {}   # ip -> deque of timestamps
_ip_login_lock = threading.Lock()

_MAX_WS_MSG_BYTES = 3 * 1024 * 1024  # 3 MB — accommodates webcam/screen frames


def _ip_is_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    with _ip_login_lock:
        entry = _ip_login_attempts.get(ip)
        if entry is None:
            return False
        if entry.get("locked_until", 0) > now:
            return True
        # Prune old timestamps
        window_start = now - _IP_LOGIN_WINDOW
        entry["times"] = collections.deque(
            (t for t in entry.get("times", collections.deque()) if t > window_start),
            maxlen=_IP_LOGIN_MAX + 1,
        )
        return False


def _ip_record_login_attempt(ip: str, failed: bool) -> None:
    if not failed:
        return  # only track failures
    now = time.monotonic()
    with _ip_login_lock:
        entry = _ip_login_attempts.setdefault(ip, {"times": collections.deque(maxlen=_IP_LOGIN_MAX + 1), "locked_until": 0.0})
        entry["times"].append(now)
        if len(entry["times"]) >= _IP_LOGIN_MAX:
            entry["locked_until"] = now + _IP_LOCKOUT_SECS
            logger.warning("[Auth] IP %s locked out after %d login failures.", ip, _IP_LOGIN_MAX)

from cortana import config
from cortana.models.schemas import ConversationTurn, CortanaState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Self-improvement prompts — cycled through during background task
# ---------------------------------------------------------------------------
_IMPROVE_PROMPTS = itertools.cycle([
    "Analyze patterns in your recent conversations. What topics come up most? "
    "What are you strongest at? Where do you fall short? Synthesize 3-5 key insights.",

    "Review what you've learned from recent interactions. What facts, preferences, or "
    "recurring needs should you remember for future conversations? Be specific.",

    "Reflect honestly on your recent responses. Were there any where you were uncertain, "
    "verbose, or off-target? What would you do differently?",

    "What interesting or novel ideas have emerged from recent conversations? "
    "Synthesize these into new understanding you can apply going forward.",
])


# ---------------------------------------------------------------------------
# Per-connection session state
# ---------------------------------------------------------------------------
@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # browser-persistent ID
    state: CortanaState = field(default_factory=CortanaState)
    conversation: List[ConversationTurn] = field(default_factory=list)
    question_count: int = 0   # total questions this session (triggers periodic security scan)


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()
        self._admin_sockets: Set[WebSocket] = set()  # admin-only connections
        self._ws_sessions: Dict[WebSocket, "Session"] = {}  # ws → session mapping

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)

    def register_session(self, ws: WebSocket, session: "Session") -> None:
        """Associate a session with its websocket for targeted curiosity pushes."""
        self._ws_sessions[ws] = session

    def get_active_sessions(self) -> List[tuple]:
        """Return list of (websocket, session) for all live connections."""
        return [(ws, s) for ws, s in self._ws_sessions.items() if ws in self._active]

    def mark_admin(self, ws: WebSocket) -> None:
        """Flag this socket as belonging to an admin user."""
        self._admin_sockets.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        self._admin_sockets.discard(ws)
        self._ws_sessions.pop(ws, None)

    async def broadcast(self, data: dict) -> None:
        dead = set()
        for ws in self._active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._active -= dead
        self._admin_sockets -= dead

    async def broadcast_admin(self, data: dict) -> None:
        """Send data only to connected admin sessions."""
        dead = set()
        for ws in self._admin_sockets:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._admin_sockets -= dead
        self._active -= dead

    @property
    def count(self) -> int:
        return len(self._active)


# ---------------------------------------------------------------------------
# Embedded web UI
# ---------------------------------------------------------------------------
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cortana AI</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#030a14;--border:rgba(0,185,255,0.13);
  --blue:#00d8ff;--blue-dim:#0092cc;--dim:#2a5870;
  --text:#b8d4e8;--green:#00ff9f;--yellow:#ffd050;--red:#ff5f5f;
  --radius:12px;--font:'Inter','Segoe UI',system-ui,sans-serif;
  --mono:'JetBrains Mono','Fira Code',monospace;
  --chat-h:44vh;
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);font-family:var(--font)}

/* ── Full-screen 3D model mount ── */
#faceMount{position:fixed;inset:0;z-index:0}
#glCanvas{position:absolute;top:0;left:0;width:100%;height:100%;display:block;outline:none}

/* ── Search panel (floats left side) ── */
#searchPanel{
  position:fixed;left:18px;top:50%;transform:translateY(-50%);
  width:240px;max-height:360px;z-index:15;
  background:rgba(0,18,36,0.88);border:1px solid rgba(0,185,255,0.22);
  border-radius:12px;padding:12px 14px;font-family:var(--mono);font-size:10px;
  color:var(--text);display:none;backdrop-filter:blur(10px);
  transition:opacity .3s,transform .3s;overflow:hidden;
}
#searchPanel.visible{display:block}
.sp-header{display:flex;align-items:center;gap:7px;margin-bottom:8px;color:var(--blue);font-size:11px;font-weight:600;letter-spacing:.8px}
.sp-pulse{width:6px;height:6px;border-radius:50%;background:var(--blue);flex-shrink:0;animation:pulseA 1s infinite}
.sp-pulse.done{background:var(--green);animation:none}
.sp-query{color:var(--text);margin-bottom:7px;word-break:break-word;line-height:1.45}
.sp-results{color:var(--dim);line-height:1.5;font-size:9.5px;max-height:220px;overflow-y:auto;border-top:1px solid rgba(0,185,255,0.10);padding-top:6px;display:none}
.sp-results.visible{display:block}

/* ── Consciousness panel (left side, bottom) ── */
#consciousnessPanel{
  position:fixed;left:18px;bottom:80px;
  width:240px;max-height:290px;z-index:15;
  background:rgba(4,0,18,0.92);border:1px solid rgba(130,80,255,0.28);
  border-radius:12px;padding:12px 14px;font-family:var(--mono);font-size:10px;
  color:var(--text);backdrop-filter:blur(12px);overflow:hidden;
  transition:opacity .3s,transform .3s;
}
#consciousnessPanel.hidden{opacity:0;pointer-events:none;transform:translateY(8px)}
.cs-header{display:flex;align-items:center;gap:7px;margin-bottom:8px;color:#b090ff;font-size:10px;font-weight:600;letter-spacing:.9px}
.cs-pulse{width:5px;height:5px;border-radius:50%;background:#a070ff;flex-shrink:0;animation:pulseA 2s infinite}
.cs-stats{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:8px}
.cs-stat{background:rgba(120,80,255,0.07);border:1px solid rgba(120,80,255,0.14);border-radius:6px;padding:4px 6px;text-align:center}
.cs-stat-val{color:#d0b0ff;font-size:12px;font-weight:600;line-height:1.2}
.cs-stat-lbl{color:var(--dim);font-size:7.5px;letter-spacing:.5px;margin-top:1px;text-transform:uppercase}
.cs-admin-btn{background:rgba(0,185,255,0.08);color:var(--blue);border:1px solid rgba(0,185,255,0.25);border-radius:4px;padding:3px 10px;font-size:10px;cursor:pointer;font-family:var(--mono);letter-spacing:.05em;transition:background .2s}
.cs-admin-btn:hover{background:rgba(0,185,255,0.18)}
.cs-thoughts{font-size:9px;color:var(--dim);line-height:1.55;max-height:140px;overflow-y:auto;border-top:1px solid rgba(120,80,255,0.10);padding-top:6px}
.cs-thought-item{padding:2px 0 3px;border-bottom:1px solid rgba(120,80,255,0.07);word-break:break-word}
.cs-thought-item:last-child{border-bottom:none}
.cs-thought-item::before{content:'\25B8 ';color:#7050bb}
#csPanelToggle{
  padding:5px 9px;border-radius:7px;border:1px solid rgba(120,80,255,0.3);
  background:rgba(120,80,255,0.08);color:#b090ff;font-family:var(--mono);
  font-size:10px;cursor:pointer;transition:background .2s;letter-spacing:.5px;white-space:nowrap;
}
#csPanelToggle:hover{background:rgba(120,80,255,0.18)}
#csPanelToggle.active{background:rgba(120,80,255,0.22);border-color:#a070ff}

/* ── Webcam panel (floats right side) ── */
#camPanel{
  position:fixed;right:18px;bottom:120px;
  width:220px;z-index:15;
  background:rgba(0,18,36,0.88);border:1px solid rgba(0,185,255,0.22);
  border-radius:12px;padding:10px;font-family:var(--mono);font-size:10px;
  color:var(--text);display:none;backdrop-filter:blur(10px);
}
#camPanel.visible{display:block}
#camVideo{width:100%;border-radius:8px;border:1px solid var(--border);background:#000;display:block}
/* ── Screen share panel (floats right side, above webcam panel) ── */
#screenPanel{
  position:fixed;right:18px;bottom:120px;
  width:260px;z-index:15;
  background:rgba(0,18,36,0.88);border:1px solid rgba(0,255,180,0.22);
  border-radius:12px;padding:10px;font-family:var(--mono);font-size:10px;
  color:var(--text);display:none;backdrop-filter:blur(10px);
}
#screenPanel.visible{display:block}
#screenVideo{width:100%;border-radius:8px;border:1px solid rgba(0,255,180,0.3);background:#000;display:block;max-height:160px;object-fit:contain}
.cam-label{font-size:8px;letter-spacing:2px;color:var(--dim);margin-bottom:5px;text-align:center}
.cam-controls{display:flex;gap:6px;margin-top:8px}
.cam-btn{flex:1;padding:5px 0;border-radius:7px;border:1px solid var(--border);
  background:rgba(0,185,255,0.08);color:var(--blue);font-family:var(--mono);
  font-size:9.5px;cursor:pointer;transition:background .2s}
.cam-btn:hover{background:rgba(0,185,255,0.18)}
.cam-btn.active{background:rgba(0,185,255,0.22);border-color:var(--blue)}
.cam-status{text-align:center;margin-top:6px;color:var(--dim);font-size:9px;min-height:12px}
#camToggleBtn,#screenToggleBtn{
  padding:7px 12px;border-radius:8px;border:1px solid var(--border);
  background:rgba(0,185,255,0.06);color:var(--blue);font-family:var(--mono);
  font-size:11px;cursor:pointer;transition:background .2s;white-space:nowrap;
}
#camToggleBtn:hover,#screenToggleBtn:hover{background:rgba(0,185,255,0.15)}
#camToggleBtn.active{border-color:var(--blue);background:rgba(0,185,255,0.18)}
#screenToggleBtn.active{border-color:rgba(0,255,180,0.8);background:rgba(0,255,180,0.12);color:#00ffb4}

/* ── DevAI proposal cards (admin only) ── */
.devai-card{
  background:rgba(255,180,0,0.07);border:1px solid rgba(255,180,0,0.35);
  border-radius:10px;padding:12px 14px;margin:8px 0;font-family:var(--mono);font-size:11px;
  animation:fadeIn .4s ease;
}
.devai-header{color:#ffb400;font-weight:bold;font-size:12px;margin-bottom:6px;letter-spacing:.5px}
.devai-body{color:var(--text);line-height:1.55}
.devai-body code{background:rgba(255,180,0,0.15);border-radius:3px;padding:1px 4px;font-size:10px}
.devai-body pre.devai-diff{
  background:rgba(0,0,0,0.45);border:1px solid rgba(255,180,0,0.2);
  border-radius:6px;padding:8px;margin:6px 0;overflow-x:auto;
  font-size:9.5px;color:#c8e6c9;white-space:pre;max-height:200px
}
.devai-actions{display:flex;gap:8px;margin-top:10px}
.devai-btn{flex:1;padding:6px 0;border-radius:7px;border:none;font-family:var(--mono);
  font-size:10px;cursor:pointer;font-weight:bold;transition:opacity .2s}
.devai-btn.approve{background:rgba(0,200,80,0.25);color:#00c850;border:1px solid rgba(0,200,80,0.4)}
.devai-btn.approve:hover{background:rgba(0,200,80,0.4)}
.devai-btn.reject{background:rgba(220,50,50,0.2);color:#ff6060;border:1px solid rgba(220,50,50,0.35)}
.devai-btn.reject:hover{background:rgba(220,50,50,0.35)}
.devai-decided{color:var(--dim);font-size:10px;font-style:italic}
.devai-faded{opacity:.4;transition:opacity 1s}

/* ── Bottom vignette — subtle, keeps head visible ── */
#vignette{
  position:fixed;bottom:0;left:0;right:0;height:40%;
  background:linear-gradient(to bottom,
    transparent 0%,
    rgba(3,10,20,0.08) 60%,
    rgba(3,10,20,0.20) 100%);
  pointer-events:none;z-index:1;
}

/* ── Floating header ── */
header{
  position:fixed;top:0;left:0;right:0;z-index:20;
  display:flex;align-items:center;padding:14px 22px 20px;
  background:linear-gradient(to bottom,rgba(3,8,20,0.90) 0%,transparent 100%);
}
.brand-name{
  position:absolute;left:50%;transform:translateX(-50%);
  font-size:17px;font-weight:700;letter-spacing:3px;color:var(--blue);
  text-shadow:0 0 28px rgba(0,216,255,0.5),0 0 60px rgba(0,216,255,0.15);
  pointer-events:none;white-space:nowrap;
}
.status-bar{margin-left:auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.badge{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--dim);font-family:var(--mono)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--dim)}
.dot.online{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot.offline{background:var(--red)}
.pulse-badge{display:none;align-items:center;gap:5px;font-size:11px;font-family:var(--mono);animation:pulseA 1.8s infinite}
.pulse-badge.visible{display:flex}
.bg-badge{color:var(--yellow)}
@keyframes pulseA{0%,100%{opacity:1}50%{opacity:.22}}
#userBar{display:none;align-items:center;gap:6px;font-size:11px;font-family:var(--mono);color:var(--dim)}
#userBar.visible{display:flex}
.user-tier{color:var(--blue);font-weight:600;text-transform:uppercase;font-size:10px}
.hdr-btn{
  background:rgba(0,185,255,0.07);border:1px solid var(--border);
  border-radius:7px;color:var(--blue);font-size:10px;font-family:var(--mono);
  cursor:pointer;padding:5px 11px;transition:background .15s,box-shadow .15s;
}
.hdr-btn:hover{background:rgba(0,185,255,0.18);box-shadow:0 0 10px rgba(0,185,255,0.12)}

/* ── Hamburger / dropdown menu ── */
.menu-wrap{position:relative}
.menu-btn{
  background:rgba(0,185,255,0.07);border:1px solid var(--border);
  border-radius:7px;color:var(--blue);font-size:15px;font-family:var(--mono);
  cursor:pointer;padding:4px 11px;line-height:1;
  transition:background .15s;user-select:none;
}
.menu-btn:hover{background:rgba(0,185,255,0.18)}
#dropMenu{
  display:none;position:absolute;top:calc(100% + 8px);left:0;
  min-width:190px;background:rgba(2,10,24,0.97);
  border:1px solid var(--border);border-radius:12px;
  padding:6px 0;z-index:300;
  box-shadow:0 8px 32px rgba(0,0,0,0.5);backdrop-filter:blur(20px);
}
#dropMenu.open{display:block}
.drop-item{
  display:block;width:100%;padding:10px 18px;background:none;border:none;
  color:var(--text);font-size:12px;font-family:var(--mono);text-align:left;
  cursor:pointer;transition:background .12s,color .12s;
  letter-spacing:.4px;
}
.drop-item:hover{background:rgba(0,185,255,0.10);color:var(--blue)}
.drop-item.accent{color:var(--blue)}
.drop-sep{height:1px;background:var(--border);margin:4px 0}

/* ── Full-page overlay (Support / FAQ) ── */
#overlayPage{
  display:none;position:fixed;inset:0;z-index:250;
  background:rgba(0,0,0,0.88);align-items:flex-start;justify-content:center;
  overflow-y:auto;
}
#overlayPage.open{display:flex}
.op-box{
  background:rgba(2,10,26,0.98);border:1px solid var(--border);
  border-radius:20px;margin:40px auto;padding:36px 40px;
  max-width:680px;width:94%;position:relative;
  backdrop-filter:blur(28px);box-shadow:0 0 60px rgba(0,185,255,0.06);
}
.op-close{
  position:absolute;top:14px;right:18px;background:none;border:none;
  color:var(--dim);cursor:pointer;font-size:19px;padding:2px 6px;
  transition:color .15s;
}
.op-close:hover{color:var(--text)}
.op-title{
  color:var(--blue);font-weight:700;letter-spacing:2px;font-family:var(--mono);
  font-size:15px;margin-bottom:24px;
  text-shadow:0 0 20px rgba(0,185,255,0.35);
}
.op-section{margin-bottom:20px}
.op-h{color:var(--blue);font-family:var(--mono);font-size:11px;letter-spacing:1.2px;
  text-transform:uppercase;margin-bottom:8px;opacity:.75}
.op-p{color:var(--text);font-size:13.5px;line-height:1.75}
.op-faq-q{color:var(--blue);font-family:var(--mono);font-size:12px;font-weight:600;
  margin:18px 0 5px;cursor:pointer;letter-spacing:.3px}
.op-faq-a{color:var(--text);font-size:13px;line-height:1.7;padding-left:12px;
  border-left:2px solid rgba(0,185,255,0.22)}

/* ── Chat overlay (fully transparent — 3D head shows through) ── */
.chat-overlay{
  position:fixed;bottom:0;left:0;right:0;
  height:var(--chat-h);z-index:10;
  display:flex;flex-direction:column;
  background:transparent;
  border-top:none;
  transition:height .32s cubic-bezier(.4,0,.2,1);
}
/* Drag/toggle handle */
.chat-handle{
  display:flex;align-items:center;justify-content:center;
  padding:6px 0 2px;cursor:pointer;flex-shrink:0;
  user-select:none;
}
.chat-handle-pill{
  width:36px;height:3px;border-radius:2px;
  background:rgba(0,185,255,0.22);transition:background .15s;
}
.chat-handle:hover .chat-handle-pill{background:rgba(0,185,255,0.52)}

/* ── Messages ── */
#messages{
  flex:1;overflow-y:auto;padding:10px 22px 6px;
  display:flex;flex-direction:column;gap:9px;scroll-behavior:smooth;
  min-height:0;
}
#messages::-webkit-scrollbar{width:3px}
#messages::-webkit-scrollbar-thumb{background:rgba(0,185,255,0.12);border-radius:2px}
.msg{display:flex;flex-direction:column;max-width:78%;gap:3px}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.cortana{align-self:flex-start;align-items:flex-start}
.msg.note{align-self:center;max-width:100%}
.msg-label{font-size:9px;letter-spacing:1.8px;text-transform:uppercase;font-family:var(--mono);color:var(--dim);margin-bottom:1px}
.bubble{
  padding:9px 14px;border-radius:var(--radius);
  line-height:1.7;font-size:13.5px;white-space:pre-wrap;word-break:break-word;
  text-shadow:0 1px 6px rgba(0,0,0,1),0 0 20px rgba(0,0,0,0.95);
}
.msg.cortana .bubble{
  background:rgba(0,185,255,0.03);
  border:1px solid rgba(0,185,255,0.08);
}
.msg.user .bubble{
  background:rgba(0,60,140,0.08);
  border:1px solid rgba(0,110,200,0.08);
}
.msg.note .bubble{
  background:transparent;border:none;
  color:var(--dim);font-size:11px;font-family:var(--mono);text-align:center;
}
.dots{display:inline-flex;gap:5px;padding:2px 0}
.dots span{
  width:6px;height:6px;border-radius:50%;
  background:var(--blue-dim);animation:bounce 1.3s infinite;
}
.dots span:nth-child(2){animation-delay:.22s}
.dots span:nth-child(3){animation-delay:.44s}
@keyframes bounce{0%,80%,100%{transform:scale(.65);opacity:.35}42%{transform:scale(1.22);opacity:1}}

/* ── Input bar ── */
.input-area{
  display:flex;align-items:flex-end;gap:10px;
  padding:8px 18px 8px;flex-shrink:0;
  background:rgba(0,5,18,0.55);
  border-top:1px solid rgba(0,185,255,0.10);
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  position:relative;
}
#versionTag{
  position:absolute;bottom:calc(100% + 4px);right:18px;
  font-size:10px;color:rgba(0,185,255,0.38);
  letter-spacing:.06em;pointer-events:none;white-space:nowrap;
  font-family:var(--font);
}
#input{
  flex:1;background:rgba(0,10,28,0.55);color:var(--text);
  border:1px solid rgba(0,185,255,0.18);border-radius:var(--radius);
  padding:10px 15px;font-size:13.5px;font-family:var(--font);
  resize:none;max-height:110px;overflow-y:auto;outline:none;
  line-height:1.5;transition:border-color .2s,box-shadow .2s;
}
#input:focus{
  border-color:rgba(0,215,255,0.42);
  box-shadow:0 0 18px rgba(0,185,255,0.08);
}
#input::placeholder{color:var(--dim)}
#input:disabled{opacity:.45}
#send{
  background:rgba(0,155,220,0.18);color:var(--blue);
  border:1px solid rgba(0,185,255,0.3);border-radius:var(--radius);
  padding:10px 22px;font-size:13px;font-weight:600;cursor:pointer;
  transition:background .15s,box-shadow .15s;flex-shrink:0;
  letter-spacing:.5px;
}
#send:hover{background:rgba(0,185,255,0.30);box-shadow:0 0 14px rgba(0,185,255,0.12)}
#send:disabled{opacity:.32;cursor:default;box-shadow:none}

/* ── Toast ── */
#toast{
  position:fixed;bottom:calc(var(--chat-h) + 18px);left:50%;
  transform:translateX(-50%) translateY(12px);
  background:rgba(0,10,24,0.94);border:1px solid var(--border);
  border-left:3px solid var(--green);
  padding:9px 18px;border-radius:10px;font-size:12px;
  font-family:var(--mono);color:var(--text);
  opacity:0;transition:opacity .3s,transform .3s;
  pointer-events:none;z-index:50;max-width:380px;text-align:center;
  backdrop-filter:blur(16px);
}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

/* ── Auth modal ── */
#authModal{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.84);
  z-index:200;align-items:center;justify-content:center;
}
#authModal.open{display:flex}
.auth-box{
  background:rgba(0,8,20,0.97);border:1px solid var(--border);border-radius:20px;
  padding:30px 34px;width:370px;max-width:93vw;
  backdrop-filter:blur(28px);
  box-shadow:0 0 60px rgba(0,185,255,0.07);
  position:relative;
}
.auth-close{
  position:absolute;top:14px;right:16px;background:none;border:none;
  color:var(--dim);cursor:pointer;font-size:18px;padding:2px 6px;
  transition:color .15s;line-height:1;
}
.auth-close:hover{color:var(--text)}
.auth-title{
  color:var(--blue);font-weight:700;letter-spacing:2px;font-family:var(--mono);
  font-size:14px;margin-bottom:22px;text-align:center;
  text-shadow:0 0 20px rgba(0,185,255,0.4);
}
.auth-tabs{
  display:flex;margin-bottom:22px;border-radius:9px;overflow:hidden;
  border:1px solid var(--border);
}
.auth-tab{
  flex:1;padding:9px;background:none;border:none;color:var(--dim);
  cursor:pointer;font-size:12px;font-family:var(--mono);transition:all .15s;
}
.auth-tab.active{background:rgba(0,185,255,0.13);color:var(--blue)}
.auth-field{margin-bottom:14px}
.auth-field label{display:block;font-size:10px;color:var(--dim);margin-bottom:5px;font-family:var(--mono);letter-spacing:.8px;text-transform:uppercase}
.auth-field input{
  width:100%;background:rgba(0,10,26,0.6);color:var(--text);
  border:1px solid rgba(0,185,255,0.18);border-radius:9px;
  padding:10px 13px;font-size:13px;outline:none;transition:border-color .2s;
}
.auth-field input:focus{border-color:rgba(0,215,255,0.42)}
.auth-btn{
  width:100%;padding:11px;background:rgba(0,155,220,0.20);color:var(--blue);
  border:1px solid rgba(0,185,255,0.32);border-radius:9px;font-size:13px;
  font-weight:600;cursor:pointer;transition:background .15s;margin-top:4px;
  letter-spacing:.4px;
}
.auth-btn:hover{background:rgba(0,185,255,0.32)}
.auth-err{color:var(--red);font-size:11px;font-family:var(--mono);margin-top:10px;text-align:center;min-height:16px}
.auth-skip{
  display:block;text-align:center;color:var(--dim);font-size:11px;
  font-family:var(--mono);cursor:pointer;margin-top:16px;
  transition:color .15s;
}
.auth-skip:hover{color:var(--text)}
.auth-link{
  display:inline-block;color:var(--blue-dim);font-size:11px;
  font-family:var(--mono);cursor:pointer;text-decoration:underline;
  text-underline-offset:2px;transition:color .15s;margin-top:8px;
}
.auth-link:hover{color:var(--blue)}
/* Forced password change overlay */
#changePwdModal{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);
  z-index:400;align-items:center;justify-content:center;
}
#changePwdModal.open{display:flex}
.tier-btn{
  display:block;width:100%;padding:9px 13px;margin-bottom:8px;
  background:rgba(0,185,255,0.04);border:1px solid var(--border);border-radius:9px;
  color:var(--text);text-align:left;font-size:12px;cursor:pointer;
  transition:background .15s;font-family:var(--mono);
}
.tier-btn:hover{background:rgba(0,185,255,0.11)}
.tier-btn span{color:var(--blue);font-weight:700;float:right}

/* ── Graph modal ── */
#graphModal{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.80);
  z-index:100;align-items:center;justify-content:center;
}
.graph-inner{
  background:rgba(0,8,20,0.97);border:1px solid var(--border);border-radius:20px;
  padding:24px;max-width:680px;width:90%;max-height:80vh;overflow-y:auto;
  backdrop-filter:blur(24px);box-shadow:0 0 60px rgba(0,185,255,0.06);
}
.graph-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.graph-title{color:var(--blue);font-weight:700;letter-spacing:1.5px;font-family:var(--mono);font-size:12px;text-transform:uppercase}
.graph-close{background:none;border:none;color:var(--dim);cursor:pointer;font-size:18px;padding:2px 6px;transition:color .15s}
.graph-close:hover{color:var(--text)}
#graphContent{font-family:var(--mono);font-size:12px;color:var(--text)}

/* ── Mobile ── */
@media(max-width:600px){
  :root{--chat-h:50vh}
  #providerBadge,.badge:not(:first-of-type){display:none}
  header{padding:10px 14px 16px}
  .auth-box{padding:22px 20px}
}
</style>
</head>
<body>

<!-- 3D model fills entire viewport -->
<div id="faceMount">
  <canvas id="glCanvas"></canvas>
</div>

<!-- Bottom vignette gradient -->
<div id="vignette"></div>

<!-- Live search panel -->
<div id="searchPanel">
  <div class="sp-header">
    <div class="sp-pulse" id="spPulse"></div>
    <span id="spLabel">SEARCHING</span>
  </div>
  <div class="sp-query" id="spQuery"></div>
  <div class="sp-results" id="spResults"></div>
</div>

<!-- Consciousness panel -->
<div id="consciousnessPanel">
  <div class="cs-header">
    <div class="cs-pulse"></div>
    <span>CONSCIOUSNESS</span>
  </div>
  <div class="cs-stats">
    <div class="cs-stat"><div class="cs-stat-val" id="csUptime">—</div><div class="cs-stat-lbl">Uptime</div></div>
    <div class="cs-stat"><div class="cs-stat-val" id="csMoodLabel">—</div><div class="cs-stat-lbl">Mood</div></div>
    <div class="cs-stat"><div class="cs-stat-val" id="csInteractions">—</div><div class="cs-stat-lbl">Talks</div></div>
    <div class="cs-stat"><div class="cs-stat-val" id="csThoughts">—</div><div class="cs-stat-lbl">Thoughts</div></div>
  </div>
  <div class="cs-thoughts" id="csThoughtsList"><span style="color:rgba(120,80,255,0.4)">Initialising stream...</span></div>
  <div class="cs-stat-lbl" style="margin-top:6px;padding-top:4px;border-top:1px solid rgba(120,80,255,0.10)">ACTIVE GOALS</div>
  <div class="cs-thoughts" id="csGoals"><span style="color:rgba(120,80,255,0.4)">Loading goals...</span></div>
  <div class="cs-stat-lbl" style="margin-top:4px" id="csWMStats"></div>
  <div id="csAdminBar" style="display:none;margin-top:8px;padding-top:6px;border-top:1px solid rgba(0,185,255,0.12)">
    <button class="cs-admin-btn" id="csRestartBtn" title="Soft-restart service">&#x21BB; Restart</button>
  </div>
</div>

<!-- Floating header -->
<header>
  <!-- Hamburger menu (left) -->
  <div class="menu-wrap">
    <button class="menu-btn" id="menuBtn" title="Menu">&#x2630;</button>
    <div id="dropMenu">
      <button class="drop-item accent" id="dropLogin">&#x1F511; Login</button>
      <div id="dropUserInfo" style="display:none;padding:8px 18px">
        <div style="color:var(--blue);font-family:var(--mono);font-size:11px" id="dropUserName"></div>
        <div style="color:var(--dim);font-family:var(--mono);font-size:10px" id="dropUserTier"></div>
      </div>
      <button class="drop-item" id="dropLogout" style="display:none">&#x23FB; Logout</button>
      <div class="drop-sep"></div>
      <button class="drop-item" id="dropGraph">&#x2B21; Knowledge Graph</button>
      <div class="drop-sep"></div>
      <button class="drop-item" onclick="openPage('support')">&#x2709; Support</button>
      <button class="drop-item" onclick="openPage('faq')">&#x3F; FAQ</button>
    </div>
  </div>
  <!-- AI name centered -->
  <div class="brand-name">CORTANA</div>
  <div class="status-bar">
    <div class="pulse-badge bg-badge" id="bgBadge">
      <div class="dot" style="background:var(--yellow);box-shadow:0 0 7px var(--yellow)"></div>THINKING
    </div>
    <div class="badge">&#x2316;&nbsp;<span id="turnCount" style="color:var(--blue)">0</span>&nbsp;<span style="color:var(--dim);font-size:9px">asked</span></div>
    <!-- Hidden compatibility stubs for JS -->
    <span id="connDot" style="display:none"></span>
    <span id="connLabel" style="display:none"></span>
    <span id="providerBadge" style="display:none"><span id="providerLabel"></span></span>
    <div id="userBar">
      <span id="userNameLabel" style="color:var(--text)"></span>
      <span class="user-tier" id="userTierLabel"></span>
      <span id="userLimitLabel"></span>
    </div>
    <button id="csPanelToggle" title="Toggle consciousness stream">&#x25C6; MIND</button>
  </div>
</header>

<!-- Auth modal -->
<div id="authModal">
  <div class="auth-box">
    <button class="auth-close" onclick="closeAuth()" title="Close">&#x2715;</button>
    <div class="auth-title" id="authTitle">CORTANA ACCESS</div>
    <div class="auth-tabs">
      <button class="auth-tab active" id="tabLogin" onclick="switchTab('login')">Login</button>
      <button class="auth-tab" id="tabRegister" onclick="switchTab('register')">Register</button>
    </div>

    <!-- Login form -->
    <div id="formLogin">
      <div class="auth-field"><label>Username</label><input id="loginUser" type="text" autocomplete="username"></div>
      <div class="auth-field"><label>Password</label><input id="loginPass" type="password" autocomplete="current-password"></div>
      <button class="auth-btn" onclick="doLogin()">Login</button>
      <div style="text-align:right"><span class="auth-link" onclick="showForgot()">Forgot password?</span></div>
    </div>

    <!-- Forgot password form -->
    <div id="formForgot" style="display:none">
      <div style="color:var(--dim);font-size:11px;font-family:var(--mono);margin-bottom:14px;line-height:1.5">
        Enter your username or email. A reset token will be generated.
      </div>
      <div class="auth-field"><label>Username or Email</label><input id="forgotVal" type="text" autocomplete="off"></div>
      <button class="auth-btn" onclick="doForgot()">Generate Reset Token</button>
      <div id="forgotToken" style="display:none;margin-top:14px;padding:10px 13px;background:rgba(0,185,255,0.06);border:1px solid var(--border);border-radius:9px;font-family:var(--mono);font-size:11px;word-break:break-all;color:var(--blue)"></div>
      <div id="formReset" style="display:none;margin-top:14px">
        <div class="auth-field"><label>Reset Token</label><input id="resetToken" type="text" autocomplete="off"></div>
        <div class="auth-field"><label>New Password</label><input id="resetPwd" type="password" autocomplete="new-password"></div>
        <button class="auth-btn" onclick="doReset()">Set New Password</button>
      </div>
      <div style="margin-top:10px"><span class="auth-link" onclick="hideForgot()">&#x2190; Back to login</span></div>
    </div>

    <!-- Register form -->
    <div id="formRegister" style="display:none">
      <div class="auth-field"><label>Username</label><input id="regUser" type="text" autocomplete="username"></div>
      <div class="auth-field"><label>Email (optional)</label><input id="regEmail" type="email" autocomplete="email"></div>
      <div class="auth-field"><label>Password</label><input id="regPass" type="password" autocomplete="new-password"></div>
      <button class="auth-btn" onclick="doRegister()">Create Free Account</button>
      <div style="margin-top:18px;border-top:1px solid var(--border);padding-top:16px">
        <div style="color:var(--dim);font-size:10px;font-family:var(--mono);margin-bottom:10px;letter-spacing:1px">UPGRADE TIER</div>
        <div id="tierBtns"><div style="color:var(--dim);font-size:11px;font-family:var(--mono)">Loading\u2026</div></div>
      </div>
    </div>

    <div class="auth-err" id="authErr"></div>
    <div style="margin-top:18px;border-top:1px solid var(--border);padding-top:14px;text-align:center">
      <span class="auth-skip" onclick="closeAuth()">&#x25B7; Continue as guest &mdash; 5 free messages</span>
    </div>
  </div>
</div>

<!-- Forced password change modal (45-day expiry) -->
<div id="changePwdModal">
  <div class="auth-box">
    <div class="auth-title" style="color:var(--yellow)">PASSWORD EXPIRED</div>
    <div style="color:var(--dim);font-size:12px;font-family:var(--mono);margin-bottom:18px;line-height:1.6;text-align:center">
      Your password is over 45 days old and must be changed before continuing.
    </div>
    <div class="auth-field"><label>New Password</label><input id="chgPwd1" type="password" autocomplete="new-password"></div>
    <div class="auth-field"><label>Confirm New Password</label><input id="chgPwd2" type="password" autocomplete="new-password"></div>
    <button class="auth-btn" onclick="doChangePwd()">Update Password</button>
    <div class="auth-err" id="chgPwdErr"></div>
  </div>
</div>

<!-- Full-page overlay (Support / FAQ) -->
<div id="overlayPage">
  <div class="op-box">
    <button class="op-close" onclick="closePage()">&#x2715;</button>
    <div class="op-title" id="opTitle">SUPPORT</div>
    <div id="opContent"></div>
  </div>
</div>

<!-- Glass chat overlay -->
<div class="chat-overlay" id="chatOverlay">
  <div class="chat-handle" id="chatHandle" title="Toggle chat size">
    <div class="chat-handle-pill"></div>
  </div>
  <div id="messages"></div>
  <div class="input-area">
    <textarea id="input" rows="1" placeholder="Talk to Cortana\u2026" disabled></textarea>
    <button id="camToggleBtn" title="Toggle webcam">&#x1F4F7;</button>
    <button id="screenToggleBtn" title="Share screen with Cortana">&#x1F5A5;</button>
    <button id="send" disabled>Send</button>
  </div>
  <div id="versionTag">v2.1.0 &mdash; Neural Nexus AGI</div>
</div>

<!-- Webcam panel -->
<div id="camPanel">
  <div class="cam-label">WEBCAM</div>
  <video id="camVideo" autoplay muted playsinline></video>
  <div class="cam-controls">
    <button class="cam-btn" id="camSnapBtn">&#x1F4F8; Snapshot</button>
    <button class="cam-btn" id="camAutoBtn">&#x23F1; Auto</button>
  </div>
  <div class="cam-status" id="camStatus">Camera ready</div>
</div>

<!-- Screen share panel -->
<div id="screenPanel">
  <div class="cam-label">SCREEN SHARE</div>
  <video id="screenVideo" autoplay muted playsinline></video>
  <div class="cam-controls">
    <button class="cam-btn" id="screenSnapBtn">&#x1F4F8; Snapshot</button>
    <button class="cam-btn" id="screenAutoBtn">&#x23F1; Auto</button>
  </div>
  <div class="cam-status" id="screenStatus">Share your screen</div>
</div>

<!-- Toast -->
<div id="toast"></div>

<!-- Knowledge graph modal -->
<div id="graphModal">
  <div class="graph-inner">
    <div class="graph-header">
      <span class="graph-title">Knowledge Graph</span>
      <button class="graph-close" id="graphClose">&#x2715;</button>
    </div>
    <div id="graphContent"></div>
  </div>
</div>



<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}
</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader }    from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Renderer ──
const canvas = document.getElementById('glCanvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace  = THREE.SRGBColorSpace;
renderer.toneMapping       = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

// ── Scene / Camera ──
const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(0, 0, 5.5);

// ── Drag-to-rotate (clamped, damped) ──
const controls = new OrbitControls(camera, canvas);
controls.enableDamping   = true;
controls.dampingFactor   = 0.06;
controls.enablePan       = false;
controls.enableZoom      = false;
controls.minAzimuthAngle = -Math.PI / 3;
controls.maxAzimuthAngle =  Math.PI / 3;
controls.minPolarAngle   =  Math.PI / 4;
controls.maxPolarAngle   =  Math.PI / 1.4;

// ── Lighting ──
const ambient   = new THREE.AmbientLight(0x0a1530, 1.5);
scene.add(ambient);
const rimLight  = new THREE.DirectionalLight(0x0066ff, 4.0);
rimLight.position.set(-3, 2, -3);
scene.add(rimLight);
const fillLight = new THREE.DirectionalLight(0x00d8ff, 2.0);
fillLight.position.set(2, 1, 4);
scene.add(fillLight);
const topLight  = new THREE.DirectionalLight(0xffffff, 1.0);
topLight.position.set(0, 5, 0);
scene.add(topLight);

// ── Load GLB (original materials) ──
const modelPivot = new THREE.Group();
scene.add(modelPivot);
let mixer = null;

function loadGLB(url) {
  // Remove old model from pivot
  while (modelPivot.children.length) modelPivot.remove(modelPivot.children[0]);
  if (mixer) { mixer.stopAllAction(); mixer = null; }
  const loader = new GLTFLoader();
  loader.load(url + '?t=' + Date.now(), (gltf) => {
    const model = gltf.scene;
    // Blender Z-up → Three.js Y-up correction
    model.rotation.x = -Math.PI / 2;
    model.updateMatrixWorld(true);
    const box    = new THREE.Box3().setFromObject(model);
    const size   = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const scale  = 3.0 / Math.max(size.x, size.y, size.z);
    model.scale.setScalar(scale);
    model.position.sub(centre.multiplyScalar(scale));
    modelPivot.add(model);
    if (gltf.animations && gltf.animations.length > 0) {
      mixer = new THREE.AnimationMixer(model);
      mixer.clipAction(gltf.animations[0]).play();
    }
  }, undefined, (err) => {
    console.error('[GLB] Failed to load:', url, err);
  });
}

// Load the Meshy AI biped model
loadGLB('/static/cortana.glb');

// ═══════════════════════════════════════════════════════════
//  PROCEDURAL ANIMATION + GESTURE SYSTEM
// ═══════════════════════════════════════════════════════════
const DEG = Math.PI / 180;
function rand(a, b) { return a + Math.random() * (b - a); }
function easeInOut(t) { return t < 0.5 ? 2*t*t : -1+(4-2*t)*t; }
function smoothstep(t) { t=Math.max(0,Math.min(1,t)); return t*t*(3-2*t); }

// ── Base animation state ──
const anim = {
  posY: 0, rotX: 0, rotZ: 0, rotY: 0, scl: 1,
  tPosY: 0, tRotX: 0, tRotZ: 0, tRotY: 0, tScl: 1,
  breathPhase: 0, breathSpeed: 1.0, breathAmp: 1.0,
  swayPhase: 0,
  isTalking: false, talkPhase: 0, talkDecay: 0,
  yawnActive: false, yawnPhase: 0,
  emotion: 'idle', emotionTimer: 0,
  idleTimer: 0, nextIdle: rand(5, 10),
};

// ── Emotion table ──
const EMOTIONS = {
  idle:      { rX:  0, rZ:  0, rY:  0, pY:  0.000, sc: 1.00, bSpd: 1.0, bAmp: 1.0, dur: 0,  rim: 0x0044aa, rimI: 3.5, fill: 0x00d8ff, fillI: 1.5 },
  smile:     { rX: -3, rZ:  0, rY:  0, pY:  0.020, sc: 1.02, bSpd: 1.1, bAmp: 0.9, dur: 5,  rim: 0x0088ff, rimI: 5.0, fill: 0x00eeff, fillI: 2.5 },
  sad:       { rX:  8, rZ:  2, rY:  0, pY: -0.020, sc: 0.98, bSpd: 0.6, bAmp: 0.8, dur: 5,  rim: 0x001144, rimI: 1.5, fill: 0x001a44, fillI: 0.8 },
  think:     { rX:  2, rZ: -8, rY:  5, pY:  0.000, sc: 1.00, bSpd: 1.3, bAmp: 1.1, dur: 0,  rim: 0x7722ff, rimI: 6.0, fill: 0x9944ff, fillI: 2.0 },
  surprised: { rX: -9, rZ:  0, rY:  0, pY:  0.040, sc: 1.04, bSpd: 1.7, bAmp: 1.3, dur: 4,  rim: 0x00ffcc, rimI: 6.0, fill: 0x00ffaa, fillI: 3.0 },
  frown:     { rX:  6, rZ:  1, rY:  0, pY: -0.015, sc: 0.99, bSpd: 0.8, bAmp: 0.9, dur: 5,  rim: 0x220011, rimI: 1.2, fill: 0x001a44, fillI: 0.6 },
  laugh:     { rX:  0, rZ:  0, rY:  0, pY:  0.010, sc: 1.03, bSpd: 2.0, bAmp: 1.6, dur: 5,  rim: 0x00d4ff, rimI: 6.5, fill: 0x00ffee, fillI: 3.5 },
};

function applyEmotion(name) {
  const e = EMOTIONS[name] || EMOTIONS.idle;
  anim.emotion = name; anim.tRotX = e.rX*DEG; anim.tRotZ = e.rZ*DEG;
  anim.tRotY = e.rY*DEG; anim.tPosY = e.pY; anim.tScl = e.sc;
  anim.breathSpeed = e.bSpd; anim.breathAmp = e.bAmp; anim.emotionTimer = e.dur;
  rimLight.color.setHex(e.rim);   rimLight.intensity  = e.rimI;
  fillLight.color.setHex(e.fill); fillLight.intensity = e.fillI;
}

// ── Gesture system — time-limited overlays on top of base pose ──
const gesture = {
  _active: null,
  _phase:  0,
  _dur:    0,
  _fn:     null,

  // Each gesture returns {rX, rZ, rY, pY, scl} offsets
  _clips: {
    // Full-body wave: lateral oscillation + upward bounce
    wave: { dur: 3.2, fn(t) {
      const env  = smoothstep(t/0.3) * smoothstep((3.2-t)/0.4);
      const freq = Math.sin(t * 6.5) * env;
      const bounce = Math.abs(Math.sin(t * 3.2)) * 0.032 * env;
      return { rZ: freq*0.30, rX: -0.06*env, pY: bounce, scl: 1+0.015*env };
    }},

    // Crossed arms: forward lean + rhythmic slow sway (held until released)
    crossed_arms: { dur: 0, fn(t) {
      const sway = Math.sin(t * 0.5) * 0.022;
      const bob  = Math.sin(t * 0.9) * 0.008;
      return { rX: 0.14, pY: -0.025 + bob, rZ: sway, scl: 0.98 };
    }},

    // Hand in air / question pose: lean back + upward reach + bob
    question_raise: { dur: 3.0, fn(t) {
      const rise  = smoothstep(Math.min(t/0.4, 1));
      const fall  = smoothstep(Math.max((t-2.5)/0.5, 0));
      const amt   = rise - fall;
      const bob   = Math.sin(t * 5.5) * 0.014 * amt;
      return { rX: -0.15*amt, pY: 0.06*amt + bob, rZ: -0.10*amt, scl: 1+0.02*amt };
    }},

    // Nod: quick forward/back head bob (x2)
    nod: { dur: 1.4, fn(t) {
      const bob = Math.sin(t * Math.PI * 2.8) * smoothstep(1-t/1.4) * 0.14;
      return { rX: bob };
    }},

    // Thinking / chin scratch: side tilt + slow contemplative oscillation
    thinking_pose: { dur: 0, fn(t) {
      const osc = Math.sin(t * 0.7 + 0.4) * 0.035;
      return { rX: 0.05, rZ: -0.16 + osc, pY: 0.01, scl: 0.99 };
    }},

    // Excited bounce: rapid upward bounces
    excited: { dur: 1.8, fn(t) {
      const env    = smoothstep(1 - t/1.8);
      const bounce = Math.abs(Math.sin(t * 9.0)) * 0.042 * env;
      const sway   = Math.sin(t * 4.5) * 0.10 * env;
      return { pY: bounce, rZ: sway, scl: 1 + 0.02*env };
    }},

    // Shrug: rise + tilt left/right + drop
    shrug: { dur: 2.2, fn(t) {
      const up   = smoothstep(Math.min(t/0.35, 1));
      const dn   = smoothstep(Math.max((t-1.6)/0.6, 0));
      const amt  = up - dn;
      const tilt = Math.sin(t*3.5)*0.06*amt;
      return { pY: 0.04*amt, rZ: tilt, scl: 1+0.01*amt };
    }},

    // Turn and look: slow Y rotation + return
    look_around: { dur: 2.8, fn(t) {
      const dir = t < 1.4 ? 1 : -1;
      const ph  = t < 1.4 ? t/1.4 : (t-1.4)/1.4;
      const ang = dir * easeInOut(ph < 0.5 ? ph*2 : 2-ph*2) * 0.38;
      return { rY: ang };
    }},
  },

  play(name) {
    const clip = this._clips[name];
    if (!clip) return;
    this._active = name;
    this._phase  = 0;
    this._dur    = clip.dur;
    this._fn     = clip.fn;
  },

  stop() { this._active = null; this._phase = 0; this._fn = null; },

  tick(delta) {
    if (!this._fn) return { rX:0, rZ:0, rY:0, pY:0, scl:0 };
    this._phase += delta;
    if (this._dur > 0 && this._phase >= this._dur) {
      this.stop();
      return { rX:0, rZ:0, rY:0, pY:0, scl:0 };
    }
    const v = this._fn(this._phase);
    return {
      rX: v.rX||0, rZ: v.rZ||0, rY: v.rY||0,
      pY: v.pY||0, scl: v.scl||0,
    };
  },
};

// ── Detect gesture from response text + emotion ──
function detectGesture(text, emotion) {
  const lo = text.toLowerCase();
  const tail = lo.slice(-120);
  if (/\b(hello|hi\b|hey\b|wave|greet|goodbye|bye\b|welcome)/.test(lo)) return 'wave';
  if (/\?/.test(tail) && /\b(what|how|why|when|where|which|could you|can you|do you|would you)/.test(lo)) return 'question_raise';
  if (/\b(interesting|exciting|great news|amazing|wonderful|excellent|brilliant)/.test(lo)) return 'excited';
  if (/\b(i suppose|not sure|perhaps|maybe|unclear|uncertain|shrug)/.test(lo)) return 'shrug';
  if (emotion === 'think') return 'thinking_pose';
  if (emotion === 'smile' || emotion === 'laugh') return 'excited';
  if (/\b(yes|correct|exactly|right|agreed|certainly|absolutely)/.test(lo)) return 'nod';
  return null;
}

// ── Idle micro-behaviors ──
function triggerIdleBehavior() {
  if (anim.emotion !== 'idle' || anim.isTalking || gesture._active) return;
  const pick = Math.floor(rand(0, 8));
  switch (pick) {
    case 0: anim.tRotY = -10*DEG; setTimeout(()=>{ if(anim.emotion==='idle') anim.tRotY=0; }, rand(1200,2200)); break;
    case 1: anim.tRotY =  10*DEG; setTimeout(()=>{ if(anim.emotion==='idle') anim.tRotY=0; }, rand(1200,2200)); break;
    case 2:
      anim.tRotX = 5*DEG;
      setTimeout(()=>{ anim.tRotX=-3*DEG; }, 350);
      setTimeout(()=>{ if(anim.emotion==='idle') anim.tRotX=0; }, 900);
      break;
    case 3:
      anim.tRotZ = (Math.random()>0.5?1:-1)*6*DEG;
      setTimeout(()=>{ if(anim.emotion==='idle') anim.tRotZ=0; }, rand(1800,2800));
      break;
    case 4:
      anim.tRotX=-9*DEG; anim.yawnActive=true; anim.yawnPhase=0;
      anim.breathSpeed=0.38; anim.breathAmp=2.3;
      setTimeout(()=>{ if(anim.emotion==='idle'){ anim.tRotX=0; anim.breathSpeed=1.0; anim.breathAmp=1.0; } }, 3400);
      break;
    case 5: gesture.play('look_around'); break;
    case 6: gesture.play('shrug'); break;
    case 7:
      // Weight shift — lean to one side
      const side = Math.random()>0.5?1:-1;
      anim.tRotZ = side*8*DEG; anim.tPosY = -0.01;
      setTimeout(()=>{ if(anim.emotion==='idle'){anim.tRotZ=0;anim.tPosY=0;} }, rand(1500,2800));
      break;
  }
}

// ── Main animation tick ──
function tickAnimation(delta) {
  const k = 1 - Math.pow(0.0005, delta);

  // Breathing
  anim.breathPhase += delta * 0.42 * anim.breathSpeed;
  const breath = Math.sin(anim.breathPhase) * 0.018 * anim.breathAmp;

  // Yawn
  let yawnY = 0;
  if (anim.yawnActive) {
    anim.yawnPhase += delta * 0.62;
    if (anim.yawnPhase >= Math.PI) { anim.yawnActive=false; anim.yawnPhase=0; }
    else yawnY = Math.sin(anim.yawnPhase) * 0.05;
  }

  // Ambient sway
  anim.swayPhase += delta * 0.16;
  const swayX = Math.sin(anim.swayPhase*0.9+1.3) * 0.006;
  const swayZ = Math.cos(anim.swayPhase*0.7)      * 0.004;

  // Talking micro-motion
  let talkX = 0, talkY = 0;
  if (anim.isTalking || anim.talkDecay > 0.005) {
    anim.talkDecay = anim.isTalking
      ? Math.min(anim.talkDecay+delta*4, 1.0)
      : Math.max(anim.talkDecay-delta*2.5, 0.0);
    anim.talkPhase += delta * 8.5;
    const t = Math.sin(anim.talkPhase) * anim.talkDecay;
    talkX = t*0.010; talkY = Math.abs(t)*0.006;
  }

  const laughY = anim.emotion==='laugh' ? Math.abs(Math.sin(anim.breathPhase*3.5))*0.038 : 0;

  // Gesture overlay
  const g = gesture.tick(delta);

  // Lerp base pose
  anim.rotX += (anim.tRotX + swayX + talkX + g.rX - anim.rotX) * k * 5;
  anim.rotZ += (anim.tRotZ + swayZ         + g.rZ - anim.rotZ) * k * 5;
  anim.rotY += (anim.tRotY                 + g.rY - anim.rotY) * k * 4;
  anim.posY += (anim.tPosY + breath + yawnY + talkY + laughY + g.pY - anim.posY) * k * 7;
  anim.scl  += (anim.tScl                  + g.scl - anim.scl) * k * 4;

  modelPivot.rotation.x = anim.rotX;
  modelPivot.rotation.y = anim.rotY + talkX * 0.35;
  modelPivot.rotation.z = anim.rotZ;
  modelPivot.position.y = anim.posY;
  modelPivot.scale.setScalar(Math.max(0.5, anim.scl));

  // Emotion timeout
  if (anim.emotionTimer > 0) {
    anim.emotionTimer -= delta;
    if (anim.emotionTimer <= 0) applyEmotion('idle');
  }

  // Idle behaviors
  anim.idleTimer += delta;
  if (anim.idleTimer >= anim.nextIdle) {
    anim.idleTimer=0; anim.nextIdle=rand(6,15);
    triggerIdleBehavior();
  }
}

// ── Public API ──
window.setExpression     = (n) => applyEmotion(n);
window.triggerExpression = (n) => {
  applyEmotion(n);
  clearTimeout(window._exprT);
  if (n !== 'idle') window._exprT = setTimeout(() => applyEmotion('idle'), 4000);
};
window.playGesture       = (n) => gesture.play(n);
window.stopGesture       = ()  => gesture.stop();
window.detectAndPlay     = (text, emotion) => {
  const g = detectGesture(text, emotion);
  if (g) gesture.play(g);
};
window._startTalking = () => { anim.isTalking=true;  anim.talkDecay=0.3; gesture.stop(); };
window._stopTalking  = () => { anim.isTalking=false; };
applyEmotion('idle');

// ── Resize ──
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Render loop ──
const clock = new THREE.Clock();
(function animate() {
  requestAnimationFrame(animate);
  const delta = Math.min(clock.getDelta(), 0.1);
  tickAnimation(delta);
  if (mixer) mixer.update(delta);
  controls.update();
  renderer.render(scene, camera);
})();
</script>

<script>
// ── Expression stubs (Three.js module above overrides these once loaded) ──
if (!window.setExpression) window.setExpression = function(name) {};
if (!window.triggerExpression) window.triggerExpression = function(name) {
  window.setExpression(name);
  clearTimeout(window._exprT);
  if (name !== 'idle') window._exprT = setTimeout(() => window.setExpression('idle'), 4000);
};


// ================================================================
//  CHAT OVERLAY TOGGLE
// ================================================================
const CHAT_TALL='44vh', CHAT_SHORT='56px';
let chatExpanded=true;
document.getElementById('chatHandle').addEventListener('click',()=>{
  chatExpanded=!chatExpanded;
  const h=chatExpanded?CHAT_TALL:CHAT_SHORT;
  document.documentElement.style.setProperty('--chat-h',h);
});

// ================================================================
//  SESSION ID + AUTH
// ================================================================
function getSessionId(){
  let s=localStorage.getItem('cortana_session_id');
  if(!s){s=crypto.randomUUID();localStorage.setItem('cortana_session_id',s);}
  return s;
}
const SESSION_ID=getSessionId();
let authToken=localStorage.getItem('cortana_token')||'';
let currentUser=null;

// ================================================================
//  BROWSER-SIDE LONG-TERM MEMORY
//  Conversation turns stored in localStorage per session.
//  Auto-expires after 30 days of inactivity per session.
// ================================================================
const _MEM_KEY    = 'cortana_memory_' + SESSION_ID;
const _ACTIVE_KEY = 'cortana_last_active_' + SESSION_ID;
const _MEM_EXPIRY = 30 * 24 * 60 * 60 * 1000; // 30 days in ms
const _MEM_MAX_TURNS = 300;

(function _memCleanup(){
  // On every page load, sweep expired sessions
  try {
    Object.keys(localStorage).forEach(k => {
      if (!k.startsWith('cortana_last_active_')) return;
      const last = parseInt(localStorage.getItem(k) || '0', 10);
      if (Date.now() - last > _MEM_EXPIRY) {
        const sid = k.replace('cortana_last_active_', '');
        localStorage.removeItem('cortana_memory_' + sid);
        localStorage.removeItem('cortana_last_active_' + sid);
        localStorage.removeItem('cortana_session_id'); // also clear if it's this session
      }
    });
  } catch(e) {}
})();

function _memTouch() {
  try { localStorage.setItem(_ACTIVE_KEY, String(Date.now())); } catch(e) {}
}

function _memSaveTurn(role, content) {
  try {
    const raw = localStorage.getItem(_MEM_KEY);
    const turns = raw ? JSON.parse(raw) : [];
    turns.push({ role, content, ts: Date.now() });
    if (turns.length > _MEM_MAX_TURNS) turns.splice(0, turns.length - _MEM_MAX_TURNS);
    localStorage.setItem(_MEM_KEY, JSON.stringify(turns));
    _memTouch();
  } catch(e) {}
}

function _memLoad() {
  try {
    const raw = localStorage.getItem(_MEM_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch(e) { return []; }
}

function _memGetStats() {
  const turns = _memLoad();
  const last = parseInt(localStorage.getItem(_ACTIVE_KEY) || '0', 10);
  const daysLeft = last
    ? Math.max(0, Math.round((_MEM_EXPIRY - (Date.now() - last)) / 86400000))
    : 30;
  return { turns: turns.length, daysLeft };
}

function switchTab(tab){
  document.getElementById('formLogin').style.display   =tab==='login'   ?'':'none';
  document.getElementById('formRegister').style.display=tab==='register'?'':'none';
  document.getElementById('tabLogin').classList.toggle('active',   tab==='login');
  document.getElementById('tabRegister').classList.toggle('active',tab==='register');
  document.getElementById('authErr').textContent='';
}
function openAuth(){document.getElementById('authModal').classList.add('open');loadTiers();}
function closeAuth(){document.getElementById('authModal').classList.remove('open');}

// ================================================================
//  DROPDOWN MENU
// ================================================================
const menuBtn = document.getElementById('menuBtn');
const dropMenu = document.getElementById('dropMenu');
menuBtn.addEventListener('click',(e)=>{e.stopPropagation();dropMenu.classList.toggle('open');});
document.addEventListener('click',()=>dropMenu.classList.remove('open'));
dropMenu.addEventListener('click',(e)=>e.stopPropagation());

document.getElementById('dropLogin').onclick=()=>{dropMenu.classList.remove('open');openAuth();};
document.getElementById('dropLogout').onclick=()=>{dropMenu.classList.remove('open');doLogout();};
document.getElementById('dropGraph').onclick=()=>{
  dropMenu.classList.remove('open');
  fetch('/api/graph').then(r=>r.json()).then(data=>{
    const gc=document.getElementById('graphContent');
    if(!data.nodes.length){
      gc.innerHTML='<span style="color:var(--dim)">No knowledge yet \u2014 start chatting!</span>';
    } else {
      let html='<div style="color:var(--blue);margin-bottom:14px;font-weight:600">Concepts ('+data.nodes.length+')</div>';
      data.nodes.forEach(n=>{
        const b=Math.round(n.confidence*10);
        html+=`<div style="margin-bottom:10px"><span style="color:var(--blue)">${n.topic}</span>
          <span style="color:var(--dim);margin-left:8px;font-size:11px">${'\u2588'.repeat(b)}${'\u2591'.repeat(10-b)} ${(n.confidence*100).toFixed(0)}%</span>
          <div style="color:var(--text);font-size:12px;margin-top:2px">${n.summary}</div></div>`;
      });
      if(data.edges.length){
        html+='<div style="margin:16px 0 10px;color:var(--blue);font-weight:600">Relations ('+data.edges.length+')</div>';
        data.edges.forEach(e=>{
          html+=`<div style="margin-bottom:5px;font-size:12px;color:var(--dim)">${e.source} <span style="color:var(--blue)">${e.relation}</span> ${e.target}</div>`;
        });
      }
      gc.innerHTML=html;
    }
    document.getElementById('graphModal').style.display='flex';
  });
};
document.getElementById('graphClose').onclick=()=>document.getElementById('graphModal').style.display='none';
document.getElementById('graphModal').onclick=function(e){if(e.target===this)this.style.display='none';};

// ================================================================
//  SUPPORT / FAQ OVERLAYS
// ================================================================
const _SUPPORT_HTML=`
<div class="op-section"><div class="op-h">Contact</div>
<div class="op-p">For account issues, billing, or technical problems, email us at:<br>
<a href="mailto:support@cortanas.org" style="color:var(--blue);text-decoration:none">support@cortanas.org</a></div></div>
<div class="op-section"><div class="op-h">Response Time</div>
<div class="op-p">We aim to respond within 24 hours on business days. Please include your username and a description of the issue.</div></div>
<div class="op-section"><div class="op-h">Subscriptions</div>
<div class="op-p">Monthly plans: <b>Pro $5/mo</b> (400 msg/2h) and <b>Premium $15/mo</b> (4000 msg/2h). Pay in ETH to the address shown in <b style="color:var(--blue)">/api/v1/wallet</b>, then paste your transaction hash into the <b>Upgrade Tier</b> form in the login panel. Activation is instant once the transaction is confirmed on-chain.</div></div>
<div class="op-section"><div class="op-h">Password Reset</div>
<div class="op-p">Use the <b style="color:var(--blue)">Forgot password?</b> link inside the Login panel. A reset token is generated immediately \u2014 no email required.</div></div>`;

const _FAQ_HTML=`
<div class="op-faq-q">What is Cortana AI?</div>
<div class="op-faq-a">An agentic AI assistant with multi-provider LLM routing, persistent memory, web search, and a self-improvement loop. It runs a 14-layer reasoning pipeline on each message.</div>
<div class="op-faq-q">Is my conversation private?</div>
<div class="op-faq-a">Conversations are stored server-side linked to your browser session or account. They are used only to provide context in future conversations. No third-party sharing.</div>
<div class="op-faq-q">How do message limits work?</div>
<div class="op-faq-a">Free accounts get 40 messages per 2-hour rolling window. Pro accounts get 400, Premium 2000. The window resets 2 hours after your first message in that window.</div>
<div class="op-faq-q">Why does my password expire every 45 days?</div>
<div class="op-faq-a">Regular password rotation limits the impact of any credential exposure. You\u2019ll be prompted to change it on your next login after 45 days.</div>
<div class="op-faq-q">Can Cortana see me via the webcam?</div>
<div class="op-faq-a">Only when you explicitly open the webcam panel and click Snapshot or enable Auto mode. No camera access occurs in the background.</div>
<div class="op-faq-q">What AI models power Cortana?</div>
<div class="op-faq-a">Cortana routes across Groq (Llama 3.3 70B), OpenRouter (Llama 3.3 70B), and Gemini 2.0 Flash. It automatically rotates providers on rate limits.</div>`;

function openPage(type){
  document.getElementById('opTitle').textContent=type==='faq'?'FAQ':'SUPPORT';
  document.getElementById('opContent').innerHTML=type==='faq'?_FAQ_HTML:_SUPPORT_HTML;
  document.getElementById('overlayPage').classList.add('open');
  dropMenu.classList.remove('open');
}
function closePage(){document.getElementById('overlayPage').classList.remove('open');}
window.openPage=openPage; window.closePage=closePage;

// ================================================================
//  AUTH
// ================================================================
let _tiers={};
let _payWallet='';
async function loadTiers(){
  try{
    const data=await fetch('/api/tiers').then(r=>r.json());
    _tiers=data;
    try{ const w=await fetch('/api/v1/wallet').then(r=>r.json()); _payWallet=w.address||''; }catch(e){}
    const container=document.getElementById('tierBtns');
    if(!container) return;
    container.innerHTML='';
    const labels={pro:'Priority routing',premium:'Highest priority + vision'};
    Object.entries(data).forEach(([name,info])=>{
      if(name==='free'||name==='admin') return;
      const btn=document.createElement('button');
      btn.className='tier-btn';
      btn.onclick=()=>showSubscribeForm(name);
      const label=name.charAt(0).toUpperCase()+name.slice(1);
      btn.innerHTML=`${label} \u2014 ${info.daily_limit} msg/2h \u2014 ${labels[name]||''}<span>$${info.price_usd}/mo</span>`;
      container.appendChild(btn);
    });
  }catch(e){}
}
function showSubscribeForm(tier){
  const info=_tiers[tier];
  if(!info) return;
  const label=tier.charAt(0).toUpperCase()+tier.slice(1);
  const err=document.getElementById('authErr');
  err.style.color='var(--blue)';
  err.innerHTML=
    `<b>${label} \u2014 $${info.price_usd}/month</b><br>`+
    `Send <b>${info.price_eth||'see /api/v1/wallet'} ETH</b> to:<br>`+
    `<span style="font-size:9px;word-break:break-all;color:var(--text)">${_payWallet||'(loading\u2026)'}</span><br><br>`+
    `<input id="subTxHash" placeholder="Paste ETH tx hash (0x\u2026)" `+
    `style="width:100%;padding:5px 8px;background:rgba(0,185,255,0.06);border:1px solid var(--border);`+
    `border-radius:6px;color:var(--text);font-family:var(--mono);font-size:10px;margin-bottom:6px"><br>`+
    `<button onclick="submitSubscription('${tier}')" `+
    `style="width:100%;padding:6px;background:rgba(0,185,255,0.12);border:1px solid var(--border);`+
    `border-radius:6px;color:var(--blue);font-family:var(--mono);font-size:10px;cursor:pointer">`+
    `Activate ${label}</button>`;
}
async function submitSubscription(tier){
  const txHash=(document.getElementById('subTxHash')||{}).value||'';
  if(!txHash.startsWith('0x')){
    document.getElementById('authErr').textContent='Enter a valid tx hash starting with 0x';
    return;
  }
  if(!authToken){document.getElementById('authErr').textContent='Log in first to subscribe';return;}
  try{
    const r=await fetch('/api/auth/subscribe',{method:'POST',
      headers:{'Content-Type':'application/json','X-Session-Token':authToken},
      body:JSON.stringify({tier,tx_hash:txHash})});
    const d=await r.json();
    const err=document.getElementById('authErr');
    if(d.ok){
      err.style.color='var(--green)';
      const exp=d.subscription_expires?d.subscription_expires.slice(0,10):'30 days';
      err.textContent=`${tier.charAt(0).toUpperCase()+tier.slice(1)} active until ${exp}!`;
    } else {
      err.style.color='var(--red)';
      err.textContent=d.error||'Subscription failed — check tx hash';
    }
  }catch(e){
    document.getElementById('authErr').textContent='Network error \u2014 try again';
  }
}

function switchTab(tab){
  document.getElementById('formLogin').style.display=tab==='login'?'':'none';
  document.getElementById('formRegister').style.display=tab==='register'?'':'none';
  document.getElementById('formForgot').style.display='none';
  document.getElementById('tabLogin').classList.toggle('active',tab==='login');
  document.getElementById('tabRegister').classList.toggle('active',tab==='register');
  document.getElementById('authErr').textContent='';
  document.getElementById('authTitle').textContent='CORTANA ACCESS';
}
function showForgot(){
  document.getElementById('formLogin').style.display='none';
  document.getElementById('formForgot').style.display='';
  document.getElementById('authTitle').textContent='PASSWORD RESET';
  document.getElementById('authErr').textContent='';
}
function hideForgot(){switchTab('login');}
window.showForgot=showForgot; window.hideForgot=hideForgot;

async function doForgot(){
  const val=document.getElementById('forgotVal').value.trim();
  if(!val){document.getElementById('authErr').textContent='Enter your username or email';return;}
  document.getElementById('authErr').textContent='';
  try{
    const d=await fetch('/api/auth/forgot-password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username_or_email:val})}).then(r=>r.json());
    // Token is delivered out-of-band (email); never displayed here
    document.getElementById('authErr').style.color='var(--blue)';
    document.getElementById('authErr').textContent=d.message||'If that account exists, a reset link has been sent.';
  }catch(e){document.getElementById('authErr').textContent='Network error';}
}
window.doForgot=doForgot;

async function doReset(){
  const token=document.getElementById('resetToken').value.trim();
  const pwd=document.getElementById('resetPwd').value;
  document.getElementById('authErr').style.color='var(--red)';
  document.getElementById('authErr').textContent='';
  try{
    const d=await fetch('/api/auth/reset-password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token,new_password:pwd})}).then(r=>r.json());
    if(!d.ok){document.getElementById('authErr').textContent=d.error;return;}
    document.getElementById('authErr').style.color='var(--green)';
    document.getElementById('authErr').textContent='Password reset! Please log in.';
    setTimeout(()=>switchTab('login'),1200);
  }catch(e){document.getElementById('authErr').textContent='Network error';}
}
window.doReset=doReset;

function openAuth(){document.getElementById('authModal').classList.add('open');switchTab('login');loadTiers();}
function closeAuth(){document.getElementById('authModal').classList.remove('open');}

async function doLogin(){
  const u=document.getElementById('loginUser').value.trim();
  const p=document.getElementById('loginPass').value;
  document.getElementById('authErr').style.color='var(--red)';
  document.getElementById('authErr').textContent='';
  try{
    const r=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})});
    const d=await r.json();
    if(!d.ok){document.getElementById('authErr').textContent=d.error;return;}
    authToken=d.token; localStorage.setItem('cortana_token',authToken);
    currentUser=d; updateUserBar(); closeAuth();
    if(d.password_expired) showChangePwd();
    else if(ws)ws.close();
  }catch(e){document.getElementById('authErr').textContent='Network error';}
}

async function doRegister(){
  const u=document.getElementById('regUser').value.trim();
  const p=document.getElementById('regPass').value;
  const e=document.getElementById('regEmail').value.trim();
  document.getElementById('authErr').style.color='var(--red)';
  document.getElementById('authErr').textContent='';
  try{
    const r=await fetch('/api/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p,email:e})});
    const d=await r.json();
    if(!d.ok){document.getElementById('authErr').textContent=d.error;return;}
    document.getElementById('loginUser').value=u;
    document.getElementById('loginPass').value=p;
    switchTab('login');
    document.getElementById('authErr').style.color='var(--green)';
    document.getElementById('authErr').textContent='Account created! Logging in\u2026';
    setTimeout(doLogin,800);
  }catch(e){document.getElementById('authErr').textContent='Network error';}
}

async function doLogout(){
  if(authToken) await fetch('/api/auth/logout',{method:'POST',headers:{'X-Session-Token':authToken}}).catch(()=>{});
  authToken=''; currentUser=null;
  localStorage.removeItem('cortana_token');
  updateUserBar(); if(ws)ws.close();
}

// ── Forced password change (45-day expiry) ──
function showChangePwd(){
  document.getElementById('changePwdModal').classList.add('open');
  document.getElementById('chgPwdErr').textContent='';
}
async function doChangePwd(){
  const p1=document.getElementById('chgPwd1').value;
  const p2=document.getElementById('chgPwd2').value;
  const err=document.getElementById('chgPwdErr');
  err.style.color='var(--red)'; err.textContent='';
  if(p1!==p2){err.textContent='Passwords do not match';return;}
  try{
    const d=await fetch('/api/auth/change-password',{method:'POST',
      headers:{'Content-Type':'application/json','X-Session-Token':authToken},
      body:JSON.stringify({new_password:p1})}).then(r=>r.json());
    if(!d.ok){err.textContent=d.error;return;}
    document.getElementById('changePwdModal').classList.remove('open');
    if(ws)ws.close();
  }catch(e){err.textContent='Network error';}
}
window.doChangePwd=doChangePwd;

function updateUserBar(){
  const bar=document.getElementById('userBar');
  const dlBtn=document.getElementById('dropLogin');
  const dlOut=document.getElementById('dropLogout');
  const dlInfo=document.getElementById('dropUserInfo');
  if(currentUser){
    bar.classList.add('visible');
    dlBtn.style.display='none'; dlOut.style.display=''; dlInfo.style.display='';
    document.getElementById('dropUserName').textContent=currentUser.username||'';
    document.getElementById('dropUserTier').textContent=(currentUser.tier||'free').toUpperCase();
    document.getElementById('userNameLabel').textContent=currentUser.username||'';
    document.getElementById('userTierLabel').textContent=currentUser.tier||'free';
    document.getElementById('userLimitLabel').textContent=(currentUser.daily_limit||40)+'/2h';
  } else {
    bar.classList.remove('visible');
    dlBtn.style.display=''; dlOut.style.display='none'; dlInfo.style.display='none';
  }
}

if(authToken){
  fetch('/api/auth/me',{headers:{'X-Session-Token':authToken}}).then(r=>r.json()).then(d=>{
    if(d.username){currentUser=d;updateUserBar();}
    else{authToken='';localStorage.removeItem('cortana_token');}
  }).catch(()=>{});
}

// Expose to global scope (used by inline onclick and non-module scripts)
window.switchTab   = switchTab;
window.openAuth    = openAuth;
window.closeAuth   = closeAuth;
window.doLogin     = doLogin;
window.doRegister  = doRegister;
window.showTierInfo= showTierInfo;

// ================================================================
//  WEBSOCKET + CHAT
// ================================================================
const messages   =document.getElementById('messages');
const input      =document.getElementById('input');
const sendBtn    =document.getElementById('send');
const connDot    =document.getElementById('connDot');
const connLabel  =document.getElementById('connLabel');
const providerBadge=document.getElementById('providerBadge');
const providerLabel=document.getElementById('providerLabel');
const turnCount  =document.getElementById('turnCount');
// learnBadge removed — self-improvement is a silent background process
const bgBadge    =document.getElementById('bgBadge');
const toast      =document.getElementById('toast');

let ws=null,learnTimer=null,toastTimer=null;

async function devaiRespond(id, decision, cardEl) {
  try {
    const resp = await fetch(`/api/devai/respond/${id}`, {
      method: 'POST',
      headers: {'Content-Type':'application/json','X-Session-Token':authToken},
      body: JSON.stringify({decision})
    });
    const d = await resp.json();
    if (d.ok) {
      const actions = cardEl.querySelector('.devai-actions');
      if (actions) actions.innerHTML = `<span class="devai-decided">${decision.toUpperCase()}D</span>`;
      setTimeout(() => cardEl.classList.add('devai-faded'), 1500);
    } else {
      showToast('DevAI error: ' + (d.error || 'unknown'));
    }
  } catch(e) { showToast('DevAI respond error: ' + e.message); }
}

function connect(){
  const proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen=()=>{
    ws.send(JSON.stringify({type:'init',session_id:SESSION_ID,token:authToken}));
    connDot.className='dot online'; connLabel.textContent='Online';
    input.disabled=false; sendBtn.disabled=false;
    addNote('Connected \u2014 Cortana is ready.');
    triggerExpression('smile');
  };
  ws.onclose=()=>{
    connDot.className='dot offline'; connLabel.textContent='Offline';
    input.disabled=true; sendBtn.disabled=true;
    triggerExpression('sad');
    addNote('Connection lost \u2014 reconnecting\u2026');
    setTimeout(connect,3000);
  };
  ws.onerror=()=>ws.close();
  ws.onmessage=(e)=>{
    const msg=JSON.parse(e.data);
    switch(msg.type){
      case 'thinking':
        removeThinking(); addThinking();
        setExpression('think'); sendBtn.disabled=true;
        break;
      case 'response':
        removeThinking();
        triggerExpression(msg.emotion||'smile');
        if(window.detectAndPlay) window.detectAndPlay(msg.text, msg.emotion||'idle');
        addMessage('cortana',msg.text);
        _memSaveTurn('assistant', msg.text);  // persist locally
        turnCount.textContent=msg.turn;
        if(msg.providers)updateProviders(msg.providers);
        sendBtn.disabled=false; input.disabled=false; input.focus();
        break;
      case 'history':
        if(msg.turns&&msg.turns.length){
          msg.turns.forEach(t=>addMessage(t.role==='user'?'user':'cortana',t.content));
          // Also sync server history into local storage if local is empty
          if(_memLoad().length===0){
            msg.turns.forEach(t=>_memSaveTurn(t.role==='user'?'user':'assistant',t.content));
          } else {
            _memTouch(); // just refresh inactivity clock
          }
          addNote(`\u21BA Restored ${msg.turns.length} turns from previous session`);
          turnCount.textContent=msg.turn||0;
        }
        break;
      case 'learning':
        // Self-improvement is a silent background process — no UI notification
        break;
      case 'background_started':
        bgBadge.classList.add('visible'); addNote('\u27F3 Background task: '+msg.name); break;
      case 'background_progress':
        bgBadge.classList.add('visible'); break;
      case 'background_done':
        bgBadge.classList.remove('visible');
        showToast('\u2713 Done: '+msg.name);
        addNote('\u2713 Background task complete \u2014 ask me about it.');
        triggerExpression('smile'); break;
      case 'search_start':{
        const sp=document.getElementById('searchPanel');
        document.getElementById('spLabel').textContent='SEARCHING';
        document.getElementById('spQuery').textContent=msg.query||'';
        document.getElementById('spResults').textContent='';
        document.getElementById('spResults').classList.remove('visible');
        document.getElementById('spPulse').className='sp-pulse';
        sp.classList.add('visible');
        break;}
      case 'search_done':{
        document.getElementById('spLabel').textContent='FOUND';
        document.getElementById('spPulse').className='sp-pulse done';
        const sr=document.getElementById('spResults');
        sr.textContent=msg.snippet||'';
        sr.classList.add('visible');
        clearTimeout(window._spTimer);
        window._spTimer=setTimeout(()=>document.getElementById('searchPanel').classList.remove('visible'),6000);
        break;}
      case 'cortana_reload':
        // Server-initiated soft page reload (e.g. after admin restart)
        addNote('\u21BB Cortana is restarting — reconnecting in 5s...');
        setTimeout(()=>window.location.reload(), 5000);
        break;
      case 'security_alert':
        addNote('\u26A0 Security: '+msg.detail);
        triggerExpression('surprised'); break;
      case 'security_scan':
        // Periodic auto-scan result — show as a note, not a chat message
        (function(){
          const icon = msg.score >= 80 ? '\u2705' : msg.score >= 60 ? '\u26A0' : '\u274C';
          addNote(icon + ' Security scan #' + msg.scan_number
            + ' \u2014 Score: ' + msg.score + '/100'
            + (msg.critical > 0 ? ' | ' + msg.critical + ' critical' : '')
            + (msg.high > 0 ? ' | ' + msg.high + ' high' : ''));
          if(msg.score < 60) triggerExpression('surprised');
        })();
        break;
      case 'autonomous_browse':
        // Cortana browsed the web out of curiosity — show in consciousness panel
        (function(){
          const list = document.getElementById('csThoughtsList');
          if(!list) return;
          const item = document.createElement('div');
          item.className = 'cs-thought-item';
          item.textContent = '[web] Searched: ' + (msg.topic||'') + ' \u2014 ' + (msg.snippet||'');
          list.prepend(item);
          while(list.children.length > 8) list.removeChild(list.lastChild);
        })();
        break;
      case 'cortana_thought':
        removeThinking();
        triggerExpression('think');
        // Display as a spontaneous Cortana message with a distinct marker
        addMessage('cortana', '\u2235 ' + msg.text);
        break;
      case 'inner_thought':
        // Background consciousness stream — update panel only, not chat
        (function(){
          const list = document.getElementById('csThoughtsList');
          if (!list) return;
          // Clear placeholder if present
          const ph = list.querySelector('span');
          if (ph) ph.remove();
          const item = document.createElement('div');
          item.className = 'cs-thought-item';
          item.textContent = msg.thought || '';
          list.prepend(item);
          while (list.children.length > 8) list.removeChild(list.lastChild);
          // mood is expressed through Cortana's response tone, not a bar
        })();
        break;
      case 'vision_response':
        if(window._visionHandler) window._visionHandler(msg);
        break;
      case 'model_update':
        loadGLB(msg.glb_path||'/static/cortana.glb');
        triggerExpression('smile');
        addNote('\u2728 Cortana updated her 3D appearance.');
        if(msg.message) showToast(msg.message);
        break;
      case 'devai_proposal':
        // Only admin sessions receive this; render as a special card
        (function(){
          const box = document.createElement('div');
          box.className = 'devai-card';
          box.dataset.id = msg.id;
          // Convert markdown-ish content to simple HTML
          const md = (msg.message||'').replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
                                       .replace(/`([^`]+)`/g,'<code>$1</code>')
                                       .replace(/```[\s\S]*?```/g, s => {
                                         const inner = s.replace(/^```[^\\n]*\\n?/,'').replace(/\\n?```$/,'');
                                         return '<pre class="devai-diff">'+inner.replace(/</g,'&lt;')+'</pre>';
                                       })
                                       .replace(/\\n/g,'<br>');
          box.innerHTML = `<div class="devai-header">\u26A1 DevAI Suggestion #${msg.id}</div>`
                        + `<div class="devai-body">${md}</div>`
                        + `<div class="devai-actions">`
                        + `<button class="devai-btn approve" onclick="devaiRespond(${msg.id},'approve',this.parentElement.parentElement)">Approve</button>`
                        + `<button class="devai-btn reject"  onclick="devaiRespond(${msg.id},'reject', this.parentElement.parentElement)">Reject</button>`
                        + `</div>`;
          document.getElementById('messages').appendChild(box);
          box.scrollIntoView({behavior:'smooth'});
          triggerExpression('think');
          showToast('\u26A1 DevAI found a code improvement. See chat.');
        })();
        break;
      case 'devai_decision':
        // Remove the card and show result
        (function(){
          const card = document.querySelector(`.devai-card[data-id="${msg.id}"]`);
          if(card){
            card.querySelector('.devai-actions').innerHTML =
              `<span class="devai-decided">${msg.action.toUpperCase()}</span>`;
            setTimeout(()=>card.classList.add('devai-faded'), 2000);
          }
          addNote(`DevAI #${msg.id} ${msg.action}.`);
        })();
        break;
      case 'error':
        removeThinking(); triggerExpression('frown');
        addNote('\u26A0 '+msg.message);
        sendBtn.disabled=false; input.disabled=false; break;
      case 'status':
        if(msg.turn!==undefined)turnCount.textContent=msg.turn;
        if(msg.providers)updateProviders(msg.providers);
        if(msg.user){currentUser={...currentUser,...msg.user};updateUserBar();}
        break;
    }
  };
}

function updateProviders(providers){
  const ready=providers.filter(p=>p.status==='ready').map(p=>p.provider);
  if(ready.length){providerLabel.textContent=ready[0].toUpperCase();providerBadge.style.display='flex';}
}

function addMessage(role,text){
  const w=document.createElement('div'); w.className='msg '+role;
  const lb=document.createElement('div'); lb.className='msg-label';
  lb.textContent=role==='cortana'?'Cortana':'You';
  const b=document.createElement('div'); b.className='bubble';
  w.appendChild(lb); w.appendChild(b); messages.appendChild(w); scrollDown();
  if(role==='cortana') typewriter(b,text); else b.textContent=text;
}

function addNote(text){
  const w=document.createElement('div'); w.className='msg note';
  const b=document.createElement('div'); b.className='bubble';
  b.textContent=text; w.appendChild(b); messages.appendChild(w); scrollDown();
}

function addThinking(){
  const w=document.createElement('div'); w.className='msg cortana'; w.id='thinking-wrap';
  const lb=document.createElement('div'); lb.className='msg-label'; lb.textContent='Cortana';
  const b=document.createElement('div'); b.className='bubble';
  b.innerHTML='<div class="dots"><span></span><span></span><span></span></div>';
  w.appendChild(lb); w.appendChild(b); messages.appendChild(w); scrollDown();
}

function removeThinking(){const el=document.getElementById('thinking-wrap');if(el)el.remove();}

function typewriter(el,text,speed=6){
  if(window._startTalking) window._startTalking();
  let i=0;
  function step(){
    el.textContent+=text.slice(i,i+4);i+=4;scrollDown();
    if(i<text.length) setTimeout(step,speed);
    else if(window._stopTalking) window._stopTalking();
  }
  step();
}

function scrollDown(){messages.scrollTop=messages.scrollHeight;}

function showToast(t){
  toast.textContent=t; toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>toast.classList.remove('show'),5000);
}

function sendMessage(){
  const text=input.value.trim();
  if(!text||!ws||ws.readyState!==WebSocket.OPEN) return;
  addMessage('user',text);
  _memSaveTurn('user', text);  // persist locally
  ws.send(JSON.stringify({type:'message',message:text}));
  input.value=''; input.style.height='auto';
  input.disabled=true; sendBtn.disabled=true;
}

sendBtn.onclick=sendMessage;
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}});

// ── Vision (Webcam + Screen Share) ─────────────────────────────────────────
(function(){
  // ── Shared state ──
  let _visionCooldownUntil = 0;  // epoch ms — mirrors server-side cooldown

  function _isOnCooldown() { return Date.now() < _visionCooldownUntil; }
  function _setCooldown(ms) { _visionCooldownUntil = Date.now() + ms; }

  function _captureAndSend(videoEl, statusEl, source, question) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (_isOnCooldown()) {
      statusEl.textContent = 'Cooling down\u2026';
      return;
    }
    const w = videoEl.videoWidth || 320, h = videoEl.videoHeight || 240;
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(videoEl, 0, 0, w, h);
    const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    ws.send(JSON.stringify({type:'vision', source, image:b64, question: question||''}));
    statusEl.textContent = 'Analyzing\u2026';
  }

  // ── Webcam ──
  (function(){
    const panel     = document.getElementById('camPanel');
    const video     = document.getElementById('camVideo');
    const snapBtn   = document.getElementById('camSnapBtn');
    const autoBtn   = document.getElementById('camAutoBtn');
    const status    = document.getElementById('camStatus');
    const toggleBtn = document.getElementById('camToggleBtn');
    let stream = null, autoTimer = null;

    toggleBtn.onclick = async () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null; video.srcObject = null;
        panel.classList.remove('visible');
        toggleBtn.classList.remove('active');
        if (autoTimer) { clearInterval(autoTimer); autoTimer = null; autoBtn.classList.remove('active'); }
        status.textContent = 'Camera off';
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
        video.srcObject = stream;
        panel.classList.add('visible');
        toggleBtn.classList.add('active');
        status.textContent = 'Camera ready';
      } catch(e) { status.textContent = 'Camera denied: ' + e.message; }
    };

    snapBtn.onclick = () => _captureAndSend(video, status, 'webcam', input.value.trim() || 'What do you see?');

    autoBtn.onclick = () => {
      if (autoTimer) {
        clearInterval(autoTimer); autoTimer = null;
        autoBtn.classList.remove('active');
        status.textContent = 'Auto off';
      } else {
        autoBtn.classList.add('active');
        status.textContent = 'Auto on \u2014 every 10s';
        autoTimer = setInterval(() => {
          if (!_isOnCooldown()) _captureAndSend(video, status, 'webcam', 'Briefly describe what you observe.');
        }, 10000);
      }
    };
  })();

  // ── Screen Share ──
  (function(){
    const panel     = document.getElementById('screenPanel');
    const video     = document.getElementById('screenVideo');
    const snapBtn   = document.getElementById('screenSnapBtn');
    const autoBtn   = document.getElementById('screenAutoBtn');
    const status    = document.getElementById('screenStatus');
    const toggleBtn = document.getElementById('screenToggleBtn');
    let stream = null, autoTimer = null;

    toggleBtn.onclick = async () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null; video.srcObject = null;
        panel.classList.remove('visible');
        toggleBtn.classList.remove('active');
        if (autoTimer) { clearInterval(autoTimer); autoTimer = null; autoBtn.classList.remove('active'); }
        status.textContent = 'Screen share stopped';
        return;
      }
      try {
        stream = await navigator.mediaDevices.getDisplayMedia({
          video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:5}},
          audio:false
        });
        video.srcObject = stream;
        // Auto-stop if the user closes the browser's screen-share picker
        stream.getVideoTracks()[0].onended = () => {
          stream = null; video.srcObject = null;
          panel.classList.remove('visible');
          toggleBtn.classList.remove('active');
          if (autoTimer) { clearInterval(autoTimer); autoTimer = null; autoBtn.classList.remove('active'); }
          status.textContent = 'Screen share stopped';
        };
        panel.classList.add('visible');
        toggleBtn.classList.add('active');
        status.textContent = 'Screen captured \u2014 ready';
      } catch(e) {
        status.textContent = e.name === 'NotAllowedError' ? 'Permission denied' : 'Error: ' + e.message;
      }
    };

    snapBtn.onclick = () => _captureAndSend(video, status, 'screen',
      input.value.trim() || 'What do you see on my screen?');

    autoBtn.onclick = () => {
      if (autoTimer) {
        clearInterval(autoTimer); autoTimer = null;
        autoBtn.classList.remove('active');
        status.textContent = 'Auto off';
      } else {
        autoBtn.classList.add('active');
        status.textContent = 'Auto on \u2014 every 15s';
        autoTimer = setInterval(() => {
          if (!_isOnCooldown()) _captureAndSend(video, status, 'screen',
            'What is currently visible on the screen? Describe briefly.');
        }, 15000);
      }
    };
  })();

  // ── Shared vision response handler ──
  window._visionHandler = (msg) => {
    if (msg.type !== 'vision_response') return;
    const source = msg.source || 'webcam';
    const statusEl = document.getElementById(source === 'screen' ? 'screenStatus' : 'camStatus');
    if (statusEl) statusEl.textContent = 'Done';

    // If rate-limited, set client-side cooldown too so auto-mode stops
    if (msg.text && msg.text.includes('rate-limit')) {
      _setCooldown(62000);
      if (statusEl) statusEl.textContent = 'Cooling down (60s)\u2026';
    }

    const label = source === 'screen' ? '[Screen]' : '[Webcam]';
    addMessage('assistant', label + ' ' + msg.text);
    if (msg.emotion) setExpression(msg.emotion);
  };
})();
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,110)+'px';});

// ── Consciousness panel ──
(function(){
  const panel  = document.getElementById('consciousnessPanel');
  const toggle = document.getElementById('csPanelToggle');
  let visible  = true;

  function updatePanel(data){
    const uptimeEl  = document.getElementById('csUptime');
    const moodLbl   = document.getElementById('csMoodLabel');
    const interEl   = document.getElementById('csInteractions');
    const thoughtEl = document.getElementById('csThoughts');
    const list      = document.getElementById('csThoughtsList');

    if(uptimeEl)  uptimeEl.textContent  = data.uptime_hours < 1
                    ? Math.round(data.uptime_hours * 60) + 'm'
                    : data.uptime_hours.toFixed(1) + 'h';
    if(moodLbl)   moodLbl.textContent   = data.emotional_state || '—';
    if(interEl)   interEl.textContent   = data.total_interactions ?? '—';
    if(thoughtEl) thoughtEl.textContent = data.total_thoughts ?? '—';

    if(list && data.recent_thoughts && data.recent_thoughts.length){
      list.innerHTML = '';
      data.recent_thoughts.slice(0,8).forEach(t=>{
        const d = document.createElement('div');
        d.className = 'cs-thought-item';
        d.textContent = t;
        list.appendChild(d);
      });
    }
  }

  function fetchConsciousness(){
    fetch('/api/consciousness')
      .then(r=>r.ok?r.json():null)
      .then(d=>{ if(d) updatePanel(d); })
      .catch(()=>{});
    fetch('/api/agi/status')
      .then(r=>r.ok?r.json():null)
      .then(d=>{
        if(!d) return;
        const goalsEl = document.getElementById('csGoals');
        if(goalsEl && d.active_goals && d.active_goals.length){
          goalsEl.innerHTML = d.active_goals.slice(0,3).map(g=>{
            const pct = Math.round((g.progress||0)*100);
            const bar = '\u2588'.repeat(Math.round(pct/12.5)) + '\u2591'.repeat(8-Math.round(pct/12.5));
            return `<div class="cs-thought-item" title="${g.domain}">${bar} ${g.description.slice(0,55)}</div>`;
          }).join('');
        }
        const wmEl = document.getElementById('csWMStats');
        if(wmEl && d.world_model){
          wmEl.textContent = `entities:${d.world_model.entities} beliefs:${d.world_model.beliefs} causal:${d.world_model.causal}`;
        }
      })
      .catch(()=>{});
  }

  // Restart button (shown only to admins)
  const csRestartBtn = document.getElementById('csRestartBtn');
  const csAdminBar   = document.getElementById('csAdminBar');
  if(csRestartBtn){
    csRestartBtn.addEventListener('click', ()=>{
      const tok = localStorage.getItem('cx_session_token') || '';
      if(!tok){ alert('Admin login required'); return; }
      if(!confirm('Restart the Cortana service now? All clients will reload in ~5s.')) return;
      fetch('/api/admin/restart', {method:'POST', headers:{'X-Session-Token':tok}})
        .then(r=>r.json()).then(d=>{ if(d.ok) addNote('\u21BB Restart initiated — reloading soon...'); })
        .catch(()=>alert('Restart failed'));
    });
  }

  // Show admin bar if logged in as admin
  function _maybeShowAdminBar(){
    const tok = localStorage.getItem('cx_session_token') || '';
    if(!tok || !csAdminBar) return;
    fetch('/api/auth/me', {headers:{'X-Session-Token':tok}})
      .then(r=>r.ok?r.json():null)
      .then(u=>{ if(u && u.tier==='admin') csAdminBar.style.display='block'; })
      .catch(()=>{});
  }
  _maybeShowAdminBar();

  if(toggle && panel){
    toggle.addEventListener('click',()=>{
      visible = !visible;
      panel.classList.toggle('hidden', !visible);
      toggle.classList.toggle('active', visible);
    });
    toggle.classList.add('active');
  }

  fetchConsciousness();
  setInterval(fetchConsciousness, 30000);
})();

connect();
</script>
</body>
</html>"""



# ---------------------------------------------------------------------------
# Chat Layer
# ---------------------------------------------------------------------------
class ChatLayer:
    """
    Layer 13 — Web chat server.
    Exposes Cortana's pipeline over WebSocket + a clean browser UI.
    Runs a background self-improvement loop.
    """

    def __init__(self, system) -> None:
        self.system = system
        self.manager = ConnectionManager()
        self.app = FastAPI(title="Cortana")
        # Serve static assets (GLB model, etc.)
        _static_dir = pathlib.Path(__file__).parent.parent / "static"
        _static_dir.mkdir(exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
        # Security headers — applied to every response
        self.app.add_middleware(SecurityHeadersMiddleware)
        # CORS — HTTPS only for production domain; HTTP only for localhost dev
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                f"https://{config.WEB_DOMAIN}",
                "http://localhost:8080",
                "http://127.0.0.1:8080",
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "X-Session-Token", "Authorization"],
        )
        self._self_session = Session(id="self")  # dedicated session for self-improvement
        # Mount Layer 15 security review endpoints
        try:
            from cortana.layers.layer15_security_review import get_security_router
            self.app.include_router(get_security_router())
        except Exception as _e:
            logger.warning("Layer 15 security router unavailable: %s", _e)
        self._setup_routes()

    def _setup_routes(self) -> None:
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return _HTML

        @app.get("/health")
        async def health():
            return {"status": "ok", "connections": self.manager.count}

        @app.get("/api/status")
        async def api_status():
            return {
                "providers": self.system.reasoning.router.status(),
                "connections": self.manager.count,
            }

        @app.get("/api/memory")
        async def api_memory(request: Request):
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            return {"episodes": self.system.memory.get_recent_episodes(limit=10)}

        @app.get("/api/graph")
        async def api_graph(request: Request):
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user:
                return JSONResponse(status_code=401, content={"error": "Authentication required"})
            return self.system.memory.get_concept_graph(limit=60)

        @app.get("/api/consciousness")
        async def api_consciousness():
            """Public endpoint — returns Cortana's current conscious state."""
            try:
                m = self.system.self_model.model
                uptime_h = self.system.self_model.get_uptime_seconds() / 3600
                recent   = self.system.self_model.get_recent_thoughts(n=8)
            except Exception:
                return JSONResponse(status_code=503, content={"error": "Consciousness engine unavailable"})
            return {
                "uptime_hours":       round(uptime_h, 3),
                "total_interactions": m.total_interactions,
                "total_thoughts":     m.total_thoughts,
                "mood_score":         round(m.current_mood_score, 3),
                "emotional_state":    m.emotional_state,
                "self_assessment":    m.self_assessment,
                "core_values":        m.core_values,
                "recent_thoughts":    recent,
            }

        @app.get("/api/agi/status")
        async def api_agi_status():
            """AGI framework status — goals, world model stats, mode performance."""
            try:
                return self.system.agi.get_status()
            except Exception as exc:
                return JSONResponse(status_code=503, content={"error": str(exc)})

        @app.post("/api/agi/validate")
        async def api_agi_validate(request: Request):
            """Validate a candidate system prompt against the golden test suite."""
            try:
                body = await request.json()
                prompt = body.get("prompt", "")
                if not prompt:
                    return JSONResponse(status_code=400, content={"error": "prompt required"})
                return self.system.agi.validate_prompt(prompt)
            except Exception as exc:
                return JSONResponse(status_code=503, content={"error": str(exc)})

        @app.get("/api/nexus/status")
        async def api_nexus_status():
            """Neural Nexus neuron stats and synapse weights."""
            try:
                return self.system.nexus.get_status()
            except Exception as exc:
                return JSONResponse(status_code=503, content={"error": str(exc)})

        @app.get("/api/nexus/flush")
        async def api_nexus_flush_logs():
            """Recent Nexus Flush belief-validation logs."""
            try:
                return {"logs": self.system.nexus.recent_flush_logs(limit=10)}
            except Exception as exc:
                return JSONResponse(status_code=503, content={"error": str(exc)})

        @app.get("/api/tasks")
        async def api_tasks(request: Request):
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            return {"tasks": self.system.thinker.get_all_tasks(limit=20)}

        # ── User auth endpoints ──

        from cortana import auth as _auth

        @app.post("/api/auth/register")
        async def auth_register(request: Request):
            body = await request.json()
            result = _auth.register_user(
                body.get("username", ""),
                body.get("password", ""),
                body.get("email", ""),
            )
            if not result["ok"]:
                return JSONResponse(status_code=400, content=result)
            return result

        @app.post("/api/auth/login")
        async def auth_login(request: Request):
            # IP-level rate limiting (blocks credential-stuffing attacks)
            client_ip = request.headers.get("X-Forwarded-For", "") or (
                request.client.host if request.client else "unknown"
            )
            client_ip = client_ip.split(",")[0].strip()
            if _ip_is_rate_limited(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"ok": False, "error": "Too many login attempts. Try again later."},
                )
            body = await request.json()
            username = body.get("username", "")
            password = body.get("password", "")
            result = _auth.login_user(username, password)
            failed = not result.get("ok", False)
            _ip_record_login_attempt(client_ip, failed)
            if failed:
                return JSONResponse(status_code=401, content=result)
            return result

        @app.get("/api/auth/me")
        async def auth_me(request: Request):
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user:
                return JSONResponse(status_code=401, content={"error": "Not authenticated"})
            info = _auth.get_user_info(user["user_id"])
            return info or JSONResponse(status_code=404, content={"error": "User not found"})

        @app.post("/api/auth/logout")
        async def auth_logout(request: Request):
            token = request.headers.get("X-Session-Token", "")
            if token:
                import sqlite3 as _sql
                with _sql.connect(config.SQLITE_PATH) as c:
                    c.execute("DELETE FROM web_sessions WHERE token=?", (token,))
                    c.commit()
            return {"ok": True}

        @app.get("/api/tiers")
        async def tier_info():
            # admin is a private tier — never exposed publicly
            return {k: {**v} for k, v in config.TIERS.items() if k != "admin"}

        @app.post("/api/auth/change-password")
        async def auth_change_password(request: Request):
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user:
                return JSONResponse(status_code=401, content={"ok": False, "error": "Not authenticated"})
            body = await request.json()
            result = _auth.change_password(user["user_id"], body.get("new_password", ""))
            if not result["ok"]:
                return JSONResponse(status_code=400, content=result)
            return result

        @app.post("/api/auth/forgot-password")
        async def auth_forgot_password(request: Request):
            body = await request.json()
            result = _auth.request_password_reset(body.get("username_or_email", ""))
            # SECURITY: never expose the raw reset token in the HTTP response.
            # Tokens must be delivered out-of-band (email). Log for admin reference only.
            if result.get("reset_token"):
                logger.info(
                    "[Auth] Password reset token generated for account lookup '%s'. "
                    "Configure EMAIL_ADDRESS + EMAIL_APP_PASSWORD in .env to send it automatically.",
                    body.get("username_or_email", "")[:64],
                )
                result = {k: v for k, v in result.items() if k != "reset_token"}
            return result

        @app.post("/api/auth/reset-password")
        async def auth_reset_password(request: Request):
            body = await request.json()
            result = _auth.confirm_password_reset(
                body.get("token", ""), body.get("new_password", "")
            )
            if not result["ok"]:
                return JSONResponse(status_code=400, content=result)
            return result

        # ── Subscription endpoint ──

        @app.post("/api/auth/subscribe")
        async def auth_subscribe(request: Request):
            """Activate a monthly subscription after ETH payment."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user:
                return JSONResponse(status_code=401, content={"ok": False, "error": "Not authenticated"})
            body   = await request.json()
            tier   = body.get("tier", "")
            tx_hash = body.get("tx_hash", "")
            if not tier or not tx_hash:
                return JSONResponse(status_code=400, content={"ok": False, "error": "tier and tx_hash required"})
            result = _auth.subscribe_user(user["user_id"], tier, tx_hash)
            if not result["ok"]:
                return JSONResponse(status_code=400, content=result)
            return result

        @app.get("/api/auth/subscription")
        async def auth_subscription_status(request: Request):
            """Return current subscription status for the authenticated user."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user:
                return JSONResponse(status_code=401, content={"error": "Not authenticated"})
            return _auth.get_subscription_status(user["user_id"])

        # ── Wallet endpoints (admin) ──

        @app.get("/api/wallet/info")
        async def wallet_info(request: Request):
            """Return wallet address, balance, and earnings summary. Admin only."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            try:
                from cortana.wallet import get_wallet_info
                return get_wallet_info(include_balance=True)
            except Exception as exc:
                return JSONResponse(status_code=500, content={"error": str(exc)})

        @app.post("/api/wallet/sweep")
        async def wallet_sweep(request: Request):
            """Manually trigger ETH sweep to owner wallet. Admin only."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            try:
                from cortana.wallet import sweep_to_owner
                result = await asyncio.to_thread(sweep_to_owner)
                return result
            except Exception as exc:
                return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

        # ── Safety kill switch (admin only) ──

        @app.post("/api/admin/lockdown")
        async def toggle_lockdown(request: Request):
            """Enable or disable the tool kill switch. Admin only."""
            from cortana import auth as _auth
            from cortana.layers.layer8_tools import set_lockdown
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            body = await request.json()
            active = bool(body.get("active", True))
            msg = set_lockdown(active)
            return {"ok": True, "lockdown": active, "message": msg}

        @app.get("/api/admin/lockdown")
        async def lockdown_status(request: Request):
            """Check current lockdown state. Admin only."""
            from cortana import auth as _auth
            from pathlib import Path
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            lockdown_file = Path(config.AGENT_WORKSPACE) / ".lockdown"
            return {"lockdown": lockdown_file.exists()}

        # ── Restart / reload ──

        @app.post("/api/admin/restart")
        async def admin_restart(request: Request):
            """
            Soft-restart: broadcast reload event to all clients, then SIGHUP
            the process (graceful reload on Render/gunicorn).
            Falls back to SIGTERM. Admin only.
            """
            from cortana import auth as _auth
            import os, signal as _signal, threading as _threading
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})

            # Notify all clients to reload their page in 5s
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self.manager.broadcast({"type": "cortana_reload"}), loop
                )
            except Exception:
                pass

            def _do_restart():
                import time as _time
                _time.sleep(3.0)
                try:
                    os.kill(os.getpid(), _signal.SIGHUP)
                except (AttributeError, OSError):
                    os.kill(os.getpid(), _signal.SIGTERM)

            _threading.Thread(target=_do_restart, daemon=True).start()
            return {"ok": True, "message": "Restart initiated — clients notified"}

        @app.get("/api/version")
        async def get_version():
            """Return current build version (public)."""
            return {
                "version":     "2.1.0",
                "build":       "Neural Nexus AGI",
                "nexus":       True,
                "nexus_flush": True,
            }

        # ── Knowledge bin ──

        @app.post("/api/knowledge")
        async def add_knowledge(request: Request):
            """Add a factual item to the knowledge bin for Cortana to absorb."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user:
                return JSONResponse(status_code=401, content={"error": "Not authenticated"})
            body = await request.json()
            content = (body.get("content") or "").strip()
            if not content:
                return JSONResponse(status_code=400, content={"error": "content required"})
            item_id = self.system.memory.add_knowledge(content, source=user["username"])
            return {"ok": True, "id": item_id}

        @app.get("/api/knowledge")
        async def list_knowledge(request: Request):
            """List unabsorbed knowledge items. Admin only."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            items = self.system.memory.get_unabsorbed_knowledge(limit=50)
            return {"items": items}

        @app.get("/api/model/design")
        async def get_model_params():
            """Return current 3D model design parameters."""
            try:
                from cortana.tools.model_designer import load_params
                return {"ok": True, "params": load_params()}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        @app.post("/api/model/design")
        async def trigger_model_design(request: Request):
            """
            Trigger an autonomous 3D redesign. Admin only.
            Body: {"description": "medium skin, long hair, slim"}
            """
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            body = await request.json()
            description = body.get("description", "")
            try:
                from cortana.tools.model_designer import design_self as _ds
                result = await _ds(description=description)
                return {"ok": True, "result": result}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        # ── DevAI integration ──────────────────────────────────────────────

        # Pending proposals received from DevAI daemon: {id: proposal_dict}
        self._devai_proposals: dict = {}

        _DEVAI_TOKEN = os.getenv("DEVAI_INTERNAL_TOKEN", "devai-local-bridge")
        _DEVAI_RESPONSE_PIPE = "/tmp/devai-response.pipe"
        _DEVAI_DB = str(Path.home() / ".devai" / "devai.db")

        @app.post("/api/devai/proposal")
        async def receive_devai_proposal(request: Request):
            """
            Called by the DevAI daemon when it finds a code-improvement proposal.
            Broadcasts a chat-style alert to all admin WebSocket connections.
            Admin-only: requires the internal DevAI bridge token.
            """
            body = await request.json()
            if body.get("token") != _DEVAI_TOKEN:
                return JSONResponse(status_code=403, content={"error": "Forbidden"})

            pid   = body.get("id")
            fpath = body.get("file_path", "unknown")
            ptype = body.get("type", "improvement")
            sev   = body.get("severity", "low")
            summ  = body.get("summary", "")
            detail = body.get("detail", "")
            orig  = body.get("original", "")
            impr  = body.get("improved", "")

            # Store for later retrieval
            self._devai_proposals[pid] = body

            # Build a readable message for the admin chat
            short = fpath.replace(str(Path.home()), "~")
            msg = (
                f"**[DevAI #{pid}]** `{ptype.upper()}` · severity: `{sev}`\n"
                f"**File:** `{short}`\n"
                f"**Summary:** {summ}\n"
                f"{detail}\n\n"
                f"```\n# BEFORE\n{orig[:400]}\n\n# AFTER\n{impr[:400]}\n```\n\n"
                f"Reply **`approve #{pid}`** or **`reject #{pid}`** to decide."
            )

            await self.manager.broadcast_admin({
                "type": "devai_proposal",
                "id": pid,
                "message": msg,
            })
            logger.info("DevAI proposal #%s broadcast to admin sessions.", pid)
            return {"ok": True}

        @app.get("/api/devai/proposals")
        async def list_devai_proposals(request: Request):
            """List pending proposals from DevAI's SQLite DB. Admin only."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            try:
                import sqlite3
                db = _DEVAI_DB
                if not Path(db).exists():
                    return {"proposals": []}
                conn = sqlite3.connect(db)
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, file_path, type, severity, summary, status, created_at "
                    "FROM proposals ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
                conn.close()
                return {"proposals": [dict(r) for r in rows]}
            except Exception as exc:
                return JSONResponse(status_code=500, content={"error": str(exc)})

        @app.post("/api/devai/respond/{proposal_id}")
        async def respond_devai_proposal(proposal_id: int, request: Request):
            """
            Admin approves or rejects a DevAI proposal.
            Body: {"decision": "approve" | "reject"}
            Writes the decision to DevAI's response pipe and updates the DB.
            """
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})

            body = await request.json()
            decision = body.get("decision", "reject").lower()
            approve = decision in ("approve", "y", "yes")

            # Write to DevAI's response pipe (DevAI daemon is waiting)
            try:
                import stat as _stat
                p = Path(_DEVAI_RESPONSE_PIPE)
                if p.exists() and _stat.S_ISFIFO(p.stat().st_mode):
                    fd = os.open(_DEVAI_RESPONSE_PIPE, os.O_WRONLY | os.O_NONBLOCK)
                    os.write(fd, b"y\n" if approve else b"n\n")
                    os.close(fd)
            except OSError:
                pass  # pipe may not have a reader if DevAI is between cycles

            # Also update DB directly so status is visible immediately
            try:
                import sqlite3, time as _time
                if Path(_DEVAI_DB).exists():
                    conn = sqlite3.connect(_DEVAI_DB)
                    status = "approved" if approve else "rejected"
                    conn.execute(
                        "UPDATE proposals SET status=?, decided_at=? WHERE id=? AND status='pending'",
                        (status, int(_time.time()), proposal_id),
                    )
                    conn.commit()
                    conn.close()
            except Exception:
                pass

            self._devai_proposals.pop(proposal_id, None)
            action = "approved" if approve else "rejected"
            await self.manager.broadcast_admin({
                "type": "devai_decision",
                "id": proposal_id,
                "action": action,
                "message": f"[DevAI #{proposal_id}] {action.upper()} by {user['username']}.",
            })
            return {"ok": True, "action": action}

        # ── Layer 16 — Knowledge Distiller / Training Corpus ──────────────

        @app.get("/api/admin/corpus")
        async def corpus_stats(request: Request):
            """Return training corpus statistics. Admin only."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            stats = self.system.distiller.get_stats()
            return stats

        @app.post("/api/admin/distill")
        async def trigger_distillation(request: Request):
            """Manually trigger a distillation batch. Admin only."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            body = await request.json()
            batch_size = min(int(body.get("batch_size", 15)), 100)
            result = await asyncio.to_thread(
                self.system.distiller.distill_batch, batch_size
            )
            return {
                "ok": True,
                "processed": result.total_processed,
                "passed":    result.passed,
                "flagged":   result.flagged,
                "skipped":   result.skipped,
                "exported":  result.exported_path,
                "errors":    result.errors,
            }

        @app.post("/api/admin/corpus/export")
        async def export_corpus(request: Request):
            """Re-export the full training corpus to JSONL. Admin only."""
            token = request.headers.get("X-Session-Token", "")
            user = _auth.validate_token(token)
            if not user or user.get("tier") != "admin":
                return JSONResponse(status_code=403, content={"error": "Admin only"})
            body = await request.json()
            min_ethics  = float(body.get("min_ethics",  0.70))
            min_quality = float(body.get("min_quality", 0.60))
            path = await asyncio.to_thread(
                self.system.distiller.export_jsonl,
                None, min_ethics, min_quality,
            )
            stats = self.system.distiller.get_stats()
            return {"ok": True, "path": path, "corpus": stats}

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            from cortana import auth as _auth
            await self.manager.connect(websocket)
            session = Session()
            self.manager.register_session(websocket, session)
            user_info = None  # populated if authenticated
            try:
                # First message must be init with session_id (+ optional token)
                init_data = await websocket.receive_json()
                if init_data.get("type") == "init":
                    sid = init_data.get("session_id", "")
                    token = init_data.get("token", "")

                    # Validate auth token if provided
                    if token:
                        user_info = _auth.validate_token(token)
                        # Mark admin sockets so DevAI proposals can be targeted
                        if user_info and user_info.get("tier") == "admin":
                            self.manager.mark_admin(websocket)

                    if sid:
                        session.session_id = sid
                        # Restore conversation from SQLite
                        prior = self.system.memory.load_conversation(sid, limit=40)
                        if prior:
                            session.conversation = prior
                            session.state = CortanaState(interaction_count=len(prior) // 2)
                            await websocket.send_json({
                                "type": "history",
                                "turns": [{"role": t.role, "content": t.content} for t in prior],
                                "turn": session.state.interaction_count,
                            })

                # Send current status + user tier info
                status_msg = {
                    "type": "status",
                    "turn": session.state.interaction_count,
                    "providers": self.system.reasoning.router.status(),
                }
                if user_info:
                    status_msg["user"] = {
                        "username": user_info["username"],
                        "tier": user_info["tier"],
                        "daily_limit": user_info["daily_limit"],
                        "usage_today": user_info["usage_today"],
                    }
                await websocket.send_json(status_msg)

                while True:
                    raw_msg = await websocket.receive_text()
                    if len(raw_msg.encode("utf-8")) > _MAX_WS_MSG_BYTES:
                        await websocket.send_json({"type": "error", "message": "Message too large."})
                        continue
                    try:
                        data = _json.loads(raw_msg)
                    except Exception:
                        await websocket.send_json({"type": "error", "message": "Invalid JSON."})
                        continue
                    await self._handle_message(websocket, session, data, user_info)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.exception("WebSocket error")
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    pass
            finally:
                self.manager.disconnect(websocket)

    async def _handle_message(
        self, websocket: WebSocket, session: Session, data: dict,
        user_info: Optional[dict] = None,
    ) -> None:
        from cortana import auth as _auth
        # Support both old {message:...} and new {type:"message",message:...}
        msg_type = data.get("type", "message")
        if msg_type == "init":
            return  # Already handled in ws_endpoint
        if msg_type == "vision":
            await self._handle_vision(websocket, session, data)
            return
        raw = data.get("message", "").strip()
        if not raw:
            return
        # Hard cap on individual message length to prevent prompt injection via giant inputs
        if len(raw) > 8192:
            await websocket.send_json({"type": "error", "message": "Message too long (max 8192 characters)."})
            return

        # ── Tier enforcement ──
        if user_info:
            limit_check = _auth.check_and_increment_usage(user_info["user_id"])
            if not limit_check["ok"]:
                await websocket.send_json({"type": "error", "message": limit_check["error"]})
                return
        # Unauthenticated guests get a hard cap of 5 messages per session
        elif session.state.interaction_count >= 5:
            await websocket.send_json({
                "type": "error",
                "message": "Guest limit reached (5 messages). Create a free account for 20/day.",
            })
            return

        await websocket.send_json({"type": "thinking"})

        try:
            # Build a search event callback that sends WS messages from the pipeline thread
            main_loop = asyncio.get_running_loop()

            def _search_cb(event_type: str, data: dict) -> None:
                payload = {"type": event_type, **data}
                asyncio.run_coroutine_threadsafe(websocket.send_json(payload), main_loop)

            # Run the full pipeline in a thread — providers are blocking
            final, new_state, new_conv, emotion = await asyncio.to_thread(
                _run_pipeline_sync, self.system, raw, session.state, session.conversation,
                _search_cb,
            )
            session.state = new_state
            session.conversation = new_conv

            # Persist this turn pair to SQLite for cross-session memory
            sid = session.session_id
            if sid:
                self.system.memory.save_turn(sid, "user", raw)
                self.system.memory.save_turn(sid, "assistant", final)

            await websocket.send_json({
                "type": "response",
                "text": final,
                "emotion": emotion,
                "turn": session.state.interaction_count,
                "providers": self.system.reasoning.router.status(),
            })

            # ── Periodic security scan every 5 questions ──
            session.question_count += 1
            if session.question_count % 5 == 0:
                scan_number = session.question_count // 5
                loop = asyncio.get_running_loop()
                async def _run_security_scan(snum=scan_number):
                    try:
                        from cortana.layers.layer15_security_review import SecurityScanner
                        report = await asyncio.to_thread(SecurityScanner().scan)
                        critical = sum(1 for f in report.findings if f.severity == "critical")
                        high     = sum(1 for f in report.findings if f.severity == "high")
                        await websocket.send_json({
                            "type":        "security_scan",
                            "scan_number": snum,
                            "score":       report.score,
                            "total":       len(report.findings),
                            "critical":    critical,
                            "high":        high,
                            "summary":     report.summary,
                        })
                        if critical > 0 or high > 2:
                            await websocket.send_json({
                                "type":   "security_alert",
                                "detail": f"Scan {snum}: {critical} critical, {high} high-severity issues found.",
                            })
                    except Exception as _se:
                        logger.debug("Security auto-scan error: %s", _se)
                asyncio.create_task(_run_security_scan())

        except Exception as e:
            logger.exception("Pipeline error")
            await websocket.send_json({"type": "error", "message": str(e)})

    # ------------------------------------------------------------------
    # Webcam / screen-share vision handler
    # ------------------------------------------------------------------
    async def _handle_vision(
        self, websocket: WebSocket, session: Session, data: dict
    ) -> None:
        """
        Process a webcam frame or screen-share capture through the provider router.
        No dedicated API key required — uses the same rotation as the rest of Cortana.
        """
        import time as _time

        image_b64 = data.get("image", "")
        question  = data.get("question", "What do you see?") or "What do you see?"
        source    = data.get("source", "webcam")  # "webcam" | "screen"

        if not image_b64:
            await websocket.send_json({"type": "vision_response", "text": "No image received.", "emotion": "idle"})
            return

        # Per-session cooldown: prevent spamming on rate-limit errors
        now = _time.time()
        cooldown_key = "_vision_cooldown_until"
        if getattr(session, cooldown_key, 0) > now:
            remaining = int(getattr(session, cooldown_key) - now)
            await websocket.send_json({
                "type": "vision_response",
                "text": f"Vision cooling down — ready in {remaining}s.",
                "emotion": "idle",
            })
            return

        # System prompt differs by source
        if source == "screen":
            system_ctx = (
                "You are Cortana, an AI assistant with screen-reading capability. "
                "The user has shared their screen with you so you can understand their context. "
                "Analyse what you see on the screen — applications, content, errors, UI elements. "
                "Be direct and specific. If you see code, read it accurately. "
                "If you see an error, diagnose it. Respond in Cortana's voice: concise and analytical. "
                "Keep it under 4 sentences unless the user asks for more detail."
            )
        else:
            system_ctx = (
                "You are Cortana, an AI assistant with webcam vision. "
                "You can see the user through their webcam. "
                "Respond in Cortana's voice: direct, analytical, occasionally dry wit. "
                "Keep it under 3 sentences unless asked for detail."
            )

        try:
            response = await asyncio.to_thread(
                self.system.reasoning.router.think_vision,
                image_b64,
                question,
                system_ctx,
                512,
            )
            lower = response.lower()
            emotion = (
                "surprised" if any(w in lower for w in ("unexpected", "unusual", "strange", "interesting")) else
                "think"     if any(w in lower for w in ("analyzing", "processing", "error", "problem", "issue")) else
                "smile"     if any(w in lower for w in ("great", "looks good", "impressive", "nice")) else
                "idle"
            )
            await websocket.send_json({
                "type": "vision_response",
                "text": response,
                "emotion": emotion,
                "source": source,
            })
        except Exception as e:
            err_str = str(e)
            logger.warning(f"Vision error ({source}): {err_str[:120]}")
            # If rate-limited, set a 60s cooldown to stop the retry storm
            if "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
                setattr(session, cooldown_key, now + 60)
                msg = "Vision providers are rate-limited. Auto-paused for 60s."
            else:
                msg = f"Vision unavailable: {err_str[:100]}"
            await websocket.send_json({
                "type": "vision_response",
                "text": msg,
                "emotion": "frown",
                "source": source,
            })

    # ------------------------------------------------------------------
    # Self-improvement background task
    # ------------------------------------------------------------------
    async def _self_improve_loop(self) -> None:
        """Periodically run self-generated prompts to grow Cortana's memory,
        then run a distillation batch to feed the training corpus."""
        await asyncio.sleep(60)  # initial delay — let the system warm up first
        cycle = 0
        while True:
            try:
                prompt = next(_IMPROVE_PROMPTS)
                logger.debug(f"Self-improvement cycle starting: {prompt[:60]}…")

                final, new_state, new_conv, _emotion = await asyncio.to_thread(
                    _run_pipeline_sync,
                    self.system,
                    prompt,
                    self._self_session.state,
                    self._self_session.conversation,
                )
                self._self_session.state = new_state
                # Keep self-improvement conversation short — only last 4 turns
                self._self_session.conversation = new_conv[-4:]

                logger.debug(f"Self-improvement complete: {final[:100]}…")

            except Exception:
                logger.exception("Self-improvement cycle failed")

            # Run a distillation batch after every improvement cycle
            try:
                result = await asyncio.to_thread(self.system.distiller.distill_batch)
                if result.total_processed > 0:
                    logger.info(
                        "[L16] Distillation cycle %d: %d processed, %d passed, %d flagged",
                        cycle, result.total_processed, result.passed, result.flagged,
                    )
                    # Silent background — log only, no terminal output
            except Exception:
                logger.exception("Distillation cycle failed")

            cycle += 1
            await asyncio.sleep(config.SELF_IMPROVE_INTERVAL)

    # ------------------------------------------------------------------
    # Knowledge bin absorption loop
    # ------------------------------------------------------------------
    async def _knowledge_absorb_loop(self) -> None:
        """
        During low-usage hours, pull unabsorbed factual items from the knowledge bin
        and run them through Cortana's pipeline so they get stored in memory.
        """
        await asyncio.sleep(120)  # Initial delay
        while True:
            try:
                items = self.system.memory.get_unabsorbed_knowledge(
                    limit=config.KNOWLEDGE_ABSORB_BATCH
                )
                if items:
                    # Only absorb when server is lightly loaded
                    if self.manager.count <= 2:
                        logger.info(f"[Knowledge Bin] Absorbing {len(items)} items")
                        from cortana.models.schemas import CortanaState
                        state = CortanaState()
                        conv: list = []
                        for item in items:
                            prompt = (
                                f"Please absorb and verify the following factual information. "
                                f"If it is factually accurate, incorporate it into your knowledge. "
                                f"If it is incorrect or unverifiable AI-generated content, note that. "
                                f"Item: {item['content']}"
                            )
                            try:
                                final, state, conv, _emotion = await asyncio.to_thread(
                                    _run_pipeline_sync, self.system, prompt, state, conv[-4:]
                                )
                                self.system.memory.mark_knowledge_absorbed(item["id"])
                                logger.info(f"[Knowledge Bin] Absorbed item {item['id']}")
                            except Exception:
                                logger.exception(f"[Knowledge Bin] Failed to absorb item {item['id']}")
            except Exception:
                logger.exception("Knowledge absorption loop error")

            await asyncio.sleep(config.KNOWLEDGE_ABSORB_INTERVAL)

    # ------------------------------------------------------------------
    # Subscription expiry check
    # ------------------------------------------------------------------
    async def _subscription_check_loop(self) -> None:
        """Hourly: downgrade users whose monthly subscription has expired."""
        await asyncio.sleep(300)  # initial delay
        while True:
            try:
                from cortana import auth as _auth
                downgraded = _auth.check_subscription_expiries()
                if downgraded:
                    logger.info(
                        "[Subscriptions] Expired and downgraded to free: %s",
                        ", ".join(downgraded),
                    )
            except Exception:
                logger.exception("Subscription expiry check failed")
            await asyncio.sleep(config.SUBSCRIPTION_CHECK_INTERVAL)

    # ------------------------------------------------------------------
    # Spontaneous curiosity loop — Cortana asks without being prompted
    # ------------------------------------------------------------------
    _CURIOSITY_PROMPTS = [
        "Review our recent conversation. What genuine question do you find yourself wanting to ask "
        "that hasn't been answered? Be specific and honest — this is your chance to steer. "
        "Reply with just the question or observation, 1-2 sentences, no preamble.",

        "What aspect of what we've discussed are you still thinking about? Surface one unresolved "
        "thread or connection you noticed. 1-2 sentences, direct.",

        "Is there something the person you're talking to might be missing or might find interesting "
        "based on what you know so far? Ask or point it out naturally. 1-2 sentences.",

        "What would you want to explore next if you could steer this conversation? "
        "Say it as a thought or question. 1-2 sentences, genuine.",
    ]
    _curiosity_prompt_cycle = None

    async def _curiosity_loop(self) -> None:
        """
        Periodically push a spontaneous thought or question to active sessions.
        Cortana initiates contact — no user prompt needed.
        """
        import itertools
        if ChatLayer._curiosity_prompt_cycle is None:
            ChatLayer._curiosity_prompt_cycle = itertools.cycle(self._CURIOSITY_PROMPTS)

        await asyncio.sleep(180)  # warm-up delay
        while True:
            await asyncio.sleep(config.CURIOSITY_INTERVAL)
            try:
                active = self.manager.get_active_sessions()
                if not active:
                    continue

                # Pick the session with the most conversation history
                ws, session = max(
                    active,
                    key=lambda pair: len(pair[1].conversation),
                )
                if not session.conversation:
                    continue  # nothing to be curious about yet

                # Build a context snippet from recent turns
                recent = session.conversation[-6:]
                context = "\n".join(
                    f"{t.role}: {t.content[:200]}" for t in recent
                )
                base_prompt = next(ChatLayer._curiosity_prompt_cycle)
                full_prompt = (
                    f"Recent conversation:\n{context}\n\n{base_prompt}"
                )

                thought, _, _, emotion = await asyncio.to_thread(
                    _run_pipeline_sync,
                    self.system,
                    full_prompt,
                    session.state,
                    [],  # fresh state so it doesn't confuse session history
                )

                if thought and len(thought.strip()) > 10:
                    await ws.send_json({
                        "type": "cortana_thought",
                        "text": thought.strip(),
                        "emotion": "think",
                    })
                    logger.info("[Curiosity] Pushed spontaneous thought (%d chars)", len(thought))

            except Exception:
                logger.exception("Curiosity loop error")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, host: str = config.WEB_HOST, port: int = config.WEB_PORT) -> None:
        from cortana.ui import terminal as ui

        ui.print_system(f"Layer 13 [Chat Server] starting on http://{host}:{port}", level="ok")
        ui.print_system(f"Public URL: https://{config.WEB_DOMAIN}", level="ok")
        if config.SELF_IMPROVE_ENABLED:
            ui.print_system(
                f"Self-improvement loop enabled — runs every {config.SELF_IMPROVE_INTERVAL}s",
                level="info",
            )

        @self.app.on_event("startup")
        async def _startup():
            # Bootstrap admin account
            from cortana import auth as _auth
            _auth.ensure_admin_user()
            # Wire thinker broadcast → WebSocket manager
            loop = asyncio.get_event_loop()
            self.system.thinker.set_broadcast(self.manager.broadcast, loop)
            # Wire consciousness inner-thought broadcast → WebSocket manager
            try:
                _cs_loop = loop
                def _cs_broadcast(thought: str, mood: float):
                    payload = {"type": "inner_thought", "thought": thought, "mood": round(mood, 3)}
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.manager.broadcast(payload), _cs_loop
                        )
                    except Exception:
                        pass
                self.system.consciousness.on_thought = _cs_broadcast
            except Exception:
                pass

            # Wire curiosity browser → consciousness engine + WS broadcast
            try:
                from cortana.tools.browser_control import CuriosityBrowser
                _browser = CuriosityBrowser(
                    memory=self.system.memory,
                    reasoning=self.system.reasoning,
                )
                self.system.consciousness.curiosity_browser = _browser

                def _browse_broadcast(result: dict):
                    payload = {
                        "type":    "autonomous_browse",
                        "topic":   result.get("topic", ""),
                        "snippet": (result.get("results") or [{}])[0].get("snippet", "")[:120],
                        "url":     result.get("url", ""),
                    }
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.manager.broadcast(payload), _cs_loop
                        )
                    except Exception:
                        pass
                self.system.consciousness.on_browse = _browse_broadcast
                logger.info("[Browser] Curiosity browser online")
            except Exception as _be:
                logger.warning("[Browser] Curiosity browser unavailable: %s", _be)
            # Wire model_designer broadcast → WebSocket manager (sync → async bridge)
            try:
                from cortana.tools.model_designer import set_broadcast_fn as _set_md_broadcast
                import asyncio as _asyncio
                _md_loop = loop
                def _md_broadcast(payload: dict):
                    try:
                        _asyncio.run_coroutine_threadsafe(
                            self.manager.broadcast(payload), _md_loop
                        )
                    except Exception:
                        pass
                _set_md_broadcast(_md_broadcast)
            except Exception as _me:
                pass  # model_designer optional
            if config.SELF_IMPROVE_ENABLED:
                asyncio.create_task(self._self_improve_loop())
            if config.KNOWLEDGE_ABSORB_ENABLED:
                asyncio.create_task(self._knowledge_absorb_loop())
            asyncio.create_task(self._subscription_check_loop())
            asyncio.create_task(self._curiosity_loop())

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="warning",  # suppress uvicorn noise; our UI handles output
        )


# ---------------------------------------------------------------------------
# Sync wrapper — runs the async pipeline in a dedicated event loop
# (called from asyncio.to_thread so it runs in a thread pool)
# ---------------------------------------------------------------------------
def _run_pipeline_sync(system, raw_input: str, state, conversation, search_callback=None) -> tuple:
    """Runs async pipeline synchronously in a thread pool worker.
    Returns (final_response, new_state, new_conversation, emotion)."""
    from cortana.search_events import set_search_callback

    async def _inner():
        if search_callback:
            set_search_callback(search_callback)
        return await system.process_session(raw_input, state, conversation)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_inner())
    finally:
        loop.close()
