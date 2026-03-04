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

import pathlib

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

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


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)

    async def broadcast(self, data: dict) -> None:
        dead = set()
        for ws in self._active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
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
.cam-controls{display:flex;gap:6px;margin-top:8px}
.cam-btn{flex:1;padding:5px 0;border-radius:7px;border:1px solid var(--border);
  background:rgba(0,185,255,0.08);color:var(--blue);font-family:var(--mono);
  font-size:9.5px;cursor:pointer;transition:background .2s}
.cam-btn:hover{background:rgba(0,185,255,0.18)}
.cam-btn.active{background:rgba(0,185,255,0.22);border-color:var(--blue)}
.cam-status{text-align:center;margin-top:6px;color:var(--dim);font-size:9px;min-height:12px}
#camToggleBtn{
  padding:7px 12px;border-radius:8px;border:1px solid var(--border);
  background:rgba(0,185,255,0.06);color:var(--blue);font-family:var(--mono);
  font-size:11px;cursor:pointer;transition:background .2s;white-space:nowrap;
}
#camToggleBtn:hover{background:rgba(0,185,255,0.15)}
#camToggleBtn.active{border-color:var(--blue);background:rgba(0,185,255,0.18)}

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
.learn-badge{color:var(--blue-dim)}
.bg-badge{color:var(--yellow)}
@keyframes pulseA{0%,100%{opacity:1}50%{opacity:.22}}
#userBar{display:none;align-items:center;gap:8px;font-size:11px;font-family:var(--mono);color:var(--dim)}
#userBar.visible{display:flex}
.user-tier{color:var(--blue);font-weight:600;text-transform:uppercase;font-size:10px}
.hdr-btn{
  background:rgba(0,185,255,0.07);border:1px solid var(--border);
  border-radius:7px;color:var(--blue);font-size:10px;font-family:var(--mono);
  cursor:pointer;padding:5px 11px;transition:background .15s,box-shadow .15s;
}
.hdr-btn:hover{background:rgba(0,185,255,0.18);box-shadow:0 0 10px rgba(0,185,255,0.12)}

/* ── Expression + graph HUD (floats just above chat) ── */
#exprHud{
  position:fixed;left:50%;transform:translateX(-50%);
  bottom:calc(var(--chat-h) + 14px);
  z-index:8;display:flex;align-items:center;gap:14px;
  transition:bottom .32s cubic-bezier(.4,0,.2,1);
}
#exprLabel{
  font-size:11px;font-family:var(--mono);color:var(--blue);
  letter-spacing:1.4px;text-transform:uppercase;
  text-shadow:0 0 12px rgba(0,216,255,0.4);
}
.graph-btn{
  padding:5px 12px;background:rgba(0,185,255,0.07);
  border:1px solid var(--border);border-radius:7px;
  color:var(--blue-dim);font-size:10px;font-family:var(--mono);
  cursor:pointer;transition:all .15s;
}
.graph-btn:hover{background:rgba(0,185,255,0.18);color:var(--blue)}

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
  padding:8px 18px 14px;flex-shrink:0;
  background:rgba(0,5,18,0.55);
  border-top:1px solid rgba(0,185,255,0.10);
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
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

<!-- Floating header -->
<header>
  <!-- AI name centered -->
  <div class="brand-name">CORTANA</div>
  <div class="status-bar">
    <div class="pulse-badge bg-badge" id="bgBadge">
      <div class="dot" style="background:var(--yellow);box-shadow:0 0 7px var(--yellow)"></div>THINKING
    </div>
    <div class="pulse-badge learn-badge" id="learnBadge">
      <div class="dot online"></div>LEARNING
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
      <button class="hdr-btn" id="logoutBtn">Logout</button>
    </div>
    <button class="hdr-btn" id="loginBtn">Login</button>
  </div>
</header>

<!-- Auth modal -->
<div id="authModal">
  <div class="auth-box">
    <button class="auth-close" onclick="closeAuth()" title="Close">&#x2715;</button>
    <div class="auth-title">CORTANA ACCESS</div>
    <div class="auth-tabs">
      <button class="auth-tab active" id="tabLogin" onclick="switchTab('login')">Login</button>
      <button class="auth-tab" id="tabRegister" onclick="switchTab('register')">Register</button>
    </div>
    <div id="formLogin">
      <div class="auth-field"><label>Username</label><input id="loginUser" type="text" autocomplete="username"></div>
      <div class="auth-field"><label>Password</label><input id="loginPass" type="password" autocomplete="current-password"></div>
      <button class="auth-btn" onclick="doLogin()">Login</button>
    </div>
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

<!-- Expression + graph HUD -->
<div id="exprHud">
  <span id="exprLabel">idle</span>
  <button class="graph-btn" id="graphBtn">&#x2B21; Knowledge Graph</button>
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
    <button id="send" disabled>Send</button>
  </div>
</div>

<!-- Webcam panel -->
<div id="camPanel">
  <video id="camVideo" autoplay muted playsinline></video>
  <div class="cam-controls">
    <button class="cam-btn" id="camSnapBtn">&#x1F4F8; Snapshot</button>
    <button class="cam-btn" id="camAutoBtn">&#x23F1; Auto</button>
  </div>
  <div class="cam-status" id="camStatus">Camera ready</div>
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

const loader = new GLTFLoader();
loader.load('/static/cortana_rigged.glb', (gltf) => {
  const model  = gltf.scene;
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
}, undefined, (err) => console.error('[GLB]', err));

// ═══════════════════════════════════════════════════════════
//  PROCEDURAL ANIMATION SYSTEM
// ═══════════════════════════════════════════════════════════
const DEG = Math.PI / 180;
function rand(a, b) { return a + Math.random() * (b - a); }

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
  const el = document.getElementById('exprLabel');
  if (el) el.textContent = name;
}

function triggerIdleBehavior() {
  if (anim.emotion !== 'idle' || anim.isTalking) return;
  switch (Math.floor(rand(0, 5))) {
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
  }
}

function tickAnimation(delta) {
  const k = 1 - Math.pow(0.0005, delta);

  anim.breathPhase += delta * 0.42 * anim.breathSpeed;
  const breath = Math.sin(anim.breathPhase) * 0.018 * anim.breathAmp;

  let yawnY = 0;
  if (anim.yawnActive) {
    anim.yawnPhase += delta * 0.62;
    if (anim.yawnPhase >= Math.PI) { anim.yawnActive=false; anim.yawnPhase=0; }
    else yawnY = Math.sin(anim.yawnPhase) * 0.05;
  }

  anim.swayPhase += delta * 0.16;
  const swayX = Math.sin(anim.swayPhase*0.9+1.3) * 0.006;
  const swayZ = Math.cos(anim.swayPhase*0.7)      * 0.004;

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

  anim.rotX += (anim.tRotX+swayX+talkX - anim.rotX) * k*5;
  anim.rotZ += (anim.tRotZ+swayZ        - anim.rotZ) * k*5;
  anim.rotY += (anim.tRotY              - anim.rotY) * k*4;
  anim.posY += (anim.tPosY+breath+yawnY+talkY+laughY - anim.posY) * k*7;
  anim.scl  += (anim.tScl               - anim.scl)  * k*4;

  modelPivot.rotation.x = anim.rotX;
  modelPivot.rotation.y = anim.rotY + talkX*0.35;
  modelPivot.rotation.z = anim.rotZ;
  modelPivot.position.y = anim.posY;
  modelPivot.scale.setScalar(anim.scl);

  if (anim.emotionTimer > 0) {
    anim.emotionTimer -= delta;
    if (anim.emotionTimer <= 0) applyEmotion('idle');
  }

  anim.idleTimer += delta;
  if (anim.idleTimer >= anim.nextIdle) {
    anim.idleTimer=0; anim.nextIdle=rand(6,15);
    triggerIdleBehavior();
  }
}

// ── Public API ──
window.setExpression     = (n) => applyEmotion(n);
window.triggerExpression = (n) => applyEmotion(n);
window._startTalking = () => { anim.isTalking=true;  anim.talkDecay=0.3; };
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
if (!window.setExpression) window.setExpression = function(name) {
  const el = document.getElementById('exprLabel'); if (el) el.textContent = name;
};
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

function switchTab(tab){
  document.getElementById('formLogin').style.display   =tab==='login'   ?'':'none';
  document.getElementById('formRegister').style.display=tab==='register'?'':'none';
  document.getElementById('tabLogin').classList.toggle('active',   tab==='login');
  document.getElementById('tabRegister').classList.toggle('active',tab==='register');
  document.getElementById('authErr').textContent='';
}
function openAuth(){document.getElementById('authModal').classList.add('open');loadTiers();}
function closeAuth(){document.getElementById('authModal').classList.remove('open');}

let _tiers={};
async function loadTiers(){
  try{
    const data=await fetch('/api/tiers').then(r=>r.json());
    _tiers=data;
    const container=document.getElementById('tierBtns');
    if(!container) return;
    container.innerHTML='';
    const labels={pro:'Priority routing',premium:'Highest priority'};
    Object.entries(data).forEach(([name,info])=>{
      if(name==='free') return;
      const btn=document.createElement('button');
      btn.className='tier-btn';
      btn.onclick=()=>showTierInfo(name);
      const label=name.charAt(0).toUpperCase()+name.slice(1);
      btn.innerHTML=`${label} &mdash; ${info.daily_limit} msg/2h &mdash; ${labels[name]||''}<span>$${info.price_usd}/mo</span>`;
      container.appendChild(btn);
    });
  }catch(e){}
}
function showTierInfo(tier){
  const info=_tiers[tier];
  const price=info?`$${info.price_usd}/mo`:'';
  const limit=info?`${info.daily_limit} messages per 2 hours`:'';
  const label=tier.charAt(0).toUpperCase()+tier.slice(1);
  document.getElementById('authErr').style.color='var(--blue)';
  document.getElementById('authErr').textContent=`${label}: ${price} \u2014 ${limit}. Pay via ETH to /api/v1/wallet, then contact support to upgrade.`;
}

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
    if(ws)ws.close();
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

function updateUserBar(){
  const bar=document.getElementById('userBar');
  const btn=document.getElementById('loginBtn');
  if(currentUser){
    bar.classList.add('visible'); btn.style.display='none';
    document.getElementById('userNameLabel').textContent=currentUser.username||'';
    document.getElementById('userTierLabel').textContent=currentUser.tier||'free';
    document.getElementById('userLimitLabel').textContent=(currentUser.daily_limit||40)+'/2h';
  } else {
    bar.classList.remove('visible'); btn.style.display='';
  }
}

if(authToken){
  fetch('/api/auth/me',{headers:{'X-Session-Token':authToken}}).then(r=>r.json()).then(d=>{
    if(d.username){currentUser=d;updateUserBar();}
    else{authToken='';localStorage.removeItem('cortana_token');}
  }).catch(()=>{});
}

document.getElementById('loginBtn').onclick=openAuth;
document.getElementById('logoutBtn').onclick=doLogout;

// Expose auth helpers to global scope so inline onclick="" attributes work
// (module scripts are scoped — window assignment bridges that gap)
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
const learnBadge =document.getElementById('learnBadge');
const bgBadge    =document.getElementById('bgBadge');
const toast      =document.getElementById('toast');

let ws=null,learnTimer=null,toastTimer=null;

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
        addMessage('cortana',msg.text);
        turnCount.textContent=msg.turn;
        if(msg.providers)updateProviders(msg.providers);
        sendBtn.disabled=false; input.disabled=false; input.focus();
        break;
      case 'history':
        if(msg.turns&&msg.turns.length){
          msg.turns.forEach(t=>addMessage(t.role==='user'?'user':'cortana',t.content));
          addNote(`\u21BA Restored ${msg.turns.length} turns from previous session`);
          turnCount.textContent=msg.turn||0;
        }
        break;
      case 'learning':
        learnBadge.classList.add('visible');
        clearTimeout(learnTimer);
        learnTimer=setTimeout(()=>learnBadge.classList.remove('visible'),7000);
        addNote('\u2736 Self-improvement cycle\u2026');
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
      case 'security_alert':
        addNote('\u26A0 Security: '+msg.detail);
        triggerExpression('surprised'); break;
      case 'vision_response':
        if(window._visionHandler) window._visionHandler(msg);
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
  ws.send(JSON.stringify({type:'message',message:text}));
  input.value=''; input.style.height='auto';
  input.disabled=true; sendBtn.disabled=true;
}

sendBtn.onclick=sendMessage;
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}});

// ── Webcam vision ──────────────────────────────────────────────────────────
(function(){
  const panel   = document.getElementById('camPanel');
  const video   = document.getElementById('camVideo');
  const snapBtn = document.getElementById('camSnapBtn');
  const autoBtn = document.getElementById('camAutoBtn');
  const status  = document.getElementById('camStatus');
  const toggleBtn = document.getElementById('camToggleBtn');
  let stream = null;
  let autoTimer = null;
  const canvas = document.createElement('canvas');

  toggleBtn.onclick = async () => {
    if (stream) {
      // Turn off
      stream.getTracks().forEach(t => t.stop());
      stream = null;
      video.srcObject = null;
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
    } catch(e) {
      status.textContent = 'Camera denied: ' + e.message;
    }
  };

  function captureFrame(question) {
    if (!stream || !ws || ws.readyState !== WebSocket.OPEN) return;
    const w = video.videoWidth || 320, h = video.videoHeight || 240;
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);
    const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    ws.send(JSON.stringify({type:'vision', image:b64, question: question||''}));
    status.textContent = 'Analyzing\u2026';
  }

  snapBtn.onclick = () => captureFrame(input.value.trim() || 'What do you see?');

  autoBtn.onclick = () => {
    if (autoTimer) {
      clearInterval(autoTimer); autoTimer = null;
      autoBtn.classList.remove('active');
      status.textContent = 'Auto off';
    } else {
      autoBtn.classList.add('active');
      status.textContent = 'Auto on \u2014 every 8s';
      autoTimer = setInterval(() => captureFrame('Briefly describe what you observe.'), 8000);
    }
  };

  // Handle vision responses
  window._visionHandler = (msg) => {
    if (msg.type === 'vision_response') {
      status.textContent = 'Done';
      addMessage('assistant', '[Vision] ' + msg.text);
      if (msg.emotion) setExpression(msg.emotion);
    }
  };
})();
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,110)+'px';});

// ── Knowledge graph modal ──
document.getElementById('graphBtn').onclick=()=>{
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
        # CORS — allow the public domain + localhost for dev
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                f"https://{config.WEB_DOMAIN}",
                f"http://{config.WEB_DOMAIN}",
                "http://localhost:8080",
                "http://127.0.0.1:8080",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
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
        async def api_memory():
            return {"episodes": self.system.memory.get_recent_episodes(limit=10)}

        @app.get("/api/graph")
        async def api_graph():
            return self.system.memory.get_concept_graph(limit=60)

        @app.get("/api/tasks")
        async def api_tasks():
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
            body = await request.json()
            result = _auth.login_user(
                body.get("username", ""),
                body.get("password", ""),
            )
            if not result["ok"]:
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
            return {k: {**v} for k, v in config.TIERS.items()}

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
        async def list_knowledge():
            """List unabsorbed knowledge items."""
            items = self.system.memory.get_unabsorbed_knowledge(limit=50)
            return {"items": items}

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            from cortana import auth as _auth
            await self.manager.connect(websocket)
            session = Session()
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
                    data = await websocket.receive_json()
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

        except Exception as e:
            logger.exception("Pipeline error")
            await websocket.send_json({"type": "error", "message": str(e)})

    # ------------------------------------------------------------------
    # Webcam vision handler
    # ------------------------------------------------------------------
    async def _handle_vision(
        self, websocket: WebSocket, session: Session, data: dict
    ) -> None:
        """Process a webcam frame with Gemini Vision and respond."""
        import base64, io
        image_b64 = data.get("image", "")
        question  = data.get("question", "What do you see?") or "What do you see?"
        if not image_b64:
            await websocket.send_json({"type": "vision_response", "text": "No image received.", "emotion": "confused"})
            return
        try:
            import google.generativeai as genai
            from cortana import config as _cfg
            from PIL import Image as PILImage
            img_bytes = base64.b64decode(image_b64)
            img = PILImage.open(io.BytesIO(img_bytes))
            genai.configure(api_key=_cfg.GEMINI_API_KEY)
            vision_model = genai.GenerativeModel("gemini-2.0-flash")
            system_ctx = (
                "You are Cortana, an AI with vision. "
                "The user has just shown you a webcam frame. "
                "Respond in Cortana's voice: direct, analytical, occasionally dry wit. "
                "Keep it under 3 sentences unless asked for detail."
            )
            response = await asyncio.to_thread(
                lambda: vision_model.generate_content([
                    system_ctx + "\n\nUser asks: " + question, img
                ]).text
            )
            # Detect emotion from content
            lower = response.lower()
            emotion = (
                "smile"     if any(w in lower for w in ["interesting","fascinating","impressive"]) else
                "think"     if any(w in lower for w in ["analyzing","processing","unclear"]) else
                "surprised" if any(w in lower for w in ["unexpected","unusual","strange"]) else
                "idle"
            )
            await websocket.send_json({
                "type": "vision_response",
                "text": response,
                "emotion": emotion,
            })
        except Exception as e:
            logger.exception("Vision error")
            await websocket.send_json({
                "type": "vision_response",
                "text": f"Vision error: {e}",
                "emotion": "frown",
            })

    # ------------------------------------------------------------------
    # Self-improvement background task
    # ------------------------------------------------------------------
    async def _self_improve_loop(self) -> None:
        """Periodically run self-generated prompts to grow Cortana's memory."""
        await asyncio.sleep(60)  # initial delay — let the system warm up first
        while True:
            try:
                prompt = next(_IMPROVE_PROMPTS)
                logger.info(f"Self-improvement cycle starting: {prompt[:60]}…")

                await self.manager.broadcast({"type": "learning"})

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

                logger.info(f"Self-improvement complete: {final[:100]}…")

            except Exception:
                logger.exception("Self-improvement cycle failed")

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
            if config.SELF_IMPROVE_ENABLED:
                asyncio.create_task(self._self_improve_loop())
            if config.KNOWLEDGE_ABSORB_ENABLED:
                asyncio.create_task(self._knowledge_absorb_loop())

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
