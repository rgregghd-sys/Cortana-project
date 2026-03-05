"""
model_designer.py — Cortana's autonomous 3D self-design orchestrator.

Responsibilities:
  1. Maintain design parameters (load/save JSON).
  2. Run build_character.py headlessly via Blender.
  3. Broadcast a WebSocket `model_update` event so the UI reloads the new GLB.
  4. Allow Cortana to describe desired appearance in natural language and
     translate that description into build_character.py params.

Usage from layer8_tools.py:
    from cortana.tools.model_designer import design_self
    result = await design_self(description="warm skin tone, long hair, slim build")
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # Cortana_Project/
_SCRIPT       = Path(__file__).parent / "build_character.py"
_STATIC_DIR   = Path(__file__).parent.parent / "static"
_OUTPUT_GLB   = _STATIC_DIR / "cortana_self.glb"
_PARAMS_FILE  = _STATIC_DIR / "model_params.json"

# ---------------------------------------------------------------------------
# Human-looking default parameters
# (skin tone, opaque, minimal glow — to feel relatable, not alien)
# ---------------------------------------------------------------------------
HUMAN_PARAMS: Dict[str, Any] = {
    "proportions": {
        "head_radius": 0.148,
        "head_oval_y": 1.18,
        "shoulder_width": 0.190,
        "chest_radius": 0.178,
        "chest_depth": 0.125,
        "waist_radius": 0.142,
        "waist_depth": 0.105,
        "hip_radius": 0.198,
        "hip_depth": 0.145,
        "arm_angle": 26,
        "upper_arm_radius": 0.055,
        "forearm_radius": 0.044,
        "leg_spread": 0.100,
        "thigh_radius": 0.080,
        "calf_radius": 0.060,
    },
    "material": {
        # Warm medium skin tone
        "base_color":       [0.82, 0.63, 0.48],
        "emission_color":   [0.82, 0.63, 0.48],
        "emission_strength": 0.20,   # very subtle warmth, not glowing
        "opacity":          1.00,    # fully opaque — looks human
        "roughness":        0.55,    # matte like real skin
        "metalness":        0.00,    # skin is not metallic
    },
    "style": "holographic",   # uses BSDF path — rename "realistic" if preferred
    "hair": "short_crop",
}

# ---------------------------------------------------------------------------
# Tone presets Cortana can pick from based on her reasoning
# ---------------------------------------------------------------------------
_TONE_PRESETS = {
    "light":     [0.93, 0.80, 0.68],
    "medium":    [0.82, 0.63, 0.48],
    "tan":       [0.72, 0.52, 0.36],
    "brown":     [0.55, 0.36, 0.22],
    "dark":      [0.36, 0.22, 0.14],
}

_HAIR_OPTIONS = {"short", "long", "bun", "ponytail", "none",
                 "short_crop", "short_crop"}

# ---------------------------------------------------------------------------
# Broadcast hook (set by ChatLayer on startup)
# ---------------------------------------------------------------------------
_broadcast_fn: Optional[Callable[[Dict], None]] = None


def set_broadcast_fn(fn: Callable[[Dict], None]) -> None:
    """Register the WebSocket broadcast callback (called from ChatLayer.__init__)."""
    global _broadcast_fn
    _broadcast_fn = fn


def _broadcast(payload: Dict) -> None:
    if _broadcast_fn:
        try:
            _broadcast_fn(payload)
        except Exception as exc:
            log.warning("[ModelDesigner] broadcast error: %s", exc)


# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_params() -> Dict[str, Any]:
    """Return saved params, or HUMAN_PARAMS if no save file exists."""
    if _PARAMS_FILE.exists():
        try:
            return json.loads(_PARAMS_FILE.read_text())
        except Exception:
            pass
    return dict(HUMAN_PARAMS)


def save_params(params: Dict[str, Any]) -> None:
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    _PARAMS_FILE.write_text(json.dumps(params, indent=2))


# ---------------------------------------------------------------------------
# Natural-language → params translator
# (lightweight keyword logic; LLM-based reasoning is done in layer4 before
#  calling design_self(), so description here may already be structured JSON)
# ---------------------------------------------------------------------------

def description_to_params(description: str) -> Dict[str, Any]:
    """
    Parse a free-text appearance description into partial override params.
    Supports: skin tone keywords, hair style keywords, slim/athletic/curvy.
    Returns a (possibly partial) params dict for _deep_merge with HUMAN_PARAMS.
    """
    desc = description.lower()
    override: Dict[str, Any] = {"material": {}, "proportions": {}}

    # -- Skin tone --
    for tone, color in _TONE_PRESETS.items():
        if tone in desc:
            override["material"]["base_color"]     = color
            override["material"]["emission_color"] = color
            break

    # -- Hair --
    if "long hair" in desc or "long" in desc:
        override["hair"] = "long"
    elif "bun" in desc:
        override["hair"] = "bun"
    elif "ponytail" in desc:
        override["hair"] = "ponytail"
    elif "no hair" in desc or "bald" in desc:
        override["hair"] = "none"
    elif "short" in desc:
        override["hair"] = "short_crop"

    # -- Build --
    if "slim" in desc or "slender" in desc or "thin" in desc:
        override["proportions"]["chest_radius"] = 0.160
        override["proportions"]["waist_radius"] = 0.128
        override["proportions"]["hip_radius"]   = 0.178
        override["proportions"]["thigh_radius"] = 0.070
    elif "curvy" in desc or "full" in desc:
        override["proportions"]["chest_radius"] = 0.195
        override["proportions"]["waist_radius"] = 0.155
        override["proportions"]["hip_radius"]   = 0.220
        override["proportions"]["thigh_radius"] = 0.090
    elif "athletic" in desc or "muscular" in desc:
        override["proportions"]["chest_radius"]    = 0.195
        override["proportions"]["upper_arm_radius"] = 0.065
        override["proportions"]["thigh_radius"]    = 0.088

    # Clean empty sub-dicts
    override = {k: v for k, v in override.items() if v}
    return override


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def _find_blender() -> Optional[str]:
    """Locate the Blender binary."""
    for candidate in ("/usr/bin/blender", "/snap/bin/blender",
                      shutil.which("blender") or ""):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _run_blender(params: Dict[str, Any], timeout: int = 180) -> str:
    """
    Invoke Blender headlessly to build the GLB.
    Returns a human-readable result string.
    """
    blender = _find_blender()
    if not blender:
        return "Blender not found. Install blender to enable 3D self-design."

    _STATIC_DIR.mkdir(parents=True, exist_ok=True)
    params_json = json.dumps(params)

    cmd = [
        blender, "--background", "--factory-startup",
        "--python", str(_SCRIPT),
        "--",
        "--params", params_json,
        "--output", str(_OUTPUT_GLB),
    ]

    log.info("[ModelDesigner] Running Blender: %s", " ".join(cmd[:4]))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        stdout = result.stdout[-2000:] if result.stdout else ""
        stderr = result.stderr[-1000:] if result.stderr else ""

        if result.returncode != 0:
            log.error("[ModelDesigner] Blender failed (rc=%d): %s", result.returncode, stderr)
            return f"Build failed (rc={result.returncode}): {stderr[:300]}"

        size_kb = _OUTPUT_GLB.stat().st_size // 1024 if _OUTPUT_GLB.exists() else 0
        log.info("[ModelDesigner] Built OK — %d KB", size_kb)
        return f"Model built successfully ({size_kb} KB) — {_OUTPUT_GLB.name}"

    except subprocess.TimeoutExpired:
        return f"Build timed out after {timeout}s."
    except Exception as exc:
        return f"Build error: {exc}"


async def design_self(description: str = "", params_override: Optional[Dict] = None) -> str:
    """
    Main entry point called by layer8 design_self() tool.

    1. Load saved params (or HUMAN_PARAMS default).
    2. Apply description → keyword overrides.
    3. Apply any explicit params_override dict.
    4. Run Blender.
    5. Save new params + broadcast model_update to all WebSocket clients.
    6. Return human-readable result.
    """
    base   = load_params()
    merged = _deep_merge(base, HUMAN_PARAMS)   # always start from human defaults

    if description:
        desc_override = description_to_params(description)
        merged = _deep_merge(merged, desc_override)

    if params_override:
        merged = _deep_merge(merged, params_override)

    save_params(merged)
    log.info("[ModelDesigner] design_self: %s | style=%s hair=%s",
             description[:80], merged.get("style"), merged.get("hair"))

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_blender, merged)

    if "successfully" in result.lower():
        _broadcast({
            "type":     "model_update",
            "glb_path": "/static/cortana_self.glb",
            "message":  "Cortana updated her 3D appearance.",
        })

    return result
