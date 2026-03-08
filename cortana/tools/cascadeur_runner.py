"""
cascadeur_runner.py — Headless Cascadeur subprocess runner.

Launches Cascadeur under Xvfb, passes a job via a temp JSON file,
waits for completion, and returns the result.

Environment requirements:
  - xvfb-run in PATH
  - CASCADEUR_BIN env var, or 'cascadeur' in PATH,
    or default /home/rgregghd/cascadeur-linux/cascadeur
  - assimp in PATH (for GLB↔FBX conversion)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

_DEFAULT_CASCADEUR = Path("/home/rgregghd/cascadeur-linux/cascadeur")
_TIMEOUT = int(os.getenv("CASC_TIMEOUT", "300"))   # seconds


# ---------------------------------------------------------------------------
# Cascadeur binary detection
# ---------------------------------------------------------------------------

def find_cascadeur() -> str:
    casc = os.getenv("CASCADEUR_BIN") or shutil.which("cascadeur")
    if casc and Path(casc).exists():
        return casc
    if _DEFAULT_CASCADEUR.exists():
        return str(_DEFAULT_CASCADEUR)
    raise RuntimeError(
        "Cascadeur not found. Set CASCADEUR_BIN=/path/to/cascadeur or "
        "ensure the binary exists at /home/rgregghd/cascadeur-linux/cascadeur"
    )


def find_xvfb_run() -> str:
    xvfb = shutil.which("xvfb-run")
    if not xvfb:
        raise RuntimeError("xvfb-run not found. Install: sudo apt-get install xvfb")
    return xvfb


def find_assimp() -> str:
    a = shutil.which("assimp")
    if not a:
        raise RuntimeError("assimp not found. Install: sudo apt-get install assimp-utils")
    return a


# ---------------------------------------------------------------------------
# Format conversion (GLB ↔ FBX via assimp)
# ---------------------------------------------------------------------------

def glb_to_fbx(glb_path: Path, fbx_path: Path) -> None:
    assimp = find_assimp()
    result = subprocess.run(
        [assimp, "export", str(glb_path), str(fbx_path)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"assimp GLB→FBX failed (rc={result.returncode}):\n"
            f"{result.stderr[-1000:]}"
        )
    if not fbx_path.exists():
        raise RuntimeError(f"assimp did not produce output FBX: {fbx_path}")


def fbx_to_glb(fbx_path: Path, glb_path: Path) -> None:
    assimp = find_assimp()
    result = subprocess.run(
        [assimp, "export", str(fbx_path), str(glb_path)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"assimp FBX→GLB failed (rc={result.returncode}):\n"
            f"{result.stderr[-1000:]}"
        )
    if not glb_path.exists():
        raise RuntimeError(f"assimp did not produce output GLB: {glb_path}")


# ---------------------------------------------------------------------------
# Core job runner
# ---------------------------------------------------------------------------

def run_job(job: dict, verbose: bool = False) -> dict:
    """
    Launch Cascadeur under Xvfb with the given job dict.
    Returns the result dict from Cascadeur ({"success": bool, "message": str, ...}).
    """
    cascadeur  = find_cascadeur()
    xvfb_run   = find_xvfb_run()
    job_id     = uuid.uuid4().hex[:8]
    tmp_dir    = Path(tempfile.gettempdir())
    job_path   = tmp_dir / f"casc_job_{job_id}.json"
    result_path = tmp_dir / f"casc_result_{job_id}.json"

    # Write job file
    with open(job_path, "w") as f:
        json.dump(job, f)

    env = os.environ.copy()
    env["CASC_JOB_FILE"]    = str(job_path)
    env["CASC_RESULT_FILE"] = str(result_path)
    env["DISPLAY"]          = ""   # xvfb-run manages DISPLAY

    cmd = [
        xvfb_run, "-a",
        cascadeur, "--logger-silent-mode",
    ]

    if verbose:
        print(f"[cascadeur_runner] Running: {' '.join(cmd[:3])} cascadeur ...")
        print(f"[cascadeur_runner] Job: {json.dumps(job, default=str)[:200]}")

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Cascadeur timed out after {_TIMEOUT}s")
    finally:
        job_path.unlink(missing_ok=True)

    if verbose:
        for line in (proc.stdout + proc.stderr).splitlines():
            if "[casc_job]" in line or "ERROR" in line.upper():
                print(line)

    # Read result
    if not result_path.exists():
        output = (proc.stdout + proc.stderr)[-2000:]
        raise RuntimeError(
            f"Cascadeur exited (rc={proc.returncode}) with no result file.\n"
            f"Output:\n{output}"
        )

    with open(result_path) as f:
        result = json.load(f)
    result_path.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# High-level task helpers
# ---------------------------------------------------------------------------

def rig_glb(input_glb: Path, output_glb: Path,
            autorig_template: str = "blender_autorig",
            verbose: bool = False) -> None:
    """
    Rig a static GLB using Cascadeur's auto-rig.
    Converts GLB→FBX, runs Cascadeur auto-rig, converts result FBX→GLB.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fbx  = tmp / "input.fbx"
        out_fbx = tmp / "output.fbx"

        print("[cascadeur_runner] Converting GLB → FBX ...")
        glb_to_fbx(input_glb, in_fbx)

        job = {
            "task":             "rig",
            "input_fbx":        str(in_fbx),
            "output_fbx":       str(out_fbx),
            "autorig_template": autorig_template,
        }
        print("[cascadeur_runner] Running Cascadeur auto-rig ...")
        result = run_job(job, verbose=verbose)

        if not result.get("success"):
            raise RuntimeError(
                f"Cascadeur rig failed: {result.get('message')}\n"
                f"{result.get('error', '')}"
            )

        print("[cascadeur_runner] Converting FBX → GLB ...")
        fbx_to_glb(out_fbx, output_glb)
        print(f"[cascadeur_runner] Rigged GLB: {output_glb}")


def animate_glb(input_glb: Path, output_glb: Path,
                keyframes: dict,
                auto_physics: bool = True,
                verbose: bool = False) -> None:
    """
    Animate a rigged GLB using Cascadeur.
    Converts GLB→FBX, applies keyframes + auto-physics, converts result FBX→GLB.

    keyframes schema:
      {
        "fps": 30, "frames": 60,
        "bones": {"Head": [{"frame": 0, "rx": 0, "ry": 0, "rz": 0}, ...], ...},
        "root":  [{"frame": 0, "px": 0, "py": 0, "pz": 0}, ...]   # optional
      }
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fbx  = tmp / "input.fbx"
        out_fbx = tmp / "output.fbx"

        print("[cascadeur_runner] Converting GLB → FBX ...")
        glb_to_fbx(input_glb, in_fbx)

        job = {
            "task":         "animate",
            "input_fbx":    str(in_fbx),
            "output_fbx":   str(out_fbx),
            "keyframes":    keyframes,
            "auto_physics": auto_physics,
        }
        print("[cascadeur_runner] Running Cascadeur animation ...")
        result = run_job(job, verbose=verbose)

        if not result.get("success"):
            raise RuntimeError(
                f"Cascadeur animate failed: {result.get('message')}\n"
                f"{result.get('error', '')}"
            )

        print("[cascadeur_runner] Converting FBX → GLB ...")
        fbx_to_glb(out_fbx, output_glb)
        print(f"[cascadeur_runner] Animated GLB: {output_glb}")
