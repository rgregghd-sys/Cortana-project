"""
build_character.py — Build/export a character GLB using Cascadeur as the base.

Cascadeur ships with 'Cascy.casc' — a fully rigged humanoid base mesh.
This script:
  1. Opens the Cascy sample scene in Cascadeur under Xvfb
  2. Cascadeur exports it as FBX via the scene_opened job runner
  3. Converts the FBX → GLB (via assimp) and writes to cortana/static/

Usage:
  python -m cortana.tools.build_character --output cortana_self.glb
  python -m cortana.tools.build_character --params '{"hair":"long","material":{"base_color":[0.8,0.6,0.4]}}'

Note: Geometry/proportion params are logged but not applied at this stage —
Cascadeur's Cascy base mesh is used as-is. Material and pose changes are
applied downstream via the animate/design pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

_CASCADEUR_DIR = Path("/home/rgregghd/cascadeur-linux")
_CASCY_CASC    = _CASCADEUR_DIR / "samples" / "Cascy.casc"
_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_STATIC_DIR    = _PROJECT_ROOT / "cortana" / "static"
_DEFAULT_OUT   = _STATIC_DIR / "cortana_self.glb"


def find_cascadeur() -> str:
    casc = os.getenv("CASCADEUR_BIN")
    if casc and Path(casc).exists():
        return casc
    default = _CASCADEUR_DIR / "cascadeur"
    if default.exists():
        return str(default)
    found = shutil.which("cascadeur")
    if found:
        return found
    raise RuntimeError("Cascadeur binary not found.")


def find_assimp() -> str:
    a = shutil.which("assimp")
    if not a:
        raise RuntimeError("assimp not found. Install: sudo apt-get install assimp-utils")
    return a


def export_cascy_to_fbx(output_fbx: Path) -> None:
    """
    Open Cascy.casc in Cascadeur under Xvfb and export as FBX.
    The scene_opened job runner handles the export_scene task.
    """
    if not _CASCY_CASC.exists():
        raise RuntimeError(f"Cascy sample not found: {_CASCY_CASC}")

    xvfb = shutil.which("xvfb-run")
    if not xvfb:
        raise RuntimeError("xvfb-run not found.")

    job_id      = uuid.uuid4().hex[:8]
    tmp_dir     = Path(tempfile.gettempdir())
    job_path    = tmp_dir / f"casc_job_{job_id}.json"
    result_path = tmp_dir / f"casc_result_{job_id}.json"

    job = {
        "task":       "export_scene",
        "output_fbx": str(output_fbx),
    }
    with open(job_path, "w") as f:
        json.dump(job, f)

    env = os.environ.copy()
    env["CASC_JOB_FILE"]    = str(job_path)
    env["CASC_RESULT_FILE"] = str(result_path)
    env["DISPLAY"]          = ""

    cascadeur = find_cascadeur()
    cmd = [xvfb, "-a", cascadeur, "--logger-silent-mode", str(_CASCY_CASC)]

    print("[build_character] Opening Cascy in Cascadeur ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    job_path.unlink(missing_ok=True)

    if not result_path.exists():
        out = (result.stdout + result.stderr)[-1500:]
        raise RuntimeError(f"Cascadeur produced no result.\nOutput:\n{out}")

    with open(result_path) as f:
        res = json.load(f)
    result_path.unlink(missing_ok=True)

    if not res.get("success"):
        raise RuntimeError(f"Cascadeur export failed: {res.get('message')}")

    print(f"[build_character] FBX exported: {output_fbx}")


def build(params: dict, output_glb: Path) -> str:
    """
    Build a character GLB from the Cascy base mesh.
    params are noted for logging; geometry changes are applied in later stages.
    """
    if params:
        print(f"[build_character] Params: {json.dumps(params)[:120]}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_fbx = tmp / "cascy.fbx"

        export_cascy_to_fbx(out_fbx)

        assimp = find_assimp()
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [assimp, "export", str(out_fbx), str(output_glb)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"assimp FBX→GLB failed:\n{result.stderr[-500:]}")

    size_kb = output_glb.stat().st_size // 1024
    return f"Character GLB built from Cascy base ({size_kb} KB) — {output_glb.name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a character GLB using Cascadeur's Cascy base mesh."
    )
    parser.add_argument("--output", "-o", default=str(_DEFAULT_OUT),
                        help=f"Output GLB path (default: {_DEFAULT_OUT.name})")
    parser.add_argument("--params", default=None,
                        help="JSON params string (logged; geometry not modified)")
    args = parser.parse_args()

    params     = json.loads(args.params) if args.params else {}
    output_glb = Path(args.output)

    try:
        msg = build(params, output_glb)
        print(f"[build_character] {msg}")
    except Exception as e:
        print(f"[build_character] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
