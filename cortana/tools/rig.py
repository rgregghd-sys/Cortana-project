"""
rig.py — CLI wrapper to add a skeleton to a static GLB.

Usage:
  python -m cortana.tools.rig
  python -m cortana.tools.rig --input cortana.glb --output cortana_rigged.glb

Workflow:
  1. Invokes Blender headlessly with rig_glb.py
  2. Blender computes mesh bounds, builds a 10-bone armature (Root→Head chain
     + shoulders + arms), auto-weight-paints all mesh objects, exports GLB.
  3. The output GLB has a full skeleton (skin + weights) and is ready for
     animate.py to apply LLM-generated bone keyframes.

Requirements:
  - blender (>= 3.4) in PATH or BLENDER_BIN env var set
  - cortana.glb present in cortana/static/
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT   = Path(__file__).parent.parent.parent
_STATIC_DIR     = _PROJECT_ROOT / "cortana" / "static"
_DEFAULT_INPUT  = _STATIC_DIR / "cortana.glb"
_DEFAULT_OUTPUT = _STATIC_DIR / "cortana_rigged.glb"
_BLENDER_SCRIPT = Path(__file__).parent / "rig_glb.py"


def find_blender() -> str:
    blender = os.getenv("BLENDER_BIN") or shutil.which("blender")
    if not blender:
        raise RuntimeError(
            "blender not found in PATH. Install it or set BLENDER_BIN=/path/to/blender"
        )
    return blender


def run_rig(input_glb: Path, output_glb: Path) -> None:
    blender = find_blender()
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python", str(_BLENDER_SCRIPT),
        "--",
        "--input",  str(input_glb),
        "--output", str(output_glb),
    ]
    print(f"[rig] Blender cmd: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Print only [rig] tagged lines + any errors for clarity
    output = result.stdout + result.stderr
    for line in output.splitlines():
        if "[rig]" in line or "Error" in line or "error" in line.lower() or "traceback" in line.lower():
            print(line)

    if result.returncode != 0:
        print("\n[rig] --- Full Blender output (last 3000 chars) ---")
        print(output[-3000:])
        raise RuntimeError(f"Blender exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a skeleton + skin weights to a static GLB."
    )
    parser.add_argument(
        "--input", "-i",
        default=str(_DEFAULT_INPUT),
        help=f"Input static GLB (default: {_DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output rigged GLB (default: {_DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="After rigging, copy output over the original input (replaces cortana.glb)",
    )
    args = parser.parse_args()

    input_glb  = Path(args.input)
    output_glb = Path(args.output)

    if not input_glb.exists():
        print(f"[rig] ERROR: input not found: {input_glb}", file=sys.stderr)
        sys.exit(1)

    print(f"[rig] Rigging  : {input_glb}")
    print(f"[rig] Output   : {output_glb}")

    run_rig(input_glb, output_glb)

    if args.replace:
        backup = input_glb.with_suffix(".unrigged.glb")
        shutil.copy2(input_glb, backup)
        shutil.copy2(output_glb, input_glb)
        print(f"[rig] Replaced {input_glb.name} (backup: {backup.name})")

    print(f"\n[rig] Done. Rigged GLB: {output_glb}")
    print(f"[rig] Now run animate.py against the rigged GLB for per-bone animations:")
    print(f"[rig]   python -m cortana.tools.animate 'nod' --input {output_glb.name}")


if __name__ == "__main__":
    main()
