"""
rig.py — CLI wrapper to add a skeleton to a static GLB via Cascadeur.

Usage:
  python -m cortana.tools.rig
  python -m cortana.tools.rig --input cortana.glb --output cortana_rigged.glb
  python -m cortana.tools.rig --template standard

Workflow:
  1. Converts the input GLB → FBX (via assimp)
  2. Launches Cascadeur under Xvfb — auto-rigs using the named template
     (default: blender_autorig, which maps Blender bone naming conventions)
  3. Exports the rigged FBX from Cascadeur
  4. Converts the rigged FBX → GLB (via assimp)

Requirements:
  - xvfb-run in PATH
  - assimp in PATH  (sudo apt-get install assimp-utils)
  - Cascadeur at /home/rgregghd/cascadeur-linux/cascadeur
    or set CASCADEUR_BIN env var
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT   = Path(__file__).parent.parent.parent
_STATIC_DIR     = _PROJECT_ROOT / "cortana" / "static"
_DEFAULT_INPUT  = _STATIC_DIR / "cortana.glb"
_DEFAULT_OUTPUT = _STATIC_DIR / "cortana_rigged.glb"


def run_rig(input_glb: Path, output_glb: Path,
            template: str = "blender_autorig",
            verbose: bool = False) -> None:
    from cortana.tools.cascadeur_runner import rig_glb
    rig_glb(input_glb, output_glb, autorig_template=template, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a skeleton to a static GLB via Cascadeur auto-rig."
    )
    parser.add_argument("--input", "-i", default=str(_DEFAULT_INPUT),
                        help=f"Input static GLB (default: {_DEFAULT_INPUT.name})")
    parser.add_argument("--output", "-o", default=str(_DEFAULT_OUTPUT),
                        help=f"Output rigged GLB (default: {_DEFAULT_OUTPUT.name})")
    parser.add_argument("--template", "-t", default="blender_autorig",
                        help="Cascadeur auto-rig template name (default: blender_autorig)")
    parser.add_argument("--replace", action="store_true",
                        help="After rigging, copy output over the original input")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed Cascadeur output")
    args = parser.parse_args()

    input_glb  = Path(args.input)
    output_glb = Path(args.output)

    if not input_glb.exists():
        print(f"[rig] ERROR: input not found: {input_glb}", file=sys.stderr)
        sys.exit(1)

    print(f"[rig] Input    : {input_glb}")
    print(f"[rig] Output   : {output_glb}")
    print(f"[rig] Template : {args.template}")

    run_rig(input_glb, output_glb, template=args.template, verbose=args.verbose)

    if args.replace:
        backup = input_glb.with_suffix(".unrigged.glb")
        shutil.copy2(input_glb, backup)
        shutil.copy2(output_glb, input_glb)
        print(f"[rig] Replaced {input_glb.name} (backup: {backup.name})")

    print(f"\n[rig] Done. Rigged GLB: {output_glb}")
    print(f"[rig] Now run animate.py against the rigged GLB:")
    print(f"[rig]   python -m cortana.tools.animate 'nod' --input {output_glb.name}")


if __name__ == "__main__":
    main()
