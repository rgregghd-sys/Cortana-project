"""
animate.py — Text-to-animation pipeline for Cortana's GLB model via Cascadeur.

Usage:
  python -m cortana.tools.animate "nod slowly while thinking" --input cortana_rigged.glb
  python -m cortana.tools.animate "shake head" --fps 30 --frames 90 --no-physics

Workflow:
  1. Sends text description to the LLM provider → returns keyframe JSON
  2. Converts the rigged input GLB → FBX (via assimp)
  3. Launches Cascadeur under Xvfb — applies keyframes + auto-physics simulation
  4. Exports the animated FBX from Cascadeur
  5. Converts FBX → GLB (via assimp) and copies to cortana/static/

Requirements:
  - xvfb-run in PATH
  - assimp in PATH  (sudo apt-get install assimp-utils)
  - Cascadeur at /home/rgregghd/cascadeur-linux/cascadeur
    or set CASCADEUR_BIN env var
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_STATIC_DIR   = _PROJECT_ROOT / "cortana" / "static"
_DEFAULT_GLB  = _STATIC_DIR / "cortana_rigged.glb"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM keyframe generation (same schema as before)
# ---------------------------------------------------------------------------
_KEYFRAME_SYSTEM = """You are an animation keyframe generator for a 3D humanoid head model.
Output ONLY valid JSON — no prose, no markdown, no code fences.

The JSON schema:
{
  "name": "<clip name>",
  "fps": 30,
  "frames": <total frame count>,
  "bones": {
    "<bone_name>": [
      {"frame": <int>, "rx": <degrees>, "ry": <degrees>, "rz": <degrees>},
      ...
    ]
  },
  "root": [
    {"frame": <int>, "px": <float>, "py": <float>, "pz": <float>},
    ...
  ]
}

Available bones (use only these names):
  Head, Neck, Spine, Spine1, Spine2,
  LeftShoulder, RightShoulder,
  LeftArm, RightArm

Rules:
- Keep rotations subtle: Head rx/ry/rz rarely exceed ±25 degrees
- Minimum 2 keyframes per bone used (start + end for clean loops)
- For a 30-fps clip: 30 frames = 1 second, 90 frames = 3 seconds
- "root" controls body position offset (px/py/pz in meters, usually 0)
- If no root movement needed, omit the "root" key
- Only include bones that actually move
"""


def generate_keyframes(description: str, fps: int = 30, frames: int = 60) -> dict:
    """Call the LLM provider to produce keyframe JSON from a text description."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    from cortana.providers.router import ProviderRouter

    router = ProviderRouter()
    prompt = (
        f"Create animation keyframes for this action: {description}\n"
        f"Target: {frames} frames at {fps} fps ({frames/fps:.1f} seconds).\n"
        f"Output ONLY the JSON object."
    )

    raw = router.think_simple(prompt, system=_KEYFRAME_SYSTEM, max_tokens=2048)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw:\n{raw}") from e

    data.setdefault("fps", fps)
    data.setdefault("frames", frames)
    data.setdefault("name", description[:40].replace(" ", "_"))
    return data


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def pipeline(
    description: str,
    input_glb: Path,
    output_glb: Path,
    fps: int = 30,
    frames: int = 60,
    skip_llm: bool = False,
    keyframe_file: Path | None = None,
    auto_physics: bool = True,
    verbose: bool = False,
) -> None:
    from cortana.tools.cascadeur_runner import animate_glb

    print(f"[animate] Description : {description}")
    print(f"[animate] Input GLB   : {input_glb}")
    print(f"[animate] Output GLB  : {output_glb}")

    # Step 1 — Get keyframes
    if keyframe_file:
        print(f"[animate] Loading keyframes from: {keyframe_file}")
        with open(keyframe_file) as f:
            kf_data = json.load(f)
    elif skip_llm:
        kf_data = _demo_nod(fps, frames)
        print("[animate] Using built-in demo nod keyframes (--skip-llm)")
    else:
        print("[animate] Generating keyframes via LLM ...")
        kf_data = generate_keyframes(description, fps=fps, frames=frames)
        print(f"[animate] Got keyframes for bones: {list(kf_data.get('bones', {}).keys())}")

    # Step 2 — Run Cascadeur
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    animate_glb(
        input_glb, output_glb,
        keyframes=kf_data,
        auto_physics=auto_physics,
        verbose=verbose,
    )

    print(f"[animate] Done! Animated GLB: {output_glb}")

    # Step 3 — Copy to static for live serving
    if output_glb.parent != _STATIC_DIR:
        live_path = _STATIC_DIR / output_glb.name
        shutil.copy2(output_glb, live_path)
        print(f"[animate] Also copied to static: {live_path}")


def _demo_nod(fps: int, frames: int) -> dict:
    mid = frames // 2
    return {
        "name": "demo_nod",
        "fps": fps,
        "frames": frames,
        "bones": {
            "Head": [
                {"frame": 0,      "rx":  0, "ry": 0, "rz": 0},
                {"frame": mid,    "rx": 20, "ry": 0, "rz": 0},
                {"frame": frames, "rx":  0, "ry": 0, "rz": 0},
            ]
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an animated GLB from a text description using Cascadeur."
    )
    parser.add_argument("description", nargs="?", default="nod slowly",
                        help="Natural language animation description")
    parser.add_argument("--input",  "-i", default=str(_DEFAULT_GLB),
                        help=f"Input rigged GLB (default: {_DEFAULT_GLB.name})")
    parser.add_argument("--output", "-o", default=None,
                        help="Output GLB path (default: <stem>_anim.glb)")
    parser.add_argument("--fps",    type=int, default=30)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Use built-in demo nod instead of LLM")
    parser.add_argument("--keyframes", default=None,
                        help="Load keyframes from a JSON file")
    parser.add_argument("--no-physics", action="store_true",
                        help="Disable Cascadeur auto-physics simulation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed Cascadeur output")
    args = parser.parse_args()

    input_glb = Path(args.input)
    if not input_glb.exists():
        print(f"[animate] ERROR: input GLB not found: {input_glb}", file=sys.stderr)
        sys.exit(1)

    output_glb = (
        Path(args.output) if args.output
        else input_glb.parent / f"{input_glb.stem}_anim.glb"
    )
    kf_file = Path(args.keyframes) if args.keyframes else None

    pipeline(
        description=args.description,
        input_glb=input_glb,
        output_glb=output_glb,
        fps=args.fps,
        frames=args.frames,
        skip_llm=args.skip_llm,
        keyframe_file=kf_file,
        auto_physics=not args.no_physics,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
