"""
animate.py — Text-to-animation pipeline for Cortana's GLB model.

Usage:
  python -m cortana.tools.animate "nod slowly while thinking" --input cortana.glb --output out.glb
  python -m cortana.tools.animate "shake head in disbelief" --fps 30 --frames 90

Workflow:
  1. Sends text description to local Llama → returns keyframe JSON
  2. Validates and saves keyframe JSON to a temp file
  3. Invokes Blender headlessly with blender_animate.py
  4. Blender applies keyframes to the GLB and exports animated GLB
  5. Copies output to cortana/static/ so it's live on the web

Requirements:
  - blender (>= 3.4) in PATH or set BLENDER_BIN env var
  - cortana.glb in cortana/static/
  - Local Llama provider running (LLAMA_ENABLED=true + model downloaded)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_STATIC_DIR   = _PROJECT_ROOT / "cortana" / "static"
_DEFAULT_GLB  = _STATIC_DIR / "cortana.glb"
_BLENDER_SCRIPT = Path(__file__).parent / "blender_animate.py"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Llama keyframe generation
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
    """Call local Llama to produce keyframe JSON from a text description."""
    # Import here to avoid circular imports at module level
    sys.path.insert(0, str(_PROJECT_ROOT))
    from cortana.providers.router import ProviderRouter

    router = ProviderRouter()

    prompt = (
        f"Create animation keyframes for this action: {description}\n"
        f"Target: {frames} frames at {fps} fps ({frames/fps:.1f} seconds).\n"
        f"Output ONLY the JSON object."
    )

    raw = router.think_simple(prompt, system=_KEYFRAME_SYSTEM, max_tokens=2048)

    # Strip any accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Llama returned invalid JSON: {e}\n\nRaw output:\n{raw}") from e

    # Inject fps + frames if Llama forgot them
    data.setdefault("fps", fps)
    data.setdefault("frames", frames)
    data.setdefault("name", description[:40].replace(" ", "_"))

    return data


# ---------------------------------------------------------------------------
# Blender invocation
# ---------------------------------------------------------------------------
def find_blender() -> str:
    """Locate the blender binary."""
    blender = os.getenv("BLENDER_BIN") or shutil.which("blender")
    if not blender:
        raise RuntimeError(
            "blender not found in PATH. Install it or set BLENDER_BIN=/path/to/blender"
        )
    return blender


def run_blender(input_glb: Path, output_glb: Path, keyframe_json: Path) -> None:
    """Invoke Blender headlessly to apply keyframes and export GLB."""
    blender = find_blender()
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python", str(_BLENDER_SCRIPT),
        "--",                         # everything after -- goes to the Python script
        "--input",  str(input_glb),
        "--output", str(output_glb),
        "--json",   str(keyframe_json),
    ]
    print(f"[animate] Running Blender: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print("[animate] Blender stdout:")
        print(result.stdout[-3000:])
        print("[animate] Blender stderr:")
        print(result.stderr[-3000:])
        raise RuntimeError(f"Blender exited with code {result.returncode}")

    # Surface any [animate] lines from the Blender script
    for line in (result.stdout + result.stderr).splitlines():
        if "[animate]" in line or "Error" in line:
            print(line)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def pipeline(
    description: str,
    input_glb: Path,
    output_glb: Path,
    fps: int = 30,
    frames: int = 60,
    skip_llama: bool = False,
    keyframe_file: Path | None = None,
) -> None:
    """Full text → keyframe → Blender → GLB pipeline."""

    print(f"[animate] Description : {description}")
    print(f"[animate] Input GLB   : {input_glb}")
    print(f"[animate] Output GLB  : {output_glb}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Step 1 — Generate or load keyframes
        if keyframe_file:
            print(f"[animate] Loading keyframes from: {keyframe_file}")
            with open(keyframe_file) as f:
                kf_data = json.load(f)
        elif skip_llama:
            # Built-in demo: simple head nod
            kf_data = _demo_nod(fps, frames)
            print("[animate] Using built-in demo nod keyframes (--skip-llama)")
        else:
            print("[animate] Generating keyframes via Llama…")
            kf_data = generate_keyframes(description, fps=fps, frames=frames)
            print(f"[animate] Got keyframes: {list(kf_data.get('bones', {}).keys())} bones")

        # Save keyframes to temp file
        kf_path = tmp / "keyframes.json"
        with open(kf_path, "w") as f:
            json.dump(kf_data, f, indent=2)
        print(f"[animate] Keyframe JSON saved to {kf_path}")

        # Step 2 — Run Blender
        run_blender(input_glb, output_glb, kf_path)

    print(f"[animate] Done! Animated GLB saved to: {output_glb}")

    # Step 3 — Optionally copy to static for live serving
    if output_glb.parent != _STATIC_DIR:
        live_path = _STATIC_DIR / output_glb.name
        shutil.copy2(output_glb, live_path)
        print(f"[animate] Also copied to static: {live_path}")


def _demo_nod(fps: int, frames: int) -> dict:
    """Built-in fallback: a simple head nod (no Llama required)."""
    mid = frames // 2
    return {
        "name": "demo_nod",
        "fps": fps,
        "frames": frames,
        "bones": {
            "Head": [
                {"frame": 0,   "rx":  0, "ry": 0, "rz": 0},
                {"frame": mid, "rx": 20, "ry": 0, "rz": 0},
                {"frame": frames, "rx": 0, "ry": 0, "rz": 0},
            ]
        },
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an animated GLB from a text description."
    )
    parser.add_argument("description", nargs="?", default="nod slowly",
                        help="Natural language animation description")
    parser.add_argument("--input",  "-i", default=str(_DEFAULT_GLB),
                        help=f"Input GLB path (default: {_DEFAULT_GLB})")
    parser.add_argument("--output", "-o", default=None,
                        help="Output GLB path (default: <input stem>_anim.glb)")
    parser.add_argument("--fps",    type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--frames", type=int, default=60, help="Total frame count (default: 60 = 2s)")
    parser.add_argument("--skip-llama", action="store_true",
                        help="Use built-in demo nod instead of calling Llama")
    parser.add_argument("--keyframes", default=None,
                        help="Load keyframes from a JSON file instead of generating them")
    args = parser.parse_args()

    input_glb = Path(args.input)
    if not input_glb.exists():
        print(f"[animate] ERROR: input GLB not found: {input_glb}", file=sys.stderr)
        sys.exit(1)

    output_glb = Path(args.output) if args.output else input_glb.parent / f"{input_glb.stem}_anim.glb"
    kf_file    = Path(args.keyframes) if args.keyframes else None

    pipeline(
        description=args.description,
        input_glb=input_glb,
        output_glb=output_glb,
        fps=args.fps,
        frames=args.frames,
        skip_llama=args.skip_llama,
        keyframe_file=kf_file,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
