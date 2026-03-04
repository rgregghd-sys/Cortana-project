"""
blender_animate.py — Headless Blender script invoked by animate.py.

Invocation (by animate.py):
  blender --background --factory-startup --python blender_animate.py \\
          -- --input cortana.glb --output cortana_anim.glb --json keyframes.json

What it does:
  1. Clears the default Blender scene
  2. Imports the GLB (or FBX from Mixamo)
  3. Reads keyframe JSON produced by Llama (or built-in demo)
  4. If the model has an armature (rigged via Mixamo), applies bone keyframes
  5. If no armature (static mesh), applies object-level transform keyframes
  6. Exports the result as an animated GLB

Keyframe JSON schema (produced by animate.py / Llama):
  {
    "name":   "nod",
    "fps":    30,
    "frames": 60,
    "bones": {
      "Head": [
        {"frame": 0,  "rx": 0,  "ry": 0, "rz": 0},
        {"frame": 30, "rx": 20, "ry": 0, "rz": 0},
        {"frame": 60, "rx": 0,  "ry": 0, "rz": 0}
      ]
    },
    "root": [
      {"frame": 0,  "px": 0, "py": 0, "pz": 0},
      {"frame": 60, "px": 0, "py": 0, "pz": 0}
    ]
  }
"""
import sys
import json
import math
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# NumPy compatibility shim — Blender 3.4 uses deprecated np.bool/np.int/etc.
# These were removed in NumPy 1.24. Patch them back before Blender imports anything.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    for _attr, _builtin in [
        ("bool", bool), ("int", int), ("float", float),
        ("complex", complex), ("object", object), ("str", str),
    ]:
        if not hasattr(np, _attr):
            setattr(np, _attr, _builtin)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Parse args passed after '--'
# ---------------------------------------------------------------------------
def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--json",   required=True, dest="json_path")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import bpy

    args = _parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output)
    json_path   = Path(args.json_path)

    print(f"[animate] Input  : {input_path}")
    print(f"[animate] Output : {output_path}")
    print(f"[animate] JSON   : {json_path}")

    # ---- Load keyframes ----
    with open(json_path) as f:
        kf = json.load(f)

    clip_name   = kf.get("name", "clip")
    fps         = kf.get("fps", 30)
    total_frames = kf.get("frames", 60)
    bones_kf    = kf.get("bones", {})
    root_kf     = kf.get("root", [])

    # ---- Clear scene ----
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end   = total_frames

    # ---- Import model ----
    suffix = input_path.suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(input_path))
        print(f"[animate] Imported GLB: {input_path.name}")
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(input_path))
        print(f"[animate] Imported FBX: {input_path.name}")
    else:
        print(f"[animate] ERROR: unsupported format: {suffix}", file=sys.stderr)
        sys.exit(1)

    # ---- Find armature and root object ----
    armature_obj = None
    root_obj     = None

    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            armature_obj = obj
        if obj.parent is None and obj.type in ("ARMATURE", "MESH", "EMPTY"):
            root_obj = obj

    # ---- Apply keyframes ----
    if armature_obj and bones_kf:
        _animate_armature(armature_obj, bones_kf, root_kf, clip_name, fps, total_frames)
    else:
        # No armature — animate root object transforms instead (procedural fallback)
        target = armature_obj or root_obj or list(bpy.context.scene.objects)[0]
        print(f"[animate] No armature found. Animating object: {target.name}")
        _animate_object(target, bones_kf, root_kf, fps, total_frames)

    # ---- Export animated GLB ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath        = str(output_path),
        export_format   = "GLB",
        export_animations = True,
        export_nla_strips = True,
        export_def_bones  = False,
        export_apply      = False,
    )
    print(f"[animate] Exported: {output_path}")


# ---------------------------------------------------------------------------
# Armature animation
# ---------------------------------------------------------------------------
def _animate_armature(arm_obj, bones_kf, root_kf, clip_name, fps, total_frames):
    import bpy

    # Select armature and enter pose mode
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    # Create or reuse action
    action = bpy.data.actions.new(name=clip_name)
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = action

    available_bones = {b.name for b in arm_obj.pose.bones}
    applied = []

    for bone_name, keyframes in bones_kf.items():
        # Fuzzy match — case insensitive
        matched = _match_bone(bone_name, available_bones)
        if not matched:
            print(f"[animate] WARN: bone '{bone_name}' not found in armature — skipping")
            continue

        pbone = arm_obj.pose.bones[matched]
        pbone.rotation_mode = "XYZ"

        for kf_entry in keyframes:
            frame = kf_entry["frame"]
            rx = math.radians(kf_entry.get("rx", 0))
            ry = math.radians(kf_entry.get("ry", 0))
            rz = math.radians(kf_entry.get("rz", 0))

            pbone.rotation_euler = (rx, ry, rz)
            pbone.keyframe_insert(data_path="rotation_euler", frame=frame)

        applied.append(matched)

    # Root translation keyframes
    if root_kf:
        arm_obj.rotation_mode = "XYZ"
        for kf_entry in root_kf:
            frame = kf_entry["frame"]
            arm_obj.location = (
                kf_entry.get("px", 0),
                kf_entry.get("py", 0),
                kf_entry.get("pz", 0),
            )
            arm_obj.keyframe_insert(data_path="location", frame=frame)

    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"[animate] Armature animated. Bones used: {applied}")


def _match_bone(name: str, available: set) -> str | None:
    """Case-insensitive prefix/exact match."""
    name_l = name.lower()
    for b in available:
        if b.lower() == name_l:
            return b
    for b in available:
        if b.lower().startswith(name_l) or name_l.startswith(b.lower()):
            return b
    return None


# ---------------------------------------------------------------------------
# Object-level animation (no armature / static mesh fallback)
# ---------------------------------------------------------------------------
def _animate_object(obj, bones_kf, root_kf, fps, total_frames):
    """
    For static GLBs with no skeleton, map bone keyframes to object rotation
    and root keyframes to object translation.
    """
    import bpy
    import math

    obj.rotation_mode = "XYZ"
    obj.animation_data_create()

    # Aggregate all rotation keyframes by frame number
    # (blend contributions from multiple "bone" entries)
    frame_rot = {}
    for bone_name, keyframes in bones_kf.items():
        for kf_entry in keyframes:
            frame = kf_entry["frame"]
            rx = math.radians(kf_entry.get("rx", 0))
            ry = math.radians(kf_entry.get("ry", 0))
            rz = math.radians(kf_entry.get("rz", 0))
            if frame not in frame_rot:
                frame_rot[frame] = [rx, ry, rz]
            else:
                frame_rot[frame][0] += rx
                frame_rot[frame][1] += ry
                frame_rot[frame][2] += rz

    for frame, (rx, ry, rz) in sorted(frame_rot.items()):
        obj.rotation_euler = (rx, ry, rz)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Root position
    for kf_entry in root_kf:
        frame = kf_entry["frame"]
        obj.location = (
            kf_entry.get("px", 0),
            kf_entry.get("py", 0),
            kf_entry.get("pz", 0),
        )
        obj.keyframe_insert(data_path="location", frame=frame)

    print(f"[animate] Object '{obj.name}' animated with {len(frame_rot)} keyframes")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
