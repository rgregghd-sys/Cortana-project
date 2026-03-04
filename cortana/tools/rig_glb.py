"""
rig_glb.py — Headless Blender script that adds a skeleton to a static GLB.

Invocation (by rig.py):
  blender --background --factory-startup --python rig_glb.py \\
          -- --input cortana.glb --output cortana_rigged.glb

What it does:
  1. Clears scene and imports the GLB
  2. Computes the mesh bounding box in Blender world space
  3. Determines the dominant "up" axis automatically
  4. Creates an armature with 10 bones:
       Root, Spine, Spine1, Spine2, Neck, Head,
       LeftShoulder, RightShoulder, LeftArm, RightArm
  5. Parents all mesh objects to the armature with automatic weight painting
  6. Exports the rigged GLB (with skeleton + skin weights)

The resulting GLB can be animated by blender_animate.py using named bones.
"""
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# NumPy compat shim — Blender 3.4 uses deprecated np.bool removed in NumPy 1.24
# Must run BEFORE any bpy import so the patched numpy is already in sys.modules.
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
# Arg parsing — everything after '--' belongs to this script
# ---------------------------------------------------------------------------
def _parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Input static GLB path")
    p.add_argument("--output", required=True, help="Output rigged GLB path")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Bone layout — built dynamically from the mesh bounding box
# ---------------------------------------------------------------------------
def _build_bone_defs(lo, hi):
    """
    Return bone definitions relative to the mesh bounding box.

    lo / hi: Vector3 tuples (x,y,z) min/max in Blender world space.

    We detect the dominant "height" axis (largest span) and orient the
    spine chain along it.  The armature covers a head+shoulders bust layout.

    Returns:
        dict  name -> (head_xyz, tail_xyz, parent_name_or_None)
    """
    spans = [hi[i] - lo[i] for i in range(3)]
    up_axis = spans.index(max(spans))          # 0=X, 1=Y, 2=Z
    side_axes = [i for i in range(3) if i != up_axis]
    # Pick the wider of the two remaining axes as the left-right (shoulder) axis
    wide_axis  = side_axes[0] if spans[side_axes[0]] >= spans[side_axes[1]] else side_axes[1]
    depth_axis = side_axes[1] if wide_axis == side_axes[0] else side_axes[0]

    # Mesh center along each axis (spine is centered inside the mesh)
    ctr = tuple((lo[i] + hi[i]) / 2.0 for i in range(3))

    def pt(up_frac, side_offset=0.0):
        """Build a 3-element position along the detected axes.

        up_frac    : 0.0=bottom of mesh, 1.0=top
        side_offset: offset along wide axis (+ve = right, -ve = left)
        """
        result = [0.0, 0.0, 0.0]
        result[up_axis]    = lo[up_axis] + spans[up_axis] * up_frac
        result[wide_axis]  = ctr[wide_axis] + side_offset   # centered + lateral offset
        result[depth_axis] = ctr[depth_axis]                 # centered depth
        return tuple(result)

    half_w = spans[wide_axis] * 0.5      # half-width of model (for shoulder X)

    # Spine chain (6 bones from base to crown)
    root_h   = pt(0.00)
    spine_h  = pt(0.10);  spine_t  = pt(0.28)
    spine1_t = pt(0.44)
    spine2_t = pt(0.55)
    neck_b   = pt(0.60);  neck_t   = pt(0.73)
    head_t   = pt(1.00)

    # Shoulders (branch from Spine2)
    s_base  = pt(0.55, 0.0)
    ls_tip  = pt(0.55, -half_w * 0.70)   # left → negative wide axis
    rs_tip  = pt(0.55,  half_w * 0.70)
    la_tip  = pt(0.30, -half_w * 0.90)
    ra_tip  = pt(0.30,  half_w * 0.90)

    return {
        "Root":          (root_h,   spine_h,   None),
        "Spine":         (spine_h,  spine_t,   "Root"),
        "Spine1":        (spine_t,  spine1_t,  "Spine"),
        "Spine2":        (spine1_t, spine2_t,  "Spine1"),
        "Neck":          (neck_b,   neck_t,    "Spine2"),
        "Head":          (neck_t,   head_t,    "Neck"),
        "LeftShoulder":  (s_base,   ls_tip,    "Spine2"),
        "RightShoulder": (s_base,   rs_tip,    "Spine2"),
        "LeftArm":       (ls_tip,   la_tip,    "LeftShoulder"),
        "RightArm":      (rs_tip,   ra_tip,    "RightShoulder"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import bpy
    from mathutils import Vector

    args  = _parse_args()
    src   = Path(args.input)
    dst   = Path(args.output)

    print(f"[rig] Input  : {src}")
    print(f"[rig] Output : {dst}")

    # ---- Clear scene ----
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # ---- Import ----
    suffix = src.suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(src))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(src))
    else:
        print(f"[rig] ERROR: unsupported format '{suffix}'", file=sys.stderr)
        sys.exit(1)
    print(f"[rig] Imported: {src.name}")

    # ---- Gather mesh objects ----
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print("[rig] ERROR: no mesh objects after import", file=sys.stderr)
        sys.exit(1)
    print(f"[rig] Meshes  : {[m.name for m in meshes]}")

    # ---- Compute world-space bounding box ----
    xs, ys, zs = [], [], []
    for obj in meshes:
        for v in obj.data.vertices:
            co = obj.matrix_world @ v.co
            xs.append(co.x); ys.append(co.y); zs.append(co.z)
    lo = (min(xs), min(ys), min(zs))
    hi = (max(xs), max(ys), max(zs))
    spans = tuple(hi[i] - lo[i] for i in range(3))
    print(f"[rig] Bounds  : X=[{lo[0]:.3f},{hi[0]:.3f}] Y=[{lo[1]:.3f},{hi[1]:.3f}] Z=[{lo[2]:.3f},{hi[2]:.3f}]")
    print(f"[rig] Spans   : X={spans[0]:.3f}  Y={spans[1]:.3f}  Z={spans[2]:.3f}")
    up_label = ["X", "Y", "Z"][spans.index(max(spans))]
    print(f"[rig] Up axis : {up_label} (dominant span)")

    # ---- Create armature at world origin ----
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    arm_obj = bpy.context.active_object
    arm_obj.name = "CortanaArmature"

    bpy.ops.object.mode_set(mode='EDIT')
    arm = arm_obj.data

    # Remove the default bone that comes with armature_add
    for b in list(arm.edit_bones):
        arm.edit_bones.remove(b)

    # Build and insert bones
    bone_defs = _build_bone_defs(lo, hi)
    created = {}
    for bname, (head, tail, parent_name) in bone_defs.items():
        eb = arm.edit_bones.new(bname)
        eb.head = Vector(head)
        eb.tail = Vector(tail)
        created[bname] = eb
        print(f"[rig]   {bname:20s}  head={tuple(round(v,3) for v in head)}")

    # Wire up parent relationships (no connect — keeps exact head/tail positions)
    for bname, (_, _, parent_name) in bone_defs.items():
        if parent_name:
            created[bname].parent      = created[parent_name]
            created[bname].use_connect = False

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[rig] Armature built: {len(bone_defs)} bones")

    # ---- Parent mesh objects to armature with automatic weight painting ----
    # Select order matters: meshes first, armature last (becomes active).
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    print(f"[rig] Auto-weighted {len(meshes)} mesh(es)")

    # Verify vertex groups were created
    for m in meshes:
        vg_names = [vg.name for vg in m.vertex_groups]
        print(f"[rig]   {m.name}: {len(vg_names)} vertex groups → {vg_names[:5]}{'…' if len(vg_names) > 5 else ''}")

    # ---- Export rigged GLB ----
    dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath            = str(dst),
        export_format       = "GLB",
        export_animations   = True,
        export_nla_strips   = False,
        export_def_bones    = False,
        export_apply        = False,
    )
    print(f"[rig] Exported: {dst}")


if __name__ == "__main__":
    main()
