"""
build_character.py — Blender headless script.

Builds a full-body stylized humanoid GLB from JSON design parameters.
Cortana calls this autonomously to design her own 3D appearance.

Usage (via model_designer.py):
  blender --background --factory-startup --python cortana/tools/build_character.py \
          -- --params '{"proportions":{...},"material":{...}}' \
             --output cortana/static/cortana_self.glb
"""
import sys
import math
import json
import argparse
from pathlib import Path

# NumPy compat shim — must run BEFORE any bpy import (Blender 3.4 / NumPy 1.24)
try:
    import numpy as np
    for _a, _b in [("bool", bool), ("int", int), ("float", float),
                   ("complex", complex), ("object", object), ("str", str)]:
        if not hasattr(np, _a):
            setattr(np, _a, _b)
except ImportError:
    pass


def _parse():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--params", default="{}", help="JSON design parameters")
    p.add_argument("--output", required=True, help="Output GLB path")
    return p.parse_args(argv)


DEFAULT_PARAMS = {
    "proportions": {
        "head_radius": 0.145,
        "head_oval_y": 1.22,
        "shoulder_width": 0.195,
        "chest_radius": 0.185,
        "chest_depth": 0.130,
        "waist_radius": 0.148,
        "waist_depth": 0.110,
        "hip_radius": 0.200,
        "hip_depth": 0.150,
        "arm_angle": 28,
        "upper_arm_radius": 0.058,
        "forearm_radius": 0.046,
        "leg_spread": 0.105,
        "thigh_radius": 0.082,
        "calf_radius": 0.062,
    },
    "material": {
        "base_color": [0.02, 0.55, 0.92],
        "emission_color": [0.0, 0.45, 0.90],
        "emission_strength": 2.0,
        "opacity": 0.85,
        "roughness": 0.15,
        "metalness": 0.40,
    },
    "style": "holographic",
    "hair": "short_crop",
}


def _merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def main():
    import bpy
    from mathutils import Vector, Euler

    args  = _parse()
    out   = Path(args.output)

    try:
        user_params = json.loads(args.params)
    except json.JSONDecodeError:
        user_params = {}

    params = _merge(DEFAULT_PARAMS, user_params)
    P = params["proportions"]
    M = params["material"]
    style = params.get("style", "holographic")
    hair  = params.get("hair", "short_crop")

    print(f"[build] style={style}  hair={hair}")
    print(f"[build] base_color={M['base_color']}  emission={M['emission_strength']}")

    # ── Clear scene ──────────────────────────────────────────────────────────
    bpy.ops.wm.read_factory_settings(use_empty=True)
    parts = []

    def add_sphere(loc, sx, sy, sz, segs=16):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=segs, ring_count=segs // 2, location=loc)
        o = bpy.context.active_object
        o.scale = (sx, sy, sz)
        parts.append(o)
        return o

    def add_cyl(loc, rx, ry, rz, ry_rot=0.0, segs=12):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segs, radius=1.0, depth=2.0, location=loc)
        o = bpy.context.active_object
        o.scale = (rx, ry, rz)
        if ry_rot:
            o.rotation_euler = Euler((0, ry_rot, 0), 'XYZ')
        parts.append(o)
        return o

    # Extract proportions
    sw   = P["shoulder_width"]
    cr   = P["chest_radius"];   cd = P["chest_depth"]
    wr   = P["waist_radius"];   wd = P["waist_depth"]
    hr   = P["hip_radius"];     hd = P["hip_depth"]
    uar  = P["upper_arm_radius"]
    far_ = P["forearm_radius"]
    ls   = P["leg_spread"]
    tr   = P["thigh_radius"]
    calf = P["calf_radius"]

    # A-pose geometry: shoulder, elbow, wrist, hand positions
    elbow_x  = sw + 0.11
    wrist_x  = elbow_x + 0.05
    hand_x   = wrist_x + 0.01
    wrist_z  = 1.155 - 0.18
    hand_z   = wrist_z - 0.09

    a_up = math.atan2(elbow_x - sw, 1.36 - 1.155)  # upper arm angle from vertical
    a_lo = math.atan2(wrist_x - elbow_x, 1.155 - wrist_z)

    arm_ctr_x  = (sw + elbow_x) / 2
    arm_ctr_z  = (1.36 + 1.155) / 2
    arm_half   = math.hypot(elbow_x - sw, 1.36 - 1.155) / 2

    fa_ctr_x   = (elbow_x + wrist_x) / 2
    fa_ctr_z   = (1.155 + wrist_z) / 2
    fa_half    = math.hypot(wrist_x - elbow_x, 1.155 - wrist_z) / 2

    head_rx = P["head_radius"]
    head_ry = head_rx * 0.90
    head_rz = head_rx * P["head_oval_y"]

    # ── HEAD ─────────────────────────────────────────────────────────────────
    add_sphere((0, 0, 1.62), head_rx, head_ry, head_rz)

    # ── HAIR ─────────────────────────────────────────────────────────────────
    if hair == "short_crop":
        add_sphere((0,     0.03,  1.800), head_rx * 0.90, head_rx * 0.78, 0.048)
        add_sphere((-0.08, 0.010, 1.775), 0.065, 0.042, 0.036)
        add_sphere(( 0.08, 0.010, 1.775), 0.065, 0.042, 0.036)
    elif hair == "long":
        add_sphere((0, -0.06, 1.50), head_rx * 0.78, 0.040, head_rx * 0.70)
        add_sphere((0, -0.08, 1.30), head_rx * 0.68, 0.035, head_rx * 0.58)
        add_sphere((0, -0.09, 1.12), head_rx * 0.55, 0.030, head_rx * 0.45)
    elif hair == "bun":
        add_sphere((0, -0.01, 1.840), 0.090, 0.078, 0.075)
    elif hair == "ponytail":
        add_sphere((0, -0.05, 1.77), 0.055, 0.040, 0.055)
        add_sphere((0, -0.10, 1.60), 0.040, 0.032, 0.110)
    # "none" → no hair

    # ── NECK ─────────────────────────────────────────────────────────────────
    add_cyl((0, 0, 1.435), 0.058, 0.058, 0.095)

    # ── COLLAR / SHOULDER SLAB ───────────────────────────────────────────────
    add_cyl((0, 0, 1.365), sw, sw * 0.76, 0.040)

    # ── CHEST ────────────────────────────────────────────────────────────────
    add_cyl((0, 0, 1.230), cr, cd, 0.135)

    # ── WAIST ────────────────────────────────────────────────────────────────
    add_cyl((0, 0, 1.025), wr, wd, 0.100)

    # ── HIPS ─────────────────────────────────────────────────────────────────
    add_cyl((0, 0, 0.890), hr, hd, 0.108)

    # ── THIGHS ───────────────────────────────────────────────────────────────
    add_cyl((-ls, 0, 0.685), tr,   tr,   0.215)
    add_cyl(( ls, 0, 0.685), tr,   tr,   0.215)

    # ── CALVES ───────────────────────────────────────────────────────────────
    add_cyl((-ls, 0, 0.330), calf, calf, 0.180)
    add_cyl(( ls, 0, 0.330), calf, calf, 0.180)

    # ── FEET ─────────────────────────────────────────────────────────────────
    add_sphere((-ls,  0.045, 0.072), calf, calf * 2.0, calf * 0.95)
    add_sphere(( ls,  0.045, 0.072), calf, calf * 2.0, calf * 0.95)

    # ── UPPER ARMS (A-pose) ───────────────────────────────────────────────────
    add_cyl((-arm_ctr_x, 0, arm_ctr_z), uar, uar, arm_half, ry_rot= a_up)
    add_cyl(( arm_ctr_x, 0, arm_ctr_z), uar, uar, arm_half, ry_rot=-a_up)

    # ── FOREARMS ─────────────────────────────────────────────────────────────
    add_cyl((-fa_ctr_x, 0, fa_ctr_z),  far_, far_, fa_half, ry_rot= a_lo)
    add_cyl(( fa_ctr_x, 0, fa_ctr_z),  far_, far_, fa_half, ry_rot=-a_lo)

    # ── HANDS ────────────────────────────────────────────────────────────────
    add_sphere((-hand_x, 0, hand_z), 0.050, 0.038, 0.065)
    add_sphere(( hand_x, 0, hand_z), 0.050, 0.038, 0.065)

    # ── JOIN ALL PARTS ────────────────────────────────────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    body = bpy.context.active_object
    body.name = "CortanaBody"
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    bpy.ops.object.shade_smooth()

    # ── SUBDIVISION (level 1 keeps poly count web-friendly) ──────────────────
    sub = body.modifiers.new("Subdiv", 'SUBSURF')
    sub.levels        = 1
    sub.render_levels = 1

    # ── MATERIAL ─────────────────────────────────────────────────────────────
    mat = bpy.data.materials.new("CortanaSkin")
    mat.use_nodes      = True
    mat.blend_method   = 'BLEND'
    mat.shadow_method  = 'NONE'
    nt = mat.node_tree
    nt.nodes.clear()

    bc  = list(M["base_color"])[:3]
    ec  = list(M["emission_color"])[:3]

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value        = (*bc, 1.0)
    bsdf.inputs["Roughness"].default_value         = float(M["roughness"])
    bsdf.inputs["Metallic"].default_value          = float(M["metalness"])
    bsdf.inputs["Emission"].default_value          = (*ec, 1.0)
    bsdf.inputs["Emission Strength"].default_value = float(M["emission_strength"])
    bsdf.inputs["Alpha"].default_value             = float(M["opacity"])
    bsdf.inputs["Specular"].default_value          = 0.8

    out_mat = nt.nodes.new("ShaderNodeOutputMaterial")
    out_mat.location = (320, 0)

    if style == "wireframe":
        wire = nt.nodes.new("ShaderNodeWireframe")
        wire.use_pixel_size = True
        wire.inputs["Size"].default_value = 0.6
        mix   = nt.nodes.new("ShaderNodeMixShader")
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        nt.links.new(wire.outputs["Fac"],    mix.inputs["Fac"])
        nt.links.new(bsdf.outputs["BSDF"],   mix.inputs[1])
        nt.links.new(transp.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"],  out_mat.inputs["Surface"])
    else:
        nt.links.new(bsdf.outputs["BSDF"], out_mat.inputs["Surface"])

    body.data.materials.append(mat)

    # ── ARMATURE — 19 bones, A-pose ───────────────────────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    arm_obj = bpy.context.active_object
    arm_obj.name = "CortanaArmature"

    bpy.ops.object.mode_set(mode='EDIT')
    arm = arm_obj.data
    for eb in list(arm.edit_bones):
        arm.edit_bones.remove(eb)

    def bone(name, head, tail, parent=None):
        eb = arm.edit_bones.new(name)
        eb.head = Vector(head)
        eb.tail = Vector(tail)
        if parent:
            eb.parent = arm.edit_bones[parent]
            eb.use_connect = False
        return eb

    bone("Hips",      (0,   0, 0.89),        (0,   0, 1.03))
    bone("Spine",     (0,   0, 1.03),        (0,   0, 1.19),   "Hips")
    bone("Spine1",    (0,   0, 1.19),        (0,   0, 1.36),   "Spine")
    bone("Neck",      (0,   0, 1.36),        (0,   0, 1.50),   "Spine1")
    bone("Head",      (0,   0, 1.50),        (0,   0, 1.80),   "Neck")

    bone("LUpLeg",    (-ls, 0, 0.89),        (-ls, 0, 0.50),   "Hips")
    bone("LLeg",      (-ls, 0, 0.50),        (-ls, 0, 0.13),   "LUpLeg")
    bone("LFoot",     (-ls, 0, 0.13),        (-ls, 0.09, 0.04),"LLeg")
    bone("RUpLeg",    ( ls, 0, 0.89),        ( ls, 0, 0.50),   "Hips")
    bone("RLeg",      ( ls, 0, 0.50),        ( ls, 0, 0.13),   "RUpLeg")
    bone("RFoot",     ( ls, 0, 0.13),        ( ls, 0.09, 0.04),"RLeg")

    bone("LShoulder", (0,   0, 1.36),        (-sw,       0, 1.36),    "Spine1")
    bone("LArm",      (-sw,        0, 1.36), (-elbow_x,  0, 1.155),  "LShoulder")
    bone("LForeArm",  (-elbow_x,   0, 1.155),(-wrist_x,  0, wrist_z),"LArm")
    bone("LHand",     (-wrist_x,   0, wrist_z),(-hand_x, 0, hand_z), "LForeArm")

    bone("RShoulder", (0,   0, 1.36),        ( sw,       0, 1.36),    "Spine1")
    bone("RArm",      ( sw,        0, 1.36), ( elbow_x,  0, 1.155),  "RShoulder")
    bone("RForeArm",  ( elbow_x,   0, 1.155),( wrist_x,  0, wrist_z),"RArm")
    bone("RHand",     ( wrist_x,   0, wrist_z),( hand_x, 0, hand_z), "RForeArm")

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[build] Armature: {len(arm.bones)} bones")

    # ── PARENT MESH TO ARMATURE (auto-weight) ────────────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    print("[build] Auto-weighted mesh to armature")

    # ── EXPORT ───────────────────────────────────────────────────────────────
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath          = str(out),
        export_format     = "GLB",
        export_animations = True,
        export_apply      = True,
        export_nla_strips = False,
        export_def_bones  = False,
    )
    size_kb = out.stat().st_size // 1024
    print(f"[build] Exported: {out}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
