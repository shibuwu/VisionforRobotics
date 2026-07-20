import bpy
import mathutils
import json
import math
import os
import sys
import numpy as np

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="scene8")
parser.add_argument("--frame", type=int, default=-1)
parser.add_argument("--output_dir", type=str, default="renders")
parser.add_argument("--detection_dir", type=str, default="output")
parser.add_argument("--max_frames", type=int, default=-1)
parser.add_argument("--skip", type=int, default=0)
parser.add_argument("--render_step", type=int, default=1,
                    help="Render every Nth frame entry (1=all, 5=every 5th)")
parser.add_argument("--all_frames", action="store_true",
                    help="Render every frame in detections.json (ignore cadence)")
parser.add_argument("--hsv_brake", action="store_true",
                    help="Use legacy HSV brake detector files (brake_lights.json). "
                         "Default is Detic (brake_lights_detic.json).")
parser.add_argument("--brake_variant", type=str, default="",
                    help="Append a variant suffix to the Detic brake light file "
                         "name, e.g. 'v1_baseline' -> brake_lights_detic_v1_baseline.json. "
                         "Also splits renders into renders_<variant>/ so outputs "
                         "don't clobber each other.")
args = parser.parse_args(argv)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_DIR, "P3Data", "Assets")
# brake variant → separate output dir
_render_subdir = args.output_dir
if _render_subdir == "renders":
    if args.hsv_brake:
        _render_subdir = "renders_hsv"
    elif args.brake_variant:
        _render_subdir = f"renders_{args.brake_variant}"
OUTPUT_DIR = os.path.join(PROJECT_DIR, _render_subdir)
_DET_BASE = os.path.join(PROJECT_DIR, args.detection_dir, args.scene)
DETECTION_FILE = os.path.join(_DET_BASE, "detections.json")
# lane files (fallback paths if not in detection dir)
LANE_FILE = os.path.join(_DET_BASE, "lane_detections.json")
if not os.path.exists(LANE_FILE):
    LANE_FILE = os.path.join(PROJECT_DIR, "output_lane", args.scene, "detections.json")
LANE_ARROW_FILE = os.path.join(_DET_BASE, "lane_arrow_detections.json")
if not os.path.exists(LANE_ARROW_FILE):
    LANE_ARROW_FILE = os.path.join(PROJECT_DIR, "output_lane_arrow",
                                    f"detections_{args.scene}_lanes.json")
if args.hsv_brake:
    _BRAKE_SUFFIX = ""
    _IND_SUFFIX = ""
else:
    _BRAKE_SUFFIX = f"_detic_{args.brake_variant}" if args.brake_variant else "_detic"
BRAKE_LIGHTS_FILE = os.path.join(_DET_BASE, f"brake_lights{_BRAKE_SUFFIX}.json")
DEPTH_DIR = os.path.join(_DET_BASE, "depth")
PARKED_MOVING_FILE = os.path.join(_DET_BASE, "detections_parked_moving.json")
CALIB_FILE = os.path.join(PROJECT_DIR, "P3Data", "Calib", "calibration_results.json")
EXTRINSICS_FILE = os.path.join(PROJECT_DIR, "P3Data", "Calib", "extrinsics.json")
BASE_SCENE = os.path.join(PROJECT_DIR, "base_scene.blend")

IMG_W, IMG_H = 1280, 960

CONF_THRESHOLDS = {
    "vehicle":          0.60,
    "pedestrian":       0.55,
    "stop_sign":        0.60,
    "traffic_light":    0.50,
    "traffic_pole":     0.50,
    "traffic_cone":     0.45,
    "traffic_cylinder": 0.45,
    "dustbin":          0.50,
    "speed_sign":       0.50,
    "road_arrow":       0.0,
}
CONF_DEFAULT = 0.6

KNOWN_HEIGHTS = {  # meters
    "pedestrian": 1.7,
    "vehicle": 1.5,
    "traffic_light": 0.9,
    "stop_sign": 0.75,
    "traffic_cone": 0.7,
    "dustbin": 1.0,
    "traffic_pole": 3.5,
    "traffic_cylinder": 0.7,
    "speed_sign": 0.6,
}

VEHICLE_HEIGHTS = {
    "sedan":      1.5,
    "hatchback":  1.5,
    "suv":        1.8,
    "pickup":     1.9,
    "truck":      3.5,
    "motorcycle": 1.4,
    "bicycle":    1.4,
}

KNOWN_WIDTHS = {
    "dustbin": 0.45,
    "traffic_pole": 0.25,
}

MOUNT_HEIGHTS = {
    "traffic_light": 3.5,
    "stop_sign": 0.0,
    "pedestrian": 0.0,
    "vehicle": 0.0,
    "traffic_cone": 0.0,
    "dustbin": 0.0,
    "traffic_pole": 0.0,
    "traffic_cylinder": 0.0,
    "speed_sign": 0.0,    # create_speed_sign builds the pole from base z
    "road_arrow": 0.0,
}

ASSET_INFO = {
    "vehicle": {
        "file": os.path.join("Vehicles", "SedanAndHatchback.blend"),
        "object": "Car",
        "scale": 0.02,
        "rotation": (0, 0, 0),  # car faces +Y (away from ego)
    },
    "pedestrian": {
        "file": "HumanBasemesh.blend",
        "object": "Male_Basemesh_01",
        "armature": "Male_Basemesh_Rig_01",
        "extra_objects": ["MaleBasemesh_Eyes_01"],
        "scale": 1.0,              # already 1.7m tall
        "z_offset": 0,
        "rotation": (0, 0, 0),
    },
    "traffic_light": {
        "file": "TrafficSignal.blend",
        "object": "Traffic_signal1",
        "scale": 1.0,
        "rotation": (math.radians(90), 0, math.radians(180)),  # upright, facing ego (-Y)
    },
    "stop_sign": {
        "file": "StopSign.blend",
        "object": "StopSign_Geo",
        "scale": 0.373,  # 6.71m raw → 2.5m
        "rotation": (math.radians(90), 0, math.radians(180)),  # upright, sign face toward ego (-Y)
    },
    "traffic_cone": {
        "file": "TrafficConeAndCylinder.blend",
        "object": "absperrhut",
        "scale": 1.28,  # raw 0.545 units tall → 0.7m
        "rotation": (math.radians(90), 0, 0),  # Y is up in source, rotate to Z
    },
    "dustbin": {
        "file": "Dustbin.blend",
        "object": "Bin_Mesh.072",
        "scale": 1.68,    # raw 0.595 units tall → 1.0m
        "rotation": (math.radians(90), 0, 0),  # Y is up in source, rotate to Z
    },
    "traffic_pole": {
        "file": None,  # use primitive cylinder
        "object": None,
        "scale": 1.0,
        "rotation": (0, 0, 0),
    },
    "traffic_cylinder": {
        "file": "TrafficConeAndCylinder.blend",
        "object": "absperrhut",
        "scale": 2.20,  # raw 0.545 → 1.2m (delineator post)
        "rotation": (math.radians(90), 0, 0),  # Y is up in source, rotate to Z
    },
}

VEHICLE_ASSETS = {
    # yaw_offset: 0 if model faces -Y at rot_z=0 (sedan), pi if faces +Y (truck)
    "sedan":      {"file": os.path.join("Vehicles", "SedanAndHatchback.blend"), "object": "Car",            "scale": 0.0195, "yaw_offset": math.pi, "z_offset": 0},
    "hatchback":  {"file": os.path.join("Vehicles", "SedanAndHatchback.blend"), "object": "Car",            "scale": 0.0175, "yaw_offset": math.pi, "z_offset": 0},
    "suv":        {"file": os.path.join("Vehicles", "SUV.blend"),               "object": "Jeep_3_",        "scale": 3.57,   "yaw_offset": 0, "z_offset": 0.02},
    "pickup":     {"file": os.path.join("Vehicles", "PickupTruck.blend"),       "object": "PickupTruck",    "scale": 0.581,  "yaw_offset": math.pi, "z_offset": 0.85, "base_rot": (math.pi/2, 0, math.pi/2)},
    "truck":      {"file": os.path.join("Vehicles", "Truck.blend"),             "object": "Truck",          "scale": 0.000895, "yaw_offset": 0, "z_offset": 0.01},
    "motorcycle": {"file": os.path.join("Vehicles", "Motorcycle.blend"),        "object": "B_Wheel",        "scale": 0.006,   "yaw_offset": 0, "z_offset": 0.0, "base_rot": (math.pi/2, 0, math.pi/2)},
    "bicycle":    {"file": os.path.join("Vehicles", "Bicycle.blend"),           "object": "roadbike 2.0.1", "scale": 0.141,  "yaw_offset": -math.pi/2, "z_offset": 0.29, "base_rot": (math.pi/2, 0, math.pi/2)},
}

# vehicle_color name → RGB for Blender material
VEHICLE_COLOR_RGB = {
    "white":  (0.95, 0.95, 0.95, 1.0),
    "black":  (0.05, 0.05, 0.05, 1.0),
    "gray":   (0.45, 0.45, 0.48, 1.0),
    "red":    (0.70, 0.10, 0.10, 1.0),
    "blue":   (0.15, 0.25, 0.65, 1.0),
    "green":  (0.15, 0.45, 0.20, 1.0),
    "yellow": (0.85, 0.75, 0.15, 1.0),
    "orange": (0.85, 0.45, 0.10, 1.0),
    "brown":  (0.40, 0.25, 0.15, 1.0),
    "golden": (0.75, 0.60, 0.30, 1.0),
    "offwhite": (0.85, 0.83, 0.78, 1.0),
}




def make_brake_material(strength=50.0):
    """Pure emissive red — strong enough to bloom onto the car body."""
    mat = bpy.data.materials.new("BrakeMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    em = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Color"].default_value = (1.0, 0.04, 0.04, 1.0)
    em.inputs["Strength"].default_value = strength
    mat.node_tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_brake_halo_material(strength=12.0):
    """Larger semi-transparent backing plane to fake a glow halo around the brake light."""
    mat = bpy.data.materials.new("BrakeHaloMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    em = nodes.new("ShaderNodeEmission")
    transp = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    out = nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Color"].default_value = (1.0, 0.15, 0.15, 1.0)
    em.inputs["Strength"].default_value = strength
    mix.inputs["Fac"].default_value = 0.65
    mat.node_tree.links.new(transp.outputs["BSDF"], mix.inputs[1])
    mat.node_tree.links.new(em.outputs["Emission"], mix.inputs[2])
    mat.node_tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    mat.blend_method = "BLEND"
    return mat


BRAKE_STRENGTH = 80.0
BRAKE_HALO_STRENGTH = 22.0


def _rear_mount_info(vehicle_obj):
    """Compute the geometry needed to place rear-facing light planes on a
    vehicle: rear plane position, up/height, lateral direction, width, yaw.
    Returns None if the vehicle is at the origin (nothing to place against)."""
    bpy.context.view_layer.update()

    mw = vehicle_obj.matrix_world
    world_corners = [mw @ mathutils.Vector(c) for c in vehicle_obj.bound_box]

    zs = [c.z for c in world_corners]
    z_bot, z_top = min(zs), max(zs)
    actual_h = z_top - z_bot

    cx = sum(c.x for c in world_corners) / 8.0
    cy = sum(c.y for c in world_corners) / 8.0
    dist = math.sqrt(cx ** 2 + cy ** 2)
    if dist < 1e-3:
        return None

    tc_x, tc_y = -cx / dist, -cy / dist

    cam_projs = [c.x * tc_x + c.y * tc_y for c in world_corners]
    rear_depth = max(cam_projs) - 0.005  # 5mm inside surface = flush
    rear_disp = rear_depth - (cx * tc_x + cy * tc_y)
    rx = cx + rear_disp * tc_x
    ry = cy + rear_disp * tc_y

    lat_x, lat_y = -tc_y, tc_x
    lat_projs = [c.x * lat_x + c.y * lat_y for c in world_corners]
    veh_w = max(1.0, min(max(lat_projs) - min(lat_projs), 2.4))

    face_yaw = math.atan2(tc_x, tc_y)
    return dict(rx=rx, ry=ry, z_bot=z_bot, actual_h=actual_h,
                tc_x=tc_x, tc_y=tc_y, lat_x=lat_x, lat_y=lat_y,
                veh_w=veh_w, face_yaw=face_yaw)


def _make_rear_plane(location, scale, rot_euler, material, parent_obj):
    """Spawn a shadow-disabled emissive plane parented to a vehicle."""
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
    plane = bpy.context.active_object
    plane.scale = scale
    plane.rotation_euler = rot_euler
    plane.data.materials.append(material)
    plane.visible_shadow = False
    plane.parent = parent_obj
    plane.matrix_parent_inverse = parent_obj.matrix_world.inverted()
    return plane


def add_brake_lights(vehicle_obj, det):
    """Attach two emissive red planes to the upper-outer corners of the
    vehicle's rear face. Planes are parented to the vehicle so they follow it.
    Binary on/off — if the detector flagged this detection, draw at full
    strength."""
    if vehicle_obj is None or vehicle_obj.type != "MESH":
        return
    info = _rear_mount_info(vehicle_obj)
    if info is None:
        return

    light_z = info["z_bot"] + info["actual_h"] * 0.62
    light_lat = (info["veh_w"] / 2.0) * 0.78
    lw = max(0.06, min(0.14, info["veh_w"] * 0.08))
    lh = max(0.04, min(0.09, info["actual_h"] * 0.06))

    brake_mat = make_brake_material(strength=BRAKE_STRENGTH)
    halo_mat = make_brake_halo_material(strength=BRAKE_HALO_STRENGTH)

    for sign in (+1, -1):
        lx = info["rx"] + sign * light_lat * info["lat_x"]
        ly = info["ry"] + sign * light_lat * info["lat_y"]

        _make_rear_plane((lx, ly, light_z),
                         (lw, lh, 0.001),
                         (math.pi / 2, 0.0, info["face_yaw"]),
                         brake_mat, vehicle_obj)

        # soft halo backing plane sitting a hair behind the main light
        _make_rear_plane(
            (lx - 0.003 * info["tc_x"], ly - 0.003 * info["tc_y"], light_z),
            (lw * 2.6, lh * 2.6, 0.001),
            (math.pi / 2, 0.0, info["face_yaw"]),
            halo_mat, vehicle_obj)


def setup_bloom_compositor():
    """Build a compositor node group with a Glare(Fog Glow) so the emissive
    brake lights bloom in the final image. Blender 5.x replaced the legacy
    CompositorNodeComposite with NodeGroupOutput inside a CompositorNodeTree
    that the scene references via scene.compositing_node_group."""
    scene = bpy.context.scene

    # already wired?
    ng = getattr(scene, "compositing_node_group", None)
    if ng is not None and any(n.name == "BrakeBloom" for n in ng.nodes):
        return

    ng = bpy.data.node_groups.new("BrakeCompositor", "CompositorNodeTree")
    # node group needs an Image output socket (the "final image" of the comp)
    if hasattr(ng, "interface"):
        ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    rl = ng.nodes.new("CompositorNodeRLayers")
    glare = ng.nodes.new("CompositorNodeGlare")
    out = ng.nodes.new("NodeGroupOutput")

    glare.name = "BrakeBloom"
    glare.inputs["Type"].default_value = "Fog Glow"
    glare.inputs["Quality"].default_value = "Medium"
    glare.inputs["Threshold"].default_value = 1.0
    glare.inputs["Size"].default_value = 0.7
    glare.inputs["Strength"].default_value = 1.0

    rl.location = (-400, 0)
    glare.location = (0, 0)
    out.location = (300, 0)

    ng.links.new(rl.outputs["Image"], glare.inputs["Image"])
    ng.links.new(glare.outputs["Image"], out.inputs[0])

    scene.compositing_node_group = ng


def load_calibration():
    with open(CALIB_FILE) as f:
        calib = json.load(f)
    with open(EXTRINSICS_FILE) as f:
        extrinsics = json.load(f)

    result = {}
    for cam in ["front", "back", "left", "right"]:
        result[cam] = {
            "K": np.array(calib[cam]["K"]),
            "R": np.array(extrinsics[cam]["R"]),
            "t": np.array(extrinsics[cam]["t"]),
        }
    return result


def compute_depth_scale(detections, cam_calib, depth_map):
    """Compute scale factor to correct depth map using known-size anchors.

    Uses both FCOS3D metric depth and bbox-based depth from known heights,
    combining them to avoid systematic bias from either source alone.

    Returns scale such that: corrected_depth = depth_map_value * scale
    """
    if depth_map is None:
        return None
    dh, dw = depth_map.shape
    fx = cam_calib["K"][0, 0]

    fcos_ratios = []
    bbox_ratios = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bbox_h = y2 - y1
        if bbox_h < 20:
            continue

        # sample depth map at detection center
        cx = min(max(int((x1+x2)/2 * dw / IMG_W), 0), dw-1)
        cy = min(max(int((y1+y2)/2 * dh / IMG_H), 0), dh-1)
        r = 5
        patch = depth_map[max(0,cy-r):cy+r+1, max(0,cx-r):cx+r+1]
        dm_val = float(np.median(patch)) if patch.size > 0 else 0
        if dm_val < 0.5:
            continue

        # anchor 1: FCOS3D metric depth
        fcos_depth = det.get("fcos3d_depth", 0)
        if fcos_depth > 3.0 and fcos_depth < 80.0:
            fcos_ratios.append(fcos_depth / dm_val)

        # anchor 2: bbox height → known object height
        if det["label"] in KNOWN_HEIGHTS and 30 < bbox_h < 400:
            z_bbox = (fx * KNOWN_HEIGHTS[det["label"]]) / bbox_h
            if 3.0 < z_bbox < 80.0:
                bbox_ratios.append(z_bbox / dm_val)

    scales = []
    if fcos_ratios:
        scales.append(float(np.median(fcos_ratios)))
    if bbox_ratios:
        scales.append(float(np.median(bbox_ratios)))

    if scales:
        scale = float(np.mean(scales))
        print(f"  Depth scale: {scale:.3f} (fcos={len(fcos_ratios)} bbox={len(bbox_ratios)} anchors)")
        return scale
    return None


def bbox_to_3d(bbox, label, cam_calib, blender_cam_pos, camera="front",
               blender_fov=None, depth_map=None, depth_scale=None,
               det=None):
    """Convert detection to 3D Blender position.

    Priority:
    1. FCOS3D 3D position (direct metric, no projection needed)
    2. Scaled depth map + back-projection
    3. Bbox height fallback
    """
    x1, y1, x2, y2 = bbox
    cx_px = (x1 + x2) / 2
    bbox_h = y2 - y1
    bbox_w = x2 - x1

    if bbox_h < 5:
        return None

    real_fx = cam_calib["K"][0, 0]
    real_fy = cam_calib["K"][1, 1]
    real_cx = cam_calib["K"][0, 2]
    real_cy = cam_calib["K"][1, 2]

    GROUND_OBJECTS = {"dustbin", "traffic_cone", "traffic_cylinder"}
    if label in GROUND_OBJECTS:
        CAM_HEIGHT = 1.3
        v_below = y2 - real_cy
        if v_below > 5:  # pixel must be below horizon
            Z = real_fy * CAM_HEIGHT / v_below
            if 1.0 < Z < 80.0:
                x_cam = (cx_px - real_cx) / real_fx * Z
                ego_y = 4.0
                return (float(x_cam), float(ego_y + Z),
                        float(MOUNT_HEIGHTS.get(label, 0.0)))
        # if pixel is at/above horizon, fall through to bbox fallback
    known_h = KNOWN_HEIGHTS.get(label, 1.5)
    if label == "vehicle" and det is not None:
        vt = det.get("vehicle_type")
        if vt in VEHICLE_HEIGHTS:
            known_h = VEHICLE_HEIGHTS[vt]
    known_w = KNOWN_WIDTHS.get(label)

    # for objects whose bbox often bleeds vertically (dustbin), drop ones with
    # extreme aspect ratio — they're misdetections of pole/sign clusters
    if label in KNOWN_WIDTHS and bbox_h / max(bbox_w, 1) > 8:
        return None

    # use FCOS3D position if available
    fcos_pos = det.get("fcos3d_pos") if det else None
    if fcos_pos and fcos_pos[2] > 1.0 and fcos_pos[2] < 80.0 and label == "vehicle":
        ego_y = 4.0
        fcos_depth = fcos_pos[2]
        x_world = fcos_pos[0]
        y_world = ego_y + fcos_depth
        z_world = MOUNT_HEIGHTS.get(label, 0.0)
    else:
        # fallback: back-project from 2D bbox
        if blender_fov:
            blender_fx = (IMG_W / 2) / math.tan(blender_fov / 2)
        else:
            blender_fx = real_fx

        # bbox-based Z: prefer width for objects with known width (more reliable
        # when bbox bleeds vertically into above-object clutter)
        if known_w is not None and bbox_w > 5:
            z_bbox = (real_fx * known_w) / bbox_w
        else:
            z_bbox = (real_fx * known_h) / bbox_h if bbox_h > 10 else None

        z_depth = None
        if depth_map is not None and depth_scale is not None:
            dh, dw = depth_map.shape
            # sample near bbox bottom for width-based objects
            sample_y = y2 - bbox_w / 2 if known_w is not None else (y1 + y2) / 2
            cx = min(max(int(cx_px * dw / IMG_W), 0), dw-1)
            cy = min(max(int(sample_y * dh / IMG_H), 0), dh-1)
            r = 5
            patch = depth_map[max(0,cy-r):cy+r+1, max(0,cx-r):cx+r+1]
            raw_d = float(np.median(patch)) if patch.size > 0 else 0
            if raw_d >= 0.5:
                z_depth = raw_d * depth_scale

        # prefer bbox-based Z when depth map disagrees
        if label in KNOWN_HEIGHTS and z_bbox is not None and 1.0 < z_bbox < 80.0:
            if z_depth is None or z_depth > 80.0 or z_depth < 1.0 \
               or z_depth > z_bbox * 2 or z_depth < z_bbox * 0.5:
                Z = z_bbox
            else:
                Z = z_depth
        elif z_depth is not None:
            Z = z_depth
        elif z_bbox is not None:
            Z = z_bbox
        else:
            return None

        if Z < 1.0 or Z > 80.0:
            return None

        # back-project using REAL camera intrinsics for consistent X.
        x_cam = (cx_px - real_cx) / real_fx * Z
        ego_y = 4.0
        x_world = x_cam
        y_world = ego_y + Z
        z_world = MOUNT_HEIGHTS.get(label, 0.0)

    # 3m boundary around ego car at origin — push out if inside
    boundary = 5.0
    dist = math.sqrt(x_world**2 + y_world**2)
    if dist < boundary and dist > 0:
        # push outward along the same direction from ego
        scale = boundary / dist
        x_world *= scale
        y_world *= scale

    return (float(x_world), float(y_world), float(z_world))


def make_material(name, color, roughness=0.5, emission=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    if emission > 0:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        mat.blend_method = 'BLEND'
    return mat


def load_asset_object(label, vehicle_type=None):
    """Load asset from .blend file. For rigged pedestrians, loads both mesh and armature."""
    # pick asset info
    if label == "vehicle" and vehicle_type and vehicle_type in VEHICLE_ASSETS:
        info = VEHICLE_ASSETS[vehicle_type]
    else:
        info = ASSET_INFO.get(label)
    if not info or info.get("file") is None:
        return None

    filepath = os.path.join(ASSETS_DIR, info["file"])
    if not os.path.exists(filepath):
        return None

    obj_name = info["object"]
    armature_name = info.get("armature")

    # load mesh + armature + extras if rigged
    names_to_load = [obj_name]
    if armature_name:
        names_to_load.append(armature_name)
    for extra in info.get("extra_objects", []):
        names_to_load.append(extra)

    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        to_load = []
        for name in names_to_load:
            if name in data_from.objects:
                to_load.append(name)
        if not to_load:
            # fallback to first object
            if data_from.objects:
                to_load = [data_from.objects[0]]
            else:
                return None
        data_to.objects = to_load

    mesh_obj = None
    arm_obj = None
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            if obj.type == 'ARMATURE':
                arm_obj = obj
            else:
                mesh_obj = obj

    # apply any baked rotation so our rotation_euler starts from (0,0,0)
    result_obj = None
    if arm_obj and mesh_obj:
        mesh_obj.parent = arm_obj
        for mod in mesh_obj.modifiers:
            if mod.type == 'ARMATURE':
                mod.object = arm_obj
        result_obj = arm_obj
    else:
        result_obj = mesh_obj or arm_obj

    pass  # rotation handled in place_object via base_rot_x

    return result_obj


def create_tl_arrow_mesh(x, y, z, arrow_dir, color_rgba, dx_face, dy_face):
    """Replace the active TL bulb with a colored disc + dark arrow inside it."""
    disc_r = 0.25
    s = disc_r * 0.85
    head_pts = [(-0.45*s, 0.3*s), (0, 0.85*s), (0.45*s, 0.3*s)]
    tail_pts = [(-0.2*s, 0.3*s), (-0.2*s, -0.55*s), (0.2*s, -0.55*s), (0.2*s, 0.3*s)]
    if arrow_dir == "left":
        rot_angle = math.radians(90)
    elif arrow_dir == "right":
        rot_angle = math.radians(-90)
    else:
        rot_angle = 0
    cos_a, sin_a = math.cos(rot_angle), math.sin(rot_angle)
    def rot2d(px, pz):
        return (px*cos_a - pz*sin_a, px*sin_a + pz*cos_a)
    head_pts = [rot2d(*p) for p in head_pts]
    tail_pts = [rot2d(*p) for p in tail_pts]

    perp_x, perp_y = -dy_face, dx_face
    off = 0.05
    verts = []
    for local_perp, local_z in head_pts + tail_pts:
        verts.append((x + dx_face*off + perp_x*local_perp,
                      y + dy_face*off + perp_y*local_perp,
                      z + local_z))
    faces = [(0,1,2), (3,4,5,6)]
    mesh = bpy.data.meshes.new("TLArrowMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("TL_Arrow", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_material("TL_Arrow_Mat", (0.02, 0.02, 0.02, 1.0)))

    # colored disc behind the arrow
    bpy.ops.mesh.primitive_circle_add(radius=disc_r, fill_type='NGON', location=(x, y, z))
    disc = bpy.context.active_object
    disc.name = "TL_Arrow_disc"
    face_yaw = math.atan2(-dx_face, -dy_face)
    disc.rotation_euler = (math.radians(90), 0, face_yaw + math.pi)
    disc.data.materials.append(make_material("TL_Arrow_disc_mat", color_rgba, emission=20.0))


def create_traffic_light(x, y, z, color_name, arrow="none"):
    """Create a simple traffic light: dark box with 3 colored spheres.
    Spheres face toward ego car (origin). If arrow != 'none', the active bulb
    is replaced with a disc + arrow indicator."""
    colors_map = {
        "red": (1, 0, 0, 1),
        "yellow": (1, 0.9, 0, 1),
        "green": (0, 1, 0, 1),
    }

    # direction from TL toward ego (for offsetting spheres to face ego)
    dist = math.sqrt(x**2 + y**2)
    if dist > 0:
        dx = -x / dist * 0.5
        dy = -y / dist * 0.5
    else:
        dx, dy = 0, -0.5
    dx_unit = dx / 0.5 if dist > 0 else 0
    dy_unit = dy / 0.5 if dist > 0 else -1

    # dark box body — oversized for visibility
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z + 1.2))
    body = bpy.context.active_object
    body.scale = (0.6, 0.6, 1.0)
    body.name = "TL_body"
    body_mat = make_material("TL_body", (0.15, 0.15, 0.15, 1.0), roughness=0.8)
    body.data.materials.append(body_mat)

    # three light spheres offset toward ego: red=top, yellow=mid, green=bottom
    offsets = {"red": 1.8, "yellow": 1.2, "green": 0.6}
    dark_color = (0.05, 0.05, 0.05, 1.0)

    for name, rgba in colors_map.items():
        z_off = offsets[name]
        is_active = (name == color_name)
        if is_active and arrow != "none":
            create_tl_arrow_mesh(x + dx, y + dy, z + z_off, arrow, rgba, dx_unit, dy_unit)
            continue
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.25,
            location=(x + dx, y + dy, z + z_off)
        )
        sphere = bpy.context.active_object
        sphere.name = f"TL_{name}_bulb"
        if is_active:
            mat = make_material(f"TL_{name}_on", rgba, emission=20.0)
        else:
            mat = make_material(f"TL_{name}_off", dark_color, roughness=0.9)
        sphere.data.materials.append(mat)

    return body


def create_speed_sign(x, y, z, speed):
    """Pole + white face + 'SPEED LIMIT' text + numeric speed, facing ego."""
    dist = max(math.sqrt(x**2 + y**2), 0.01)
    # face toward ego
    face_yaw = math.atan2(-x, y)
    dx_ego = -x / dist
    dy_ego = -y / dist

    bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=2.5, location=(x, y, z + 1.25))
    pole = bpy.context.active_object
    pole.name = "SpeedSign_pole"
    pole.data.materials.append(make_material("SSPole", (0.4, 0.4, 0.4, 1.0), roughness=0.7))

    # flip plane to face ego
    plane_yaw = face_yaw + math.pi

    bpy.ops.mesh.primitive_plane_add(size=1,
        location=(x - dx_ego * 0.03, y - dy_ego * 0.03, z + 2.3))
    border = bpy.context.active_object
    border.scale = (0.50, 0.60, 1)
    border.rotation_euler = (math.radians(90), 0, plane_yaw)
    border.name = "SpeedSign_border"
    border.data.materials.append(make_material("SSBlack", (0.05, 0.05, 0.05, 1.0)))

    bpy.ops.mesh.primitive_plane_add(size=1,
        location=(x + dx_ego * 0.01, y + dy_ego * 0.01, z + 2.3))
    face = bpy.context.active_object
    face.scale = (0.45, 0.55, 1)
    face.rotation_euler = (math.radians(90), 0, plane_yaw)
    face.name = "SpeedSign_face"
    face.data.materials.append(make_material("SSWhite", (0.95, 0.95, 0.95, 1.0), roughness=0.3))

    tx = x + dx_ego * 0.06
    ty = y + dy_ego * 0.06

    bpy.ops.object.text_add(location=(tx, ty, z + 2.55))
    title = bpy.context.active_object
    title.data.body = "SPEED\nLIMIT"
    title.data.size = 0.18
    title.data.align_x = 'CENTER'
    title.data.align_y = 'CENTER'
    title.rotation_euler = (math.radians(90), 0, face_yaw)
    title.name = "SpeedSign_title"
    title.data.materials.append(make_material("SSTitle", (0.05, 0.05, 0.05, 1.0)))

    bpy.ops.object.text_add(location=(tx, ty, z + 2.25))
    num = bpy.context.active_object
    num.data.body = str(speed)
    num.data.size = 0.5
    num.data.align_x = 'CENTER'
    num.data.align_y = 'CENTER'
    num.rotation_euler = (math.radians(90), 0, face_yaw)
    num.name = "SpeedSign_number"
    num.data.materials.append(make_material("SSNum", (0.05, 0.05, 0.05, 1.0)))

    return pole


_dir_arrow_counter = [0]


def create_direction_arrow(x, y, z, move_dir):
    """Create flat chevron arrows projected ahead of vehicle.
    Green = same direction, Red = oncoming, Blue = lateral.
    """
    dir_map = {
        'oncoming': (0, -1),
        'ahead':    (0, 1),
        'left':     (-1, 0),
        'right':    (1, 0),
    }
    color_map = {
        'ahead':    (0.1, 0.9, 0.2, 1.0),
        'oncoming': (0.9, 0.1, 0.1, 1.0),
        'left':     (0.1, 0.5, 1.0, 1.0),
        'right':    (0.1, 0.5, 1.0, 1.0),
    }
    bx, by = dir_map.get(move_dir, (0, 1))
    color = color_map.get(move_dir, (0.1, 0.5, 1.0, 1.0))
    az = 0.10

    _dir_arrow_counter[0] += 1
    mat = make_material(f"DirArrowMat_{_dir_arrow_counter[0]}", color, emission=5.0)
    px, py = -by, bx

    if move_dir in ('ahead', 'oncoming'):
        start_x = x
        start_y = y - 3.0
    else:
        start_x = x - bx * 3.0
        start_y = y - by * 3.0

    for i in range(3):
        if move_dir in ('ahead', 'oncoming'):
            cx = start_x + bx * (i * 1.5)
            cy = start_y - (i * 1.5)
        else:
            cx = start_x + bx * (i * 1.5)
            cy = start_y + by * (i * 1.5)
        tip = (cx + bx * 0.9, cy + by * 0.9, az)
        left = (cx + px * 0.6, cy + py * 0.6, az)
        right = (cx - px * 0.6, cy - py * 0.6, az)
        verts = [tip, left, right]
        mesh = bpy.data.meshes.new(f"Chevron_{i}")
        mesh.from_pydata(verts, [], [(0, 1, 2)])
        mesh.update()
        obj = bpy.data.objects.new(f"Chevron_{i}", mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(mat)


def create_ground_arrow(x, y, z, direction):
    """Create a flat arrow mesh on the ground plane."""
    s = 1.2
    head_pts = [(-0.4*s, -0.2*s, 0), (0, 0.5*s, 0), (0.4*s, -0.2*s, 0)]
    tail_pts = [(-0.15*s, -0.2*s, 0), (-0.15*s, -1.0*s, 0),
                (0.15*s, -1.0*s, 0), (0.15*s, -0.2*s, 0)]
    rot = 0
    if direction == "left":
        rot = math.radians(90)
    elif direction == "right":
        rot = math.radians(-90)
    def rot_xy(px, py, pz):
        nx = px * math.cos(rot) - py * math.sin(rot)
        ny = px * math.sin(rot) + py * math.cos(rot)
        return (nx, ny, pz)
    verts = [rot_xy(*p) for p in head_pts + tail_pts]
    verts = [(v[0] + x, v[1] + y, v[2] + z + 0.02) for v in verts]
    faces = [(0, 1, 2), (3, 4, 5, 6)]
    mesh = bpy.data.meshes.new("GroundArrow")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Road_Arrow", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_material("ArrowWhite", (0.9, 0.9, 0.9, 1.0),
                                             roughness=0.9, emission=1.5))
    return obj


def apply_stop_sign_texture(obj):
    """Apply StopSignImage.png texture to stop sign."""
    tex_path = os.path.join(ASSETS_DIR, "StopSignImage.png")
    if not os.path.exists(tex_path):
        return

    mat = bpy.data.materials.new("StopSignMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes["Principled BSDF"]
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(tex_path)
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.5

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def ped_facing_from_keypoints(keypoints, keypoints_3d=None):
    """Estimate pedestrian facing using 3D shoulder depth + 2D stride.
    3D depth disambiguates facing vs away. 2D stride gives walking angle.
    Returns Blender Z rotation."""
    if not keypoints or len(keypoints) < 17:
        return None

    # 3D shoulder line → facing direction
    if keypoints_3d and len(keypoints_3d) >= 17:
        ls3d, rs3d = keypoints_3d[5], keypoints_3d[6]
        dx = rs3d[0] - ls3d[0]
        dy = rs3d[1] - ls3d[1]
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            # perpendicular to shoulder line in XY plane
            face_x = dy
            face_y = -dx
            yaw = math.atan2(face_x, face_y)
            # model faces -Y at rot_z=0
            return yaw + math.pi

    # fallback: ankle stride direction (2D)
    lank, rank = keypoints[15], keypoints[16]
    if lank[2] < 0.4 or rank[2] < 0.4:
        return None
    if lank[1] < rank[1]:
        stride_dx = lank[0] - rank[0]
        stride_dy = lank[1] - rank[1]
    else:
        stride_dx = rank[0] - lank[0]
        stride_dy = rank[1] - lank[1]
    if abs(stride_dx) < 5 and abs(stride_dy) < 5:
        return None
    bx = stride_dx
    by = -stride_dy
    walk_yaw = math.atan2(bx, by)
    return walk_yaw + math.pi


# COCO keypoint pairs → Rigify bone names
COCO_TO_BONE = {
    (5, 7):   "upper_arm_fk.L",  # left shoulder → left elbow
    (7, 9):   "forearm_fk.L",    # left elbow → left wrist
    (6, 8):   "upper_arm_fk.R",  # right shoulder → right elbow
    (8, 10):  "forearm_fk.R",    # right elbow → right wrist
    (11, 13): "thigh_fk.L",      # left hip → left knee
    (13, 15): "shin_fk.L",       # left knee → left ankle
    (12, 14): "thigh_fk.R",      # right hip → right knee
    (14, 16): "shin_fk.R",       # right knee → right ankle
}

KPT_CONF_THRESH = 0.4


def pose_pedestrian(armature_obj, keypoints, keypoints_3d=None):
    """Pose pedestrian using 2D keypoint angles on DEF bones.
    Simple approach: compute limb angle from image keypoints,
    apply as X rotation on deformation bones.
    """
    if not keypoints or len(keypoints) < 17:
        return

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    # switch Rigify to FK mode and set euler rotation
    for pb in armature_obj.pose.bones:
        pb.rotation_mode = 'XYZ'
        for key in pb.keys():
            if 'IK_FK' in key:
                pb[key] = 1.0

    for (kpt_a, kpt_b), bone_name in COCO_TO_BONE.items():
        pa, pb_kpt = keypoints[kpt_a], keypoints[kpt_b]
        if pa[2] < KPT_CONF_THRESH or pb_kpt[2] < KPT_CONF_THRESH:
            continue

        pbone = armature_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue

        # 2D angle from vertical (image: x=right, y=down)
        dx = pb_kpt[0] - pa[0]
        dy = pb_kpt[1] - pa[1]
        img_angle = math.atan2(dx, dy)

        # wide clamp so big walking/raised-arm angles are visible
        img_angle = max(-math.pi/2, min(math.pi/2, img_angle))
        if abs(img_angle) < 0.05:
            continue

        pbone.rotation_euler = (img_angle, 0, 0)

    bpy.ops.object.mode_set(mode='OBJECT')


def pose_pedestrian_3d(armature_obj, keypoints_3d):
    """Pose pedestrian using RTMPose3D 3D keypoints.
    For each FK bone, align its Y axis to the (child - parent) world vector,
    expressed in armature-local space. Anatomically correct for arms+legs.
    Returns True if pose was applied, False if data unusable.
    """
    if not keypoints_3d or len(keypoints_3d) < 17:
        return False

    # reject all-zero or degenerate data
    nonzero = sum(1 for k in keypoints_3d[:17]
                  if abs(k[0]) + abs(k[1]) + abs(k[2]) > 1e-4)
    if nonzero < 8:
        return False

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    # force FK and quaternion mode
    for pb in armature_obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        for key in pb.keys():
            if 'IK_FK' in key:
                pb[key] = 1.0

    # process parents before children so pb.matrix reflects updated parent
    BONE_CHAIN = [
        (5, 7,  "upper_arm_fk.L"),
        (7, 9,  "forearm_fk.L"),
        (6, 8,  "upper_arm_fk.R"),
        (8, 10, "forearm_fk.R"),
        (11, 13, "thigh_fk.L"),
        (13, 15, "shin_fk.L"),
        (12, 14, "thigh_fk.R"),
        (14, 16, "shin_fk.R"),
    ]

    arm_world_3x3 = armature_obj.matrix_world.to_3x3()
    arm_world_inv = arm_world_3x3.inverted()
    applied = 0

    for ka, kb, bone_name in BONE_CHAIN:
        pb = armature_obj.pose.bones.get(bone_name)
        if pb is None:
            continue
        p3 = keypoints_3d[ka]
        c3 = keypoints_3d[kb]
        v_world = mathutils.Vector((c3[0] - p3[0], c3[1] - p3[1], c3[2] - p3[2]))
        if v_world.length < 1e-4:
            continue
        v_world.normalize()

        # transform target direction into armature-local space
        v_local = arm_world_inv @ v_world
        v_local.normalize()

        # current bone direction in armature space (after parent updates)
        bpy.context.view_layer.update()
        cur_dir = (pb.tail - pb.head)
        if cur_dir.length < 1e-4:
            continue
        cur_dir.normalize()

        # rotation that takes current direction to target direction
        q_delta = cur_dir.rotation_difference(v_local)

        # compose with existing pose matrix in armature space
        new_basis_3x3 = q_delta.to_matrix() @ pb.matrix.to_3x3()
        new_mat = mathutils.Matrix.Translation(pb.head) @ new_basis_3x3.to_4x4()
        pb.matrix = new_mat
        applied += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    return applied > 0


def compute_facing_ego_yaw(x, y):
    """Compute yaw angle so object faces toward ego car at origin.
    Returns rotation around Z axis."""
    return math.atan2(-x, -y)


def place_object(label, position, extra_data=None):
    """Place a detected object at a 3D position."""
    x, y_fwd, z = position

    # traffic lights: use simple primitives (no asset loading)
    if label == "traffic_light":
        color_name = extra_data.get("color", "red") if extra_data else "red"
        arrow = extra_data.get("arrow", "none") if extra_data else "none"
        return create_traffic_light(x, y_fwd, z, color_name, arrow=arrow)

    if label == "speed_sign":
        speed = extra_data.get("speed", 25) if extra_data else 25
        return create_speed_sign(x, y_fwd, z, speed)

    if label == "road_arrow":
        direction = extra_data.get("direction", "straight") if extra_data else "straight"
        return create_ground_arrow(x, y_fwd, z, direction)

    vehicle_type = extra_data.get("vehicle_type") if extra_data else None
    obj = load_asset_object(label, vehicle_type=vehicle_type)

    if obj:
        # get scale from vehicle-specific or default asset info
        if label == "vehicle" and vehicle_type and vehicle_type in VEHICLE_ASSETS:
            info = VEHICLE_ASSETS[vehicle_type]
        else:
            info = ASSET_INFO.get(label)
            if not info:
                print(f"    WARNING: no asset info for {label}, skipping")
                return None
        s = info["scale"]
        z_off = info.get("z_offset", 0)  # already in meters
        obj.scale = (s, s, s)
        obj.location = (x, y_fwd, z + z_off)

        if label == "vehicle":
            fcos_yaw = extra_data.get("fcos3d_yaw") if extra_data else None
            asset_info = VEHICLE_ASSETS.get(vehicle_type, {})
            yaw_off = asset_info.get("yaw_offset", 0)
            if fcos_yaw is not None:
                rot_z = -fcos_yaw + yaw_off - math.pi/2
            else:
                # no FCOS3D — use move_dir as orientation source
                move_dir = extra_data.get("move_dir") if extra_data else None
                if move_dir == "oncoming":
                    rot_z = yaw_off - math.pi
                elif move_dir == "left":
                    rot_z = yaw_off - math.pi/2
                elif move_dir == "right":
                    rot_z = yaw_off + math.pi/2
                else:
                    # "ahead" or unknown — face away from ego
                    rot_z = yaw_off
            # compose base rotation (uprighting) with yaw
            base_rot = asset_info.get("base_rot")
            base_rx = asset_info.get("base_rot_x", 0)
            if base_rot:
                base_mat = mathutils.Euler(base_rot).to_matrix().to_4x4()
                yaw_mat = mathutils.Matrix.Rotation(rot_z, 4, 'Z')
                obj.rotation_euler = (yaw_mat @ base_mat).to_euler()
            elif base_rx != 0:
                base_mat = mathutils.Euler((base_rx, 0, 0)).to_matrix().to_4x4()
                yaw_mat = mathutils.Matrix.Rotation(rot_z, 4, 'Z')
                obj.rotation_euler = (yaw_mat @ base_mat).to_euler()
            else:
                obj.rotation_euler = (0, 0, rot_z)
        elif label == "stop_sign":
            # model face points -Y at yaw=0, add pi to face toward ego
            face_yaw = math.atan2(x, y_fwd) + math.pi
            base_rot = info.get("rotation", (0, 0, 0))
            obj.rotation_euler = (base_rot[0], base_rot[1], face_yaw)
        elif label == "pedestrian":
            kpts = extra_data.get("keypoints") if extra_data else None
            kpts_3d = extra_data.get("keypoints_3d") if extra_data else None
            kp_yaw = ped_facing_from_keypoints(kpts, keypoints_3d=kpts_3d)
            face_yaw = kp_yaw if kp_yaw is not None else math.atan2(x, y_fwd)
            base_rot = info.get("rotation", (0, 0, 0))
            obj.rotation_euler = (base_rot[0], base_rot[1], face_yaw)
        else:
            obj.rotation_euler = info.get("rotation", (0, 0, 0))

        # apply materials
        if label == "stop_sign":
            apply_stop_sign_texture(obj)
        elif label == "vehicle":
            is_moving = extra_data.get("moving", True) if extra_data else True
            if is_moving:
                veh_color = extra_data.get("vehicle_color", "gray") if extra_data else "gray"
                color_rgba = VEHICLE_COLOR_RGB.get(veh_color, (0.45, 0.50, 0.58, 1.0))
                car_mat = make_material("VehPaint", color_rgba, roughness=0.35)
                car_mat.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.6
                move_dir = extra_data.get("move_dir", "ahead") if extra_data else "ahead"
                create_direction_arrow(x, y_fwd, z, move_dir)
            else:
                # parked: translucent gray
                car_mat = make_material("VehPaint", (0.55, 0.55, 0.55, 0.6), roughness=0.6, alpha=0.6)
            obj.data.materials.clear()
            obj.data.materials.append(car_mat)

            # brake lights (from Detic detection)
            if extra_data and extra_data.get("brake_light"):
                add_brake_lights(obj, extra_data)
        elif label == "pedestrian":
            ped_mat = make_material("PedMat", (0.65, 0.65, 0.68, 1.0), roughness=0.7)
            # apply material to mesh child if obj is armature
            mesh_target = obj
            if obj.type == 'ARMATURE':
                for child in obj.children:
                    if child.type == 'MESH':
                        mesh_target = child
                        break
            if mesh_target.data and hasattr(mesh_target.data, 'materials'):
                mesh_target.data.materials.clear()
                mesh_target.data.materials.append(ped_mat)

        return obj

    # primitive fallbacks for assets with no .blend (traffic_pole) or that failed to load
    if label == "traffic_light":
        color_name = extra_data.get("color", "red") if extra_data else "red"
        cmap = {"red": (1, 0, 0, 1), "green": (0, 1, 0, 1), "yellow": (1, 0.9, 0, 1)}
        bpy.ops.mesh.primitive_cube_add(size=0.4, location=(x, y_fwd, 5.0))
        obj = bpy.context.active_object
        obj.scale = (0.4, 0.4, 1.2)
        obj.data.materials.append(make_material(f"TL_{color_name}",
                                                 cmap.get(color_name, (1, 1, 1, 1)), emission=8.0))
    elif label == "traffic_pole":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=4.0,
                                             location=(x, y_fwd, 2.0))
        obj = bpy.context.active_object
        obj.data.materials.append(make_material("PoleMat", (0.35, 0.35, 0.35, 1.0)))
    elif label in ("traffic_cone", "traffic_cylinder"):
        bpy.ops.mesh.primitive_cone_add(radius1=0.2, radius2=0.02, depth=0.7,
                                         location=(x, y_fwd, 0.35))
        obj = bpy.context.active_object
        obj.data.materials.append(make_material("ConeMat", (1.0, 0.4, 0.0, 1.0)))
    elif label == "dustbin":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=1.0,
                                             location=(x, y_fwd, 0.5))
        obj = bpy.context.active_object
        obj.data.materials.append(make_material("BinMat", (0.2, 0.35, 0.2, 1.0)))
    return obj


def clear_detections():
    """Remove all objects except the base scene (EgoCar, Road, Camera, Lights)."""
    keep = {"EgoCar", "EgoCamera", "Road", "Shoulder_Left", "Shoulder_Right", "Sun", "FillLight"}
    for obj in list(bpy.data.objects):
        if obj.name not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)


def pixel_to_ground(u, v, K, R, t, camera_height=1.3):
    """Project a 2D image pixel onto the ground plane (z=0)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    ray_world_cam = np.array([ray_cam[0], ray_cam[2], -ray_cam[1]])
    ray_world = R @ ray_world_cam

    cam_pos = t.copy()

    if abs(ray_world[2]) < 1e-6:
        return None
    t_param = -cam_pos[2] / ray_world[2]
    if t_param < 0:
        return None

    point = cam_pos + t_param * ray_world
    dist = math.sqrt(point[0]**2 + point[1]**2)
    if dist > 80.0:
        return None
    return (float(point[0]), float(point[1]))


def render_lanes(lanes, calib):
    """Render lane lines as flat ribbon meshes on the ground plane."""
    if not lanes:
        return

    cam_calib = calib["front"]
    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]

    LANE_COLORS = {
        "white":  (0.90, 0.90, 0.92, 1.0),
        "yellow": (1.00, 0.85, 0.10, 1.0),
    }
    LANE_WIDTH = 0.15

    for lane_idx, lane in enumerate(lanes):
        points_2d = lane.get("points", [])
        color_name = lane.get("color", "white")
        lane_type = lane.get("type", "solid")

        if len(points_2d) < 2:
            continue

        points_3d = []
        for pt in points_2d:
            gnd = pixel_to_ground(float(pt[0]), float(pt[1]), K, R, t)
            if gnd is not None:
                points_3d.append((gnd[0], gnd[1], 0.05))

        if len(points_3d) < 2:
            continue

        # extrapolate toward car and horizon
        if len(points_3d) >= 3:
            points_3d.sort(key=lambda p: p[1])
            p0 = points_3d[0]
            p1 = points_3d[min(3, len(points_3d)-1)]
            dy = p0[1] - p1[1]
            dx = p0[0] - p1[0]
            if abs(dy) > 0.5:
                slope_x = dx / dy
                y_target = max(-3.0, p0[1] - 30.0)  # extend lanes close to ego
                y_cur = p0[1]
                while y_cur > y_target:
                    y_cur -= 0.5
                    x_new = p0[0] + slope_x * (y_cur - p0[1])
                    points_3d.insert(0, (x_new, y_cur, 0.05))

            p_last = points_3d[-1]
            p_prev = points_3d[max(-4, -len(points_3d))]
            dy = p_last[1] - p_prev[1]
            dx = p_last[0] - p_prev[0]
            if abs(dy) > 0.5:
                slope_x = max(-0.3, min(0.3, dx / dy))
                y_cur = p_last[1]
                y_target = min(60.0, p_last[1] + 25.0)
                while y_cur < y_target:
                    y_cur += 0.5
                    x_new = p_last[0] + slope_x * (y_cur - p_last[1])
                    points_3d.append((x_new, y_cur, 0.05))

        # dashed lanes: dash-gap pattern
        if lane_type == "dashed":
            segments = []
            DASH_LEN, GAP_LEN = 3.0, 3.0
            cum_dist = 0.0
            in_dash = True
            seg = [points_3d[0]]
            for i in range(1, len(points_3d)):
                dx = points_3d[i][0] - points_3d[i-1][0]
                dy = points_3d[i][1] - points_3d[i-1][1]
                cum_dist += math.sqrt(dx*dx + dy*dy)
                limit = DASH_LEN if in_dash else GAP_LEN
                if cum_dist >= limit:
                    if in_dash and len(seg) >= 2:
                        segments.append(seg)
                    seg = [points_3d[i]]
                    cum_dist = 0.0
                    in_dash = not in_dash
                elif in_dash:
                    seg.append(points_3d[i])
            if in_dash and len(seg) >= 2:
                segments.append(seg)
        else:
            segments = [points_3d]

        lane_color = LANE_COLORS.get(color_name, LANE_COLORS["white"])
        mat = make_material(f"LaneMat_{color_name}_{lane_idx}", lane_color,
                           roughness=0.8, emission=2.0)

        for seg_idx, seg_pts in enumerate(segments):
            if len(seg_pts) < 2:
                continue
            verts = []
            faces = []
            for i, (x, y, z) in enumerate(seg_pts):
                if i < len(seg_pts) - 1:
                    dx = seg_pts[i+1][0] - x
                    dy = seg_pts[i+1][1] - y
                else:
                    dx = x - seg_pts[i-1][0]
                    dy = y - seg_pts[i-1][1]
                length = math.sqrt(dx*dx + dy*dy)
                if length < 1e-6:
                    perp_x, perp_y = LANE_WIDTH/2, 0
                else:
                    perp_x = -dy / length * LANE_WIDTH / 2
                    perp_y = dx / length * LANE_WIDTH / 2
                verts.append((x + perp_x, y + perp_y, z))
                verts.append((x - perp_x, y - perp_y, z))

            for i in range(len(seg_pts) - 1):
                v0 = 2 * i
                faces.append((v0, v0+1, v0+3, v0+2))

            if not verts or not faces:
                continue
            mesh = bpy.data.meshes.new(f"Lane_{lane_idx}_{seg_idx}")
            mesh.from_pydata(verts, [], faces)
            mesh.update()
            obj = bpy.data.objects.new(f"Lane_{lane_idx}_{seg_idx}", mesh)
            bpy.context.collection.objects.link(obj)
            obj.data.materials.append(mat)


# ---- persistent scene: load once, clean between frames ----
_base_object_names = set()
_scene_initialized = False


def init_scene_once():
    """Load the base .blend, configure EEVEE + compositor once.
    Saves the names of base objects so we can delete only placed objects later."""
    global _base_object_names, _scene_initialized
    if _scene_initialized:
        return
    bpy.ops.wm.open_mainfile(filepath=BASE_SCENE)
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    setup_bloom_compositor()
    _base_object_names = {obj.name for obj in bpy.data.objects}
    _scene_initialized = True


def clean_placed_objects():
    """Remove everything that was added after the base scene was loaded."""
    to_delete = [obj for obj in bpy.data.objects
                 if obj.name not in _base_object_names]
    for obj in to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)
    # purge orphan data to free memory from deleted meshes/materials
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def render_frame(frame_data, scene_name, frame_idx, calib):
    init_scene_once()
    clean_placed_objects()

    # load depth map if available
    depth_map = None
    depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_idx:05d}.npy")
    if os.path.exists(depth_path):
        depth_map = np.load(depth_path)

    # get Blender camera position and FOV for coordinate transform
    cam_obj = bpy.data.objects.get("EgoCamera")
    blender_cam_pos = list(cam_obj.location) if cam_obj else [0, -5, 3.5]
    blender_fov = cam_obj.data.angle if cam_obj else math.radians(71.5)

    # front camera only
    cam_calib = calib["front"]
    front_dets = [d for d in frame_data["detections"]
                  if d.get("camera", "front") == "front"]

    # dedupe by track_id: detections.json has each tracked vehicle twice (a 2D
    # entry with fcos3d_pos=None and a FCOS3D entry with fcos3d_pos set). keep
    # the one carrying fcos3d_pos when available so we don't render two trucks.
    by_track = {}
    untracked = []
    for d in front_dets:
        tid = d.get("track_id")
        if tid is None:
            untracked.append(d)
            continue
        prev = by_track.get(tid)
        if prev is None or (d.get("fcos3d_pos") and not prev.get("fcos3d_pos")):
            by_track[tid] = d
    front_dets = list(by_track.values()) + untracked

    # deduplicate stacked FCOS3D positions
    pos_counts = {}
    for d in front_dets:
        p = d.get("fcos3d_pos")
        if p is None:
            continue
        key = (round(p[0], 2), round(p[1], 2), round(p[2], 2))
        pos_counts[key] = pos_counts.get(key, 0) + 1
    nulled = 0
    for d in front_dets:
        p = d.get("fcos3d_pos")
        if p is None:
            continue
        key = (round(p[0], 2), round(p[1], 2), round(p[2], 2))
        if pos_counts[key] > 1:
            d["fcos3d_pos"] = None
            nulled += 1
    if nulled:
        print(f"  Nulled {nulled} shared fcos3d_pos entries (depth-map fallback)")

    # null fcos3d_pos that disagrees with depth map
    if depth_map is not None:
        dh, dw = depth_map.shape
        # rough depth scale guess from the depths we have so far
        scale_guess = compute_depth_scale(front_dets, cam_calib, depth_map)
        bad = 0
        if scale_guess is not None:
            for d in front_dets:
                p = d.get("fcos3d_pos")
                if p is None:
                    continue
                x1, y1, x2, y2 = d["bbox"]
                cxp = min(max(int((x1 + x2) / 2 * dw / IMG_W), 0), dw - 1)
                cyp = min(max(int((y1 + y2) / 2 * dh / IMG_H), 0), dh - 1)
                r = 5
                patch = depth_map[max(0, cyp - r):cyp + r + 1,
                                   max(0, cxp - r):cxp + r + 1]
                raw_d = float(np.median(patch)) if patch.size > 0 else 0
                if raw_d < 0.5:
                    continue
                z_dm = raw_d * scale_guess
                z_fc = p[2]
                if z_dm > 1.0 and z_dm < 80.0 and (z_dm > 2 * z_fc or z_fc > 2 * z_dm):
                    d["fcos3d_pos"] = None
                    bad += 1
        if bad:
            print(f"  Nulled {bad} fcos3d_pos entries that disagreed with depth map")

    # compute per-frame depth scale using detections as anchors
    depth_scale = compute_depth_scale(front_dets, cam_calib, depth_map)

    # render lanes first so we can filter objects by proximity
    lanes = frame_data.get("lanes", [])
    if lanes:
        render_lanes(lanes, calib)

    # collect lane x positions for filtering (project closest points to ground)
    lane_xs = []
    cam_K = cam_calib["K"]
    cam_R = cam_calib["R"]
    cam_t = cam_calib["t"]
    for lane in lanes:
        for pt in lane.get("points", []):
            gnd = pixel_to_ground(float(pt[0]), float(pt[1]), cam_K, cam_R, cam_t)
            if gnd:
                lane_xs.append(gnd[0])

    placed = {"vehicle": 0, "pedestrian": 0, "other": 0}
    for det in front_dets:
        label = det["label"]

        # traffic poles add visual clutter without value — skip
        if label == "traffic_pole":
            continue

        # skip low confidence detections (per-class thresholds)
        conf_thr = CONF_THRESHOLDS.get(label, CONF_DEFAULT)
        if det.get("confidence", 0) < conf_thr:
            continue

        # skip cyclists — they're rendered as part of the motorcycle/bicycle
        if det.get("is_cyclist"):
            continue

        # vehicles: only render if FCOS3D matched
        if label == "vehicle":
            if det.get("heading_source") not in ("fcos3d", "fcos3d_track"):
                continue

        # skip edge-clipped detections (except ground objects)
        bbox = det["bbox"]
        edge_ok_labels = {"traffic_cone", "traffic_cylinder", "dustbin"}
        if label not in edge_ok_labels:
            if bbox[0] < 5 or bbox[2] > IMG_W - 5:
                continue

        # road arrows carry a pre-computed world_pos in ego frame, no projection
        if label == "road_arrow" and det.get("world_pos") is not None:
            wp = det["world_pos"]
            pos = (float(wp[0]), float(wp[1]), float(wp[2]))
        else:
            pos = bbox_to_3d(det["bbox"], label, cam_calib,
                             blender_cam_pos, "front", blender_fov,
                             depth_map, depth_scale, det=det)
        if pos is None:
            continue

        # skip vehicles too far from any lane (>8m lateral)
        if lane_xs and label == "vehicle":
            min_lane_dist = min(abs(pos[0] - lx) for lx in lane_xs)
            if min_lane_dist > 8.0:
                continue

        if label == "vehicle":
            print(f"  PLACE {det.get('vehicle_type','')} {det.get('vehicle_color','')} at ({pos[0]:.1f},{pos[1]:.1f}) depth={det.get('fcos3d_depth',0):.0f}m heading={math.degrees(det.get('heading',0)):.0f}d")
            placed["vehicle"] += 1
        else:
            print(f"  PLACE {label} at ({pos[0]:.1f},{pos[1]:.1f})")
            placed["other" if label not in placed else label] += 1

        place_object(label, pos, extra_data=det)

    counts = ", ".join(f"{v} {k}s" for k, v in placed.items() if v > 0)
    print(f"Placed: {counts}")

    # render
    out_dir = os.path.join(OUTPUT_DIR, scene_name)
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"render_{frame_idx:05d}.png")
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

    print(f"Rendered frame {frame_idx} -> {filepath}")


def main():
    calib = load_calibration()

    with open(DETECTION_FILE) as f:
        all_frames = json.load(f)
    # smooth vehicle_type/color per track (majority vote)
    from collections import Counter
    tracks = {}
    for entry in all_frames:
        for det in entry.get("detections", []):
            if det.get("label") != "vehicle":
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            tracks.setdefault(tid, []).append(det)
    type_smoothed = color_smoothed = 0
    for tid, dets in tracks.items():
        if len(dets) < 2:
            continue
        type_maj = Counter(d.get("vehicle_type", "sedan") for d in dets).most_common(1)[0][0]
        color_maj = Counter(d.get("vehicle_color", "gray") for d in dets).most_common(1)[0][0]
        for d in dets:
            if d.get("vehicle_type") != type_maj:
                d["vehicle_type"] = type_maj
                type_smoothed += 1
            if d.get("vehicle_color") != color_maj:
                d["vehicle_color"] = color_maj
                color_smoothed += 1
    print(f"Track-smoothed vehicle types: {type_smoothed} type, {color_smoothed} color flips")

    # merge brake light detections from separate file
    if os.path.exists(BRAKE_LIGHTS_FILE):
        with open(BRAKE_LIGHTS_FILE) as f:
            brake_data = json.load(f)
        n_lit = 0
        conf_sum = 0.0
        for entry in all_frames:
            fkey = str(entry["frame_idx"])
            lit_tracks = brake_data.get(fkey, {})
            if not lit_tracks:
                continue
            for det in entry.get("detections", []):
                if det.get("label") != "vehicle":
                    continue
                tid = det.get("track_id")
                if tid is None:
                    continue
                v = lit_tracks.get(str(tid))
                if v is not None and v is not False:
                    det["brake_light"] = True
                    det["brake_confidence"] = float(v) if isinstance(v, (int, float)) else 1.0
                    n_lit += 1
                    conf_sum += det["brake_confidence"]
        mean_conf = conf_sum / n_lit if n_lit else 0.0
        print(f"Brake lights merged from {os.path.basename(BRAKE_LIGHTS_FILE)}: "
              f"{n_lit} lit detections (mean conf {mean_conf:.2f})")
    else:
        print(f"No brake lights file found — run detect_brake_lights_detic.py first")

    # Phase 3: merge parked/moving labels from separate detection file
    if os.path.exists(PARKED_MOVING_FILE):
        with open(PARKED_MOVING_FILE) as f:
            pm_frames = json.load(f)
        # build lookup: frame_idx → list of pm detections (match by bbox IoU)
        pm_by_frame = {e["frame_idx"]: e.get("detections", []) for e in pm_frames}

        def _iou(a, b):
            x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
            x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
            return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0

        n_merged = 0
        n_arrows = 0
        n_speed = 0
        for entry in all_frames:
            pm_dets = pm_by_frame.get(entry["frame_idx"], [])
            if not pm_dets:
                continue
            # merge moving/move_dir into vehicles by bbox IoU
            for det in entry.get("detections", []):
                if det.get("label") != "vehicle":
                    continue
                best_iou, best_pm = 0, None
                for pd in pm_dets:
                    if pd.get("label") != "vehicle":
                        continue
                    iou = _iou(det["bbox"], pd["bbox"])
                    if iou > best_iou:
                        best_iou, best_pm = iou, pd
                if best_iou > 0.5 and best_pm is not None:
                    det["moving"] = best_pm.get("moving", True)
                    if "move_dir" in best_pm:
                        det["move_dir"] = best_pm["move_dir"]
                    n_merged += 1
            # add road_arrow and speed_sign detections directly
            for pd in pm_dets:
                if pd.get("label") == "road_arrow":
                    entry.setdefault("detections", []).append(pd)
                    n_arrows += 1
                elif pd.get("label") == "speed_sign":
                    entry.setdefault("detections", []).append(pd)
                    n_speed += 1
        print(f"Parked/moving merged from {os.path.basename(PARKED_MOVING_FILE)}: "
              f"{n_merged} vehicles, {n_arrows} road_arrows, {n_speed} speed_signs")
    else:
        print(f"No detections_parked_moving.json — all vehicles default to moving")

    # merge lane/arrow data from fallback files if not already in detections.json
    det_by_frame = {e["frame_idx"]: e for e in all_frames}

    main_has_lanes = sum(1 for e in all_frames if e.get("lanes"))
    main_has_arrows = sum(1 for e in all_frames
                          for d in e.get("detections", [])
                          if d.get("label") == "road_arrow")
    main_has_speed = sum(1 for e in all_frames
                         for d in e.get("detections", [])
                         if d.get("label") == "speed_sign")
    main_has_tl_arrow = sum(1 for e in all_frames
                            for d in e.get("detections", [])
                            if d.get("label") == "traffic_light"
                            and d.get("arrow", "none") != "none")
    print(f"detections.json carries: lanes={main_has_lanes} "
          f"road_arrow={main_has_arrows} speed_sign={main_has_speed} "
          f"TL_arrow={main_has_tl_arrow}")

    def bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0

    def patch_tl_from(det_entry, src_dets, force=False):
        """For each TL in det_entry, copy color/arrow from the best-IoU TL in
        src_dets. With force=True overwrite existing values; otherwise only
        fill missing/'unknown'/'none' values."""
        n = 0
        src_tls = [d for d in src_dets if d.get("label") == "traffic_light"]
        if not src_tls:
            return 0
        for det in det_entry["detections"]:
            if det.get("label") != "traffic_light":
                continue
            best, best_iou = None, 0
            for s in src_tls:
                iou = bbox_iou(det["bbox"], s["bbox"])
                if iou > best_iou:
                    best, best_iou = s, iou
            if best is None or best_iou < 0.3:
                continue
            sc = best.get("color")
            if sc and sc != "unknown":
                cur_color = det.get("color")
                if force or not cur_color or cur_color == "unknown":
                    det["color"] = sc
                    n += 1
            sa = best.get("arrow")
            if sa and sa != "none":
                cur_arrow = det.get("arrow", "none")
                if force or cur_arrow == "none":
                    det["arrow"] = sa
                    n += 1
        return n

    def apply_lane_source(src_data, label, take_arrows_and_speed=True):
        """Overlay data from src_data onto det_by_frame.
        Always takes lanes + TL color/arrow patches. Only takes road_arrow and
        speed_sign detections when take_arrows_and_speed=True (primary source)."""
        src_by_frame = {e["frame_idx"]: e for e in src_data}
        lanes_set = arrows_added = speeds_added = tl_patched = 0
        for fidx, det_entry in det_by_frame.items():
            se = src_by_frame.get(fidx)
            if se is None:
                continue
            if se.get("lanes"):
                det_entry["lanes"] = se["lanes"]
                lanes_set += 1
            src_dets = se.get("detections", [])
            if take_arrows_and_speed:
                arrows = [d for d in src_dets if d.get("label") == "road_arrow"]
                if arrows:
                    det_entry["detections"] = [d for d in det_entry.get("detections", [])
                                               if d.get("label") != "road_arrow"]
                    det_entry["detections"].extend(arrows)
                    arrows_added += len(arrows)
                speeds = [d for d in src_dets if d.get("label") == "speed_sign"]
                if speeds:
                    det_entry["detections"] = [d for d in det_entry.get("detections", [])
                                               if d.get("label") != "speed_sign"]
                    det_entry["detections"].extend(speeds)
                    speeds_added += len(speeds)
            tl_patched += patch_tl_from(det_entry, src_dets, force=True)
        if lanes_set or arrows_added or speeds_added or tl_patched:
            print(f"  [{label}] lanes={lanes_set} road_arrow=+{arrows_added} "
                  f"speed_sign=+{speeds_added} TL_fields={tl_patched}")

    # apply lane file (secondary), then arrow file (primary, overwrites)
    if os.path.exists(LANE_FILE):
        with open(LANE_FILE) as f:
            apply_lane_source(json.load(f), "output_lane", take_arrows_and_speed=False)
    else:
        print(f"  no {os.path.basename(LANE_FILE)} (output_lane)")

    if os.path.exists(LANE_ARROW_FILE):
        with open(LANE_ARROW_FILE) as f:
            apply_lane_source(json.load(f), "output_lane_arrow", take_arrows_and_speed=True)
    else:
        print(f"  no {os.path.basename(LANE_ARROW_FILE)} (output_lane_arrow)")

    # propagate TL arrow/color across tracks
    from collections import Counter
    tl_arrow_by_track = {}
    tl_color_by_track = {}
    for entry in all_frames:
        for det in entry.get("detections", []):
            if det.get("label") != "traffic_light":
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            a = det.get("arrow")
            if a and a != "none":
                tl_arrow_by_track.setdefault(tid, Counter())[a] += 1
            c = det.get("color")
            if c and c != "unknown":
                tl_color_by_track.setdefault(tid, Counter())[c] += 1
    # majority vote per track, then apply
    tl_arrow_maj = {tid: cnt.most_common(1)[0][0] for tid, cnt in tl_arrow_by_track.items()}
    n_propagated = 0
    for entry in all_frames:
        for det in entry.get("detections", []):
            if det.get("label") != "traffic_light":
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            if tid in tl_arrow_maj and det.get("arrow", "none") == "none":
                det["arrow"] = tl_arrow_maj[tid]
                n_propagated += 1
    if n_propagated:
        print(f"TL arrows propagated across tracks: {n_propagated} "
              f"({len(tl_arrow_maj)} tracks with arrows)")

    # temporal smoothing: remove short-lived detections, fill 1-frame gaps
    KEEP_UNTRACKED = {"road_arrow", "speed_sign", "traffic_light",
                      "dustbin", "traffic_cone", "traffic_cylinder", "traffic_pole"}

    # build frame index for tracked gap-filling
    track_frames = {}  # tid → [(frame_idx, det), ...]
    for entry in all_frames:
        new_dets = []
        for det in entry.get("detections", []):
            tid = det.get("track_id")
            if tid is not None:
                track_frames.setdefault(tid, []).append((entry["frame_idx"], det))
                new_dets.append(det)
            elif det.get("label") in KEEP_UNTRACKED:
                new_dets.append(det)
            # else: drop untracked non-essential detections (flicker source)
        entry["detections"] = new_dets

    # remove short tracks (<3 frames)
    n_short_removed = 0
    short_tids = {tid for tid, frames in track_frames.items() if len(frames) < 3}
    if short_tids:
        for entry in all_frames:
            before = len(entry["detections"])
            entry["detections"] = [d for d in entry["detections"]
                                   if d.get("track_id") not in short_tids]
            n_short_removed += before - len(entry["detections"])

    # fill 1-frame gaps in tracks
    n_gaps_filled = 0
    frame_idx_map = {e["frame_idx"]: i for i, e in enumerate(all_frames)}
    for tid, frame_det_list in track_frames.items():
        if tid in short_tids:
            continue
        frame_det_list.sort(key=lambda x: x[0])
        for i in range(len(frame_det_list) - 1):
            fidx_a, det_a = frame_det_list[i]
            fidx_b, det_b = frame_det_list[i + 1]
            gap = fidx_b - fidx_a
            if 2 <= gap <= 5:  # fill gaps up to 4 missing frames
                for fill_fidx in range(fidx_a + 1, fidx_b):
                    if fill_fidx in frame_idx_map:
                        fill_entry = all_frames[frame_idx_map[fill_fidx]]
                        # use det_a as template, already in the list won't duplicate
                        existing_tids = {d.get("track_id") for d in fill_entry["detections"]}
                        if tid not in existing_tids:
                            fill_entry["detections"].append(dict(det_a))
                            n_gaps_filled += 1

    # filter flickery untracked detections
    FLICKER_LABELS = {"road_arrow", "speed_sign", "dustbin", "traffic_cone",
                      "traffic_cylinder"}
    # collect all untracked dets by label
    untracked_by_label = {}
    for entry in all_frames:
        for det in entry.get("detections", []):
            if det.get("track_id") is not None:
                continue
            if det.get("label") not in FLICKER_LABELS:
                continue
            untracked_by_label.setdefault(det["label"], []).append(
                (entry["frame_idx"], det))

    # cluster by bbox center, keep only persistent ones (>=3 frames)
    n_flicker_removed = 0
    keep_keys = set()
    for label, detections in untracked_by_label.items():
        clusters = []
        for fidx, det in detections:
            b = det["bbox"]
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            matched = False
            for cluster in clusters:
                rcx, rcy = cluster["center"]
                if abs(cx - rcx) < 50 and abs(cy - rcy) < 50:
                    cluster["frames"].append((fidx, det))
                    # update running center
                    n = len(cluster["frames"])
                    cluster["center"] = (rcx + (cx-rcx)/n, rcy + (cy-rcy)/n)
                    matched = True
                    break
            if not matched:
                clusters.append({"center": (cx, cy), "frames": [(fidx, det)]})

        for cluster in clusters:
            if len(cluster["frames"]) >= 3:
                for fidx, det in cluster["frames"]:
                    keep_keys.add((fidx, label, int(det["bbox"][0]*100)))
            else:
                n_flicker_removed += len(cluster["frames"])

    # apply filter
    for entry in all_frames:
        new_dets = []
        for det in entry.get("detections", []):
            if det.get("track_id") is not None:
                new_dets.append(det)
                continue
            label = det.get("label")
            if label not in FLICKER_LABELS:
                new_dets.append(det)
                continue
            key = (entry["frame_idx"], label, int(det["bbox"][0]*100))
            if key in keep_keys:
                new_dets.append(det)
        entry["detections"] = new_dets

    print(f"Temporal smoothing: removed {n_short_removed} short-track dets "
          f"({len(short_tids)} tracks), filled {n_gaps_filled} gaps, "
          f"removed {n_flicker_removed} flickery untracked dets")

    # ensure every frame has a (possibly empty) lanes field
    for det_entry in det_by_frame.values():
        det_entry.setdefault("lanes", [])

    # render every 36th frame + any frame with rare detections (arrows, speed signs)
    def is_rare(e):
        for d in e.get("detections", []):
            l = d.get("label")
            if l in ("road_arrow", "speed_sign"):
                return True
            if l == "traffic_light" and d.get("arrow", "none") != "none":
                return True
        return False

    if args.all_frames:
        frames_to_process = list(all_frames)
        frames_to_process.sort(key=lambda e: e["frame_idx"])
        print(f"All frames: {len(frames_to_process)}")
    else:
        if all_frames:
            max_idx = max(e["frame_idx"] for e in all_frames)
            cadence = set(range(0, max_idx + 1, 36))
        else:
            cadence = set()
        rare_idxs = {e["frame_idx"] for e in all_frames if is_rare(e)}
        keep = cadence | rare_idxs
        frames_to_process = [e for e in all_frames if e["frame_idx"] in keep]
        frames_to_process.sort(key=lambda e: e["frame_idx"])
        print(f"Cadence: {len(frames_to_process)} frames "
              f"(every 36 + {len(rare_idxs - cadence)} rare-only)")

    print(f"Loaded {len(frames_to_process)} frames for {args.scene} (lane-driven cadence)")

    if args.frame >= 0:
        # explicit frame request: render even if not in lane file (no lanes shown)
        frame_data = det_by_frame.get(args.frame)
        if frame_data:
            if "lanes" not in frame_data:
                frame_data["lanes"] = []
            render_frame(frame_data, args.scene, args.frame, calib)
        else:
            print(f"Frame {args.frame} not in detection file")
    else:
        stepped = frames_to_process[::args.render_step]
        frames_to_render = stepped[args.skip:]
        if args.max_frames > 0:
            frames_to_render = frames_to_render[:args.max_frames]
        print(f"Rendering {len(frames_to_render)} frames (skip={args.skip} step={args.render_step})")
        failed = []
        for i, frame_data in enumerate(frames_to_render):
            try:
                render_frame(frame_data, args.scene, frame_data["frame_idx"], calib)
            except Exception as e:
                fidx = frame_data["frame_idx"]
                print(f"  FAILED frame {fidx}: {e}")
                failed.append(fidx)
                # reset scene state so next frame starts clean
                global _scene_initialized
                _scene_initialized = False
        if failed:
            print(f"WARNING: {len(failed)} frames failed: {failed[:20]}")


if __name__ == "__main__":
    main()
