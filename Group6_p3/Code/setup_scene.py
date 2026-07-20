"""
Create the base Blender scene (Tesla-style visualization).
Run with: blender --background --python setup_scene.py

Outputs: base_scene.blend
"""

import bpy
import math
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PROJECT_DIR, "P3Data", "Assets")
OUTPUT_BLEND = os.path.join(PROJECT_DIR, "base_scene.blend")


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block_list in [bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.lights, bpy.data.cameras]:
        for block in block_list:
            if block.users == 0:
                block_list.remove(block)


def make_material(name, color, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def setup_road():
    # main road — light gray like the Tesla viz
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 100, 0))
    road = bpy.context.active_object
    road.name = "Road"
    road.scale = (20, 200, 1)
    road.data.materials.append(make_material("RoadMat", (0.35, 0.38, 0.40, 1.0), roughness=0.95))

    # sidewalks — slightly lighter
    for side, x in [("Left", -10.5), ("Right", 10.5)]:
        bpy.ops.mesh.primitive_plane_add(size=1, location=(x, 100, 0.02))
        sw = bpy.context.active_object
        sw.name = f"Shoulder_{side}"
        sw.scale = (3, 200, 1)
        sw.data.materials.append(make_material(f"ShoulderMat_{side}", (0.45, 0.47, 0.48, 1.0), roughness=0.9))


def setup_ego_car():
    blend_path = os.path.join(ASSETS_DIR, "Vehicles", "SedanAndHatchback.blend")
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        if "Car" in data_from.objects:
            data_to.objects = ["Car"]

    ego = None
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            ego = obj

    if ego is None:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.7))
        ego = bpy.context.active_object
        ego.scale = (1.0, 2.3, 0.6)

    s = 0.02
    ego.scale = (s, s, s)
    ego.rotation_euler = (0, 0, 0)
    ego.location = (0, 0, 0.024)
    ego.name = "EgoCar"

    # dark blue-black metallic paint (like Tesla in Fig 5/8)
    dark_mat = make_material("EgoCarPaint", (0.03, 0.03, 0.08, 1.0), roughness=0.2)
    if ego.data and hasattr(ego.data, 'materials'):
        ego.data.materials.clear()
        ego.data.materials.append(dark_mat)

    return ego


def setup_camera():
    # Tesla-style: low behind car, slightly elevated, looking forward
    bpy.ops.object.camera_add(location=(0, -5.0, 3.5))
    cam = bpy.context.active_object
    cam.name = "EgoCamera"
    cam.rotation_euler = (math.radians(75), 0, 0)  # mostly forward, slight downward
    cam.data.lens = 25  # wide enough to see sides
    cam.data.clip_end = 500

    bpy.context.scene.camera = cam

    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    return cam


def setup_lighting():
    # bright, even lighting — no harsh shadows
    bpy.ops.object.light_add(type='SUN', location=(0, 10, 30))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(60), 0, 0)

    # strong ambient fill
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 15))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 200.0
    fill.data.size = 30
    fill.rotation_euler = (0, 0, 0)  # pointing straight down


def setup_world():
    world = bpy.data.worlds.new("EinsteinWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    # light gray misty background like Tesla viz
    bg.inputs["Color"].default_value = (0.75, 0.78, 0.80, 1.0)
    bg.inputs["Strength"].default_value = 2.0


def main():
    clear_scene()
    setup_world()
    setup_road()
    setup_ego_car()
    setup_camera()
    setup_lighting()

    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
    print(f"Saved base scene to {OUTPUT_BLEND}")

    test_path = os.path.join(PROJECT_DIR, "renders", "base_scene_test.png")
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    bpy.context.scene.render.filepath = test_path
    bpy.ops.render.render(write_still=True)
    print(f"Test render saved to {test_path}")


if __name__ == "__main__":
    main()
