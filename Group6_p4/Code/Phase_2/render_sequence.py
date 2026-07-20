"""render_sequence.py — Blender headless rendering from pre-generated poses."""
import bpy
import sys
import os
import argparse
import json
import numpy as np
from mathutils import Quaternion, Vector


def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', type=str, required=True)
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 240])
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)


def setup_plane(size, texture_path, uv_scale=1.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = 'Floor'

    mat = bpy.data.materials.new('FloorMat')
    bsdf = mat.node_tree.nodes['Principled BSDF']
    tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(texture_path)

    # add UV scale via mapping node
    coord = mat.node_tree.nodes.new('ShaderNodeTexCoord')
    mapping = mat.node_tree.nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (uv_scale, uv_scale, 1.0)
    mat.node_tree.links.new(mapping.inputs['Vector'], coord.outputs['UV'])
    mat.node_tree.links.new(tex.inputs['Vector'], mapping.outputs['Vector'])
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex.outputs['Color'])
    plane.data.materials.append(mat)

    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')
    return plane


def setup_camera(focal_length=35):
    bpy.ops.object.camera_add(location=(0, 0, 2))
    cam = bpy.context.active_object
    cam.name = 'Camera'
    cam.data.lens = focal_length
    bpy.context.scene.camera = cam
    return cam


def setup_light():
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    bpy.context.active_object.data.energy = 3.0


def setup_render(res_x, res_y):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'FLAT'
    scene.display.shading.color_type = 'TEXTURE'
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False


def get_intrinsics(cam, res_x, res_y):
    f_mm = cam.data.lens
    sensor_w = cam.data.sensor_width
    # pixel focal lengths
    fx = f_mm * res_x / sensor_w
    fy = fx
    cx = res_x / 2.0
    cy = res_y / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])


def main():
    args = parse_args()
    seq_dir = args.seq_dir

    with open(os.path.join(seq_dir, 'config.json')) as f:
        config = json.load(f)

    cam_data = np.load(os.path.join(seq_dir, 'camera.npz'))
    positions = cam_data['cam_positions']
    quaternions = cam_data['cam_quaternions']

    # find texture file
    texture_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        p = os.path.join(seq_dir, 'texture_used' + ext)
        if os.path.exists(p):
            texture_path = p
            break
    if texture_path is None:
        tex_dir = config.get('texture_dir', os.path.join(os.path.dirname(__file__), 'textures'))
        texture_path = os.path.join(tex_dir, config['texture'])

    img_dir = os.path.join(seq_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    clear_scene()
    plane_size = config.get('plane_size', 50.0)
    focal_length = config.get('focal_length', 35)
    uv_scale = config.get('uv_scale', 1.0)
    setup_plane(plane_size, texture_path, uv_scale)
    cam = setup_camera(focal_length)
    setup_light()
    setup_render(args.resolution[0], args.resolution[1])

    # save camera intrinsics (fixed, no distortion)
    K = get_intrinsics(cam, args.resolution[0], args.resolution[1])
    np.save(os.path.join(seq_dir, 'intrinsics.npy'), K)

    n_frames = len(positions)
    print(f"Rendering {n_frames} frames...")
    for i in range(n_frames):
        cam.location = (float(positions[i, 0]),
                        float(positions[i, 1]),
                        float(positions[i, 2]))
        qw, qx, qy, qz = quaternions[i]
        cam.rotation_euler = Quaternion((qw, qx, qy, qz)).to_euler('XYZ')

        filepath = os.path.join(img_dir, f'frame_{i:05d}.png')
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        if i % 100 == 0:
            print(f"  {i}/{n_frames}")

    print(f"Done rendering {n_frames} frames to {img_dir}")


if __name__ == '__main__':
    main()
