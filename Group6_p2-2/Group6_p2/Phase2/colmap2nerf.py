import numpy as np
import json
import os
import sys
from pathlib import Path
from PIL import Image

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def parse_cameras(path):
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {
                'model': model, 'width': width, 'height': height, 'params': params
            }
    return cameras

def parse_images(path):
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('#'):
            i += 1
            continue
        parts = lines[i].strip().split()
        if len(parts) < 10:
            i += 1
            continue
        img_id = int(parts[0])
        qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        cam_id = int(parts[8])
        name = parts[9]
        images[img_id] = {
            'qvec': qvec, 'tvec': tvec, 'cam_id': cam_id, 'name': name
        }
        i += 2
    return images

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    colmap_dir = os.path.join(base_dir, 'colmap_ws', 'sparse', '0')
    image_dir = base_dir

    cameras = parse_cameras(os.path.join(colmap_dir, 'cameras.txt'))
    images = parse_images(os.path.join(colmap_dir, 'images.txt'))

    print(f"Found {len(cameras)} camera(s), {len(images)} images")

    cam = cameras[1]
    w, h = cam['width'], cam['height']

    if cam['model'] == 'SIMPLE_RADIAL':
        fl = cam['params'][0]
    elif cam['model'] == 'SIMPLE_PINHOLE':
        fl = cam['params'][0]
    elif cam['model'] == 'PINHOLE':
        fl = (cam['params'][0] + cam['params'][1]) / 2
    elif cam['model'] == 'RADIAL':
        fl = cam['params'][0]
    else:
        fl = cam['params'][0]
        print(f"Warning: unknown model {cam['model']}, using first param as focal length")

    camera_angle_x = 2 * np.arctan(w / (2 * fl))
    print(f"Focal length: {fl:.2f}px, FOV: {np.degrees(camera_angle_x):.1f}°")

    target_size = 800
    out_dir = os.path.join(base_dir, 'nerf_custom')
    img_out_dir = os.path.join(out_dir, 'train')
    os.makedirs(img_out_dir, exist_ok=True)

    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    frames = []
    for img_id in sorted(images.keys()):
        img = images[img_id]
        name = img['name']

        R = qvec2rotmat(img['qvec'])
        t = img['tvec']

        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t

        src_path = os.path.join(image_dir, name)
        if not os.path.exists(src_path):
            print(f"Warning: {name} not found, skipping")
            continue

        dst_name = name.replace('.jpg', '.png').replace('.JPG', '.png')
        dst_path = os.path.join(img_out_dir, dst_name)

        pil_img = Image.open(src_path)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        pil_img.save(dst_path)

        frames.append({
            'file_path': f'./train/{dst_name}',
            'transform_matrix': c2w.tolist()
        })

    transforms = {
        'camera_angle_x': float(camera_angle_x),
        'frames': frames
    }

    out_path = os.path.join(out_dir, 'transforms_train.json')
    with open(out_path, 'w') as f:
        json.dump(transforms, f)

    test_frames = [frames[i] for i in range(0, len(frames), 8)]
    transforms_test = {
        'camera_angle_x': float(camera_angle_x),
        'frames': test_frames
    }
    with open(os.path.join(out_dir, 'transforms_test.json'), 'w') as f:
        json.dump(transforms_test, f)

    val_frames = [frames[i] for i in range(4, len(frames), 8)]
    transforms_val = {
        'camera_angle_x': float(camera_angle_x),
        'frames': val_frames
    }
    with open(os.path.join(out_dir, 'transforms_val.json'), 'w') as f:
        json.dump(transforms_val, f)

    print(f"\nDone! Output in: {out_dir}")
    print(f"  Images resized to: {new_w}x{new_h}")
    print(f"  Train frames: {len(frames)}")
    print(f"  Test frames: {len(test_frames)}")
    print(f"  Val frames: {len(val_frames)}")

if __name__ == '__main__':
    main()
