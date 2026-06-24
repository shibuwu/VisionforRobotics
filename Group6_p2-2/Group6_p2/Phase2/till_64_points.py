import json
import os
import numpy as np
from PIL import Image


def load_nerf_data(base_dir, split="train", half_res=False):
    json_path = os.path.join(base_dir, f"transforms_{split}.json")
    with open(json_path, "r") as f:
        meta = json.load(f)

    camera_angle_x = meta["camera_angle_x"]
    images, poses = [], []

    for frame in meta["frames"]:
        img_path = os.path.join(base_dir, frame["file_path"] + ".png")
        img = Image.open(img_path)

        if half_res:
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)

        img = np.array(img, dtype=np.float32) / 255.0

        if img.shape[2] == 4:
            alpha = img[:, :, 3:4]
            img = img[:, :, :3] * alpha + (1.0 - alpha)

        images.append(img)
        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

    images = np.stack(images, axis=0)
    poses = np.stack(poses, axis=0)

    H, W = images.shape[1], images.shape[2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    return images, poses, focal


def get_rays(H, W, focal, pose):
    j, i = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32))

    dirs_cam = np.stack([
         (j - 0.5 * W) / focal,
        -(i - 0.5 * H) / focal,
        -np.ones_like(i)
    ], axis=-1)

    R = pose[:3, :3]
    rays_d = dirs_cam @ R.T
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape).copy()

    return rays_o, rays_d


def sample_points_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True):
    t_vals = np.linspace(near, far, N_samples, dtype=np.float32)
    batch_shape = rays_o.shape[:-1]
    t_vals = np.broadcast_to(t_vals, (*batch_shape, N_samples)).copy()

    if perturb:
        bin_width = (far - near) / N_samples
        t_vals = t_vals + np.random.uniform(0.0, bin_width, size=t_vals.shape).astype(np.float32)

    pts = rays_o[..., None, :] + t_vals[..., :, None] * rays_d[..., None, :]
    return pts, t_vals


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "lego")

    images, poses, focal = load_nerf_data(base_dir, split="train")
    H, W = images.shape[1], images.shape[2]
    print(f"Loaded {len(images)} images at {H}x{W}, focal={focal:.2f}")

    rays_o, rays_d = get_rays(H, W, focal, poses[0])
    print(f"Rays: origins {rays_o.shape}, directions {rays_d.shape}")

    pts, t_vals = sample_points_along_rays(rays_o, rays_d, 2.0, 6.0, 64)
    print(f"Sample points: {pts.shape}, t_vals: {t_vals.shape}")
