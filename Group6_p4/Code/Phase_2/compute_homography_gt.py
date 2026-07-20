import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
def quat_wxyz_to_rotmat(q_wxyz):
    q_xyzw = np.roll(q_wxyz, -1)
    return R.from_quat(q_xyzw).as_matrix()
def compute_homography(K, R_world_i, t_world_i, R_world_j, t_world_j, plane_z=0.0):
    n_world = np.array([0.0, 0.0, 1.0])
    n_i = R_world_i.T @ n_world
    d_i = float(t_world_i[2] - plane_z)
    if d_i <= 0:
        d_i = max(abs(d_i), 1e-3)
    R_ij = R_world_j.T @ R_world_i
    t_ij = R_world_j.T @ (t_world_i - t_world_j)
    R_ji = R_world_j.T @ R_world_i
    t_ji_in_i = R_world_i.T @ (t_world_j - t_world_i)
    H_normalized = R_ji - np.outer(t_ji_in_i, n_i) / d_i
    H = K @ H_normalized @ np.linalg.inv(K)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H
def homography_to_4point(H, image_size):
    h_img, w_img = image_size
    corners = np.array([
        [0, 0, 1],
        [w_img - 1, 0, 1],
        [w_img - 1, h_img - 1, 1],
        [0, h_img - 1, 1],
    ], dtype=np.float64).T
    warped = H @ corners
    warped_xy = warped[:2, :] / warped[2:3, :]
    original_xy = corners[:2, :]
    displacements = (warped_xy - original_xy).T
    return displacements.flatten().astype(np.float32)
def process_sequence(seq_path, image_size=(240, 320)):
    cam = np.load(seq_path / "camera.npz")
    K = np.load(seq_path / "intrinsics.npy")
    cam_pos = cam["cam_positions"]
    cam_q = cam["cam_quaternions"]
    n_frames = len(cam_pos)
    R_mats = np.stack([quat_wxyz_to_rotmat(q) for q in cam_q], axis=0)
    H_all = np.zeros((n_frames - 1, 3, 3), dtype=np.float32)
    h4pt_all = np.zeros((n_frames - 1, 8), dtype=np.float32)
    for i in range(n_frames - 1):
        H = compute_homography(
            K=K,
            R_world_i=R_mats[i],
            t_world_i=cam_pos[i],
            R_world_j=R_mats[i + 1],
            t_world_j=cam_pos[i + 1],
            plane_z=0.0,
        )
        H_all[i] = H.astype(np.float32)
        h4pt_all[i] = homography_to_4point(H, image_size)
    np.savez(seq_path / "homography_gt.npz", H=H_all, h4pt=h4pt_all)
    return H_all, h4pt_all
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    args = ap.parse_args()
    root = Path(args.root)
    seqs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seq_")])
    print(f"Found {len(seqs)} sequences in {root}")
    stats = {"total_pairs": 0, "h4pt_min": [], "h4pt_max": [], "h4pt_mean_abs": []}
    for seq_path in tqdm(seqs, desc="Computing GT homography"):
        H, h4pt = process_sequence(seq_path, image_size=(args.height, args.width))
        stats["total_pairs"] += len(h4pt)
        stats["h4pt_min"].append(h4pt.min())
        stats["h4pt_max"].append(h4pt.max())
        stats["h4pt_mean_abs"].append(np.abs(h4pt).mean())
    print(f"\nDone. Total frame pairs: {stats['total_pairs']}")
    print(f"4-point displacement stats (pixels):")
    print(f"  Min:        {min(stats['h4pt_min']):.2f}")
    print(f"  Max:        {max(stats['h4pt_max']):.2f}")
    print(f"  Mean |abs|: {np.mean(stats['h4pt_mean_abs']):.2f}")
    print(f"\nSaved homography_gt.npz to each seq_* folder under {root}")
if __name__ == "__main__":
    main()
