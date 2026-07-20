"""generate_sequence.py — one sequence of trajectory + IMU + camera data."""
import argparse
import json
import os
import shutil
import numpy as np
from trajectory_generator import generate_trajectory, relative_poses
from imu_simulator import simulate_imu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='data/seq_000')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--freq', type=int, default=1000)
    parser.add_argument('--cam_freq', type=int, default=100)
    parser.add_argument('--focal_length', type=float, default=35.0)
    parser.add_argument('--texture_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures'))
    parser.add_argument('--plane_size', type=float, default=50.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    n_wp = max(4, int(args.duration * 0.8))
    traj = generate_trajectory(duration=args.duration, freq=args.freq,
                               n_wp=n_wp, seed=args.seed)
    np.random.seed(args.seed + 10000)
    imu = simulate_imu(traj, freq=args.freq)

    # camera frame indices
    step = args.freq // args.cam_freq
    cam_idx = np.arange(0, len(traj['positions']), step)
    cam_pos = traj['positions'][cam_idx]
    cam_quats = traj['quaternions'][cam_idx]
    rel_t, rel_q = relative_poses(cam_quats, cam_pos)

    # pick random OSM map texture
    all_tex = [f for f in os.listdir(args.texture_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    map_tex = [f for f in all_tex if f.startswith('osm_')]
    textures = map_tex if map_tex else all_tex
    tex_name = str(rng.choice(textures))
    tex_src = os.path.join(args.texture_dir, tex_name)
    tex_ext = os.path.splitext(tex_name)[1]
    tex_dst = os.path.join(args.out_dir, 'texture_used' + tex_ext)
    shutil.copy2(tex_src, tex_dst)

    plane_size = args.plane_size + rng.uniform(-5, 5)
    uv_scale = float(rng.uniform(1.0, 4.0))

    np.savez(os.path.join(args.out_dir, 'groundtruth.npz'),
             timestamps=traj['timestamps'],
             positions=traj['positions'],
             quaternions=traj['quaternions'],
             velocities=traj['velocities'],
             angular_velocities=traj['angular_velocities'])

    np.savez(os.path.join(args.out_dir, 'imu.npz'),
             timestamps=imu['timestamps'],
             gyro=imu['gyro'],
             accel=imu['accel'])

    np.savez(os.path.join(args.out_dir, 'camera.npz'),
             cam_indices=cam_idx,
             cam_timestamps=traj['timestamps'][cam_idx],
             cam_positions=cam_pos,
             cam_quaternions=cam_quats,
             rel_translations=rel_t,
             rel_quaternions=rel_q)

    # IMU windows between consecutive camera frames
    imu_windows = []
    for i in range(len(cam_idx) - 1):
        start = cam_idx[i]
        end = cam_idx[i + 1]
        window = np.hstack([imu['gyro'][start:end], imu['accel'][start:end]])
        imu_windows.append(window)
    np.savez(os.path.join(args.out_dir, 'imu_windows.npz'),
             *imu_windows)

    config = {
        'seed': args.seed,
        'duration': args.duration,
        'freq': args.freq,
        'cam_freq': args.cam_freq,
        'focal_length': args.focal_length,
        'texture': tex_name,
        'plane_size': float(plane_size),
        'uv_scale': uv_scale,
        'n_cam_frames': int(len(cam_idx)),
        'n_imu_samples': int(len(imu['gyro'])),
        'imu_between_frames': step,
    }
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Sequence generated: {args.out_dir}")
    print(f"  {len(cam_idx)} camera frames, {len(imu['gyro'])} IMU samples")
    print(f"  Texture: {tex_name}, Plane size: {plane_size:.1f}m")
    print(f"  IMU window size: {step} samples between each camera pair")


if __name__ == '__main__':
    main()
