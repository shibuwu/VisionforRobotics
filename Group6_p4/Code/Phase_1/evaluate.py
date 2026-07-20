import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_tum(path):
    # TUM: timestamp tx ty tz qx qy qz qw
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1:4], data[:, 4:8]   # ts, xyz, quat_xyzw


def load_euroc_gt(path):
    # EuRoC state_groundtruth_estimate0/data.csv columns:
    # ts[ns], px, py, pz, qw, qx, qy, qz, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    ts = data[:, 0] * 1e-9
    xyz = data[:, 1:4]
    qw, qx, qy, qz = data[:, 4], data[:, 5], data[:, 6], data[:, 7]
    quat_xyzw = np.column_stack([qx, qy, qz, qw])
    return ts, xyz, quat_xyzw


def associate(ts_a, ts_b, max_diff=0.02):
    # For each ts in ts_a, find closest ts in ts_b within max_diff seconds.
    ts_b = np.asarray(ts_b)
    idx_a, idx_b = [], []
    for i, t in enumerate(ts_a):
        j = int(np.argmin(np.abs(ts_b - t)))
        if abs(ts_b[j] - t) <= max_diff:
            idx_a.append(i)
            idx_b.append(j)
    return np.array(idx_a, dtype=int), np.array(idx_b, dtype=int)


def align_se3(src, dst):
    # R, t minimizing ||R @ src_i + t - dst_i||^2 (Umeyama, no scale).
    src = np.asarray(src); dst = np.asarray(dst)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    Ss = src - mu_s
    Sd = dst - mu_d
    H = Ss.T @ Sd
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    aligned = (R @ src.T).T + t
    return R, t, aligned


def ate_metrics(aligned, gt):
    err = np.linalg.norm(aligned - gt, axis=1)
    return {
        'rmse':   float(np.sqrt(np.mean(err ** 2))),
        'mean':   float(np.mean(err)),
        'median': float(np.median(err)),
        'std':    float(np.std(err)),
        'min':    float(np.min(err)),
        'max':    float(np.max(err)),
        'n':      int(len(err)),
        'errors': err,
    }


def plot_comparison(est, gt, errors, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(gt[:, 0],  gt[:, 1],  color='magenta', linewidth=1.5, label='Ground truth')
    axes[0].plot(est[:, 0], est[:, 1], color='blue',    linewidth=1.0, label='Estimate', alpha=0.85)
    axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('y [m]')
    axes[0].set_title('Top-down (x-y)')
    axes[0].axis('equal'); axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')

    axes[1].plot(gt[:, 0],  gt[:, 2],  color='magenta', linewidth=1.5, label='Ground truth')
    axes[1].plot(est[:, 0], est[:, 2], color='blue',    linewidth=1.0, label='Estimate', alpha=0.85)
    axes[1].set_xlabel('x [m]'); axes[1].set_ylabel('z [m]')
    axes[1].set_title('Side (x-z)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')

    rmse = np.sqrt(np.mean(errors ** 2))
    fig.suptitle(f'S-MSCKF vs EuRoC ground truth  -  RMSE-ATE = {rmse:.4f} m',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_3d(est, gt, out_path):
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:, 0],  gt[:, 1],  gt[:, 2],  color='magenta', linewidth=1.5, label='Ground truth')
    ax.plot(est[:, 0], est[:, 1], est[:, 2], color='blue',    linewidth=1.0, label='Estimate')
    ax.scatter(est[0, 0], est[0, 1], est[0, 2], color='green', s=60, label='start')
    ax.scatter(est[-1, 0], est[-1, 1], est[-1, 2], color='red', s=60, label='end')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_title('Trajectory vs Ground truth (3D)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_error_over_time(ts, errors, out_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(ts - ts[0], errors, color='blue', linewidth=1.0)
    rmse = np.sqrt(np.mean(errors ** 2))
    ax.axhline(rmse, color='red', linestyle='--', linewidth=1.0,
               label=f'RMSE = {rmse:.4f} m')
    ax.set_xlabel('time since start [s]'); ax.set_ylabel('APE [m]')
    ax.set_title('Absolute Position Error over time')
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', default='vio_output/trajectory.tum',
                    help='Estimated trajectory (TUM format)')
    ap.add_argument('--gt', required=True,
                    help='EuRoC state_groundtruth_estimate0/data.csv')
    ap.add_argument('--max_diff', type=float, default=0.02,
                    help='max timestamp diff for association (seconds)')
    ap.add_argument('--outdir', default='vio_output',
                    help='where to save plots + aligned trajectory')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f'[load] estimate : {args.est}')
    ts_est, xyz_est, q_est = load_tum(args.est)
    print(f'       {len(ts_est)} poses, t = {ts_est[0]:.3f} .. {ts_est[-1]:.3f}')

    print(f'[load] gt       : {args.gt}')
    ts_gt, xyz_gt, q_gt = load_euroc_gt(args.gt)
    print(f'       {len(ts_gt)} poses, t = {ts_gt[0]:.3f} .. {ts_gt[-1]:.3f}')

    idx_e, idx_g = associate(ts_est, ts_gt, max_diff=args.max_diff)
    if len(idx_e) < 10:
        raise SystemExit(
            f'ERROR: only {len(idx_e)} associated pairs (max_diff={args.max_diff}s). '
            f'Check that estimate timestamps are in the same epoch as GT.')
    print(f'[assoc] {len(idx_e)} / {len(ts_est)} estimate poses associated '
          f'(max_diff = {args.max_diff}s)')

    est_pts = xyz_est[idx_e]
    gt_pts  = xyz_gt[idx_g]
    ts_pair = ts_est[idx_e]

    R, t, est_aligned = align_se3(est_pts, gt_pts)
    print('[align] SE(3) translation:', np.round(t, 4))

    m = ate_metrics(est_aligned, gt_pts)
    print('\nAPE / ATE (translation, SE(3)-aligned)')
    print(f'  samples : {m["n"]}')
    print(f'  RMSE    : {m["rmse"]:.4f} m')
    print(f'  mean    : {m["mean"]:.4f} m')
    print(f'  median  : {m["median"]:.4f} m')
    print(f'  std     : {m["std"]:.4f} m')
    print(f'  min     : {m["min"]:.4f} m')
    print(f'  max     : {m["max"]:.4f} m')
    print()

    p1 = os.path.join(args.outdir, 'evaluation.png')
    p2 = os.path.join(args.outdir, 'evaluation_3d.png')
    p3 = os.path.join(args.outdir, 'error_over_time.png')
    plot_comparison(est_aligned, gt_pts, m['errors'], p1)
    plot_3d(est_aligned, gt_pts, p2)
    plot_error_over_time(ts_pair, m['errors'], p3)
    print(f'[save] {p1}')
    print(f'[save] {p2}')
    print(f'[save] {p3}')

    aligned_path = os.path.join(args.outdir, 'aligned_estimate.tum')
    with open(aligned_path, 'w') as f:
        f.write('# timestamp tx ty tz qx qy qz qw (SE3-aligned to GT)\n')
        q_al = q_est[idx_e]
        for ts, p, q in zip(ts_pair, est_aligned, q_al):
            f.write(f'{ts:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} '
                    f'{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n')
    print(f'[save] {aligned_path}')


if __name__ == '__main__':
    main()