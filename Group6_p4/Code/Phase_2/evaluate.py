import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import wxyz_to_xyzw

try:
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import sync, metrics, lie_algebra
    from evo.core.metrics import PoseRelation, Unit
    HAS_EVO = True
except ImportError:
    HAS_EVO = False

def umeyama_alignment(src, dst, with_scale=False):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt.T) < 0:
        S[2, 2] = -1
    Rmat = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0
    t = mu_dst - s * Rmat @ mu_src
    return s, Rmat, t

def compute_ate_handrolled(pred_pos, gt_pos):
    s, Rm, t = umeyama_alignment(pred_pos, gt_pos, with_scale=False)
    aligned = (s * (Rm @ pred_pos.T)).T + t
    errs = np.linalg.norm(aligned - gt_pos, axis=1)
    return {
        "ate_rmse":   float(np.sqrt((errs ** 2).mean())),
        "ate_mean":   float(errs.mean()),
        "ate_median": float(np.median(errs)),
        "ate_std":    float(errs.std()),
        "ate_max":    float(errs.max()),
        "n_poses":    int(len(errs)),
        "alignment_used": "Umeyama (hand-rolled, no scale)",
        "aligned_positions": aligned.astype(np.float32),
    }

def compute_ate_evo(pred_pos, pred_q_wxyz, gt_pos, gt_q_wxyz, timestamps):
    pred_traj = PoseTrajectory3D(
        positions_xyz=pred_pos.astype(np.float64),
        orientations_quat_wxyz=pred_q_wxyz.astype(np.float64),
        timestamps=timestamps.astype(np.float64),
    )
    gt_traj = PoseTrajectory3D(
        positions_xyz=gt_pos.astype(np.float64),
        orientations_quat_wxyz=gt_q_wxyz.astype(np.float64),
        timestamps=timestamps.astype(np.float64),
    )
    gt_synced, pred_synced = sync.associate_trajectories(gt_traj, pred_traj)
    pred_aligned = pred_synced.align(gt_synced, correct_scale=False, correct_only_scale=False)
    ape = metrics.APE(PoseRelation.translation_part)
    ape.process_data((gt_synced, pred_synced))
    stats = ape.get_all_statistics()
    return {
        "ate_rmse":   float(stats["rmse"]),
        "ate_mean":   float(stats["mean"]),
        "ate_median": float(stats["median"]),
        "ate_std":    float(stats["std"]),
        "ate_max":    float(stats["max"]),
        "n_poses":    int(len(gt_synced.positions_xyz)),
        "alignment_used": "evo SE(3), no scale",
        "aligned_positions": np.asarray(pred_synced.positions_xyz, dtype=np.float32),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--save_aligned", default=None)
    ap.add_argument("--force_handrolled", action="store_true")
    args = ap.parse_args()
    data = np.load(args.input)
    pred_pos = data["pred_positions"]
    pred_q   = data["pred_quaternions"]
    gt_pos   = data["gt_positions"]
    gt_q     = data["gt_quaternions"]
    ts       = data["timestamps"]
    use_evo = HAS_EVO and not args.force_handrolled
    print(f"Using {'evo' if use_evo else 'hand-rolled'} alignment / ATE")
    if use_evo:
        try:
            metrics_out = compute_ate_evo(pred_pos, pred_q, gt_pos, gt_q, ts)
        except Exception as e:
            print(f"  evo failed ({e}), falling back to hand-rolled")
            metrics_out = compute_ate_handrolled(pred_pos, gt_pos)
    else:
        metrics_out = compute_ate_handrolled(pred_pos, gt_pos)
    aligned = metrics_out.pop("aligned_positions")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    if args.save_aligned is not None:
        np.save(args.save_aligned, aligned)
    print(f"\nResults:")
    for k, v in metrics_out.items():
        if isinstance(v, float):
            print(f"  {k:15}: {v:.4f}")
        else:
            print(f"  {k:15}: {v}")
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
