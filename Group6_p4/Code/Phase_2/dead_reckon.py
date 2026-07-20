import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import wxyz_to_xyzw, xyzw_to_wxyz, quat_multiply_wxyz

def chain_poses(start_pos, start_q_wxyz, delta_t_body, delta_q_wxyz):
    N = delta_t_body.shape[0] + 1
    positions = np.zeros((N, 3), dtype=np.float64)
    quaternions = np.zeros((N, 4), dtype=np.float64)
    positions[0] = start_pos
    quaternions[0] = start_q_wxyz
    for i in range(N - 1):
        Ri = R.from_quat(wxyz_to_xyzw(quaternions[i]))
        positions[i + 1] = positions[i] + Ri.apply(delta_t_body[i])
        quaternions[i + 1] = quat_multiply_wxyz(quaternions[i], delta_q_wxyz[i])
        n = np.linalg.norm(quaternions[i + 1])
        if n > 1e-12:
            quaternions[i + 1] /= n
    return (positions, quaternions)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='path to predicted_deltas.npz')
    ap.add_argument('--output', required=True, help='path to dead_reckoned.npz')
    args = ap.parse_args()
    data = np.load(args.input)
    delta_t = data['delta_t_pred']
    delta_q = data['delta_q_pred']
    cam_positions = data['cam_positions']
    cam_quaternions = data['cam_quaternions']
    timestamps = data['cam_timestamps']
    start_pos = cam_positions[0]
    start_q = cam_quaternions[0]
    (pred_positions, pred_quaternions) = chain_poses(start_pos, start_q, delta_t, delta_q)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, pred_positions=pred_positions.astype(np.float32), pred_quaternions=pred_quaternions.astype(np.float32), gt_positions=cam_positions.astype(np.float32), gt_quaternions=cam_quaternions.astype(np.float32), timestamps=timestamps.astype(np.float32))
    final_drift = np.linalg.norm(pred_positions[-1] - cam_positions[-1])
    total_path = np.sum(np.linalg.norm(np.diff(cam_positions, axis=0), axis=1))
    print(f'Saved {out_path}')
    print(f'  final position drift: {final_drift:.4f} m')
    print(f'  total path length:    {total_path:.4f} m')
    print(f'  drift / path:         {100 * final_drift / max(total_path, 1e-06):.2f}%')
if __name__ == '__main__':
    main()
