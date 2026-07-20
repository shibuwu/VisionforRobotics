import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

COLORS = {
    "groundtruth": "red",
    "vision":      "tab:blue",
    "inertial":    "tab:green",
    "fusion":      "tab:purple",
}

def load_metrics(metrics_path):
    if metrics_path is None or not Path(metrics_path).exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)

def plot_one_sequence(seq_name, pred_data_per_network, out_path):
    fig = plt.figure(figsize=(14, 6))

    ax2d = fig.add_subplot(1, 2, 1)
    first_net = next(iter(pred_data_per_network.values()))
    gt = first_net["gt_pos"]
    ax2d.plot(gt[:, 0], gt[:, 1], color=COLORS["groundtruth"], lw=2.0,
              label="Ground Truth", zorder=3)
    ax2d.scatter([gt[0, 0]], [gt[0, 1]], c="black", marker="o", s=60,
                 zorder=5, label="Start")

    for name, d in pred_data_per_network.items():
        pred = d["pred_pos"]
        ate = d["metrics"]["ate_rmse"] if d["metrics"] else float("nan")
        ax2d.plot(pred[:, 0], pred[:, 1], color=COLORS[name], lw=1.4, alpha=0.85,
                  label=f"{name} (ATE={ate:.3f}m)", zorder=2)
    ax2d.set_xlabel("X (m)")
    ax2d.set_ylabel("Y (m)")
    ax2d.set_title(f"{seq_name} — Top-down (X-Y)")
    ax2d.set_aspect("equal", adjustable="datalim")
    ax2d.grid(True, alpha=0.3)
    ax2d.legend(loc="best", fontsize=9)

    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax3d.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=COLORS["groundtruth"], lw=2.0,
              label="Ground Truth")
    ax3d.scatter([gt[0, 0]], [gt[0, 1]], [gt[0, 2]], c="black", marker="o", s=60,
                 label="Start")
    for name, d in pred_data_per_network.items():
        pred = d["pred_pos"]
        ax3d.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=COLORS[name],
                  lw=1.2, alpha=0.85, label=name)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"{seq_name} — 3D")
    ax3d.legend(loc="best", fontsize=9)

    fig.suptitle(f"Trajectory comparison: {seq_name}", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--networks", nargs="+", default=["vision", "inertial", "fusion"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.results_root)
    pred_data = {}
    for name in args.networks:
        dr_path = root / name / args.seq / "dead_reckoned.npz"
        m_path  = root / name / args.seq / "metrics.json"
        if not dr_path.exists():
            print(f"  warning: {dr_path} missing, skipping {name}")
            continue
        d = np.load(dr_path)
        pred_data[name] = {
            "pred_pos": d["pred_positions"],
            "pred_q":   d["pred_quaternions"],
            "gt_pos":   d["gt_positions"],
            "gt_q":     d["gt_quaternions"],
            "metrics":  load_metrics(m_path),
        }
    if not pred_data:
        raise RuntimeError(f"No prediction data found under {root} for {args.seq}")

    plot_one_sequence(args.seq, pred_data, args.output)

if __name__ == "__main__":
    main()
