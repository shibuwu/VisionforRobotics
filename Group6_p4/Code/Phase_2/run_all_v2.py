import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import VIODataset
from models import VIOModel
from encoder_inertial import InertialEncoder
from encoder_vision_v2 import VisionEncoderV2
from encoder_fusion_v2 import FusionEncoderV2
from dead_reckon import chain_poses
from evaluate import compute_ate_evo, compute_ate_handrolled, HAS_EVO
from plot_trajectories import plot_one_sequence, load_metrics

ENCODER_REGISTRY = {
    "inertial":   InertialEncoder,
    "vision_v2":  VisionEncoderV2,
    "fusion_v2":  FusionEncoderV2,
}

def build_model_from_checkpoint(ckpt_path, encoder_type, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    enc_kwargs = cfg.get("encoder_kwargs", {})
    encoder = ENCODER_REGISTRY[encoder_type](**enc_kwargs)
    model = VIOModel(encoder, feature_dim=encoder.feature_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

@torch.no_grad()
def run_inference(model, test_root, seq_name, mean_path, std_path, device,
                  batch_size=32, frame_gap=5):
    ds = VIODataset(test_root, frame_gap=frame_gap, image_size=None,
                    mean_path=mean_path, std_path=std_path, sequences=[seq_name])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_t, all_q = [], []
    for batch in loader:
        batch_dev = {"img_pair": batch["img_pair"].to(device, non_blocking=True),
                     "imu":      batch["imu"].to(device, non_blocking=True)}
        pred = model(batch_dev)
        all_t.append(pred["delta_t"].cpu().numpy())
        all_q.append(pred["delta_q"].cpu().numpy())

    delta_t_all = np.concatenate(all_t, 0)
    delta_q_all = np.concatenate(all_q, 0)
    sign = np.where(delta_q_all[:, :1] < 0, -1.0, 1.0)
    delta_q_all = delta_q_all * sign

    delta_t_pred = delta_t_all[::frame_gap]
    delta_q_pred = delta_q_all[::frame_gap]

    cam = np.load(Path(test_root) / seq_name / "camera.npz")
    n_full = cam["cam_positions"].shape[0]

    n_pos = len(delta_t_pred) + 1
    gt_indices = np.arange(0, n_pos * frame_gap, frame_gap)
    gt_indices = gt_indices[gt_indices < n_full]
    delta_t_pred = delta_t_pred[:len(gt_indices) - 1]
    delta_q_pred = delta_q_pred[:len(gt_indices) - 1]

    pos_strided = cam["cam_positions"][gt_indices]
    q_strided   = cam["cam_quaternions"][gt_indices]
    ts_strided  = cam["cam_timestamps"][gt_indices]

    return {
        "delta_t_pred": delta_t_pred.astype(np.float32),
        "delta_q_pred": delta_q_pred.astype(np.float32),
        "cam_positions":   pos_strided.astype(np.float32),
        "cam_quaternions": q_strided.astype(np.float32),
        "cam_timestamps":  ts_strided.astype(np.float32),
    }

def list_test_sequences(test_root):
    return sorted([p.name for p in Path(test_root).iterdir()
                   if p.is_dir() and p.name.startswith("seq_")])

def run_one(encoder_type, label, ckpt_path, test_root, seq, mean, std,
            out_dir, device, batch_size=32, frame_gap=5, use_evo=True):
    seq_out = Path(out_dir) / label / seq
    seq_out.mkdir(parents=True, exist_ok=True)

    deltas_path = seq_out / "predicted_deltas.npz"
    if deltas_path.exists():
        deltas = dict(np.load(deltas_path))
    else:
        model, _ = build_model_from_checkpoint(ckpt_path, encoder_type, device)
        deltas = run_inference(model, test_root, seq, mean, std, device,
                               batch_size=batch_size, frame_gap=frame_gap)
        np.savez(deltas_path, **deltas)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred_pos, pred_q = chain_poses(
        start_pos=deltas["cam_positions"][0],
        start_q_wxyz=deltas["cam_quaternions"][0],
        delta_t_body=deltas["delta_t_pred"],
        delta_q_wxyz=deltas["delta_q_pred"])

    np.savez(seq_out / "dead_reckoned.npz",
             pred_positions=pred_pos.astype(np.float32),
             pred_quaternions=pred_q.astype(np.float32),
             gt_positions=deltas["cam_positions"].astype(np.float32),
             gt_quaternions=deltas["cam_quaternions"].astype(np.float32),
             timestamps=deltas["cam_timestamps"].astype(np.float32))

    if use_evo and HAS_EVO:
        try:
            m = compute_ate_evo(pred_pos, pred_q, deltas["cam_positions"],
                                deltas["cam_quaternions"], deltas["cam_timestamps"])
        except Exception:
            m = compute_ate_handrolled(pred_pos, deltas["cam_positions"])
    else:
        m = compute_ate_handrolled(pred_pos, deltas["cam_positions"])

    aligned = m.pop("aligned_positions")
    with open(seq_out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    np.save(seq_out / "aligned_positions.npy", aligned)
    return m

def make_summary(all_metrics, labels, seqs, out_path):
    lines = ["| Network | " + " | ".join(seqs) + " | Mean |",
             "|---" * (len(seqs) + 2) + "|"]
    for net in labels:
        row, nums = [], []
        for seq in seqs:
            m = all_metrics.get(net, {}).get(seq)
            if m is None:
                row.append("—")
            else:
                row.append(f"{m['ate_rmse']:.4f}")
                nums.append(m['ate_rmse'])
        mean_str = f"{np.mean(nums):.4f}" if nums else "—"
        lines.append(f"| {net} | " + " | ".join(row) + f" | **{mean_str}** |")
    table = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write("# RMSE ATE V2 (m, lower is better)\n\n")
        f.write(table + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir",  default="./runs")
    ap.add_argument("--test_root", default="./dataset/test")
    ap.add_argument("--mean",      default="./stats/mean.npy")
    ap.add_argument("--std",       default="./stats/std.npy")
    ap.add_argument("--out_dir",   default="./stage3_results_v2")
    ap.add_argument("--frame_gap", type=int, default=5)
    ap.add_argument("--ckpt_name", default="best.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_evo", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    networks = {"inertial_v2": "inertial",
                "vision_v2":   "vision_v2",
                "fusion_v2":   "fusion_v2"}

    seqs = list_test_sequences(args.test_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for label, encoder_type in networks.items():
        ckpt_path = Path(args.runs_dir) / label / args.ckpt_name
        if not ckpt_path.exists():
            continue
        all_metrics[label] = {}
        for seq in seqs:
            m = run_one(encoder_type, label, ckpt_path, args.test_root, seq,
                        args.mean, args.std, out_dir, device,
                        batch_size=args.batch_size, frame_gap=args.frame_gap,
                        use_evo=not args.no_evo)
            all_metrics[label][seq] = m

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for seq in seqs:
        pred_data = {}
        for label in networks:
            dr = out_dir / label / seq / "dead_reckoned.npz"
            mp = out_dir / label / seq / "metrics.json"
            if not dr.exists():
                continue
            d = np.load(dr)
            color_key = label.replace("_v2", "")
            pred_data[color_key] = {
                "pred_pos": d["pred_positions"],
                "pred_q":   d["pred_quaternions"],
                "gt_pos":   d["gt_positions"],
                "gt_q":     d["gt_quaternions"],
                "metrics":  load_metrics(mp),
            }
        if pred_data:
            plot_one_sequence(seq, pred_data, plots_dir / f"{seq}_comparison.png")

    make_summary(all_metrics, list(networks.keys()), seqs, out_dir / "results_table.md")

if __name__ == "__main__":
    main()
