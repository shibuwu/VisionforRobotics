import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", default="./stats")
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seq_")])
    assert len(seqs) > 0, f"No seq_* in {root}"
    mean = np.zeros(3, dtype=np.float64)
    M2   = np.zeros(3, dtype=np.float64)
    n    = 0
    img_files = []
    for s in seqs:
        files = sorted((s / "images").glob("frame_*.png"))[::args.stride]
        img_files.extend(files)
    print(f"Found {len(img_files)} images across {len(seqs)} sequences (stride={args.stride})")
    for f in tqdm(img_files, desc="streaming"):
        arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        n_batch = flat.shape[0]
        batch_mean = flat.mean(axis=0)
        batch_M2   = ((flat - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - mean
        new_n = n + n_batch
        mean = mean + delta * (n_batch / new_n)
        M2   = M2 + batch_M2 + (delta ** 2) * (n * n_batch / new_n)
        n = new_n
    var = M2 / max(n - 1, 1)
    std = np.sqrt(var)
    print(f"per-channel mean (R,G,B) = {mean}")
    print(f"per-channel std  (R,G,B) = {std}")
    np.save(out_dir / "mean.npy", mean.astype(np.float32))
    np.save(out_dir / "std.npy",  std.astype(np.float32))
    print(f"Saved {out_dir/'mean.npy'} and {out_dir/'std.npy'}")
if __name__ == "__main__":
    main()
