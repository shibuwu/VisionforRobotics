import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
DATA_ROOT = Path("./dataset_100hz/train")
OUT_DIR = Path("./stats_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
n_pixels = 0
sum_c = np.zeros(3, dtype=np.float64)
sumsq_c = np.zeros(3, dtype=np.float64)
seqs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("seq_")])
print(f"Found {len(seqs)} sequences")
for seq in tqdm(seqs):
    img_dir = seq / "images"
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float64) / 255.0
        sum_c   += img.sum(axis=(0, 1))
        sumsq_c += (img ** 2).sum(axis=(0, 1))
        n_pixels += img.shape[0] * img.shape[1]
mean = sum_c / n_pixels
var = sumsq_c / n_pixels - mean ** 2
std = np.sqrt(np.maximum(var, 1e-12))
print(f"\nmean: {mean}")
print(f"std:  {std}")
np.save(OUT_DIR / "mean.npy", mean.astype(np.float32))
np.save(OUT_DIR / "std.npy",  std.astype(np.float32))
print(f"\nSaved to {OUT_DIR}/mean.npy and {OUT_DIR}/std.npy")
