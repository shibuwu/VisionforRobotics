import time
import torch
from torch.utils.data import DataLoader

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
    log_every: int = 50,
) -> dict:
    model.train()
    n_batches = 0
    sum_total, sum_t, sum_r = 0.0, 0.0, 0.0
    t_start = time.time()
    for step, batch in enumerate(loader):
        batch_dev = {
            "img_pair": batch["img_pair"].to(device, non_blocking=True),
            "imu":      batch["imu"].to(device, non_blocking=True),
        }
        target = {
            "delta_t":  batch["delta_t"].to(device, non_blocking=True),
            "delta_q":  batch["delta_q"].to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch_dev)
        loss, comp = loss_fn(pred, target)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        sum_total += comp["loss_total"]
        sum_t     += comp["loss_t"]
        sum_r     += comp["loss_r"]
        n_batches += 1
        if log_every > 0 and (step + 1) % log_every == 0:
            elapsed = time.time() - t_start
            print(f"    step {step+1:5d}/{len(loader)}  loss={comp['loss_total']:.4e}  t={comp['loss_t']:.3e}  r={comp['loss_r']:.3e}  ({elapsed:.1f}s)")
    duration = time.time() - t_start
    return {
        "train/loss_total": sum_total / max(n_batches, 1),
        "train/loss_t":     sum_t     / max(n_batches, 1),
        "train/loss_r":     sum_r     / max(n_batches, 1),
        "train/duration_s": duration,
        "train/n_batches":  n_batches,
    }

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    n_batches = 0
    sum_total, sum_t, sum_r = 0.0, 0.0, 0.0
    t_start = time.time()
    for batch in loader:
        batch_dev = {
            "img_pair": batch["img_pair"].to(device, non_blocking=True),
            "imu":      batch["imu"].to(device, non_blocking=True),
        }
        target = {
            "delta_t":  batch["delta_t"].to(device, non_blocking=True),
            "delta_q":  batch["delta_q"].to(device, non_blocking=True),
        }
        pred = model(batch_dev)
        _, comp = loss_fn(pred, target)
        sum_total += comp["loss_total"]
        sum_t     += comp["loss_t"]
        sum_r     += comp["loss_r"]
        n_batches += 1
    duration = time.time() - t_start
    return {
        "val/loss_total": sum_total / max(n_batches, 1),
        "val/loss_t":     sum_t     / max(n_batches, 1),
        "val/loss_r":     sum_r     / max(n_batches, 1),
        "val/duration_s": duration,
        "val/n_batches":  n_batches,
    }
