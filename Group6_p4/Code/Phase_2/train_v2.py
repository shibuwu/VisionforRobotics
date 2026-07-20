import argparse
import importlib
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import VIODataset
from losses import PoseLoss
from models import VIOModel
from encoder_inertial import InertialEncoder
from encoder_vision import VisionEncoder
from encoder_fusion import FusionEncoder
from encoder_vision_v2 import VisionEncoderV2
from encoder_fusion_v2 import FusionEncoderV2
from engine import train_one_epoch, validate
from checkpoint import save_checkpoint, load_checkpoint
from logger import JsonlLogger

ENCODER_REGISTRY = {'inertial': InertialEncoder, 'vision': VisionEncoder, 'fusion': FusionEncoder, 'vision_v2': VisionEncoderV2, 'fusion_v2': FusionEncoderV2}

def build_encoder(encoder_type: str, kwargs: dict):
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f'Unknown encoder_type {encoder_type!r}')
    return ENCODER_REGISTRY[encoder_type](**kwargs)

def build_split(data_root: str, val_ratio: int, split_seed: int):
    root = Path(data_root)
    all_seqs = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith('seq_')])
    if val_ratio >= len(all_seqs):
        raise ValueError(f'val_ratio={val_ratio} >= len(all_seqs)={len(all_seqs)}')
    train_seqs = all_seqs[:-val_ratio]
    val_seqs = all_seqs[-val_ratio:]
    return (train_seqs, val_seqs)

def build_optimizer(name: str, params, lr: float, weight_decay: float):
    if name.lower() == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name.lower() == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unknown optimizer {name!r}')

def build_scheduler(name, optimizer, total_epochs: int, min_lr: float):
    if name is None:
        return None
    if name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
    raise ValueError(f'Unknown scheduler {name!r}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg_module = importlib.import_module(args.config)
    cfg = cfg_module.config
    print(f'Loaded config: {args.config}')
    print(json.dumps(cfg, indent=2, default=str))
    out_dir = Path(cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2, default=str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(cfg['split_seed'])
    (train_seqs, val_seqs) = build_split(cfg['data_root'], cfg['val_ratio'], cfg['split_seed'])
    print(f'Train seqs ({len(train_seqs)}): {train_seqs[:3]}...{train_seqs[-1:]}')
    print(f'Val   seqs ({len(val_seqs)}): {val_seqs}')
    train_ds = VIODataset(cfg['data_root'], frame_gap=cfg['frame_gap'], image_size=cfg['image_size'], mean_path=cfg['mean_path'], std_path=cfg['std_path'], sequences=train_seqs)
    val_ds = VIODataset(cfg['data_root'], frame_gap=cfg['frame_gap'], image_size=cfg['image_size'], mean_path=cfg['mean_path'], std_path=cfg['std_path'], sequences=val_seqs)
    print(f'len(train_ds)={len(train_ds)}, len(val_ds)={len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=device.type == 'cuda', persistent_workers=cfg['num_workers'] > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=device.type == 'cuda', persistent_workers=cfg['num_workers'] > 0)
    encoder = build_encoder(cfg['encoder_type'], cfg['encoder_kwargs'])
    model = VIOModel(encoder, feature_dim=encoder.feature_dim).to(device)
    n_params = sum((p.numel() for p in model.parameters()))
    print(f"Model: encoder={cfg['encoder_type']}, total params={n_params:,}")
    loss_fn = PoseLoss(lambda_rot=cfg['lambda_rot'])
    optimizer = build_optimizer(cfg['optimizer'], model.parameters(), cfg['lr'], cfg['weight_decay'])
    scheduler = build_scheduler(cfg.get('scheduler'), optimizer, cfg['epochs'], cfg.get('min_lr', 0.0))
    start_epoch = 1
    best_val_loss = float('inf')
    if cfg.get('resume_from') is not None:
        print(f"Resuming from {cfg['resume_from']}")
        ckpt = load_checkpoint(cfg['resume_from'], model, optimizer, device=device)
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('metrics', {}).get('val/loss_total', float('inf'))
        if scheduler is not None:
            for _ in range(ckpt['epoch']):
                scheduler.step()
    print(f"\nStarting training: epochs {start_epoch} -> {cfg['epochs']}")
    t_global = time.time()
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        print(f"\n=== Epoch {epoch}/{cfg['epochs']} ===")
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=cfg.get('grad_clip'), log_every=cfg.get('log_every_n_steps', 0))
        print(f"  train: total={train_metrics['train/loss_total']:.4e}  t={train_metrics['train/loss_t']:.3e}  r={train_metrics['train/loss_r']:.3e}  ({train_metrics['train/duration_s']:.1f}s)")
        val_metrics = validate(model, val_loader, loss_fn, device)
        print(f"  val:   total={val_metrics['val/loss_total']:.4e}  t={val_metrics['val/loss_t']:.3e}  r={val_metrics['val/loss_r']:.3e}  ({val_metrics['val/duration_s']:.1f}s)")
        if scheduler is not None:
            scheduler.step()
        cur_val = val_metrics['val/loss_total']
        is_best = cur_val < best_val_loss
        if is_best:
            best_val_loss = cur_val
        all_metrics = {**train_metrics, **val_metrics}
        all_metrics['epoch'] = epoch
        all_metrics['lr'] = optimizer.param_groups[0]['lr']
        all_metrics['is_best'] = is_best
        all_metrics['epoch_duration_s'] = time.time() - epoch_start
        all_metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        logger.log(all_metrics)
        save_checkpoint(out_dir, epoch, model, optimizer, metrics=all_metrics, config=cfg, is_best=is_best)
        print(f'  saved checkpoint (best={is_best}, best_val={best_val_loss:.4e})')
    total_time = time.time() - t_global
    print(f'\nTraining done in {total_time / 60:.1f} min. Best val loss: {best_val_loss:.4e}. Output: {out_dir}')

if __name__ == '__main__':
    main()
