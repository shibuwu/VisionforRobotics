import argparse, importlib, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data_v4 import VIODatasetV4
from losses_v4 import MultiTaskLoss
from models_v4 import VIOModelV4
from encoder_inertial import InertialEncoder
from encoder_vision import VisionEncoder
from encoder_fusion import FusionEncoder
from encoder_vision_v2 import VisionEncoderV2
from encoder_fusion_v2 import FusionEncoderV2
from checkpoint import save_checkpoint, load_checkpoint
from logger import JsonlLogger

ENCODER_REGISTRY = {'inertial': InertialEncoder, 'vision': VisionEncoder, 'fusion': FusionEncoder, 'vision_v2': VisionEncoderV2, 'fusion_v2': FusionEncoderV2}

def build_encoder(encoder_type, kwargs):
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f'Unknown encoder_type {encoder_type!r}')
    return ENCODER_REGISTRY[encoder_type](**kwargs)

def build_split(data_root, val_ratio, split_seed):
    root = Path(data_root)
    all_seqs = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith('seq_')])
    if val_ratio >= len(all_seqs):
        raise ValueError(f'val_ratio={val_ratio} >= len(all_seqs)={len(all_seqs)}')
    train_seqs = all_seqs[:-val_ratio]
    val_seqs = all_seqs[-val_ratio:]
    return (train_seqs, val_seqs)

def build_optimizer(name, params, lr, weight_decay):
    if name.lower() == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name.lower() == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unknown optimizer {name!r}')

def build_scheduler(name, optimizer, total_epochs, min_lr):
    if name is None:
        return None
    if name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
    raise ValueError(f'Unknown scheduler {name!r}')

def train_one_epoch_v4(model, loader, loss_fn, optimizer, device, grad_clip=None, log_every=0):
    model.train()
    total = 0.0
    n = 0
    sum_t = 0.0
    sum_r = 0.0
    sum_h = 0.0
    t0 = time.time()
    for (step, batch) in enumerate(loader):
        batch_dev = {'img_pair': batch['img_pair'].to(device, non_blocking=True), 'imu': batch['imu'].to(device, non_blocking=True)}
        target = {'delta_t': batch['delta_t'].to(device, non_blocking=True), 'delta_q': batch['delta_q'].to(device, non_blocking=True), 'h4pt_gt': batch['h4pt_gt'].to(device, non_blocking=True)}
        pred = model(batch_dev)
        (loss, components) = loss_fn(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += components['loss_total']
        sum_t += components['loss_t']
        sum_r += components['loss_r']
        sum_h += components['loss_homo']
        n += 1
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}: total={components['loss_total']:.4e}  t={components['loss_t']:.3e}  r={components['loss_r']:.3e}  h={components['loss_homo']:.3e}")
    return {'train/loss_total': total / max(n, 1), 'train/loss_t': sum_t / max(n, 1), 'train/loss_r': sum_r / max(n, 1), 'train/loss_homo': sum_h / max(n, 1), 'train/duration_s': time.time() - t0, 'train/n_batches': n}

@torch.no_grad()
def validate_v4(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    sum_t = 0.0
    sum_r = 0.0
    sum_h = 0.0
    t0 = time.time()
    for batch in loader:
        batch_dev = {'img_pair': batch['img_pair'].to(device, non_blocking=True), 'imu': batch['imu'].to(device, non_blocking=True)}
        target = {'delta_t': batch['delta_t'].to(device, non_blocking=True), 'delta_q': batch['delta_q'].to(device, non_blocking=True), 'h4pt_gt': batch['h4pt_gt'].to(device, non_blocking=True)}
        pred = model(batch_dev)
        (_, components) = loss_fn(pred, target)
        total += components['loss_total']
        sum_t += components['loss_t']
        sum_r += components['loss_r']
        sum_h += components['loss_homo']
        n += 1
    return {'val/loss_total': total / max(n, 1), 'val/loss_t': sum_t / max(n, 1), 'val/loss_r': sum_r / max(n, 1), 'val/loss_homo': sum_h / max(n, 1), 'val/duration_s': time.time() - t0, 'val/n_batches': n}

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
    train_ds = VIODatasetV4(cfg['data_root'], frame_gap=cfg['frame_gap'], image_size=cfg['image_size'], mean_path=cfg['mean_path'], std_path=cfg['std_path'], sequences=train_seqs)
    val_ds = VIODatasetV4(cfg['data_root'], frame_gap=cfg['frame_gap'], image_size=cfg['image_size'], mean_path=cfg['mean_path'], std_path=cfg['std_path'], sequences=val_seqs)
    print(f'len(train_ds)={len(train_ds)}, len(val_ds)={len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=device.type == 'cuda', persistent_workers=cfg['num_workers'] > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=device.type == 'cuda', persistent_workers=cfg['num_workers'] > 0)
    encoder = build_encoder(cfg['encoder_type'], cfg['encoder_kwargs'])
    model = VIOModelV4(encoder, feature_dim=encoder.feature_dim).to(device)
    n_params = sum((p.numel() for p in model.parameters()))
    print(f"Model: encoder={cfg['encoder_type']}, total params={n_params:,}")
    loss_fn = MultiTaskLoss(lambda_rot=cfg.get('lambda_rot', 10.0), lambda_homo=cfg.get('lambda_homo', 1.0), h4pt_scale=cfg.get('h4pt_scale', 400.0))
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
    print(f"\nStarting V4 training: epochs {start_epoch} -> {cfg['epochs']}")
    t_global = time.time()
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        print(f"\n=== Epoch {epoch}/{cfg['epochs']} ===")
        epoch_start = time.time()
        train_metrics = train_one_epoch_v4(model, train_loader, loss_fn, optimizer, device, grad_clip=cfg.get('grad_clip'), log_every=cfg.get('log_every_n_steps', 0))
        print(f"  train: total={train_metrics['train/loss_total']:.4e}  t={train_metrics['train/loss_t']:.3e}  r={train_metrics['train/loss_r']:.3e}  h={train_metrics['train/loss_homo']:.3e}  ({train_metrics['train/duration_s']:.1f}s)")
        val_metrics = validate_v4(model, val_loader, loss_fn, device)
        print(f"  val:   total={val_metrics['val/loss_total']:.4e}  t={val_metrics['val/loss_t']:.3e}  r={val_metrics['val/loss_r']:.3e}  h={val_metrics['val/loss_homo']:.3e}  ({val_metrics['val/duration_s']:.1f}s)")
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
    print(f'\nV4 training done in {(time.time() - t_global) / 60:.1f} min. Best val loss: {best_val_loss:.4e}. Output: {out_dir}')

if __name__ == '__main__':
    main()
