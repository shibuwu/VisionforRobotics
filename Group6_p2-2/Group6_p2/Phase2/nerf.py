import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from till_64_points import load_nerf_data, get_rays, sample_points_along_rays
from skimage.metrics import structural_similarity as ssim



class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.freqs = 2.0 ** torch.arange(L, dtype=torch.float32)

    def forward(self, x):
        out = [x]                          
        for f in self.freqs.to(x.device):
            out += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    def __init__(self, no_pe=False):
        super().__init__()
        if no_pe:
            pos_dim = 3; dir_dim = 3
            self.pe_x = nn.Identity()
            self.pe_d = nn.Identity()
        else:
            pos_dim = 63; dir_dim = 27
            self.pe_x = PositionalEncoding(L=10)
            self.pe_d = PositionalEncoding(L=4)

        D = 256  
        self.fc1 = nn.Sequential(               
            nn.Linear(pos_dim, D), nn.ReLU(True),
            nn.Linear(D, D),  nn.ReLU(True),
            nn.Linear(D, D),  nn.ReLU(True),
            nn.Linear(D, D),  nn.ReLU(True),
        )
        self.fc2 = nn.Sequential(               
            nn.Linear(D + pos_dim, D), nn.ReLU(True),
            nn.Linear(D, D), nn.ReLU(True),
            nn.Linear(D, D), nn.ReLU(True),
            nn.Linear(D, D), nn.ReLU(True),
        )
        self.sigma_head  = nn.Linear(D, 1)
        self.no_pe = no_pe
        if no_pe:
            nn.init.constant_(self.sigma_head.bias, 0.1)
        self.feat_head   = nn.Linear(D, D)
        self.colour_head = nn.Sequential(
            nn.Linear(D + dir_dim, 128), nn.ReLU(True),
            nn.Linear(128, 3), nn.Sigmoid()
        )

    def forward(self, pts, dirs):
        ex = self.pe_x(pts)
        ed = self.pe_d(dirs)
        h  = self.fc1(ex)
        h  = self.fc2(torch.cat([h, ex], dim=-1))
        sigma = F.softplus(self.sigma_head(h)) if self.no_pe else F.relu(self.sigma_head(h))
        rgb   = self.colour_head(torch.cat([self.feat_head(h), ed], dim=-1))
        return rgb, sigma



def volume_render(rgb, sigma, t_vals, white_bkgd=True):
    deltas  = torch.cat([t_vals[..., 1:] - t_vals[..., :-1],
                         torch.full_like(t_vals[..., :1], 1e10)], dim=-1)
    alpha   = 1 - torch.exp(-sigma[..., 0] * deltas)
    T       = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                       1 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = T * alpha
    rgb_map = (weights[..., None] * rgb).sum(-2)
    if white_bkgd:
        rgb_map = rgb_map + (1 - weights.sum(-1, keepdim=True))
    return rgb_map, weights



def sample_fine(t_coarse, weights, N_fine):
    N, N_c  = t_coarse.shape
    weights = weights + 1e-5
    pdf     = weights / weights.sum(-1, keepdim=True)
    cdf     = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], dim=-1)
    u       = torch.rand(N, N_fine, device=t_coarse.device).contiguous()
    inds    = torch.searchsorted(cdf.contiguous(), u, right=True)
    lo = (inds - 1).clamp(0, N_c - 1);  hi = inds.clamp(0, N_c - 1)
    cdf_lo  = torch.gather(cdf, 1, lo);  cdf_hi = torch.gather(cdf, 1, hi)
    t_lo    = torch.gather(t_coarse, 1, lo);  t_hi = torch.gather(t_coarse, 1, hi)
    denom   = (cdf_hi - cdf_lo).clamp(min=1e-5)
    t_fine  = t_lo + (u - cdf_lo) / denom * (t_hi - t_lo)
    t_all, _ = torch.sort(torch.cat([t_coarse, t_fine.detach()], dim=-1), dim=-1)
    return t_all



def render_rays(coarse, fine, rays_o, rays_d, near, far, N_c=64, N_f=128, perturb=True):
    N = rays_o.shape[0]

    t_c = torch.linspace(near, far, N_c, device=rays_o.device).expand(N, N_c).clone()
    if perturb:
        t_c = t_c + torch.rand_like(t_c) * (far - near) / N_c
    pts_c = rays_o[:, None] + t_c[:, :, None] * rays_d[:, None]
    d_c   = rays_d[:, None].expand_as(pts_c)

    rgb_c, sig_c = coarse(pts_c.reshape(-1, 3), d_c.reshape(-1, 3))
    rgb_c = rgb_c.reshape(N, N_c, 3);  sig_c = sig_c.reshape(N, N_c, 1)
    rgb_map_c, w_c = volume_render(rgb_c, sig_c, t_c)

    t_f   = sample_fine(t_c, w_c.detach(), N_f)
    pts_f = rays_o[:, None] + t_f[:, :, None] * rays_d[:, None]
    d_f   = rays_d[:, None].expand_as(pts_f)
    N_tot = t_f.shape[1]

    rgb_f, sig_f = fine(pts_f.reshape(-1, 3), d_f.reshape(-1, 3))
    rgb_f = rgb_f.reshape(N, N_tot, 3);  sig_f = sig_f.reshape(N, N_tot, 1)
    rgb_map_f, _ = volume_render(rgb_f, sig_f, t_f)

    return rgb_map_c, rgb_map_f



class RayDataset(Dataset):
    def __init__(self, images, poses, focal):
        H, W = images.shape[1:3]
        os, ds, cs = [], [], []
        for img, pose in zip(images, poses):
            ro, rd = get_rays(H, W, focal, pose)
            os.append(ro.reshape(-1, 3));  ds.append(rd.reshape(-1, 3))
            cs.append(img.reshape(-1, 3))
        self.rays_o = torch.from_numpy(np.concatenate(os))
        self.rays_d = torch.from_numpy(np.concatenate(ds))
        self.rgb    = torch.from_numpy(np.concatenate(cs))

    def __len__(self): return len(self.rays_o)
    def __getitem__(self, i): return self.rays_o[i], self.rays_d[i], self.rgb[i]



def train(scene="lego", half_res=True, max_steps=300_000, save_every=10_000, no_pe=False):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = os.path.join(os.path.dirname(__file__), "nerf_synthetic", scene)
    suffix    = "_nope" if no_pe else ""
    ckpt_dir  = os.path.join(os.path.dirname(__file__), "checkpoints", scene + suffix)
    log_path  = os.path.join(ckpt_dir, "train_log.csv")
    os.makedirs(ckpt_dir, exist_ok=True)

    images, poses, focal = load_nerf_data(data_dir, "train", half_res)
    print(f"Loaded {len(images)} images {images.shape[1]}x{images.shape[2]}, focal={focal:.1f}")

    loader  = DataLoader(RayDataset(images, poses, focal),
                         batch_size=4096, shuffle=True, num_workers=0, pin_memory=True)
    coarse  = NeRF(no_pe=no_pe).to(device)
    fine    = NeRF(no_pe=no_pe).to(device)
    opt     = torch.optim.Adam(list(coarse.parameters()) + list(fine.parameters()), lr=5e-4)
    sched   = torch.optim.lr_scheduler.ExponentialLR(opt, (5e-5 / 5e-4) ** (1 / max_steps))

    import csv
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "psnr"])

    step = 0
    while step < max_steps:
        for ro, rd, gt in loader:
            if step >= max_steps: break
            ro, rd, gt = ro.to(device), rd.to(device), gt.to(device)
            rc, rf = render_rays(coarse, fine, ro, rd, near=2.0, far=6.0)
            loss   = F.mse_loss(rc, gt) + F.mse_loss(rf, gt)   # Eq. 6
            opt.zero_grad(); loss.backward()
            opt.step(); sched.step()
            step += 1
            if step % 1000 == 0:
                psnr = -10 * torch.log10(F.mse_loss(rf, gt).detach())
                print(f"step {step}  loss={loss.item():.4f}  PSNR={psnr.item():.2f}")
                log_writer.writerow([step, f"{loss.item():.6f}", f"{psnr.item():.2f}"])
                log_file.flush()
            if step % save_every == 0:
                p = os.path.join(ckpt_dir, f"ckpt_{step:06d}.pt")
                torch.save({"step": step, "coarse": coarse.state_dict(),
                             "fine": fine.state_dict(), "opt": opt.state_dict()}, p)
                print(f"saved {p}")



@torch.no_grad()
def evaluate(scene="lego", ckpt_path="", half_res=True, no_pe=False):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.path.dirname(__file__), "nerf_synthetic", scene)
    coarse   = NeRF(no_pe=no_pe).to(device); fine = NeRF(no_pe=no_pe).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    coarse.load_state_dict(ck["coarse"]); fine.load_state_dict(ck["fine"])
    coarse.eval(); fine.eval()

    images, poses, focal = load_nerf_data(data_dir, "test", half_res)
    H, W  = images.shape[1:3]
    suffix = "_nope" if no_pe else ""
    save  = os.path.join(os.path.dirname(__file__), "renders", scene + suffix)
    os.makedirs(save, exist_ok=True)

    psnrs, ssims = [], []
    for i, (gt, pose) in enumerate(zip(images, poses)):
        ro, rd = get_rays(H, W, focal, pose)
        ro = torch.from_numpy(ro.reshape(-1, 3)).to(device)
        rd = torch.from_numpy(rd.reshape(-1, 3)).to(device)
        chunks = []
        for j in range(0, ro.shape[0], 4096):
            _, rf = render_rays(coarse, fine, ro[j:j+4096], rd[j:j+4096], 2.0, 6.0, perturb=False)
            chunks.append(rf.cpu())
        pred = torch.cat(chunks).reshape(H, W, 3).numpy().clip(0, 1)
        gt3 = gt[..., :3]  # drop alpha if present
        psnr = -10 * np.log10(np.mean((pred - gt3) ** 2) + 1e-8)
        s = ssim(pred, gt3, data_range=1.0, channel_axis=-1)
        psnrs.append(psnr); ssims.append(s)
        Image.fromarray((pred * 255).astype(np.uint8)).save(
            os.path.join(save, f"render_{i:04d}.png"))
        print(f"[{i+1}/{len(images)}] PSNR={psnr:.2f}  SSIM={s:.4f}")
    print(f"Mean PSNR={np.mean(psnrs):.2f}  Mean SSIM={np.mean(ssims):.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scene",    default="lego")
    p.add_argument("--half_res", action="store_true", default=True)
    p.add_argument("--eval",     action="store_true")
    p.add_argument("--ckpt",     default="")
    p.add_argument("--no_pe",    action="store_true")
    args = p.parse_args()
    if args.eval:
        evaluate(args.scene, args.ckpt, args.half_res, no_pe=args.no_pe)
    else:
        train(args.scene, args.half_res, no_pe=args.no_pe)
