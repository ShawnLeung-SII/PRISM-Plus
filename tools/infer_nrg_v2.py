#!/usr/bin/env python3
"""PRISM+ C2 — generate visualisation samples from a trained NRGStandalone.

Usage:
    python tools/infer_nrg_v2.py \
        --bnd_ckpt /.../stage1_v6/best.pt \
        --nrg_ckpt /.../stage2a_nrg/best.pt \
        --data_root /inspire/hdd/.../ByteCameraDepth \
        --n_samples 100 --num_steps 50 \
        --output_dir /.../stage2a_nrg/vis
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data import ByteCamDepthDataset
from prism_plus.models import create_bnd_plus, NRGStandalone


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser('PRISM+ C2 NRG inference visualiser')
    p.add_argument('--bnd_ckpt', required=True)
    p.add_argument('--nrg_ckpt', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--vfm_checkpoint', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/weights/moge2.pt')
    p.add_argument('--sd_checkpoint', default='/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth/latpixdepth/checkpoints/Latent_SD.ckpt')
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--num_steps', type=int, default=50)
    p.add_argument('--guidance', type=float, default=1.0)
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    grid_dir = out_dir / 'grids'
    raw_dir  = out_dir / 'raw'
    grid_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/4] Loading frozen BND v0.6.1 from {args.bnd_ckpt}')
    bnd = create_bnd_plus(
        vfm_type='moge2', vfm_checkpoint=args.vfm_checkpoint,
        encoder_size='large', deep_supervision=True,
        vfm_cross_attn_dim=128, boundary_band_radius=3,
    ).to(device).eval()
    sd = torch.load(args.bnd_ckpt, map_location=device, weights_only=False)
    bnd.load_state_dict(sd.get('model', sd), strict=False)
    for p_ in bnd.parameters():
        p_.requires_grad = False

    print(f'[2/4] Loading NRGStandalone from {args.nrg_ckpt}')
    nrg = NRGStandalone(
        sd_checkpoint=args.sd_checkpoint,
        residual_scale=2.0, lambda_boundary=0.3,
        boundary_radius=5, freeze_vae=True, freeze_unet=False, freeze_text=True,
    ).to(device).eval()
    sd = torch.load(args.nrg_ckpt, map_location=device, weights_only=False)
    nrg.load_state_dict(sd.get('model', sd), strict=False)

    print(f'[3/4] Building dataset val split (n_samples={args.n_samples})')
    ds = ByteCamDepthDataset(data_root=args.data_root, split='val',
                              resolution=args.resolution, augment=False)
    rng = np.random.RandomState(args.seed)
    if len(ds) > args.n_samples:
        idx = sorted(rng.choice(len(ds), args.n_samples, replace=False).tolist())
    else:
        idx = list(range(len(ds)))
    loader = DataLoader(Subset(ds, idx), batch_size=4, shuffle=False, num_workers=2)

    print(f'[4/4] Inference -> {out_dir}')
    saved = 0
    for batch in tqdm(loader):
        rgb   = batch['rgb'].to(device)
        sim_d = batch['sim_depth'].to(device)
        real_d= batch['real_depth'].to(device)
        hole  = batch['hole_mask'].to(device)
        with torch.no_grad():
            bnd_out = bnd(rgb, sim_d)
            density = bnd_out['pred_failure'].float()
            d_pred = nrg.sample(sim_depth=sim_d, mask_density=density,
                                 num_steps=args.num_steps)
        for b in range(rgb.size(0)):
            i = saved + b
            if i >= args.n_samples: break
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes[0,0].imshow(_to_np(rgb[b]).transpose(1,2,0)); axes[0,0].set_title('RGB'); axes[0,0].axis('off')
            axes[0,1].imshow(_to_np(sim_d[b,0]), cmap='viridis'); axes[0,1].set_title('sim_depth (clean)'); axes[0,1].axis('off')
            axes[0,2].imshow(_to_np(real_d[b,0]), cmap='viridis'); axes[0,2].set_title('real_depth (noisy GT)'); axes[0,2].axis('off')
            axes[1,0].imshow(_to_np(density[b,0]), cmap='hot', vmin=0, vmax=1); axes[1,0].set_title('BND density'); axes[1,0].axis('off')
            axes[1,1].imshow(_to_np(hole[b,0]), cmap='gray', vmin=0, vmax=1); axes[1,1].set_title('hole_mask GT'); axes[1,1].axis('off')
            axes[1,2].imshow(_to_np(d_pred[b,0]), cmap='viridis'); axes[1,2].set_title('NRG synthesized depth'); axes[1,2].axis('off')
            plt.tight_layout()
            fig.savefig(grid_dir / f'sample_{i:03d}.png', dpi=120)
            plt.close(fig)
            np.savez_compressed(raw_dir / f'sample_{i:03d}.npz',
                rgb=_to_np(rgb[b]), sim_depth=_to_np(sim_d[b,0]),
                real_depth=_to_np(real_d[b,0]), hole_mask=_to_np(hole[b,0]),
                density=_to_np(density[b,0]), nrg_depth=_to_np(d_pred[b,0]))
        saved += rgb.size(0)
        if saved >= args.n_samples: break

    print(f'\nDone. {saved} samples saved to {grid_dir}')


if __name__ == '__main__':
    main()
