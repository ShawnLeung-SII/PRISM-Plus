#!/usr/bin/env python3
"""PRISM+ C2 — NRG visualisation, aligned with stage1 visualisation scheme.

Stage1 conventions (from prism_plus/utils/visualize.py):
    * depth     -> matplotlib Spectral_r
    * pred_prob -> hot (vmin=0, vmax=1)
    * mask      -> gray binary
    * vmin/vmax = percentile 2/98 of the *GT-valid* region of real_depth
    * hole pixels are painted BLACK on the depth visualisation

NRG additions (PRISM tripartite pipeline output):
    * nrg_depth                = NRG synthesised metric depth
    * nrg_depth_bnd_hole       = nrg_depth with BND mask painted black
                                  (== final PRISM-synthesised noisy depth)
    * nrg_depth_gt_hole        = nrg_depth with GT mask painted black
                                  (for direct visual A/B with real_depth_gt_hole)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data import ByteCamDepthDataset
from prism_plus.models import create_bnd_plus, NRGStandalone
from prism_plus.utils.visualize import (
    _to_2d, _to_uint8_rgb, _colorize_depth, _depth_with_hole_mask,
    _tp_fp_fn_overlay,
)


def save_nrg_sample(
    out_dir: Path,
    sample_id: str,
    rgb: torch.Tensor,
    sim_depth: torch.Tensor,
    real_depth: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_prob: torch.Tensor,            # BND density [0,1]
    nrg_depth: torch.Tensor,            # NRG synthesised depth (metres)
    threshold: float = 0.5,
):
    folder = out_dir / sample_id
    folder.mkdir(parents=True, exist_ok=True)

    rgb_u8 = _to_uint8_rgb(rgb)
    sim_d  = _to_2d(sim_depth)
    real_d = _to_2d(real_depth)
    gt_m   = _to_2d(gt_mask)
    prob   = _to_2d(pred_prob)
    pred_m = (prob > threshold).astype(np.float32)
    nrg_d  = _to_2d(nrg_depth)

    # vmin/vmax from GT-valid region of real_depth (stage1 convention)
    valid = (1.0 - gt_m).astype(bool)
    if valid.any():
        vmin = float(np.nanpercentile(real_d[valid], 2))
        vmax = float(np.nanpercentile(real_d[valid], 98))
    else:
        vmin, vmax = float(real_d.min()), float(real_d.max())

    Image.fromarray(rgb_u8).save(folder / 'rgb.png')
    Image.fromarray(_colorize_depth(sim_d,  vmin=vmin, vmax=vmax)).save(folder / 'sim_depth.png')
    Image.fromarray(_depth_with_hole_mask(real_d, gt_m,  vmin, vmax)).save(folder / 'real_depth_gt_hole.png')
    Image.fromarray(_depth_with_hole_mask(real_d, pred_m, vmin, vmax)).save(folder / 'real_depth_pred_hole.png')
    Image.fromarray(_depth_with_hole_mask(nrg_d,  gt_m,  vmin, vmax)).save(folder / 'nrg_depth_gt_hole.png')
    Image.fromarray(_depth_with_hole_mask(nrg_d,  pred_m, vmin, vmax)).save(folder / 'nrg_depth_bnd_hole.png')
    Image.fromarray((gt_m  * 255).astype(np.uint8), mode='L').save(folder / 'gt_mask.png')
    Image.fromarray((pred_m * 255).astype(np.uint8), mode='L').save(folder / 'pred_mask.png')

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(prob, cmap='hot', vmin=0, vmax=1); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(folder / 'pred_prob.png', dpi=120, bbox_inches='tight'); plt.close(fig)
    Image.fromarray(_tp_fp_fn_overlay(rgb_u8, pred_m, gt_m)).save(folder / 'overlay_tp_fp_fn.png')

    # Final summary grid (3x4, the PRISM-tripartite story)
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes[0,0].imshow(rgb_u8);                                                axes[0,0].set_title('RGB')
    axes[0,1].imshow(_colorize_depth(sim_d, vmin=vmin, vmax=vmax));          axes[0,1].set_title('Sim Depth (clean)')
    axes[0,2].imshow(_depth_with_hole_mask(real_d, gt_m,  vmin, vmax));      axes[0,2].set_title('Real Depth (GT hole)')
    axes[0,3].imshow(_depth_with_hole_mask(real_d, pred_m, vmin, vmax));     axes[0,3].set_title('Real Depth (Pred hole)')
    axes[1,0].imshow(gt_m,  cmap='gray');                                    axes[1,0].set_title('GT Mask')
    axes[1,1].imshow(pred_m, cmap='gray');                                   axes[1,1].set_title(f'Pred Mask (t={threshold})')
    axes[1,2].imshow(prob,  cmap='hot', vmin=0, vmax=1);                     axes[1,2].set_title('BND density')
    axes[1,3].imshow(_tp_fp_fn_overlay(rgb_u8, pred_m, gt_m));               axes[1,3].set_title('TP=G FP=R FN=B')
    axes[2,0].imshow(_colorize_depth(nrg_d, vmin=vmin, vmax=vmax));          axes[2,0].set_title('NRG synth (raw)')
    axes[2,1].imshow(_depth_with_hole_mask(nrg_d, gt_m,  vmin, vmax));       axes[2,1].set_title('NRG synth (GT hole)')
    axes[2,2].imshow(_depth_with_hole_mask(nrg_d, pred_m, vmin, vmax));      axes[2,2].set_title('NRG synth (BND hole)')
    # Residual visualisation: nrg vs real_GT
    res = nrg_d - real_d
    res_v = np.nanpercentile(np.abs(res[valid]), 98) if valid.any() else 1.0
    axes[2,3].imshow(res, cmap='seismic', vmin=-res_v, vmax=res_v);          axes[2,3].set_title('NRG - Real (residual)')
    for ax in axes.flat: ax.axis('off')
    fig.suptitle(sample_id, fontsize=14); fig.tight_layout()
    fig.savefig(folder / 'grid.png', dpi=120, bbox_inches='tight'); plt.close(fig)


def main():
    p = argparse.ArgumentParser('PRISM+ C2 NRG inference viz (stage1-aligned)')
    p.add_argument('--bnd_ckpt', required=True)
    p.add_argument('--nrg_ckpt', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--vfm_checkpoint', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/weights/moge2.pt')
    p.add_argument('--sd_checkpoint', default='/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth/latpixdepth/checkpoints/Latent_SD.ckpt')
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--num_steps', type=int, default=50)
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/3] Loading frozen BND (v0.6.1) from {args.bnd_ckpt}')
    bnd = create_bnd_plus(
        vfm_type='moge2', vfm_checkpoint=args.vfm_checkpoint,
        encoder_size='large', deep_supervision=True,
        vfm_cross_attn_dim=128, boundary_band_radius=3,
    ).to(device).eval()
    sd = torch.load(args.bnd_ckpt, map_location=device, weights_only=False)
    bnd.load_state_dict(sd.get('model', sd), strict=False)
    for q in bnd.parameters(): q.requires_grad = False

    print(f'[2/3] Loading NRGStandalone from {args.nrg_ckpt}')
    nrg = NRGStandalone(
        sd_checkpoint=args.sd_checkpoint,
        residual_scale=2.0, lambda_boundary=0.3,
        freeze_vae=True, freeze_unet=False, freeze_text=True,
    ).to(device).eval()
    sd = torch.load(args.nrg_ckpt, map_location=device, weights_only=False)
    nrg.load_state_dict(sd.get('model', sd), strict=False)

    print(f'[3/3] Inference -> {out_dir}')
    ds = ByteCamDepthDataset(data_root=args.data_root, split='val',
                              resolution=args.resolution, augment=False)
    rng = np.random.RandomState(args.seed)
    idx = sorted(rng.choice(len(ds), min(args.n_samples, len(ds)), replace=False).tolist())
    loader = DataLoader(Subset(ds, idx), batch_size=4, shuffle=False, num_workers=2)

    saved = 0
    for batch in tqdm(loader):
        rgb   = batch['rgb'].to(device)
        sim_d = batch['sim_depth'].to(device)
        real_d= batch['real_depth'].to(device)
        hole  = batch['hole_mask'].to(device)
        with torch.no_grad():
            density = bnd(rgb, sim_d)['pred_failure'].float()
            d_pred  = nrg.sample(sim_depth=sim_d, mask_density=density, num_steps=args.num_steps)
        for b in range(rgb.size(0)):
            i = saved + b
            if i >= args.n_samples: break
            save_nrg_sample(
                out_dir=out_dir, sample_id=f'sample_{i:03d}',
                rgb=rgb[b].cpu(), sim_depth=sim_d[b].cpu(),
                real_depth=real_d[b].cpu(), gt_mask=hole[b].cpu(),
                pred_prob=density[b].cpu(), nrg_depth=d_pred[b].cpu(),
                threshold=args.threshold,
            )
        saved += rgb.size(0)
        if saved >= args.n_samples: break

    print(f'\nDone. {saved} samples -> {out_dir}')


if __name__ == '__main__':
    main()
