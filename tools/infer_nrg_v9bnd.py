#!/usr/bin/env python3
"""PRISM+ C2 — full BND+NRG pipeline using LEGACY v9 backbone.

Identical visualisation contract to infer_nrg_v2.py (stage1 colour scheme,
Spectral_r + GT-percentile + hole black painting, 3x4 grid with the full
PRISM tripartite story), but the BND density used as conditioning for the
NRG is produced by the legacy v9 dual-stream pixel branch instead of
PRISMPlusBND v0.6.1.

This is a direct A/B for the user — visual quality of the synthesised
depth in hole regions when conditioned on v9 mask vs v0.6.1 mask.

NOTE: v9 was trained at 512x512 — we run BND at 512 then downsample its
mask to NRG's native 256 to keep NRG behaviour identical to vis_v2.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LATPIX = '/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth'
if _LATPIX not in sys.path:
    sys.path.insert(0, _LATPIX)

from prism_plus.data import ByteCamDepthDataset
from prism_plus.models import NRGStandalone
from prism_plus.utils.visualize import (
    _to_2d, _to_uint8_rgb, _colorize_depth, _depth_with_hole_mask,
    _tp_fp_fn_overlay,
)
from latpixdepth.models.dual_stream_pixel_branch_v9 import create_pixel_branch_v9


def save_nrg_sample(out_dir, sample_id, rgb, sim_depth, real_depth, gt_mask,
                     pred_prob, nrg_depth, threshold=0.5, label=''):
    folder = out_dir / sample_id; folder.mkdir(parents=True, exist_ok=True)
    rgb_u8 = _to_uint8_rgb(rgb)
    sim_d  = _to_2d(sim_depth)
    real_d = _to_2d(real_depth)
    gt_m   = _to_2d(gt_mask)
    prob   = _to_2d(pred_prob)
    pred_m = (prob > threshold).astype(np.float32)
    nrg_d  = _to_2d(nrg_depth)

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
    res = nrg_d - real_d
    res_v = float(np.nanpercentile(np.abs(res[valid]), 98)) if valid.any() else 1.0
    axes[2,3].imshow(res, cmap='seismic', vmin=-res_v, vmax=res_v);          axes[2,3].set_title('NRG - Real (residual)')
    for ax in axes.flat: ax.axis('off')
    fig.suptitle(f'{sample_id}  [{label}]', fontsize=14); fig.tight_layout()
    fig.savefig(folder / 'grid.png', dpi=120, bbox_inches='tight'); plt.close(fig)


def main():
    p = argparse.ArgumentParser('PRISM+ C2 vis with v9 BND + NRG combo')
    p.add_argument('--v9_ckpt', default='/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth/outputs/pixel_branch_v9/best.pt')
    p.add_argument('--nrg_ckpt', required=True)
    p.add_argument('--data_root', default='/inspire/hdd/project/robot-dna/liangxiujian-253308390319/ByteCameraDepth')
    p.add_argument('--vfm_checkpoint', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/weights/moge2.pt')
    p.add_argument('--sd_checkpoint', default='/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth/latpixdepth/checkpoints/Latent_SD.ckpt')
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--num_steps', type=int, default=50)
    p.add_argument('--bnd_res', type=int, default=512)     # v9 trained at 512
    p.add_argument('--nrg_res', type=int, default=256)     # NRG native res
    p.add_argument('--threshold', type=float, default=0.25)  # v9 default
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/4] Loading v9 BND from {args.v9_ckpt}')
    bnd = create_pixel_branch_v9(
        vfm_type='moge2', vfm_checkpoint=args.vfm_checkpoint,
        encoder_size='large',
        use_semantic_guidance=True, deep_supervision=True,
    ).to(device).eval()
    ck = torch.load(args.v9_ckpt, map_location=device, weights_only=False)
    sd = ck.get('model_state_dict', ck.get('model', ck))
    m, u = bnd.load_state_dict(sd, strict=False)
    print(f'  v9 loaded: missing={len(m)} unexpected={len(u)}')

    print(f'[2/4] Loading NRGStandalone from {args.nrg_ckpt}')
    nrg = NRGStandalone(
        sd_checkpoint=args.sd_checkpoint,
        residual_scale=2.0, lambda_boundary=0.3,
        freeze_vae=True, freeze_unet=False, freeze_text=True,
    ).to(device).eval()
    ck = torch.load(args.nrg_ckpt, map_location=device, weights_only=False)
    nrg.load_state_dict(ck.get('model', ck), strict=False)

    print(f'[3/4] Dataset (n={args.n_samples}, bnd_res={args.bnd_res}, nrg_res={args.nrg_res})')
    ds = ByteCamDepthDataset(data_root=args.data_root, split='val',
                              resolution=args.bnd_res, augment=False)
    rng = np.random.RandomState(args.seed)
    idx = sorted(rng.choice(len(ds), min(args.n_samples, len(ds)), replace=False).tolist())
    loader = DataLoader(Subset(ds, idx), batch_size=2, shuffle=False, num_workers=2)

    print(f'[4/4] Inference -> {out_dir}')
    saved = 0
    for batch in tqdm(loader):
        rgb_hi   = batch['rgb'].to(device)            # bnd_res
        sim_hi   = batch['sim_depth'].to(device)
        real_hi  = batch['real_depth'].to(device)
        hole_hi  = batch['hole_mask'].to(device)

        with torch.no_grad():
            v9_out = bnd(rgb_hi, sim_hi)
            density_hi = v9_out['pred_failure'].float()    # [B,1,bnd_res,bnd_res]

        # Downsample to NRG resolution
        rgb_lo     = F.interpolate(rgb_hi,    size=args.nrg_res, mode='bilinear', align_corners=False)
        sim_lo     = F.interpolate(sim_hi,    size=args.nrg_res, mode='nearest')
        real_lo    = F.interpolate(real_hi,   size=args.nrg_res, mode='nearest')
        hole_lo    = F.interpolate(hole_hi,   size=args.nrg_res, mode='nearest')
        density_lo = F.interpolate(density_hi, size=args.nrg_res, mode='bilinear', align_corners=False).clamp(0, 1)

        with torch.no_grad():
            d_pred = nrg.sample(sim_depth=sim_lo, mask_density=density_lo, num_steps=args.num_steps)

        for b in range(rgb_lo.size(0)):
            i = saved + b
            if i >= args.n_samples: break
            save_nrg_sample(
                out_dir, f'sample_{i:03d}',
                rgb_lo[b].cpu(), sim_lo[b].cpu(),
                real_lo[b].cpu(), hole_lo[b].cpu(),
                density_lo[b].cpu(), d_pred[b].cpu(),
                threshold=args.threshold,
                label=f'v9 BND ({args.bnd_res}->{args.nrg_res}) + NRG',
            )
        saved += rgb_lo.size(0)
        if saved >= args.n_samples: break

    print(f'\nDone. {saved} samples -> {out_dir}')


if __name__ == '__main__':
    main()
