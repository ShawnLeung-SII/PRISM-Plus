#!/usr/bin/env python3
"""BND-only inference for legacy v9 (latpixdepth/pixel_branch_v9).

Same visual layout as the stage1 PRISMPlusBND vis (Spectral_r + GT-percentile
+ hole black painting) so v9 and v0.6.1 outputs are directly comparable.

Does NOT run NRG — only the BND mask + hole-region depth visualisation.
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
# Make latpixdepth.models.* importable
_LATPIX = '/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth'
if _LATPIX not in sys.path:
    sys.path.insert(0, _LATPIX)

from prism_plus.data import ByteCamDepthDataset
from prism_plus.utils.visualize import (
    _to_2d, _to_uint8_rgb, _colorize_depth, _depth_with_hole_mask,
    _tp_fp_fn_overlay,
)
from latpixdepth.models.dual_stream_pixel_branch_v9 import create_pixel_branch_v9


def save_v9_sample(out_dir: Path, sample_id: str,
                    rgb, sim_depth, real_depth, gt_mask,
                    pred_prob, pred_main, pred_detail, threshold=0.25):
    folder = out_dir / sample_id; folder.mkdir(parents=True, exist_ok=True)
    rgb_u8 = _to_uint8_rgb(rgb)
    sim_d  = _to_2d(sim_depth)
    real_d = _to_2d(real_depth)
    gt_m   = _to_2d(gt_mask)
    prob   = _to_2d(pred_prob)
    main_p = _to_2d(pred_main)
    det_p  = _to_2d(pred_detail)
    pred_m = (prob > threshold).astype(np.float32)

    valid = (1.0 - gt_m).astype(bool)
    if valid.any():
        vmin = float(np.nanpercentile(real_d[valid], 2))
        vmax = float(np.nanpercentile(real_d[valid], 98))
    else:
        vmin, vmax = float(real_d.min()), float(real_d.max())

    Image.fromarray(rgb_u8).save(folder / 'rgb.png')
    Image.fromarray(_colorize_depth(sim_d, vmin=vmin, vmax=vmax)).save(folder / 'sim_depth.png')
    Image.fromarray(_depth_with_hole_mask(real_d, gt_m,  vmin, vmax)).save(folder / 'real_depth_gt_hole.png')
    Image.fromarray(_depth_with_hole_mask(real_d, pred_m, vmin, vmax)).save(folder / 'real_depth_pred_hole.png')
    Image.fromarray((gt_m * 255).astype(np.uint8), mode='L').save(folder / 'gt_mask.png')
    Image.fromarray((pred_m * 255).astype(np.uint8), mode='L').save(folder / 'pred_mask.png')
    Image.fromarray(_tp_fp_fn_overlay(rgb_u8, pred_m, gt_m)).save(folder / 'overlay_tp_fp_fn.png')

    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(prob, cmap='hot', vmin=0, vmax=1); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(folder / 'pred_prob.png', dpi=120, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0,0].imshow(rgb_u8);                                              axes[0,0].set_title('RGB')
    axes[0,1].imshow(_colorize_depth(sim_d, vmin=vmin, vmax=vmax));        axes[0,1].set_title('Sim Depth')
    axes[0,2].imshow(_depth_with_hole_mask(real_d, gt_m,  vmin, vmax));    axes[0,2].set_title('Real Depth (GT hole)')
    axes[0,3].imshow(_depth_with_hole_mask(real_d, pred_m, vmin, vmax));   axes[0,3].set_title(f'Real Depth (Pred hole, t={threshold})')
    axes[1,0].imshow(gt_m, cmap='gray');                                   axes[1,0].set_title('GT Mask')
    axes[1,1].imshow(pred_m, cmap='gray');                                 axes[1,1].set_title(f'Pred Mask (t={threshold})')
    axes[1,2].imshow(prob,   cmap='hot', vmin=0, vmax=1);                  axes[1,2].set_title('pred_failure (final)')
    axes[1,3].imshow(_tp_fp_fn_overlay(rgb_u8, pred_m, gt_m));             axes[1,3].set_title('TP=G FP=R FN=B')
    for ax in axes.flat: ax.axis('off')
    fig.suptitle(f'{sample_id}  [v9 backbone]', fontsize=14); fig.tight_layout()
    fig.savefig(folder / 'grid.png', dpi=120, bbox_inches='tight'); plt.close(fig)


def main():
    p = argparse.ArgumentParser('v9 BND inference vis')
    p.add_argument('--ckpt', default='/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth/outputs/pixel_branch_v9/best.pt')
    p.add_argument('--vfm_checkpoint', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/weights/moge2.pt')
    p.add_argument('--data_root', default='/inspire/hdd/project/robot-dna/liangxiujian-253308390319/ByteCameraDepth')
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--resolution', type=int, default=512)         # v9 was trained at 512
    p.add_argument('--threshold', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/3] Building v9 model')
    model = create_pixel_branch_v9(
        vfm_type='moge2', vfm_checkpoint=args.vfm_checkpoint,
        encoder_size='large',
        use_semantic_guidance=True, deep_supervision=True,
    ).to(device).eval()

    print(f'[2/3] Loading {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f'  loaded: missing={len(miss)} unexpected={len(unexp)}')

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
            out = model(rgb, sim_d)
        prob = out['pred_failure'].float()
        main_p = out['pred_main'].float() if 'pred_main' in out else prob
        det_p  = out['pred_detail'].float() if 'pred_detail' in out else prob
        for b in range(rgb.size(0)):
            i = saved + b
            if i >= args.n_samples: break
            save_v9_sample(
                out_dir, f'sample_{i:03d}',
                rgb[b].cpu(), sim_d[b].cpu(), real_d[b].cpu(), hole[b].cpu(),
                prob[b].cpu(), main_p[b].cpu(), det_p[b].cpu(),
                threshold=args.threshold,
            )
        saved += rgb.size(0)
        if saved >= args.n_samples: break

    print(f'\nDone. {saved} samples -> {out_dir}')


if __name__ == '__main__':
    main()
