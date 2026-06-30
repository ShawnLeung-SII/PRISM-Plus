#!/usr/bin/env python3
"""PRISM+ C4 — TNSM temporal vis: BND-only vs BND+TNSM (fused logits).

Per T-frame window we emit one combined grid:
    Row 0 : RGB[t0..t3]
    Row 1 : BND-only density[t0..t3]
    Row 2 : TNSM-fused density[t0..t3]    (BND + TNSM proj_out)
    Row 3 : GT mask[t0..t3]
The visual story is whether TNSM reduces inter-frame flicker.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data.dreds import DREDSDataset
from prism_plus.data.temporal_window import TemporalWindow
from prism_plus.models import create_bnd_plus, TNSM, RAFTFlow
from prism_plus.metrics_temporal import temporal_noise_flicker_rate


def main():
    p = argparse.ArgumentParser('TNSM vs BND-only temporal vis')
    p.add_argument('--bnd_ckpt', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/checkpoints/prism_plus/stage1_v6/best.pt')
    p.add_argument('--tnsm_ckpt', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/checkpoints/prism_plus/stage4_tnsm/best_tnsm.pt')
    p.add_argument('--vfm_checkpoint', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/weights/moge2.pt')
    p.add_argument('--dreds_root', default='/inspire/hdd/global_user/liangxiujian-253308390319/datasets/DREDS/DREDS-CatKnown')
    p.add_argument('--dreds_split', default='shapenet_generate_1216/val_part2')
    p.add_argument('--n_windows', type=int, default=25)     # 25 windows x 4 frames = 100 frames
    p.add_argument('--T', type=int, default=4)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/4] Loading BND v0.6.1 (frozen) from {args.bnd_ckpt}')
    bnd = create_bnd_plus(
        vfm_type='moge2', vfm_checkpoint=args.vfm_checkpoint,
        encoder_size='large', deep_supervision=True,
        vfm_cross_attn_dim=128, boundary_band_radius=3,
    ).to(device).eval()
    sd = torch.load(args.bnd_ckpt, map_location=device, weights_only=False)
    bnd.load_state_dict(sd.get('model', sd), strict=False)
    for q in bnd.parameters(): q.requires_grad = False

    print(f'[2/4] Loading TNSM from {args.tnsm_ckpt}')
    tnsm = TNSM(
        enc_in_channels=1, state_channels=128, spatial_div=4, use_flow_warp=True,
    ).to(device).eval()
    sd_t = torch.load(args.tnsm_ckpt, map_location=device, weights_only=False)
    tnsm_state = sd_t.get('tnsm_state', sd_t.get('model', sd_t))
    tnsm.load_state_dict(tnsm_state, strict=False)
    for q in tnsm.parameters(): q.requires_grad = False

    print(f'[3/4] Loading RAFT (torchvision C_T_V2)')
    raft = RAFTFlow(weights_name='C_T_V2', iters=12, freeze=True).to(device).eval()

    print(f'[4/4] Building temporal windows (T={args.T}, stride={args.stride})')
    base = DREDSDataset(
        root=args.dreds_root, splits=[args.dreds_split],
        resolution=args.resolution, sensor_id='dreds_d415',
    )
    ds = TemporalWindow(base, T=args.T, stride=args.stride)
    rng = np.random.RandomState(args.seed)
    idx = sorted(rng.choice(len(ds), min(args.n_windows, len(ds)), replace=False).tolist())
    loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, num_workers=2)
    print(f'  {len(idx)} windows selected from {len(ds)} total')

    saved = 0
    tnfr_bnd_all, tnfr_tnsm_all = [], []
    for batch in tqdm(loader):
        rgb_seq  = batch['rgb'].to(device)          # [1, T, 3, H, W]
        sim_seq  = batch['sim_depth'].to(device)
        real_seq = batch['real_depth'].to(device)
        gt_seq   = batch['hole_mask'].to(device)
        T_ = rgb_seq.shape[1]; H = rgb_seq.shape[-2]; W = rgb_seq.shape[-1]

        with torch.no_grad():
            flows = []
            for t in range(1, T_):
                flows.append(raft(rgb_seq[0, t:t+1], rgb_seq[0, t-1:t]))

            bnd_probs = []
            tnsm_probs = []
            h_prev = None
            for t in range(T_):
                out_b = bnd(rgb_seq[0, t:t+1], sim_seq[0, t:t+1])
                coarse = out_b['coarse_logits']
                target_hw = (H//4, W//4)
                enc_feat = F.interpolate(coarse, size=target_hw, mode='bilinear', align_corners=False)
                flow_in = flows[t-1] if t > 0 else None
                h_prev, feat_out, _ = tnsm.step(enc_feat, h_prev, flow_in)
                delta = F.interpolate(feat_out, size=out_b['failure_logits'].shape[-2:],
                                       mode='bilinear', align_corners=False)
                fused = out_b['failure_logits'] + delta
                bnd_probs.append(torch.sigmoid(out_b['failure_logits'])[0,0].cpu().numpy())
                tnsm_probs.append(torch.sigmoid(fused)[0,0].cpu().numpy())

        # Compute per-clip TNFR
        bnd_seq  = torch.tensor(np.stack(bnd_probs)).unsqueeze(1).unsqueeze(1)   # [T,1,1,H,W]
        tnsm_seq = torch.tensor(np.stack(tnsm_probs)).unsqueeze(1).unsqueeze(1)
        gt_clip  = gt_seq[0].unsqueeze(1)                                          # [T,1,1,H,W] -> reshape
        gt_clip  = gt_clip.cpu()
        tnfr_b = temporal_noise_flicker_rate(bnd_seq,  gt_clip, args.threshold)
        tnfr_t = temporal_noise_flicker_rate(tnsm_seq, gt_clip, args.threshold)
        tnfr_bnd_all.append(tnfr_b); tnfr_tnsm_all.append(tnfr_t)

        fig, axes = plt.subplots(4, T_, figsize=(4 * T_, 14))
        for t in range(T_):
            rgb_np = rgb_seq[0,t].cpu().permute(1,2,0).numpy()
            rgb_np = np.clip(rgb_np, 0, 1)
            axes[0, t].imshow(rgb_np);                                            axes[0,t].set_title(f'RGB t{t}')
            axes[1, t].imshow(bnd_probs[t],  cmap='hot', vmin=0, vmax=1);         axes[1,t].set_title(f'BND-only density')
            axes[2, t].imshow(tnsm_probs[t], cmap='hot', vmin=0, vmax=1);         axes[2,t].set_title(f'BND+TNSM density')
            axes[3, t].imshow(gt_seq[0,t,0].cpu().numpy(), cmap='gray');          axes[3,t].set_title(f'GT mask t{t}')
        for ax in axes.flat: ax.axis('off')
        fig.suptitle(f'window {saved:03d}  TNFR  BND={tnfr_b:.4f}  BND+TNSM={tnfr_t:.4f}', fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / f'window_{saved:03d}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        saved += 1

    # Aggregate TNFR
    print(f'\n=== TNFR aggregate over {len(tnfr_bnd_all)} windows ===')
    print(f'BND-only      : mean {np.mean(tnfr_bnd_all):.4f}  median {np.median(tnfr_bnd_all):.4f}')
    print(f'BND+TNSM      : mean {np.mean(tnfr_tnsm_all):.4f}  median {np.median(tnfr_tnsm_all):.4f}')
    print(f'TNSM reduction: {(np.mean(tnfr_bnd_all)-np.mean(tnfr_tnsm_all))/max(np.mean(tnfr_bnd_all),1e-6)*100:.1f}%')
    with open(out_dir / 'tnfr_summary.txt', 'w') as f:
        f.write(f'BND-only mean TNFR: {np.mean(tnfr_bnd_all):.4f}\n')
        f.write(f'BND+TNSM mean TNFR: {np.mean(tnfr_tnsm_all):.4f}\n')
    print(f'{saved} windows -> {out_dir}')


if __name__ == '__main__':
    main()
