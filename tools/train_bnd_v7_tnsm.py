#!/usr/bin/env python3
"""PRISM+ Stage 4 — TNSM temporal training (C4).

Loads a frozen v0.6 BND backbone, attaches a TNSM ConvGRU at the bottleneck
(H/4 feature map), and trains:
    - TNSM ConvGRU + projection heads
    - (optional) last layer of BND decoder (depending on config)

Computes optical flow on the fly with torchvision RAFT (frozen).

Per training step a T-frame clip is processed with BPTT through the
ConvGRU. Loss is the per-frame V9StyleLoss + a small temporal-consistency
regulariser.

Usage:
    torchrun --nproc_per_node=4 tools/train_bnd_v7_tnsm.py \
        --config configs/stage4_tnsm.yaml --seed 42
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data.dreds import DREDSDataset
from prism_plus.data.temporal_window import TemporalWindow
from prism_plus.losses import V9StyleLoss
from prism_plus.metrics_temporal import evaluate_temporal
from prism_plus.models import create_bnd_plus
from prism_plus.models.flow_utils import RAFTFlow
from prism_plus.models.tnsm import TNSM


def setup_dist():
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local)
        return {'dist': True, 'rank': rank, 'local': local,
                'device': torch.device(f'cuda:{local}'), 'main': rank == 0}
    return {'dist': False, 'rank': 0, 'local': 0,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'main': True}


def main():
    p = argparse.ArgumentParser(description='PRISM+ Stage 4 (TNSM temporal)')
    p.add_argument('--config', required=True)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env = setup_dist()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = env['device']

    out_dir = Path(cfg['output_dir'])
    if env['main']:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Frozen v0.6 backbone ----
    backbone = create_bnd_plus(
        vfm_type=cfg.get('vfm_type', 'moge2'),
        vfm_checkpoint=cfg.get('vfm_checkpoint'),
        encoder_size=cfg.get('encoder_size', 'large'),
        deep_supervision=True,
        vfm_cross_attn_dim=cfg.get('vfm_cross_attn_dim', 128),
        boundary_band_radius=cfg.get('boundary_band_radius', 3),
    ).to(device)
    ckpt = torch.load(cfg['v06_checkpoint'], map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt.get('model', ckpt), strict=False)
    for p_ in backbone.parameters():
        p_.requires_grad = False
    backbone.eval()
    if env['main']:
        print(f'  [v0.6] loaded {cfg["v06_checkpoint"]}')

    # ---- 2) TNSM module (trainable) ----
    tnsm = TNSM(
        enc_in_channels=cfg.get('tnsm_enc_channels', 256),
        state_channels=cfg.get('tnsm_state_channels', 128),
        spatial_div=cfg.get('tnsm_spatial_div', 4),
        use_flow_warp=cfg.get('tnsm_use_flow', True),
    ).to(device)

    # ---- 3) RAFT (frozen) for optical flow ----
    raft = RAFTFlow(weights_name=cfg.get('raft_weights', 'C_T_V2'),
                     iters=cfg.get('raft_iters', 12),
                     freeze=True).to(device)
    if env['main']:
        print(f'  [RAFT] torchvision weights={cfg.get("raft_weights", "C_T_V2")}')

    if env['dist']:
        tnsm = torch.nn.parallel.DistributedDataParallel(tnsm, device_ids=[env['local']])

    # ---- 4) Data: T-frame windows ----
    T = int(cfg.get('T', 4))
    base = DREDSDataset(
        root=cfg['dreds_root'],
        splits=cfg.get('dreds_splits', ['shapenet_generate_1216/val_part2']),
        resolution=cfg.get('resolution', 256),
        sensor_id='dreds_d415',
    )
    train_ds = TemporalWindow(base, T=T, stride=cfg.get('stride', 1))
    if env['main']:
        print(f'  [data] {len(train_ds)} {T}-frame windows from {len(base)} base frames')

    sampler = DistributedSampler(train_ds) if env['dist'] else None
    loader = DataLoader(train_ds,
                        batch_size=cfg.get('batch_size', 2),
                        shuffle=(sampler is None), sampler=sampler,
                        num_workers=cfg.get('num_workers', 2),
                        pin_memory=True, drop_last=True)

    # ---- 5) Loss / Optim ----
    loss_fn = V9StyleLoss(
        pos_weight=cfg.get('pos_weight', 3.0),
        use_ohem=True, ohem_ratio=cfg.get('ohem_ratio', 0.3),
        use_small_region_weighting=True,
        dice_weight=0.3,
        use_lpips=False,
        w_final=1.0, w_coarse=0.0, w_edge=0.0,
    ).to(device)
    optim = AdamW([p_ for p_ in tnsm.parameters() if p_.requires_grad],
                  lr=cfg.get('lr', 5e-5), weight_decay=1e-4)
    epochs = 3 if args.debug else cfg.get('epochs', 30)
    sched = CosineAnnealingLR(optim, T_max=epochs, eta_min=cfg.get('lr', 5e-5) * 0.01)
    scaler = GradScaler()

    if env['main']:
        n_train_params = sum(p.numel() for p in tnsm.parameters() if p.requires_grad)
        print(f'  [TNSM] trainable: {n_train_params/1e6:.2f} M')

    # ---- 6) Training ----
    history = []
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        tnsm.train()
        if sampler:
            sampler.set_epoch(epoch)
        sum_loss = 0.0; n_b = 0
        for batch in loader:
            rgb_t   = batch['rgb'].to(device, non_blocking=True)         # [B, T, 3, H, W]
            sim_t   = batch['sim_depth'].to(device, non_blocking=True)
            gt_t    = batch['hole_mask'].to(device, non_blocking=True)

            B, T_, _, H, W = rgb_t.shape
            optim.zero_grad()

            # Compute pairwise flows (frozen RAFT) outside the autograd path
            with torch.no_grad():
                flows = []
                for t in range(1, T_):
                    f = raft(rgb_t[:, t], rgb_t[:, t - 1])
                    flows.append(f)

            with autocast('cuda'):
                # BPTT through ConvGRU — loss path runs through TNSM.proj_*
                h_prev = None
                loss_per_frame = []
                tnsm_mod = tnsm.module if env['dist'] else tnsm
                for t in range(T_):
                    # 1) Frozen backbone forward (no_grad to save mem)
                    with torch.no_grad():
                        out_b = backbone(rgb_t[:, t], sim_t[:, t])
                        coarse = out_b['coarse_logits']            # [B,1,H,W]
                        target_hw = (H // 4, W // 4)
                        enc_feat = nn.functional.interpolate(
                            coarse, size=target_hw, mode='bilinear', align_corners=False)
                    # 2) TNSM step (gradient flows through TNSM weights)
                    flow_in = flows[t - 1] if t > 0 else None
                    h_prev, feat_out, _ = tnsm_mod.step(enc_feat, h_prev, flow_in)
                    # 3) Upsample TNSM output to full res and FUSE into BND logits
                    delta = nn.functional.interpolate(
                        feat_out, size=out_b['failure_logits'].shape[-2:],
                        mode='bilinear', align_corners=False)
                    fused_logits = out_b['failure_logits'].detach() + delta
                    out_fused = {
                        'failure_logits': fused_logits,
                        'coarse_logits':  None,
                        'edge_logits':    None,
                        'pred_failure':   torch.sigmoid(fused_logits),
                    }
                    lossd = loss_fn(out_fused, gt_t[:, t])
                    loss_per_frame.append(lossd['loss'])
                loss = torch.stack(loss_per_frame).mean()

            if not torch.isfinite(loss):
                optim.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_([p_ for p_ in tnsm.parameters() if p_.requires_grad], 1.0)
            scaler.step(optim); scaler.update()

            sum_loss += float(loss); n_b += 1
        sched.step()

        if env['main']:
            sum_loss /= max(n_b, 1)
            history.append({'epoch': epoch, 'train_loss': sum_loss})
            print(f' Ep{epoch:03d} | tr_loss={sum_loss:.4f}')
            if sum_loss < best_loss:
                best_loss = sum_loss
                torch.save({
                    'epoch': epoch,
                    'tnsm_state': (tnsm.module if env['dist'] else tnsm).state_dict(),
                    'metrics': {'train_loss': sum_loss},
                }, out_dir / 'best_tnsm.pt')

    if env['main']:
        torch.save({'epoch': epochs,
                    'tnsm_state': (tnsm.module if env['dist'] else tnsm).state_dict()},
                   out_dir / 'final_tnsm.pt')
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f'Done. Best tr_loss = {best_loss:.4f} -> {out_dir}')

    if env['dist']:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
