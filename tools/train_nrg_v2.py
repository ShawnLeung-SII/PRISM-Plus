#!/usr/bin/env python3
"""PRISM+ Stage 2a — Mask-conditioned NRG training (C2, standalone NRG).

Pipeline per batch:
    1. Frozen BND v0.4 forward         → mask_density (continuous [0,1])
    2. NRGStandalone.forward           → diffusion loss + L_boundary
    3. DDP backward
       (UNet is trainable by default; VAE and text encoder are frozen.)

The NRG is initialised from a stock SD-1.5 checkpoint (Latent_SD.ckpt) via
UNet first-conv 4 → 9 channel expansion (new 5 ch zero-initialised).

Usage:
    torchrun --nproc_per_node=4 tools/train_nrg_v2.py \
        --config configs/stage2a_nrg.yaml --seed 42
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data import ByteCamDepthDataset
from prism_plus.models import create_bnd_plus, NRGStandalone


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


def _make_loaders(cfg, args, env):
    res = cfg.get('resolution', 256)
    train_ds = ByteCamDepthDataset(data_root=cfg['data_root'], split='train',
                                   resolution=res, augment=True)
    val_ds   = ByteCamDepthDataset(data_root=cfg['data_root'], split='val',
                                   resolution=res, augment=False)
    if args.debug:
        train_ds = Subset(train_ds, range(min(16, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(8,  len(val_ds))))
        cfg['batch_size'] = min(cfg.get('batch_size', 2), 2)

    tr_s = DistributedSampler(train_ds) if env['dist'] else None
    va_s = DistributedSampler(val_ds, shuffle=False) if env['dist'] else None
    tr_l = DataLoader(train_ds, batch_size=cfg.get('batch_size', 2),
                      shuffle=(tr_s is None), sampler=tr_s,
                      num_workers=cfg.get('num_workers', 4),
                      pin_memory=True, drop_last=True)
    va_l = DataLoader(val_ds, batch_size=cfg.get('batch_size', 2),
                      shuffle=False, sampler=va_s,
                      num_workers=2, pin_memory=True, drop_last=False)
    return tr_l, va_l, tr_s, va_s


def load_bnd_v4(cfg, device):
    bnd = create_bnd_plus(
        vfm_type=cfg.get('vfm_type', 'moge2'),
        vfm_checkpoint=cfg.get('vfm_checkpoint'),
        encoder_size=cfg.get('encoder_size', 'large'),
        deep_supervision=True,
        vfm_cross_attn_dim=cfg.get('vfm_cross_attn_dim', 128),
        boundary_band_radius=cfg.get('boundary_band_radius', 3),
    ).to(device).eval()
    ckpt = torch.load(cfg['bnd_v4_checkpoint'], map_location=device, weights_only=False)
    sd = ckpt.get('model', ckpt)
    miss, unexp = bnd.load_state_dict(sd, strict=False)
    print(f'  [BND v0.4] loaded {cfg["bnd_v4_checkpoint"]}: missing={len(miss)} unexpected={len(unexp)}')
    for p in bnd.parameters():
        p.requires_grad = False
    return bnd


def build_nrg(cfg, device):
    nrg = NRGStandalone(
        sd_checkpoint=cfg['sd_checkpoint'],
        d_min=cfg.get('d_min', 0.05),
        d_max=cfg.get('d_max', 5.0),
        filter_max=cfg.get('filter_max', 4.9),
        residual_scale=cfg.get('residual_scale', 2.0),
        scale_factor=cfg.get('scale_factor', 0.18215),
        num_timesteps=cfg.get('num_timesteps', 1000),
        linear_start=cfg.get('linear_start', 0.00085),
        linear_end=cfg.get('linear_end', 0.0120),
        lambda_boundary=cfg.get('lambda_boundary', 0.3),
        boundary_radius=cfg.get('boundary_radius', 5),
        freeze_vae=cfg.get('freeze_vae', True),
        freeze_unet=cfg.get('freeze_unet', False),
        freeze_text=cfg.get('freeze_text', True),
    ).to(device)
    return nrg


def main():
    p = argparse.ArgumentParser(description='PRISM+ Stage 2a (M-cond standalone NRG)')
    p.add_argument('--config', required=True)
    p.add_argument('--resume', default=None)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env = setup_dist()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # GPU-adaptive batch size
    by_gpu = cfg.get('batch_size_by_gpu') or {}
    if torch.cuda.is_available() and by_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        for k, v in by_gpu.items():
            if k in gpu_name:
                if env['main']:
                    print(f'  [auto] GPU={gpu_name!r} -> batch_size = {v}')
                cfg['batch_size'] = v
                break

    out_dir = Path(cfg['output_dir'])
    if env['main']:
        out_dir.mkdir(parents=True, exist_ok=True)

    bnd = load_bnd_v4(cfg, env['device'])
    nrg = build_nrg(cfg, env['device'])
    if args.resume:
        ckpt = torch.load(args.resume, map_location=env['device'], weights_only=False)
        nrg.load_state_dict(ckpt.get('model', ckpt), strict=False)
        if env['main']:
            print(f'resumed NRG from {args.resume}')
    if env['dist']:
        nrg = DDP(nrg, device_ids=[env['local']], find_unused_parameters=True)

    train_loader, val_loader, train_sampler, _ = _make_loaders(cfg, args, env)

    epochs = 5 if args.debug else cfg.get('epochs', 50)
    trainable = [p_ for p_ in nrg.parameters() if p_.requires_grad]
    optim = AdamW(trainable, lr=cfg.get('lr', 1e-5), weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=epochs, eta_min=cfg.get('lr', 1e-5) * 0.01)
    scaler = GradScaler()
    if env['main']:
        n_train = sum(p.numel() for p in trainable)
        print(f'  [NRG-S] trainable params: {n_train/1e6:.2f} M')

    best_val = float('inf')
    history: list = []

    for epoch in range(1, epochs + 1):
        nrg.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        sums = {'loss': 0.0, 'loss_diff': 0.0, 'loss_boundary': 0.0,
                'valid_ratio': 0.0, 'mask_mean': 0.0}
        n_batches = 0
        for batch in tqdm(train_loader, desc=f'Ep{epoch:03d}/train',
                          leave=False, disable=not env['main']):
            rgb    = batch['rgb'].to(env['device'], non_blocking=True)
            sim_d  = batch['sim_depth'].to(env['device'], non_blocking=True)
            real_d = batch['real_depth'].to(env['device'], non_blocking=True)
            hole_m = batch['hole_mask'].to(env['device'], non_blocking=True)
            with torch.no_grad():
                with autocast('cuda'):
                    bnd_out = bnd(rgb, sim_d)
                density = bnd_out['pred_failure'].float()

            optim.zero_grad()
            with autocast('cuda'):
                log = nrg(sim_depth=sim_d, real_depth=real_d,
                          hole_mask=hole_m, mask_density=density, rgb=rgb)
                loss = log['loss']
            if not torch.isfinite(loss):
                if env['main']:
                    print('  [warn] non-finite loss skipped')
                optim.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optim); scaler.update()
            for k in sums:
                v = log.get(k, 0.0)
                sums[k] += float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
            n_batches += 1
        sched.step()
        n_batches = max(n_batches, 1)
        for k in sums: sums[k] /= n_batches

        if env['main']:
            # ---- val ----
            nrg.eval()
            v_sums = {'loss': 0.0, 'loss_diff': 0.0, 'loss_boundary': 0.0}
            v_n = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='val', leave=False):
                    rgb    = batch['rgb'].to(env['device'], non_blocking=True)
                    sim_d  = batch['sim_depth'].to(env['device'], non_blocking=True)
                    real_d = batch['real_depth'].to(env['device'], non_blocking=True)
                    hole_m = batch['hole_mask'].to(env['device'], non_blocking=True)
                    with autocast('cuda'):
                        bnd_out = bnd(rgb, sim_d)
                    density = bnd_out['pred_failure'].float()
                    with autocast('cuda'):
                        log = nrg(sim_depth=sim_d, real_depth=real_d,
                                  hole_mask=hole_m, mask_density=density, rgb=rgb)
                    for k in v_sums:
                        v = log.get(k, 0.0)
                        v_sums[k] += float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
                    v_n += 1
            v_n = max(v_n, 1)
            for k in v_sums: v_sums[k] /= v_n

            metrics = {'epoch': epoch,
                       **{f'tr_{k}': v for k, v in sums.items()},
                       **{f'val_{k}': v for k, v in v_sums.items()}}
            history.append(metrics)
            print(f'Ep{epoch:03d} | tr_loss={sums["loss"]:.4f} val_loss={v_sums["loss"]:.4f}'
                  f' | tr_diff={sums["loss_diff"]:.4f} tr_bnd={sums["loss_boundary"]:.4f}'
                  f' val_diff={v_sums["loss_diff"]:.4f} val_bnd={v_sums["loss_boundary"]:.4f}'
                  f' | mask%={sums["mask_mean"]*100:.1f}')

            mdl = nrg.module if env['dist'] else nrg
            if v_sums['loss'] < best_val:
                best_val = v_sums['loss']
                torch.save({'epoch': epoch, 'model': mdl.state_dict(),
                            'metrics': metrics}, out_dir / 'best.pt')
            if epoch % 5 == 0:
                torch.save({'epoch': epoch, 'model': mdl.state_dict()},
                           out_dir / f'epoch_{epoch:03d}.pt')

    if env['main']:
        mdl = nrg.module if env['dist'] else nrg
        torch.save({'epoch': epochs, 'model': mdl.state_dict()},
                   out_dir / 'final.pt')
        with open(out_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f'\nDone. Best val loss = {best_val:.4f}  ->  {out_dir}')

    if env['dist']:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
