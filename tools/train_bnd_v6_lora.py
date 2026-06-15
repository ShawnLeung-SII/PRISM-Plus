#!/usr/bin/env python3
"""PRISM+ Stage 3 — LoRA-SPA cross-sensor fine-tune (C3).

Loads a frozen v0.6 BND backbone (PRISMPlusBND + V9StyleLoss-trained
weights), attaches rank-r LoRA per target sensor, and trains ONLY the
LoRA tensors. This is the implementation of the FINAL_PROPOSAL B5 data
efficiency curve experiment: sweep N_train ∈ {10, 50, 100, 200, 500}.

Usage (single sensor + single sample size):
    torchrun --nproc_per_node=1 tools/train_bnd_v6_lora.py \
        --config configs/stage3_lora.yaml \
        --sensor dreds_d415 --n_train 100 --seed 42

Multi-run B5 sweep (delegated to scripts/lora_b5_sweep.sh).
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data.dreds import DREDSDataset
from prism_plus.losses import V9StyleLoss
from prism_plus.metrics import evaluate_bnd, evaluate_density_full
from prism_plus.models import create_bnd_plus
from prism_plus.models.lora_spa import LoRASPA


# Registry: sensor_id -> dataset builder
def _build_sensor_dataset(sensor_id: str, cfg: dict):
    if sensor_id.startswith('dreds_'):
        return DREDSDataset(
            root=cfg['dreds_root'],
            splits=cfg.get('dreds_splits', ['shapenet_generate_1216/val_part2']),
            resolution=cfg.get('resolution', 256),
            sensor_id=sensor_id,
        )
    raise ValueError(f'Unsupported sensor_id: {sensor_id}')


def _split_indices(n: int, n_train: int, n_val: int, n_test: int, seed: int):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return (
        idx[:n_train].tolist(),
        idx[n_train:n_train + n_val].tolist(),
        idx[n_train + n_val:n_train + n_val + n_test].tolist(),
    )


def main():
    p = argparse.ArgumentParser(description='PRISM+ C3 LoRA-SPA fine-tune')
    p.add_argument('--config', required=True)
    p.add_argument('--sensor', required=True, help='target sensor id, e.g. dreds_d415')
    p.add_argument('--n_train', type=int, required=True, help='few-shot train sample count')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(cfg['output_dir']) / f'{args.sensor}_n{args.n_train}_s{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Backbone (frozen v0.6) ----
    print(f'[1/4] Loading v0.6 backbone from {cfg["v06_checkpoint"]!r}')
    backbone = create_bnd_plus(
        vfm_type=cfg.get('vfm_type', 'moge2'),
        vfm_checkpoint=cfg.get('vfm_checkpoint'),
        encoder_size=cfg.get('encoder_size', 'large'),
        deep_supervision=True,
        vfm_cross_attn_dim=cfg.get('vfm_cross_attn_dim', 128),
        boundary_band_radius=cfg.get('boundary_band_radius', 3),
    )
    ckpt = torch.load(cfg['v06_checkpoint'], map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    miss, unexp = backbone.load_state_dict(sd, strict=False)
    print(f'  loaded: missing={len(miss)} unexpected={len(unexp)}')

    # ---- 2) LoRA wrapper ----
    model = LoRASPA(
        base_model=backbone,
        sensor_ids=[args.sensor],
        d_sem=cfg.get('d_sem', 1024),
        rank=cfg.get('lora_rank', 4),
        alpha=cfg.get('lora_alpha', 1.0),
        dropout=cfg.get('lora_dropout', 0.0),
        freeze_base=True,
    ).to(device)
    model.set_active_sensor(args.sensor)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[2/4] LoRA installed. trainable: {n_train_params / 1e6:.4f} M (rank={cfg.get("lora_rank", 4)})')

    # ---- 3) Data ----
    print(f'[3/4] Building dataset for sensor={args.sensor}')
    full_ds = _build_sensor_dataset(args.sensor, cfg)
    print(f'  total {len(full_ds)} frames')
    n_val = min(cfg.get('n_val', 200), max(len(full_ds) - args.n_train - 100, 100))
    n_test = max(len(full_ds) - args.n_train - n_val, 0)
    tr_idx, va_idx, te_idx = _split_indices(len(full_ds), args.n_train, n_val, n_test, args.seed)
    print(f'  split  train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}')

    train_loader = DataLoader(Subset(full_ds, tr_idx),
                              batch_size=cfg.get('batch_size', 4),
                              shuffle=True, num_workers=cfg.get('num_workers', 2),
                              pin_memory=True, drop_last=False)
    val_loader   = DataLoader(Subset(full_ds, va_idx),
                              batch_size=cfg.get('batch_size', 4),
                              shuffle=False, num_workers=2,
                              pin_memory=True, drop_last=False)
    test_loader  = DataLoader(Subset(full_ds, te_idx),
                              batch_size=cfg.get('batch_size', 4),
                              shuffle=False, num_workers=2,
                              pin_memory=True, drop_last=False)

    # ---- 4) Loss + optim + schedule ----
    loss_fn = V9StyleLoss(
        pos_weight=cfg.get('pos_weight', 3.0),
        use_ohem=True, ohem_ratio=cfg.get('ohem_ratio', 0.3),
        use_small_region_weighting=True, small_region_weight=2.0,
        dice_weight=0.3,
        use_lpips=False,        # LoRA fine-tune doesn't need LPIPS (slower, marginal)
        w_final=1.0, w_coarse=0.0, w_edge=0.0,
    ).to(device)

    epochs = cfg.get('epochs', 30)
    optim  = AdamW(model.parameters(), lr=cfg.get('lr', 1e-3), weight_decay=1e-4)
    sched  = CosineAnnealingLR(optim, T_max=epochs, eta_min=cfg.get('lr', 1e-3) * 0.01)
    scaler = GradScaler()

    print(f'[4/4] Training {epochs} epochs (lr={cfg.get("lr", 1e-3):.1e})')

    best = {'epoch': 0, 'iou_cleaned': 0.0, 'metrics': None}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = 0.0; n_b = 0
        for batch in tqdm(train_loader, desc=f'Ep{epoch:03d}/tr', leave=False):
            rgb   = batch['rgb'].to(device, non_blocking=True)
            sim_d = batch['sim_depth'].to(device, non_blocking=True)
            gt    = batch['hole_mask'].to(device, non_blocking=True)
            optim.zero_grad()
            with autocast('cuda'):
                out = model(rgb, sim_d)
                lossd = loss_fn(out, gt)
                loss = lossd['loss']
            if not torch.isfinite(loss):
                optim.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            sum_loss += float(loss); n_b += 1
        sched.step()
        n_b = max(n_b, 1)

        # ---- Eval on val ----
        model.eval()
        agg = {}
        with torch.no_grad():
            for batch in val_loader:
                rgb   = batch['rgb'].to(device, non_blocking=True)
                sim_d = batch['sim_depth'].to(device, non_blocking=True)
                gt    = batch['hole_mask'].to(device, non_blocking=True)
                with autocast('cuda'):
                    out = model(rgb, sim_d)
                pred_p = out['pred_failure'].float()
                bnd = evaluate_bnd(pred_p, gt, threshold=0.25)
                for k, v in bnd.items():
                    agg.setdefault(k, []).append(v)
        metrics = {k: float(np.mean(v)) for k, v in agg.items()}
        metrics['epoch'] = epoch
        metrics['train_loss'] = sum_loss / n_b
        history.append(metrics)

        iou = metrics.get('inv_iou', 0.0)
        marker = '*' if iou > best['iou_cleaned'] else ' '
        print(f' Ep{epoch:03d} | loss={metrics["train_loss"]:.4f}'
              f' | val inv_iou@0.25={iou:.4f}'
              f' P={metrics.get("precision", 0):.3f}'
              f' R={metrics.get("recall", 0):.3f}'
              f' F1={metrics.get("f1", 0):.3f}  {marker}')

        if iou > best['iou_cleaned']:
            best = {'epoch': epoch, 'iou_cleaned': iou, 'metrics': metrics}
            torch.save({
                'lora_state': {k: v.detach().cpu()
                                for k, v in model.adapters[args.sensor].state_dict().items()},
                'sensor_id': args.sensor,
                'n_train': args.n_train,
                'seed': args.seed,
                'metrics': metrics,
            }, out_dir / 'best_lora.pt')

    # ---- Final test pass ----
    model.eval()
    agg = {}
    with torch.no_grad():
        for batch in test_loader:
            rgb   = batch['rgb'].to(device, non_blocking=True)
            sim_d = batch['sim_depth'].to(device, non_blocking=True)
            gt    = batch['hole_mask'].to(device, non_blocking=True)
            with autocast('cuda'):
                out = model(rgb, sim_d)
            pred_p = out['pred_failure'].float()
            bnd = evaluate_bnd(pred_p, gt, threshold=0.25)
            for k, v in bnd.items():
                agg.setdefault(k, []).append(v)
    test_metrics = {k: float(np.mean(v)) for k, v in agg.items()}

    summary = {
        'sensor': args.sensor,
        'n_train': args.n_train,
        'seed': args.seed,
        'best_val': best,
        'test': test_metrics,
        'config': cfg,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print(f'=== B5 sample sensor={args.sensor} n_train={args.n_train} ===')
    print(f'best val inv_iou@0.25 = {best["iou_cleaned"]:.4f}  (epoch {best["epoch"]})')
    print(f'test inv_iou@0.25     = {test_metrics.get("inv_iou", 0):.4f}')
    print(f'saved -> {out_dir}')


if __name__ == '__main__':
    main()
