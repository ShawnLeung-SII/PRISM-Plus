#!/usr/bin/env python3
"""PRISM+ BND v0.6.0 — C1 architecture + v9-style loss.

Design rationale (vs v0.4.0):
    - Training target  : RAW hole_mask (was M_target). Keeps fine-grained
      detail; the model is free to learn even tiny holes.
    - Evaluation target: BOTH raw GT AND M_target. We report iou@0.5 (raw)
      AND iou_on_target@0.5 (M_target) so 'visual richness' and 'fairness
      against intrinsic noise' are both reflected.
    - Loss          : V9StyleLoss with pos_weight=3.0, OHEM ratio=0.3,
      small-region weighting 2x (kernel=5), Dice (w=0.3), LPIPS-VGG (w=1.0).
      OHEM ratio bumped from v9's 0.1 → 0.3 for signal stability.
    - Model         : PRISMPlusBND (Gated VFM cross-attn + boundary branch);
      keeps the C1 contribution intact.
    - Eval threshold : default 0.25 (v9-style detail recall) + multi-thresh
      (0.3/0.5/0.7) reported via evaluate_density_full.

Usage:
    torchrun --nproc_per_node=4 tools/train_bnd_v6.py \
        --config configs/stage1_v6.yaml --seed 42
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
from prism_plus.data.gt_decompose_wrapper import GTDecomposeWrapper
from prism_plus.losses import V9StyleLoss
from prism_plus.metrics import evaluate_bnd, evaluate_density_full
from prism_plus.models import create_bnd_plus
from prism_plus.utils import dump_eval_batch


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
    res = cfg.get('resolution', 512)
    base_train = ByteCamDepthDataset(data_root=cfg['data_root'], split='train',
                                     resolution=res, augment=True)
    base_val   = ByteCamDepthDataset(data_root=cfg['data_root'], split='val',
                                     resolution=res, augment=False)
    # Eval uses M_target — wrap val side with GTDecomposeWrapper.
    # Train side also wraps so dataloader signature is consistent (we just ignore
    # m_target/m_ignore at training time).
    preset = cfg.get('gt_preset', 'medium')
    train_ds = GTDecomposeWrapper(base_train, preset=preset)
    val_ds   = GTDecomposeWrapper(base_val,   preset=preset)
    if env['main']:
        print(f'  [data] GT decomposition preset (eval only) = {preset!r}')

    if args.debug:
        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(16, len(val_ds))))
        cfg['batch_size'] = min(cfg.get('batch_size', 16), 4)

    tr_s = DistributedSampler(train_ds) if env['dist'] else None
    va_s = DistributedSampler(val_ds, shuffle=False) if env['dist'] else None
    tr_l = DataLoader(train_ds, batch_size=cfg.get('batch_size', 16),
                      shuffle=(tr_s is None), sampler=tr_s,
                      num_workers=cfg.get('num_workers', 4),
                      pin_memory=True, drop_last=True)
    va_l = DataLoader(val_ds, batch_size=cfg.get('batch_size', 16),
                      shuffle=False, sampler=va_s,
                      num_workers=4, pin_memory=True, drop_last=False)
    return tr_l, va_l, tr_s, va_s


def main():
    p = argparse.ArgumentParser(description='PRISM+ BND v0.6.0 (C1 + v9 loss)')
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
    vis_dir = out_dir / 'vis'
    if env['main']:
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model (PRISMPlusBND, C1 architecture) ----
    model = create_bnd_plus(
        vfm_type=cfg.get('vfm_type', 'moge2'),
        vfm_checkpoint=cfg.get('vfm_checkpoint'),
        encoder_size=cfg.get('encoder_size', 'large'),
        deep_supervision=True,
        vfm_cross_attn_dim=cfg.get('vfm_cross_attn_dim', 128),
        boundary_band_radius=cfg.get('boundary_band_radius', 3),
    ).to(env['device'])
    for n, p_ in model.named_parameters():
        p_.requires_grad = ('vfm' not in n.lower())
    for mod in [model.gate_f3, model.gate_f4,
                model.boundary_branch, model.boundary_refiner]:
        for p_ in mod.parameters():
            p_.requires_grad = True

    if args.resume:
        ckpt = torch.load(args.resume, map_location=env['device'], weights_only=False)
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        if env['main']:
            print(f'Resumed from {args.resume}')

    if env['dist']:
        model = DDP(model, device_ids=[env['local']], find_unused_parameters=True)

    # ---- Loss ----
    loss_fn = V9StyleLoss(
        pos_weight=cfg.get('pos_weight', 3.0),
        use_ohem=cfg.get('use_ohem', True),
        ohem_ratio=cfg.get('ohem_ratio', 0.3),
        use_small_region_weighting=cfg.get('use_small_region', True),
        small_region_weight=cfg.get('small_region_weight', 2.0),
        small_region_kernel=cfg.get('small_region_kernel', 5),
        dice_weight=cfg.get('dice_weight', 0.3),
        use_lpips=cfg.get('use_lpips', True),
        lpips_weight=cfg.get('lpips_weight', 1.0),
        w_final=cfg.get('w_final', 1.0),
        w_coarse=cfg.get('w_coarse', 0.3),
        w_edge=cfg.get('w_edge', 0.0),
        label_smoothing=cfg.get('label_smoothing', 0.0),
        focal_gamma=cfg.get('focal_gamma', 0.0),
    ).to(env['device'])

    # ---- Data ----
    train_loader, val_loader, train_sampler, _ = _make_loaders(cfg, args, env)

    epochs = 5 if args.debug else cfg.get('epochs', 100)
    optim  = AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                   lr=cfg.get('lr', 1e-4), weight_decay=cfg.get('weight_decay', 1e-2))
    sched  = CosineAnnealingLR(optim, T_max=epochs, eta_min=cfg.get('lr', 1e-4) * 0.01)
    scaler = GradScaler()

    threshold_main = float(cfg.get('eval_threshold', 0.25))
    vis_n          = int(cfg.get('vis_n_samples', 10))

    best_iou_target = 0.0
    history: list = []

    for epoch in range(1, epochs + 1):
        # -------- train --------
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        sum_t = 0.0
        sum_comps = {'l_final_bce':0.0,'l_final_dice':0.0,'l_final_lpips':0.0,
                     'l_coarse_total':0.0,'l_edge_total':0.0}
        n_b = 0
        for batch in tqdm(train_loader, desc=f'Ep{epoch:03d}/train',
                          leave=False, disable=not env['main']):
            rgb   = batch['rgb'].to(env['device'], non_blocking=True)
            sim_d = batch['sim_depth'].to(env['device'], non_blocking=True)
            gt_raw= batch['hole_mask'].to(env['device'], non_blocking=True)   # RAW supervision

            optim.zero_grad()
            with autocast('cuda'):
                out = model(rgb, sim_d)
                lossd = loss_fn(out, gt_raw)
                loss = lossd['loss']
            if not torch.isfinite(loss):
                if env['main']: print(f'  [warn] non-finite loss skipped')
                optim.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            sum_t += loss.item()
            for k in sum_comps:
                v = lossd.get(k, 0.0)
                sum_comps[k] += float(v) if isinstance(v, (int,float)) else float(v.detach())
            n_b += 1
        sched.step()
        n_b = max(n_b, 1)
        sum_t /= n_b
        for k in sum_comps: sum_comps[k] /= n_b

        # -------- eval --------
        if env['main']:
            model.eval()
            agg_d, agg_b = {}, {}
            vis_rgb, vis_sim, vis_real, vis_gt, vis_pred = [], [], [], [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='eval', leave=False):
                    rgb    = batch['rgb'].to(env['device'], non_blocking=True)
                    sim_d  = batch['sim_depth'].to(env['device'], non_blocking=True)
                    real_d = batch.get('real_depth', sim_d).to(env['device'], non_blocking=True)
                    gt_raw = batch['hole_mask'].to(env['device'], non_blocking=True)
                    m_tgt  = batch['m_target'].to(env['device'], non_blocking=True)
                    m_ign  = batch['m_ignore'].to(env['device'], non_blocking=True)
                    with autocast('cuda'):
                        out = model(rgb, sim_d)
                    pred_p = out['pred_failure'].float()

                    # Full density metrics (raw + on_target)
                    dens = evaluate_density_full(pred_p, gt_raw, m_tgt, m_ign)
                    for k, v in dens.items():
                        agg_d.setdefault(k, []).append(v)
                    # legacy at threshold_main
                    bnd = evaluate_bnd(pred_p, gt_raw, threshold=threshold_main)
                    for k, v in bnd.items():
                        agg_b.setdefault(f'main_{k}', []).append(v)

                    # vis
                    if sum(b.shape[0] for b in vis_rgb) < vis_n:
                        need = vis_n - sum(b.shape[0] for b in vis_rgb)
                        vis_rgb .append(rgb [:need].detach().float().cpu())
                        vis_sim .append(sim_d[:need].detach().float().cpu())
                        vis_real.append(real_d[:need].detach().float().cpu())
                        vis_gt  .append(gt_raw[:need].detach().float().cpu())
                        vis_pred.append(pred_p[:need].detach().float().cpu())

            metrics = {'epoch': epoch, 'train_loss': sum_t}
            for k, vs in agg_d.items():
                metrics[k] = float(np.mean(vs))
            for k, vs in agg_b.items():
                metrics[k] = float(np.mean(vs))
            metrics.update(sum_comps)
            history.append(metrics)

            print(f'Ep{epoch:03d} | loss={sum_t:.4f}'
                  f' | iou_cleaned@.25={metrics.get("iou_cleaned@0.25",0):.4f}'      # ★ main fair metric
                  f' iou_cleaned@.5={metrics.get("iou_cleaned@0.5",0):.4f}'
                  f' | iou_tgt@.5={metrics.get("iou_on_target@0.5",0):.4f}'           # one-sided clean (GT only)
                  f' iou_raw@.5={metrics.get("iou@0.5",0):.4f}'                       # no cleaning
                  f' | ECE={metrics.get("ece",0):.4f}'
                  f' AUROC={metrics.get("auroc",0):.4f}'
                  f' Brier={metrics.get("brier",0):.4f}'
                  f' | P@.25={metrics.get("P@0.3",0):.3f}'
                  f' R@.25={metrics.get("R@0.3",0):.3f}'
                  f' F1@.25={metrics.get("F1@0.3",0):.3f}')

            if vis_rgb:
                v_rgb=torch.cat(vis_rgb); v_sim=torch.cat(vis_sim); v_real=torch.cat(vis_real)
                v_gt=torch.cat(vis_gt);   v_pred=torch.cat(vis_pred)
                ep_dir = vis_dir / f'ep_{epoch:03d}'
                try:
                    dump_eval_batch(ep_dir, v_rgb, v_sim, v_real, v_gt, v_pred,
                                    threshold=threshold_main)
                except Exception as e:
                    print(f'  [warn] vis dump failed: {e}')

            mdl = model.module if env['dist'] else model
            iou_main = metrics.get('iou_cleaned@0.25', 0.0)
            if iou_main > best_iou_target:
                best_iou_target = iou_main
                torch.save({'epoch': epoch, 'model': mdl.state_dict(),
                            'metrics': metrics}, out_dir / 'best.pt')
            if epoch % 10 == 0:
                torch.save({'epoch': epoch, 'model': mdl.state_dict()},
                           out_dir / f'epoch_{epoch:03d}.pt')

    if env['main']:
        mdl = model.module if env['dist'] else model
        torch.save({'epoch': epochs, 'model': mdl.state_dict()},
                   out_dir / 'final.pt')
        with open(out_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f'\nDone. Best iou_cleaned@0.25 = {best_iou_target:.4f} -> {out_dir}')

    if env['dist']:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
