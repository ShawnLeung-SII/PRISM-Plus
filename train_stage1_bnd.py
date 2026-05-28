"""
Stage 1: Train SpatialSPRBND (PRISM+ C1)

Uses DualModalNoiseDatasetV3 + ByteCameraDepth (same as V9 baseline).
Freezes VFM; trains only new cross-attn modules + fine-tunes decoder.

Usage:
    torchrun --nproc_per_node=N train_stage1_bnd.py --config configs/stage1_spatial_spr.yaml
"""

import argparse, logging, os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

_LAT  = '/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth'
_PLUS = '/inspire/ssd/project/robot-dna/liangxiujian-253308390319/0-XIUJIANLIANG/prism_plus'
for p in [_LAT, _PLUS]:
    if p not in sys.path: sys.path.insert(0, p)

from latpixdepth.data.dual_modal_dataset_v3 import DualModalNoiseDatasetV3
from models.spatial_spr_bnd import SpatialSPRBND
from utils.metrics_prism_plus import inv_iou, boundary_iou

try:
    import wandb; WANDB = True
except ImportError:
    WANDB = False


# ---------------------------------------------------------------------------
# Loss (same H-PPS structure as V9 but simplified for Stage 1)
# ---------------------------------------------------------------------------

def hpps_loss(pred_dict, gt_mask, pos_weight=3.0, dice_w=0.3):
    """
    Hierarchical Positive-Prioritized Supervision (H-PPS) — same as PRISM.
    Uses main + aux failure logits when available.
    """
    pw = torch.tensor([pos_weight], device=gt_mask.device)
    main_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_dict['failure_logits'], gt_mask.float(), pos_weight=pw
    )
    # Dice on main output
    p = torch.sigmoid(pred_dict['failure_logits'])
    flat_p = p.flatten(1); flat_t = gt_mask.float().flatten(1)
    dice = 1 - (2*(flat_p*flat_t).sum(1)+1e-6) / (flat_p.sum(1)+flat_t.sum(1)+1e-6)
    loss = main_loss + dice_w * dice.mean()

    if 'aux_failure_logits' in pred_dict:
        for aux_logit in pred_dict['aux_failure_logits']:
            gt_s = nn.functional.interpolate(gt_mask.float(), size=aux_logit.shape[-2:],
                                             mode='nearest')
            loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(
                aux_logit, gt_s, pos_weight=pw
            )
    return loss


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_dist():
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = int(os.environ['RANK'])
        local = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local)
        return {'dist': True, 'rank': rank, 'local': local,
                'device': torch.device(f'cuda:{local}'), 'main': rank == 0}
    return {'dist': False, 'rank': 0, 'local': 0,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'main': True}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--debug',  action='store_true')
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env = setup_dist()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(cfg.get('output_dir',
        '/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/checkpoints/prism_plus/stage1'))
    if env['main']:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model ----
    model = SpatialSPRBND(
        vfm_type      = cfg.get('vfm_type', 'moge2'),
        vfm_checkpoint= cfg.get('vfm_checkpoint', None),
        encoder_size  = cfg.get('encoder_size', 'large'),
        deep_supervision = True,
    ).to(env['device'])

    # Freeze VFM; unfreeze cross-attn + decoder
    for n, p in model.named_parameters():
        p.requires_grad = 'vfm' not in n.lower() or 'cross_attn' in n.lower()
    # Make sure new cross-attn modules are trainable
    for p in model.vfm_cross_attns.parameters():
        p.requires_grad = True

    if args.resume:
        ckpt = torch.load(args.resume, map_location=env['device'])
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)

    if env['dist']:
        model = DDP(model, device_ids=[env['local']], find_unused_parameters=True)

    # ---- Data ----
    data_root = cfg['data_root']
    train_ds = DualModalNoiseDatasetV3(
        data_root=data_root, split='train',
        resolution=cfg.get('resolution', 512), augment=True)
    val_ds   = DualModalNoiseDatasetV3(
        data_root=data_root, split='val',
        resolution=cfg.get('resolution', 512), augment=False)

    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(16, len(val_ds))))
        cfg['batch_size'] = min(cfg.get('batch_size', 16), 4)  # limit for debug GPU

    train_sampler = DistributedSampler(train_ds) if env['dist'] else None
    train_loader  = DataLoader(train_ds, batch_size=cfg.get('batch_size', 16),
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=cfg.get('num_workers', 4), pin_memory=True, drop_last=True)
    val_loader    = DataLoader(val_ds, batch_size=cfg.get('batch_size', 16),
        shuffle=False, num_workers=4, pin_memory=True)

    # ---- Optimiser ----
    epochs = 5 if args.debug else cfg.get('epochs', 100)
    opt    = AdamW([p for p in model.parameters() if p.requires_grad],
                   lr=cfg.get('lr', 1e-4), weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.get('lr', 1e-4)*0.01)
    scaler = GradScaler()

    if WANDB and env['main'] and not args.debug:
        wandb.init(project='prism_plus', name=f'stage1_spatial_spr_s{args.seed}', config=cfg)

    best_iou = 0.0; history = []

    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler: train_sampler.set_epoch(epoch)
        t_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Ep{epoch:03d}/train', leave=False, disable=not env['main']):
            rgb   = batch['rgb'].to(env['device'])
            sim_d = batch['sim_depth'].to(env['device'])
            gt_m  = batch['hole_mask'].to(env['device'])

            opt.zero_grad()
            with autocast('cuda'):
                out  = model(rgb, sim_d)
                loss = hpps_loss(out, gt_m,
                                 pos_weight=cfg.get('pos_weight', 3.0),
                                 dice_w=cfg.get('dice_weight', 0.3))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            t_loss += loss.item()

        sched.step()
        t_loss /= len(train_loader)

        # Eval
        if env['main']:
            model.eval()
            iou_l, biou_l = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='eval', leave=False):
                    rgb  = batch['rgb'].to(env['device'])
                    sim_d= batch['sim_depth'].to(env['device'])
                    gt_m = batch['hole_mask'].to(env['device'])
                    out  = model(rgb, sim_d)
                    iou_l.append(inv_iou(out['pred_failure'], gt_m).item())
                    biou_l.append(boundary_iou(out['pred_failure'], gt_m).item())

            metrics = {'epoch': epoch, 'train_loss': t_loss,
                       'inv_iou': np.mean(iou_l), 'boundary_iou': np.mean(biou_l)}
            history.append(metrics)
            print(f"Ep{epoch:03d} | train_loss={t_loss:.4f} | inv_iou={metrics['inv_iou']:.4f} | boundary_iou={metrics['boundary_iou']:.4f}")

            if WANDB and not args.debug: wandb.log(metrics)

            mdl = model.module if env['dist'] else model
            if metrics['inv_iou'] > best_iou:
                best_iou = metrics['inv_iou']
                torch.save({'epoch': epoch, 'model': mdl.state_dict(), 'metrics': metrics},
                           out_dir / 'best.pt')
            if epoch % 10 == 0:
                torch.save({'epoch': epoch, 'model': mdl.state_dict()},
                           out_dir / f'epoch_{epoch:03d}.pt')

    if env['main']:
        mdl = model.module if env['dist'] else model
        torch.save({'epoch': epochs, 'model': mdl.state_dict()}, out_dir / 'final.pt')
        with open(out_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nDone. Best inv_iou={best_iou:.4f}  →  {out_dir}")

    if env['dist']: dist.destroy_process_group()


if __name__ == '__main__':
    main()
