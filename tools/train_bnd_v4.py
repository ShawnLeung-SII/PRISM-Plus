#!/usr/bin/env python3
"""PRISM+ BND v0.4.0 training entrypoint — density-aware.

Key differences vs v0.3.x:
    * Dataset wrapped with GTDecomposeWrapper → returns M_target + M_ignore
    * Loss: DensityLoss = vanilla weighted BCE (no asymmetric, no Tversky,
      no sharpness, no FP/FN biasing — see prism_plus/losses/density.py)
    * Eval: evaluate_density_full → reports
        - IoU @ {0.3, 0.5, 0.7}  (multi-threshold)
        - IoU on M_target @ 0.5
        - ECE, Brier, AUROC  (on raw GT and on M_target)
        - Precision/Recall/F1 @ 9 thresholds (full PR curve)
      Plus the original inv_iou/boundary_iou/coverage from evaluate_bnd().

Theoretical foundation: sigmoid+BCE optimum is P(y|x), so the trained
network output IS a calibrated density. No "soft GT" needed.

Usage:
    torchrun --nproc_per_node=4 tools/train_bnd_v4.py \\
        --config configs/stage1_v4.yaml --seed 42
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
from prism_plus.losses import DensityLoss
from prism_plus.metrics import (
    evaluate_bnd,                # 兼容指标 (inv_iou, boundary_iou, P/R/F1, polar, cov)
    evaluate_density_full,       # v0.4.0 新加 (ECE, Brier, AUROC, multi-IoU, PR-curve)
)
from prism_plus.models import create_bnd_plus
from prism_plus.utils import dump_eval_batch

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False


def setup_dist() -> dict:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return {"dist": True, "rank": rank, "local": local,
                "device": torch.device(f"cuda:{local}"), "main": rank == 0}
    return {"dist": False, "rank": 0, "local": 0,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "main": True}


def _make_loaders(cfg, args, env):
    base_train = ByteCamDepthDataset(data_root=cfg["data_root"], split="train",
                                     resolution=cfg.get("resolution", 512),
                                     augment=True)
    base_val   = ByteCamDepthDataset(data_root=cfg["data_root"], split="val",
                                     resolution=cfg.get("resolution", 512),
                                     augment=False)
    preset = cfg.get("gt_preset", "medium")
    train_ds = GTDecomposeWrapper(base_train, preset=preset)
    val_ds   = GTDecomposeWrapper(base_val,   preset=preset)
    if env["main"]:
        print(f"  [data] GT decomposition preset = {preset!r}")

    if args.debug:
        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(16, len(val_ds))))
        cfg["batch_size"] = min(cfg.get("batch_size", 16), 4)

    train_sampler = DistributedSampler(train_ds) if env["dist"] else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if env["dist"] else None
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 16),
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=cfg.get("num_workers", 4),
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.get("batch_size", 16),
                              shuffle=False, sampler=val_sampler,
                              num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, val_loader, train_sampler, val_sampler


def main():
    p = argparse.ArgumentParser(description="PRISM+ BND v0.4.0 (density-aware)")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env = setup_dist()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # GPU-adaptive batch_size
    by_gpu = cfg.get("batch_size_by_gpu") or {}
    if torch.cuda.is_available() and by_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        for key, value in by_gpu.items():
            if key in gpu_name:
                if env["main"]:
                    print(f"  [auto] GPU detected: {gpu_name!r} -> batch_size = {value}")
                cfg["batch_size"] = value
                break

    out_dir = Path(cfg["output_dir"])
    vis_dir = out_dir / "vis"
    if env["main"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model (unchanged from v0.3.x) ----
    model = create_bnd_plus(
        vfm_type=cfg.get("vfm_type", "moge2"),
        vfm_checkpoint=cfg.get("vfm_checkpoint"),
        encoder_size=cfg.get("encoder_size", "large"),
        deep_supervision=True,
        vfm_cross_attn_dim=cfg.get("vfm_cross_attn_dim", 128),
        boundary_band_radius=cfg.get("boundary_band_radius", 3),
    ).to(env["device"])

    for n, p_ in model.named_parameters():
        p_.requires_grad = ("vfm" not in n.lower())
    for mod in [model.gate_f3, model.gate_f4,
                model.boundary_branch, model.boundary_refiner]:
        for p_ in mod.parameters():
            p_.requires_grad = True

    if args.resume:
        ckpt = torch.load(args.resume, map_location=env["device"])
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        if env["main"]:
            print(f"Resumed from {args.resume}")

    if env["dist"]:
        model = DDP(model, device_ids=[env["local"]], find_unused_parameters=True)

    # ---- Data ----
    train_loader, val_loader, train_sampler, val_sampler = _make_loaders(cfg, args, env)

    # ---- Optimiser + loss ----
    epochs  = 5 if args.debug else cfg.get("epochs", 100)
    optim   = AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                    lr=cfg.get("lr", 1e-4), weight_decay=1e-4)
    sched   = CosineAnnealingLR(optim, T_max=epochs, eta_min=cfg.get("lr", 1e-4) * 0.01)
    scaler  = GradScaler()
    loss_fn = DensityLoss(
        w_main=cfg.get("w_main", 1.0),
        w_coarse=cfg.get("w_coarse", 0.3),
        w_edge=cfg.get("w_edge", 0.0),
        edge_pos_weight=cfg.get("edge_pos_weight", 5.0),
    )

    if WANDB_OK and env["main"] and not args.debug and os.environ.get("WANDB_MODE") != "disabled":
        wandb.init(project="prism_plus", name=f"v0.4.0_density_s{args.seed}", config=cfg)

    best_iou_target = 0.0
    history: list = []
    vis_n = int(cfg.get("vis_n_samples", 10))

    for epoch in range(1, epochs + 1):
        # -------- train --------
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        sum_t = 0.0
        sum_comps = {"l_main": 0.0, "l_coarse": 0.0, "l_edge": 0.0}
        for batch in tqdm(train_loader, desc=f"Ep{epoch:03d}/train",
                          leave=False, disable=not env["main"]):
            rgb   = batch["rgb"].to(env["device"], non_blocking=True)
            sim_d = batch["sim_depth"].to(env["device"], non_blocking=True)
            m_tgt = batch["m_target"].to(env["device"], non_blocking=True)
            m_ign = batch["m_ignore"].to(env["device"], non_blocking=True)

            optim.zero_grad()
            with autocast("cuda"):
                out = model(rgb, sim_d)
                lossd = loss_fn(
                    final_logits=out["failure_logits"],
                    coarse_logits=out["coarse_logits"],
                    edge_logits=out["edge_logits"],
                    M_target=m_tgt,
                    M_ignore=m_ign,
                )
                loss = lossd["loss"]
            if not torch.isfinite(loss):
                if env["main"]:
                    print(f"  [warn] non-finite loss = {loss.item()}, skipping batch")
                optim.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optim); scaler.update()
            sum_t += loss.item()
            for k in sum_comps:
                sum_comps[k] += float(lossd[k])
        sched.step()
        n_batches = max(len(train_loader), 1)
        sum_t /= n_batches
        for k in sum_comps: sum_comps[k] /= n_batches

        # -------- eval --------
        if env["main"]:
            model.eval()
            agg_density = {}    # full density metrics
            agg_bnd     = {}    # legacy bnd metrics
            vis_rgb, vis_sim, vis_real, vis_gt, vis_pred = [], [], [], [], []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="eval", leave=False):
                    rgb    = batch["rgb"].to(env["device"], non_blocking=True)
                    sim_d  = batch["sim_depth"].to(env["device"], non_blocking=True)
                    real_d = batch.get("real_depth", sim_d).to(env["device"], non_blocking=True)
                    gt_raw = batch["hole_mask"].to(env["device"], non_blocking=True)
                    m_tgt  = batch["m_target"].to(env["device"], non_blocking=True)
                    m_ign  = batch["m_ignore"].to(env["device"], non_blocking=True)
                    with autocast("cuda"):
                        out = model(rgb, sim_d)
                    pred_p = out["pred_failure"].float()

                    # Density metrics (raw GT + M_target views, ECE, AUROC, ...)
                    dens = evaluate_density_full(pred_p, gt_raw, m_tgt, m_ign)
                    for k, v in dens.items():
                        agg_density.setdefault(k, []).append(v)

                    # Legacy BND metrics for backward comparison
                    bnd = evaluate_bnd(pred_p, gt_raw)
                    for k, v in bnd.items():
                        agg_bnd.setdefault(k, []).append(v)

                    if sum(b.shape[0] for b in vis_rgb) < vis_n:
                        need = vis_n - sum(b.shape[0] for b in vis_rgb)
                        vis_rgb.append(rgb[:need].detach().float().cpu())
                        vis_sim.append(sim_d[:need].detach().float().cpu())
                        vis_real.append(real_d[:need].detach().float().cpu())
                        vis_gt.append(gt_raw[:need].detach().float().cpu())
                        vis_pred.append(pred_p[:need].detach().float().cpu())

            metrics = {"epoch": epoch, "train_loss": sum_t}
            for k, vs in agg_density.items():
                metrics[k] = float(np.mean(vs))
            for k, vs in agg_bnd.items():
                if k not in metrics:
                    metrics[k] = float(np.mean(vs))
            metrics.update(sum_comps)
            history.append(metrics)

            # === Print key line (most important indicators) ===
            print(f"Ep{epoch:03d} | loss={sum_t:.4f}"
                  f" | iou_raw@0.5={metrics.get('iou@0.5', metrics.get('inv_iou', 0)):.4f}"
                  f" iou_tgt@0.5={metrics.get('iou_on_target@0.5', 0):.4f}"
                  f" | iou@0.3={metrics.get('iou@0.3', 0):.4f}"
                  f" iou@0.7={metrics.get('iou@0.7', 0):.4f}"
                  f" | ECE={metrics.get('ece', 0):.4f}"
                  f" AUROC={metrics.get('auroc', 0):.4f}"
                  f" Brier={metrics.get('brier', 0):.4f}"
                  f" | P@0.5={metrics.get('P@0.5', metrics.get('precision', 0)):.3f}"
                  f" R@0.5={metrics.get('R@0.5', metrics.get('recall', 0)):.3f}"
                  f" F1@0.5={metrics.get('F1@0.5', metrics.get('f1', 0)):.3f}")

            if WANDB_OK and not args.debug and os.environ.get("WANDB_MODE") != "disabled":
                wandb.log(metrics)

            # Visualisation
            if vis_rgb:
                v_rgb  = torch.cat(vis_rgb, dim=0)
                v_sim  = torch.cat(vis_sim, dim=0)
                v_real = torch.cat(vis_real, dim=0)
                v_gt   = torch.cat(vis_gt, dim=0)
                v_pred = torch.cat(vis_pred, dim=0)
                ep_dir = vis_dir / f"ep_{epoch:03d}"
                try:
                    dump_eval_batch(ep_dir, v_rgb, v_sim, v_real, v_gt, v_pred, threshold=0.5)
                except Exception as e:
                    print(f"  [warn] vis dump failed: {e}")

            mdl = model.module if env["dist"] else model
            iou_target = metrics.get("iou_on_target@0.5", 0.0)
            if iou_target > best_iou_target:
                best_iou_target = iou_target
                torch.save({"epoch": epoch, "model": mdl.state_dict(), "metrics": metrics},
                           out_dir / "best.pt")
            if epoch % 10 == 0:
                torch.save({"epoch": epoch, "model": mdl.state_dict()},
                           out_dir / f"epoch_{epoch:03d}.pt")

    if env["main"]:
        mdl = model.module if env["dist"] else model
        torch.save({"epoch": epochs, "model": mdl.state_dict()}, out_dir / "final.pt")
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nDone. Best iou_on_target@0.5 = {best_iou_target:.4f}  ->  {out_dir}")

    if env["dist"]:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
