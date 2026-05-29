#!/usr/bin/env python3
"""PRISM+ BND training entrypoint (Stage 1).

Trains either:
    * BND          : PRISM baseline (ICML 2026)
    * SpatialBND   : PRISM+ C1 (default)

Usage
-----
Single GPU debug:
    python tools/train_bnd.py --config configs/stage1_bnd_spatial.yaml --debug

Multi-GPU DDP via torchrun:
    torchrun --nproc_per_node=4 tools/train_bnd.py \\
        --config configs/stage1_bnd_spatial.yaml --seed 42
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

# Make `prism_plus` importable when running as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data import ByteCamDepthDataset
from prism_plus.losses import HPPSLoss
from prism_plus.metrics import inv_iou, boundary_iou
from prism_plus.models import SpatialBND, create_spatial_bnd

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

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


def main():
    p = argparse.ArgumentParser(description="PRISM+ BND training (Stage 1)")
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

    out_dir = Path(cfg["output_dir"])
    if env["main"]:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model ----
    model = create_spatial_bnd(
        vfm_type=cfg.get("vfm_type", "moge2"),
        vfm_checkpoint=cfg.get("vfm_checkpoint"),
        encoder_size=cfg.get("encoder_size", "large"),
        deep_supervision=True,
    ).to(env["device"])

    # Freeze VFM; train rest
    for n, p_ in model.named_parameters():
        p_.requires_grad = ("vfm" not in n.lower()) or ("cross_attn" in n.lower())
    for p_ in model.vfm_cross_attns.parameters():
        p_.requires_grad = True

    if args.resume:
        ckpt = torch.load(args.resume, map_location=env["device"])
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    if env["dist"]:
        model = DDP(model, device_ids=[env["local"]], find_unused_parameters=False)

    # ---- Data ----
    data_root = cfg["data_root"]
    train_ds = ByteCamDepthDataset(data_root=data_root, split="train",
                                   resolution=cfg.get("resolution", 512), augment=True)
    val_ds   = ByteCamDepthDataset(data_root=data_root, split="val",
                                   resolution=cfg.get("resolution", 512), augment=False)

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

    # ---- Optimiser + loss ----
    epochs  = 5 if args.debug else cfg.get("epochs", 100)
    optim   = AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                    lr=cfg.get("lr", 1e-4), weight_decay=1e-4)
    sched   = CosineAnnealingLR(optim, T_max=epochs, eta_min=cfg.get("lr", 1e-4) * 0.01)
    scaler  = GradScaler()
    loss_fn = HPPSLoss(pos_weight=cfg.get("pos_weight", 3.0),
                       dice_weight=cfg.get("dice_weight", 0.3))

    if WANDB_OK and env["main"] and not args.debug and os.environ.get("WANDB_MODE") != "disabled":
        wandb.init(project="prism_plus", name=f"stage1_bnd_spatial_s{args.seed}", config=cfg)

    best_iou = 0.0
    history: list = []

    for epoch in range(1, epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        t_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Ep{epoch:03d}/train",
                          leave=False, disable=not env["main"]):
            rgb   = batch["rgb"].to(env["device"])
            sim_d = batch["sim_depth"].to(env["device"])
            gt_m  = batch["hole_mask"].to(env["device"])

            optim.zero_grad()
            with autocast("cuda"):
                out = model(rgb, sim_d)
                loss = loss_fn(out, gt_m)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            t_loss += loss.item()
        sched.step()
        t_loss /= len(train_loader)

        if env["main"]:
            model.eval()
            iou_l, biou_l = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="eval", leave=False):
                    rgb   = batch["rgb"].to(env["device"])
                    sim_d = batch["sim_depth"].to(env["device"])
                    gt_m  = batch["hole_mask"].to(env["device"])
                    with autocast("cuda"):
                        out = model(rgb, sim_d)
                    iou_l.append(inv_iou(out["pred_failure"], gt_m).item())
                    biou_l.append(boundary_iou(out["pred_failure"], gt_m).item())

            metrics = {"epoch": epoch, "train_loss": t_loss,
                       "inv_iou": float(np.mean(iou_l)),
                       "boundary_iou": float(np.mean(biou_l))}
            history.append(metrics)
            print(f"Ep{epoch:03d} | train_loss={t_loss:.4f} "
                  f"| inv_iou={metrics['inv_iou']:.4f} "
                  f"| boundary_iou={metrics['boundary_iou']:.4f}")

            if WANDB_OK and not args.debug and os.environ.get("WANDB_MODE") != "disabled":
                wandb.log(metrics)

            mdl = model.module if env["dist"] else model
            if metrics["inv_iou"] > best_iou:
                best_iou = metrics["inv_iou"]
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
        print(f"\nDone. Best inv_iou={best_iou:.4f}  ->  {out_dir}")

    if env["dist"]:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
