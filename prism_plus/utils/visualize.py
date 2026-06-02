"""Eval-time visualisation utility.

Saves a small grid of N samples per eval epoch. Used by tools/train_bnd_plus.py
to bridge the "metrics-vs-visual-quality" gap.

For each sample we save:
    rgb.png                  – RGB input
    sim_depth.png            – clean simulation depth (Spectral_r)
    real_depth_gt_hole.png   – real depth, holes from GT
    real_depth_pred_hole.png – real depth, holes from prediction
    pred_prob.png            – probability heatmap (hot cmap)
    pred_mask.png            – binary at threshold 0.5
    gt_mask.png              – GT binary mask
    overlay_tp_fp_fn.png     – TP=green, FP=red, FN=blue on RGB

Plus one combined grid:
    grid.png                 – 2x4 panel for quick scrolling
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """[3,H,W] or [H,W,3] tensor in [0,1] -> uint8 HxWx3."""
    if t.ndim == 3 and t.shape[0] == 3:
        a = t.detach().cpu().permute(1, 2, 0).numpy()
    elif t.ndim == 3 and t.shape[-1] == 3:
        a = t.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected RGB shape {tuple(t.shape)}")
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


def _to_2d(t: torch.Tensor) -> np.ndarray:
    """[1,H,W] or [H,W] tensor -> 2D numpy."""
    if t.ndim == 3 and t.shape[0] == 1:
        return t.detach().cpu().squeeze(0).numpy()
    if t.ndim == 2:
        return t.detach().cpu().numpy()
    raise ValueError(f"Expected 2D-ish tensor, got {tuple(t.shape)}")


def _colorize_depth(d: np.ndarray, cmap: str = "Spectral_r",
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None) -> np.ndarray:
    """Map a 2D depth array to RGB via matplotlib colormap."""
    d = d.astype(np.float32)
    if vmin is None: vmin = float(np.nanmin(d))
    if vmax is None: vmax = float(np.nanmax(d)) if np.nanmax(d) > vmin else vmin + 1e-6
    n = (d - vmin) / max(vmax - vmin, 1e-6)
    n = np.clip(n, 0, 1)
    rgb = (plt.get_cmap(cmap)(n)[..., :3] * 255).astype(np.uint8)
    return rgb


def _depth_with_hole_mask(d: np.ndarray, mask01: np.ndarray,
                           vmin: float, vmax: float,
                           cmap: str = "Spectral_r") -> np.ndarray:
    """Color depth where mask==0, paint mask==1 as black."""
    rgb = _colorize_depth(d, cmap, vmin, vmax)
    hole = mask01.astype(bool)
    rgb[hole] = (0, 0, 0)
    return rgb


def _tp_fp_fn_overlay(rgb_uint8: np.ndarray,
                       pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Green=TP, Red=FP, Blue=FN (RGB-uint8 base)."""
    out = rgb_uint8.copy()
    pm = pred_mask.astype(bool)
    gm = gt_mask.astype(bool)
    out[pm & gm]   = (0, 255, 0)
    out[pm & ~gm]  = (255, 0, 0)
    out[~pm & gm]  = (0, 0, 255)
    return out


# ---------------------------------------------------------------------------
# Per-sample / per-epoch dump
# ---------------------------------------------------------------------------

def save_eval_sample(
    out_dir: Path,
    sample_idx: int,
    rgb: torch.Tensor,
    sim_depth: torch.Tensor,
    real_depth: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_prob: torch.Tensor,
    threshold: float = 0.5,
    sample_id: Optional[str] = None,
) -> None:
    """Save a complete visualisation for one sample.

    All tensors are single samples (no batch dim). Move to CPU before calling.
    """
    folder = out_dir / (sample_id if sample_id else f"sample_{sample_idx:02d}")
    folder.mkdir(parents=True, exist_ok=True)

    rgb_u8     = _to_uint8_rgb(rgb)
    sim_d      = _to_2d(sim_depth)
    real_d     = _to_2d(real_depth)
    gt_m_arr   = _to_2d(gt_mask)
    pred_p_arr = _to_2d(pred_prob)
    pred_m_arr = (pred_p_arr > threshold).astype(np.float32)

    # Use GT valid region to fix the depth color range
    valid = (1.0 - gt_m_arr).astype(bool)
    if valid.any():
        vmin = float(np.nanpercentile(real_d[valid], 2))
        vmax = float(np.nanpercentile(real_d[valid], 98))
    else:
        vmin, vmax = float(real_d.min()), float(real_d.max())

    Image.fromarray(rgb_u8).save(folder / "rgb.png")
    Image.fromarray(_colorize_depth(sim_d, vmin=vmin, vmax=vmax)).save(folder / "sim_depth.png")
    Image.fromarray(_depth_with_hole_mask(real_d, gt_m_arr, vmin, vmax)).save(folder / "real_depth_gt_hole.png")
    Image.fromarray(_depth_with_hole_mask(real_d, pred_m_arr, vmin, vmax)).save(folder / "real_depth_pred_hole.png")
    Image.fromarray((gt_m_arr * 255).astype(np.uint8), mode="L").save(folder / "gt_mask.png")
    Image.fromarray((pred_m_arr * 255).astype(np.uint8), mode="L").save(folder / "pred_mask.png")
    Image.fromarray(_tp_fp_fn_overlay(rgb_u8, pred_m_arr, gt_m_arr)).save(folder / "overlay_tp_fp_fn.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(pred_p_arr, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(folder / "pred_prob.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Combined 2x4 grid for quick scrolling
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(rgb_u8);                                                axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(_colorize_depth(sim_d, vmin=vmin, vmax=vmax));          axes[0, 1].set_title("Sim Depth")
    axes[0, 2].imshow(_depth_with_hole_mask(real_d, gt_m_arr, vmin, vmax));   axes[0, 2].set_title("Real (GT hole)")
    axes[0, 3].imshow(_depth_with_hole_mask(real_d, pred_m_arr, vmin, vmax)); axes[0, 3].set_title("Real (Pred hole)")
    axes[1, 0].imshow(gt_m_arr, cmap="gray");                                 axes[1, 0].set_title("GT Mask")
    axes[1, 1].imshow(pred_m_arr, cmap="gray");                               axes[1, 1].set_title(f"Pred Mask (t={threshold})")
    axes[1, 2].imshow(pred_p_arr, cmap="hot", vmin=0, vmax=1);                axes[1, 2].set_title("Pred Prob")
    axes[1, 3].imshow(_tp_fp_fn_overlay(rgb_u8, pred_m_arr, gt_m_arr));       axes[1, 3].set_title("TP=G FP=R FN=B")
    for ax in axes.flat: ax.axis("off")
    fig.suptitle(folder.name, fontsize=14)
    fig.tight_layout()
    fig.savefig(folder / "grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def dump_eval_batch(
    epoch_dir: Path,
    rgb: torch.Tensor,            # [N,3,H,W]
    sim_depth: torch.Tensor,      # [N,1,H,W]
    real_depth: torch.Tensor,     # [N,1,H,W]
    gt_mask: torch.Tensor,        # [N,1,H,W]
    pred_prob: torch.Tensor,      # [N,1,H,W]
    threshold: float = 0.5,
    sample_ids: Optional[Iterable[str]] = None,
) -> None:
    """Save up to N samples; called once per epoch."""
    epoch_dir.mkdir(parents=True, exist_ok=True)
    N = rgb.shape[0]
    ids = list(sample_ids) if sample_ids else [f"sample_{i:02d}" for i in range(N)]
    for i in range(N):
        save_eval_sample(
            out_dir=epoch_dir, sample_idx=i,
            rgb=rgb[i].cpu(),
            sim_depth=sim_depth[i].cpu(),
            real_depth=real_depth[i].cpu(),
            gt_mask=gt_mask[i].cpu(),
            pred_prob=pred_prob[i].cpu(),
            threshold=threshold,
            sample_id=ids[i] if i < len(ids) else None,
        )
