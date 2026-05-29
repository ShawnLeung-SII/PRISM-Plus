"""Evaluation metrics for PRISM and PRISM+.

Includes both PRISM (ICML) original metrics and PRISM+ (TPAMI) new metrics.

PRISM original:
    * inv_iou             – invalidation IoU (hole mask overlap)
    * standard depth      – MAE / RMSE / AbsRel / delta<1.25 (in compute_depth_metrics_full)

PRISM+ new (for TPAMI):
    * boundary_mae        – MAE within radius px of GT hole boundary (isolates C2)
    * flying_pixel_rate   – valid pixels adjacent to holes with err > k * global_mae
    * boundary_iou        – IoU computed only on boundary region (for C1 thin-geom eval)
    * tnfr                – Temporal Noise Flicker Rate (video sequences, C4)


Implements:
  - boundary_mae()      : MAE within 5px of GT hole boundary
  - flying_pixel_rate() : fraction of valid pixels adj to holes with large depth error
  - tnfr()              : Temporal Noise Flicker Rate (video sequences)
  - inv_iou()           : Invalidation IoU (hole mask overlap)
  - boundary_iou()      : IoU computed on boundary region only
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dilate_mask(mask: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """Binary dilation of a [B,1,H,W] mask by `radius` pixels."""
    k = 2 * radius + 1
    kernel = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype)
    dilated = F.conv2d(mask.float(), kernel, padding=radius)
    return (dilated > 0).float()


def _boundary_region(gt_mask: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """
    Returns a binary mask of the region within `radius` pixels of the
    GT hole boundary (transition between hole and valid).
    """
    dilated_hole  = _dilate_mask(gt_mask, radius)
    dilated_valid = _dilate_mask(1.0 - gt_mask, radius)
    boundary = dilated_hole * dilated_valid   # intersection = boundary region
    return boundary


# ---------------------------------------------------------------------------
# Boundary-MAE
# ---------------------------------------------------------------------------

def boundary_mae(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_hole_mask: torch.Tensor,
    radius: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean Absolute Error of depth prediction within `radius` pixels of GT
    hole boundary, evaluated only on valid (non-hole) pixels.

    Args:
        pred_depth:   [B, 1, H, W]  predicted depth (meters)
        gt_depth:     [B, 1, H, W]  ground-truth real depth (meters)
        gt_hole_mask: [B, 1, H, W]  1=hole, 0=valid
        radius:       boundary dilation radius (default 5px per TPAMI plan)

    Returns:
        Scalar tensor: mean boundary-MAE across batch
    """
    boundary = _boundary_region(gt_hole_mask, radius)        # [B,1,H,W]
    valid = (1.0 - gt_hole_mask) * boundary                  # valid pixels near boundary
    mae = torch.abs(pred_depth - gt_depth) * valid
    return mae.sum() / (valid.sum() + eps)


# ---------------------------------------------------------------------------
# Flying-Pixel Rate
# ---------------------------------------------------------------------------

def flying_pixel_rate(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_hole_mask: torch.Tensor,
    multiplier: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fraction of valid pixels adjacent to holes whose depth error exceeds
    `multiplier` × the global MAE (flying pixel criterion).

    Args:
        pred_depth:   [B, 1, H, W]
        gt_depth:     [B, 1, H, W]
        gt_hole_mask: [B, 1, H, W]  1=hole, 0=valid
        multiplier:   threshold = multiplier × global_mae (default 2.0)

    Returns:
        Scalar tensor: flying pixel rate (fraction)
    """
    valid = 1.0 - gt_hole_mask
    adj_to_hole = _dilate_mask(gt_hole_mask, radius=1) * valid    # valid pixels adjacent to holes

    err = torch.abs(pred_depth - gt_depth)
    global_mae = (err * valid).sum() / (valid.sum() + eps)
    threshold = multiplier * global_mae

    flying = (err > threshold) * adj_to_hole
    return flying.sum() / (adj_to_hole.sum() + eps)


# ---------------------------------------------------------------------------
# Invalidation IoU
# ---------------------------------------------------------------------------

def inv_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    IoU between predicted and GT hole masks (invalidation regions).

    Args:
        pred_mask: [B, 1, H, W]  predicted hole probability or binary mask
        gt_mask:   [B, 1, H, W]  GT hole mask (binary)
    """
    pred_bin = (pred_mask > threshold).float()
    intersection = (pred_bin * gt_mask).sum(dim=[1,2,3])
    union = ((pred_bin + gt_mask) > 0).float().sum(dim=[1,2,3])
    return (intersection / (union + eps)).mean()


# ---------------------------------------------------------------------------
# Boundary IoU
# ---------------------------------------------------------------------------

def boundary_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    radius: int = 5,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """IoU computed only within the boundary region (±radius of GT boundary)."""
    boundary = _boundary_region(gt_mask, radius)
    pred_bin = (pred_mask > threshold).float()
    pred_b = pred_bin * boundary
    gt_b   = gt_mask  * boundary
    intersection = (pred_b * gt_b).sum(dim=[1,2,3])
    union = ((pred_b + gt_b) > 0).float().sum(dim=[1,2,3])
    return (intersection / (union + eps)).mean()


# ---------------------------------------------------------------------------
# Temporal Noise Flicker Rate (TNFR)
# ---------------------------------------------------------------------------

def tnfr(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Temporal Noise Flicker Rate: fraction of pixels where the predicted mask
    flips between consecutive frames while the GT does NOT change.

    Measures temporal inconsistency of hole prediction on video sequences.

    Args:
        pred_masks: [B, T, 1, H, W]  predicted hole masks (binary), T frames
        gt_masks:   [B, T, 1, H, W]  GT hole masks (binary)

    Returns:
        Scalar tensor: TNFR (lower is better)
    """
    assert pred_masks.shape == gt_masks.shape
    assert pred_masks.ndim == 5, "Expected [B, T, 1, H, W]"

    # Consecutive-frame differences
    pred_flip = (pred_masks[:, 1:] != pred_masks[:, :-1]).float()  # [B,T-1,1,H,W]
    gt_stable = (gt_masks[:, 1:] == gt_masks[:, :-1]).float()      # GT did NOT change

    # Flicker = pred flipped but GT was stable
    flicker = pred_flip * gt_stable
    return flicker.sum() / (gt_stable.sum() + eps)


# ---------------------------------------------------------------------------
# Standard depth metrics (for compatibility)
# ---------------------------------------------------------------------------

def compute_depth_metrics_full(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_hole_mask: torch.Tensor,
    max_depth: float = 5.0,
) -> dict:
    """
    Full metric suite for PRISM+ evaluation.

    Returns dict with: mae, rmse, abs_rel, delta1.25,
                       inv_iou, boundary_mae, flying_pixel_rate
    """
    valid = (1.0 - gt_hole_mask) * (gt_depth > 0.05).float() * (gt_depth < max_depth).float()
    eps = 1e-6

    err = torch.abs(pred_depth - gt_depth)
    mae      = (err * valid).sum() / (valid.sum() + eps)
    rmse     = ((err**2 * valid).sum() / (valid.sum() + eps)).sqrt()
    abs_rel  = ((err / (gt_depth.clamp(min=eps))) * valid).sum() / (valid.sum() + eps)
    ratio    = torch.max(pred_depth / gt_depth.clamp(min=eps), gt_depth.clamp(min=eps) / pred_depth.clamp(min=eps))
    delta125 = ((ratio < 1.25).float() * valid).sum() / (valid.sum() + eps)

    return {
        'mae':               mae.item(),
        'rmse':              rmse.item(),
        'abs_rel':           abs_rel.item(),
        'delta1.25':         delta125.item(),
        'boundary_mae':      boundary_mae(pred_depth, gt_depth, gt_hole_mask).item(),
        'flying_pixel_rate': flying_pixel_rate(pred_depth, gt_depth, gt_hole_mask).item(),
    }
