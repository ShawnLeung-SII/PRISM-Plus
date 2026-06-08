"""Evaluation metrics for PRISM and PRISM+.

Includes both PRISM (ICML) original metrics and PRISM+ (TPAMI) new metrics.

PRISM original:
    * inv_iou             – invalidation IoU (hole mask overlap)
    * standard depth      – MAE / RMSE / AbsRel / delta<1.25

PRISM+ new (for TPAMI):
    * boundary_mae        – MAE within r px of GT hole boundary  (C2)
    * flying_pixel_rate   – valid pixels adjacent to holes with err > k × global_mae
    * boundary_iou        – Boundary IoU (Cheng+ CVPR'21 definition, fixed) (C1)
    * tnfr                – Temporal Noise Flicker Rate (video sequences, C4)

PRISM+ NEW (for diagnosing precision/recall imbalance, added 2026-06-08):
    * precision_recall_f1 – per-pixel precision, recall, F1
    * coverage_on_det     – recall on STRUCTURED failure (open(GT,k))
    * coverage_on_noise   – recall on RANDOM dots (GT - open(GT,k))
    * fp_density          – false positive density (FP / (H*W - GT_pos))
    * prob_polarisation   – fraction of predicted prob in [0.2, 0.8] (lower=more polarised)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers — accept binary {0,1} or float, treat >0.5 as positive.
# All ops are GPU-friendly and vectorised.
# ---------------------------------------------------------------------------

def _as_binary(t: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Convert prob or already-binary tensor into hard {0,1} float."""
    return (t > thresh).float()


def _dilate_bin(mask01: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """Binary dilation via max-pool (kernel = 2r+1). Input must be 0/1."""
    k = 2 * radius + 1
    return F.max_pool2d(mask01, k, stride=1, padding=radius)


def _erode_bin(mask01: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """Binary erosion via min-pool. Input must be 0/1."""
    k = 2 * radius + 1
    return -F.max_pool2d(-mask01, k, stride=1, padding=radius)


def _morph_open(mask01: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Open = erode then dilate; removes isolated dots smaller than (2r+1)²."""
    return _dilate_bin(_erode_bin(mask01, radius), radius)


def _boundary_of(mask01: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """1-px-wide boundary band of a binary mask (dilate ⊕ erode)."""
    dil = _dilate_bin(mask01, radius)
    ero = _erode_bin(mask01, radius)
    return (dil - ero).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Core IoU family
# ---------------------------------------------------------------------------

def inv_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Invalidation IoU.

    Both tensors are [B,1,H,W]; ``pred_mask`` may be sigmoid prob OR binary
    (any value > ``threshold`` counts as positive). ``gt_mask`` is treated
    as binary the same way (robust to soft labels / smoothed GT).

    Returns scalar = mean per-sample IoU. Samples whose union is empty are
    excluded from the mean (instead of contributing 0, which used to bias
    the metric downward).
    """
    pred_b = _as_binary(pred_mask, threshold)
    gt_b   = _as_binary(gt_mask, 0.5)
    inter  = (pred_b * gt_b).sum(dim=[1, 2, 3])
    union  = ((pred_b + gt_b) > 0).float().sum(dim=[1, 2, 3])

    # Mask out samples with no GT positives AND no predictions
    valid_sample = union > 0
    if valid_sample.sum() == 0:
        return pred_mask.new_zeros(())
    per_sample = inter / (union + eps)
    return per_sample[valid_sample].mean()


def boundary_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    radius: int = 2,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Boundary IoU (Cheng et al., CVPR 2021) – the standard definition.

    Boundary IoU = | (pred ∩ pred_boundary) ∩ (gt ∩ gt_boundary) |
                  / | (pred ∩ pred_boundary) ∪ (gt ∩ gt_boundary) |

    where ``X_boundary`` is the 1-px-wide boundary of X. Each side uses
    its OWN boundary (the previous implementation shared GT's boundary,
    which gave a systematic under-estimate).

    ``radius`` controls boundary thickness; CVPR'21 uses 2 (≈ 0.4% × diag).
    """
    pred_b = _as_binary(pred_mask, threshold)
    gt_b   = _as_binary(gt_mask,   0.5)

    pred_boundary = _boundary_of(pred_b, radius)
    gt_boundary   = _boundary_of(gt_b,   radius)

    p_b = pred_b * pred_boundary
    g_b = gt_b   * gt_boundary

    inter = (p_b * g_b).sum(dim=[1, 2, 3])
    union = ((p_b + g_b) > 0).float().sum(dim=[1, 2, 3])

    valid_sample = union > 0
    if valid_sample.sum() == 0:
        return pred_mask.new_zeros(())
    return (inter / (union + eps))[valid_sample].mean()


# ---------------------------------------------------------------------------
# Precision / Recall / F1  (added 2026-06-08 to diagnose precision deficit)
# ---------------------------------------------------------------------------

def precision_recall_f1(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Pixel-level precision, recall, F1 for binary segmentation.

    Reports the BATCH-AVERAGED metric (computed once over all pixels),
    not per-sample averaged — this is the standard reporting form for
    sparse-positive tasks like ours.
    """
    p = _as_binary(pred_mask, threshold)
    g = _as_binary(gt_mask,   0.5)
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Structural-vs-Noise decomposition coverage (added for diagnosis)
# ---------------------------------------------------------------------------

def coverage_decomposed(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    kernel: int = 3,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """How well does the prediction cover structured failure vs random dots?

    Decomposes GT into:
        gt_det   = open(gt, kernel)         # structured failure (recoverable)
        gt_noise = gt - gt_det              # isolated dots (mostly unrecoverable)

    Returns:
        coverage_on_det:    recall on gt_det   (target: as close to 1.0 as possible)
        coverage_on_noise:  recall on gt_noise (low value = healthy, model didn't
                                                 over-fit to per-pixel noise)
        noise_fraction:     gt_noise / gt_raw  (how noisy the dataset itself is)
        fp_density:         FP / (H*W - GT_pos)  (lower = better precision)
    """
    p = _as_binary(pred_mask, threshold)
    g = _as_binary(gt_mask,   0.5)
    g_det   = _morph_open(g, radius=(kernel - 1) // 2)
    g_noise = (g - g_det).clamp(0, 1)

    cov_det   = (p * g_det).sum()   / (g_det.sum()   + eps)
    cov_noise = (p * g_noise).sum() / (g_noise.sum() + eps)
    noise_fr  = g_noise.sum() / (g.sum() + eps)

    # FP density over true-negative pixels
    fp = (p * (1 - g)).sum()
    neg = (1 - g).sum()
    fp_dens = fp / (neg + eps)

    return {
        "coverage_on_det":   cov_det,
        "coverage_on_noise": cov_noise,
        "noise_fraction":    noise_fr,
        "fp_density":        fp_dens,
    }


# ---------------------------------------------------------------------------
# Prob polarisation – diagnose "hedging" behaviour
# ---------------------------------------------------------------------------

def prob_polarisation(
    pred_prob: torch.Tensor,
    low: float = 0.2,
    high: float = 0.8,
) -> torch.Tensor:
    """Fraction of pixels whose predicted prob is in [low, high].

    A confident binary classifier polarises its outputs to ≈ 0 or ≈ 1.
    A "hedging" classifier (the failure mode you suspected) leaves a lot
    of mass in [0.2, 0.8]. Tracking this each epoch is cheap and tells
    us immediately if asymmetric BCE / sharpness penalty is working.

    Returns a scalar in [0, 1]; lower = better polarisation.
    """
    p = pred_prob.clamp(0.0, 1.0)
    return ((p > low) & (p < high)).float().mean()


# ---------------------------------------------------------------------------
# Boundary-MAE — depth MAE in the boundary band
# ---------------------------------------------------------------------------

def boundary_mae(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_hole_mask: torch.Tensor,
    radius: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Depth MAE within ``radius`` px of GT hole boundary, on VALID pixels
    only (excludes pixels inside holes).

    ``gt_hole_mask`` is treated as binary via _as_binary.
    """
    gt_b = _as_binary(gt_hole_mask, 0.5)
    # boundary band = pixels close to the hole/valid transition
    dil_hole  = _dilate_bin(gt_b,       radius)
    dil_valid = _dilate_bin(1.0 - gt_b, radius)
    band = (dil_hole * dil_valid).clamp(0, 1)
    valid_in_band = (1.0 - gt_b) * band

    mae = torch.abs(pred_depth - gt_depth) * valid_in_band
    return mae.sum() / (valid_in_band.sum() + eps)


# ---------------------------------------------------------------------------
# Flying-pixel rate
# ---------------------------------------------------------------------------

def flying_pixel_rate(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_hole_mask: torch.Tensor,
    multiplier: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fraction of valid pixels adjacent to holes whose depth error exceeds
    ``multiplier`` × the global MAE."""
    gt_b = _as_binary(gt_hole_mask, 0.5)
    valid = 1.0 - gt_b
    adj   = _dilate_bin(gt_b, 1) * valid     # valid pixels adjacent to a hole

    err = torch.abs(pred_depth - gt_depth)
    global_mae = (err * valid).sum() / (valid.sum() + eps)
    flying = (err > multiplier * global_mae).float() * adj
    return flying.sum() / (adj.sum() + eps)


# ---------------------------------------------------------------------------
# Temporal Noise Flicker Rate
# ---------------------------------------------------------------------------

def tnfr(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fraction of pixels where the predicted mask flips between frames
    while GT did NOT change.

    Inputs are [B, T, 1, H, W]. Now binarised internally (previous impl
    used float != which is bit-fragile)."""
    assert pred_masks.shape == gt_masks.shape and pred_masks.ndim == 5
    p = _as_binary(pred_masks, threshold)
    g = _as_binary(gt_masks,   0.5)

    pred_flip = (p[:, 1:] != p[:, :-1]).float()
    gt_stable = (g[:, 1:] == g[:, :-1]).float()
    flicker = pred_flip * gt_stable
    return flicker.sum() / (gt_stable.sum() + eps)


# ---------------------------------------------------------------------------
# All-in-one evaluation
# ---------------------------------------------------------------------------

def evaluate_bnd(
    pred_prob: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
    boundary_radius: int = 2,
    noise_kernel: int = 3,
) -> Dict[str, float]:
    """One-shot evaluation for BND. Returns a flat dict of scalars.

    Use this in train_bnd_plus.py's eval loop in place of calling each
    metric individually."""
    res = {
        "inv_iou":          inv_iou(pred_prob, gt_mask, threshold).item(),
        "boundary_iou":     boundary_iou(pred_prob, gt_mask, boundary_radius, threshold).item(),
        "prob_polarisation": prob_polarisation(pred_prob).item(),
    }
    prf = precision_recall_f1(pred_prob, gt_mask, threshold)
    res.update({k: v.item() for k, v in prf.items()})
    cov = coverage_decomposed(pred_prob, gt_mask, noise_kernel, threshold)
    res.update({k: v.item() for k, v in cov.items()})
    return res


# ---------------------------------------------------------------------------
# Standard depth metrics (unchanged)
# ---------------------------------------------------------------------------

def compute_depth_metrics_full(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_hole_mask: torch.Tensor,
    max_depth: float = 5.0,
) -> dict:
    valid = (1.0 - _as_binary(gt_hole_mask, 0.5)) \
            * (gt_depth > 0.05).float() * (gt_depth < max_depth).float()
    eps = 1e-6

    err = torch.abs(pred_depth - gt_depth)
    mae      = (err * valid).sum() / (valid.sum() + eps)
    rmse     = ((err**2 * valid).sum() / (valid.sum() + eps)).sqrt()
    abs_rel  = ((err / gt_depth.clamp(min=eps)) * valid).sum() / (valid.sum() + eps)
    ratio    = torch.max(pred_depth / gt_depth.clamp(min=eps),
                         gt_depth / pred_depth.clamp(min=eps))
    delta125 = ((ratio < 1.25).float() * valid).sum() / (valid.sum() + eps)

    return {
        "mae":               mae.item(),
        "rmse":              rmse.item(),
        "abs_rel":           abs_rel.item(),
        "delta1.25":         delta125.item(),
        "boundary_mae":      boundary_mae(pred_depth, gt_depth, gt_hole_mask).item(),
        "flying_pixel_rate": flying_pixel_rate(pred_depth, gt_depth, gt_hole_mask).item(),
    }
