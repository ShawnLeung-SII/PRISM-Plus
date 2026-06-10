"""Calibration & ranking metrics for v0.4.0 density-aware BND.

Adds:
    * expected_calibration_error (ECE)
    * brier_score
    * auroc
    * precision_recall_curve
    * iou_multi_threshold
    * iou_on_target (IoU computed against M_target instead of GT_raw)

Designed to complement (not replace) inv_iou / boundary_iou in metrics.py.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Calibration: ECE
# ---------------------------------------------------------------------------

def expected_calibration_error(
    pred_prob: torch.Tensor,
    gt: torch.Tensor,
    n_bins: int = 15,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Expected Calibration Error.

    ECE = Σ_bin |bin_size / total| × |bin_accuracy - bin_confidence|

    Lower = better. A model that says "0.7 hole" should be hole 70% of the time.

    Args:
        pred_prob: [B,1,H,W] in [0,1]
        gt:        [B,1,H,W] binary
        mask:      [B,1,H,W] in {0,1}; pixels with mask=0 are excluded
    """
    p = pred_prob.flatten().detach().cpu().numpy()
    g = (gt > 0.5).float().flatten().detach().cpu().numpy()
    if mask is not None:
        m = mask.flatten().detach().cpu().numpy()
        p = p[m > 0]; g = g[m > 0]
    if p.size == 0:
        return torch.tensor(0.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n_bin = int(idx.sum())
        if n_bin == 0:
            continue
        bin_conf = float(p[idx].mean())
        bin_acc  = float(g[idx].mean())
        ece += (n_bin / p.size) * abs(bin_acc - bin_conf)
    return torch.tensor(ece)


# ---------------------------------------------------------------------------
# Calibration: Brier
# ---------------------------------------------------------------------------

def brier_score(
    pred_prob: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean squared error in probability space. Lower = better."""
    p = pred_prob.flatten().float()
    g = (gt > 0.5).float().flatten()
    err2 = (p - g) ** 2
    if mask is not None:
        m = mask.flatten().float()
        n = m.sum().clamp(min=1.0)
        return (err2 * m).sum() / n
    return err2.mean()


# ---------------------------------------------------------------------------
# Ranking: AUROC (threshold-free)
# ---------------------------------------------------------------------------

def auroc(
    pred_prob: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    n_subsample: int = 500_000,
) -> torch.Tensor:
    """ROC-AUC via Mann-Whitney U.

    For speed we subsample to `n_subsample` pixels (random) if total > that.
    """
    p = pred_prob.flatten().detach().cpu().numpy()
    g = (gt > 0.5).flatten().detach().cpu().numpy().astype(np.int8)
    if mask is not None:
        m = mask.flatten().detach().cpu().numpy()
        p = p[m > 0]; g = g[m > 0]
    if p.size == 0 or g.sum() == 0 or g.sum() == g.size:
        return torch.tensor(0.5)

    if p.size > n_subsample:
        idx = np.random.choice(p.size, n_subsample, replace=False)
        p = p[idx]; g = g[idx]

    # Wilcoxon / Mann-Whitney: rank-based
    order = np.argsort(p, kind="mergesort")
    g_sorted = g[order]
    n_pos = int(g_sorted.sum())
    n_neg = int((1 - g_sorted).sum())
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.5)
    # ranks: 1..N
    ranks = np.arange(1, len(g_sorted) + 1)
    sum_ranks_pos = ranks[g_sorted == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return torch.tensor(float(u / (n_pos * n_neg)))


# ---------------------------------------------------------------------------
# Precision–Recall curve
# ---------------------------------------------------------------------------

def precision_recall_curve(
    pred_prob: torch.Tensor,
    gt: torch.Tensor,
    thresholds: Optional[List[float]] = None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    """Precision and recall at a list of thresholds. Default thresholds:
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    p = pred_prob.flatten().float()
    g = (gt > 0.5).flatten().float()
    if mask is not None:
        m = mask.flatten().float()
        p = p[m > 0]; g = g[m > 0]
    if p.numel() == 0:
        return {"thresholds": thresholds,
                "precision":  [0.0] * len(thresholds),
                "recall":     [0.0] * len(thresholds),
                "f1":         [0.0] * len(thresholds)}

    eps = 1e-8
    out = {"thresholds": thresholds, "precision": [], "recall": [], "f1": []}
    for t in thresholds:
        pred_b = (p > t).float()
        tp = (pred_b * g).sum()
        fp = (pred_b * (1 - g)).sum()
        fn = ((1 - pred_b) * g).sum()
        prec = float(tp / (tp + fp + eps))
        rec  = float(tp / (tp + fn + eps))
        f1   = float(2 * prec * rec / (prec + rec + eps))
        out["precision"].append(prec)
        out["recall"].append(rec)
        out["f1"].append(f1)
    return out


# ---------------------------------------------------------------------------
# Multi-threshold IoU
# ---------------------------------------------------------------------------

def iou_multi_threshold(
    pred_prob: torch.Tensor,
    gt: torch.Tensor,
    thresholds: Optional[List[float]] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Returns {iou_at_t: value} for each threshold."""
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]
    g = (gt > 0.5).float()
    out = {}
    for t in thresholds:
        p = (pred_prob > t).float()
        inter = (p * g).sum(dim=[1, 2, 3])
        union = ((p + g) > 0).float().sum(dim=[1, 2, 3])
        valid = union > 0
        iou_val = (inter[valid] / (union[valid] + eps)).mean().item() if valid.any() else 0.0
        out[f"iou@{t}"] = iou_val
    return out


def iou_on_target(
    pred_prob: torch.Tensor,
    M_target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """IoU computed against M_target (i.e. denoised GT)."""
    p = (pred_prob > threshold).float()
    g = (M_target > 0.5).float()
    inter = (p * g).sum(dim=[1, 2, 3])
    union = ((p + g) > 0).float().sum(dim=[1, 2, 3])
    valid = union > 0
    return (inter[valid] / (union[valid] + eps)).mean().item() if valid.any() else 0.0


# ---------------------------------------------------------------------------
# All-in-one for v0.4.0 eval loop
# ---------------------------------------------------------------------------

def evaluate_density_full(
    pred_prob: torch.Tensor,
    gt_raw: torch.Tensor,
    M_target: torch.Tensor,
    M_ignore: torch.Tensor,
    iou_thresholds: Optional[List[float]] = None,
    pr_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """One-shot evaluation for density-aware BND.

    Returns a flat dict (everything float) ready for logging / wandb.
    """
    res: Dict[str, float] = {}

    # IoU at multiple thresholds vs raw GT (论文硬指标)
    res.update(iou_multi_threshold(pred_prob, gt_raw, iou_thresholds))

    # IoU @ 0.5 vs M_target (反映真实学习能力)
    res["iou_on_target@0.5"] = iou_on_target(pred_prob, M_target, 0.5)

    # Calibration & ranking (whole image)
    res["ece"]   = float(expected_calibration_error(pred_prob, gt_raw))
    res["brier"] = float(brier_score(pred_prob, gt_raw))
    res["auroc"] = float(auroc(pred_prob, gt_raw))

    # Same metrics on the supervised pixels only (excluding ignored noise)
    keep = (1.0 - M_ignore.float())
    res["ece_on_target"]   = float(expected_calibration_error(pred_prob, gt_raw, mask=keep))
    res["brier_on_target"] = float(brier_score(pred_prob, gt_raw, mask=keep))
    res["auroc_on_target"] = float(auroc(pred_prob, gt_raw, mask=keep))

    # Precision-Recall curve (returned as separate lists)
    pr = precision_recall_curve(pred_prob, gt_raw, pr_thresholds)
    for t, p, r, f in zip(pr["thresholds"], pr["precision"], pr["recall"], pr["f1"]):
        res[f"P@{t}"] = p
        res[f"R@{t}"] = r
        res[f"F1@{t}"] = f

    return res
