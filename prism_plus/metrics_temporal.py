"""PRISM+ C4 — Temporal evaluation metrics: TNFR + Temporal-IoU.

TNFR (Temporal Noise Flicker Rate):
    Fraction of pixels whose binary mask flips between consecutive frames
    where the underlying GT did NOT flip (i.e. spurious mask flicker).

    For each pair (t-1, t):
        flip_pred = (M_pred_t != M_pred_{t-1})
        flip_gt   = (M_gt_t   != M_gt_{t-1})
        flicker   = flip_pred & ~flip_gt
        TNFR = mean(flicker)

Temporal-IoU:
    Mean GT-mask IoU averaged across consecutive frame pairs.
    Pred-mask matches GT consistently if Temporal-IoU is high.

Both expect lists/tensors of [T, B, 1, H, W] over a video clip.
"""
from __future__ import annotations
from typing import Dict, List

import torch


@torch.no_grad()
def temporal_noise_flicker_rate(
    pred_seq: torch.Tensor,
    gt_seq: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """TNFR over a video clip.

    Args:
        pred_seq : [T, B, 1, H, W] in [0,1]
        gt_seq   : [T, B, 1, H, W] binary {0, 1}
        threshold: binarisation threshold for pred_seq
    """
    if pred_seq.shape != gt_seq.shape:
        raise ValueError(f'pred {tuple(pred_seq.shape)} vs gt {tuple(gt_seq.shape)}')
    T = pred_seq.shape[0]
    if T < 2:
        return 0.0

    pred_bin = (pred_seq > threshold).float()
    gt_bin   = (gt_seq   > 0.5).float()

    flicker_count = 0.0
    pixel_count = 0.0
    for t in range(1, T):
        flip_pred = (pred_bin[t] != pred_bin[t - 1]).float()
        flip_gt   = (gt_bin[t]   != gt_bin[t - 1]).float()
        spurious  = flip_pred * (1.0 - flip_gt)
        flicker_count += spurious.sum().item()
        pixel_count += spurious.numel()
    return flicker_count / max(pixel_count, 1.0)


@torch.no_grad()
def temporal_iou(
    pred_seq: torch.Tensor,
    gt_seq: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Mean per-frame IoU averaged across a clip."""
    T = pred_seq.shape[0]
    pred_bin = (pred_seq > threshold).float()
    gt_bin   = (gt_seq   > 0.5).float()
    ious = []
    for t in range(T):
        inter = (pred_bin[t] * gt_bin[t]).sum(dim=[1, 2, 3])
        union = ((pred_bin[t] + gt_bin[t]) > 0).float().sum(dim=[1, 2, 3])
        valid = union > 0
        if valid.any():
            ious.append((inter[valid] / (union[valid] + eps)).mean().item())
    return float(sum(ious) / max(len(ious), 1))


def evaluate_temporal(
    pred_seq: torch.Tensor,
    gt_seq: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """One-shot wrapper: returns both TNFR and Temporal-IoU as a flat dict."""
    return {
        'tnfr':         temporal_noise_flicker_rate(pred_seq, gt_seq, threshold),
        'temporal_iou': temporal_iou(pred_seq, gt_seq, threshold),
    }
