"""H-PPS: Hierarchical Positive-Prioritized Supervision for BND.

Reference: PRISM (ICML 2026), Section 3.4 / Equations (7)-(9).

Combines three mechanisms:
    1. Multi-scale positive-weighted BCE (boost recall on rare hole pixels).
    2. PP-DHM (Positive-Preserving Dynamic Hard Mining) - all positives kept,
       only hardest fraction of negatives back-propagated.
    3. Dice loss on the main output for shape compactness.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def positive_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 3.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE that boosts positive (hole) pixels by ``pos_weight``."""
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), pos_weight=pw, reduction=reduction
    )


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard 2D Dice loss over the full mask (no class weighting)."""
    p = prob.flatten(1)
    t = target.to(p.dtype).flatten(1)
    inter = (p * t).sum(1)
    return (1.0 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def pp_dhm_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 3.0,
    hard_neg_ratio: float = 0.25,
) -> torch.Tensor:
    """Positive-Preserving Dynamic Hard Mining BCE.

    All positive pixels contribute their BCE; among negatives, only the
    top-``hard_neg_ratio`` fraction by per-pixel loss is back-propagated.
    """
    raw = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype), reduction="none"
    )
    pos_mask = (target > 0.5).flatten(1)        # [B, N]
    raw_flat = raw.flatten(1)
    B, N = raw_flat.shape
    loss_per_sample = torch.zeros(B, device=raw.device)
    for b in range(B):
        pos = raw_flat[b][pos_mask[b]] * pos_weight
        neg = raw_flat[b][~pos_mask[b]]
        if neg.numel() > 0:
            k = max(1, int(neg.numel() * hard_neg_ratio))
            neg = neg.topk(k).values
        loss_per_sample[b] = torch.cat([pos, neg]).mean() if (pos.numel() + neg.numel()) > 0 else raw_flat[b].mean()
    return loss_per_sample.mean()


class HPPSLoss(nn.Module):
    """End-to-end H-PPS loss used in PRISM BND training.

    Args:
        pos_weight: BCE positive weight (Eq. 7).
        dice_weight: weight of the dice term (default 0.3).
        aux_weight: weight applied to auxiliary (deep-supervision) heads.
        use_dhm: enable PP-DHM (off by default for simplicity).
        hard_neg_ratio: gamma in PP-DHM (decays 0.25 -> 0.1 in PRISM ICML).
    """

    def __init__(
        self,
        pos_weight: float = 3.0,
        dice_weight: float = 0.3,
        aux_weight: float = 0.5,
        use_dhm: bool = False,
        hard_neg_ratio: float = 0.25,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.use_dhm = use_dhm
        self.hard_neg_ratio = hard_neg_ratio

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1) main BCE
        main_logits = pred_dict["failure_logits"]
        if self.use_dhm:
            main_loss = pp_dhm_bce(main_logits, gt_mask, self.pos_weight, self.hard_neg_ratio)
        else:
            main_loss = positive_weighted_bce(main_logits, gt_mask, self.pos_weight)

        # 2) dice on main probabilities
        main_prob = torch.sigmoid(main_logits)
        main_dice = dice_loss(main_prob, gt_mask)

        loss = main_loss + self.dice_weight * main_dice

        # 3) deep supervision (if available)
        aux_logits = pred_dict.get("aux_failure_logits")
        if aux_logits is not None:
            for aux in aux_logits:
                gt_s = F.interpolate(
                    gt_mask.to(aux.dtype), size=aux.shape[-2:], mode="nearest"
                )
                loss = loss + self.aux_weight * positive_weighted_bce(aux, gt_s, self.pos_weight)

        return loss
