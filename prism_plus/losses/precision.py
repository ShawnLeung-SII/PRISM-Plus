"""Precision-focused losses for PRISM+ BND v0.3.0.

Designed to fix the diagnosed failure mode of v0.2.0:
    coverage_on_det = 0.96  (recall is fine)
    IoU             = 0.62  (precision is the bottleneck)
    -> model is over-predicting; we need to penalise FP > FN.

Components:
    1. asymmetric_bce  – BCE where FP cost > FN cost (inverts pos_weight)
    2. tversky_loss    – soft Dice generalisation with α>β to favour precision
    3. sharpness_loss  – pushes predicted prob to {0, 1}, prevents hedging
    4. PrecisionFocusedLoss – combined module with adjustable weights
    5. Sobel + small-region weighting and OHEM imported from boundary.py / hpps.py
       (kept available for callers that want them, but NOT included by default —
        the diagnosis showed those mechanisms are not the actual root cause).

Reference: diagnose_noise_hypothesis.py result, 2026-06-08.
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Asymmetric BCE — heavier penalty on FP than FN
# ---------------------------------------------------------------------------

def asymmetric_bce(
    pred_logits: torch.Tensor,
    gt: torch.Tensor,
    fp_weight: float = 2.0,   # penalty on FP (predicting hole where there is none)
    fn_weight: float = 1.0,   # penalty on FN (missing a real hole)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted BCE that penalises FP more heavily than FN.

    Standard BCE = -[gt log p + (1-gt) log(1-p)]
    Here we re-weight the two halves:
        loss = -[fn_weight * gt * log p  +  fp_weight * (1-gt) * log(1-p)]

    A simple way to invert PyTorch's pos_weight (which boosts the positive
    half — exactly the opposite of what we want when recall ≫ precision).
    """
    pred = torch.sigmoid(pred_logits).clamp(eps, 1 - eps)
    g = gt.to(pred.dtype)

    # Positive term (FN penalty when g=1 and pred<1)
    loss_pos = -g * pred.log()                            # shape: pred
    # Negative term (FP penalty when g=0 and pred>0)
    loss_neg = -(1.0 - g) * (1.0 - pred).log()
    loss = fn_weight * loss_pos + fp_weight * loss_neg
    return loss.mean()


# ---------------------------------------------------------------------------
# Tversky loss — generalises Dice; α/β control FP/FN weighting
# ---------------------------------------------------------------------------

def tversky_loss(
    pred_prob: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 0.7,        # FP weight
    beta: float = 0.3,         # FN weight
    eps: float = 1e-6,
) -> torch.Tensor:
    """Tversky index loss; with α=β=0.5 this reduces to Dice loss.

    α > β  →  penalises FP more (improves precision)
    α < β  →  penalises FN more (improves recall, what Dice-like losses do)
    """
    p = pred_prob.flatten(1)
    t = gt.to(p.dtype).flatten(1)
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - tv).mean()


# ---------------------------------------------------------------------------
# Sharpness penalty — push predicted prob to {0, 1}
# ---------------------------------------------------------------------------

def sharpness_loss(pred_prob: torch.Tensor) -> torch.Tensor:
    """Penalises predicted probabilities in the [0.2, 0.8] hedging zone.

    Using p * (1 - p):
        p = 0.0 or 1.0  ->  0          (no penalty when confident)
        p = 0.5          ->  0.25       (maximum penalty when hedging)
    This is the differential entropy peak of a Bernoulli's variance.
    """
    return (pred_prob * (1.0 - pred_prob)).mean()


# ---------------------------------------------------------------------------
# Boundary BCE+Dice on extracted edge band (kept from v0.2.0)
# ---------------------------------------------------------------------------

def _erode(mask: torch.Tensor, r: int) -> torch.Tensor:
    return -F.max_pool2d(-mask.float(), 2 * r + 1, 1, padding=r)


def _dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    return F.max_pool2d(mask.float(), 2 * r + 1, 1, padding=r)


def gt_boundary_band(gt: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Boundary band of GT for use as a per-pixel weight map."""
    dil = _dilate(gt, radius)
    ero = _erode(gt,  radius)
    return (dil - ero).clamp(0.0, 1.0)


def boundary_weighted_bce(
    pred_logits: torch.Tensor,
    gt: torch.Tensor,
    radius: int = 2,
    band_weight: float = 4.0,
    fp_weight: float = 2.0,
    fn_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Asymmetric BCE with extra weight on the boundary band."""
    band = gt_boundary_band(gt, radius)
    weight = 1.0 + (band_weight - 1.0) * band

    pred = torch.sigmoid(pred_logits).clamp(eps, 1 - eps)
    g = gt.to(pred.dtype)
    raw = -(fn_weight * g * pred.log() + fp_weight * (1 - g) * (1 - pred).log())
    return (raw * weight).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class PrecisionFocusedLoss(nn.Module):
    """v0.3.0 default loss for PRISMPlusBND.

    L = w_abce  · asymmetric_bce(final_logits, gt)
      + w_tversky · tversky_loss(final_prob,   gt)
      + w_sharp · sharpness_loss(final_prob)
      + w_bedge · boundary_weighted_bce(final_logits, gt)
      + w_coarse · asymmetric_bce(coarse_logits, downsampled gt)
      + w_edge_sup · BCE(edge_logits, gt_boundary)      # weak edge head supervision

    All weights are tunable from YAML. Default values come from the
    diagnostic finding that PRECISION is the bottleneck.
    """
    def __init__(
        self,
        # main losses
        w_abce: float = 1.0,
        w_tversky: float = 0.5,
        w_sharp: float = 0.2,
        w_bedge: float = 0.5,
        # auxiliary
        w_coarse: float = 0.3,
        w_edge_sup: float = 0.3,
        # asymmetric BCE balance
        fp_weight: float = 2.0,
        fn_weight: float = 1.0,
        # tversky balance
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        # boundary band
        band_radius: int = 2,
        band_weight: float = 4.0,
        edge_radius: int = 1,
    ):
        super().__init__()
        self.w_abce = w_abce
        self.w_tversky = w_tversky
        self.w_sharp = w_sharp
        self.w_bedge = w_bedge
        self.w_coarse = w_coarse
        self.w_edge_sup = w_edge_sup
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.band_radius = band_radius
        self.band_weight = band_weight
        self.edge_radius = edge_radius

    def forward(
        self,
        final_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        final_prob = torch.sigmoid(final_logits)
        g = gt_mask.to(final_logits.dtype)

        # 1) asymmetric BCE (FP-heavy)
        l_abce = asymmetric_bce(
            final_logits, g, self.fp_weight, self.fn_weight
        )
        # 2) Tversky (precision-leaning Dice)
        l_tversky = tversky_loss(
            final_prob, g, self.tversky_alpha, self.tversky_beta
        )
        # 3) Sharpness (anti-hedging)
        l_sharp = sharpness_loss(final_prob)
        # 4) Boundary-weighted asymmetric BCE
        l_bedge = boundary_weighted_bce(
            final_logits, g, self.band_radius, self.band_weight,
            self.fp_weight, self.fn_weight,
        )

        # 5) Coarse supervision (downsampled GT)
        gt_c = F.interpolate(g, size=coarse_logits.shape[-2:], mode="nearest")
        l_coarse = asymmetric_bce(
            coarse_logits, gt_c, self.fp_weight, self.fn_weight
        )

        # 6) Edge head supervision (1-px GT boundary)
        gt_edge = gt_boundary_band(g, self.edge_radius)
        # NOTE: edge GT is even rarer than hole GT; keep pos_weight high
        l_edge_sup = F.binary_cross_entropy_with_logits(
            edge_logits, gt_edge,
            pos_weight=torch.tensor([5.0], device=edge_logits.device,
                                    dtype=edge_logits.dtype),
        )

        total = (self.w_abce    * l_abce
                 + self.w_tversky * l_tversky
                 + self.w_sharp   * l_sharp
                 + self.w_bedge   * l_bedge
                 + self.w_coarse  * l_coarse
                 + self.w_edge_sup * l_edge_sup)

        return {
            "loss":      total,
            "l_abce":    l_abce.detach(),
            "l_tversky": l_tversky.detach(),
            "l_sharp":   l_sharp.detach(),
            "l_bedge":   l_bedge.detach(),
            "l_coarse":  l_coarse.detach(),
            "l_edge_sup": l_edge_sup.detach(),
        }
