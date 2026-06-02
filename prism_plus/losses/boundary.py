"""Boundary-aware losses for PRISM+ BND.

Implements:
    - edge_bce_dice : BCE+Dice on boundary-only logits (forces edge supervision)
    - band_weighted_bce : weighted BCE concentrated in ±r px around GT boundary
    - signed_distance_loss : penalizes prediction at locations far from GT boundary
    - mask_boundary : extract 1-pixel GT boundary (used as target for edge_logits)
    - boundary_band : dilated boundary region (used to focus the refiner)
    - sobel : edge operator for use as a CNN input feature

These follow the design recommended by the GPT-5 architecture review
(refine-logs/BND_ARCHITECTURE_REVIEW.md).
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers — boundary extraction
# ---------------------------------------------------------------------------

def mask_boundary(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Extract a thin boundary band of `radius` pixels from a binary mask.

    boundary = dilate(mask) ⊕ erode(mask)
    Args:
        mask: [B, 1, H, W] binary in {0, 1}
    Returns:
        [B, 1, H, W] binary boundary mask
    """
    k = 2 * radius + 1
    pad = radius
    m = mask.float()
    # min-pool implements erosion; max-pool implements dilation
    dil = F.max_pool2d(m, k, stride=1, padding=pad)
    ero = -F.max_pool2d(-m, k, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def boundary_band(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Dilated boundary band (where the refiner is allowed to act).

    Returns the union of dilate(hole) ∧ dilate(valid) — i.e. the
    transition region within `radius` pixels of the GT boundary.
    """
    m = mask.float()
    dil_hole  = F.max_pool2d(m,        2 * radius + 1, 1, padding=radius)
    dil_valid = F.max_pool2d(1.0 - m,  2 * radius + 1, 1, padding=radius)
    return (dil_hole * dil_valid).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Sobel (depth-aware edge operator) for CNN BoundaryBranch input
# ---------------------------------------------------------------------------

class Sobel(nn.Module):
    """Fixed Sobel operator. Returns ||∇x||_2 magnitude."""

    def __init__(self):
        super().__init__()
        gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        gy = gx.t()
        self.register_buffer("k", torch.stack([gx, gy]).unsqueeze(1))  # [2,1,3,3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, 1, H, W] magnitude (averaged across C)."""
        B, C, H, W = x.shape
        # Apply same kernel per-channel, then average to 1 channel
        x_flat = x.reshape(B * C, 1, H, W)
        g = F.conv2d(x_flat, self.k, padding=1)        # [B*C, 2, H, W]
        mag = g.pow(2).sum(dim=1, keepdim=True).sqrt() # [B*C, 1, H, W]
        return mag.view(B, C, H, W).mean(dim=1, keepdim=True)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Functional helper (allocates kernel per call — prefer Sobel module)."""
    gx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=x.dtype)
    gy = gx.t()
    k = torch.stack([gx, gy]).unsqueeze(1)
    B, C, H, W = x.shape
    flat = x.reshape(B * C, 1, H, W)
    g = F.conv2d(flat, k, padding=1)
    mag = g.pow(2).sum(1, keepdim=True).sqrt()
    return mag.view(B, C, H, W).mean(1, keepdim=True)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def edge_bce_dice(
    edge_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    radius: int = 1,
    pos_weight: float = 5.0,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """BCE + Dice on the GT 1-pixel boundary.

    Forces the edge branch to predict where the boundary IS, which is what
    the refiner needs. Boundary pixels are extremely rare (<<1%) so we
    apply a strong positive weight (5.0).

    Args:
        edge_logits: [B, 1, H, W] raw logits from BoundaryBranch
        gt_mask:     [B, 1, H, W] binary hole mask
    """
    edge_gt = mask_boundary(gt_mask, radius=radius)
    pw = torch.tensor([pos_weight], device=edge_logits.device, dtype=edge_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        edge_logits, edge_gt, pos_weight=pw
    )
    p = torch.sigmoid(edge_logits).flatten(1)
    t = edge_gt.flatten(1)
    inter = (p * t).sum(1)
    dice = 1.0 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)
    return bce + dice_weight * dice.mean()


def band_weighted_bce(
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    radius: int = 3,
    band_weight: float = 4.0,
    pos_weight: float = 3.0,
) -> torch.Tensor:
    """BCE with extra weight on pixels in the boundary band (±radius px).

    This is the boundary-precision regularizer. Pixels far from any boundary
    are weighted 1.0; pixels in the band are weighted `band_weight`.
    """
    band = boundary_band(gt_mask, radius=radius)          # [B,1,H,W] in {0,1}
    weight = 1.0 + (band_weight - 1.0) * band
    pw = torch.tensor([pos_weight], device=pred_logits.device, dtype=pred_logits.dtype)
    raw = F.binary_cross_entropy_with_logits(
        pred_logits, gt_mask.to(pred_logits.dtype), pos_weight=pw, reduction="none"
    )
    return (raw * weight).mean()


def signed_distance_loss(
    pred_prob: torch.Tensor,
    gt_mask: torch.Tensor,
    sdf_max: float = 30.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """A lightweight signed-distance-style loss.

    For each predicted positive (hole) pixel, penalise its distance to the
    nearest GT positive (and vice-versa). We approximate the GT SDF by an
    iterative max-pool dilation — exact, fast, GPU-friendly, no scipy.

    Implements Kervadec-style boundary loss spirit but works directly on
    logits without precomputed distance maps.
    """
    # Approximate SDF outside GT mask
    sdf = _approx_sdf(gt_mask, max_iter=int(sdf_max))     # [B,1,H,W] in [0, sdf_max]
    # Penalty = pred_prob * sdf  (high pred where it's far from GT = punished)
    fp_pen = (pred_prob * sdf).mean()
    # Also penalise false negatives (pred low where GT is high; weighted by inverse-sdf)
    inv_sdf = _approx_sdf(1.0 - gt_mask, max_iter=int(sdf_max))
    fn_pen = ((1.0 - pred_prob) * inv_sdf * gt_mask).mean()
    return fp_pen + fn_pen


def _approx_sdf(mask: torch.Tensor, max_iter: int = 30) -> torch.Tensor:
    """Approximate Euclidean distance transform via iterative dilation.

    For each pixel NOT in `mask`, returns the number of dilation steps
    needed to reach `mask`. Cheap O(max_iter) on GPU.
    """
    m = mask.float()
    dist = torch.zeros_like(m)
    cur = m.clone()
    for i in range(1, max_iter + 1):
        nxt = F.max_pool2d(cur, 3, 1, padding=1)
        new = (nxt > 0).float() - (cur > 0).float()
        dist = dist + new * float(i)
        cur = nxt
        if (nxt.min() > 0).item():
            break
    return dist


# ---------------------------------------------------------------------------
# Combined loss for PRISMPlusBND
# ---------------------------------------------------------------------------

class BNDPlusLoss(nn.Module):
    """Combined loss for PRISMPlusBND.

    L = w_region * BCE_Dice(final, gt)
      + w_coarse * BCE_Dice(coarse, down(gt))
      + w_edge   * BCE_Dice(edge,   boundary(gt))
      + w_band   * band-weighted-BCE(final, gt)
      + w_sdf    * SDF_loss(sigmoid(final), gt)

    Weights default to the values recommended by the GPT-5 architectural
    review. They can be ablated individually.
    """

    def __init__(
        self,
        w_region: float = 1.0,
        w_coarse: float = 0.3,
        w_edge:   float = 0.5,
        w_band:   float = 1.0,
        w_sdf:    float = 0.05,
        pos_weight: float = 3.0,
        edge_pos_weight: float = 5.0,
        dice_weight: float = 0.3,
        band_radius: int = 3,
        edge_radius: int = 1,
    ):
        super().__init__()
        self.w_region = w_region
        self.w_coarse = w_coarse
        self.w_edge = w_edge
        self.w_band = w_band
        self.w_sdf = w_sdf
        self.pos_weight = pos_weight
        self.edge_pos_weight = edge_pos_weight
        self.dice_weight = dice_weight
        self.band_radius = band_radius
        self.edge_radius = edge_radius

    @staticmethod
    def _bce_dice(logits, gt, pos_weight=3.0, dice_w=0.3, eps=1e-6):
        pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, gt.to(logits.dtype), pos_weight=pw
        )
        p = torch.sigmoid(logits).flatten(1)
        t = gt.to(logits.dtype).flatten(1)
        inter = (p * t).sum(1)
        dice = 1.0 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)
        return bce + dice_w * dice.mean()

    def forward(
        self,
        final_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> dict:
        # 1) region loss on final
        l_region = self._bce_dice(
            final_logits, gt_mask, self.pos_weight, self.dice_weight
        )

        # 2) coarse mask supervision (downsample GT to match coarse_logits)
        gt_coarse = F.interpolate(
            gt_mask.float(), size=coarse_logits.shape[-2:], mode="nearest"
        )
        l_coarse = self._bce_dice(
            coarse_logits, gt_coarse, self.pos_weight, self.dice_weight
        )

        # 3) edge supervision
        l_edge = edge_bce_dice(
            edge_logits, gt_mask,
            radius=self.edge_radius,
            pos_weight=self.edge_pos_weight,
            dice_weight=self.dice_weight,
        )

        # 4) band-weighted BCE on final
        l_band = band_weighted_bce(
            final_logits, gt_mask,
            radius=self.band_radius,
            pos_weight=self.pos_weight,
        )

        # 5) SDF loss
        l_sdf = signed_distance_loss(torch.sigmoid(final_logits), gt_mask)

        total = (self.w_region * l_region + self.w_coarse * l_coarse
                 + self.w_edge   * l_edge   + self.w_band   * l_band
                 + self.w_sdf    * l_sdf)

        return {
            "loss":     total,
            "l_region": l_region.detach(),
            "l_coarse": l_coarse.detach(),
            "l_edge":   l_edge.detach(),
            "l_band":   l_band.detach(),
            "l_sdf":    l_sdf.detach(),
        }
