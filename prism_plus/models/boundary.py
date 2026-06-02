"""High-resolution boundary branch + boundary-band refiner.

Design rationale (from refine-logs/BND_ARCHITECTURE_REVIEW.md):
    "VFM should know WHERE failures might happen.
     CNN/edge/refinement should decide WHICH pixels are failures."

This module is responsible for the second half: turning a coarse
semantic mask + raw RGB/depth gradients into a pixel-precise failure
mask, without ever touching VFM patch tokens.
"""
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.boundary import Sobel, boundary_band


# ---------------------------------------------------------------------------
# BoundaryBranch — pure CNN, no VFM
# ---------------------------------------------------------------------------

class BoundaryBranch(nn.Module):
    """Predict per-pixel boundary probability from RGB/depth + early CNN features.

    Inputs:
        rgb:   [B, 3, H,   W]
        depth: [B, 1, H,   W]   sim depth (normalized 0..1)
        f0:    [B, 32, H,  W]   full-res CNN feature  (from BND encoder)
        f1:    [B, 64, H/2,W/2]
        f2:    [B,128, H/4,W/4]

    Output:
        edge_logits: [B, 1, H, W]   raw logits; sigmoid -> P(pixel is on a hole boundary)
    """

    def __init__(self, f0_ch: int = 32, f1_ch: int = 64, f2_ch: int = 128,
                 hidden_ch: int = 64):
        super().__init__()
        self.sobel = Sobel()    # frozen Sobel filter
        in_ch = f0_ch + f1_ch + f2_ch + 1 + 1   # + sobel(rgb_gray) + sobel(depth)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden_ch, 1, 1)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor,
                f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        H, W = rgb.shape[-2:]
        # Sobel on RGB luminance and on depth
        rgb_gray = rgb.mean(dim=1, keepdim=True)
        rgb_grad = self.sobel(rgb_gray)
        # Mask invalid depth before Sobel (zero out)
        valid = (depth > 1e-3).float()
        dep_grad = self.sobel(depth * valid)

        # Upsample low-res features to full resolution
        f1_up = F.interpolate(f1, size=(H, W), mode="bilinear", align_corners=False)
        f2_up = F.interpolate(f2, size=(H, W), mode="bilinear", align_corners=False)
        # Make sure f0 spatial size matches (encoder may produce H/2 by default → upsample)
        if f0.shape[-2:] != (H, W):
            f0 = F.interpolate(f0, size=(H, W), mode="bilinear", align_corners=False)

        x = torch.cat([f0, f1_up, f2_up, rgb_grad, dep_grad], dim=1)
        x = self.fuse(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# BoundaryRefiner — band-limited residual correction of coarse mask
# ---------------------------------------------------------------------------

class BoundaryRefiner(nn.Module):
    """Refine the coarse mask only within a thin band around the predicted boundary.

    final_logits = coarse_up + band * residual
    where:
        coarse_up = bilinear_up(coarse_logits, H, W)
        band      = soft_dilate(sigmoid(edge_logits), radius) detached
        residual  = refine_head([coarse_up, edge_prob, rgb, depth,
                                 sobel(rgb), sobel(depth), f0, f1])

    Detaching the band ensures the refiner doesn't accidentally
    over-amplify residual at every location.
    """

    def __init__(
        self,
        f0_ch: int = 32,
        f1_ch: int = 64,
        hidden_ch: int = 64,
        band_radius: int = 3,
        residual_clip: float = 4.0,
    ):
        super().__init__()
        self.sobel = Sobel()
        self.band_radius = band_radius
        self.residual_clip = residual_clip

        # Inputs: coarse(1) + edge_prob(1) + rgb(3) + depth(1) + sobel_rgb(1) + sobel_dep(1) + f0 + f1_up
        in_ch = 1 + 1 + 3 + 1 + 1 + 1 + f0_ch + f1_ch
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1),
        )
        # Zero-init the last conv so refiner starts with no effect
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    @staticmethod
    def _soft_dilate(p: torch.Tensor, radius: int) -> torch.Tensor:
        """Soft dilation = max-pool on a probability map."""
        k = 2 * radius + 1
        return F.max_pool2d(p, k, stride=1, padding=radius)

    def forward(
        self,
        coarse_logits: torch.Tensor,    # [B,1,Hc,Wc]
        edge_logits:   torch.Tensor,    # [B,1,H, W ]
        rgb:           torch.Tensor,    # [B,3,H, W ]
        depth:         torch.Tensor,    # [B,1,H, W ]
        f0:            torch.Tensor,    # [B,32,H,W]   (may need upsample)
        f1:            torch.Tensor,    # [B,64,H/2,W/2]
    ) -> torch.Tensor:
        H, W = rgb.shape[-2:]
        # 1) upsample coarse mask to full res
        coarse_up = F.interpolate(coarse_logits, size=(H, W),
                                  mode="bilinear", align_corners=False)
        edge_prob = torch.sigmoid(edge_logits)

        # 2) compute residual band (where refinement may act)
        band = self._soft_dilate(edge_prob, self.band_radius).detach()

        # 3) gather inputs for the refine head
        rgb_gray  = rgb.mean(dim=1, keepdim=True)
        rgb_grad  = self.sobel(rgb_gray)
        valid     = (depth > 1e-3).float()
        dep_grad  = self.sobel(depth * valid)
        f1_up     = F.interpolate(f1, size=(H, W), mode="bilinear", align_corners=False)
        if f0.shape[-2:] != (H, W):
            f0 = F.interpolate(f0, size=(H, W), mode="bilinear", align_corners=False)

        x = torch.cat([coarse_up, edge_prob, rgb, depth,
                       rgb_grad, dep_grad, f0, f1_up], dim=1)
        residual = self.refine(x).clamp(-self.residual_clip, self.residual_clip)

        # 4) apply band-gated residual
        return coarse_up + band * residual
