"""PRISM+ C4 — Optical flow utilities (torchvision RAFT + flow warp).

Uses torchvision's bundled RAFT large model. The 'C_T_V2' weights are
trained on FlyingChairs + FlyingThings3D, equivalent to the canonical
'raft-things.pth' from princeton-vl/RAFT.

No external checkpoint download needed — torchvision auto-fetches to
$TORCH_HOME/checkpoints/ on first use (~50 MB).

Provides:
    1. RAFTFlow         — frozen optical flow estimator
    2. backward_warp    — differentiable bilinear warp by flow
    3. downsample_flow  — spatial rescale + magnitude rescale
    4. make_zero_flow   — identity fallback for textureless / single-frame
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def backward_warp(features: torch.Tensor, flow: torch.Tensor,
                   mode: str = 'bilinear', padding_mode: str = 'zeros') -> torch.Tensor:
    """Warp  (frame t-1) to frame t coordinate frame.

    For each output pixel (x, y), samples src at (x + flow_x, y + flow_y).

    Args:
        features : [B, C, H, W]
        flow     : [B, 2, H, W]   in pixels, flow channel 0 = du, 1 = dv

    Returns:
        warped   : [B, C, H, W]
    """
    B, C, H, W = features.shape
    device = features.device

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=features.dtype),
        torch.arange(W, device=device, dtype=features.dtype),
        indexing='ij',
    )
    grid_x = xx.unsqueeze(0).expand(B, -1, -1)
    grid_y = yy.unsqueeze(0).expand(B, -1, -1)

    src_x = grid_x + flow[:, 0]
    src_y = grid_y + flow[:, 1]

    norm_x = 2.0 * src_x / max(W - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(H - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)

    return F.grid_sample(features, grid, mode=mode,
                         padding_mode=padding_mode, align_corners=True)


def downsample_flow(flow: torch.Tensor, target_size) -> torch.Tensor:
    """Spatially resize a flow field and rescale magnitudes.

    Flow lives in pixel units, so shrinking H,W by k must shrink magnitudes by k.
    """
    H_t, W_t = target_size
    H_s, W_s = flow.shape[-2:]
    scale_h = H_t / H_s
    scale_w = W_t / W_s
    flow_ds = F.interpolate(flow, size=(H_t, W_t), mode='bilinear', align_corners=True)
    flow_ds = flow_ds.clone()
    flow_ds[:, 0] *= scale_w
    flow_ds[:, 1] *= scale_h
    return flow_ds


def make_zero_flow(reference: torch.Tensor) -> torch.Tensor:
    B, _, H, W = reference.shape
    return torch.zeros(B, 2, H, W, device=reference.device, dtype=reference.dtype)


class RAFTFlow(nn.Module):
    """Frozen RAFT optical flow estimator via torchvision.

    Args:
        weights_name : one of 'C_T_V2' (= raft-things, default),
                       'C_T_SKHT_V2' (+ Sintel + KITTI, stronger),
                       'C_T_V1' (legacy).
        iters        : RAFT GRU update iterations (default 12 ~ paper)

    Inputs: image1, image2 of shape [B, 3, H, W] in [0, 1] (auto-rescaled to
    [-1, 1] internally to match RAFT's preprocessing). H, W should be a
    multiple of 8 (RAFT internally pads otherwise).
    """

    def __init__(self, weights_name: str = 'C_T_V2', iters: int = 12,
                  freeze: bool = True):
        super().__init__()
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights = getattr(Raft_Large_Weights, weights_name)
        self.model = raft_large(weights=weights, progress=False)
        self.iters = int(iters)
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Estimate flow image1 -> image2.

        Args:
            image1, image2 : [B, 3, H, W] in [0, 1]
        Returns:
            flow_12 : [B, 2, H, W] in pixels
        """
        # torchvision RAFT expects images in [-1, 1]
        x1 = image1 * 2.0 - 1.0
        x2 = image2 * 2.0 - 1.0
        # Returns list of flow maps from coarse to fine; last is final estimate
        flows = self.model(x1, x2, num_flow_updates=self.iters)
        return flows[-1]


# Backward-compat alias (older code referenced RAFTWrapper)
RAFTWrapper = RAFTFlow
