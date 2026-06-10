"""DensityLoss — v0.4.0 minimal weighted BCE for density estimation.

Theoretical foundation:
    The Bayes-optimal solution of binary cross-entropy is the posterior
    probability:  σ(f*(x)) = P(y=1 | x).

    So a sigmoid + BCE network, trained on clean hard binary labels, will
    naturally output a calibrated density estimate. No special loss is needed.

The only departure from vanilla BCE is the per-pixel `weight` (= 1 - M_ignore),
which excludes isolated noise pixels that the model CANNOT learn (their
positions are random per RealSense capture).

What this module does NOT include (all removed from v0.3.x — see commit log):
    * asymmetric BCE (FP vs FN weighting)        — biases predictions, hurts recall
    * Tversky loss with α≠β                       — same problem
    * sharpness loss (p × (1-p))                  — kills natural calibration
    * boundary-band weighted BCE                  — over-constrains predictions
    * SDF loss / edge BCE                         — replaced by data-driven calibration

Why "less is more": BCE already converges to P(y|x). Any extra "smart" term
adds a bias that the optimal solution must overcome — that's exactly what
crippled v0.3.1 (precision=0.93 but recall and visual granularity collapsed).
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityLoss(nn.Module):
    """v0.4.0 BND loss.

    L = weighted_BCE(final_logits, M_target, weight=1-M_ignore)
        + w_coarse · weighted_BCE(coarse_logits, M_target_down)
        + w_edge   · BCE(edge_logits, edge_GT)      # weak edge supervision

    All terms are unbiased BCE — no pos_weight, no asymmetry, nothing fancy.
    The network learns the true conditional probability P(hole | x), which
    can later be thresholded at any value to produce a binary mask.
    """

    def __init__(
        self,
        w_main:   float = 1.0,
        w_coarse: float = 0.3,
        w_edge:   float = 0.0,    # 默认关闭, 试一下纯净 BCE 的效果; 必要时再开
        edge_pos_weight: float = 5.0,
    ):
        super().__init__()
        self.w_main = w_main
        self.w_coarse = w_coarse
        self.w_edge = w_edge
        self.edge_pos_weight = edge_pos_weight

    @staticmethod
    def _weighted_bce(logits: torch.Tensor,
                      target: torch.Tensor,
                      weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard BCE with per-pixel weight, AMP-safe via logsigmoid path."""
        return F.binary_cross_entropy_with_logits(
            logits, target.to(logits.dtype),
            weight=weight, reduction="mean",
        )

    def forward(
        self,
        final_logits:  torch.Tensor,   # [B,1,H,W]
        coarse_logits: torch.Tensor,   # [B,1,Hc,Wc]
        edge_logits:   torch.Tensor,   # [B,1,H,W]
        M_target:      torch.Tensor,   # [B,1,H,W]  hard binary
        M_ignore:      torch.Tensor,   # [B,1,H,W]  hard binary, 1 = exclude
    ) -> Dict[str, torch.Tensor]:
        # 1) Main loss: weighted BCE on structured targets only
        pixel_w = (1.0 - M_ignore.to(final_logits.dtype))
        l_main = self._weighted_bce(final_logits, M_target, pixel_w)

        # 2) Coarse supervision — downsample both target and ignore mask
        tgt_c = F.interpolate(M_target.to(coarse_logits.dtype),
                              size=coarse_logits.shape[-2:], mode="nearest")
        ign_c = F.interpolate(M_ignore.to(coarse_logits.dtype),
                              size=coarse_logits.shape[-2:], mode="nearest")
        pixel_wc = (1.0 - ign_c)
        l_coarse = self._weighted_bce(coarse_logits, tgt_c, pixel_wc)

        # 3) Edge supervision (optional, default off)
        if self.w_edge > 0:
            edge_gt = _gt_edge_band(M_target, radius=1)
            pw = torch.tensor([self.edge_pos_weight],
                              device=edge_logits.device, dtype=edge_logits.dtype)
            l_edge = F.binary_cross_entropy_with_logits(
                edge_logits, edge_gt, pos_weight=pw, reduction="mean",
            )
        else:
            l_edge = final_logits.new_zeros(())

        total = self.w_main * l_main + self.w_coarse * l_coarse + self.w_edge * l_edge

        return {
            "loss":     total,
            "l_main":   l_main.detach(),
            "l_coarse": l_coarse.detach(),
            "l_edge":   l_edge.detach(),
        }


# ---------------------------------------------------------------------------
# Helper used only when w_edge > 0
# ---------------------------------------------------------------------------

def _gt_edge_band(m: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """1-pixel boundary of M as edge GT."""
    pad = radius
    k = 2 * radius + 1
    m = m.float()
    dil =  F.max_pool2d( m, k, 1, padding=pad)
    ero = -F.max_pool2d(-m, k, 1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)
