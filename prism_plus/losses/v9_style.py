"""PRISM+ v0.6 — V9-style loss adapted for PRISMPlusBND (C1 architecture).

Built on the recipe that gave the latpixdepth v9 pixel branch its visually
sharp masks. Key components retained from v9:

    * BCE with pos_weight (default 3.0) — pushes recall up
    * OHEM (default ratio 0.3) — focuses gradient on hardest pixels;
        NOTE: v9 used 0.1 which is too aggressive (high variance, can
        collapse to a handful of extreme pixels). 0.3 is the Faster-R-CNN
        sweet spot — same hard-mining benefit, more stable signal.
    * Small-region weighting via 5×5 erosion — fine-grained hole detail
    * Dice loss — region overlap bonus on top of BCE
    * LPIPS (VGG) on the final mask — perceptual sharpness

Adapted for the PRISMPlusBND output dict {failure_logits, coarse_logits,
edge_logits, pred_failure} instead of the v9 dual-stream branches.

The supervision target is **raw hole_mask** (not the de-noised M_target):
we want the model to learn every detail in the data, and let the v0.6
evaluation lift the noise from the metric side instead (eval reports both
raw and M_target-based IoU).
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class V9StyleLoss(nn.Module):
    """V9-recipe loss for PRISMPlusBND. raw-GT supervision."""

    def __init__(
        self,
        pos_weight: float = 3.0,
        use_ohem: bool = True,
        ohem_ratio: float = 0.3,
        use_small_region_weighting: bool = True,
        small_region_weight: float = 2.0,
        small_region_kernel: int = 5,
        dice_weight: float = 0.3,
        use_lpips: bool = True,
        lpips_weight: float = 1.0,
        # head weighting
        w_final: float = 1.0,
        w_coarse: float = 0.3,
        w_edge:   float = 0.0,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,        # 0 disables focal
    ):
        super().__init__()
        self.pos_weight   = float(pos_weight)
        self.use_ohem     = bool(use_ohem)
        self.ohem_ratio   = float(ohem_ratio)
        self.use_small_region_weighting = bool(use_small_region_weighting)
        self.small_region_weight = float(small_region_weight)
        self.small_region_kernel = int(small_region_kernel)
        self.dice_weight  = float(dice_weight)
        self.use_lpips    = bool(use_lpips)
        self.lpips_weight = float(lpips_weight)
        self.w_final      = float(w_final)
        self.w_coarse     = float(w_coarse)
        self.w_edge       = float(w_edge)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma  = float(focal_gamma)

        if self.use_lpips:
            import lpips
            self.lpips_net = lpips.LPIPS(net='vgg', verbose=False)
            for p in self.lpips_net.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # primitives
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _small_region_weight(self, target: torch.Tensor) -> torch.Tensor:
        """Erode the positive mask (kxk); the difference highlights thin
        small regions. Weight = 1 + (sr_w - 1) * small_region_mask."""
        k = self.small_region_kernel
        pad = k // 2
        eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size=k, stride=1, padding=pad)
        sr = (target - eroded).clamp(0.0, 1.0)
        return 1.0 + (self.small_region_weight - 1.0) * sr

    def _label_smooth(self, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            return target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return target

    def _ohem_reduce(self, loss_map: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_flat = loss_map.reshape(-1)
        target_flat = target.reshape(-1)
        pos = target_flat > 0.5
        pos_loss = loss_flat[pos]
        neg_loss = loss_flat[~pos]
        if neg_loss.numel() > 0:
            k = max(1, int(neg_loss.numel() * self.ohem_ratio))
            neg_loss = torch.topk(neg_loss, k=k).values
        if pos_loss.numel() > 0 and neg_loss.numel() > 0:
            return torch.cat([pos_loss, neg_loss]).mean()
        if pos_loss.numel() > 0: return pos_loss.mean()
        if neg_loss.numel() > 0: return neg_loss.mean()
        return loss_flat.mean()

    def _bce(self, pred_prob: torch.Tensor, target: torch.Tensor,
              pixel_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        eps = 1e-6
        p = pred_prob.clamp(eps, 1.0 - eps)
        t = self._label_smooth(target)
        bce_pos = -t * p.log()
        bce_neg = -(1.0 - t) * (1.0 - p).log()
        loss_map = bce_pos * self.pos_weight + bce_neg

        if self.focal_gamma > 0:
            pt = torch.where(target >= 0.5, p, 1.0 - p)
            loss_map = loss_map * (1.0 - pt).pow(self.focal_gamma)

        if pixel_weight is not None:
            loss_map = loss_map * pixel_weight
        if self.use_ohem:
            loss = self._ohem_reduce(loss_map, target)
        else:
            loss = loss_map.mean()
        if not torch.isfinite(loss):
            return loss_map.new_tensor(0.0, requires_grad=True)
        return loss

    @staticmethod
    def _dice(pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred_prob.reshape(pred_prob.shape[0], -1)
        t = target.reshape(target.shape[0], -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

    def _lpips(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.use_lpips:
            return pred_prob.new_tensor(0.0, requires_grad=True)
        if self.lpips_net.parameters().__next__().device != pred_prob.device:
            self.lpips_net = self.lpips_net.to(pred_prob.device)
        # LPIPS expects [-1, 1] RGB images
        pred_rgb   = pred_prob.repeat(1, 3, 1, 1) * 2.0 - 1.0
        target_rgb = target.repeat(1, 3, 1, 1)    * 2.0 - 1.0
        l = self.lpips_net(pred_rgb, target_rgb).mean()
        if not torch.isfinite(l):
            return pred_prob.new_tensor(0.0, requires_grad=True)
        return l

    # ------------------------------------------------------------------
    # head loss assembly
    # ------------------------------------------------------------------

    def _head_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pixel_weight: Optional[torch.Tensor],
        use_lpips: bool,
    ) -> Dict[str, torch.Tensor]:
        prob = torch.sigmoid(logits.float())
        bce_l   = self._bce(prob, target, pixel_weight)
        dice_l  = self._dice(prob, target)
        if use_lpips:
            lpips_l = self._lpips(prob, target)
        else:
            lpips_l = prob.new_tensor(0.0)
        total = bce_l + self.dice_weight * dice_l + self.lpips_weight * lpips_l
        return {
            'total': total,
            'bce':   bce_l.detach(),
            'dice':  (self.dice_weight * dice_l).detach(),
            'lpips': (self.lpips_weight * lpips_l).detach(),
        }

    # ------------------------------------------------------------------
    # forward — accepts PRISMPlusBND output dict
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,           # raw hole_mask [B,1,H,W]
    ) -> Dict[str, torch.Tensor]:
        """outputs dict from PRISMPlusBND.forward(rgb, sim_depth).
        Required keys:
            'failure_logits' : main BND output
            'coarse_logits'  : coarse semantic branch
            'edge_logits'    : optional boundary branch (may be empty)
        """
        pixel_w = self._small_region_weight(target) if self.use_small_region_weighting else None

        # main (final) — gets LPIPS
        l_final = self._head_loss(outputs['failure_logits'], target,
                                  pixel_weight=pixel_w, use_lpips=self.use_lpips)
        out: Dict[str, torch.Tensor] = {
            'l_final_total': l_final['total'].detach(),
            'l_final_bce':   l_final['bce'],
            'l_final_dice':  l_final['dice'],
            'l_final_lpips': l_final['lpips'],
        }
        loss_total = self.w_final * l_final['total']

        # coarse aux — no LPIPS, no small-region weight (deep supervision)
        if self.w_coarse > 0 and 'coarse_logits' in outputs and outputs['coarse_logits'] is not None:
            l_coarse = self._head_loss(outputs['coarse_logits'], target,
                                       pixel_weight=None, use_lpips=False)
            loss_total = loss_total + self.w_coarse * l_coarse['total']
            out['l_coarse_total'] = (self.w_coarse * l_coarse['total']).detach()

        # edge aux — typically OFF (w_edge=0), enable for boundary-rich supervision
        if self.w_edge > 0 and 'edge_logits' in outputs and outputs['edge_logits'] is not None:
            l_edge = self._head_loss(outputs['edge_logits'], target,
                                     pixel_weight=pixel_w, use_lpips=False)
            loss_total = loss_total + self.w_edge * l_edge['total']
            out['l_edge_total'] = (self.w_edge * l_edge['total']).detach()

        if not torch.isfinite(loss_total):
            loss_total = l_final['total'].clamp(0.0, 10.0)

        out['loss'] = loss_total
        return out
