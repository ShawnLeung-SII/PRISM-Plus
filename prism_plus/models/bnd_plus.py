"""PRISM+ BND v0.2.0 — Coarse Semantic Prior + Pixel Boundary Refinement.

Design rationale (per refine-logs/BND_ARCHITECTURE_REVIEW.md):

    VFM provides material/scene SEMANTIC PRIOR ("where failures might occur").
    CNN + edge branch + refiner decide PIXEL-PRECISE boundaries.

Architecture::

    RGB+depth → CNN encoder → f0, f1, f2, f3, f4
                                          ↑     ↑
                          GatedVFMCrossAttn (α=0 init) on F3, F4 only
                          using ONLY DINOv2 layer 23 (deepest)
                                          ↓
                              CoarseDecoder → coarse_logits (low-res, semantic)

                          BoundaryBranch (pure CNN + Sobel)
                              ↓
                              edge_logits (high-res, boundary-aware)

                          BoundaryRefiner
                              ↓
                              final_logits = coarse_up + band * residual

Crucial differences from v0.1.0 (SpatialBND):

    1. VFM cross-attn ONLY on F3, F4 (coarse) — never on F0, F1, F2.
    2. cross-attn output gated by `α.tanh()`, α zero-init -> starts as no-op.
    3. NEW boundary branch (CNN + Sobel(RGB) + Sobel(depth)) for pixel-precise edges.
    4. Refiner adds residual ONLY in the predicted boundary band (±3px).
    5. NO double modulation: coarse decoder owns channel attention; refiner owns
       boundary; they never overlap.

This guarantees:
    * Training starts from the BND baseline behaviour (α=0, refiner=0)
    * VFM contribution is monotonic improvement (ablatable)
    * Patch-level VFM tokens are NEVER bilinear-upsampled to full resolution.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bnd import (
    BND,
    PixelLevelEncoder,
    GlobalSemanticContext,
    UNetDecoder,
    PixelPredictionLogitHead,
)
from .boundary import BoundaryBranch, BoundaryRefiner
from .vfm import VFMInterface, create_vfm


# ---------------------------------------------------------------------------
# Gated VFM cross-attention (zero-init alpha, only used on coarse scales)
# ---------------------------------------------------------------------------

class GatedVFMCrossAttn(nn.Module):
    """Coarse-stage VFM cross-attention with zero-init learnable gate.

    Q  : CNN features at coarse scale  [B, C_enc, H_s, W_s]
    KV : VFM patch tokens at one layer [B, N_vfm, D_vfm]

    Output = feat + α.tanh() * GN(proj(attn))      α init = 0

    At init, output = feat exactly (matches baseline BND).
    During training, α grows as long as the gradient signal supports it.
    """

    def __init__(
        self,
        enc_dim: int,
        vfm_dim: int = 1024,
        d_model: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        self.q = nn.Conv2d(enc_dim, d_model, 1)
        self.k = nn.Linear(vfm_dim, d_model)
        self.v = nn.Linear(vfm_dim, d_model)
        self.proj = nn.Conv2d(d_model, enc_dim, 1)
        self.norm = nn.GroupNorm(8, enc_dim)

        # zero-init: training starts indistinguishable from baseline BND
        self.alpha = nn.Parameter(torch.zeros(1))
        # Also zero-init proj output so very first iteration is exact baseline
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, vfm_tok: torch.Tensor) -> torch.Tensor:
        """feat: [B,C,H,W]  vfm_tok: [B,N,D_vfm]"""
        B, C, H, W = feat.shape

        # Q from CNN features
        q = self.q(feat).flatten(2).transpose(1, 2)   # [B, HW, d_model]
        k = self.k(vfm_tok)                            # [B, N,  d_model]
        v = self.v(vfm_tok)                            # [B, N,  d_model]

        # Multi-head reshape
        def _mh(t):
            return t.reshape(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        q_h, k_h, v_h = _mh(q), _mh(k), _mh(v)

        # FlashAttention path (O(N) memory)
        attn = F.scaled_dot_product_attention(q_h, k_h, v_h)         # [B, h, HW, d_h]
        attn = attn.transpose(1, 2).reshape(B, H * W, self.d_model)
        delta = attn.transpose(1, 2).reshape(B, self.d_model, H, W)
        delta = self.proj(delta)                                      # [B, C, H, W]
        delta = self.norm(delta)

        return feat + self.alpha.tanh() * delta


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PRISMPlusBND(BND):
    """PRISM+ BND v0.2.0.

    Inherits the baseline BND encoder / decoder / heads from PRISM, then:
      * adds GatedVFMCrossAttn on F3, F4 (deep VFM layer 23 only)
      * adds BoundaryBranch (pure CNN + Sobel)
      * adds BoundaryRefiner (band-limited residual)
    """

    # We only ever use the deepest layer (the only layer that's semantic enough
    # to safely upsample to coarse decoder stages without injecting patch-level
    # texture artefacts).
    VFM_DEEP_LAYER = 23

    def __init__(
        self,
        vfm_type: str = "moge2",
        vfm_checkpoint: Optional[str] = None,
        encoder_size: str = "large",
        use_semantic_guidance: bool = True,
        deep_supervision: bool = True,
        vfm_cross_attn_dim: int = 128,
        boundary_band_radius: int = 3,
    ):
        super().__init__(
            vfm_type=vfm_type,
            vfm_checkpoint=vfm_checkpoint,
            encoder_size=encoder_size,
            use_semantic_guidance=use_semantic_guidance,
            deep_supervision=deep_supervision,
        )

        # Reuse the VFM loaded by GlobalSemanticContext (frozen backbone).
        if self.semantic_context is not None:
            self._vfm: VFMInterface = self.semantic_context.vfm
        else:
            self._vfm = create_vfm(vfm_type, checkpoint_path=vfm_checkpoint, freeze=True)

        # Detect real backbone patch size (MoGe2/DINOv2 = 14)
        try:
            self._backbone_ps = int(
                self._vfm.model.encoder.backbone.patch_embed.proj.kernel_size[0]
            )
        except AttributeError:
            self._backbone_ps = getattr(self._vfm, "patch_size", 14)

        vfm_dim = getattr(self._vfm, "embed_dim", 1024)

        # Gated cross-attn ONLY for F3 (256ch, H/8) and F4 (512ch, H/16)
        self.gate_f3 = GatedVFMCrossAttn(256, vfm_dim=vfm_dim, d_model=vfm_cross_attn_dim)
        self.gate_f4 = GatedVFMCrossAttn(512, vfm_dim=vfm_dim, d_model=vfm_cross_attn_dim)

        # Boundary branch + refiner (PRISM+ contribution that owns pixel-level boundary)
        self.boundary_branch  = BoundaryBranch(
            f0_ch=PixelLevelEncoder.F0_CHANNELS,  # 32
            f1_ch=64, f2_ch=128, hidden_ch=64,
        )
        self.boundary_refiner = BoundaryRefiner(
            f0_ch=PixelLevelEncoder.F0_CHANNELS,
            f1_ch=64, hidden_ch=64,
            band_radius=boundary_band_radius,
        )

        # PRISMPlusBND does NOT use the parent's detail_head / detail_scale
        # (we replace the detail path with BoundaryBranch + BoundaryRefiner).
        # Remove them so DDP does not see "unused parameters".
        for attr in ("detail_head", "detail_scale"):
            if hasattr(self, attr):
                delattr(self, attr)

        n_new = sum(p.numel() for p in [
            *self.gate_f3.parameters(), *self.gate_f4.parameters(),
            *self.boundary_branch.parameters(), *self.boundary_refiner.parameters(),
        ])
        print(f"PRISMPlusBND: +{n_new/1e6:.2f}M new params (gates + boundary)")

    # ------------------------------------------------------------------ #
    def _extract_vfm_deep(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Extract ONLY the deepest VFM layer, with size guard for patch divisibility."""
        H, W = rgb.shape[-2], rgb.shape[-1]
        ps   = self._backbone_ps
        H_pad = math.ceil(H / ps) * ps
        W_pad = math.ceil(W / ps) * ps
        if (H_pad, W_pad) != (H, W):
            rgb_in = F.interpolate(rgb, size=(H_pad, W_pad),
                                   mode="bilinear", align_corners=False)
        else:
            rgb_in = rgb
        with torch.no_grad():
            feats = self._vfm.get_intermediate_layers(rgb_in, n=[self.VFM_DEEP_LAYER])
        tok = feats[-1]                       # [B, N, D]
        return tok, H_pad // ps, W_pad // ps

    @staticmethod
    def _vfm_at(tok: torch.Tensor, h_vfm: int, w_vfm: int,
                tgt_h: int, tgt_w: int) -> torch.Tensor:
        """Bilinear-interpolate VFM tokens to (tgt_h, tgt_w)."""
        B, N, D = tok.shape
        feat = tok.transpose(1, 2).reshape(B, D, h_vfm, w_vfm)
        feat = F.interpolate(feat, size=(tgt_h, tgt_w),
                             mode="bilinear", align_corners=False)
        return feat.flatten(2).transpose(1, 2)   # [B, tgt_h*tgt_w, D]

    # ------------------------------------------------------------------ #
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns:
            failure_logits  : [B,1,H,W]  final refined prediction
            coarse_logits   : [B,1,Hc,Wc] from baseline decoder + heads
            edge_logits     : [B,1,H,W]   raw edge probability
            pred_failure    : sigmoid(failure_logits)
            pred_coarse     : sigmoid(coarse_logits)
            pred_edge       : sigmoid(edge_logits)
        """
        B, _, H, W = rgb.shape

        # 1) Encoder (CNN) — unchanged
        encoder_out  = self.encoder(rgb, depth)
        enc_feats    = list(encoder_out["features"])      # [F0, F1, F2, F3, F4]
        f0, f1, f2, f3, f4 = enc_feats

        # 2) GAP channel attention — only on F1/F2 now (NOT on F3/F4)
        #    (F3/F4 get richer SPATIAL gating from gate_f3 / gate_f4 below)
        if self.semantic_context is not None:
            channel_weights = self.semantic_context(rgb)   # list of 4 [B,C,1,1]
            # Disable GAP gating on F3/F4 (zero out those weights)
            # so they don't fight with the spatial cross-attn we add.
            channel_weights = [
                channel_weights[0],          # w1 for F1 — keep
                channel_weights[1],          # w2 for F2 — keep
                torch.zeros_like(channel_weights[2]),     # w3 for F3 — disable
                torch.zeros_like(channel_weights[3]),     # w4 for F4 — disable
            ]
        else:
            channel_weights = None

        # 3) VFM cross-attn ONLY on coarse stages (F3, F4) using DEEPEST layer
        vfm_tok, h_vfm, w_vfm = self._extract_vfm_deep(rgb)
        tok_f3 = self._vfm_at(vfm_tok, h_vfm, w_vfm, f3.shape[-2], f3.shape[-1])
        tok_f4 = self._vfm_at(vfm_tok, h_vfm, w_vfm, f4.shape[-2], f4.shape[-1])
        f3 = self.gate_f3(f3, tok_f3)
        f4 = self.gate_f4(f4, tok_f4)
        enc_feats = [f0, f1, f2, f3, f4]

        # 4) Baseline-style coarse decoder + main head
        dec_out, pyramid = self.decoder(enc_feats, channel_weights)
        if dec_out.shape[2:] != (H, W):
            dec_out = F.interpolate(dec_out, size=(H, W),
                                    mode="bilinear", align_corners=False)
        coarse_logits = self.failure_head(dec_out)         # [B,1,H,W]

        # 5) High-res boundary branch (pure CNN, no VFM)
        edge_logits = self.boundary_branch(rgb, depth, f0, f1, f2)

        # 6) Band-limited residual refinement
        final_logits = self.boundary_refiner(
            coarse_logits=coarse_logits,
            edge_logits=edge_logits,
            rgb=rgb, depth=depth,
            f0=f0, f1=f1,
        )

        result = {
            "failure_logits": final_logits,
            "coarse_logits":  coarse_logits,
            "edge_logits":    edge_logits,
            "pred_failure":   torch.sigmoid(final_logits),
            "pred_coarse":    torch.sigmoid(coarse_logits),
            "pred_edge":      torch.sigmoid(edge_logits),
        }
        if self.deep_supervision and self.training:
            aux_logits = self.aux_failure_heads(pyramid, (H, W))
            result["aux_failure_logits"] = aux_logits

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_bnd_plus(
    vfm_type: str = "moge2",
    vfm_checkpoint: Optional[str] = None,
    encoder_size: str = "large",
    deep_supervision: bool = True,
    **kwargs,
) -> PRISMPlusBND:
    return PRISMPlusBND(
        vfm_type=vfm_type,
        vfm_checkpoint=vfm_checkpoint,
        encoder_size=encoder_size,
        deep_supervision=deep_supervision,
        **kwargs,
    )
