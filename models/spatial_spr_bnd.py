"""
Spatial-SPR BND (C1 for PRISM+)

Extends DualStreamPixelBranchV9 by adding VFM spatial cross-attention
to the encoder skip features at each scale, replacing the GAP-based
global channel attention with pixel-level material-noise alignment.

C1 design (from FINAL_PROPOSAL):
  At each BND scale l:
    Q = D_l  (skip/decoder features at scale l)
    K, V = bilinear_interp(Φ_VFM, scale_l)   ← pixel-level alignment
    enhanced_l = D_l + softmax(Q·K^T/√d)·V   ← residual add

Implementation strategy:
  Subclass DualStreamPixelBranchV9, inject VFM cross-attention into the
  encoder skip features (F1..F4) BEFORE they enter the existing decoder.
  This avoids modifying DecoderBlock and keeps the baseline decoder intact.

New trainable params: ~1.5M (4× VFMCrossAttn modules)
Frozen params: VFM backbone (unchanged)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# ---- path setup ----
_LAT = '/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth'
if _LAT not in sys.path:
    sys.path.insert(0, _LAT)

from latpixdepth.models.dual_stream_pixel_branch_v9 import DualStreamPixelBranchV9
from latpixdepth.models.vfm_interface_v2 import VFMInterface


# ---------------------------------------------------------------------------
# Per-scale VFM spatial cross-attention
# ---------------------------------------------------------------------------

class VFMScaleCrossAttn(nn.Module):
    """
    Cross-attention at one BND scale.

    Q: encoder skip features D_l   [B, C_l, h_l, w_l]
    K, V: VFM tokens (bilinear-aligned to scale l) [B, h_l*w_l, D_vfm]

    Output: spatial residual [B, C_l, h_l, w_l] added to D_l.
    Only adds ~375K params per scale (d_model=256, 8 heads).
    """

    def __init__(self, enc_dim: int, vfm_dim: int = 1024,
                 d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.q_proj  = nn.Linear(enc_dim, d_model)
        self.kv_proj = nn.Linear(vfm_dim, d_model * 2)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out     = nn.Linear(d_model, enc_dim)
        self.norm    = nn.LayerNorm(enc_dim)

    def forward(self, feat: torch.Tensor, vfm_tok: torch.Tensor) -> torch.Tensor:
        """
        feat:    [B, C, h, w]
        vfm_tok: [B, N_vfm, D_vfm]  (already interpolated to (h, w))
        """
        B, C, h, w = feat.shape
        q_flat = feat.flatten(2).transpose(1, 2)          # [B, hw, C]
        q  = self.q_proj(q_flat)                          # [B, hw, d]
        kv = self.kv_proj(vfm_tok)                        # [B, hw, 2d]
        k, v = kv.chunk(2, dim=-1)
        attn_out, _ = self.attn(q, k, v)                  # [B, hw, d]
        res = self.out(attn_out)                           # [B, hw, C]
        res = self.norm(q_flat + res)                      # residual + norm  [B,hw,C]
        return feat + res.transpose(1, 2).reshape(B, C, h, w)


# ---------------------------------------------------------------------------
# Main model: SpatialSPRBND
# ---------------------------------------------------------------------------

class SpatialSPRBND(DualStreamPixelBranchV9):
    """
    PRISM+ BND with Spatial-SPR (C1).

    Inherits DualStreamPixelBranchV9 in full; adds VFM cross-attention
    modules that inject spatial physics priors into encoder skip features
    F1..F4 before the UNet decoder.
    """

    # VFM intermediate layers for multi-scale extraction (DINOv2-Large: 24 blocks)
    VFM_LAYERS = [4, 11, 17, 23]
    # Matches encoder dims: F1=64, F2=128, F3=256, F4=512  (V9 defaults)
    ENC_DIMS   = [64, 128, 256, 512]

    def __init__(
        self,
        vfm_type: str = 'moge2',
        vfm_checkpoint: Optional[str] = None,
        encoder_size: str = 'large',
        use_semantic_guidance: bool = True,
        deep_supervision: bool = True,
        vfm_cross_attn_dim: int = 256,
        **kwargs,
    ):
        # Build parent (V9)
        super().__init__(
            vfm_type=vfm_type,
            vfm_checkpoint=vfm_checkpoint,
            encoder_size=encoder_size,
            use_semantic_guidance=use_semantic_guidance,
            deep_supervision=deep_supervision,
        )

        # Reuse the VFM already loaded by GlobalSemanticContext
        # (avoids loading a second copy of the 300M VFM)
        if self.semantic_context is not None:
            self._vfm: VFMInterface = self.semantic_context.vfm
        else:
            from latpixdepth.models.vfm_interface_v2 import create_vfm
            self._vfm = create_vfm(vfm_type, checkpoint_path=vfm_checkpoint, freeze=True)

        # Detect actual backbone patch size (MoGe2 uses DINOv2 patch=14 internally,
        # but vfm.patch_size returns 16 for forward_semantics; must use real value)
        try:
            backbone = self._vfm.model.encoder.backbone
            self._backbone_ps = int(backbone.patch_embed.proj.kernel_size[0])
        except AttributeError:
            self._backbone_ps = getattr(self._vfm, 'patch_size', 14)
        print(f"  VFM backbone patch_size: {self._backbone_ps}")
        vfm_dim = getattr(self._vfm, 'embed_dim', 1024)

        # One cross-attention module per encoder scale (F1..F4)
        self.vfm_cross_attns = nn.ModuleList([
            VFMScaleCrossAttn(c, vfm_dim=vfm_dim, d_model=vfm_cross_attn_dim)
            for c in self.ENC_DIMS
        ])

        n_new = sum(p.numel() for p in self.vfm_cross_attns.parameters())
        print(f"SpatialSPRBND: added {n_new/1e6:.2f}M cross-attn params")

    # ------------------------------------------------------------------ #
    def _extract_vfm_last(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Extract last VFM intermediate layer.
        Resizes RGB to nearest multiple of backbone patch_size (14 for MoGe2/DINOv2).
        """
        import math
        H, W = rgb.shape[-2], rgb.shape[-1]
        ps = self._backbone_ps                          # true backbone patch size
        H_pad = math.ceil(H / ps) * ps
        W_pad = math.ceil(W / ps) * ps
        if H_pad != H or W_pad != W:
            rgb_in = F.interpolate(rgb, size=(H_pad, W_pad),
                                   mode="bilinear", align_corners=False)
        else:
            rgb_in = rgb
        with torch.no_grad():
            feats = self._vfm.get_intermediate_layers(rgb_in, n=self.VFM_LAYERS)
        tok = feats[-1]                                 # [B, N, D]
        return tok, H_pad // ps, W_pad // ps

    def _vfm_at(self, tok: torch.Tensor, h_vfm: int, w_vfm: int,
                tgt_h: int, tgt_w: int) -> torch.Tensor:
        """Bilinear-interpolate VFM tokens to target spatial resolution."""
        B, N, D = tok.shape
        feat = tok.transpose(1, 2).reshape(B, D, h_vfm, w_vfm)
        feat = F.interpolate(feat, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
        return feat.flatten(2).transpose(1, 2)   # [B, tgt_h*tgt_w, D]

    # ------------------------------------------------------------------ #
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Same interface as DualStreamPixelBranchV9.forward.

        C1 injection: after encoder, before decoder,
        each skip feature F_l is enhanced with VFM cross-attention.
        """
        B, _, H, W = rgb.shape

        # 1. CNN encoder (unchanged)
        encoder_out = self.encoder(rgb, depth)
        enc_feats = list(encoder_out['features'])   # [F0, F1, F2, F3, F4]
        f0 = enc_feats[0]

        # 2. GAP channel attention (unchanged — keeps the PRISM baseline behaviour)
        if self.semantic_context is not None:
            channel_weights = self.semantic_context(rgb)
        else:
            channel_weights = None

        # 3. C1: VFM spatial cross-attention on F1..F4
        vfm_tok, h_vfm, w_vfm = self._extract_vfm_last(rgb)
        for i, ca in enumerate(self.vfm_cross_attns):
            fi = enc_feats[i + 1]                 # F1..F4  (i=0→F1, …, i=3→F4)
            tok_l = self._vfm_at(vfm_tok, h_vfm, w_vfm, fi.shape[-2], fi.shape[-1])
            enc_feats[i + 1] = ca(fi, tok_l)      # residual-enhanced skip

        # 4. UNet decoder (unchanged, receives enhanced skips)
        out, pyramid = self.decoder(enc_feats, channel_weights)

        # 5. Prediction heads (unchanged)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        main_logits   = self.failure_head(out)
        main_prob     = torch.sigmoid(main_logits)
        detail_input  = torch.cat([
            out, f0, encoder_out['rgb0'], encoder_out['geo0'],
            encoder_out['geometry_maps'], main_prob,
        ], dim=1)
        detail_logits = self.detail_head(detail_input)
        final_logits  = main_logits + self.detail_scale * detail_logits
        pred_failure  = torch.sigmoid(final_logits)

        result = {
            'failure_logits': final_logits,
            'main_logits':    main_logits,
            'detail_logits':  detail_logits,
            'pred_failure':   pred_failure,
            'pred_main':      main_prob,
            'pred_detail':    torch.sigmoid(detail_logits),
        }

        if self.deep_supervision and self.training:
            aux_logits = self.aux_failure_heads(pyramid, (H, W))
            result['aux_failure_logits'] = aux_logits
            result['aux_failure_maps']   = [torch.sigmoid(p) for p in aux_logits]

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_spatial_spr_bnd(
    vfm_type: str = 'moge2',
    vfm_checkpoint: Optional[str] = None,
    encoder_size: str = 'large',
    deep_supervision: bool = True,
    **kwargs,
) -> SpatialSPRBND:
    return SpatialSPRBND(
        vfm_type=vfm_type,
        vfm_checkpoint=vfm_checkpoint,
        encoder_size=encoder_size,
        deep_supervision=deep_supervision,
        **kwargs,
    )
