"""PRISM+ C4 — Temporal Noise State Module (TNSM).

A flow-guided ConvGRU at H/4 latent resolution that maintains a temporally
consistent noise state h_t across frames of a video, then injects it back
into BND (decoder feature) and NRG (time embedding).

Pipeline per training step (4-frame BPTT):
    1. RAFT (frozen) computes flow F_{t -> t-1} from RGB sequence
    2. h_tilde_{t-1} = backward_warp(h_{t-1}, F_{t -> t-1})   (registered to frame t)
    3. h_t = ConvGRU(enc_feat_t, h_tilde_{t-1})              (update)
    4. h_t fed back to BND decoder as additive feature
    5. GAP(h_t) added to NRG time embedding (optional)

Param budget per FINAL_PROPOSAL: ~2M.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_utils import backward_warp, downsample_flow


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell. Operates at single resolution.

    h_new = (1 - z) * h_prev + z * h_cand
        z      = sigmoid(W_z [x, h_prev])
        r      = sigmoid(W_r [x, h_prev])
        h_cand = tanh   (W_h [x, r * h_prev])
    """

    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_z = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size, padding=pad)
        self.conv_r = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size, padding=pad)
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size, padding=pad)
        self.hid_ch = hid_ch

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))
        xh_r = torch.cat([x, r * h_prev], dim=1)
        h_cand = torch.tanh(self.conv_h(xh_r))
        return (1.0 - z) * h_prev + z * h_cand

    def init_hidden(self, batch: int, H: int, W: int,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.hid_ch, H, W, device=device, dtype=dtype)


class TNSM(nn.Module):
    """Temporal Noise State Module.

    Args:
        enc_in_channels : channel dim of the BND encoder feature at H/4
                          that we receive as input each frame.
        state_channels  : hidden channels of the ConvGRU (default 128).
        spatial_div     : H_state = H_image / spatial_div (default 4 -> H/4).
        use_flow_warp   : if True, warps h_{t-1} by RAFT flow before update.

    The output 'h_t' is fed back into the BND decoder (concat or add) and
    its GAP can be sent to NRG t_emb.
    """

    def __init__(
        self,
        enc_in_channels: int,
        state_channels: int = 128,
        spatial_div: int = 4,
        use_flow_warp: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.enc_in_channels = int(enc_in_channels)
        self.state_channels  = int(state_channels)
        self.spatial_div     = int(spatial_div)
        self.use_flow_warp   = bool(use_flow_warp)

        # Project encoder feature into the ConvGRU input dim
        self.proj_enc = nn.Sequential(
            nn.Conv2d(self.enc_in_channels, self.state_channels, 1),
            nn.GroupNorm(min(8, max(1, self.state_channels)), self.state_channels),
            nn.SiLU(),
        )
        self.gru = ConvGRUCell(in_ch=self.state_channels,
                                hid_ch=self.state_channels,
                                kernel_size=kernel_size)
        # Output projection (to feed back to BND decoder; same channels as input)
        self.proj_out = nn.Sequential(
            nn.Conv2d(self.state_channels, self.enc_in_channels, 1),
            nn.GroupNorm(min(8, max(1, self.enc_in_channels)), self.enc_in_channels),
        )
        # Time-embedding projection for NRG (320 = SD default)
        self.proj_temb = nn.Linear(self.state_channels, 320)
        nn.init.zeros_(self.proj_temb.weight)
        nn.init.zeros_(self.proj_temb.bias)

    @torch.no_grad()
    def init_state(
        self,
        batch: int,
        spatial_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        H, W = spatial_hw
        return self.gru.init_hidden(batch, H, W, device, dtype)

    def step(
        self,
        enc_feat_t: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        flow_t_to_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward step.

        Args:
            enc_feat_t      : [B, C_enc, H_s, W_s]  encoder feature at state scale
            h_prev          : [B, C_state, H_s, W_s] previous state, or None for t=0
            flow_t_to_prev  : [B, 2, H_img, W_img] optical flow image-space, or None
                              (will be downsampled internally to state scale)

        Returns:
            h_t           : [B, C_state, H_s, W_s]  updated state
            feat_out      : [B, C_enc, H_s, W_s]    feature to feed back to BND
            t_emb_delta   : [B, 320]                additive delta for NRG t_emb
        """
        x = self.proj_enc(enc_feat_t)

        if h_prev is None:
            h_prev = self.gru.init_hidden(
                x.size(0), x.size(2), x.size(3), x.device, x.dtype
            )
        else:
            if self.use_flow_warp and flow_t_to_prev is not None:
                # Downsample flow to state-resolution and rescale magnitudes
                flow_state = downsample_flow(flow_t_to_prev, (x.size(2), x.size(3)))
                h_prev = backward_warp(h_prev, flow_state)

        h_t = self.gru(x, h_prev)

        feat_out  = self.proj_out(h_t)
        # GAP -> 320-dim delta for NRG conditioning
        t_emb_delta = self.proj_temb(h_t.mean(dim=(2, 3)))

        return h_t, feat_out, t_emb_delta
