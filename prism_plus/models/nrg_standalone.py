"""PRISM+ C2 — Standalone Mask-Conditioned NRG.

A clean re-implementation borrowing the proven recipe of Stable-Sim2Real
(arxiv 2507.23483) but replacing its ControlNet trunk with the lighter
SD-inpainting channel-concat conditioning.

What we BORROWED from Stable-Sim2Real:
    * SD-1.5 latent diffusion as the prior            (load Latent_SD.ckpt)
    * Depth is VAE-encoded as a 3-ch RGB-style image  (sim_depth.repeat(3))
    * Network learns the *residual* (real - sim), not real
    * Valid mask removes hole pixels from supervision
    * Empty text prompt + frozen text encoder (depth task needs no text)

What we DEPARTED FROM:
    * Stable-Sim2Real uses a side ControlNet trunk processing image-space
      hint through strided convs. We instead concatenate the sim-depth
      latent and the mask into the U-Net input channels — the same scheme
      as official SD-1.5 Inpainting where the U-Net accepts
      [z_t, image_latent, mask] = 4 + 4 + 1 = 9 channels. This is lighter
      (no 360M-param ControlNet) and lets the mask reach every U-Net block.

Design — 9-channel U-Net input:
    ch  0..3 : noisy residual latent  z_t
    ch  4..7 : VAE-encoded sim_depth  (frozen condition)
    ch  8    : down-sampled mask_density (in [-1, 1])

The mask is therefore visible to *every* denoising step at the input layer of
the U-Net, not just propagated via gradient reweighting. Concretely this is
how SD inpainting works (Rombach et al., 2022) and it is known to be a stable
conditioning channel.

Loading: weights come from Latent_SD.ckpt (or any compatible SD-1.5 single
file). We rebuild the U-Net first conv from 4 → 9 in-channels; the original
4-ch weight tensor is copied into ch0..3 and ch4..8 are zero-initialised so
the model starts as an exact no-op extension of the pre-trained U-Net.

Training loss:
    L = L_diff + lambda_boundary * L_boundary
where L_boundary uses Tweedie's one-step x0 estimate to decode an image-space
residual prediction, and MSE is taken only inside dilate(M_gt) ∪ dilate(M_hat).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

try:
    from ..diffusion.ldm.models.autoencoder import AutoencoderKL
    from ..diffusion.ldm.modules.diffusionmodules.openaimodel import UNetModel
    from ..diffusion.ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
except ImportError:
    from prism_plus.diffusion.ldm.models.autoencoder import AutoencoderKL
    from prism_plus.diffusion.ldm.modules.diffusionmodules.openaimodel import UNetModel
    from prism_plus.diffusion.ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder


# ---------------------------------------------------------------------------
# Diffusion schedule helpers (SD-1.5 default: 1000 step linear 0.00085→0.0120)
# ---------------------------------------------------------------------------

def _make_beta_schedule(num_steps: int = 1000,
                        linear_start: float = 0.00085,
                        linear_end: float = 0.0120) -> torch.Tensor:
    return torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_steps,
                          dtype=torch.float64) ** 2


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    return F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=radius)


# ---------------------------------------------------------------------------
# AutoencoderKL config (SD-1.5 default)
# ---------------------------------------------------------------------------

_VAE_DDCONFIG = {
    'double_z': True, 'z_channels': 4, 'resolution': 256,
    'in_channels': 3, 'out_ch': 3, 'ch': 128,
    'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2,
    'attn_resolutions': [], 'dropout': 0.0,
}
_VAE_LOSSCONFIG = {'target': 'torch.nn.Identity'}

# UNet config (SD-1.5 default + our 9-ch input)
_UNET_KWARGS = dict(
    image_size=32,
    in_channels=9,        # 4 noisy + 4 sim_latent + 1 mask  (was 4 in SD-1.5)
    out_channels=4,
    model_channels=320,
    attention_resolutions=[4, 2, 1],
    num_res_blocks=2,
    channel_mult=[1, 2, 4, 4],
    num_head_channels=64,
    use_spatial_transformer=True,
    use_linear_in_transformer=True,
    transformer_depth=1,
    context_dim=1024,     # OpenCLIP-bigG width
    legacy=False,
)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class NRGStandalone(nn.Module):
    """Mask-conditioned NRG, fully self-contained."""

    def __init__(
        self,
        sd_checkpoint: str,
        d_min: float = 0.05,
        d_max: float = 5.0,
        filter_max: float = 4.9,
        residual_scale: float = 2.0,
        scale_factor: float = 0.18215,
        # diffusion schedule
        num_timesteps: int = 1000,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        # losses
        lambda_boundary: float = 0.3,
        boundary_radius: int = 5,
        # frozen flags
        freeze_vae: bool = True,
        freeze_unet: bool = False,
        freeze_text: bool = True,
    ):
        super().__init__()
        self.d_min = float(d_min); self.d_max = float(d_max)
        self.filter_max = float(filter_max)
        self.residual_scale = float(residual_scale)
        self.scale_factor   = float(scale_factor)
        self.lambda_boundary = float(lambda_boundary)
        self.boundary_radius = int(boundary_radius)
        self.num_timesteps   = int(num_timesteps)

        # ---- sub-modules -------------------------------------------------
        self.vae = AutoencoderKL(
            embed_dim=4, monitor='val/rec_loss',
            ddconfig=_VAE_DDCONFIG, lossconfig=_VAE_LOSSCONFIG,
        )
        self.unet = UNetModel(**_UNET_KWARGS)
        self.text_encoder = FrozenOpenCLIPEmbedder(
            freeze=True, layer='penultimate', version='laion2b_s32b_b79k',
        )

        # ---- load SD checkpoint -----------------------------------------
        self._load_sd_checkpoint(sd_checkpoint)

        # ---- diffusion schedule -----------------------------------------
        betas = _make_beta_schedule(num_timesteps, linear_start, linear_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('sqrt_alphas_cumprod',
                             alphas_cumprod.sqrt().float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             (1.0 - alphas_cumprod).sqrt().float())

        # ---- freeze handles ---------------------------------------------
        if freeze_vae:
            for p in self.vae.parameters():     p.requires_grad = False
            self.vae.eval()
        if freeze_text:
            for p in self.text_encoder.parameters(): p.requires_grad = False
            self.text_encoder.eval()
        if freeze_unet:
            for p in self.unet.parameters():    p.requires_grad = False

        # ---- cache: empty-prompt text embedding -------------------------
        self.register_buffer('_empty_text_emb',
                             torch.zeros(1, 77, 1024), persistent=False)
        self._empty_text_emb_filled = False

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_sd_checkpoint(self, path: str) -> None:
        sd_full = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(sd_full, dict):
            sd_full = sd_full.get('state_dict', sd_full)

        # VAE
        vae_sd = {k[len('first_stage_model.'):]: v
                  for k, v in sd_full.items()
                  if k.startswith('first_stage_model.')}
        miss, unexp = self.vae.load_state_dict(vae_sd, strict=False)
        print(f'  [NRG-S] VAE  loaded: missing={len(miss)} unexpected={len(unexp)}')

        # text encoder
        txt_sd = {k[len('cond_stage_model.'):]: v
                  for k, v in sd_full.items()
                  if k.startswith('cond_stage_model.')}
        if txt_sd:
            miss, unexp = self.text_encoder.load_state_dict(txt_sd, strict=False)
            print(f'  [NRG-S] TXT  loaded: missing={len(miss)} unexpected={len(unexp)}')

        # UNet — first conv expand 4 → 9
        unet_sd = {k[len('model.diffusion_model.'):]: v
                   for k, v in sd_full.items()
                   if k.startswith('model.diffusion_model.')}
        old_w = unet_sd.get('input_blocks.0.0.weight')
        if old_w is not None and old_w.shape[1] == 4:
            target_in = _UNET_KWARGS['in_channels']
            if target_in != 4:
                new_w = torch.zeros(old_w.shape[0], target_in,
                                    *old_w.shape[2:], dtype=old_w.dtype)
                new_w[:, :4] = old_w
                unet_sd['input_blocks.0.0.weight'] = new_w
                print(f'  [NRG-S] expanded UNet conv_in: 4 -> {target_in} '
                      f'(new ch {4}..{target_in-1} zero-init)')
        miss, unexp = self.unet.load_state_dict(unet_sd, strict=False)
        print(f'  [NRG-S] UNet loaded: missing={len(miss)} unexpected={len(unexp)}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_scale_depth(self, depth: torch.Tensor) -> torch.Tensor:
        valid = (depth >= self.d_min) & (depth <= self.filter_max)
        d = depth.clamp(self.d_min, self.d_max)
        log_ratio = torch.log(torch.tensor(self.d_max / self.d_min,
                                            device=depth.device, dtype=depth.dtype))
        log_d = torch.log(d / self.d_min) / log_ratio
        invalid_fill = torch.ones_like(log_d)  # invalid → 1.0 in log-norm space
        return torch.where(valid, log_d, invalid_fill)

    def _inv_log_scale_depth(self, log_d: torch.Tensor) -> torch.Tensor:
        log_ratio = torch.log(torch.tensor(self.d_max / self.d_min,
                                            device=log_d.device, dtype=log_d.dtype))
        return self.d_min * torch.exp(log_d * log_ratio)

    def _to_3ch_norm(self, x_1ch: torch.Tensor) -> torch.Tensor:
        # VAE encoder wants 3-ch input in [-1, 1]; replicate the single channel.
        return x_1ch.repeat(1, 3, 1, 1)

    @torch.no_grad()
    def _encode_vae(self, x_norm_1ch: torch.Tensor) -> torch.Tensor:
        """x_norm_1ch in [-1, 1], returns scaled latent."""
        x_3ch = self._to_3ch_norm(x_norm_1ch)
        z = self.vae.encode(x_3ch).sample() * self.scale_factor
        return z

    def _decode_vae(self, z: torch.Tensor) -> torch.Tensor:
        """Decode scaled latent back to 3-ch image; caller averages channels."""
        return self.vae.decode(z / self.scale_factor)

    def _get_text_emb(self, bs: int, device: torch.device) -> torch.Tensor:
        if not self._empty_text_emb_filled:
            with torch.no_grad():
                emb = self.text_encoder([''])    # [1, 77, 1024]
            self._empty_text_emb = emb.detach().to(device)
            self._empty_text_emb_filled = True
        return self._empty_text_emb.expand(bs, -1, -1).to(device)

    def _q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                   noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return a * x_0 + s * noise

    def _tweedie_x0(self, z_noisy: torch.Tensor, eps_pred: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        s = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return (z_noisy - s * eps_pred) / a.clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sim_depth: torch.Tensor,            # [B,1,H,W] in metres
        real_depth: torch.Tensor,           # [B,1,H,W] in metres
        hole_mask: torch.Tensor,            # [B,1,H,W] binary GT (0/1)
        mask_density: torch.Tensor,         # [B,1,H,W] in [0,1] from BND v0.4
        rgb: Optional[torch.Tensor] = None,  # unused (text-empty); kept for API
    ) -> Dict[str, torch.Tensor]:
        device = sim_depth.device
        B = sim_depth.shape[0]

        # 1) image-space normalisation
        sim_log  = self._log_scale_depth(sim_depth)
        real_log = self._log_scale_depth(real_depth)
        sim_norm  = sim_log  * 2.0 - 1.0          # [-1, 1]
        real_norm = real_log * 2.0 - 1.0
        valid_mask = 1.0 - hole_mask
        residual = ((real_norm - sim_norm) / self.residual_scale) * valid_mask

        # 2) latent encoding (VAE frozen, no grad)
        z_target   = self._encode_vae(residual)     # [B,4,h,w], h=H/8
        z_sim      = self._encode_vae(sim_norm)     # frozen condition

        # 3) mask latent — downsample then re-range to [-1,1]
        h_lat, w_lat = z_target.shape[-2], z_target.shape[-1]
        mask_lat = F.interpolate(mask_density.clamp(0, 1),
                                 size=(h_lat, w_lat),
                                 mode='bilinear', align_corners=False)
        mask_lat = mask_lat * 2.0 - 1.0             # [-1, 1]

        # 4) diffusion forward
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        noise = torch.randn_like(z_target)
        z_noisy = self._q_sample(z_target, t, noise)

        # 5) build 9-ch UNet input
        unet_in = torch.cat([z_noisy, z_sim, mask_lat], dim=1)  # [B,9,h,w]
        text_emb = self._get_text_emb(B, device)
        eps_pred = self.unet(unet_in, t, context=text_emb)

        # 6) L_diff (epsilon-prediction MSE)
        l_diff = F.mse_loss(eps_pred, noise)

        # 7) L_boundary  (Tweedie 1-step → decode → image-space MSE in band)
        l_boundary = self._compute_boundary_loss(
            z_noisy=z_noisy, eps_pred=eps_pred, t=t,
            residual_gt=residual, hole_mask=hole_mask,
            mask_density=mask_density,
        )

        l_total = l_diff + self.lambda_boundary * l_boundary
        return {
            'loss':           l_total,
            'loss_diff':      l_diff.detach(),
            'loss_boundary':  l_boundary.detach(),
            'valid_ratio':    valid_mask.mean().detach(),
            'mask_mean':      mask_density.mean().detach(),
        }

    def _compute_boundary_loss(
        self,
        z_noisy: torch.Tensor,
        eps_pred: torch.Tensor,
        t: torch.Tensor,
        residual_gt: torch.Tensor,
        hole_mask: torch.Tensor,
        mask_density: torch.Tensor,
    ) -> torch.Tensor:
        z0 = self._tweedie_x0(z_noisy, eps_pred, t)
        # Decode (VAE frozen). decoder.forward goes through 3-ch path;
        # we average to 1ch since residual lives in 1-ch space.
        r_pred_3 = self._decode_vae(z0)             # [B,3,H,W]
        r_pred   = r_pred_3.mean(dim=1, keepdim=True)

        m_gt  = (hole_mask    > 0.5).float()
        m_hat = (mask_density > 0.5).float()
        r = self.boundary_radius
        band_gt  = (_dilate(m_gt,  r) - m_gt ).clamp(0.0, 1.0)
        band_hat = (_dilate(m_hat, r) - m_hat).clamp(0.0, 1.0)
        band = torch.maximum(band_gt, band_hat)
        denom = band.sum().clamp(min=1.0)
        return ((r_pred - residual_gt) * band).pow(2).sum() / denom

    # ------------------------------------------------------------------
    # Inference (ancestral DDIM)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        sim_depth: torch.Tensor,
        mask_density: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        rgb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = sim_depth.device
        B = sim_depth.shape[0]
        sim_log = self._log_scale_depth(sim_depth)
        sim_norm = sim_log * 2.0 - 1.0
        z_sim = self._encode_vae(sim_norm)
        h_lat, w_lat = z_sim.shape[-2], z_sim.shape[-1]
        mask_lat = F.interpolate(mask_density.clamp(0, 1),
                                 size=(h_lat, w_lat),
                                 mode='bilinear', align_corners=False)
        mask_lat = mask_lat * 2.0 - 1.0

        # DDIM step indices
        idx = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1,
                             device=device).long()
        z = torch.randn(B, 4, h_lat, w_lat, device=device)
        text_emb = self._get_text_emb(B, device)

        for i in range(num_steps):
            t_cur, t_nxt = idx[i], idx[i + 1]
            t_b = torch.full((B,), t_cur, device=device, dtype=torch.long)
            unet_in = torch.cat([z, z_sim, mask_lat], dim=1)
            eps = self.unet(unet_in, t_b, context=text_emb)
            a_t   = self.sqrt_alphas_cumprod[t_cur]
            s_t   = self.sqrt_one_minus_alphas_cumprod[t_cur]
            a_nxt = self.sqrt_alphas_cumprod[t_nxt]
            s_nxt = self.sqrt_one_minus_alphas_cumprod[t_nxt]
            x0_hat = (z - s_t * eps) / a_t.clamp(min=1e-6)
            z = a_nxt * x0_hat + s_nxt * eps

        # Decode → residual → reconstruct depth
        r_pred = self._decode_vae(z).mean(dim=1, keepdim=True)
        valid = (sim_depth >= self.d_min) & (sim_depth <= self.filter_max)
        sim_norm_v = sim_norm * valid.float()
        pred_real_norm = sim_norm_v + r_pred * self.residual_scale
        pred_real_log  = ((pred_real_norm + 1.0) / 2.0).clamp(0.0, 1.0)
        pred_real      = self._inv_log_scale_depth(pred_real_log) * valid.float()
        return pred_real
