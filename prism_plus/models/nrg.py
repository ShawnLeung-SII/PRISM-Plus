"""PRISM Noise Residual Generator (NRG).

Baseline implementation from PRISM (ICML 2026) — Stable Diffusion ControlNet
that synthesizes the measurement residual R given log-scale sim depth.

VFM features are injected as a global style prompt into ControlNet's time
embedding (eq. 5 of the paper).

For the PRISM+ C2 (mask-conditioned) variant, see
:class:`prism_plus.models.mnrg.MaskConditionedNRG` (TODO).
"""
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .vfm import create_vfm

try:
    from ..diffusion.cldm.model import load_state_dict
    from ..diffusion.ldm.models.diffusion.ddim import DDIMSampler
    from ..diffusion.ldm.util import instantiate_from_config
except ImportError:
    from prism_plus.diffusion.cldm.model import load_state_dict
    from prism_plus.diffusion.ldm.models.diffusion.ddim import DDIMSampler
    from prism_plus.diffusion.ldm.util import instantiate_from_config


class NRG(nn.Module):
    def __init__(
        self,
        vfm_type: str = "moge2",
        vfm_checkpoint: Optional[str] = None,
        vfm_freeze: bool = True,
        stable_s2r_config: Optional[str] = None,
        stable_s2r_checkpoint: Optional[str] = None,
        use_vfm_injection: bool = True,
        freeze_vae: bool = True,
        freeze_unet: bool = True,
        num_ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        guidance_scale: float = 9.0,
        residual_scale: float = 2.0,
        d_min: float = 0.05,
        d_max: float = 5.0,
        filter_max: float = 4.9,
    ):
        super().__init__()

        self.use_vfm_injection = use_vfm_injection
        self.num_ddim_steps = num_ddim_steps
        self.ddim_eta = ddim_eta
        self.guidance_scale = guidance_scale
        self.residual_scale = residual_scale
        self.d_min = d_min
        self.d_max = d_max
        self.filter_max = filter_max
        self.invalid_norm_token = 1.0
        self.ddim_sampler = None

        if stable_s2r_config is None:
            raise ValueError("stable_s2r_config must be provided")

        self.vfm = None
        self.vfm_dim = None
        if self.use_vfm_injection:
            self.vfm = create_vfm(vfm_type=vfm_type, checkpoint_path=vfm_checkpoint, freeze=vfm_freeze)
            self.vfm_dim = getattr(self.vfm, "feature_dim", getattr(self.vfm, "embed_dim", None))
            if self.vfm_dim is None:
                raise ValueError("Failed to infer VFM feature dimension")

        config = OmegaConf.load(stable_s2r_config)
        control_cfg = config.model.params.control_stage_config
        if "params" not in control_cfg:
            control_cfg["params"] = {}
        control_cfg.params["use_vfm"] = bool(self.use_vfm_injection)
        control_cfg.params["vfm_dim"] = self.vfm_dim

        self.model = instantiate_from_config(config.model).cpu()

        if stable_s2r_checkpoint and os.path.exists(stable_s2r_checkpoint):
            state_dict = load_state_dict(stable_s2r_checkpoint, location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        if freeze_vae:
            for param in self.model.first_stage_model.parameters():
                param.requires_grad = False
            self.model.first_stage_model.eval()
            self.model.first_stage_model.float()

        if freeze_unet:
            for param in self.model.model.diffusion_model.parameters():
                param.requires_grad = False

    def _log_scale_depth(self, depth: torch.Tensor) -> torch.Tensor:
        valid_mask = (depth >= self.d_min) & (depth <= self.filter_max)
        depth_clamped = torch.clamp(depth, min=self.d_min, max=self.d_max)
        log_depth = torch.log(depth_clamped / self.d_min) / torch.log(
            torch.tensor(self.d_max / self.d_min, device=depth.device, dtype=depth.dtype)
        )
        invalid_fill = torch.ones_like(log_depth)
        return torch.where(valid_mask, log_depth, invalid_fill)

    def _inverse_log_scale_depth(self, log_depth: torch.Tensor) -> torch.Tensor:
        return self.d_min * torch.exp(
            log_depth * torch.log(torch.tensor(self.d_max / self.d_min, device=log_depth.device, dtype=log_depth.dtype))
        )

    @torch.no_grad()
    def extract_vfm_features(self, rgb: torch.Tensor) -> torch.Tensor:
        features = self.vfm(rgb)
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "last_hidden_state"):
                if key in features:
                    return features[key]
            return next(iter(features.values()))
        if isinstance(features, list):
            return features[-1]
        return features

    def _prepare_training_batch(
        self,
        sim_depth: torch.Tensor,
        real_depth: torch.Tensor,
        hole_mask: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        valid_mask = 1.0 - hole_mask

        sim_depth_log = self._log_scale_depth(sim_depth)
        real_depth_log = self._log_scale_depth(real_depth)

        sim_depth_norm = sim_depth_log * 2.0 - 1.0
        real_depth_norm = real_depth_log * 2.0 - 1.0

        condition_norm = sim_depth_norm * valid_mask + self.invalid_norm_token * hole_mask
        target_residual = ((real_depth_norm - sim_depth_norm) / self.residual_scale) * valid_mask

        batch = {
            "jpg": target_residual.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            "txt": [""] * sim_depth.shape[0],
            "hint": condition_norm.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            "prior": condition_norm.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            "w_map": valid_mask,
        }

        if self.use_vfm_injection and rgb is not None:
            batch["vfm_features"] = self.extract_vfm_features(rgb)
        else:
            batch["vfm_features"] = None

        return batch, {
            "sim_depth_norm": sim_depth_norm,
            "real_depth_norm": real_depth_norm,
            "condition_norm": condition_norm,
            "target_residual": target_residual,
            "valid_mask": valid_mask,
        }

    def forward(
        self,
        sim_depth: torch.Tensor,
        rgb: torch.Tensor,
        real_depth: torch.Tensor,
        hole_mask: torch.Tensor,
        training: bool = True,
    ):
        batch, aux = self._prepare_training_batch(sim_depth, real_depth, hole_mask, rgb)
        diffusion_loss, diffusion_loss_dict = self.model.shared_step(batch)

        log_loss = diffusion_loss_dict.get("train/loss" if self.training else "val/loss", diffusion_loss)
        loss_dict = {
            "loss": diffusion_loss,
            "loss_diffusion": log_loss.detach() if isinstance(log_loss, torch.Tensor) else diffusion_loss.detach(),
            "valid_ratio": aux["valid_mask"].mean().detach(),
        }
        return None, loss_dict, {"vfm_injected": self.use_vfm_injection}

    @torch.no_grad()
    def inference(
        self,
        sim_depth: torch.Tensor,
        rgb: torch.Tensor,
        hole_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        eta: Optional[float] = None,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if hole_mask is None:
            hole_mask = torch.zeros_like(sim_depth)

        if num_steps is None:
            num_steps = self.num_ddim_steps
        if eta is None:
            eta = self.ddim_eta
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        valid_mask = 1.0 - hole_mask
        sim_depth_log = self._log_scale_depth(sim_depth)
        sim_depth_norm = sim_depth_log * 2.0 - 1.0
        condition_norm = sim_depth_norm * valid_mask + self.invalid_norm_token * hole_mask
        prior = condition_norm.repeat(1, 3, 1, 1)

        if self.ddim_sampler is None:
            self.ddim_sampler = DDIMSampler(self.model)

        c_text = self.model.get_learned_conditioning([""] * sim_depth.shape[0])
        control_posterior = self.model.encode_first_stage(prior)
        c_control = self.model.get_first_stage_encoding(control_posterior).detach()
        cond = {"c_concat": [c_control], "c_crossattn": [c_text]}
        uc = {
            "c_concat": [c_control],
            "c_crossattn": [self.model.get_unconditional_conditioning(sim_depth.shape[0])],
        }

        if self.use_vfm_injection:
            vfm_features = self.extract_vfm_features(rgb)
            cond["vfm_features"] = vfm_features
            uc["vfm_features"] = vfm_features

        shape = (4, sim_depth.shape[-2] // 8, sim_depth.shape[-1] // 8)
        samples, _ = self.ddim_sampler.sample(
            S=num_steps,
            conditioning=cond,
            batch_size=sim_depth.shape[0],
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
            eta=eta,
        )

        decoded = self.model.decode_first_stage(samples)
        pred_residual = decoded.mean(dim=1, keepdim=True) * valid_mask
        pred_real_norm = sim_depth_norm + pred_residual * self.residual_scale
        pred_real_log = ((pred_real_norm + 1.0) / 2.0).clamp(0.0, 1.0)
        noisy_depth = self._inverse_log_scale_depth(pred_real_log)
        noisy_depth = noisy_depth * valid_mask
        return noisy_depth

    def configure_training_stage(self, stage: int):
        for param in self.parameters():
            param.requires_grad = False

        if stage == 1:
            for param in self.model.control_model.parameters():
                param.requires_grad = True
        elif stage == 2:
            for param in self.model.control_model.parameters():
                param.requires_grad = True
            for param in self.model.model.diffusion_model.parameters():
                param.requires_grad = True
            for param in self.model.first_stage_model.decoder.parameters():
                param.requires_grad = True
        else:
            raise ValueError("NRG only supports stage 1 (ControlNet) and stage 2 (end-to-end)")

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

