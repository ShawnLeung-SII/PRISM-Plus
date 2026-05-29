import einops
import torch
import torch as th
import torch.nn as nn

from prism_plus.diffusion.ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from prism_plus.diffusion.ldm.modules.attention import SpatialTransformer
from prism_plus.diffusion.ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from prism_plus.diffusion.ldm.models.diffusion.ddpm_stage1 import LatentDiffusion
from prism_plus.diffusion.ldm.util import log_txt_as_img, exists, instantiate_from_config
from prism_plus.diffusion.ldm.models.diffusion.ddim import DDIMSampler
from prism_plus.diffusion.sampler.registry_class import SAMPLER
from prism_plus.diffusion.cldm.cldm import ControlNet
import pdb


class ControlLDMVAE(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control, prior_model=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        if prior_model is not None:
            from tqlt.prior_model.registry_class import NORMAL_PRIOR
            self.prior = NORMAL_PRIOR.build(dict(type=prior_model.name))
        else:
            self.prior = None

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)  # x is actually latent code
        control = batch[self.control_key]  # hint
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()

        # using prior model to capture x0
        if self.prior is not None:
            # [-1,1] -> [0, 255]
            prior_hint = (batch['hint']+1.) / 2 * 255.
            _, hint_h, hint_w, _ = prior_hint.shape
            size_info = [hint_w, hint_h, 0, 0]
            self.prior.to(self.device)
            # [-1, 1]
            prior_out = self.prior(prior_hint, size_info=size_info)['abs_vals']
            # prior embedding
            prior_posterior = self.encode_first_stage(prior_out)
            prior_z = self.get_first_stage_encoding(prior_posterior).detach()
        else:
            prior_z = None

        res = dict(c_crossattn=[c], c_concat=[control_z], prior_out=[prior_z])
        
        # Handle VFM features
        if 'vfm_features' in batch and batch['vfm_features'] is not None:
            vfm_feat = batch['vfm_features']
            if bs is not None:
                vfm_feat = vfm_feat[:bs]
            vfm_feat = vfm_feat.to(self.device)
            res['vfm_features'] = vfm_feat
        
        # Handle uncertainty weight map for uncertainty-aware training
        # Downsample to latent space resolution (H/8, W/8)
        if 'w_map' in batch and batch['w_map'] is not None:
            w_map = batch['w_map']
            if bs is not None:
                w_map = w_map[:bs]
            w_map = w_map.to(self.device)
            
            # w_map could be [B, 1, H, W] (channel first) or [B, H, W] or [B, H, W, 1] (channel last)
            # Normalize to [B, 1, H, W] format
            if w_map.dim() == 3:
                w_map = w_map.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            elif w_map.dim() == 4 and w_map.shape[-1] == 1:
                w_map = einops.rearrange(w_map, 'b h w c -> b c h w')  # [B, H, W, C] -> [B, C, H, W]
            # else: already [B, C, H, W]
            
            # Downsample to latent space (H/8, W/8)
            latent_h, latent_w = x.shape[2], x.shape[3]  # x is already in latent space
            w_map_latent = torch.nn.functional.interpolate(
                w_map.float(), 
                size=(latent_h, latent_w), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Keep as [B, 1, H, W] format for get_loss()
            res['w_map'] = w_map_latent
        
        return x, res

    @torch.no_grad()
    def get_control(self, control):
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()


        return control_z


    @torch.no_grad()
    def get_hints(self, image_normalized):
        control = image_normalized
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()


        return {"c_concat" : [control_z]}


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        vfm_features = cond.get('vfm_features', None)
        
        # Get weight map for uncertainty-aware training
        # w_map should be in latent resolution (H/8, W/8)
        w_map = cond.get('w_map', None)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            hint = torch.cat(cond['c_concat'], 1)
            control = self.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt, vfm_features=vfm_features)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        # Return (eps, w_map) tuple only when w_map is provided (training mode)
        # Otherwise return eps only (inference mode - for DDIM sampler compatibility)
        if w_map is not None:
            return eps, w_map
        else:
            return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=5, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)  # bs valid for control.....


        c_cat, c = c["c_concat"][0][:N],  c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        z = z[:N]
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = self.decode_first_stage(c_cat)
        _,_, img_size_h, img_size_w = log['reconstruction'].shape
        log["conditioning"] = log_txt_as_img((img_size_w, img_size_h), batch[self.cond_stage_key][:N], size=16)

        log_sequence=['control', 'conditioning']

        if self.prior is not None:
            c_prior = c["prior_out"][0][:N]
            log['prior'] = self.decode_first_stage(c_prior)

            log_sequence.append('prior')
        else:
            c_prior = None
        log_sequence.append('samples')

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row (get the denoised lated code given noise and condition)
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)  # decode
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

            log_sequence.append(f"samples_cfg_scale_{unconditional_guidance_scale:.2f}")

        log_sequence.append('reconstruction')
        log['visualized'] = torch.cat([log[key].detach().cpu() for key in log_sequence], dim = -2)

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        ddim_sampler = SAMPLER.build(dict(type=self.sampler_type), model=self)

        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h, w)

        if self.prior is not None:
            c_prior = cond.pop('c_prior')
            kwargs['x0'] = c_prior[0]

        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.model.diffusion_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


class ControlNetVAE(ControlNet):
    def forward(self, x, hint, timesteps, context, vfm_features=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        outs = []

        h = hint.type(self.dtype)
        
        # ============ VFM 语义调制 (改进版) ============
        # 预先计算 VFM channel modulation，避免块状伪影
        vfm_channel_weight = None
        if self.use_vfm and vfm_features is not None:
            target_size = h.shape[-2:]
            if vfm_features.device != h.device:
                vfm_features = vfm_features.to(h.device)
            vfm_output = self.vfm_projector(vfm_features, target_size)
            
            # 检查是否是新版 Channel Modulation 输出
            if vfm_output.dim() == 4 and vfm_output.shape[-2:] == (1, 1):
                vfm_channel_weight = vfm_output  # [B, C, 1, 1]
            else:
                # 旧版兼容: 保存空间特征用于后续注入
                vfm_channel_weight = vfm_output  # 作为空间特征
        
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            
            # Inject VFM features after the first block (which converts in_channels -> model_channels)
            if i == 0 and vfm_channel_weight is not None:
                if vfm_channel_weight.shape[-2:] == (1, 1):
                    # 新版: Channel Modulation，使用残差调制
                    h = h * (1.0 + vfm_channel_weight)
                else:
                    # 旧版兼容: 空间特征直接相加
                    h = h + vfm_channel_weight
                
            outs.append(h)
        h = self.middle_block(h, emb, context)
        outs.append(h)

        return outs

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            # training all paramters
            params += list(self.model.diffusion_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

import torch.nn.functional as F
from prism_plus.diffusion.ldm.util import default

class MaskedControlLDMVAE(ControlLDMVAE):
    """
    ControlLDMVAE with Masked Loss Support
    Used for Latent Branch to ignore hole regions during training.
    """
    def p_losses(self, x_start, cond, t, noise=None, valid_mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # Calculate element-wise loss (mean=False)
        loss_elementwise = self.get_loss(model_output, target, mean=False)
        
        # Apply mask if provided
        if valid_mask is not None:
            # valid_mask is [B, 1, H, W] (original resolution)
            # loss_elementwise is [B, C, H_latent, W_latent]
            
            # 1. Resize mask to latent resolution
            if valid_mask.shape[-1] != loss_elementwise.shape[-1]:
                valid_mask = F.interpolate(valid_mask, size=loss_elementwise.shape[-2:], mode='nearest')
            
            # 2. Expand mask to channel dimension
            if valid_mask.shape[1] != loss_elementwise.shape[1]:
                valid_mask = valid_mask.repeat(1, loss_elementwise.shape[1], 1, 1)
                
            # 3. Apply mask
            loss = (loss_elementwise * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            
            # Log mask ratio
            loss_dict.update({f'{prefix}/mask_ratio': valid_mask.mean()})
        else:
            loss = loss_elementwise.mean()

        loss_dict.update({f'{prefix}/loss_simple': loss})

        # We ignore VLB loss for simplicity in masked training as it's complex to mask correctly
        # and usually small.
        
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
