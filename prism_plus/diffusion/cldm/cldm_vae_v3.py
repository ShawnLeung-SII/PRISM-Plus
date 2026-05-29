import torch
import torch.nn as nn

from prism_plus.diffusion.cldm.cldm import ControlNet
from prism_plus.diffusion.cldm.cldm_vae import ControlLDMVAE
from prism_plus.diffusion.ldm.modules.diffusionmodules.util import timestep_embedding


def _zero_init_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class WeakVFMSemanticAdaptor(nn.Module):
    """
    Weak global VFM conditioning.

    The adapter starts from an almost-zero contribution so the model can fall
    back to Stable-S2R behavior, but it still has a tiny gradient path.
    """

    def __init__(
        self,
        vfm_dim: int,
        model_channels: int,
        time_embed_dim: int,
        scale_floor: float = 1e-3,
    ):
        super().__init__()
        hidden_dim = max(vfm_dim // 4, 128)
        self.encoder = nn.Sequential(
            nn.Linear(vfm_dim, vfm_dim // 2),
            nn.LayerNorm(vfm_dim // 2),
            nn.GELU(),
            nn.Linear(vfm_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            _zero_init_linear(nn.Linear(hidden_dim, time_embed_dim)),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            _zero_init_linear(nn.Linear(hidden_dim, model_channels)),
            nn.Tanh(),
        )

        self.alpha_t = nn.Parameter(torch.zeros(1))
        self.alpha_c = nn.Parameter(torch.zeros(1))
        self.scale_floor = scale_floor

    def _pool_tokens(self, vfm_features: torch.Tensor) -> torch.Tensor:
        if vfm_features.dim() == 3:
            return vfm_features.mean(dim=1)
        if vfm_features.dim() == 4:
            return vfm_features.mean(dim=(2, 3))
        raise RuntimeError(f"Unexpected VFM feature shape: {tuple(vfm_features.shape)}")

    def forward(self, vfm_features: torch.Tensor):
        pooled = self._pool_tokens(vfm_features)
        code = self.encoder(pooled)
        delta_t = self.time_mlp(code)
        gate = self.channel_mlp(code).unsqueeze(-1).unsqueeze(-1)

        time_scale = self.alpha_t + self.scale_floor
        channel_scale = self.alpha_c + self.scale_floor

        return delta_t * time_scale, gate * channel_scale


class ControlNetV3(ControlNet):
    def __init__(self, *args, use_vfm: bool = False, vfm_dim: int = None, **kwargs):
        super().__init__(*args, use_vfm=False, vfm_dim=None, **kwargs)
        self.use_vfm = use_vfm
        self.vfm_dim = vfm_dim
        if self.use_vfm:
            if self.vfm_dim is None:
                raise ValueError("vfm_dim must be provided when use_vfm=True")
            self.vfm_adaptor = WeakVFMSemanticAdaptor(
                vfm_dim=self.vfm_dim,
                model_channels=self.model_channels,
                time_embed_dim=self.model_channels * 4,
            )
        else:
            self.vfm_adaptor = None

    def forward(self, x, hint, timesteps, context, vfm_features=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        gate = None
        if self.use_vfm and self.vfm_adaptor is not None and vfm_features is not None:
            if vfm_features.device != emb.device:
                vfm_features = vfm_features.to(emb.device)
            delta_t, gate = self.vfm_adaptor(vfm_features)
            emb = emb + delta_t

        guided_hint = self.input_hint_block(hint.type(self.dtype), emb, context)
        if gate is not None:
            guided_hint = guided_hint * (1.0 + gate)

        outs = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h = h + guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        return outs


class ControlLDMVAEV3(ControlLDMVAE):
    pass

