"""PRISM+ C3 — LoRA-SPA (Sensor-Prompt Adaptation via low-rank adapter).

Implements per-sensor rank-r LoRA on top of the SPR semantic embedding
pathway of PRISMPlusBND. From the FINAL_PROPOSAL:

    S_l^s = S_l^attn + (A_s · B_s^T) · z_sem

where:
    z_sem is the global VFM semantic embedding (GAP of patch tokens, 1024-dim)
    A_s, B_s are per-sensor matrices (rank r=4 by default)
    The product (A_s · B_s^T) modulates how z_sem maps into the BND decoder.

Design:
    - Backbone (PRISMPlusBND + VFM + decoder) is FROZEN.
    - Only the per-sensor LoRA tensors {A_s, B_s} are trainable.
    - Total new params per sensor ≈ 2 * d_sem * r = 2 * 1024 * 4 = 8K (0.008M).
    - Multiple sensors share the same backbone but each has own A_s, B_s.

Usage:
    wrapper = LoRASPA(base_model, sensor_ids=['dreds_d415', 'dreds_l515'])
    wrapper.set_active_sensor('dreds_d415')
    out = wrapper(rgb, sim_depth)   # backbone forward with sensor-specific delta
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Single rank-r LoRA on a (d_in -> d_out) linear path.

    Initialised so that delta = A @ B.T applied to input gives 0 at start
    (B is zero-init); after training, delta provides a low-rank perturbation
    of the original semantic-to-projection map.
    """

    def __init__(self, d_in: int, d_out: int, rank: int = 4,
                 alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        # A: d_in -> r ;  B: r -> d_out
        self.A = nn.Parameter(torch.empty(d_in, rank))
        self.B = nn.Parameter(torch.zeros(rank, d_out))   # zero-init → delta = 0 at step 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] -> delta: [..., d_out]."""
        x = self.dropout(x)
        return (x @ self.A) @ self.B * self.scaling


class LoRASPA(nn.Module):
    """Multi-sensor LoRA wrapper for PRISMPlusBND.

    Each sensor gets a private (A_s, B_s) modulating the semantic vector.

    The wrapper monkey-patches the base model's z_sem pathway:
        z_sem' = z_sem + LoRA_s(z_sem)
    where the LoRA is applied AS A RESIDUAL CORRECTION inside the same dim.

    Backbone is fully frozen — only the LoRA tensors require grad.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sensor_ids: Iterable[str],
        d_sem: int = 1024,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base_model
        self.sensor_ids = list(sensor_ids)
        self.d_sem = int(d_sem)
        self.rank = int(rank)

        # Per-sensor LoRA
        self.adapters = nn.ModuleDict({
            sid: LoRAAdapter(d_in=d_sem, d_out=d_sem, rank=rank,
                              alpha=alpha, dropout=dropout)
            for sid in self.sensor_ids
        })

        # Freeze base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # Active sensor — must be set before forward
        self._active_sensor: Optional[str] = None
        if len(self.sensor_ids) == 1:
            self._active_sensor = self.sensor_ids[0]

        # Hook into base model's VFM pathway
        self._install_hook()

    def set_active_sensor(self, sensor_id: str) -> None:
        if sensor_id not in self.sensor_ids:
            raise KeyError(f"unknown sensor_id '{sensor_id}'. available: {self.sensor_ids}")
        self._active_sensor = sensor_id

    def get_trainable_params(self, sensor_id: Optional[str] = None):
        """Return only the LoRA params for one (or all) sensors."""
        if sensor_id is None:
            return [p for ad in self.adapters.values() for p in ad.parameters()]
        return list(self.adapters[sensor_id].parameters())

    def _install_hook(self) -> None:
        """Register a forward-hook on the VFM that adds the active LoRA delta.

        We hook on the VFM backbone to intercept z_sem. PRISMPlusBND's VFM
        is  and the global semantic embedding is the GAP-mean
        of patch tokens. We patch the VFM forward to add LoRA(z_sem).
        """
        # PRISMPlusBND keeps the VFM under semantic_context.vfm; older BND has it
        # at self.vfm directly. Probe both.
        vfm = getattr(self.base, 'vfm', None)
        if vfm is None:
            sc = getattr(self.base, 'semantic_context', None)
            if sc is not None:
                vfm = getattr(sc, 'vfm', None)
        if vfm is None:
            vfm = getattr(self.base, '_vfm', None)
        if vfm is None:
            raise RuntimeError(
                'base model has no .vfm / .semantic_context.vfm / ._vfm attribute'
            )

        original_forward = vfm.forward
        wrapper = self

        def lora_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            sid = wrapper._active_sensor
            if sid is None:
                return out
            ad = wrapper.adapters[sid]
            # SemanticContext wraps the VFM call in torch.no_grad(), so we
            # must locally re-enable autograd around the LoRA delta or the
            # backward pass will fail. tok itself stays detached; only the
            # delta carries gradient.
            with torch.enable_grad():
                if isinstance(out, dict):
                    for key in ('x_norm_patchtokens', 'last_hidden_state'):
                        if key in out and isinstance(out[key], torch.Tensor):
                            out[key] = out[key] + ad(out[key])
                            return out
                    return out
                if isinstance(out, list):
                    if out and isinstance(out[-1], torch.Tensor) and out[-1].dim() == 3:
                        out = list(out)
                        out[-1] = out[-1] + ad(out[-1])
                    return out
                if isinstance(out, torch.Tensor):
                    if out.dim() in (2, 3):
                        return out + ad(out)
            return out

        vfm.forward = lora_forward

    def forward(self, *args, **kwargs):
        if self._active_sensor is None:
            raise RuntimeError('Call .set_active_sensor(sid) before forward')
        return self.base(*args, **kwargs)


def create_lora_spa(
    base_model: nn.Module,
    sensor_ids: Iterable[str],
    rank: int = 4,
    alpha: float = 1.0,
    d_sem: int = 1024,
) -> LoRASPA:
    return LoRASPA(base_model=base_model, sensor_ids=sensor_ids,
                   d_sem=d_sem, rank=rank, alpha=alpha)
