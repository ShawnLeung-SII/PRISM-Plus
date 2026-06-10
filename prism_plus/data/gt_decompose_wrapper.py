"""Wrap any existing dataset to return (M_target, M_ignore) instead of M_raw.

Usage:
    base = ByteCamDepthDataset(...)
    ds   = GTDecomposeWrapper(base, preset="medium")
    # ds[i] yields rgb, sim_depth, real_depth, hole_mask, m_target, m_ignore
"""
from __future__ import annotations
from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from .gt_decompose import decompose_gt_preset, DEFAULT_PRESET


class GTDecomposeWrapper(Dataset):
    """On-the-fly GT decomposition for v0.4.0 density-aware training.

    Adds two keys to each sample:
        m_target : hard binary, supervised target
        m_ignore : hard binary, pixels excluded from loss
    `hole_mask` (raw GT) is preserved for backward compatibility & evaluation.
    """

    def __init__(self, base: Dataset, preset: str = DEFAULT_PRESET):
        self.base = base
        self.preset = preset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        gt = sample["hole_mask"]    # [1, H, W] or [H, W]
        m_target, m_ignore = decompose_gt_preset(gt, self.preset)
        sample["m_target"] = m_target
        sample["m_ignore"] = m_ignore
        return sample
