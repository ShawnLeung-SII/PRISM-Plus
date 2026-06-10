"""GT mask multi-dimensional decomposition (for v0.4.0 density-aware training).

Splits the raw single-shot binary GT mask into:
    M_target  : hard binary, structured failure that the model SHOULD learn
    M_ignore  : hard binary, isolated/random noise pixels EXCLUDED from loss

The decomposition uses three independent criteria, ALL must pass for a hole
pixel to be considered "structured" (i.e. included in M_target):

    1. Morphological open (removes ≤kernel×kernel dots)
    2. Connected-component area filter (removes blobs with area < area_thresh)
    3. Local density (removes pixels whose 5×5 neighbourhood has <density_thresh holes)

Rationale: see refine-logs/BND_ARCHITECTURE_REVIEW.md and the user discussion
on 2026-06-10. We do NOT want soft labels (0.5) for "uncertain" pixels —
sigmoid + BCE training on the structured-only target naturally yields a
calibrated density σ(f*(x)) = P(hole|x).
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Preset configurations (chosen 2026-06-10 after 100-sample visual review)
# ---------------------------------------------------------------------------

PRESETS = {
    "conservative": dict(kernel=3, area_thresh=5,  density_window=5, density_thresh=3),
    "medium":       dict(kernel=3, area_thresh=10, density_window=5, density_thresh=5),
    "aggressive":   dict(kernel=5, area_thresh=20, density_window=5, density_thresh=5),
}

DEFAULT_PRESET = "medium"   # User decision 2026-06-10


# ---------------------------------------------------------------------------
# GPU-friendly morphological operators
# ---------------------------------------------------------------------------

def _dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    """Binary dilation via max-pool. Accepts [B,C,H,W] or [C,H,W] or [H,W]."""
    while m.dim() < 4:
        m = m.unsqueeze(0)
    out = F.max_pool2d(m.float(), 2 * r + 1, 1, padding=r)
    return out


def _erode(m: torch.Tensor, r: int) -> torch.Tensor:
    while m.dim() < 4:
        m = m.unsqueeze(0)
    out = -F.max_pool2d(-m.float(), 2 * r + 1, 1, padding=r)
    return out


def morphological_open(m: torch.Tensor, kernel: int) -> torch.Tensor:
    """Open = erode then dilate. Removes isolated dots smaller than kernel²."""
    r = (kernel - 1) // 2
    return _dilate(_erode(m, r), r)


def remove_small_components_np(mask01: np.ndarray, min_area: int) -> np.ndarray:
    """CPU connected-component filter (numpy + scipy).

    Removes connected components with area < min_area.
    """
    from scipy import ndimage
    if mask01.sum() < min_area:
        return np.zeros_like(mask01)
    labels, n = ndimage.label(mask01 > 0.5)
    if n == 0:
        return mask01
    sizes = ndimage.sum(mask01 > 0.5, labels, range(1, n + 1))
    # LUT: idx 0 = background, idx i+1 = label i+1
    lut = np.concatenate(([0], (sizes >= min_area).astype(mask01.dtype)))
    return lut[labels]


def local_density_filter(m: torch.Tensor, window: int, thresh: int) -> torch.Tensor:
    """For each pixel, count neighbouring GT positives.

    Returns binary mask where count >= thresh.
    """
    while m.dim() < 4:
        m = m.unsqueeze(0)
    s = F.avg_pool2d(m.float(), window, 1, padding=window // 2) * (window * window)
    return (s >= thresh).float()


# ---------------------------------------------------------------------------
# Main API: decompose_gt
# ---------------------------------------------------------------------------

def decompose_gt(
    gt_raw: torch.Tensor,
    kernel: int = 3,
    area_thresh: int = 10,
    density_window: int = 5,
    density_thresh: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose hard binary GT into (M_target, M_ignore).

    Args:
        gt_raw: shape [..., H, W] hard binary in {0, 1}.
                Accepts [H,W], [1,H,W], [B,1,H,W] — output keeps input shape.
        kernel, area_thresh, density_window, density_thresh: see module docstring.

    Returns:
        M_target : hard binary, structured failure pixels
        M_ignore : hard binary, GT positives excluded from supervision

    Note:
        Both outputs are HARD binary (no soft labels). The density behaviour
        of the trained model emerges from BCE + sigmoid, not from soft GT.
    """
    orig_shape = gt_raw.shape
    orig_device = gt_raw.device

    # Normalise shape to [B,1,H,W]
    g = gt_raw.float()
    while g.dim() < 4:
        g = g.unsqueeze(0)
    B = g.shape[0]

    # 1) Morphological open
    m_open = morphological_open(g, kernel)

    # 2) Connected-component filter — must go to CPU (scipy.ndimage)
    m_open_np = m_open.detach().cpu().numpy()
    m_cc_np = np.stack([
        remove_small_components_np(m_open_np[b, 0], area_thresh)
        for b in range(B)
    ])[:, None]  # add channel dim back
    m_cc = torch.from_numpy(m_cc_np).to(orig_device, dtype=g.dtype)

    # 3) Local density filter (back on GPU)
    m_dense = local_density_filter(g, density_window, density_thresh) * g

    # M_target = intersection of all three criteria
    m_target = (m_cc * m_dense).clamp(0, 1)

    # M_ignore = original GT positives NOT in m_target
    m_ignore = g * (1.0 - m_target)

    # Restore original shape
    return m_target.view(orig_shape), m_ignore.view(orig_shape)


def decompose_gt_preset(gt_raw: torch.Tensor, preset: str = DEFAULT_PRESET):
    """Convenience wrapper using a named preset."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choices: {list(PRESETS)}")
    return decompose_gt(gt_raw, **PRESETS[preset])
