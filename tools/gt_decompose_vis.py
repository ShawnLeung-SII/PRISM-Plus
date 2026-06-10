"""GT 多维度分解 — 三档参数（保守 / 中等 / 激进）可视化对比工具。

为 v0.4.0 density-aware 训练做 GT 预处理参数选择。
对每个样本输出 4x4 panel：
    Row 1: RGB | Sim Depth | Real Depth (GT hole) | GT_raw (binary)
    Row 2: 保守 M_target | 保守 M_ignore | 保守 overlay (G=target/B=ignore)
    Row 3: 中等 M_target | 中等 M_ignore | 中等 overlay
    Row 4: 激进 M_target | 激进 M_ignore | 激进 overlay

同时输出 summary.csv 含每档的：
    n_target_pixels, n_ignore_pixels, ignore_ratio (vs GT_raw)

Run:
    python tools/gt_decompose_vis.py \\
        --config configs/stage1_bnd_plus.yaml \\
        --n 100 \\
        --out /inspire/hdd/.../gt_decompose_vis/
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data import ByteCamDepthDataset


# ---------------------------------------------------------------------------
# Morphological / connected-component / density operators (GPU-friendly)
# ---------------------------------------------------------------------------

def _dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    return F.max_pool2d(m.float(), 2 * r + 1, 1, padding=r)


def _erode(m: torch.Tensor, r: int) -> torch.Tensor:
    return -F.max_pool2d(-m.float(), 2 * r + 1, 1, padding=r)


def morphological_open(m: torch.Tensor, kernel: int) -> torch.Tensor:
    """Open = erode then dilate; removes isolated dots smaller than kernel²."""
    r = (kernel - 1) // 2
    return _dilate(_erode(m, r), r)


def remove_small_components(m_bin: np.ndarray, min_area: int) -> np.ndarray:
    """CPU numpy connected-components; removes blobs with area < min_area."""
    from scipy import ndimage
    labels, n = ndimage.label(m_bin > 0.5)
    if n == 0:
        return m_bin
    sizes = ndimage.sum(m_bin > 0.5, labels, range(1, n + 1))
    # build size LUT, idx 0 = background = 0
    lut = np.concatenate(([0], (sizes >= min_area).astype(m_bin.dtype)))
    return lut[labels]


def local_density_filter(m: torch.Tensor, window: int, thresh: int) -> torch.Tensor:
    """For each pixel, count GT pixels in window×window neighbourhood.
    Returns binary mask where count >= thresh."""
    s = F.avg_pool2d(m.float(), window, 1, padding=window // 2) * (window * window)
    return (s >= thresh).float()


# ---------------------------------------------------------------------------
# Decompose: return (M_target, M_ignore), both hard binary
# ---------------------------------------------------------------------------

def decompose_gt(
    gt_raw: torch.Tensor,      # [1, H, W] in {0, 1}
    kernel: int = 3,
    area_thresh: int = 5,
    density_window: int = 5,
    density_thresh: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-dimension GT decomposition.

    Returns:
        M_target  : hard binary, pixels the model SHOULD learn
        M_ignore  : hard binary, pixels excluded from loss
    """
    assert gt_raw.dim() == 3 and gt_raw.shape[0] == 1

    # 1) Morphological open
    m_open = morphological_open(gt_raw.unsqueeze(0), kernel).squeeze(0)

    # 2) Connected-components on CPU
    m_open_np = m_open.cpu().numpy()[0]
    m_cc_np = remove_small_components(m_open_np, area_thresh)
    m_cc = torch.from_numpy(m_cc_np).to(gt_raw.device).unsqueeze(0)

    # 3) Local density
    m_dense = local_density_filter(
        gt_raw.unsqueeze(0), density_window, density_thresh
    ).squeeze(0) * gt_raw

    # Target = intersection of all checks
    m_target = (m_cc * m_dense).clamp(0, 1)

    # Ignore = GT hole pixels NOT in target (isolated noise / unstructured)
    m_ignore = gt_raw * (1.0 - m_target)

    return m_target, m_ignore


# ---------------------------------------------------------------------------
# Three preset configs
# ---------------------------------------------------------------------------

PRESETS = {
    "conservative": dict(kernel=3, area_thresh=5,  density_window=5, density_thresh=3),
    "medium":       dict(kernel=3, area_thresh=10, density_window=5, density_thresh=5),
    "aggressive":   dict(kernel=5, area_thresh=20, density_window=5, density_thresh=5),
}


# ---------------------------------------------------------------------------
# Visualisation helpers (re-using style from utils.visualize)
# ---------------------------------------------------------------------------

def _to_rgb_u8(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def _to_2d(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().squeeze(0).numpy() if t.dim() == 3 else t.detach().cpu().numpy()


def _colorize_depth(d: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    if vmin is None:
        vmin = float(np.nanmin(d))
    if vmax is None:
        vmax = float(np.nanmax(d))
    n = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return (plt.get_cmap("Spectral_r")(n)[..., :3] * 255).astype(np.uint8)


def _overlay_target_ignore(rgb_u8: np.ndarray, target: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """G channel = target, B channel = ignore. RGB base."""
    out = rgb_u8.copy().astype(np.int32)
    t_mask = target > 0.5
    i_mask = ignore > 0.5
    out[t_mask] = (0, 255, 0)
    out[i_mask] = (0, 0, 255)
    return out.clip(0, 255).astype(np.uint8)


def render_sample(
    rgb_u8: np.ndarray,
    sim_d:  np.ndarray,
    real_d: np.ndarray,
    gt_raw: np.ndarray,
    decomp: dict,   # {preset_name: (target_np, ignore_np)}
    save_path: Path,
    sample_id: str,
):
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))

    # Depth color range from valid pixels
    valid = (1 - gt_raw).astype(bool)
    if valid.any():
        vmin = float(np.nanpercentile(real_d[valid], 2))
        vmax = float(np.nanpercentile(real_d[valid], 98))
    else:
        vmin, vmax = float(real_d.min()), float(real_d.max())

    # Row 1: inputs + raw GT
    axes[0, 0].imshow(rgb_u8);                      axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(_colorize_depth(sim_d, vmin, vmax)); axes[0, 1].set_title("Sim Depth")
    real_color = _colorize_depth(real_d, vmin, vmax).copy()
    real_color[gt_raw > 0.5] = (0, 0, 0)
    axes[0, 2].imshow(real_color);                  axes[0, 2].set_title("Real (GT hole)")
    axes[0, 3].imshow(gt_raw, cmap="gray");         axes[0, 3].set_title("GT_raw (binary)")

    # Row 2-4: three presets
    for row, name in enumerate(["conservative", "medium", "aggressive"], start=1):
        tgt, ign = decomp[name]
        n_tgt = int(tgt.sum()); n_ign = int(ign.sum()); n_gt = int(gt_raw.sum())
        ratio = n_ign / max(n_gt, 1)
        axes[row, 0].imshow(tgt, cmap="gray")
        axes[row, 0].set_title(f"{name} M_target  ({n_tgt})")
        axes[row, 1].imshow(ign, cmap="gray")
        axes[row, 1].set_title(f"{name} M_ignore  ({n_ign}, {ratio*100:.1f}% of GT)")
        axes[row, 2].imshow(_overlay_target_ignore(rgb_u8, tgt, ign))
        axes[row, 2].set_title(f"{name} overlay (G=target, B=ignore)")
        axes[row, 3].axis("off")

    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{sample_id}  |  GT decomposition presets", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n",      type=int, default=100)
    ap.add_argument("--out",    required=True, help="Output dir for vis + summary.csv")
    ap.add_argument("--split",  default="val", choices=["train", "val"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ByteCamDepthDataset(
        data_root=cfg["data_root"],
        split=args.split,
        resolution=cfg.get("resolution", 512),
        augment=False,
    )
    if args.n > 0:
        ds = Subset(ds, list(range(min(args.n, len(ds)))))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    summary_rows = []

    for idx, batch in enumerate(tqdm(loader, desc="decompose")):
        rgb     = batch["rgb"][0].to(device)         # [3, H, W]
        sim_d   = batch["sim_depth"][0].to(device)   # [1, H, W]
        real_d  = batch.get("real_depth", batch["sim_depth"])[0].to(device)
        gt_raw  = batch["hole_mask"][0].to(device)   # [1, H, W]

        decomp = {}
        for name, params in PRESETS.items():
            tgt, ign = decompose_gt(gt_raw, **params)
            decomp[name] = (_to_2d(tgt), _to_2d(ign))

        # Save visualisation
        save_path = out_dir / f"sample_{idx:03d}.png"
        render_sample(
            rgb_u8=_to_rgb_u8(rgb),
            sim_d=_to_2d(sim_d),
            real_d=_to_2d(real_d),
            gt_raw=_to_2d(gt_raw),
            decomp=decomp,
            save_path=save_path,
            sample_id=f"sample_{idx:03d}",
        )

        # Record stats
        gt_pixels = int(_to_2d(gt_raw).sum())
        row = {"sample": idx, "gt_pixels": gt_pixels}
        for name, (t, i) in decomp.items():
            row[f"{name}_target"] = int(t.sum())
            row[f"{name}_ignore"] = int(i.sum())
            row[f"{name}_ignore_ratio"] = i.sum() / max(gt_pixels, 1)
        summary_rows.append(row)

    # ---- Summary CSV ----
    csv_path = out_dir / "summary.csv"
    keys = ["sample", "gt_pixels",
            "conservative_target", "conservative_ignore", "conservative_ignore_ratio",
            "medium_target",       "medium_ignore",       "medium_ignore_ratio",
            "aggressive_target",   "aggressive_ignore",   "aggressive_ignore_ratio"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in summary_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    # ---- Print aggregate ----
    print("\n" + "=" * 70)
    print(f"GT 分解三档参数 — 总览 (n={len(summary_rows)} samples)")
    print("=" * 70)
    print(f"{'preset':<14s}  {'avg ignore ratio':>20s}  {'p50':>8s}  {'p95':>8s}")
    print("-" * 60)
    for name in ["conservative", "medium", "aggressive"]:
        ratios = [r[f"{name}_ignore_ratio"] for r in summary_rows if r["gt_pixels"] > 0]
        if ratios:
            ratios = np.array(ratios)
            print(f"{name:<14s}  {ratios.mean()*100:>18.1f}%  "
                  f"{np.median(ratios)*100:>6.1f}%  {np.percentile(ratios, 95)*100:>6.1f}%")
    print("=" * 70)
    print(f"\n可视化图 ({len(summary_rows)} 张) 已存到: {out_dir}")
    print(f"详细 CSV: {csv_path}")


if __name__ == "__main__":
    main()
