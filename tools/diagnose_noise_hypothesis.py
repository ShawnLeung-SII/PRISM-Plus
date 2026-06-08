"""诊断实验：验证"细小点噪声"假说

思路：
    用 v0.2.0 best.pt 在 val 集上跑推理，得到 pred_mask。
    然后用形态学开运算把 GT 分成：
        gt_det   = 结构化失效（边缘大块 + 材料空洞）= morphological_open(gt_raw)
        gt_noise = 细小随机噪声点 = gt_raw - gt_det

    分别评估：
        IoU(pred, gt_raw)      # 当前指标，应该 ~0.64
        IoU(pred, gt_det)      # 只看可学部分；如果跳到 0.75+，假说成立
        IoU(pred, gt_noise)    # 只看噪声点；应该很低（不可学）

    同时记录：
        gt_noise / gt_raw 像素占比（量化噪声比例）
        pred 在 gt_noise 区域的命中率（"过拟合噪声"程度）

Run on code server:
    python tools/diagnose_noise_hypothesis.py \\
        --ckpt /inspire/hdd/.../checkpoints/prism_plus/stage1_v2/best.pt \\
        --config configs/stage1_bnd_plus.yaml \\
        --kernel 3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.data import ByteCamDepthDataset
from prism_plus.models import create_bnd_plus


# ---------------------------------------------------------------------------
# Morphological open / close on binary masks (GPU)
# ---------------------------------------------------------------------------

def binary_erode(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    """min-pool = erosion"""
    pad = k // 2
    return -F.max_pool2d(-m.float(), k, stride=1, padding=pad)


def binary_dilate(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    """max-pool = dilation"""
    pad = k // 2
    return F.max_pool2d(m.float(), k, stride=1, padding=pad)


def morphological_open(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Open = erode then dilate. Removes isolated dots smaller than k×k."""
    return binary_dilate(binary_erode(m, k), k)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Both [B,1,H,W] binary. Returns mean IoU over batch."""
    p = pred.float()
    t = target.float()
    inter = (p * t).sum(dim=[1, 2, 3])
    union = ((p + t) > 0).float().sum(dim=[1, 2, 3])
    return ((inter / (union + eps))).mean().item()


def coverage(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Fraction of target pixels that pred also predicts (recall on target)."""
    p = pred.float(); t = target.float()
    return ((p * t).sum() / (t.sum() + eps)).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--kernel", type=int, default=3,
                    help="Morphological open kernel; bigger = more aggressive noise removal")
    ap.add_argument("--n",      type=int, default=200,
                    help="Number of val samples to evaluate (0 = all)")
    ap.add_argument("--out",    default="diagnose_result.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build model ----
    model = create_bnd_plus(
        vfm_type=cfg.get("vfm_type", "moge2"),
        vfm_checkpoint=cfg.get("vfm_checkpoint"),
        encoder_size=cfg.get("encoder_size", "large"),
        deep_supervision=False,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    # ---- Val data ----
    val_ds = ByteCamDepthDataset(
        data_root=cfg["data_root"], split="val",
        resolution=cfg.get("resolution", 512), augment=False,
    )
    if args.n > 0:
        val_ds = Subset(val_ds, range(min(args.n, len(val_ds))))
    loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                        num_workers=4, pin_memory=True)

    K = args.kernel
    # Collectors
    iou_raw_l, iou_det_l, iou_noise_l = [], [], []
    cov_det_l, cov_noise_l            = [], []
    noise_frac_l                       = []
    sample_metrics = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="diagnose"):
            rgb   = batch["rgb"].to(device)
            sim_d = batch["sim_depth"].to(device)
            gt    = batch["hole_mask"].to(device).float()

            out = model(rgb, sim_d)
            pred_p = out["pred_failure"]
            pred_b = (pred_p > 0.5).float()

            # GT decomposition
            gt_det   = morphological_open(gt, K)        # structured failure
            gt_noise = (gt - gt_det).clamp(0, 1)        # isolated dots

            iou_raw_l.append(iou(pred_b, gt))
            iou_det_l.append(iou(pred_b, gt_det))
            # IoU on noise mask is meaningless when pred not learned for it, but coverage is
            iou_noise_l.append(iou(pred_b, gt_noise))
            cov_det_l.append(coverage(pred_b, gt_det))
            cov_noise_l.append(coverage(pred_b, gt_noise))

            # Per-sample noise fraction
            for b in range(gt.shape[0]):
                gt_sum = gt[b].sum().item()
                ns = gt_noise[b].sum().item()
                if gt_sum > 0:
                    noise_frac_l.append(ns / gt_sum)
                sample_metrics.append({
                    "iou_raw":   iou(pred_b[b:b+1], gt[b:b+1]),
                    "iou_det":   iou(pred_b[b:b+1], gt_det[b:b+1]),
                    "noise_frac": (ns / gt_sum) if gt_sum > 0 else 0.0,
                })

    res = {
        "kernel": K,
        "n_samples": len(sample_metrics),
        "iou_vs_raw_gt":      float(np.mean(iou_raw_l)),
        "iou_vs_det_gt":      float(np.mean(iou_det_l)),
        "iou_vs_noise_gt":    float(np.mean(iou_noise_l)),
        "coverage_on_det":    float(np.mean(cov_det_l)),
        "coverage_on_noise":  float(np.mean(cov_noise_l)),
        "noise_fraction_mean": float(np.mean(noise_frac_l)) if noise_frac_l else 0.0,
        "noise_fraction_p50":  float(np.median(noise_frac_l)) if noise_frac_l else 0.0,
        "noise_fraction_p95":  float(np.percentile(noise_frac_l, 95)) if noise_frac_l else 0.0,
    }

    print("\n" + "="*70)
    print(f"诊断结果 (kernel={K}, n={res['n_samples']})")
    print("="*70)
    print(f"  IoU(pred, GT_raw)                     = {res['iou_vs_raw_gt']:.4f}   ← 当前训练评估的")
    print(f"  IoU(pred, GT_det = open(GT,{K}))        = {res['iou_vs_det_gt']:.4f}   ← 只看可学失效")
    print(f"  IoU(pred, GT_noise = GT - GT_det)     = {res['iou_vs_noise_gt']:.4f}   ← 噪声部分（应该很低）")
    print(f"  coverage(pred on GT_det)               = {res['coverage_on_det']:.4f}   ← 结构化失效召回")
    print(f"  coverage(pred on GT_noise)             = {res['coverage_on_noise']:.4f}   ← 模型对噪声的'记忆度'")
    print(f"")
    print(f"  GT 中噪声占比 (mean / p50 / p95)        = {res['noise_fraction_mean']:.3f} / {res['noise_fraction_p50']:.3f} / {res['noise_fraction_p95']:.3f}")
    print("="*70)
    delta = res['iou_vs_det_gt'] - res['iou_vs_raw_gt']
    print(f"\n  ⇒ 去除噪声后 IoU 提升: {delta:+.4f} (+{delta*100:.2f}pp)")
    if delta > 0.08:
        print("  ⇒ 假说强支持: 噪声地板是主要瓶颈, R1 双 head 路径预期收益大")
    elif delta > 0.03:
        print("  ⇒ 假说部分支持: 噪声贡献部分瓶颈, R1 值得尝试")
    else:
        print("  ⇒ 假说不支持: 瓶颈在别处, 需要换思路")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  详细结果已写入: {args.out}")


if __name__ == "__main__":
    main()
