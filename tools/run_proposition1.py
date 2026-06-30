#!/usr/bin/env python3
"""PRISM+ C5 — Proposition 1 validation runner.

Procedure:
    1. Sample N=1000 frames from ByteCameraDepth val
    2. For each frame collect:
         m   = mean(BND density) using PRISMPlusBND v0.6.1
         r   = mean(|NRG residual|) using NRGStandalone best.pt
         z   = GAP(VFM patch tokens) from the same MoGe2 used by both
    3. Report:
         rho_raw         = Pearson(m, r) — expected > 0.3 (both driven by material type)
         rho_partial     = Pearson(resid(m | z), resid(r | z)) — expected < 0.1
         bootstrap 95% CI on both
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prism_plus.analysis import evaluate_proposition1
from prism_plus.data import ByteCamDepthDataset
from prism_plus.models import create_bnd_plus, NRGStandalone


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bnd_ckpt', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/checkpoints/prism_plus/stage1_v6/best.pt')
    p.add_argument('--nrg_ckpt', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/checkpoints/prism_plus/stage2a_nrg/best.pt')
    p.add_argument('--vfm_checkpoint', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/weights/moge2.pt')
    p.add_argument('--sd_checkpoint', default='/inspire/ssd/project/robot-dna/liangxiujian-253308390319/latpixdepth/latpixdepth/checkpoints/Latent_SD.ckpt')
    p.add_argument('--data_root', default='/inspire/hdd/project/robot-dna/liangxiujian-253308390319/ByteCameraDepth')
    p.add_argument('--n_samples', type=int, default=1000)
    p.add_argument('--num_steps', type=int, default=20)
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='/inspire/hdd/global_user/liangxiujian-253308390319/0-XIUJIANLIANG/checkpoints/prism_plus/proposition1_results.json')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[1/3] Loading BND v0.6.1 + NRGStandalone')
    bnd = create_bnd_plus(
        vfm_type='moge2', vfm_checkpoint=args.vfm_checkpoint,
        encoder_size='large', deep_supervision=True,
        vfm_cross_attn_dim=128, boundary_band_radius=3,
    ).to(device).eval()
    sd = torch.load(args.bnd_ckpt, map_location=device, weights_only=False)
    bnd.load_state_dict(sd.get('model', sd), strict=False)
    for q in bnd.parameters(): q.requires_grad = False

    nrg = NRGStandalone(
        sd_checkpoint=args.sd_checkpoint, residual_scale=2.0,
        freeze_vae=True, freeze_unet=False, freeze_text=True,
    ).to(device).eval()
    sd = torch.load(args.nrg_ckpt, map_location=device, weights_only=False)
    nrg.load_state_dict(sd.get('model', sd), strict=False)

    print(f'[2/3] Sampling {args.n_samples} frames from val')
    ds = ByteCamDepthDataset(data_root=args.data_root, split='val',
                              resolution=args.resolution, augment=False)
    rng = np.random.RandomState(args.seed)
    idx = sorted(rng.choice(len(ds), min(args.n_samples, len(ds)), replace=False).tolist())
    loader = DataLoader(Subset(ds, idx), batch_size=4, shuffle=False, num_workers=2)

    print(f'[3/3] Computing per-frame (m, r, z)')
    M, R, Z = [], [], []
    for batch in tqdm(loader):
        rgb   = batch['rgb'].to(device)
        sim_d = batch['sim_depth'].to(device)
        with torch.no_grad():
            out  = bnd(rgb, sim_d)
            dens = out['pred_failure'].float()
            d_pred = nrg.sample(sim_depth=sim_d, mask_density=dens, num_steps=args.num_steps)
            # VFM features via BND's semantic_context
            vfm_feats = bnd.semantic_context.vfm(rgb)
            tok = None
            if isinstance(vfm_feats, dict):
                for k in ('x_norm_patchtokens', 'last_hidden_state'):
                    if k in vfm_feats:
                        tok = vfm_feats[k]; break
            if tok is None:
                tok = vfm_feats
            if tok.dim() == 3: z = tok.mean(dim=1)
            elif tok.dim() == 4: z = tok.mean(dim=[2,3])
            else: z = tok
        # per-sample scalars
        m_b = dens.mean(dim=[1,2,3]).cpu().numpy()
        r_b = torch.abs(d_pred - sim_d).mean(dim=[1,2,3]).cpu().numpy()
        z_b = z.cpu().numpy()
        M.extend(m_b.tolist()); R.extend(r_b.tolist()); Z.append(z_b)
    M = np.array(M); R = np.array(R); Z = np.concatenate(Z, axis=0)
    print(f'shapes: M={M.shape} R={R.shape} Z={Z.shape}')

    res = evaluate_proposition1(M, R, Z, bootstrap_n=200, seed=args.seed)
    print()
    print('=== Proposition 1 Validation ===')
    print(f"  rho(M, R)        = {res['rho_raw']:.4f}  95%CI {res.get('rho_raw_ci95')}")
    print(f"  rho(M, R | zsem) = {res['rho_partial']:.4f}  95%CI {res.get('rho_partial_ci95')}")
    print(f"  abs drop         = {res['delta_drop']:.4f}")
    print(f"  n samples        = {res['n']}")
    print()
    print(f"  Expected:  rho_raw > 0.3,  rho_partial < 0.1  (Prop. 1 holds)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, 'w'), indent=2)
    print(f'-> {args.output}')


if __name__ == '__main__':
    main()
