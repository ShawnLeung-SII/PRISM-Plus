"""PRISM+ C5 — Proposition 1 partial-correlation validation.

Proposition 1 (Bounded Bimodal Identifiability):
    I(M; R | z_sem, D_sim) <= delta(I_min),
    with delta monotonically decreasing in I_min.

Empirical surrogate:
    - rho_raw  = Pearson(M, R)               (should be > 0.3)
    - rho_part = partial Pearson(M, R | z_sem)  (should be < 0.1 if Prop. 1 holds)

We compute these via per-sample averaging:
    For each frame:
        m  = mean(BND density)      [scalar per frame]
        r  = mean(|NRG residual|)   [scalar per frame]
        z  = GAP(VFM patch tokens)  [1024-dim per frame]

    rho_raw   = Pearson(m_i, r_i) over all i
    rho_part  = Pearson(resid(m | z),  resid(r | z))
                where resid(x | z) = x - z @ beta_x   (least-squares regression)

Comparison: PRISM (GAP-SPR) vs PRISM+ (Spatial-SPR) — PRISM+ should give
strictly lower rho_part (better disentanglement).
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np


def _ridge_residual(y: np.ndarray, X: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Return y - X @ beta where beta solves min ||X beta - y||^2 + lam ||beta||^2."""
    # X: [n, d], y: [n]
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    return y - X @ beta


def compute_partial_correlation(
    M: np.ndarray, R: np.ndarray, Z: Optional[np.ndarray] = None,
    ridge_lam: float = 1e-2,
) -> Dict[str, float]:
    """
    Args:
        M : [N] per-sample mean BND density
        R : [N] per-sample mean residual magnitude
        Z : [N, d] per-sample VFM semantic embeddings (None -> raw correlation only)

    Returns dict with 'rho_raw', and if Z given 'rho_partial', 'n', etc.
    """
    M = M.astype(np.float64); R = R.astype(np.float64)
    if M.std() < 1e-9 or R.std() < 1e-9:
        return {'rho_raw': float('nan'), 'n': len(M)}
    rho_raw = float(np.corrcoef(M, R)[0, 1])
    out = {'rho_raw': rho_raw, 'n': int(len(M))}
    if Z is not None:
        Z = Z.astype(np.float64)
        # Center
        Zc = Z - Z.mean(axis=0, keepdims=True)
        Mc = M - M.mean()
        Rc = R - R.mean()
        Mres = _ridge_residual(Mc, Zc, ridge_lam)
        Rres = _ridge_residual(Rc, Zc, ridge_lam)
        if Mres.std() < 1e-9 or Rres.std() < 1e-9:
            out['rho_partial'] = float('nan')
        else:
            out['rho_partial'] = float(np.corrcoef(Mres, Rres)[0, 1])
        out['delta_drop'] = abs(rho_raw) - abs(out['rho_partial'])
    return out


def evaluate_proposition1(
    M: np.ndarray, R: np.ndarray, Z: np.ndarray,
    bootstrap_n: int = 200, seed: int = 42,
) -> Dict[str, float]:
    """Compute partial correlation + bootstrap CI."""
    base = compute_partial_correlation(M, R, Z)
    rng = np.random.RandomState(seed)
    rps, rrs = [], []
    n = len(M)
    for _ in range(bootstrap_n):
        idx = rng.choice(n, n, replace=True)
        r = compute_partial_correlation(M[idx], R[idx], Z[idx])
        if not np.isnan(r.get('rho_raw', np.nan)):     rps.append(r['rho_raw'])
        if not np.isnan(r.get('rho_partial', np.nan)): rrs.append(r['rho_partial'])
    if rps:
        base['rho_raw_ci95']     = (float(np.percentile(rps, 2.5)),  float(np.percentile(rps, 97.5)))
    if rrs:
        base['rho_partial_ci95'] = (float(np.percentile(rrs, 2.5)),  float(np.percentile(rrs, 97.5)))
    base['bootstrap_n'] = bootstrap_n
    return base
