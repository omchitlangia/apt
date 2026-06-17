"""Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).

Pure functions, no I/O. Combinatorially-Symmetric Cross-Validation: split the
per-period return matrix (T observations × N candidate configurations) into
``S`` disjoint time blocks; for every way of choosing ``S/2`` blocks as
in-sample (IS), the complement is out-of-sample (OOS). Pick the IS-best
configuration, look up its OOS relative rank ``ω`` and logit
``λ = ln(ω/(1-ω))``. PBO is the fraction of splits whose IS-best lands below
the OOS median (``λ ≤ 0``) — i.e. the probability the selected strategy is no
better than the median out-of-sample.

Reference: Bailey, Borwein, López de Prado, Zhu (2017), "The Probability of
Backtest Overfitting," *Journal of Computational Finance* 20(4), 39-69.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    n_splits_blocks: int
    n_combinations: int
    n_strategies: int
    frac_is_best_also_oos_best: float


def _block_moments(matrix: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block sum, sumsq, count over the first axis split into ``s`` blocks."""
    t, n = matrix.shape
    edges = np.linspace(0, t, s + 1).astype(int)
    bsum = np.empty((s, n))
    bsq = np.empty((s, n))
    bcount = np.empty(s, dtype=int)
    for b in range(s):
        seg = matrix[edges[b] : edges[b + 1]]
        bsum[b] = seg.sum(axis=0)
        bsq[b] = (seg * seg).sum(axis=0)
        bcount[b] = seg.shape[0]
    return bsum, bsq, bcount


def _sharpe(sum_, sq_, count_) -> np.ndarray:
    mean = sum_ / count_
    var = sq_ / count_ - mean * mean
    var = np.where(var <= 0, np.nan, var)
    return mean / np.sqrt(var)


def pbo_cscv(matrix: np.ndarray, *, n_blocks: int = 16) -> PBOResult:
    """Probability of backtest overfitting via CSCV.

    Parameters
    ----------
    matrix
        ``(T, N)`` per-period returns: T observations, N candidate configs.
    n_blocks
        Even number of disjoint time blocks ``S`` (default 16 ⇒ C(16,8)=12870
        symmetric splits).
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D (T, N)")
    t, n = m.shape
    if n < 2:
        raise ValueError("need >= 2 candidate configurations")
    if n_blocks % 2 != 0 or n_blocks < 2:
        raise ValueError("n_blocks must be even and >= 2")
    if t < n_blocks:
        raise ValueError("need at least n_blocks observations")

    bsum, bsq, bcount = _block_moments(m, n_blocks)
    all_blocks = set(range(n_blocks))
    logits: list[float] = []
    is_best_oos_best = 0
    for is_blocks in combinations(range(n_blocks), n_blocks // 2):
        is_idx = list(is_blocks)
        oos_idx = list(all_blocks - set(is_blocks))
        is_sharpe = _sharpe(bsum[is_idx].sum(0), bsq[is_idx].sum(0), bcount[is_idx].sum())
        oos_sharpe = _sharpe(bsum[oos_idx].sum(0), bsq[oos_idx].sum(0), bcount[oos_idx].sum())
        if np.all(np.isnan(is_sharpe)):
            continue
        n_star = int(np.nanargmax(is_sharpe))
        # OOS relative rank of the IS-best (1 = worst ... N = best)
        ranks = rankdata(np.nan_to_num(oos_sharpe, nan=-np.inf), method="average")
        omega = ranks[n_star] / (n + 1.0)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))
        if n_star == int(np.nanargmax(oos_sharpe)):
            is_best_oos_best += 1

    lam = np.asarray(logits)
    pbo = float(np.mean(lam <= 0.0)) if lam.size else float("nan")
    return PBOResult(
        pbo=pbo,
        logits=lam,
        n_splits_blocks=n_blocks,
        n_combinations=int(lam.size),
        n_strategies=n,
        frac_is_best_also_oos_best=float(is_best_oos_best / lam.size) if lam.size else float("nan"),
    )


__all__ = ["PBOResult", "pbo_cscv"]
