"""Johansen cointegration test wrapper (Phase 4 Section 5).

Pure functions, no I/O. Order-independent multivariate cointegration via
``statsmodels.tsa.vector_ar.vecm.coint_johansen``. For a bivariate system
(a pair) the trace statistic for rank 0 vs its critical value decides whether
the pair is cointegrated; unlike Engle-Granger there is no asymmetric
"regress y on x" choice, so the result does not depend on leg ordering.

Reference: Johansen (1991). statsmodels critical-value columns are
[90%, 95%, 99%].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_LEVELS = {"90": 0, "95": 1, "99": 2}


@dataclass(frozen=True)
class JohansenResult:
    trace_stat_r0: float  # trace statistic, H0: rank = 0
    trace_crit_r0: tuple[float, float, float]  # 90/95/99 critical values
    trace_stat_r1: float  # trace statistic, H0: rank <= 1
    rank_95: int  # cointegration rank at 95%
    beta: float  # hedge ratio from the first cointegrating vector (β on x)
    eig: float  # largest eigenvalue
    fit_ok: bool
    reason: str


def johansen_pair_test(
    y: np.ndarray, x: np.ndarray, *, det_order: int = 0, k_ar_diff: int = 1
) -> JohansenResult:
    """Johansen test on the bivariate system ``[y, x]`` (log-levels).

    ``det_order=0`` ⇒ constant term in the cointegrating relation (no trend).
    Returns ``fit_ok=False`` on degenerate / too-short input.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.column_stack([y, x])
    finite = np.all(np.isfinite(m), axis=1)
    m = m[finite]
    if m.shape[0] < 50 or np.allclose(m[:, 0], m[0, 0]) or np.allclose(m[:, 1], m[0, 1]):
        return _bad("too short or degenerate")
    try:
        res = coint_johansen(m, det_order, k_ar_diff)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _bad(f"coint_johansen failed: {exc}")

    trace = res.lr1  # [r=0, r<=1]
    cvt = res.cvt  # rows: r=0, r<=1; cols: 90/95/99
    rank = 0
    for i in range(len(trace)):
        if trace[i] > cvt[i, _LEVELS["95"]]:
            rank = i + 1
        else:
            break
    # cointegrating vector (first column of evec), normalized so y-coef = 1
    evec = res.evec[:, 0]
    beta = float(-evec[1] / evec[0]) if evec[0] != 0 else float("nan")
    return JohansenResult(
        trace_stat_r0=float(trace[0]),
        trace_crit_r0=(float(cvt[0, 0]), float(cvt[0, 1]), float(cvt[0, 2])),
        trace_stat_r1=float(trace[1]),
        rank_95=int(rank),
        beta=beta,
        eig=float(res.eig[0]),
        fit_ok=True,
        reason="",
    )


def is_cointegrated(res: JohansenResult, level: str = "95") -> bool:
    """Pair is cointegrated iff rank ≥ 1 (reject H0: rank=0) at ``level``."""
    if not res.fit_ok:
        return False
    return res.trace_stat_r0 > res.trace_crit_r0[_LEVELS[level]]


def _bad(reason: str) -> JohansenResult:
    return JohansenResult(
        trace_stat_r0=float("nan"),
        trace_crit_r0=(float("nan"),) * 3,
        trace_stat_r1=float("nan"),
        rank_95=0,
        beta=float("nan"),
        eig=float("nan"),
        fit_ok=False,
        reason=reason,
    )


__all__ = ["JohansenResult", "is_cointegrated", "johansen_pair_test"]
