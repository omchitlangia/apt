"""Rolling cointegration-stability gate (Phase 4 Section 4).

Pure functions, no I/O. The spread of a frozen-(α, β) pair IS the
Engle-Granger residual (β, α were fit by EG on train), so a rolling ADF test
on the spread level directly tracks whether the cointegrating relationship
still holds out-of-sample. When the spread's ADF p-value degrades past a
threshold for several consecutive windows, the pair is **blacklisted** (no new
entries) from that point.

References: Engle & Granger (1987); Augmented Dickey-Fuller via statsmodels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller

# Labeled defaults (→ ASSUMPTIONS A6).
ADF_PVALUE_THRESHOLD: float = 0.10
CONSECUTIVE_WINDOWS_TO_GATE: int = 3
ROLLING_WINDOW: int = 60
ROLLING_STEP: int = 5


@dataclass(frozen=True)
class RollingADF:
    end_idx: np.ndarray  # index (into the input series) of each window's last obs
    pvalues: np.ndarray  # ADF p-value per window (nan if window degenerate)
    n_windows: int


def rolling_adf_pvalues(
    spread: np.ndarray,
    *,
    window: int = ROLLING_WINDOW,
    step: int = ROLLING_STEP,
    min_finite: int = 30,
) -> RollingADF:
    """ADF p-value of the spread level over a rolling trailing window."""
    s = np.asarray(spread, dtype=float)
    n = s.size
    ends, pvals = [], []
    if n >= window:
        for end in range(window - 1, n, step):
            seg = s[end - window + 1 : end + 1]
            seg = seg[np.isfinite(seg)]
            ends.append(end)
            if seg.size < min_finite or np.allclose(seg, seg[0]):
                pvals.append(np.nan)
                continue
            try:
                pvals.append(float(adfuller(seg, autolag="AIC")[1]))
            except (ValueError, np.linalg.LinAlgError):
                pvals.append(np.nan)
    return RollingADF(np.asarray(ends, dtype=int), np.asarray(pvals, dtype=float), len(ends))


def gate_index(
    roll: RollingADF,
    *,
    threshold: float = ADF_PVALUE_THRESHOLD,
    consecutive: int = CONSECUTIVE_WINDOWS_TO_GATE,
) -> int | None:
    """First window-end index after ``consecutive`` consecutive windows with
    ADF p-value > ``threshold`` (degraded cointegration). ``None`` if never."""
    run = 0
    for p, end in zip(roll.pvalues, roll.end_idx, strict=True):
        if not np.isfinite(p):
            run = 0
            continue
        if p > threshold:
            run += 1
            if run >= consecutive:
                return int(end)
        else:
            run = 0
    return None


def gate_summary(
    roll: RollingADF,
    *,
    threshold: float = ADF_PVALUE_THRESHOLD,
    consecutive: int = CONSECUTIVE_WINDOWS_TO_GATE,
) -> dict:
    finite = roll.pvalues[np.isfinite(roll.pvalues)]
    gi = gate_index(roll, threshold=threshold, consecutive=consecutive)
    return {
        "n_windows": roll.n_windows,
        "n_windows_degraded": int(np.sum(finite > threshold)),
        "frac_degraded": float(np.mean(finite > threshold)) if finite.size else float("nan"),
        "mean_pvalue": float(np.mean(finite)) if finite.size else float("nan"),
        "max_pvalue": float(np.max(finite)) if finite.size else float("nan"),
        "gated": gi is not None,
        "gate_end_idx": gi if gi is not None else -1,
    }


__all__ = [
    "ADF_PVALUE_THRESHOLD",
    "CONSECUTIVE_WINDOWS_TO_GATE",
    "ROLLING_STEP",
    "ROLLING_WINDOW",
    "RollingADF",
    "gate_index",
    "gate_summary",
    "rolling_adf_pvalues",
]
