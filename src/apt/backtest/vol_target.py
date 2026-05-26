"""Portfolio-level vol-target overlay (Phase 2B post-processing).

Equalises rungs onto a common risk axis so net return, drawdown and 2018-19
P&L become directly comparable. The ladder runs at very different effective
deployments (e.g. R3 at ~1.4% ann vol vs R0 at ~15%); without this overlay
only Sharpe is comparable across rungs.

Mechanism: for each day ``t``, set leverage
    L_t = min(target_vol / realized_vol_t, max_leverage)
where ``realized_vol_t`` is the annualised std of the trailing ``window`` days
of net returns *up to but not including ``t``* (strictly causal). Until
``min_periods`` of prior history are available, leverage = 1.0. The output
return at ``t`` is ``L_t · r_t`` (small-return log approximation).

The overlay is applied at the portfolio level — knowing nothing about
positions, pairs or clusters. It is a multiplicative gain on the daily
P&L series, exactly as an external vol-targeting wrapper would do.
"""

from __future__ import annotations

import math

import numpy as np


def apply_vol_target_overlay(
    daily_log_returns: np.ndarray,
    *,
    target_vol_annual: float = 0.10,
    window: int = 60,
    max_leverage: float = 3.0,
    min_periods: int = 20,
    periods_per_year: int = 252,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a causal portfolio-level vol-target overlay.

    Parameters
    ----------
    daily_log_returns
        1-D array of portfolio daily log returns (raw, pre-overlay).
    target_vol_annual
        Target annualised volatility (e.g. ``0.10`` for 10%).
    window
        Lookback in trading days for the trailing realized-vol estimate.
    max_leverage
        Hard cap on per-day leverage (safety brake when realized vol is tiny).
    min_periods
        Minimum prior-day count before any scaling kicks in (leverage = 1.0
        until reached). Prevents tiny samples from driving wild leverage.
    periods_per_year
        Annualisation factor (252 for daily equities).

    Returns
    -------
    (overlaid_returns, leverage)
        Both shape ``(n,)``. ``overlaid_returns[t] = leverage[t] * daily_log_returns[t]``.
    """
    if target_vol_annual <= 0:
        raise ValueError(f"target_vol_annual must be > 0, got {target_vol_annual}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if max_leverage <= 0:
        raise ValueError(f"max_leverage must be > 0, got {max_leverage}")
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")

    arr = np.asarray(daily_log_returns, dtype=float)
    n = arr.size
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    leverage = np.ones(n, dtype=float)
    annualizer = math.sqrt(periods_per_year)

    for t in range(n):
        # Strictly causal: at day t, use returns[t-window:t] (excludes t).
        prior_start = max(0, t - window)
        prior = arr[prior_start:t]
        if prior.size < min_periods:
            continue  # leverage stays at 1.0
        sd = float(np.std(prior, ddof=1))
        realized_vol = sd * annualizer
        if realized_vol > 1e-12:
            leverage[t] = min(target_vol_annual / realized_vol, max_leverage)
        else:
            leverage[t] = max_leverage

    overlaid = leverage * arr
    return overlaid, leverage
