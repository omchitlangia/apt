"""Session-aware rolling z-score + time-of-day volatility adjustment.

Three z-score flavors are provided:

* **session-local z** — :func:`sessionized_rolling_zscore`. The rolling
  mean and std are computed inside ONE session only — the window resets
  at every session boundary. Useful only when the pair's half-life fits
  inside a single session.

* **multi-session chronological z** — :func:`intraday_rolling_zscore`.
  The rolling window spans up to a few sessions (level data only, so
  there is no overnight-gap RETURN entering the window — the level just
  jumps once and is treated as one sample, the same way the previous
  bar's close→close transition is one increment). A small **intraday
  warm-up** (the first ``session_warmup_bars`` of each session) is
  suppressed to NaN regardless of global window state, killing signals
  fired during the noisy open. This is the function the Phase 3 engine
  uses by default because most cointegrated pairs have multi-day
  half-lives.

* **TOD-adjusted z** — :func:`tod_adjusted_zscore`. Deflates the
  spread-minus-rollmean innovation by the typical volatility at that
  minute-of-session, learned from a training window. Compensates for the
  intraday vol U-curve (loud open, quiet midday, lively close), so an
  identical move at 09:15 and 12:00 doesn't carry an identical z.

All three are CAUSAL: value at index ``t`` depends only on bars at or
before ``t`` (and, for the TOD profile, on a strictly separate training
window).

Overnight-gap handling
----------------------
The rolling stats here are over LEVELS, not returns. A multi-session
window therefore does NOT "blend" the overnight return into the moving
stats — it simply contains both sessions' levels. The overnight gap
RETURN is excluded from the P&L by :mod:`apt.intraday.backtest`
(Regime A) or realized as a single one-bar PnL at the next session's
open (Regime B). Together the two layers honor the task's
"no-overnight-blending" requirement without forcing a session-local
window that would never fill for slow-mean-reverting pairs.
"""

from __future__ import annotations

import numpy as np

from apt.intraday.calendar import NSE_BARS_PER_SESSION, session_segments
from apt.signals.spread import rolling_zscore


def intraday_rolling_zscore(
    spread: np.ndarray,
    session_id: np.ndarray,
    *,
    window: int,
    min_periods: int | None = None,
    session_warmup_bars: int = 0,
) -> np.ndarray:
    """Chronological multi-session rolling z + per-session warm-up suppression.

    Computes a CAUSAL trailing rolling z over the full bar sequence (the
    window may span multiple sessions). Then suppresses the first
    ``session_warmup_bars`` bars of each session to NaN regardless of
    global window state — these are the most-volatile minutes after the
    open auction and signals there are unreliable.

    The rolling stats are over level data (the log spread), so spanning
    sessions does NOT inject the overnight RETURN into the moving mean.
    The overnight gap is handled at the P&L layer (Regime A excludes it;
    Regime B realizes it as a single one-bar return at the next session's
    first bar) — see :mod:`apt.intraday.backtest`.

    Parameters
    ----------
    spread
        1-D log-spread array, length N.
    session_id
        1-D int array of length N (dense rank of each bar's session).
    window
        Trailing window in MINUTES.
    min_periods
        Minimum observations before z is non-NaN (default = window).
    session_warmup_bars
        Bars at the start of each session whose z is forced to NaN. 0
        disables (then this function is identical to a plain rolling z).
    """
    s = np.asarray(spread, dtype=float)
    sids = np.asarray(session_id)
    if s.shape != sids.shape:
        raise ValueError(f"shape mismatch: spread={s.shape} session_id={sids.shape}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    z = rolling_zscore(s, window=window, min_periods=min_periods or window)
    if session_warmup_bars > 0:
        for a, b in session_segments(sids):
            k = min(session_warmup_bars, b - a)
            z[a : a + k] = np.nan
    return z


def sessionized_rolling_zscore(
    spread: np.ndarray,
    session_id: np.ndarray,
    *,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    """Rolling z-score that never blends across the overnight gap.

    The trailing window at bar ``t`` is restricted to the current session.
    Bars before the in-session window has filled (``min_periods``) return NaN.

    Parameters
    ----------
    spread
        1-D log-spread array, length N. NaN bars are preserved (the window
        treats them as missing — same semantics as :func:`apt.signals.spread.rolling_zscore`).
    session_id
        1-D int array of length N assigning each bar to a session (e.g.,
        dense rank of the bar's date).
    window
        Rolling window in minutes. Reset at each session start.
    min_periods
        Minimum in-session observations before z is non-NaN (default = window).
    """
    s = np.asarray(spread, dtype=float)
    sids = np.asarray(session_id)
    if s.shape != sids.shape:
        raise ValueError(f"shape mismatch: spread={s.shape} session_id={sids.shape}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    out = np.full(s.size, np.nan, dtype=float)
    for a, b in session_segments(sids):
        z = rolling_zscore(s[a:b], window=window, min_periods=min_periods or window)
        out[a:b] = z
    return out


def _chronological_rolling_mean(
    spread: np.ndarray,
    *,
    window: int,
    min_periods: int | None = None,
) -> np.ndarray:
    """Plain chronological trailing rolling mean (helper for TOD-adjusted z).

    The window may span sessions — see the module-level "Overnight-gap
    handling" note. Used by :func:`fit_tod_vol_profile` and
    :func:`tod_adjusted_zscore` for consistency with
    :func:`intraday_rolling_zscore`.
    """
    import polars as pl

    s = np.asarray(spread, dtype=float)
    mp = min_periods or window
    ps = pl.Series("s", s)
    return ps.rolling_mean(window_size=window, min_samples=mp).to_numpy()


def _chronological_rolling_std(
    spread: np.ndarray,
    *,
    window: int,
    min_periods: int | None = None,
    ddof: int = 1,
) -> np.ndarray:
    """Plain chronological trailing rolling std."""
    import polars as pl

    s = np.asarray(spread, dtype=float)
    mp = min_periods or window
    ps = pl.Series("s", s)
    return ps.rolling_std(window_size=window, min_samples=mp, ddof=ddof).to_numpy()


def fit_tod_vol_profile(
    spread_train: np.ndarray,
    session_id_train: np.ndarray,
    bar_in_session_train: np.ndarray,
    *,
    window: int,
    min_periods: int | None = None,
    smooth_radius: int = 2,
) -> np.ndarray:
    """Fit a per-minute-of-session volatility profile on training data.

    For each minute-of-session ``m in [0, 375)``, computes the std of
    ``spread - sessionized_rollmean`` restricted to bars at that minute.
    The result is then smoothed with a centered moving average of half-
    width ``smooth_radius`` to damp single-minute noise from low-sample
    minutes.

    Parameters
    ----------
    spread_train, session_id_train, bar_in_session_train
        Same-length 1-D arrays describing the TRAIN window only.
    window, min_periods
        Same parameters as the rolling z (to keep the innovation
        distribution consistent across train and test).
    smooth_radius
        Half-width of the boxcar smoother across adjacent minutes
        (``smooth_radius=2`` → 5-minute boxcar). 0 disables smoothing.

    Returns
    -------
    sigma_tod : 1-D float array of length 375. NaN at minutes with no
        finite observations.
    """
    s = np.asarray(spread_train, dtype=float)
    sids = np.asarray(session_id_train)
    bins = np.asarray(bar_in_session_train, dtype=np.int32)
    if s.shape != sids.shape or s.shape != bins.shape:
        raise ValueError("shape mismatch among train arrays")
    _ = sids  # session ids are used only by callers; kept for API symmetry
    rollmean = _chronological_rolling_mean(s, window=window, min_periods=min_periods or window)
    innovation = s - rollmean

    sigma = np.full(NSE_BARS_PER_SESSION, np.nan, dtype=float)
    finite = np.isfinite(innovation)
    for m in range(NSE_BARS_PER_SESSION):
        mask = (bins == m) & finite
        if mask.sum() >= 2:
            sigma[m] = float(innovation[mask].std(ddof=1))

    if smooth_radius > 0:
        # Boxcar smoothing over adjacent minutes, NaN-aware
        out = np.full(NSE_BARS_PER_SESSION, np.nan, dtype=float)
        for m in range(NSE_BARS_PER_SESSION):
            lo = max(0, m - smooth_radius)
            hi = min(NSE_BARS_PER_SESSION, m + smooth_radius + 1)
            vals = sigma[lo:hi]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[m] = float(vals.mean())
        sigma = out

    return sigma


def tod_adjusted_zscore(
    spread: np.ndarray,
    session_id: np.ndarray,
    bar_in_session: np.ndarray,
    sigma_tod: np.ndarray,
    *,
    window: int,
    min_periods: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """TOD-adjusted z: innovation deflated by the minute-of-session vol.

    Defined as::

        z_tod[t] = (spread[t] - sessionized_rollmean[t]) / sigma_tod[m(t)]

    where ``m(t)`` is the bar's minute-of-session and ``sigma_tod`` comes
    from :func:`fit_tod_vol_profile` on the TRAIN window. Where the
    training profile is NaN (a minute with no fitting samples), the
    sessionized rolling std at ``t`` is used as a fallback so the value is
    never NaN purely for profile reasons.

    Parameters
    ----------
    spread, session_id, bar_in_session
        Length-N 1-D arrays for the TEST (or full) window.
    sigma_tod
        Per-minute-of-session volatility profile, length 375.
    window, min_periods
        Same as the flat sessionized z; required to compute the rolling
        mean (and the fallback std).
    eps
        Tiny floor on the divisor to keep z finite when sigma_tod is zero.
    """
    s = np.asarray(spread, dtype=float)
    sids = np.asarray(session_id)
    bins = np.asarray(bar_in_session, dtype=np.int32)
    if s.shape != sids.shape or s.shape != bins.shape:
        raise ValueError("shape mismatch")
    if sigma_tod.shape != (NSE_BARS_PER_SESSION,):
        raise ValueError(f"sigma_tod must have length {NSE_BARS_PER_SESSION}")
    mp = min_periods or window
    rollmean = _chronological_rolling_mean(s, window=window, min_periods=mp)
    rollstd = _chronological_rolling_std(s, window=window, min_periods=mp)
    innovation = s - rollmean
    # Pick divisor: prefer the training TOD profile; fall back to rollstd
    divisor = sigma_tod[bins]
    fallback_needed = ~np.isfinite(divisor) | (divisor <= eps)
    divisor = np.where(fallback_needed, rollstd, divisor)

    out = np.full(s.size, np.nan, dtype=float)
    valid = np.isfinite(innovation) & np.isfinite(divisor) & (divisor > eps)
    out[valid] = innovation[valid] / divisor[valid]
    return out


__all__ = [
    "sessionized_rolling_zscore",
    "fit_tod_vol_profile",
    "tod_adjusted_zscore",
]
