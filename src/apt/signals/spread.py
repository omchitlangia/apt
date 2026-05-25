"""Causal spread + rolling z-score + signal layer (asset-agnostic, lean).

Phase 1 Day 7. Turns a cointegrated pair into entry/exit/stop signals. No
backtester, no P&L — that lives in Phase 2.

**Asset-agnostic**: functions take generic positive price series + a hedge
ratio. No equity-specific assumptions; the same engine ports straight to
US futures, FX, or any asset class where log-price levels make sense.

**Causal**: every function uses trailing windows only.
  * :func:`rolling_zscore` at index ``t`` depends only on
    ``spread[max(0, t-window+1) : t+1]``.
  * :func:`generate_signals` at index ``t`` depends only on ``z[0..t]``.
  These are CI-tested with explicit lookahead-detection tests.

**Phase-2 contract**: :func:`compute_spread` applies a hedge ratio that
must be FIT IN A SEPARATE TRAINING WINDOW (typically by
:func:`apt.signals.cointegration.cointegrate_pairs`) and carried forward.
This module never re-fits anything on the series it's applied to.

Public API:
  * :func:`compute_spread`
  * :func:`rolling_zscore`
  * :func:`generate_signals`     → :class:`SignalSeries`
  * :func:`signal_diagnostics`   → summary dict
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# compute_spread
# ---------------------------------------------------------------------------


def compute_spread(
    p1: Sequence[float] | np.ndarray,
    p2: Sequence[float] | np.ndarray,
    *,
    beta: float,
    intercept: float = 0.0,
) -> np.ndarray:
    """Compute the log spread ``log(p1) - beta·log(p2) - intercept``.

    ``p1`` and ``p2`` are any pair of positive price series aligned on a
    common time axis. Asset-agnostic — no equity-specific assumptions.

    ``beta`` and ``intercept`` are the OLS hedge ratio + intercept from a
    SEPARATE training window (Phase-2 contract). This function NEVER re-fits
    anything on the inputs — it just applies the carried-forward parameters.
    """
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: p1={a.shape} p2={b.shape}")
    if a.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got p1.ndim={a.ndim}")
    if a.size == 0:
        return np.empty(0, dtype=float)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("prices must be strictly positive (log undefined)")
    return np.log(a) - beta * np.log(b) - intercept


# ---------------------------------------------------------------------------
# rolling_zscore
# ---------------------------------------------------------------------------


def rolling_zscore(
    spread: Sequence[float] | np.ndarray,
    *,
    window: int = 60,
    min_periods: int | None = None,
    ddof: int = 1,
) -> np.ndarray:
    """Trailing rolling z-score: ``(spread - rollmean) / rollstd``.

    CAUSAL — value at index ``t`` depends only on
    ``spread[max(0, t-window+1) : t+1]``. The first ``min_periods - 1``
    indices are NaN. Indices where the trailing window's std is 0 (constant
    spread in-window) are also NaN — never ±inf.

    ``ddof=1`` (sample std) by default.
    """
    s = np.asarray(spread, dtype=float)
    n = s.size
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if min_periods is None:
        min_periods = window
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")
    if min_periods > window:
        raise ValueError(f"min_periods ({min_periods}) > window ({window})")
    if n == 0:
        return np.empty(0, dtype=float)

    ps = pl.Series("s", s)
    roll_mean = ps.rolling_mean(window_size=window, min_samples=min_periods).to_numpy()
    roll_std = ps.rolling_std(window_size=window, min_samples=min_periods, ddof=ddof).to_numpy()

    z = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(roll_mean) & np.isfinite(roll_std) & (roll_std > 0)
    z[valid] = (s[valid] - roll_mean[valid]) / roll_std[valid]
    return z


# ---------------------------------------------------------------------------
# generate_signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalSeries:
    """Output of :func:`generate_signals`. Length-``n`` aligned by index.

    Attributes
    ----------
    position : np.ndarray[int8]
        In ``{-1, 0, +1}``. ``+1`` = long the spread (entered when z was
        deeply negative), ``-1`` = short the spread, ``0`` = flat.
    days_in_trade : np.ndarray[int32]
        Trading-day count since the most recent entry. ``0`` while flat;
        the entry day counts as ``1``. Frozen across NaN-z bars.
    exit_reason : list[str | None]
        ``None`` on all bars except the bar an exit occurs. Values:
        ``'mean_revert'``, ``'stop'``, ``'time'``.
    """

    position: np.ndarray
    days_in_trade: np.ndarray
    exit_reason: list


def generate_signals(
    z: Sequence[float] | np.ndarray,
    *,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 3.5,
    max_holding: int = 60,
) -> SignalSeries:
    """Generate ``{-1, 0, +1}`` positions from a z-score series.

    State machine (each step decides based on ``z[t]`` only — causal):

    * **flat → long**       when ``z < -entry``  (spread deep below mean)
    * **flat → short**      when ``z > +entry``  (spread deep above mean)
    * **long → flat**       when ``z >= -exit``  (mean-revert) ; reason ``'mean_revert'``
                          or ``z <= -stop``     (diverging)   ; reason ``'stop'``
                          or ``days_in_trade >= max_holding`` ; reason ``'time'``
    * **short → flat**      mirror of long.

    Conventions
    -----------
    * Exit-then-wait: on a bar that closes a position, the new state is flat;
      any re-entry happens on a subsequent bar at the earliest (no same-bar
      flip from long to short).
    * NaN-z bars (warm-up, degenerate-std windows): state is CARRIED
      FORWARD unchanged — ``days_in_trade`` does NOT increment across NaN
      gaps, so the time stop doesn't fire through a data blackout.
    """
    arr = np.asarray(z, dtype=float)
    n = arr.size
    if entry <= 0 or exit <= 0 or stop <= 0:
        raise ValueError("entry, exit, stop must all be positive")
    if exit >= entry:
        raise ValueError(f"exit ({exit}) must be < entry ({entry})")
    if stop <= entry:
        raise ValueError(f"stop ({stop}) must be > entry ({entry})")
    if max_holding < 1:
        raise ValueError(f"max_holding must be >= 1, got {max_holding}")

    position = np.zeros(n, dtype=np.int8)
    days_in_trade = np.zeros(n, dtype=np.int32)
    exit_reason: list = [None] * n

    pos = 0
    held = 0
    for t in range(n):
        zt = arr[t]
        if not np.isfinite(zt):
            # No-info bar: carry state forward; don't increment held
            position[t] = pos
            days_in_trade[t] = held
            continue

        if pos == 0:
            if zt > entry:
                pos = -1
                held = 1
            elif zt < -entry:
                pos = +1
                held = 1
        elif pos == +1:
            if zt <= -stop:
                pos = 0
                held = 0
                exit_reason[t] = "stop"
            elif zt >= -exit:
                pos = 0
                held = 0
                exit_reason[t] = "mean_revert"
            elif held >= max_holding:
                pos = 0
                held = 0
                exit_reason[t] = "time"
            else:
                held += 1
        else:  # pos == -1
            if zt >= stop:
                pos = 0
                held = 0
                exit_reason[t] = "stop"
            elif zt <= exit:
                pos = 0
                held = 0
                exit_reason[t] = "mean_revert"
            elif held >= max_holding:
                pos = 0
                held = 0
                exit_reason[t] = "time"
            else:
                held += 1

        position[t] = pos
        days_in_trade[t] = held

    return SignalSeries(
        position=position,
        days_in_trade=days_in_trade,
        exit_reason=exit_reason,
    )


# ---------------------------------------------------------------------------
# signal_diagnostics
# ---------------------------------------------------------------------------


def signal_diagnostics(sig: SignalSeries) -> dict:
    """Summary statistics over a :class:`SignalSeries`.

    Returns
    -------
    dict with keys:
      ``n_obs``                 — total length
      ``n_round_trips``         — completed entry→exit cycles
      ``n_open_at_end``         — 1 if series ends with an open position
      ``avg_holding_days``      — mean duration of completed trades (None if 0)
      ``max_holding_days``      — max duration of completed trades (0 if 0)
      ``pct_time_in_position``  — fraction of bars where position != 0
      ``n_long_entries`` / ``n_short_entries``
      ``n_exits_mean_revert`` / ``n_exits_stop`` / ``n_exits_time``
    """
    pos = sig.position
    n = pos.size
    if n == 0:
        return {
            "n_obs": 0,
            "n_round_trips": 0,
            "n_open_at_end": 0,
            "avg_holding_days": None,
            "max_holding_days": 0,
            "pct_time_in_position": 0.0,
            "n_long_entries": 0,
            "n_short_entries": 0,
            "n_exits_mean_revert": 0,
            "n_exits_stop": 0,
            "n_exits_time": 0,
        }

    prev = np.concatenate([[0], pos[:-1].astype(np.int32)])
    entries_mask = (prev == 0) & (pos != 0)
    n_long_entries = int(((pos == 1) & entries_mask).sum())
    n_short_entries = int(((pos == -1) & entries_mask).sum())

    n_mean_revert = sum(1 for r in sig.exit_reason if r == "mean_revert")
    n_stop = sum(1 for r in sig.exit_reason if r == "stop")
    n_time = sum(1 for r in sig.exit_reason if r == "time")
    n_round_trips = n_mean_revert + n_stop + n_time

    durations: list[int] = []
    for t in range(n):
        if sig.exit_reason[t] is not None and t > 0:
            durations.append(int(sig.days_in_trade[t - 1]))
    avg_hold = float(np.mean(durations)) if durations else None
    max_hold = int(max(durations)) if durations else 0

    in_pos = int((pos != 0).sum())
    return {
        "n_obs": n,
        "n_round_trips": n_round_trips,
        "n_open_at_end": int(pos[-1] != 0),
        "avg_holding_days": avg_hold,
        "max_holding_days": max_hold,
        "pct_time_in_position": in_pos / n,
        "n_long_entries": n_long_entries,
        "n_short_entries": n_short_entries,
        "n_exits_mean_revert": n_mean_revert,
        "n_exits_stop": n_stop,
        "n_exits_time": n_time,
    }
