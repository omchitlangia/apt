"""Phase 2B §3.7 pair-kill switch.

PRIMARY trigger: RELATIONSHIP BREAKDOWN evidence on the spread itself —
the property the pair was selected for has degraded. Specifically, ANY of:

  (a) ADF on the last ``window`` days no longer rejects the unit root at
      ``adf_alpha`` (e.g. p > 0.05). The cointegration claim is dead on
      current data.
  (b) Half-life estimated from the same window blows out past the tradeable
      band (``hl > halflife_max_days``, ``hl ≤ 0``, or non-finite).
  (c) Vol regime shift: std of the second half of the window exceeds the
      first half by a factor of ``vol_ratio_max`` (e.g. 2.0). Spread vol
      has structurally re-scaled — sizing baselines no longer hold.

BACKSTOP (loose): K consecutive stops on the pair within the fold,
default K=4. This is intentionally weak — a normal losing streak on a
still-cointegrating pair must NOT bench it.

A normal mean-reverting trade hitting its stop once or twice is signal noise,
not breakdown evidence — that's why the K backstop is set high. The
relationship test catches the fast structural break; per-fold re-selection
catches slow drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from statsmodels.tsa.stattools import adfuller

from apt.signals.cointegration import half_life_ar1

KillMode = Literal["none", "loss_only", "relationship"]


@dataclass(frozen=True)
class KillVerdict:
    killed: bool
    reason: str  # 'adf_fail' | 'halflife_blowout' | 'vol_regime_shift' | 'loss_backstop' | 'consecutive_stops' | 'ok'
    detail: dict


def check_relationship_breakdown(
    spread_history: np.ndarray,
    *,
    window: int = 252,
    adf_alpha: float = 0.05,
    halflife_max_days: int = 60,
    vol_ratio_max: float = 2.0,
) -> KillVerdict:
    """Rolling re-test on the last ``window`` spread observations.

    Returns first failure in order (ADF → half-life → vol regime).
    """
    s = np.asarray(spread_history, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < window:
        return KillVerdict(False, "ok", {"reason": "insufficient_history", "n": int(s.size)})
    win = s[-window:]
    # (a) ADF
    try:
        adf_stat, adf_p, *_ = adfuller(win, autolag="AIC")
    except Exception as e:  # pragma: no cover  — adfuller is robust on real data
        return KillVerdict(False, "ok", {"adf_error": str(e)})
    if adf_p > adf_alpha:
        return KillVerdict(
            True,
            "adf_fail",
            {"adf_pvalue": float(adf_p), "adf_stat": float(adf_stat), "n": int(win.size)},
        )
    # (b) half-life on the same window
    hl = half_life_ar1(win)
    if not np.isfinite(hl) or hl <= 0 or hl > halflife_max_days:
        return KillVerdict(
            True,
            "halflife_blowout",
            {
                "half_life": float(hl) if np.isfinite(hl) else None,
                "halflife_max_days": int(halflife_max_days),
            },
        )
    # (c) vol regime shift
    half = window // 2
    std_first = float(np.std(win[:half], ddof=1)) if half >= 2 else float("nan")
    std_second = float(np.std(win[half:], ddof=1)) if window - half >= 2 else float("nan")
    if std_first > 0 and np.isfinite(std_second):
        ratio = std_second / std_first
        if ratio > vol_ratio_max:
            return KillVerdict(
                True,
                "vol_regime_shift",
                {
                    "std_first": std_first,
                    "std_second": std_second,
                    "ratio": ratio,
                    "ratio_max": float(vol_ratio_max),
                },
            )
    return KillVerdict(False, "ok", {"adf_pvalue": float(adf_p), "half_life": float(hl)})


def check_loss_backstop(
    consecutive_stops: int,
    cum_pair_loss: float,
    *,
    kill_K: int = 4,
    kill_cap: float = 0.04,
) -> KillVerdict:
    """Loose loss-count backstop.

    ``consecutive_stops`` is the count of back-to-back stop-out exits on
    this pair within the current fold; ``cum_pair_loss`` is the cumulative
    PORTFOLIO-fraction net loss attributable to this pair within the fold
    (negative when losing).
    """
    if consecutive_stops >= kill_K:
        return KillVerdict(
            True,
            "consecutive_stops",
            {"consecutive_stops": int(consecutive_stops), "K": int(kill_K)},
        )
    if cum_pair_loss <= -abs(kill_cap):
        return KillVerdict(
            True,
            "loss_backstop",
            {"cum_pair_loss": float(cum_pair_loss), "kill_cap": float(kill_cap)},
        )
    return KillVerdict(False, "ok", {})


def evaluate_kill(
    *,
    mode: KillMode,
    spread_history: np.ndarray,
    consecutive_stops: int,
    cum_pair_loss: float,
    window: int = 252,
    adf_alpha: float = 0.05,
    halflife_max_days: int = 60,
    vol_ratio_max: float = 2.0,
    kill_K: int = 4,
    kill_cap: float = 0.04,
) -> KillVerdict:
    """Dispatch kill checks per ``mode``.

    * ``'none'``         — never kill (sweep arm baseline)
    * ``'loss_only'``    — only the loose loss/stop backstop
    * ``'relationship'`` — relationship test + loss backstop
    """
    if mode == "none":
        return KillVerdict(False, "ok", {})
    if mode == "loss_only":
        return check_loss_backstop(
            consecutive_stops, cum_pair_loss, kill_K=kill_K, kill_cap=kill_cap
        )
    if mode == "relationship":
        v = check_relationship_breakdown(
            spread_history,
            window=window,
            adf_alpha=adf_alpha,
            halflife_max_days=halflife_max_days,
            vol_ratio_max=vol_ratio_max,
        )
        if v.killed:
            return v
        return check_loss_backstop(
            consecutive_stops, cum_pair_loss, kill_K=kill_K, kill_cap=kill_cap
        )
    raise ValueError(f"unknown kill mode: {mode!r}")
