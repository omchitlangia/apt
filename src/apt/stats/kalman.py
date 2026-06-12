"""Per-session local-level (adaptive-equilibrium) filter for the spread.

Unit K. Tracks a slowly-drifting equilibrium level ``mu_t`` on an
already-selected pair-fold's (log-)spread ``X_t = log P^Y - beta*log P^X
- alpha`` (``beta``, ``alpha`` frozen from the daily EG fit — NOT tracked
here; see ``docs/kalman_design.md`` Decision Log Q1).

Model (local level / random-walk equilibrium, observed at the SESSION
frequency):

    mu_s   = mu_{s-1} + w_s          (random-walk equilibrium, per session)
    ybar_s = mu_s     + v_s          (session-mean spread observed around mu)

The **steady-state** Kalman filter for this model is a constant-gain
exponential update — equivalently the West & Harrison discount form. We
parameterize the single degree of freedom by the **re-anchor half-life
``H`` in SESSIONS** (``docs/kalman_design.md`` Q2/Q3):

    K = 1 - 2**(-1/H)        (steady-state gain; H -> inf  =>  K = 0)
    mu_{s+1} = (1 - K) * mu_s + K * ybar_s

Causality (the property the leakage tests pin): the ``mu`` *applied while
trading session s* is the value carried from the **close of session
s-1**; it depends only on sessions ``< s``. The session-mean ``ybar_s``
updates ``mu_{s+1}``, used from the open of session ``s+1``. No bar in
session ``s`` uses session-``s`` (or later) data to set its own center.

``H = inf`` (``K = 0``) holds ``mu_t`` at ``mu_init`` for the whole
window ⇒ the **frozen-mu control** (reproduces the OU unit exactly).

All functions are pure (no I/O, no config), matching :mod:`apt.stats.ou`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Sentinel for "frozen" (no re-anchoring). math.inf is accepted too.
FROZEN_HALF_LIFE = math.inf


@dataclass(frozen=True)
class LocalLevelResult:
    """Output of :func:`run_local_level_mu` for one pair-fold/window.

    ``mu_path`` is the per-bar center actually applied at each bar (the
    carried-from-previous-session value, constant within a session).
    ``residual = X - mu_path``. ``session_mu`` is the length-S center
    active during each dense-ranked session.
    """

    mu_path: np.ndarray
    residual: np.ndarray
    session_mu: np.ndarray
    session_mean: np.ndarray
    k_gain: float
    half_life_sessions: float
    mu_init: float
    n_sessions: int
    fit_ok: bool
    reason: str


def session_gain_from_half_life(half_life_sessions: float) -> float:
    """Steady-state local-level gain ``K = 1 - 2**(-1/H)``.

    ``H = inf`` (or non-positive treated as invalid by the caller) maps to
    ``K = 0`` (frozen). ``H = 1`` maps to ``K = 0.5``. Monotone decreasing
    in ``H``: a longer half-life re-anchors more slowly.
    """
    if not math.isfinite(half_life_sessions):
        return 0.0
    if half_life_sessions <= 0:
        raise ValueError(f"half_life_sessions must be > 0 or inf, got {half_life_sessions}")
    return float(1.0 - 2.0 ** (-1.0 / half_life_sessions))


def run_local_level_mu(
    spread: np.ndarray,
    session_id: np.ndarray,
    *,
    mu_init: float,
    half_life_sessions: float,
    tradeable: np.ndarray | None = None,
) -> LocalLevelResult:
    """Run the per-session causal local-level filter over one window.

    Parameters
    ----------
    spread
        Length-N (log-)spread. NaN at non-tradeable bars is ignored when
        forming each session's mean observation.
    session_id
        Length-N session index. Need NOT be dense — sessions are taken in
        first-appearance order; ``mu`` updates at each session boundary.
    mu_init
        Initial equilibrium level (the TRAIN OU ``mu``; the train fit is
        the burn-in — see Decision Log Q2). Applied during the first
        session.
    half_life_sessions
        Re-anchor half-life ``H`` in sessions. ``inf`` ⇒ frozen.
    tradeable
        Optional length-N bool; if given, only tradeable bars contribute
        to a session's mean observation. NaN spreads are always excluded.

    Returns
    -------
    LocalLevelResult
        ``fit_ok=False`` only if the input is degenerate (no finite
        spread, or length 0). The frozen case (H=inf) always fits.
    """
    s = np.asarray(spread, dtype=float)
    sids = np.asarray(session_id)
    n = s.size
    if n == 0 or sids.size != n:
        return _bad(mu_init, half_life_sessions, "empty or shape-mismatched input")
    if not math.isfinite(mu_init):
        return _bad(mu_init, half_life_sessions, "mu_init not finite")

    finite = np.isfinite(s)
    if tradeable is not None:
        finite = finite & np.asarray(tradeable, dtype=bool)
    if not finite.any():
        return _bad(mu_init, half_life_sessions, "no finite/tradeable spread")

    k = session_gain_from_half_life(half_life_sessions)

    # Sessions in first-appearance order (NOT assuming dense or sorted).
    # We require the session id to be piecewise-constant in blocks (it is,
    # by construction of the intraday panel); use run-boundaries.
    boundaries = np.flatnonzero(np.diff(sids)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))  # half-open [start, end)
    n_sessions = len(starts)

    mu_path = np.empty(n, dtype=float)
    session_mu = np.empty(n_sessions, dtype=float)
    session_mean = np.full(n_sessions, np.nan, dtype=float)

    mu_cur = float(mu_init)
    for j in range(n_sessions):
        a, b = starts[j], ends[j]
        # Apply the carried center to every bar in this session (causal:
        # mu_cur depends only on sessions < j).
        mu_path[a:b] = mu_cur
        session_mu[j] = mu_cur
        # Observe this session's mean spread level (tradeable, finite bars).
        seg_finite = finite[a:b]
        if seg_finite.any():
            ybar = float(np.mean(s[a:b][seg_finite]))
            session_mean[j] = ybar
            # Update for the NEXT session's open.
            mu_cur = (1.0 - k) * mu_cur + k * ybar
        # else: no observation this session -> carry mu_cur unchanged.

    residual = s - mu_path
    return LocalLevelResult(
        mu_path=mu_path,
        residual=residual,
        session_mu=session_mu,
        session_mean=session_mean,
        k_gain=float(k),
        half_life_sessions=float(half_life_sessions),
        mu_init=float(mu_init),
        n_sessions=int(n_sessions),
        fit_ok=True,
        reason="",
    )


def _bad(mu_init: float, hl: float, reason: str) -> LocalLevelResult:
    empty = np.empty(0, dtype=float)
    return LocalLevelResult(
        mu_path=empty,
        residual=empty,
        session_mu=empty,
        session_mean=empty,
        k_gain=float("nan"),
        half_life_sessions=float(hl),
        mu_init=float(mu_init),
        n_sessions=0,
        fit_ok=False,
        reason=reason,
    )


__all__ = [
    "FROZEN_HALF_LIFE",
    "LocalLevelResult",
    "run_local_level_mu",
    "session_gain_from_half_life",
]
