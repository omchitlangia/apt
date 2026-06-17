"""Joint (β, μ) per-session causal local-level filter (Unit K + β-escalation).

Extends :mod:`apt.stats.kalman` (μ-only) by letting the hedge ratio β track
alongside the equilibrium level. State ``θ_s = (β_s, c_s)`` where
``c_s = α + μ_s`` folds the frozen intercept and the tracking level; updated
once per session, carried causally to the next session. Diagonal steady-state
(constant-gain / West-Harrison discount) updates, one re-anchor half-life per
dimension. See ``docs/beta_escalation_design.md`` for the full derivation.

Equations (per session ``s``):

    β̂_s = cov_s(Δx, Δy) / var_s(Δx)          (returns regression; collapse-prone)
    ℓ̂_s = mean_s(y − β_s · x)                 (level concentrated on carried β)
    β_{s+1} = (1−K_β) β_s + K_β β̂_s           (if identified; else carry)
    c_{s+1} = (1−K_c) c_s + K_c ℓ̂_s
    K_x     = 1 − 2^(−1/H_x)                   (H in sessions; ∞ ⇒ K=0)

Trading residual (causal): ``r_t = y_t − β_s x_t − c_s`` for bar t in session s.

``H_β = ∞`` reproduces :func:`apt.stats.kalman.run_local_level_mu` exactly
(the frozen-control equivalence pin).

Pure functions, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from apt.stats.kalman import session_gain_from_half_life

# Labeled defaults (β-escalation design §3) → ASSUMPTIONS A5.
BETA_COLLAPSE_RATIO: float = 0.5
RESID_VAR_TOL: float = 3.0
MIN_SESSION_INCREMENTS: int = 10
MIN_VAR_X: float = 1e-10
BETA_STABILITY_BAND: tuple[float, float] = (0.25, 4.0)


@dataclass(frozen=True)
class JointBetaMuResult:
    beta_path: np.ndarray  # per-bar β applied (carried, const within session)
    c_path: np.ndarray  # per-bar level c applied
    residual: np.ndarray  # y − β_path·x − c_path
    session_beta: np.ndarray  # length-S carried β per session
    session_c: np.ndarray
    session_beta_obs: np.ndarray  # length-S session returns-β observation (nan if unidentified)
    session_identified: np.ndarray  # length-S bool
    k_beta: float
    k_c: float
    half_life_beta_sessions: float
    half_life_c_sessions: float
    beta_init: float
    c_init: float
    n_sessions: int
    fit_ok: bool
    reason: str


def _session_bounds(session_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.flatnonzero(np.diff(session_id)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [session_id.size]))
    return starts, ends


def run_joint_beta_mu(
    log_y: np.ndarray,
    log_x: np.ndarray,
    session_id: np.ndarray,
    *,
    beta_init: float,
    c_init: float,
    half_life_beta_sessions: float,
    half_life_c_sessions: float,
    tradeable: np.ndarray | None = None,
    min_session_increments: int = MIN_SESSION_INCREMENTS,
    min_var_x: float = MIN_VAR_X,
) -> JointBetaMuResult:
    """Run the per-session causal joint (β, c) filter over one window."""
    y = np.asarray(log_y, dtype=float)
    x = np.asarray(log_x, dtype=float)
    sids = np.asarray(session_id)
    n = y.size
    if n == 0 or x.size != n or sids.size != n:
        return _bad(beta_init, c_init, half_life_beta_sessions, half_life_c_sessions, "shape")
    if not (math.isfinite(beta_init) and math.isfinite(c_init)):
        return _bad(
            beta_init, c_init, half_life_beta_sessions, half_life_c_sessions, "init not finite"
        )

    finite = np.isfinite(y) & np.isfinite(x)
    if tradeable is not None:
        finite = finite & np.asarray(tradeable, dtype=bool)
    if not finite.any():
        return _bad(
            beta_init, c_init, half_life_beta_sessions, half_life_c_sessions, "no finite bars"
        )

    k_beta = session_gain_from_half_life(half_life_beta_sessions)
    k_c = session_gain_from_half_life(half_life_c_sessions)

    starts, ends = _session_bounds(sids)
    n_sessions = len(starts)
    beta_path = np.empty(n)
    c_path = np.empty(n)
    session_beta = np.empty(n_sessions)
    session_c = np.empty(n_sessions)
    session_beta_obs = np.full(n_sessions, np.nan)
    session_identified = np.zeros(n_sessions, dtype=bool)

    beta_cur = float(beta_init)
    c_cur = float(c_init)
    for j in range(n_sessions):
        a, b = starts[j], ends[j]
        # Causal: apply carried state to every bar in this session.
        beta_path[a:b] = beta_cur
        c_path[a:b] = c_cur
        session_beta[j] = beta_cur
        session_c[j] = c_cur

        seg = slice(a, b)
        fin = finite[seg]
        ys = y[seg][fin]
        xs = x[seg][fin]

        # Level observation concentrated on the carried β (carry if no data).
        level_obs = float(np.mean(ys - beta_cur * xs)) if ys.size else c_cur

        # β observation: returns regression on within-session increments.
        if ys.size >= min_session_increments + 1:
            dx = np.diff(xs)
            dy = np.diff(ys)
            vx = float(np.var(dx))
            if dx.size >= min_session_increments and vx >= min_var_x:
                beta_obs = float(np.cov(dx, dy, ddof=0)[0, 1] / vx)
                session_beta_obs[j] = beta_obs
                session_identified[j] = True
                beta_cur = (1.0 - k_beta) * beta_cur + k_beta * beta_obs
        # else: unidentified -> carry β unchanged.

        c_cur = (1.0 - k_c) * c_cur + k_c * level_obs

    residual = y - beta_path * x - c_path
    return JointBetaMuResult(
        beta_path=beta_path,
        c_path=c_path,
        residual=residual,
        session_beta=session_beta,
        session_c=session_c,
        session_beta_obs=session_beta_obs,
        session_identified=session_identified,
        k_beta=float(k_beta),
        k_c=float(k_c),
        half_life_beta_sessions=float(half_life_beta_sessions),
        half_life_c_sessions=float(half_life_c_sessions),
        beta_init=float(beta_init),
        c_init=float(c_init),
        n_sessions=int(n_sessions),
        fit_ok=True,
        reason="",
    )


def beta_collapse_flag(
    res: JointBetaMuResult,
    *,
    train_residual_var: float,
    collapse_ratio: float = BETA_COLLAPSE_RATIO,
    resid_var_tol: float = RESID_VAR_TOL,
) -> dict:
    """Raise the collapse flag when β drifts toward 0 AND residual var destabilizes.

    Returns a dict with the components so the driver can persist them.
    """
    if not res.fit_ok or res.session_beta.size == 0 or res.beta_init == 0:
        return {"collapsed": False, "min_beta_ratio": float("nan"), "resid_var_ratio": float("nan")}
    min_ratio = float(np.min(res.session_beta) / res.beta_init)
    test_var = float(np.var(res.residual[np.isfinite(res.residual)]))
    var_ratio = test_var / train_residual_var if train_residual_var > 0 else float("inf")
    beta_collapsed = min_ratio < collapse_ratio
    var_unstable = (var_ratio > resid_var_tol) or (var_ratio < 1.0 / resid_var_tol)
    return {
        "collapsed": bool(beta_collapsed and var_unstable),
        "beta_toward_zero": bool(beta_collapsed),
        "resid_var_unstable": bool(var_unstable),
        "min_beta_ratio": min_ratio,
        "terminal_beta_ratio": float(res.session_beta[-1] / res.beta_init),
        "resid_var_ratio": var_ratio,
        "frac_sessions_unidentified": float(1.0 - res.session_identified.mean()),
    }


def beta_stable_on_train(
    res: JointBetaMuResult, band: tuple[float, float] = BETA_STABILITY_BAND
) -> bool:
    """β-stability guard for selection: β_s within [lo,hi]×β_0 over the window."""
    if not res.fit_ok or res.beta_init == 0:
        return False
    ratios = res.session_beta / res.beta_init
    return bool(np.all(ratios >= band[0]) and np.all(ratios <= band[1]))


def _bad(b0: float, c0: float, hb: float, hc: float, reason: str) -> JointBetaMuResult:
    e = np.empty(0)
    return JointBetaMuResult(
        beta_path=e,
        c_path=e,
        residual=e,
        session_beta=e,
        session_c=e,
        session_beta_obs=e,
        session_identified=np.zeros(0, dtype=bool),
        k_beta=float("nan"),
        k_c=float("nan"),
        half_life_beta_sessions=float(hb),
        half_life_c_sessions=float(hc),
        beta_init=float(b0),
        c_init=float(c0),
        n_sessions=0,
        fit_ok=False,
        reason=reason,
    )


__all__ = [
    "BETA_COLLAPSE_RATIO",
    "BETA_STABILITY_BAND",
    "JointBetaMuResult",
    "MIN_SESSION_INCREMENTS",
    "MIN_VAR_X",
    "RESID_VAR_TOL",
    "beta_collapse_flag",
    "beta_stable_on_train",
    "run_joint_beta_mu",
]
