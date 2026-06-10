"""Ornstein-Uhlenbeck fitting and Bertram (2010) optimal-threshold solver.

Pure functions, no I/O, no config. The signal-layer wrapper that turns
an :class:`OUFit` + :class:`OUThresholds` into an
``IntradaySignalSeries`` lives in :mod:`apt.intraday.signals` next to
``generate_signals_two_regime``.

Process
-------
The OU SDE on the (log-)spread::

    dX_t = kappa * (mu - X_t) * dt + sigma * dW_t                (1)

For minute bars sampled at ``dt = 1 bar``, the exact discretisation is
an AR(1)::

    X_{t+1} = mu * (1 - phi) + phi * X_t + eps_t,
    eps_t ~ N(0, sigma_eps^2),
    phi   = exp(-kappa),
    sigma_eps^2 = sigma^2 * (1 - phi^2) / (2 kappa).

Bertram (2010) — trading rule and objective
-------------------------------------------
Trading rule (centred on ``mu``; the strategy is mu-translation
invariant):

* Enter short the spread at ``X = mu + a`` (spread overpriced).
* Exit at ``X = mu``.
* Symmetrically enter long at ``X = mu - a``, exit at ``X = mu``.

Long-run expected log-return per unit time (the Bertram objective)::

    mu_objective(a, c) = (a - c) / L(a)                          (2)

where ``L(a)`` is the expected cycle length::

    L(a) = T_out(a) + T_in(a)                                    (3)

* ``T_out(a) = E[first passage from mu+a to mu]`` (drift toward mean,
  fast).
* ``T_in(a)  = E[first passage from mu to mu+a OR mu-a]`` (drift
  away from mean, slow — exponential in (a/sigma_eq)^2).

Derivation (this module)
------------------------
We derive ``L(a)`` from the backward Kolmogorov ODE for the centred OU
``dX = -kappa X dt + sigma dW``::

    (1/2) * sigma^2 * v''(x) - kappa * x * v'(x) = -1            (4)

Substitute the dimensionless ``z = x / sigma_eq`` where
``sigma_eq = sigma / sqrt(2 kappa)``:

* ``T_out(a)``: solve (4) on ``x in (0, +inf)`` with ``v(0) = 0`` and
  the minimal-growth-at-infinity condition. Result in dimensionless::

      T_out(a) = (1/kappa) * sqrt(pi/2)
                 * integral_{0}^{z=a/sigma_eq} exp(y^2/2) * erfc(y/sqrt(2)) dy

* ``T_in(a)``: solve (4) on ``x in (-a, +a)`` with ``v(-a)=v(+a)=0``.
  Result::

      T_in(a)  = (1/kappa) * sqrt(pi/2)
                 * integral_{0}^{z=a/sigma_eq} exp(y^2/2) * erf(y/sqrt(2)) dy

* Sum (``erfc + erf = 1``)::

      L(a) = T_out + T_in
           = (1/kappa) * sqrt(pi/2) * integral_{0}^{z} exp(y^2/2) dy
           = (pi / (2 * kappa)) * erfi(z / sqrt(2))               (5)

  with ``z = a/sigma_eq``. So ``L(a) = (pi/(2 kappa)) *
  erfi(a/(sigma_eq * sqrt(2)))``.

Combining (2) and (5)::

    objective(a, c) = (a - c) * (2 * kappa) / (pi * erfi(a / (sigma_eq * sqrt(2))))   (6)

In Z-OU units ``a_z = a / sigma_eq`` and ``c_z = c / sigma_eq``::

    objective_dimensionless(a_z, c_z) = (a_z - c_z) / erfi(a_z / sqrt(2))   (7)

(scaled by ``sigma_eq * 2 * kappa / pi`` which is constant in ``a``.)

We solve (7) by 1-D bounded optimization. Bertram (2010) eq (5)-(7)
agrees with this form (after notation reconciliation).

The Monte-Carlo first-passage validator in
``tests/stats/test_ou_mc.py`` cross-checks the analytic objective at the
optimum against pathwise simulation — that is the decisive transcription
check (see ``docs/ou_thresholds_design.md`` §8.3).

References
----------
Bertram, W. K. (2010). "Analytic solutions for optimal statistical
arbitrage trading." Physica A 389(11), 2234-2243.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.special import erfi


@dataclass(frozen=True)
class OUFit:
    """Per-pair OU parameters fit on the TRAIN window only.

    All time-bearing fields use the FIT FREQUENCY's "bar" as the
    discrete-time unit. ``half_life_minutes = half_life_bars *
    freq_minutes``.
    """

    kappa: float
    mu: float
    sigma: float
    sigma_eq: float
    half_life_bars: float
    half_life_minutes: float
    phi: float
    sigma_eps: float
    n_obs: int
    freq_minutes: int
    fit_ok: bool
    reason: str


@dataclass(frozen=True)
class OUThresholds:
    """Bertram-derived optimal entry threshold in Z-OU units.

    The trading rule enters short-spread at ``Z >= +a_entry_z``,
    long-spread at ``Z <= -a_entry_z``, exits when ``Z`` crosses 0.
    """

    a_entry_z: float
    cost_log: float
    expected_return_per_unit_time: float
    note: str
    fit_ok: bool


_OU_PAPER_TAG = "bertram2010-eq5to7-derived"


def fit_ou_params(
    spread: np.ndarray,
    *,
    freq_minutes: int = 1,
    min_obs: int = 60,
) -> OUFit:
    """Fit OU parameters by OLS AR(1) on a TRAIN-window (log-)spread.

    NaN observations are dropped before fitting (a non-contiguous NaN
    pattern is fine — the regression treats consecutive valid pairs
    only). Returns ``fit_ok=False`` with a populated ``reason`` if:

    * fewer than ``min_obs`` consecutive-pair observations,
    * ``phi <= 0`` (anti-persistent — not mean-reverting in OU sense),
    * ``phi >= 1`` (unit-root / explosive regime).
    """
    s = np.asarray(spread, dtype=float)
    if s.size < 2:
        return _bad_fit("input too short", freq_minutes)

    # Form consecutive-pair regression rows, dropping any pair containing NaN.
    x_prev = s[:-1]
    y_curr = s[1:]
    valid = np.isfinite(x_prev) & np.isfinite(y_curr)
    if int(valid.sum()) < min_obs:
        return _bad_fit(f"n_obs<{min_obs} ({int(valid.sum())})", freq_minutes)

    xp = x_prev[valid]
    yc = y_curr[valid]
    X = sm.add_constant(xp)
    fit = sm.OLS(yc, X).fit()
    c = float(fit.params[0])
    phi = float(fit.params[1])
    resid = np.asarray(fit.resid, dtype=float)
    sigma_eps = float(resid.std(ddof=1)) if resid.size > 1 else float("nan")

    if not np.isfinite(phi):
        return _bad_fit("phi non-finite", freq_minutes)
    if phi <= 0.0:
        return _bad_fit(f"phi<=0 ({phi:.4f}) anti-persistent", freq_minutes)
    if phi >= 1.0:
        return _bad_fit(f"phi>=1 ({phi:.4f}) random-walk", freq_minutes)
    if not np.isfinite(sigma_eps) or sigma_eps <= 0.0:
        return _bad_fit("sigma_eps non-positive", freq_minutes)

    kappa = -math.log(phi)  # per bar
    mu = c / (1.0 - phi)
    # sigma from sigma_eps^2 = sigma^2 (1 - phi^2) / (2 kappa)
    sigma = sigma_eps * math.sqrt(2.0 * kappa / (1.0 - phi * phi))
    sigma_eq = sigma / math.sqrt(2.0 * kappa)
    half_life_bars = math.log(2.0) / kappa
    half_life_minutes = half_life_bars * float(freq_minutes)

    return OUFit(
        kappa=kappa,
        mu=mu,
        sigma=sigma,
        sigma_eq=sigma_eq,
        half_life_bars=half_life_bars,
        half_life_minutes=half_life_minutes,
        phi=phi,
        sigma_eps=sigma_eps,
        n_obs=int(valid.sum()),
        freq_minutes=int(freq_minutes),
        fit_ok=True,
        reason="",
    )


def _bad_fit(reason: str, freq_minutes: int) -> OUFit:
    nan = float("nan")
    return OUFit(
        kappa=nan,
        mu=nan,
        sigma=nan,
        sigma_eq=nan,
        half_life_bars=nan,
        half_life_minutes=nan,
        phi=nan,
        sigma_eps=nan,
        n_obs=0,
        freq_minutes=int(freq_minutes),
        fit_ok=False,
        reason=reason,
    )


def cycle_length(a: float, *, kappa: float, sigma_eq: float) -> float:
    """Expected cycle length L(a) per Bertram (2010) eq (5).

    ``L(a) = (pi / (2 kappa)) * erfi(a / (sigma_eq * sqrt(2)))``.

    Returns ``inf`` if ``a <= 0`` (no trade) or non-finite params.
    """
    if not (np.isfinite(a) and np.isfinite(kappa) and np.isfinite(sigma_eq)):
        return float("inf")
    if a <= 0.0 or kappa <= 0.0 or sigma_eq <= 0.0:
        return float("inf")
    z = a / (sigma_eq * math.sqrt(2.0))
    erfi_val = float(erfi(z))
    if not np.isfinite(erfi_val) or erfi_val <= 0.0:
        return float("inf")
    return (math.pi / (2.0 * kappa)) * erfi_val


def expected_return_per_unit_time(a: float, *, cost: float, kappa: float, sigma_eq: float) -> float:
    """Bertram objective ``(a - c) / L(a)`` at threshold ``a`` and cost ``c``."""
    if a <= cost:
        return 0.0  # would-be-trade pays <= 0 net
    L = cycle_length(a, kappa=kappa, sigma_eq=sigma_eq)
    if not np.isfinite(L) or L <= 0.0:
        return 0.0
    return (a - cost) / L


def bertram_threshold(
    fit: OUFit,
    *,
    cost_log_per_round_trip: float,
) -> OUThresholds:
    """Solve for the optimal entry threshold ``a*`` maximising eq (2).

    The objective is 1-D and unimodal on ``a > c``. We use
    ``scipy.optimize.minimize_scalar`` with ``method='bounded'`` on a
    wide grid in ``a`` (in spread units).

    Returns the optimum expressed in Z-OU units (``a_z = a / sigma_eq``)
    so the signal layer is scale-free.
    """
    if not fit.fit_ok:
        return OUThresholds(
            a_entry_z=float("nan"),
            cost_log=float(cost_log_per_round_trip),
            expected_return_per_unit_time=float("nan"),
            note=f"{_OU_PAPER_TAG}; aborted: fit_ok=False",
            fit_ok=False,
        )
    if not (cost_log_per_round_trip >= 0.0 and math.isfinite(cost_log_per_round_trip)):
        raise ValueError(
            f"cost_log_per_round_trip must be non-negative finite, got {cost_log_per_round_trip}"
        )

    c = float(cost_log_per_round_trip)
    kappa = fit.kappa
    sigma_eq = fit.sigma_eq

    # Search bounds in real spread units.
    # Lower: must exceed cost (else net <= 0).
    # Upper: erfi grows ~ exp(z^2) / (z sqrt(pi)); for a = 10 sigma_eq, z = ~7
    # and erfi(7) ~ 4e20 -- objective is effectively zero. Bound at 15 sigma_eq.
    a_lo = max(c + 1e-9 * max(sigma_eq, 1.0), 1e-12)
    a_hi = max(15.0 * sigma_eq, a_lo + 1e-6)

    if a_hi <= a_lo:
        return OUThresholds(
            a_entry_z=float("nan"),
            cost_log=c,
            expected_return_per_unit_time=0.0,
            note=f"{_OU_PAPER_TAG}; infeasible search interval",
            fit_ok=False,
        )

    def neg_obj(a: float) -> float:
        v = expected_return_per_unit_time(a, cost=c, kappa=kappa, sigma_eq=sigma_eq)
        # minimize_scalar minimises; negate. Penalise non-positive objective slightly
        # below zero so the optimiser strictly prefers feasible interior points.
        if v <= 0.0:
            return 1e-12 + (a / max(a_hi, 1.0)) * 1e-15
        return -v

    res = minimize_scalar(
        neg_obj,
        bounds=(a_lo, a_hi),
        method="bounded",
        options={"xatol": 1e-10},
    )
    a_star = float(res.x)
    obj_star = expected_return_per_unit_time(a_star, cost=c, kappa=kappa, sigma_eq=sigma_eq)

    if obj_star <= 0.0 or not np.isfinite(obj_star):
        return OUThresholds(
            a_entry_z=float("nan"),
            cost_log=c,
            expected_return_per_unit_time=0.0,
            note=f"{_OU_PAPER_TAG}; objective<=0 at any a (infeasible at this cost)",
            fit_ok=False,
        )

    a_entry_z = a_star / sigma_eq
    return OUThresholds(
        a_entry_z=a_entry_z,
        cost_log=c,
        expected_return_per_unit_time=obj_star,
        note=_OU_PAPER_TAG,
        fit_ok=True,
    )


__all__ = [
    "OUFit",
    "OUThresholds",
    "fit_ou_params",
    "bertram_threshold",
    "cycle_length",
    "expected_return_per_unit_time",
]
