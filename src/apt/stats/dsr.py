"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Pure functions, no I/O, no config — matches :mod:`apt.stats.ou`.

References
----------
Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
*Journal of Portfolio Management*, 40(5), 94-107.

All Sharpe ratios here are **per-period** (NOT annualized). If you have an
annualized Sharpe, divide by ``sqrt(periods_per_year)`` before passing it in,
and pass the matching per-period return series.

The Probabilistic Sharpe Ratio (PSR) of a candidate against a benchmark
``sr_benchmark`` is::

    PSR = Phi[ (sr_hat - sr_benchmark) * sqrt(T - 1)
               / sqrt(1 - skew*sr_hat + (kurt - 1)/4 * sr_hat^2) ]

where ``skew`` (γ3) and ``kurt`` (γ4, NON-excess: 3 for a Gaussian) are the
3rd/4th standardized moments of the return series and ``T`` the sample
length. The Deflated Sharpe Ratio is ``PSR`` evaluated against the
**expected maximum** Sharpe under ``N`` independent trials::

    SR_0 = sqrt(V) * [ (1 - γ) * Phi^{-1}(1 - 1/N)
                       + γ * Phi^{-1}(1 - 1/(N*e)) ]

with ``γ`` the Euler-Mascheroni constant, ``V`` the cross-trial variance of
the per-period Sharpe estimates, and ``e`` Euler's number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_MASCHERONI: float = 0.5772156649015329


@dataclass(frozen=True)
class DSRResult:
    sr_hat: float  # per-period observed Sharpe
    sr_benchmark: float  # SR_0 (expected max under N trials)
    psr_vs_zero: float  # PSR against 0 benchmark
    dsr: float  # PSR against SR_0  == deflated probability
    p_value: float  # 1 - dsr
    n_trials: int
    sample_length: int
    skew: float
    kurtosis_nonexcess: float
    trial_sharpe_var: float


def probabilistic_sharpe_ratio(
    sr_hat: float,
    *,
    sr_benchmark: float,
    n: int,
    gamma3: float,
    gamma4_nonexcess: float,
) -> float:
    """PSR = Phi[(sr_hat - sr_benchmark) sqrt(n-1) / sqrt(1 - γ3 sr + (γ4-1)/4 sr²)]."""
    if n < 2:
        return float("nan")
    denom = 1.0 - gamma3 * sr_hat + (gamma4_nonexcess - 1.0) / 4.0 * sr_hat * sr_hat
    if denom <= 0:
        return float("nan")
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, trial_sharpe_var: float) -> float:
    """Expected maximum of ``n_trials`` iid Sharpe estimates (the SR_0 deflator).

    ``SR_0 = sqrt(V) [ (1-γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)) ]``.
    """
    if n_trials < 1 or trial_sharpe_var < 0:
        return float("nan")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(trial_sharpe_var) * ((1.0 - g) * q1 + g * q2))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    n_trials: int,
    trial_sharpe_var: float,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio of a per-period return series.

    Parameters
    ----------
    returns
        Per-period (e.g. daily) return series of the candidate strategy.
    n_trials
        Honest number of trials/configurations searched (N).
    trial_sharpe_var
        Cross-trial variance ``V`` of the per-period Sharpe estimates.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = int(r.size)
    if t < 3 or r.std(ddof=1) == 0:
        return DSRResult(
            sr_hat=float("nan"),
            sr_benchmark=float("nan"),
            psr_vs_zero=float("nan"),
            dsr=float("nan"),
            p_value=float("nan"),
            n_trials=int(n_trials),
            sample_length=t,
            skew=float("nan"),
            kurtosis_nonexcess=float("nan"),
            trial_sharpe_var=float(trial_sharpe_var),
        )
    sr_hat = float(r.mean() / r.std(ddof=1))
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))  # non-excess (3 for normal)
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_var)
    psr0 = probabilistic_sharpe_ratio(sr_hat, sr_benchmark=0.0, n=t, gamma3=g3, gamma4_nonexcess=g4)
    dsr = probabilistic_sharpe_ratio(sr_hat, sr_benchmark=sr0, n=t, gamma3=g3, gamma4_nonexcess=g4)
    return DSRResult(
        sr_hat=sr_hat,
        sr_benchmark=sr0,
        psr_vs_zero=psr0,
        dsr=dsr,
        p_value=1.0 - dsr,
        n_trials=int(n_trials),
        sample_length=t,
        skew=g3,
        kurtosis_nonexcess=g4,
        trial_sharpe_var=float(trial_sharpe_var),
    )


__all__ = [
    "DSRResult",
    "EULER_MASCHERONI",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
]
