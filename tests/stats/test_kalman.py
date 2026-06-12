"""Unit tests for the per-session local-level (adaptive-equilibrium) filter.

Covers: synthetic μ-path recovery, residual OU recovery, strict
causality (future mutations cannot change a past center), hyperparameter/
init provenance (leakage analogue of test_ou.py:297), and the frozen
(H=inf) degenerate case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from apt.stats.kalman import (
    run_local_level_mu,
    session_gain_from_half_life,
)
from apt.stats.ou import bertram_threshold, fit_ou_params


def _make_sessions(n_sessions: int, bars_per_session: int) -> np.ndarray:
    """Dense session ids, contiguous blocks."""
    return np.repeat(np.arange(n_sessions), bars_per_session).astype(np.int32)


def _simulate_local_level_plus_ou(
    *,
    n_sessions: int,
    bars_per_session: int,
    mu0: float,
    mu_drift_per_session: float,
    kappa: float,
    sigma: float,
    seed: int,
    random_walk_mu: bool = False,
    mu_rw_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spread = drifting μ path (per session) + mean-zero OU residual (per bar).

    ``random_walk_mu=True`` makes μ a per-session random walk with innovation
    std ``mu_rw_std`` (the local-level model the filter actually assumes).
    Otherwise μ ramps linearly at ``mu_drift_per_session`` (a stress shape
    an EWMA provably lags).

    Returns (spread, session_id, true_mu_path_per_bar).
    """
    rng = np.random.default_rng(seed)
    sids = _make_sessions(n_sessions, bars_per_session)
    n = sids.size
    if random_walk_mu:
        steps = rng.normal(0.0, mu_rw_std, size=n_sessions)
        steps[0] = 0.0
        mu_sess = mu0 + np.cumsum(steps)
    else:
        mu_sess = mu0 + mu_drift_per_session * np.arange(n_sessions)
    true_mu = np.repeat(mu_sess, bars_per_session)
    # OU residual around 0
    phi = math.exp(-kappa)
    eps_std = sigma * math.sqrt((1.0 - phi * phi) / (2.0 * kappa))
    resid = np.empty(n, dtype=float)
    resid[0] = 0.0
    for t in range(1, n):
        resid[t] = phi * resid[t - 1] + rng.normal(0.0, eps_std)
    return true_mu + resid, sids, true_mu


# ---------------------------------------------------------------------------
# Gain arithmetic
# ---------------------------------------------------------------------------


def test_gain_frozen_and_finite() -> None:
    assert session_gain_from_half_life(math.inf) == 0.0
    # H=1 -> K = 1 - 2^-1 = 0.5
    assert session_gain_from_half_life(1.0) == pytest.approx(0.5)
    # monotone decreasing in H
    assert session_gain_from_half_life(5.0) > session_gain_from_half_life(20.0)
    with pytest.raises(ValueError):
        session_gain_from_half_life(0.0)
    with pytest.raises(ValueError):
        session_gain_from_half_life(-3.0)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_recovers_random_walk_mu_path() -> None:
    """Filter tracks a random-walk μ (the local-level model it assumes)."""
    spread, sids, true_mu = _simulate_local_level_plus_ou(
        n_sessions=150,
        bars_per_session=75,
        mu0=0.0,
        mu_drift_per_session=0.0,
        kappa=0.05,
        sigma=0.01,
        seed=1,
        random_walk_mu=True,
        mu_rw_std=0.02,
    )
    res = run_local_level_mu(spread, sids, mu_init=true_mu[0], half_life_sessions=10.0)
    assert res.fit_ok
    warm = 20 * 75  # warm-up of a few half-lives
    tracked = res.mu_path[warm:]
    truth = true_mu[warm:]
    # High correlation with the true drifting level.
    corr = float(np.corrcoef(tracked, truth)[0, 1])
    assert corr > 0.9, corr
    # RMS tracking error small vs the realized μ range.
    rms = float(np.sqrt(np.mean((tracked - truth) ** 2)))
    mu_range = float(truth.max() - truth.min())
    assert rms < 0.25 * mu_range, (rms, mu_range)


def test_ewma_lags_a_pure_ramp_by_predicted_amount() -> None:
    """Sanity pin on the EWMA bias: a deterministic ramp of slope d/session
    is lagged in steady state by ~ d*(1-K)/K. Documents WHY a ramp is the
    wrong recovery target (see test_recovers_random_walk_mu_path)."""
    d, H = 0.01, 10.0
    spread, sids, true_mu = _simulate_local_level_plus_ou(
        n_sessions=200,
        bars_per_session=75,
        mu0=0.0,
        mu_drift_per_session=d,
        kappa=0.05,
        sigma=0.0001,
        seed=11,
    )
    res = run_local_level_mu(spread, sids, mu_init=0.0, half_life_sessions=H)
    k = res.k_gain
    predicted_lag = d * (1.0 - k) / k
    warm = 80 * 75
    observed_lag = float(np.median(true_mu[warm:] - res.mu_path[warm:]))
    assert observed_lag == pytest.approx(predicted_lag, rel=0.15), (observed_lag, predicted_lag)


def test_residual_recovers_ou_params() -> None:
    """After detrending with the filter, the residual's OU params match the sim."""
    kappa_true, sigma_true = 0.05, 0.01
    spread, sids, _ = _simulate_local_level_plus_ou(
        n_sessions=200,
        bars_per_session=75,
        mu0=0.0,
        mu_drift_per_session=0.005,
        kappa=kappa_true,
        sigma=sigma_true,
        seed=2,
    )
    res = run_local_level_mu(spread, sids, mu_init=0.0, half_life_sessions=10.0)
    fit = fit_ou_params(res.residual, freq_minutes=1)
    assert fit.fit_ok, fit.reason
    # κ within 35% (the per-session detrend injects some low-freq leakage;
    # this is a recovery sanity check, not a precision claim).
    assert fit.kappa == pytest.approx(kappa_true, rel=0.35)
    assert fit.sigma == pytest.approx(sigma_true, rel=0.35)


def test_frozen_is_constant_center() -> None:
    """H=inf holds μ at μ_init for the whole window (frozen control)."""
    spread, sids, true_mu = _simulate_local_level_plus_ou(
        n_sessions=50,
        bars_per_session=75,
        mu0=0.3,
        mu_drift_per_session=0.02,
        kappa=0.05,
        sigma=0.01,
        seed=3,
    )
    res = run_local_level_mu(spread, sids, mu_init=0.3, half_life_sessions=math.inf)
    assert res.fit_ok
    assert res.k_gain == 0.0
    assert np.allclose(res.mu_path, 0.3)
    # residual = spread - 0.3 exactly
    assert np.allclose(res.residual, spread - 0.3, equal_nan=True)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_mu_path_is_strictly_causal() -> None:
    """Mutating data in session s (and later) must NOT change μ applied
    in sessions <= s. The center applied during session s depends only on
    sessions < s."""
    spread, sids, _ = _simulate_local_level_plus_ou(
        n_sessions=40,
        bars_per_session=50,
        mu0=0.0,
        mu_drift_per_session=0.01,
        kappa=0.05,
        sigma=0.01,
        seed=4,
    )
    res_full = run_local_level_mu(spread, sids, mu_init=0.0, half_life_sessions=8.0)

    # Mutate everything from the start of session 25 onward.
    cut_session = 25
    cut_idx = int(np.flatnonzero(sids == cut_session)[0])
    mutated = spread.copy()
    rng = np.random.default_rng(999)
    mutated[cut_idx:] = rng.normal(50.0, 10.0, size=mutated.size - cut_idx)
    res_mut = run_local_level_mu(mutated, sids, mu_init=0.0, half_life_sessions=8.0)

    # μ applied in sessions 0..cut_session (inclusive) must be identical:
    # session ``cut_session``'s center is carried from session cut-1's close,
    # which used only sessions < cut_session.
    np.testing.assert_allclose(res_full.mu_path[: cut_idx + 1], res_mut.mu_path[: cut_idx + 1])
    # session_mu up to and including cut_session is unchanged.
    np.testing.assert_allclose(
        res_full.session_mu[: cut_session + 1], res_mut.session_mu[: cut_session + 1]
    )


def test_truncation_invariance() -> None:
    """Running on a prefix yields the same centers as running on the whole
    and truncating (the rolling-z causality contract, lifted to sessions)."""
    spread, sids, _ = _simulate_local_level_plus_ou(
        n_sessions=30,
        bars_per_session=40,
        mu0=0.0,
        mu_drift_per_session=0.01,
        kappa=0.05,
        sigma=0.01,
        seed=5,
    )
    full = run_local_level_mu(spread, sids, mu_init=0.0, half_life_sessions=6.0)
    for k_sess in (10, 18, 25):
        cut = int(np.flatnonzero(sids == k_sess)[0])
        pre = run_local_level_mu(spread[:cut], sids[:cut], mu_init=0.0, half_life_sessions=6.0)
        np.testing.assert_allclose(pre.mu_path, full.mu_path[:cut])


# ---------------------------------------------------------------------------
# Leakage analogue (hyperparameter / init provenance)
# ---------------------------------------------------------------------------


def test_init_and_hyperparameter_provenance() -> None:
    """The filter's only knobs are mu_init and half_life_sessions; appending
    post-window garbage cannot change the centers for the original window
    (mirror of test_ou.py::test_fit_does_not_see_post_train_data)."""
    spread, sids, _ = _simulate_local_level_plus_ou(
        n_sessions=20,
        bars_per_session=30,
        mu0=0.0,
        mu_drift_per_session=0.0,
        kappa=0.05,
        sigma=0.01,
        seed=6,
    )
    a = run_local_level_mu(spread, sids, mu_init=0.0, half_life_sessions=10.0)
    # Append garbage sessions; the centers for the first 20 sessions must
    # be unchanged because the recursion is forward-causal.
    rng = np.random.default_rng(7)
    extra_sids = np.repeat(np.arange(20, 25), 30).astype(np.int32)
    extra_spread = rng.normal(100.0, 5.0, size=extra_sids.size)
    spread2 = np.concatenate([spread, extra_spread])
    sids2 = np.concatenate([sids, extra_sids])
    b = run_local_level_mu(spread2, sids2, mu_init=0.0, half_life_sessions=10.0)
    np.testing.assert_allclose(a.mu_path, b.mu_path[: spread.size])


# ---------------------------------------------------------------------------
# Absorption guard + frozen-control equivalence (integration-level, pure math)
# ---------------------------------------------------------------------------


def test_absorption_no_drift_fast_filter_collapses_residual_hl() -> None:
    """On a PURE-OU sim (no μ drift), an aggressive (small-H) filter starts
    tracking the OU oscillation itself, collapsing the residual half-life
    far below the frozen half-life. The absorption guard ([0.5x,1.5x]) must
    therefore REJECT the fast filter — the filter must not invent drift."""
    # Pure OU around a constant mean, MULTI-SESSION half-life (HL ~ 3
    # sessions at 75 bars/session), no level drift — mirrors the real
    # data where the spread reverts over several sessions. A fast
    # session-level filter then chases the slow oscillation itself.
    n_sessions, bars = 300, 75
    spread, sids, true_mu = _simulate_local_level_plus_ou(
        n_sessions=n_sessions,
        bars_per_session=bars,
        mu0=0.0,
        mu_drift_per_session=0.0,
        kappa=0.003,
        sigma=0.01,
        seed=21,
        random_walk_mu=False,
    )
    frozen = fit_ou_params(spread, freq_minutes=1)
    assert frozen.fit_ok
    hl_frozen = frozen.half_life_minutes

    def residual_hl_ratio(H: float) -> float:
        res = run_local_level_mu(spread, sids, mu_init=frozen.mu, half_life_sessions=H)
        rfit = fit_ou_params(res.residual, freq_minutes=1)
        return rfit.half_life_minutes / hl_frozen if rfit.fit_ok else float("nan")

    # Frozen (H=inf) is identity: ratio == 1.0 (within numerical noise).
    assert residual_hl_ratio(math.inf) == pytest.approx(1.0, abs=1e-9)
    # A very fast filter absorbs the signal -> residual HL collapses -> ratio < 0.5
    fast_ratio = residual_hl_ratio(2.0)
    assert fast_ratio < 0.5, fast_ratio
    # Guard verdict: only H=inf (and slow enough H) are admissible here.
    assert not (0.5 <= fast_ratio <= 1.5)


def test_frozen_control_equivalence_ou_translation_invariance() -> None:
    """H=inf reproduces the OU-unit signal EXACTLY: the residual is X − μ
    (constant shift), and OU κ/σ_eq/half-life and the Bertram a* are
    translation-invariant, so Z_k == Z_OU and the trade set is identical."""
    # Any AR(1)-ish series; use a simulated OU.
    spread, sids, _ = _simulate_local_level_plus_ou(
        n_sessions=120,
        bars_per_session=75,
        mu0=0.7,
        mu_drift_per_session=0.0,
        kappa=0.04,
        sigma=0.012,
        seed=31,
    )
    frozen = fit_ou_params(spread, freq_minutes=5)
    assert frozen.fit_ok
    # Frozen-control: H=inf residual = X − μ exactly.
    res = run_local_level_mu(spread, sids, mu_init=frozen.mu, half_life_sessions=math.inf)
    resid_fit = fit_ou_params(res.residual, freq_minutes=5)
    assert resid_fit.fit_ok
    # κ, σ_eq, half-life translation-invariant (residual is X shifted by const).
    assert resid_fit.kappa == pytest.approx(frozen.kappa, rel=1e-6)
    assert resid_fit.sigma_eq == pytest.approx(frozen.sigma_eq, rel=1e-6)
    assert resid_fit.half_life_minutes == pytest.approx(frozen.half_life_minutes, rel=1e-6)
    # Bertram a* identical at any cost.
    c = 0.0015
    a_frozen = bertram_threshold(frozen, cost_log_per_round_trip=c).a_entry_z
    a_resid = bertram_threshold(resid_fit, cost_log_per_round_trip=c).a_entry_z
    assert a_resid == pytest.approx(a_frozen, rel=1e-6)
    # Z_k == Z_OU pointwise (the actual signal input is identical).
    z_ou = (spread - frozen.mu) / frozen.sigma_eq
    z_k = (spread - res.mu_path) / resid_fit.sigma_eq
    np.testing.assert_allclose(z_k, z_ou, rtol=1e-6, atol=1e-9, equal_nan=True)
