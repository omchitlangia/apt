"""Unit tests for the joint (β, μ) per-session causal filter (Section 3)."""

from __future__ import annotations

import math

import numpy as np

from apt.stats.kalman import run_local_level_mu
from apt.stats.kalman_beta import (
    beta_collapse_flag,
    beta_stable_on_train,
    run_joint_beta_mu,
)


def _make_sessions(n_sessions: int, bars: int, rng) -> np.ndarray:
    return np.repeat(np.arange(n_sessions), bars)


def test_frozen_control_equivalence_reproduces_mu_only():
    # H_beta = inf  ->  joint filter residual == mu-only filter residual, exactly.
    rng = np.random.default_rng(0)
    n_sessions, bars = 40, 30
    sids = _make_sessions(n_sessions, bars, rng)
    n = sids.size
    x = np.cumsum(rng.standard_normal(n)) * 0.01 + 5.0
    beta, alpha = 1.3, -0.4
    spread = 0.02 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n) + rng.standard_normal(n) * 0.005
    y = spread + alpha + beta * x  # so spread = y - beta x - alpha
    mu_init = float(np.mean(spread[:bars]))

    mu_only = run_local_level_mu(spread, sids, mu_init=mu_init, half_life_sessions=20.0)
    joint = run_joint_beta_mu(
        y,
        x,
        sids,
        beta_init=beta,
        c_init=alpha + mu_init,
        half_life_beta_sessions=math.inf,
        half_life_c_sessions=20.0,
    )
    assert joint.fit_ok
    np.testing.assert_allclose(joint.residual, mu_only.residual, atol=1e-9)
    # beta never moved
    np.testing.assert_allclose(joint.session_beta, beta)


def test_synthetic_beta_recovery():
    # Slowly-ramping true beta should be tracked by the filtered session beta.
    rng = np.random.default_rng(1)
    n_sessions, bars = 60, 60
    sids = _make_sessions(n_sessions, bars, rng)
    n = sids.size
    x = np.cumsum(rng.standard_normal(n)) * 0.02 + 4.0
    beta_true = np.repeat(np.linspace(1.0, 1.6, n_sessions), bars)
    c_true = 0.5
    y = beta_true * x + c_true + rng.standard_normal(n) * 0.002
    res = run_joint_beta_mu(
        y,
        x,
        sids,
        beta_init=1.0,
        c_init=0.5,
        half_life_beta_sessions=5.0,
        half_life_c_sessions=20.0,
    )
    assert res.fit_ok and res.session_identified.mean() > 0.9
    # per-session observation tracks true beta; filtered beta ends near 1.6
    assert abs(res.session_beta[-1] - 1.6) < 0.1
    assert np.corrcoef(res.session_beta, np.linspace(1.0, 1.6, n_sessions))[0, 1] > 0.95


def test_truncation_invariance_causal():
    # Truncating the window to the first K sessions must not change the
    # carried state on the overlap (state at s depends only on sessions < s).
    rng = np.random.default_rng(2)
    n_sessions, bars = 30, 40
    sids = _make_sessions(n_sessions, bars, rng)
    n = sids.size
    x = np.cumsum(rng.standard_normal(n)) * 0.02 + 5.0
    y = 1.2 * x + 0.3 + rng.standard_normal(n) * 0.01
    full = run_joint_beta_mu(
        y,
        x,
        sids,
        beta_init=1.2,
        c_init=0.3,
        half_life_beta_sessions=8.0,
        half_life_c_sessions=15.0,
    )
    k = 18 * bars
    trunc = run_joint_beta_mu(
        y[:k],
        x[:k],
        sids[:k],
        beta_init=1.2,
        c_init=0.3,
        half_life_beta_sessions=8.0,
        half_life_c_sessions=15.0,
    )
    np.testing.assert_allclose(full.beta_path[:k], trunc.beta_path, atol=1e-12)
    np.testing.assert_allclose(full.c_path[:k], trunc.c_path, atol=1e-12)


def test_beta_collapse_when_comovement_vanishes():
    # First half: y co-moves with x (beta=1). Second half: y decouples (its
    # increments are independent of x) -> session beta-hat -> ~0 -> beta_s
    # collapses toward zero.
    rng = np.random.default_rng(3)
    n_sessions, bars = 60, 80
    sids = _make_sessions(n_sessions, bars, rng)
    half = n_sessions // 2 * bars
    n = sids.size
    x = np.cumsum(rng.standard_normal(n)) * 0.02 + 5.0
    y = np.empty(n)
    y[:half] = 1.0 * x[:half] + 0.2 + rng.standard_normal(half) * 0.002
    # decoupled second half: own random walk, no x dependence
    y[half:] = 0.2 + 5.0 + np.cumsum(rng.standard_normal(n - half)) * 0.02
    res = run_joint_beta_mu(
        y,
        x,
        sids,
        beta_init=1.0,
        c_init=0.2,
        half_life_beta_sessions=5.0,
        half_life_c_sessions=20.0,
    )
    assert res.session_beta[-1] < 0.5  # collapsed toward 0
    flag = beta_collapse_flag(res, train_residual_var=float(np.var(res.residual[:half])))
    assert flag["beta_toward_zero"] is True


def test_beta_stable_guard():
    rng = np.random.default_rng(4)
    n_sessions, bars = 30, 50
    sids = _make_sessions(n_sessions, bars, rng)
    n = sids.size
    x = np.cumsum(rng.standard_normal(n)) * 0.02 + 5.0
    y = 1.1 * x + 0.4 + rng.standard_normal(n) * 0.003
    res = run_joint_beta_mu(
        y,
        x,
        sids,
        beta_init=1.1,
        c_init=0.4,
        half_life_beta_sessions=10.0,
        half_life_c_sessions=20.0,
    )
    assert beta_stable_on_train(res) is True


def test_degenerate_inputs():
    bad = run_joint_beta_mu(
        np.array([]),
        np.array([]),
        np.array([]),
        beta_init=1.0,
        c_init=0.0,
        half_life_beta_sessions=10.0,
        half_life_c_sessions=20.0,
    )
    assert not bad.fit_ok
