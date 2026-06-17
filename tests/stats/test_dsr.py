"""Unit tests for the Deflated Sharpe Ratio (Bailey & López de Prado, 2014)."""

from __future__ import annotations

import numpy as np
import pytest

from apt.stats.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_expected_max_sharpe_monotone_in_n_and_v():
    # More trials -> higher expected max; more cross-trial variance -> higher.
    v = 0.01
    assert expected_max_sharpe(2, v) < expected_max_sharpe(10, v) < expected_max_sharpe(100, v)
    assert expected_max_sharpe(50, 0.005) < expected_max_sharpe(50, 0.02)
    # N=1 -> no selection -> 0 deflator.
    assert expected_max_sharpe(1, v) == 0.0


def test_psr_against_zero_high_for_strong_series():
    rng = np.random.default_rng(0)
    # strongly positive, low-vol series -> PSR(0) near 1
    r = 0.01 + 0.001 * rng.standard_normal(500)
    sr = float(r.mean() / r.std(ddof=1))
    psr = probabilistic_sharpe_ratio(
        sr, sr_benchmark=0.0, n=r.size, gamma3=0.0, gamma4_nonexcess=3.0
    )
    assert psr > 0.99


def test_psr_against_zero_half_for_noise():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(2000)  # zero-mean noise
    sr = float(r.mean() / r.std(ddof=1))
    psr = probabilistic_sharpe_ratio(
        sr, sr_benchmark=0.0, n=r.size, gamma3=0.0, gamma4_nonexcess=3.0
    )
    assert 0.2 < psr < 0.8  # near 0.5, not significant


def test_dsr_below_psr_vs_zero():
    # Deflation against a positive SR_0 can only reduce the probability.
    rng = np.random.default_rng(2)
    r = 0.002 + 0.01 * rng.standard_normal(500)
    res = deflated_sharpe_ratio(r, n_trials=50, trial_sharpe_var=0.01)
    assert res.dsr <= res.psr_vs_zero
    assert 0.0 <= res.dsr <= 1.0
    assert res.p_value == pytest.approx(1.0 - res.dsr)


def test_dsr_drops_as_trials_increase():
    rng = np.random.default_rng(3)
    r = 0.0015 + 0.01 * rng.standard_normal(400)
    few = deflated_sharpe_ratio(r, n_trials=2, trial_sharpe_var=0.02)
    many = deflated_sharpe_ratio(r, n_trials=500, trial_sharpe_var=0.02)
    assert many.dsr < few.dsr  # more searching -> harder to be significant


def test_dsr_of_pure_noise_is_low():
    rng = np.random.default_rng(4)
    r = 0.01 * rng.standard_normal(500)  # zero-mean
    res = deflated_sharpe_ratio(r, n_trials=100, trial_sharpe_var=0.02)
    assert res.dsr < 0.5


def test_degenerate_inputs():
    res = deflated_sharpe_ratio(np.zeros(100), n_trials=10, trial_sharpe_var=0.01)
    assert np.isnan(res.dsr)
    short = deflated_sharpe_ratio(np.array([0.1, 0.2]), n_trials=10, trial_sharpe_var=0.01)
    assert np.isnan(short.dsr)
