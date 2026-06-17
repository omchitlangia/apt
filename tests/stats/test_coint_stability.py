"""Unit tests for the rolling cointegration-stability gate (Section 4)."""

from __future__ import annotations

import numpy as np

from apt.stats.coint_stability import gate_index, gate_summary, rolling_adf_pvalues


def _ar1(n, phi, rng, sigma=1.0):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + sigma * rng.standard_normal()
    return x


def test_stationary_series_low_pvalue_no_gate():
    rng = np.random.default_rng(0)
    s = _ar1(400, 0.5, rng)  # strongly mean-reverting
    roll = rolling_adf_pvalues(s, window=60, step=5)
    assert roll.n_windows > 0
    summ = gate_summary(roll)
    assert summ["mean_pvalue"] < 0.10
    assert summ["gated"] is False
    assert gate_index(roll) is None


def test_random_walk_high_pvalue_gates():
    rng = np.random.default_rng(1)
    s = np.cumsum(rng.standard_normal(400))  # unit root
    roll = rolling_adf_pvalues(s, window=60, step=5)
    summ = gate_summary(roll)
    assert summ["mean_pvalue"] > 0.30
    assert summ["gated"] is True
    assert gate_index(roll) is not None


def test_regime_break_gates_partway():
    # Stationary first half, random-walk second half -> gate fires in 2nd half.
    rng = np.random.default_rng(2)
    a = _ar1(250, 0.3, rng)
    b = a[-1] + np.cumsum(rng.standard_normal(250))
    s = np.concatenate([a, b])
    roll = rolling_adf_pvalues(s, window=60, step=5)
    gi = gate_index(roll)
    assert gi is not None
    assert gi > 250  # gate fires after the break, not before


def test_degenerate_short_series():
    roll = rolling_adf_pvalues(np.arange(10.0), window=60, step=5)
    assert roll.n_windows == 0
    assert gate_summary(roll)["gated"] is False
