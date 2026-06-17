"""Unit tests for the Johansen pair cointegration test (Section 5)."""

from __future__ import annotations

import numpy as np

from apt.stats.johansen import is_cointegrated, johansen_pair_test


def test_cointegrated_pair_detected():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(600)) + 10.0  # random walk
    # y = 1.5 x + stationary noise -> cointegrated, true beta 1.5
    y = 1.5 * x + rng.standard_normal(600) * 0.5
    res = johansen_pair_test(y, x)
    assert res.fit_ok and is_cointegrated(res)
    assert res.rank_95 >= 1
    assert abs(abs(res.beta) - 1.5) < 0.3


def test_independent_random_walks_not_cointegrated():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(600))
    y = np.cumsum(rng.standard_normal(600))  # independent unit roots
    res = johansen_pair_test(y, x)
    assert res.fit_ok and not is_cointegrated(res)
    assert res.rank_95 == 0


def test_order_independence():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(600)) + 5.0
    y = 0.8 * x + rng.standard_normal(600) * 0.3
    r1 = johansen_pair_test(y, x)
    r2 = johansen_pair_test(x, y)
    # cointegration verdict is the same regardless of leg order
    assert is_cointegrated(r1) == is_cointegrated(r2)


def test_degenerate_input():
    res = johansen_pair_test(np.ones(10), np.arange(10.0))
    assert not res.fit_ok
