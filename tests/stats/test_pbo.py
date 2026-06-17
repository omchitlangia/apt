"""Unit tests for PBO via CSCV (Bailey et al., 2017)."""

from __future__ import annotations

import numpy as np
import pytest

from apt.stats.pbo import pbo_cscv


def test_all_noise_strategies_pbo_near_half():
    # N independent pure-noise strategies: the IS-best is overfit ~half the
    # time out-of-sample. PBO should sit near 0.5.
    rng = np.random.default_rng(0)
    m = rng.standard_normal((600, 12))
    res = pbo_cscv(m, n_blocks=12)
    assert 0.35 < res.pbo < 0.65


def test_one_genuine_winner_low_pbo():
    # One strategy carries a real, persistent drift; the rest are noise.
    # The IS-best is almost always the real winner and it generalizes OOS,
    # so PBO is low.
    rng = np.random.default_rng(1)
    m = rng.standard_normal((600, 10)) * 0.01
    m[:, 0] += 0.01  # strong constant edge
    res = pbo_cscv(m, n_blocks=12)
    assert res.pbo < 0.1
    assert res.frac_is_best_also_oos_best > 0.8


def test_pbo_bounds_and_shape():
    rng = np.random.default_rng(2)
    m = rng.standard_normal((300, 6))
    res = pbo_cscv(m, n_blocks=10)
    assert 0.0 <= res.pbo <= 1.0
    assert res.n_combinations == 252  # C(10,5)
    assert res.logits.size == 252
    assert res.n_strategies == 6


def test_validation_errors():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 1)), n_blocks=10)  # need >= 2 strategies
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 4)), n_blocks=9)  # odd blocks
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((5, 4)), n_blocks=10)  # too few observations
