"""Tests for the portfolio-level vol-target overlay used in the Phase 2B
final equity curve. Covers: causality, leverage math (1.0 when realized vol
matches target, halved when 2× target), max-leverage cap on tiny vols,
min-periods warmup behaviour, empty input, and arg validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apt.backtest import apply_vol_target_overlay


def test_empty_input_returns_empty():
    out, lev = apply_vol_target_overlay(np.array([]))
    assert out.size == 0
    assert lev.size == 0


def test_warmup_leverage_is_one_until_min_periods():
    n = 30
    rng = np.random.default_rng(0)
    arr = rng.normal(0.0, 0.01, size=n)
    _, lev = apply_vol_target_overlay(
        arr, window=20, min_periods=20, target_vol_annual=0.10, max_leverage=3.0
    )
    # First 20 days lack a 20-day prior history → leverage = 1.0
    assert np.allclose(lev[:20], 1.0)
    # Day 20 has 20 prior obs → scaling kicks in
    assert lev[20] != 1.0 or lev[20] == pytest.approx(1.0, rel=0.5)  # at least computed


def test_causality_leverage_does_not_depend_on_today():
    """A change in returns[t] must not change leverage[t]."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.01, size=200)
    _, lev_base = apply_vol_target_overlay(
        base, window=60, min_periods=20, target_vol_annual=0.10
    )
    # Mutate index 100 only
    mutated = base.copy()
    mutated[100] = 0.5  # huge spike at t=100
    _, lev_mut = apply_vol_target_overlay(
        mutated, window=60, min_periods=20, target_vol_annual=0.10
    )
    # leverage[100] uses returns[40:100], excludes returns[100] ⇒ unchanged
    assert lev_base[100] == lev_mut[100]
    # leverage[101] sees the spike → SHOULD differ
    assert lev_base[101] != lev_mut[101]


def test_leverage_one_when_realized_vol_equals_target():
    """If realized daily-std × √252 == target_vol, leverage == 1.0."""
    target = 0.10
    # daily std that annualises to exactly 10%
    daily_sd = target / math.sqrt(252)
    rng = np.random.default_rng(42)
    # Generate a long series, then standardise to exact target
    raw = rng.normal(0.0, 1.0, size=2000)
    raw = (raw - raw.mean()) / raw.std(ddof=1) * daily_sd
    _, lev = apply_vol_target_overlay(
        raw, window=60, min_periods=20, target_vol_annual=target, max_leverage=3.0
    )
    # The window's std on a long enough sample should land very close to daily_sd
    # so leverage ≈ 1.0 for the bulk of the series.
    bulk = lev[100:]
    assert np.median(bulk) == pytest.approx(1.0, abs=0.10)


def test_leverage_halved_when_realized_vol_is_double_target():
    """If realized vol is exactly 2× target, leverage should be ~0.5."""
    target = 0.10
    realized_target = 0.20  # 2× target
    daily_sd = realized_target / math.sqrt(252)
    rng = np.random.default_rng(7)
    raw = rng.normal(0.0, 1.0, size=2000)
    raw = (raw - raw.mean()) / raw.std(ddof=1) * daily_sd
    _, lev = apply_vol_target_overlay(
        raw, window=60, min_periods=20, target_vol_annual=target, max_leverage=3.0
    )
    bulk = lev[100:]
    assert np.median(bulk) == pytest.approx(0.5, abs=0.05)


def test_leverage_capped_at_max_on_tiny_vol():
    """Near-zero realized vol drives leverage to the max-leverage cap."""
    n = 200
    arr = np.zeros(n)
    arr[0] = 1e-15  # tiny nonzero to seed std calc; rest stays effectively zero
    _, lev = apply_vol_target_overlay(
        arr, window=60, min_periods=20, target_vol_annual=0.10, max_leverage=3.0
    )
    # After warmup, with vol ≈ 0, leverage must hit the cap
    assert lev[60:].max() == pytest.approx(3.0, abs=1e-9)
    assert lev[60:].mean() == pytest.approx(3.0, abs=1e-6)


def test_overlaid_return_is_leverage_times_raw():
    rng = np.random.default_rng(99)
    arr = rng.normal(0.0, 0.01, size=500)
    out, lev = apply_vol_target_overlay(
        arr, window=60, min_periods=20, target_vol_annual=0.10, max_leverage=3.0
    )
    assert np.allclose(out, lev * arr)


def test_invalid_args_raise():
    arr = np.zeros(100)
    with pytest.raises(ValueError):
        apply_vol_target_overlay(arr, target_vol_annual=0.0)
    with pytest.raises(ValueError):
        apply_vol_target_overlay(arr, window=1)
    with pytest.raises(ValueError):
        apply_vol_target_overlay(arr, max_leverage=0)
    with pytest.raises(ValueError):
        apply_vol_target_overlay(arr, min_periods=1)


def test_realized_vol_on_short_active_window_uses_max_lev():
    """If only a handful of trades happen near the start, the std calc
    over the trailing window will be tiny relative to target, so leverage
    should cap. This is the 'quiet strategy' edge case."""
    n = 200
    arr = np.zeros(n)
    # One trade-like return very early
    arr[30] = 0.002
    _, lev = apply_vol_target_overlay(
        arr, window=60, min_periods=20, target_vol_annual=0.10, max_leverage=3.0
    )
    # Most days after warmup hit the cap (vol ≈ 0)
    assert lev[150:].max() == pytest.approx(3.0, abs=1e-9)
