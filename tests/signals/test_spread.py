"""Tests for apt.signals.spread — spread math, causal rolling z, every
state-machine transition, and the asset-agnostic numpy/list interface."""

from __future__ import annotations

import numpy as np
import pytest

from apt.signals.spread import (
    SignalSeries,
    compute_spread,
    generate_signals,
    rolling_zscore,
    signal_diagnostics,
)

# ---------------------------------------------------------------------------
# compute_spread
# ---------------------------------------------------------------------------


def test_compute_spread_matches_log_formula():
    p1 = np.array([100.0, 110.0, 120.0, 105.0])
    p2 = np.array([50.0, 55.0, 60.0, 52.0])
    spread = compute_spread(p1, p2, beta=1.0, intercept=0.5)
    expected = np.log(p1) - 1.0 * np.log(p2) - 0.5
    np.testing.assert_allclose(spread, expected)


def test_compute_spread_default_intercept_is_zero():
    p1 = np.array([100.0, 110.0])
    p2 = np.array([50.0, 55.0])
    np.testing.assert_allclose(
        compute_spread(p1, p2, beta=0.5),
        np.log(p1) - 0.5 * np.log(p2),
    )


def test_compute_spread_beta_carries_forward_unchanged():
    """Same (beta, intercept, p1, p2) ⇒ identical spread — no internal re-fit."""
    p1 = np.array([100.0, 105.0, 110.0, 115.0])
    p2 = np.array([50.0, 52.0, 54.0, 56.0])
    s1 = compute_spread(p1, p2, beta=1.0, intercept=0.3)
    s2 = compute_spread(p1, p2, beta=1.0, intercept=0.3)
    np.testing.assert_array_equal(s1, s2)


def test_compute_spread_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_spread(np.zeros(10), np.zeros(11), beta=1.0)


def test_compute_spread_non_positive_prices_raise():
    with pytest.raises(ValueError, match="positive"):
        compute_spread(np.array([100.0, 0.0]), np.array([50.0, 60.0]), beta=1.0)
    with pytest.raises(ValueError, match="positive"):
        compute_spread(np.array([100.0, -1.0]), np.array([50.0, 60.0]), beta=1.0)


def test_compute_spread_accepts_python_lists_asset_agnostic():
    """Engine must accept plain lists, not just numpy arrays — port portability."""
    spread = compute_spread([100.0, 110.0], [50.0, 55.0], beta=1.0)
    assert isinstance(spread, np.ndarray)
    assert spread.shape == (2,)


def test_compute_spread_empty_returns_empty():
    spread = compute_spread(np.array([]), np.array([]), beta=1.0)
    assert spread.shape == (0,)


def test_compute_spread_2d_raises():
    with pytest.raises(ValueError, match="1-D"):
        compute_spread(np.zeros((10, 2)), np.zeros((10, 2)), beta=1.0)


# ---------------------------------------------------------------------------
# rolling_zscore — warm-up, causality, math
# ---------------------------------------------------------------------------


def test_rolling_zscore_warmup_first_n_minus_1_are_nan():
    spread = np.arange(100, dtype=float)
    z = rolling_zscore(spread, window=10)
    assert np.isnan(z[:9]).all()
    assert np.isfinite(z[9])


def test_rolling_zscore_causality_future_changes_do_not_affect_past():
    """The defining causality test: change z[100:], assert z[:100] is unchanged."""
    rng = np.random.default_rng(0)
    spread = rng.normal(size=200)
    z_orig = rolling_zscore(spread, window=20)
    spread_mod = spread.copy()
    spread_mod[100:] = 999.0
    z_mod = rolling_zscore(spread_mod, window=20)
    # Indices before the mutation point must be byte-identical
    np.testing.assert_array_equal(z_orig[:100], z_mod[:100])
    # And the mutation actually changed the post-100 part (sanity)
    both_finite = np.isfinite(z_orig[100:]) & np.isfinite(z_mod[100:])
    assert both_finite.any(), "mutation produced all-NaN post-mutation z"
    assert not np.allclose(z_orig[100:][both_finite], z_mod[100:][both_finite])


def test_rolling_zscore_window_only_looks_back():
    """Independent realisation: changing spread[t+1:] does NOT change z[t]."""
    rng = np.random.default_rng(1)
    spread = rng.normal(size=80)
    z_full = rolling_zscore(spread, window=10)
    # Pick a middle index t and re-compute z on the truncated series
    t = 50
    z_truncated = rolling_zscore(spread[: t + 1], window=10)
    # z at index t with full series == z at index t with truncated series
    assert z_full[t] == pytest.approx(z_truncated[t], rel=1e-10)


def test_rolling_zscore_constant_series_is_nan_not_inf():
    spread = np.full(50, 3.14)
    z = rolling_zscore(spread, window=20)
    # Once we have a full window, std=0 → z must be NaN, never ±inf
    assert np.isnan(z[20:]).all()
    assert not np.isinf(z).any()


def test_rolling_zscore_known_value_at_last_index():
    """Manually check the last z against the explicit trailing-window mean/std."""
    spread = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 6)  # n=30
    z = rolling_zscore(spread, window=5, ddof=1)
    win = spread[-5:]
    expected = (spread[-1] - win.mean()) / win.std(ddof=1)
    assert z[-1] == pytest.approx(expected, abs=1e-10)


def test_rolling_zscore_invalid_window_raises():
    with pytest.raises(ValueError, match="window"):
        rolling_zscore(np.zeros(50), window=1)


def test_rolling_zscore_invalid_min_periods_raises():
    with pytest.raises(ValueError, match="min_periods"):
        rolling_zscore(np.zeros(50), window=10, min_periods=1)
    with pytest.raises(ValueError, match="min_periods"):
        rolling_zscore(np.zeros(50), window=10, min_periods=20)


def test_rolling_zscore_empty_returns_empty():
    z = rolling_zscore(np.array([]), window=10)
    assert z.shape == (0,)


def test_rolling_zscore_accepts_lists():
    z = rolling_zscore([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], window=3)
    assert isinstance(z, np.ndarray)
    assert z.shape == (6,)


# ---------------------------------------------------------------------------
# generate_signals — entry/exit/stop/time transitions
# ---------------------------------------------------------------------------


def test_signals_long_entry():
    z = np.array([0.0, -2.5])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.position[0] == 0
    assert sig.position[1] == +1
    assert sig.days_in_trade[1] == 1


def test_signals_short_entry():
    z = np.array([0.0, +2.5])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.position[1] == -1
    assert sig.days_in_trade[1] == 1


def test_signals_long_mean_revert_exit():
    z = np.array([-2.5, -1.0, +0.3])  # enter, hold, cross exit band
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.position.tolist() == [+1, +1, 0]
    assert sig.exit_reason == [None, None, "mean_revert"]


def test_signals_short_mean_revert_exit():
    z = np.array([+2.5, +1.0, -0.1])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.position.tolist() == [-1, -1, 0]
    assert sig.exit_reason[2] == "mean_revert"


def test_signals_long_stop_loss():
    z = np.array([-2.5, -3.0, -4.0])  # diverging
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.exit_reason[2] == "stop"
    assert sig.position[2] == 0


def test_signals_short_stop_loss():
    z = np.array([+2.5, +3.0, +4.0])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.exit_reason[2] == "stop"


def test_signals_time_stop_fires_after_max_holding():
    # Long entry at t=0; z stays inside the bands; time stop at days_in_trade >= max
    z = np.array([-2.5] + [-1.0] * 10)
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=5)
    # t=0 enter held=1; t=1 held=2; ...; t=4 held=5 (after increment from 4)
    # t=5: held >= 5 at entry → time exit
    assert sig.position[5] == 0
    assert sig.exit_reason[5] == "time"


def test_signals_no_same_bar_flip_long_to_short():
    """A position closes on bar t; any opposite-side re-entry waits to t+1."""
    z = np.array([-2.5, +2.5, +2.5])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.position[0] == +1
    # t=1: exit triggers (mean_revert via the -exit band), pos must be 0 — NOT -1
    assert sig.position[1] == 0
    assert sig.exit_reason[1] == "mean_revert"
    # t=2: now flat, opposite-side entry takes
    assert sig.position[2] == -1


def test_signals_nan_z_before_any_entry_stays_flat():
    z = np.array([np.nan] * 5 + [-2.5])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.position[:5].tolist() == [0, 0, 0, 0, 0]
    assert sig.position[5] == +1


def test_signals_nan_mid_trade_carries_position_unchanged():
    z = np.array([-2.5, -1.0, np.nan, -1.0, +0.3])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    assert sig.position[0] == +1
    assert sig.position[1] == +1
    # NaN bar: position carries; days_in_trade does NOT increment
    assert sig.position[2] == +1
    assert sig.days_in_trade[2] == sig.days_in_trade[1]
    # Then resumes: still in band → still long; finally exits at mean_revert
    assert sig.position[3] == +1
    assert sig.position[4] == 0
    assert sig.exit_reason[4] == "mean_revert"


def test_signals_invalid_thresholds_raise():
    z = np.zeros(10)
    with pytest.raises(ValueError, match="exit"):
        generate_signals(z, entry=1.0, exit=2.0)
    with pytest.raises(ValueError, match="stop"):
        generate_signals(z, entry=2.0, stop=1.5)
    with pytest.raises(ValueError, match="positive"):
        generate_signals(z, entry=-1.0)
    with pytest.raises(ValueError, match="max_holding"):
        generate_signals(z, max_holding=0)


def test_signals_only_z_t_used_at_step_t():
    """Causality: signal at index t depends only on z[0..t]."""
    z = np.array([-2.5, -1.0, 0.3, 1.0, 0.5, -0.5, -2.5])
    sig_full = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=20)
    # Truncate at t=3, recompute, assert prefix matches
    sig_trunc = generate_signals(z[:4], entry=2.0, exit=0.5, stop=3.5, max_holding=20)
    np.testing.assert_array_equal(sig_full.position[:4], sig_trunc.position)
    np.testing.assert_array_equal(sig_full.days_in_trade[:4], sig_trunc.days_in_trade)
    assert sig_full.exit_reason[:4] == sig_trunc.exit_reason


def test_signals_position_dtype_is_compact():
    """int8 keeps the signal series cheap to store/serialize."""
    sig = generate_signals(np.zeros(10))
    assert sig.position.dtype == np.int8
    assert sig.days_in_trade.dtype == np.int32


# ---------------------------------------------------------------------------
# signal_diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_counts_round_trips_and_holding():
    # Two clean round-trips: long then short
    z = np.array([0.0, -2.5, -1.0, +0.3, 0.0, +2.5, +1.0, -0.1])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    d = signal_diagnostics(sig)
    assert d["n_round_trips"] == 2
    assert d["n_long_entries"] == 1
    assert d["n_short_entries"] == 1
    assert d["n_exits_mean_revert"] == 2
    assert d["n_exits_stop"] == 0
    assert d["n_exits_time"] == 0
    assert d["n_open_at_end"] == 0
    # Each trade: enter at t, hold 1 bar, exit. days_in_trade before exit = 2.
    assert d["avg_holding_days"] == pytest.approx(2.0)
    assert d["max_holding_days"] == 2


def test_diagnostics_open_at_end_no_round_trip_no_avg():
    z = np.array([0.0, -2.5, -1.0])  # entered, still long at end
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    d = signal_diagnostics(sig)
    assert d["n_open_at_end"] == 1
    assert d["n_round_trips"] == 0
    assert d["avg_holding_days"] is None
    assert d["max_holding_days"] == 0


def test_diagnostics_pct_time_in_position():
    # 4 of 8 bars in position
    z = np.array([0.0, -2.5, -1.0, +0.3, 0.0, +2.5, +1.0, -0.1])
    sig = generate_signals(z, entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    d = signal_diagnostics(sig)
    in_pos = int((sig.position != 0).sum())
    assert d["pct_time_in_position"] == pytest.approx(in_pos / 8)


def test_diagnostics_empty_signal_series():
    sig = SignalSeries(
        position=np.array([], dtype=np.int8),
        days_in_trade=np.array([], dtype=np.int32),
        exit_reason=[],
    )
    d = signal_diagnostics(sig)
    assert d["n_obs"] == 0
    assert d["n_round_trips"] == 0
    assert d["avg_holding_days"] is None
    assert d["pct_time_in_position"] == 0.0


# ---------------------------------------------------------------------------
# Asset-agnostic end-to-end: prices → spread → z → signals → diagnostics
# ---------------------------------------------------------------------------


def test_pipeline_end_to_end_on_synthetic_cointegrated_pair():
    """Smoke test the full asset-agnostic pipeline on a synthetic cointegrated
    pair (no equity-specific code paths exercised)."""
    rng = np.random.default_rng(7)
    n = 600
    common = rng.normal(scale=0.01, size=n).cumsum()
    noise_a = np.zeros(n)
    noise_b = np.zeros(n)
    for i in range(1, n):
        noise_a[i] = 0.4 * noise_a[i - 1] + rng.normal(scale=0.005)
        noise_b[i] = 0.4 * noise_b[i - 1] + rng.normal(scale=0.005)
    p_a = np.exp(4.0 + common + noise_a)
    p_b = np.exp(3.0 + 0.7 * common + noise_b)

    # Hedge ratio "carried forward" — true beta is 0.7
    spread = compute_spread(p_a, p_b, beta=0.7, intercept=1.0)
    z = rolling_zscore(spread, window=60)
    sig = generate_signals(z)
    diag = signal_diagnostics(sig)

    # Shape contract
    assert spread.shape == (n,)
    assert z.shape == (n,)
    assert sig.position.shape == (n,)
    # Some non-trivial trading happened
    assert diag["n_round_trips"] >= 1
    assert diag["pct_time_in_position"] > 0.0


def test_pipeline_asset_agnostic_no_polars_in_inputs():
    """Reinforces the asset-agnostic contract: all signal-layer inputs are
    plain numpy or python lists. No polars/equity-data shapes required."""
    p1_list = [100.0, 101.0, 102.0, 99.0, 98.0, 100.0, 102.0, 104.0] * 12
    p2_list = [50.0, 50.5, 51.0, 49.5, 49.0, 50.0, 51.0, 52.0] * 12
    spread = compute_spread(p1_list, p2_list, beta=1.0)
    z = rolling_zscore(spread.tolist(), window=10)
    sig = generate_signals(z.tolist(), entry=2.0, exit=0.5, stop=3.5, max_holding=10)
    # Did not crash and produced consistent shapes
    assert sig.position.size == len(p1_list)
