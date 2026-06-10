"""Tests for the session-aware rolling z and TOD-vol profile."""

import numpy as np

from apt.intraday.calendar import NSE_BARS_PER_SESSION
from apt.intraday.zscore import (
    fit_tod_vol_profile,
    intraday_rolling_zscore,
    sessionized_rolling_zscore,
    tod_adjusted_zscore,
)


def _two_sessions(n_per: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build two synthetic sessions with a big level jump across the gap."""
    rng = np.random.default_rng(seed=0)
    s1 = rng.normal(0.0, 1.0, n_per)
    s2 = rng.normal(100.0, 1.0, n_per)  # huge level jump on the overnight
    spread = np.concatenate([s1, s2])
    sids = np.concatenate([np.zeros(n_per, dtype=np.int32), np.ones(n_per, dtype=np.int32)])
    bins = np.concatenate([np.arange(n_per, dtype=np.int32), np.arange(n_per, dtype=np.int32)])
    return spread, sids, bins


def test_sessionized_z_does_not_blend_overnight_gap() -> None:
    spread, sids, _ = _two_sessions(n_per=30)
    z = sessionized_rolling_zscore(spread, sids, window=10)
    # Session 2 bars 10..29 must NOT see the level jump — the z should not
    # explode (which it would if the window blended session 1 means in).
    s2 = z[30 + 10 :]
    assert np.isfinite(s2).all()
    assert np.abs(s2).max() < 5.0


def test_sessionized_z_warmup_returns_nan() -> None:
    spread = np.arange(50, dtype=float)
    sids = np.zeros(50, dtype=np.int32)
    z = sessionized_rolling_zscore(spread, sids, window=20)
    # First 19 bars must be NaN; the 20th onward must be finite.
    assert np.isnan(z[:19]).all()
    assert np.isfinite(z[19:]).all()


def test_sessionized_z_is_causal() -> None:
    """Truncating the future never changes a past value."""
    rng = np.random.default_rng(seed=42)
    s = rng.normal(0.0, 1.0, 100)
    sids = np.zeros(100, dtype=np.int32)
    z_full = sessionized_rolling_zscore(s, sids, window=20)
    for k in (30, 50, 80):
        z_trunc = sessionized_rolling_zscore(s[:k], sids[:k], window=20)
        np.testing.assert_allclose(z_trunc, z_full[:k], equal_nan=True)


def test_tod_vol_profile_length_and_smoothing() -> None:
    spread, sids, bins = _two_sessions(n_per=80)
    sigma = fit_tod_vol_profile(spread, sids, bins, window=10, smooth_radius=2)
    assert sigma.shape == (NSE_BARS_PER_SESSION,)
    # Minutes well inside the trained range must be finite. The rolling-mean
    # warm-up makes minutes 0..8 NaN; the smoother (radius 2) reaches into
    # bar 9 from bar 7 onward and reaches OUTSIDE the trained range up to
    # bar 81 (it sees the trained bar 79 from up to bar 81).
    assert np.isfinite(sigma[10:78]).all()
    # Bars far outside the trained range (more than smooth_radius beyond)
    # must remain NaN — no training samples could leak there.
    assert np.isnan(sigma[82:]).all()


def test_tod_adjusted_z_finite_in_training_range() -> None:
    spread, sids, bins = _two_sessions(n_per=80)
    sigma = fit_tod_vol_profile(spread, sids, bins, window=10)
    z = tod_adjusted_zscore(spread, sids, bins, sigma, window=10)
    # Should be finite from index 9 onward (rolling window of 10 fills at bar 9).
    assert np.isnan(z[:9]).all()
    assert np.isfinite(z[9:]).all()


def test_intraday_z_session_warmup_suppression() -> None:
    """First K bars of each session are forced to NaN by warmup, even
    when the global rolling window would already be finite."""
    spread = np.arange(100, dtype=float)
    sids = np.concatenate([np.zeros(50, dtype=np.int32), np.ones(50, dtype=np.int32)])
    z = intraday_rolling_zscore(spread, sids, window=10, session_warmup_bars=5)
    # session 1 first 5 bars suppressed (indices 50..54), then finite
    assert np.isnan(z[50:55]).all()
    assert np.isfinite(z[55:]).all()


def test_intraday_z_multi_session_window_fills_after_warmup() -> None:
    """A window > 1 session must still produce finite z once the global
    rolling window is filled — this is the case session-local z can't handle."""
    spread = np.cumsum(np.random.default_rng(0).normal(0, 1, 800))
    sids = np.concatenate(
        [
            np.zeros(100, dtype=np.int32),
            np.ones(100, dtype=np.int32),
            np.full(100, 2, dtype=np.int32),
            np.full(100, 3, dtype=np.int32),
            np.full(100, 4, dtype=np.int32),
            np.full(100, 5, dtype=np.int32),
            np.full(100, 6, dtype=np.int32),
            np.full(100, 7, dtype=np.int32),
        ]
    )
    # Window spans 3 sessions (300 bars)
    z = intraday_rolling_zscore(spread, sids, window=300, session_warmup_bars=0)
    # The session-local function would produce ALL NaN (window > session length).
    session_local = sessionized_rolling_zscore(spread, sids, window=300)
    assert np.isnan(session_local).all()
    # Multi-session chronological: finite from bar 299 onward.
    assert np.isfinite(z[299:]).all()
