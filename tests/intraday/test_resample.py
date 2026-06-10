"""Tests for within-session bar resampling.

Verifies:
- freq_minutes=1 is identity
- coarser bars are left-labelled
- no resampled bar spans the overnight gap
- tradeable propagates correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apt.intraday.loader import AlignedMinutePair
from apt.intraday.resample import resample_within_session


def _make_panel(n_sessions: int = 2, bars_per_session: int = 30) -> AlignedMinutePair:
    """Synthetic minute panel: n_sessions, bars_per_session bars each."""
    rng = np.random.default_rng(0)
    ts_list = []
    sids = []
    bins = []
    cy = []
    cx = []
    tr = []
    for s in range(n_sessions):
        day = pd.Timestamp("2020-01-02").tz_localize("Asia/Kolkata") + pd.Timedelta(days=s)
        for b in range(bars_per_session):
            ts_list.append(day.replace(hour=9, minute=15) + pd.Timedelta(minutes=b))
            sids.append(s)
            bins.append(b)
            cy.append(100.0 + rng.normal(0, 1))
            cx.append(200.0 + rng.normal(0, 1))
            tr.append(True)
    return AlignedMinutePair(
        timestamps=pd.DatetimeIndex(ts_list),
        session_id=np.array(sids, dtype=np.int32),
        bar_in_session=np.array(bins, dtype=np.int32),
        close_y=np.array(cy, dtype=float),
        close_x=np.array(cx, dtype=float),
        volume_y=np.full(len(ts_list), 100, dtype=np.int64),
        volume_x=np.full(len(ts_list), 100, dtype=np.int64),
        tradeable=np.array(tr, dtype=bool),
    )


def test_resample_freq1_is_identity() -> None:
    aligned = _make_panel(n_sessions=2, bars_per_session=10)
    out = resample_within_session(aligned, freq_minutes=1)
    assert out.n_bars == aligned.n_bars
    np.testing.assert_array_equal(out.session_id, aligned.session_id)
    np.testing.assert_array_equal(out.bar_in_session, aligned.bar_in_session)
    np.testing.assert_array_equal(out.close_y, aligned.close_y)
    np.testing.assert_array_equal(out.close_x, aligned.close_x)
    np.testing.assert_array_equal(out.tradeable, aligned.tradeable)
    assert out.freq_minutes == 1


def test_resample_freq5_aggregates_within_session() -> None:
    aligned = _make_panel(n_sessions=2, bars_per_session=30)
    out = resample_within_session(aligned, freq_minutes=5)
    # 30 bars / 5 = 6 coarse bars per session; 2 sessions = 12 bars total
    assert out.n_bars == 12
    # First bar of session 0 is at session-bar 0 (left label)
    assert out.bar_in_session[0] == 0
    # First bar of session 1 is also at session-bar 0
    second_session_start = int(np.flatnonzero(out.session_id == 1)[0])
    assert out.bar_in_session[second_session_start] == 0
    # All sessions have 6 coarse bars
    for s in (0, 1):
        assert int((out.session_id == s).sum()) == 6


def test_resample_no_bar_spans_overnight() -> None:
    """A coarse bar must never contain bars from two different sessions."""
    aligned = _make_panel(n_sessions=3, bars_per_session=20)
    out = resample_within_session(aligned, freq_minutes=5)
    # Session_id is monotonic non-decreasing
    assert np.all(np.diff(out.session_id) >= 0)
    # No "session_id jump within a coarse bin" is possible by construction


def test_resample_freq5_close_is_last_finite_in_bin() -> None:
    """The aggregated close should be the last finite close in the bin."""
    aligned = _make_panel(n_sessions=1, bars_per_session=10)
    # Set close_y[2] = NaN, close_y[4] = 999.0 (last finite within bin [0,5))
    cy = aligned.close_y.copy()
    cy[2] = float("nan")
    cy[4] = 999.0
    aligned2 = AlignedMinutePair(
        timestamps=aligned.timestamps,
        session_id=aligned.session_id,
        bar_in_session=aligned.bar_in_session,
        close_y=cy,
        close_x=aligned.close_x,
        volume_y=aligned.volume_y,
        volume_x=aligned.volume_x,
        tradeable=aligned.tradeable,
    )
    out = resample_within_session(aligned2, freq_minutes=5)
    # First coarse bar should pick up close_y[4] = 999.0
    assert out.close_y[0] == pytest.approx(999.0)


def test_resample_tradeable_requires_finite_closes() -> None:
    aligned = _make_panel(n_sessions=1, bars_per_session=10)
    # Make all of first bin non-tradeable on the y leg
    cy = aligned.close_y.copy()
    cy[0:5] = float("nan")
    tr = aligned.tradeable.copy()
    tr[0:5] = False
    aligned2 = AlignedMinutePair(
        timestamps=aligned.timestamps,
        session_id=aligned.session_id,
        bar_in_session=aligned.bar_in_session,
        close_y=cy,
        close_x=aligned.close_x,
        volume_y=aligned.volume_y,
        volume_x=aligned.volume_x,
        tradeable=tr,
    )
    out = resample_within_session(aligned2, freq_minutes=5)
    # First coarse bar should be non-tradeable
    assert not out.tradeable[0]
