"""Tests for the intraday session calendar primitives."""

from datetime import date

import numpy as np
import pandas as pd

from apt.intraday.calendar import (
    NSE_BARS_PER_SESSION,
    bar_of_session,
    build_session_grid,
    session_segments,
)


def test_session_grid_has_375_bars_starting_at_0915() -> None:
    g = build_session_grid(date(2020, 1, 2))
    assert len(g) == NSE_BARS_PER_SESSION == 375
    assert g[0].hour == 9 and g[0].minute == 15
    assert g[-1].hour == 15 and g[-1].minute == 29
    # Continuous 1-min cadence (format-agnostic to ns vs µs resolution)
    deltas = np.unique(np.diff(g).astype("timedelta64[s]"))
    assert deltas.tolist() == [np.timedelta64(60, "s")]


def test_bar_of_session_maps_corners() -> None:
    g = build_session_grid(date(2020, 1, 2))
    b = bar_of_session(g)
    assert b[0] == 0  # 09:15 -> minute 0
    assert b[-1] == 374  # 15:29 -> minute 374
    # Monotone increasing within the session
    assert (np.diff(b) == 1).all()


def test_session_segments_handles_basic_layout() -> None:
    sids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    assert session_segments(sids) == [(0, 3), (3, 5), (5, 6)]


def test_session_segments_handles_empty_and_single() -> None:
    assert session_segments(np.array([], dtype=np.int32)) == []
    assert session_segments(np.array([0, 0, 0], dtype=np.int32)) == [(0, 3)]


def test_bar_of_session_off_session_returns_negative_or_oob() -> None:
    # 09:00 is pre-open; minute index should be < 0
    ts = pd.DatetimeIndex([pd.Timestamp("2020-01-02 09:00:00+05:30")])
    assert bar_of_session(ts)[0] < 0
