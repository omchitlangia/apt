"""NSE intraday session calendar and bar-of-session utilities.

Conventions
-----------
* Minute bars are OPEN-labeled. The bar timestamped ``09:15:00+05:30`` covers
  the interval ``[09:15, 09:16)``; the last bar of a session is
  ``15:29:00+05:30`` covering ``[15:29, 15:30)``. A full session therefore has
  exactly 375 one-minute bars (09:15..15:29 inclusive).
* The continuous trading session is ``09:15-15:30`` IST. Pre-open auction
  prints (typically ``09:00-09:15``) and any post-close prints are dropped
  upstream by the loader.
* All timestamps are tz-aware in ``Asia/Kolkata``.

Why this matters
----------------
* The bar-stamp convention gates causality: a signal computed using
  ``close[t]`` must act no earlier than bar ``t+1`` (Phase 2A `_identify_trades_and_returns`
  uses ``position[t-1] * (spread[t] - spread[t-1])`` — same convention here).
* Session boundaries gate the rolling z-window: the close→next-open move is
  an overnight gap, not a 1-min return, so the window must reset across
  sessions and never blend prior-session bars into the current session's
  rolling statistics.
"""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd

NSE_SESSION_START: time = time(9, 15)
NSE_SESSION_END: time = time(15, 29)  # last bar of the session (covers 15:29-15:30)
NSE_BARS_PER_SESSION: int = 375


def build_session_grid(session_date: date) -> pd.DatetimeIndex:
    """Canonical 375-bar timestamp grid for one NSE session.

    Returns a tz-aware ``DatetimeIndex`` in ``Asia/Kolkata`` with exactly
    :data:`NSE_BARS_PER_SESSION` minutes starting at 09:15 and ending at
    15:29 (inclusive). Useful for reindexing the loader's per-session bars
    onto a uniform axis.
    """
    start = pd.Timestamp.combine(session_date, NSE_SESSION_START).tz_localize("Asia/Kolkata")
    return pd.date_range(start=start, periods=NSE_BARS_PER_SESSION, freq="1min")


def bar_of_session(timestamps: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Minute-of-session index (0..374) for each timestamp.

    Computed as ``(hour - 9) * 60 + (minute - 15)``. Values outside the
    continuous session map outside ``[0, 375)``; callers should drop those
    rows beforehand.
    """
    ts = pd.DatetimeIndex(timestamps)
    return ((ts.hour - 9) * 60 + (ts.minute - 15)).to_numpy(dtype=np.int32)


def session_segments(session_ids: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(start, end_exclusive)]`` slices for each contiguous session.

    ``session_ids`` is a 1-D integer array assigning every bar to a session
    (e.g. a dense rank of the bar's date). Boundaries are detected via
    diff; the returned ranges are non-overlapping and cover every bar.

    Example::

        sids = np.array([0, 0, 0, 1, 1, 2])
        session_segments(sids) == [(0, 3), (3, 5), (5, 6)]
    """
    if session_ids.size == 0:
        return []
    s = np.asarray(session_ids)
    # change-points where the session id increments
    cp = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate(([0], cp))
    ends = np.concatenate((cp, [s.size]))
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]
