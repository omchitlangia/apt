"""Load and align minute bars for a pair on a common intraday grid.

The minute panel is partitioned hive-style at ``data/interim/minute_raw``:

    symbol=<SYM>/year=<YYYY>/data.parquet

Each row is one OPEN-labeled bar (see :mod:`apt.intraday.calendar`):

    timestamp (tz=Asia/Kolkata), date, open, high, low, close, volume,
    partial_day_flag

Epps-safe alignment
-------------------
The Epps effect is the well-known decay of measured correlation between two
assets as sampling frequency increases when their trades arrive
asynchronously. Naively forward-filling no-trade minutes fabricates
synchronous prices and corrupts the cointegration spread. So:

* Only bars present in BOTH legs' raw feeds for that minute are tradeable.
* Where either leg has no trade in a given minute, the bar is marked
  ``tradeable=False`` and the spread/z/return is NaN there.
* This is propagated downstream by carrying state through NaN bars in
  :func:`apt.signals.spread.generate_signals` — no position flip on a
  no-info bar.

The loader does NOT forward-fill close prices to keep the spread alive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from apt.intraday.calendar import (
    NSE_BARS_PER_SESSION,
    NSE_SESSION_END,
    NSE_SESSION_START,
    bar_of_session,
)


@dataclass(frozen=True)
class AlignedMinutePair:
    """Output of :func:`load_minute_pair`.

    Arrays are length-aligned. ``tradeable`` is True iff BOTH legs had a
    trade in that minute (volume > 0 AND a finite close). Non-tradeable
    bars carry NaN in ``close_y`` and ``close_x`` rather than a forward-
    filled price.
    """

    timestamps: pd.DatetimeIndex  # tz-aware, length N
    session_id: np.ndarray  # int32, dense rank 0..S-1 over distinct sessions
    bar_in_session: np.ndarray  # int32 in [0, 375)
    close_y: np.ndarray  # float64, NaN where leg-y untraded
    close_x: np.ndarray  # float64, NaN where leg-x untraded
    volume_y: np.ndarray  # int64
    volume_x: np.ndarray  # int64
    tradeable: np.ndarray  # bool — both legs have a real trade this minute

    @property
    def n_bars(self) -> int:
        return int(self.close_y.size)

    @property
    def n_sessions(self) -> int:
        return int(self.session_id.max() + 1) if self.session_id.size else 0


def _years_overlapping(start: date, end: date) -> list[int]:
    return list(range(start.year, end.year + 1))


def _load_symbol(
    symbol: str,
    start: date,
    end: date,
    root: Path,
) -> pd.DataFrame:
    """Load a single symbol's minute bars for the requested calendar window.

    Reads only the year partitions overlapping ``[start, end]``, filters to
    the continuous session, drops the (very rare) duplicate timestamps that
    appear in the raw feed, and indexes by ``timestamp`` (tz-aware IST).
    """
    base = root / f"symbol={symbol}"
    if not base.is_dir():
        return pd.DataFrame()
    frames = []
    for yr in _years_overlapping(start, end):
        f = base / f"year={yr}" / "data.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Drop pre-open / post-close prints
    t = df["timestamp"]
    in_sess = (t.dt.time >= NSE_SESSION_START) & (t.dt.time <= NSE_SESSION_END)
    df = df.loc[in_sess]
    # Restrict by calendar date
    df = df.loc[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)]
    if df.empty:
        return df
    # Defensively drop duplicate timestamps (keep first)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df[["timestamp", "close", "volume"]]


def load_minute_pair(
    y_sym: str,
    x_sym: str,
    start: date,
    end: date,
    *,
    root: Path | str = Path("data/interim/minute_raw"),
) -> AlignedMinutePair:
    """Load both legs and align onto a common intraday grid for ``[start, end]``.

    Parameters
    ----------
    y_sym, x_sym
        Pair-leg symbols (``y`` is the OLS dependent leg from Phase 2A).
    start, end
        Inclusive calendar dates. Only bars within the NSE continuous
        session (``09:15..15:29``) are kept.
    root
        Directory containing ``symbol=<SYM>/year=<YYYY>/data.parquet``.

    Returns
    -------
    AlignedMinutePair
        Bars on the union timeline of trading sessions present in EITHER
        leg. Bars where either leg lacks a real trade have ``tradeable=False``
        and NaN closes — never a forward-filled price.
    """
    root = Path(root)
    df_y = _load_symbol(y_sym, start, end, root)
    df_x = _load_symbol(x_sym, start, end, root)
    if df_y.empty or df_x.empty:
        empty_ts = pd.DatetimeIndex([], tz="Asia/Kolkata")
        z = np.zeros(0)
        return AlignedMinutePair(
            timestamps=empty_ts,
            session_id=np.zeros(0, dtype=np.int32),
            bar_in_session=np.zeros(0, dtype=np.int32),
            close_y=z.copy(),
            close_x=z.copy(),
            volume_y=np.zeros(0, dtype=np.int64),
            volume_x=np.zeros(0, dtype=np.int64),
            tradeable=np.zeros(0, dtype=bool),
        )

    # Build the union grid: every minute that appears in EITHER leg's session-
    # filtered feed. Sessions where both legs are entirely absent never appear.
    union_ts = pd.DatetimeIndex(
        np.union1d(df_y["timestamp"].to_numpy(), df_x["timestamp"].to_numpy())
    )
    # tz info is preserved by numpy's datetime64[ns] on tz-naive views — we have
    # to re-localize the result.
    if union_ts.tz is None:
        union_ts = union_ts.tz_localize("UTC").tz_convert("Asia/Kolkata")

    # Map each leg's prints onto the union grid via index alignment.
    y = df_y.set_index("timestamp").reindex(union_ts)
    x = df_x.set_index("timestamp").reindex(union_ts)

    close_y = y["close"].to_numpy(dtype=float)
    close_x = x["close"].to_numpy(dtype=float)
    vol_y = np.nan_to_num(y["volume"].to_numpy(dtype=float), nan=0.0).astype(np.int64)
    vol_x = np.nan_to_num(x["volume"].to_numpy(dtype=float), nan=0.0).astype(np.int64)

    # tradeable iff both legs have a real trade (finite close AND positive volume)
    tradeable = np.isfinite(close_y) & np.isfinite(close_x) & (vol_y > 0) & (vol_x > 0)

    # session_id = dense rank of session date over the union grid; tracks
    # contiguous in-session bars.
    session_date = union_ts.date  # numpy array of date objects
    unique_dates, session_id = np.unique(session_date, return_inverse=True)
    bar_idx = bar_of_session(union_ts)

    return AlignedMinutePair(
        timestamps=union_ts,
        session_id=session_id.astype(np.int32),
        bar_in_session=bar_idx,
        close_y=close_y,
        close_x=close_x,
        volume_y=vol_y,
        volume_x=vol_x,
        tradeable=tradeable,
    )


__all__ = ["AlignedMinutePair", "load_minute_pair", "NSE_BARS_PER_SESSION"]
