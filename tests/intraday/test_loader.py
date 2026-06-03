"""Smoke + correctness tests for the Epps-safe minute pair loader.

These tests synthesize a tiny on-disk parquet layout to avoid coupling
to ``/Data6/db/`` or the project's actual minute panel.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from apt.intraday.loader import load_minute_pair


def _write_symbol(root: Path, symbol: str, bars: pd.DataFrame) -> None:
    """Write a single-year parquet under ``symbol=<S>/year=<Y>/data.parquet``."""
    year = bars["timestamp"].iloc[0].year
    out_dir = root / f"symbol={symbol}" / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out_dir / "data.parquet")


def _bars(timestamps: pd.DatetimeIndex, closes: np.ndarray, vols: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "date": timestamps.date,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": vols.astype(np.int64),
            "partial_day_flag": False,
        }
    )


def test_load_minute_pair_marks_no_trade_minutes_non_tradeable(tmp_path: Path) -> None:
    # 3 bars at 09:15, 09:16, 09:17 on the same session.
    ts = pd.date_range("2020-01-02 09:15", periods=3, freq="1min", tz="Asia/Kolkata")
    # Leg Y trades every minute; leg X skips the middle bar (no volume).
    y_bars = _bars(ts, np.array([100.0, 101.0, 102.0]), np.array([10, 10, 10]))
    # Leg X is missing the middle bar entirely (asynchronous print)
    x_ts = pd.DatetimeIndex([ts[0], ts[2]])
    x_bars = _bars(x_ts, np.array([50.0, 51.0]), np.array([5, 5]))
    _write_symbol(tmp_path, "YYY", y_bars)
    _write_symbol(tmp_path, "XXX", x_bars)

    res = load_minute_pair("YYY", "XXX", date(2020, 1, 2), date(2020, 1, 2), root=tmp_path)
    assert res.n_bars == 3
    # The middle bar must be marked non-tradeable; flanking bars tradeable.
    assert res.tradeable.tolist() == [True, False, True]
    # close_x at the middle bar must be NaN — never forward-filled.
    assert np.isnan(res.close_x[1])
    # session_id all zero (one session)
    assert (res.session_id == 0).all()
    # bar_in_session = [0, 1, 2]
    assert res.bar_in_session.tolist() == [0, 1, 2]


def test_load_minute_pair_skips_post_close_prints(tmp_path: Path) -> None:
    # 09:14 (pre-open) + 09:15 (session) + 15:30 (post-close, should drop)
    ts = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-02 09:14:00", tz="Asia/Kolkata"),
            pd.Timestamp("2020-01-02 09:15:00", tz="Asia/Kolkata"),
            pd.Timestamp("2020-01-02 15:30:00", tz="Asia/Kolkata"),
        ]
    )
    bars = _bars(ts, np.array([100.0, 101.0, 102.0]), np.array([1, 2, 3]))
    _write_symbol(tmp_path, "YYY", bars)
    _write_symbol(tmp_path, "XXX", bars)
    res = load_minute_pair("YYY", "XXX", date(2020, 1, 2), date(2020, 1, 2), root=tmp_path)
    # Only the 09:15 bar should survive (09:14 is pre-open, 15:30 is post-close).
    assert res.n_bars == 1
    assert res.timestamps[0].time().hour == 9
    assert res.timestamps[0].time().minute == 15
