"""Unit tests for the crypto loader (Section 6) — minimal cleaning rules."""

from __future__ import annotations

import polars as pl

from apt.crypto import clean_minute_minimal
from apt.crypto.loader import BINANCE_KLINE_COLUMNS


def _row(open_time, o, h, l, c, v, qv):  # noqa: E741
    return {
        "open_time": open_time,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "quote_volume": qv,
    }


def test_minimal_cleaning_rules():
    df = pl.DataFrame(
        [
            _row(1, 10, 11, 9, 10.5, 5.0, 100.0),  # ok
            _row(1, 10, 11, 9, 10.6, 5.0, 100.0),  # dup open_time -> keep last
            _row(2, 10, 11, 9, 0.0, 5.0, 100.0),  # non-positive close -> drop
            _row(3, 10, 11, 9, 10.0, 0.0, 0.0),  # zero volume -> drop
            _row(4, 10, 11, 9, 10.0, 2.0, 50.0),  # ok
        ]
    )
    out = clean_minute_minimal(df)
    assert out.height == 2  # rows at open_time 1 (last) and 4
    assert out.filter(pl.col("open_time") == 1)["close"].item() == 10.6
    assert set(out["open_time"].to_list()) == {1, 4}


def test_kline_column_names_count():
    assert len(BINANCE_KLINE_COLUMNS) == 12
    assert BINANCE_KLINE_COLUMNS[0] == "open_time"
    assert BINANCE_KLINE_COLUMNS[4] == "close"
    assert BINANCE_KLINE_COLUMNS[7] == "quote_volume"
