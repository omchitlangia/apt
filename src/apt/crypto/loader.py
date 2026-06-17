"""Binance 1-minute kline loader + minimal crypto cleaning (Section 6).

File convention: ``<SYMBOL>-1m-<YYYY>-<MM>.csv`` under :data:`CRYPTO_ROOT`,
headerless 12-column Binance klines, ``open_time`` in **UTC milliseconds**.

MINIMAL cleaning (``clean_minute_minimal``) only — the NSE seven-rule cascade
(corporate-action back-adjust, session calendar, etc.) does NOT transfer:
crypto has no splits/dividends and trades 24/7. What we DO apply:

1. drop exact-duplicate ``open_time`` rows (keep last),
2. drop non-positive / non-finite OHLC,
3. drop zero-volume bars (untradeable).

[TODO data] still missing for parity with NSE: outlier/fat-finger screen,
exchange-halt handling, cross-venue consolidation, a turnover-stability gate.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

CRYPTO_ROOT = Path("/home/om/data_combined")

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "n_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def discover_symbols(root: Path = CRYPTO_ROOT) -> list[str]:
    """All distinct symbols present (prefix before ``-1m-``)."""
    syms = {p.name.split("-1m-")[0] for p in root.glob("*-1m-*.csv")}
    return sorted(syms)


def clean_minute_minimal(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the minimal crypto cleaning rules (see module docstring)."""
    return (
        df.unique(subset=["open_time"], keep="last")
        .filter(
            (pl.col("close") > 0)
            & (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & pl.col("close").is_finite()
            & (pl.col("volume") > 0)
        )
        .sort("open_time")
    )


def _read_month(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, has_header=False, new_columns=BINANCE_KLINE_COLUMNS)
    return df.select(["open_time", "open", "high", "low", "close", "volume", "quote_volume"])


def load_symbol_daily(symbol: str, root: Path = CRYPTO_ROOT) -> pl.DataFrame:
    """Daily (UTC) OHLC-close + summed volume for one symbol, minimally cleaned.

    Returns columns ``[date, close, quote_volume, n_bars]``; the daily close is
    the last 1-minute close of the UTC day.
    """
    files = sorted(root.glob(f"{symbol}-1m-*.csv"))
    if not files:
        return pl.DataFrame()
    frames = [clean_minute_minimal(_read_month(f)) for f in files]
    df = pl.concat(frames).unique(subset=["open_time"], keep="last").sort("open_time")
    # Binance switched kline open_time from MILLISECONDS to MICROSECONDS in
    # 2025; the panel mixes both. Normalize to microseconds by magnitude
    # (ms ≈ 1e12–1e13, µs ≈ 1e15–1e16; threshold 1e14 separates cleanly).
    df = df.with_columns(
        pl.when(pl.col("open_time") > 1e14)
        .then(pl.col("open_time"))
        .otherwise(pl.col("open_time") * 1000)
        .cast(pl.Datetime("us"))
        .dt.date()
        .alias("date")
    )
    return (
        df.group_by("date")
        .agg(
            close=pl.col("close").last(),
            quote_volume=pl.col("quote_volume").sum(),
            n_bars=pl.len(),
        )
        .sort("date")
    )


def build_daily_panel(symbols: list[str] | None = None, root: Path = CRYPTO_ROOT) -> pl.DataFrame:
    """Long-format daily panel ``[symbol, date, close, quote_volume, n_bars]``."""
    symbols = symbols or discover_symbols(root)
    out = []
    for sym in symbols:
        d = load_symbol_daily(sym, root)
        if d.height == 0:
            continue
        out.append(d.with_columns(pl.lit(sym).alias("symbol")))
    if not out:
        return pl.DataFrame()
    return pl.concat(out).select(["symbol", "date", "close", "quote_volume", "n_bars"])


__all__ = [
    "BINANCE_KLINE_COLUMNS",
    "CRYPTO_ROOT",
    "build_daily_panel",
    "clean_minute_minimal",
    "discover_symbols",
    "load_symbol_daily",
]
