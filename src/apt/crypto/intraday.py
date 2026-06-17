"""Crypto intraday resampling for the A10 adaptive experiment (Section 6 / Part 2).

Crypto trades 24/7, so a "session" is defined as a **UTC calendar day**
(Regime A squares off at the UTC-day boundary; the per-session μ-filter updates
once per UTC day — the direct analogue of an NSE session). Resamples 1-minute
klines to {1,5,15}-minute bars aligned within the UTC day, and aligns a pair's
two legs onto common bars.

In-process caches (``_MINUTE_CACHE`` / ``_RESAMPLED_CACHE``) avoid re-reading
the (large) 1-minute files per pair/freq.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from apt.crypto.loader import CRYPTO_ROOT, _read_month, clean_minute_minimal

_MINUTE_CACHE: dict[str, pl.DataFrame] = {}
_RESAMPLED_CACHE: dict[tuple[str, int], pd.DataFrame] = {}

# Optional (start_ym, end_ym) "YYYY-MM" bound on which monthly files are read
# — set by the driver to cap memory to the experiment window.
_WINDOW_MONTHS: tuple[str, str] | None = None


def set_window_months(start_ym: str, end_ym: str) -> None:
    """Restrict file loading to months within [start_ym, end_ym] inclusive."""
    global _WINDOW_MONTHS
    _WINDOW_MONTHS = (start_ym, end_ym)
    _MINUTE_CACHE.clear()
    _RESAMPLED_CACHE.clear()


def _files_in_window(symbol: str, root) -> list:
    files = sorted(root.glob(f"{symbol}-1m-*.csv"))
    if _WINDOW_MONTHS is None:
        return files
    lo, hi = _WINDOW_MONTHS
    return [f for f in files if lo <= f.name.split("-1m-")[1][:7] <= hi]


def _normalize_open_time(df: pl.DataFrame) -> pl.DataFrame:
    """Binance switched open_time ms->µs in 2025; normalize to a UTC datetime."""
    return df.with_columns(
        pl.when(pl.col("open_time") > 1e14)
        .then(pl.col("open_time"))
        .otherwise(pl.col("open_time") * 1000)
        .cast(pl.Datetime("us"))
        .alias("ts")
    )


def load_symbol_minute(symbol: str, root=CRYPTO_ROOT) -> pl.DataFrame:
    """Cleaned 1-minute bars (full history) with a UTC ``ts`` column, cached."""
    if symbol in _MINUTE_CACHE:
        return _MINUTE_CACHE[symbol]
    files = _files_in_window(symbol, root)
    frames = [clean_minute_minimal(_read_month(f)) for f in files]
    df = pl.concat(frames).unique(subset=["open_time"], keep="last").sort("open_time")
    df = _normalize_open_time(df).select(["ts", "close"])
    _MINUTE_CACHE[symbol] = df
    return df


def resample_symbol(symbol: str, freq_min: int, root=CRYPTO_ROOT) -> pd.DataFrame:
    """Resample to ``freq_min`` bars within the UTC day. Cached per (symbol,freq).

    Returns a pandas frame indexed by bar timestamp with columns
    ``[close, session]`` where ``session`` is the UTC date.
    """
    key = (symbol, freq_min)
    if key in _RESAMPLED_CACHE:
        return _RESAMPLED_CACHE[key]
    m = load_symbol_minute(symbol, root)
    if freq_min == 1:
        r = m.with_columns(pl.col("ts").dt.date().alias("session"))
    else:
        r = (
            m.with_columns(pl.col("ts").dt.truncate(f"{freq_min}m").alias("bar"))
            .group_by("bar")
            .agg(close=pl.col("close").last())
            .sort("bar")
            .rename({"bar": "ts"})
            .with_columns(pl.col("ts").dt.date().alias("session"))
        )
    pdf = r.to_pandas().set_index("ts")
    _RESAMPLED_CACHE[key] = pdf
    return pdf


@dataclass(frozen=True)
class CryptoPairBars:
    timestamps: pd.DatetimeIndex
    session_id: np.ndarray  # dense-rank UTC-day session index
    log_y: np.ndarray
    log_x: np.ndarray
    tradeable: np.ndarray


def load_pair_resampled(
    y_sym: str, x_sym: str, start: date, end: date, freq_min: int, root=CRYPTO_ROOT
) -> CryptoPairBars | None:
    """Aligned resampled bars for a pair over ``[start, end]`` (UTC days)."""
    ry = resample_symbol(y_sym, freq_min, root)
    rx = resample_symbol(x_sym, freq_min, root)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    ry = ry[(ry.index >= lo) & (ry.index < hi)]
    rx = rx[(rx.index >= lo) & (rx.index < hi)]
    common = ry.index.intersection(rx.index)
    if len(common) < 100:
        return None
    ry, rx = ry.loc[common], rx.loc[common]
    sessions = pd.Index(ry.session.values)
    _, sid = np.unique(sessions, return_inverse=True)
    return CryptoPairBars(
        timestamps=pd.DatetimeIndex(common),
        session_id=sid.astype(np.int32),
        log_y=np.log(ry.close.to_numpy()),
        log_x=np.log(rx.close.to_numpy()),
        tradeable=np.ones(len(common), dtype=bool),
    )


__all__ = ["CryptoPairBars", "load_pair_resampled", "load_symbol_minute", "resample_symbol"]
