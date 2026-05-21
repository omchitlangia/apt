"""Corporate actions: yfinance cache, residual-jump audit, dividend-flavor
classifier, and validated daily passthrough.

Day-3 of the project verifies — but does NOT re-apply — split/bonus adjustments.
The raw daily and minute datasets are already split-/bonus-adjusted back to
2003; this module:

1. Caches yfinance ``splits`` and ``dividends`` for the universe.
2. Audits each reported split ex-date for a residual unadjusted jump.
3. Classifies high-dividend names as ``split_only`` vs ``total_return`` by
   comparing daily_raw close against yfinance ``Close`` and ``Adj Close``.
4. Writes a validated passthrough of daily_raw → daily_adjusted.

Functions here are pure where practical. Network I/O is isolated to the
``fetch_*`` helpers so they can be stubbed in tests.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger

# Schema written to data/interim/corporate_actions.parquet (long format).
_ACTIONS_SCHEMA: dict[str, type[pl.DataType]] = {
    "symbol": pl.Utf8,
    "action_type": pl.Utf8,  # "split" or "dividend"
    "ex_date": pl.Date,
    "value": pl.Float64,
}

# Five high-dividend NSE names used for the flavor diagnosis in Day-3 step 3.
DIVIDEND_FLAVOR_PROBES: tuple[str, ...] = (
    "ITC",
    "COALINDIA",
    "HINDPETRO",
    "NTPC",
    "POWERGRID",
)


# ---------------------------------------------------------------------------
# Step 1 — fetch + cache yfinance corporate actions
# ---------------------------------------------------------------------------


def _empty_actions_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [], "action_type": [], "ex_date": [], "value": []},
        schema=_ACTIONS_SCHEMA,
    )


def _yf_ticker_actions(symbol: str):  # pragma: no cover — thin wrapper for stubbing
    """Return the raw ``Ticker(symbol+".NS").actions`` pandas DataFrame.

    Isolated so tests can monkey-patch it without hitting the network.
    """
    import yfinance as yf  # local import — slow + optional in some envs

    return yf.Ticker(f"{symbol}.NS").actions


def fetch_corporate_actions_for_symbol(
    symbol: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    ticker_fn: Callable[[str], object] | None = None,
) -> dict:
    """Fetch splits + dividends for one NSE symbol via yfinance.

    Returns a dict with::

        {
            "symbol": str,
            "splits":   [(ex_date, ratio), ...],
            "dividends":[(ex_date, amount), ...],
            "error":    None | str,
        }

    On failure, ``error`` is a string and the two lists are empty. The
    coverage table built by ``fetch_all_corporate_actions`` uses ``error``
    to distinguish "no actions" (success) from "fetch failed".
    """
    fetch = ticker_fn or _yf_ticker_actions
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            actions = fetch(symbol)
        except Exception as exc:  # noqa: BLE001 — yfinance raises a zoo
            last_exc = exc
            sleep_for = base_delay * (2**attempt) + random.random() * 0.25
            logger.debug(
                "{}: fetch attempt {} failed ({}); sleeping {:.2f}s",
                symbol,
                attempt + 1,
                exc.__class__.__name__,
                sleep_for,
            )
            time.sleep(sleep_for)
            continue

        # actions: pandas DataFrame with "Dividends" and "Stock Splits" columns,
        # indexed by tz-aware DatetimeIndex. May be empty.
        if actions is None or actions.empty:
            return {"symbol": symbol, "splits": [], "dividends": [], "error": None}

        splits: list[tuple[date, float]] = []
        dividends: list[tuple[date, float]] = []
        for ts, row in actions.iterrows():
            ex_date = ts.date() if hasattr(ts, "date") else ts
            split_val = float(row.get("Stock Splits", 0.0) or 0.0)
            div_val = float(row.get("Dividends", 0.0) or 0.0)
            if split_val and not math.isnan(split_val):
                splits.append((ex_date, split_val))
            if div_val and not math.isnan(div_val):
                dividends.append((ex_date, div_val))
        return {"symbol": symbol, "splits": splits, "dividends": dividends, "error": None}

    logger.warning("{}: all {} fetch attempts failed: {}", symbol, max_retries, last_exc)
    return {
        "symbol": symbol,
        "splits": [],
        "dividends": [],
        "error": f"{last_exc.__class__.__name__}: {last_exc}" if last_exc else "unknown",
    }


def _result_to_long(result: dict) -> pl.DataFrame:
    """Flatten one fetch result dict into a long-format polars frame."""
    rows: list[dict] = []
    for ex_date, ratio in result["splits"]:
        rows.append(
            {
                "symbol": result["symbol"],
                "action_type": "split",
                "ex_date": ex_date,
                "value": float(ratio),
            }
        )
    for ex_date, amount in result["dividends"]:
        rows.append(
            {
                "symbol": result["symbol"],
                "action_type": "dividend",
                "ex_date": ex_date,
                "value": float(amount),
            }
        )
    if not rows:
        return _empty_actions_frame()
    return pl.DataFrame(rows, schema=_ACTIONS_SCHEMA)


def fetch_all_corporate_actions(
    symbols: list[str],
    *,
    n_jobs: int = 8,
    max_retries: int = 3,
    ticker_fn: Callable[[str], object] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch corporate actions for every symbol; return (actions, coverage).

    Args:
        symbols: list of NSE root symbols (no ".NS").
        n_jobs: parallel HTTP workers (yfinance is rate-sensitive — keep modest).
        max_retries: per-symbol retry budget.
        ticker_fn: override for ``_yf_ticker_actions`` (used in tests).

    Returns:
        ``actions``: long-format DataFrame with ``_ACTIONS_SCHEMA``.
        ``coverage``: per-symbol metadata: ``symbol, status, error,
        n_splits, n_dividends``.
    """
    from apt.utils.parallel import parallel_map

    def _one(sym: str) -> dict:
        return fetch_corporate_actions_for_symbol(sym, max_retries=max_retries, ticker_fn=ticker_fn)

    results = parallel_map(
        _one,
        symbols,
        n_jobs=n_jobs,
        desc="Fetching yfinance corporate actions",
        prefer="threads",
    )

    frames = [_result_to_long(r) for r in results]
    actions = pl.concat(frames, how="vertical") if frames else _empty_actions_frame()

    coverage = pl.DataFrame(
        {
            "symbol": [r["symbol"] for r in results],
            "status": ["ok" if r["error"] is None else "failed" for r in results],
            "error": [r["error"] or "" for r in results],
            "n_splits": [len(r["splits"]) for r in results],
            "n_dividends": [len(r["dividends"]) for r in results],
        }
    )
    return actions, coverage


def write_corporate_actions_cache(actions: pl.DataFrame, path: Path) -> None:
    """Write the actions long-frame to ``path`` (parquet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    actions.sort(["symbol", "ex_date", "action_type"]).write_parquet(path, use_pyarrow=True)


# ---------------------------------------------------------------------------
# Step 2 — residual split-jump audit
# ---------------------------------------------------------------------------


def audit_split_residuals(
    daily: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    ratio_tol: float = 0.05,
) -> pl.DataFrame:
    """Flag splits whose unadjusted ex-date jump is still present in ``daily``.

    For each ``(symbol, ex_date, value)`` split row, looks up ``close`` on
    ``ex_date`` and on the prior trading day for that symbol within ``daily``.
    ``observed_ratio = close / prev_close``. If the data were unadjusted the
    observed ratio would equal ``1/value`` (e.g. 0.5 for a 2:1 split). A
    discrepancy of less than ``ratio_tol`` flags a residual jump.

    Splits whose ex-date is outside the daily window, or whose prior trading
    day is unavailable, are skipped (status reported on the row).

    Returns:
        DataFrame with columns ``symbol, ex_date, split_ratio, prev_close,
        close, observed_ratio, expected_unadjusted_ratio, residual_distance,
        status``. Only rows with ``status="residual_jump"`` are failures —
        the caller may filter accordingly.
    """
    splits = actions.filter(pl.col("action_type") == "split").rename(
        {"ex_date": "date", "value": "split_ratio"}
    )
    if splits.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "ex_date": pl.Date,
                "split_ratio": pl.Float64,
                "prev_close": pl.Float64,
                "close": pl.Float64,
                "observed_ratio": pl.Float64,
                "expected_unadjusted_ratio": pl.Float64,
                "residual_distance": pl.Float64,
                "status": pl.Utf8,
            }
        )

    enriched = daily.sort(["symbol", "date"]).with_columns(
        pl.col("close").shift(1).over("symbol").alias("prev_close")
    )

    joined = splits.join(
        enriched.select(["symbol", "date", "close", "prev_close"]),
        on=["symbol", "date"],
        how="left",
    )

    joined = joined.with_columns(
        (pl.col("close") / pl.col("prev_close")).alias("observed_ratio"),
        (1.0 / pl.col("split_ratio")).alias("expected_unadjusted_ratio"),
    ).with_columns(
        (pl.col("observed_ratio") - pl.col("expected_unadjusted_ratio"))
        .abs()
        .alias("residual_distance"),
    )

    # A split ratio very close to 1.0 (e.g. yfinance's 1.0008 for GODREJIND)
    # gives an expected unadjusted ratio of ~1.0, which can't be distinguished
    # from normal day-to-day price movement — the test is degenerate.
    joined = joined.with_columns(
        pl.when(pl.col("close").is_null())
        .then(pl.lit("out_of_window"))
        .when(pl.col("prev_close").is_null())
        .then(pl.lit("no_prev_close"))
        .when((pl.col("expected_unadjusted_ratio") - 1.0).abs() < ratio_tol)
        .then(pl.lit("trivial_split"))
        .when(pl.col("residual_distance") < ratio_tol)
        .then(pl.lit("residual_jump"))
        .otherwise(pl.lit("ok"))
        .alias("status")
    )

    return joined.rename({"date": "ex_date"}).select(
        [
            "symbol",
            "ex_date",
            "split_ratio",
            "prev_close",
            "close",
            "observed_ratio",
            "expected_unadjusted_ratio",
            "residual_distance",
            "status",
        ]
    )


# ---------------------------------------------------------------------------
# Step 3 — dividend-flavor classifier
# ---------------------------------------------------------------------------


def _yf_history(
    symbol: str, start: date, end: date
):  # pragma: no cover — thin wrapper for stubbing
    """Return yfinance daily history with both Close and Adj Close (pandas)."""
    import yfinance as yf

    return yf.download(
        f"{symbol}.NS",
        start=start.isoformat(),
        end=(end.isoformat()),
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
        threads=False,
    )


def _coefficient_of_variation(s: pl.Series) -> float:
    """std / mean of a strictly positive series. Returns inf if undefined."""
    s = s.drop_nulls()
    if s.len() == 0:
        return float("inf")
    m = s.mean()
    if m is None or m == 0:
        return float("inf")
    sd = s.std()
    if sd is None:
        return float("inf")
    return abs(sd / m)


def classify_adjustment_flavor(
    symbol: str,
    daily: pl.DataFrame,
    *,
    history_fn: Callable[[str, date, date], object] | None = None,
) -> dict:
    """Compare daily_raw close against yfinance Close vs Adj Close.

    Computes ``cv(daily.close / yf.Close)`` and ``cv(daily.close / yf.AdjClose)``
    over the overlapping date set. The lower-CV ratio identifies which flavor
    of adjustment is encoded in ``daily``:

      * lower CV against ``Close`` (auto_adjust=False) → split/bonus-only
      * lower CV against ``Adj Close``                  → total-return

    The function is observational — no decision is made by the classifier.

    Returns a dict with: symbol, n_overlap, cv_vs_close, cv_vs_adj_close,
    verdict, confidence (one of ``high|medium|low``), error.
    """
    get_history = history_fn or _yf_history
    sym_daily = daily.filter(pl.col("symbol") == symbol).select(["date", "close"]).sort("date")
    if sym_daily.is_empty():
        return {
            "symbol": symbol,
            "n_overlap": 0,
            "cv_vs_close": float("nan"),
            "cv_vs_adj_close": float("nan"),
            "verdict": "no_data",
            "confidence": "low",
            "error": "symbol not in daily",
        }
    start = sym_daily["date"].min()
    end_obj = sym_daily["date"].max()
    # yfinance end is exclusive — bump by one day to include the last bar.
    end = (
        end_obj if not hasattr(end_obj, "toordinal") else date.fromordinal(end_obj.toordinal() + 1)
    )
    try:
        yf_df = get_history(symbol, start, end)
    except Exception as exc:  # noqa: BLE001
        logger.warning("{}: yfinance history fetch failed: {}", symbol, exc)
        return {
            "symbol": symbol,
            "n_overlap": 0,
            "cv_vs_close": float("nan"),
            "cv_vs_adj_close": float("nan"),
            "verdict": "fetch_failed",
            "confidence": "low",
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    if yf_df is None or len(yf_df) == 0:
        return {
            "symbol": symbol,
            "n_overlap": 0,
            "cv_vs_close": float("nan"),
            "cv_vs_adj_close": float("nan"),
            "verdict": "no_yfinance_data",
            "confidence": "low",
            "error": "empty yfinance history",
        }

    # Convert yf pandas → polars, aligning on calendar date.
    yf_df = yf_df.reset_index()
    yf_pl = pl.from_pandas(yf_df).select(
        pl.col("Date").dt.date().alias("date"),
        pl.col("Close").cast(pl.Float64).alias("yf_close"),
        pl.col("Adj Close").cast(pl.Float64).alias("yf_adj_close"),
    )

    merged = (
        sym_daily.join(yf_pl, on="date", how="inner")
        .filter(
            pl.col("close").is_not_null()
            & pl.col("yf_close").is_not_null()
            & pl.col("yf_adj_close").is_not_null()
            & (pl.col("yf_close") > 0)
            & (pl.col("yf_adj_close") > 0)
        )
        .with_columns(
            (pl.col("close") / pl.col("yf_close")).alias("ratio_vs_close"),
            (pl.col("close") / pl.col("yf_adj_close")).alias("ratio_vs_adj_close"),
        )
    )
    n = merged.height
    if n < 20:
        return {
            "symbol": symbol,
            "n_overlap": n,
            "cv_vs_close": float("nan"),
            "cv_vs_adj_close": float("nan"),
            "verdict": "insufficient_overlap",
            "confidence": "low",
            "error": f"only {n} overlapping rows",
        }

    cv_close = _coefficient_of_variation(merged["ratio_vs_close"])
    cv_adj = _coefficient_of_variation(merged["ratio_vs_adj_close"])
    verdict = "split_only" if cv_close < cv_adj else "total_return"
    # Confidence: how big is the gap relative to the smaller CV?
    smaller = min(cv_close, cv_adj)
    gap = abs(cv_close - cv_adj)
    if smaller > 0 and gap / smaller > 5:
        confidence = "high"
    elif smaller > 0 and gap / smaller > 1:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "symbol": symbol,
        "n_overlap": n,
        "cv_vs_close": float(cv_close),
        "cv_vs_adj_close": float(cv_adj),
        "verdict": verdict,
        "confidence": confidence,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Step 4 — validated passthrough
# ---------------------------------------------------------------------------


def passthrough_daily_adjusted(daily_raw_path: Path, output_path: Path) -> dict:
    """Copy daily_raw → daily_adjusted unchanged, with validation.

    Reads the source frame, asserts schema & non-emptiness, writes the same
    frame to ``output_path``, then reloads the output and confirms equal
    row counts and symbol sets. Returns a summary dict.
    """
    df = pl.read_parquet(daily_raw_path)
    if df.is_empty():
        raise RuntimeError(f"{daily_raw_path}: empty source frame, cannot passthrough")
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{daily_raw_path}: missing columns {missing}")

    n_in = df.height
    n_sym_in = df["symbol"].n_unique()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, use_pyarrow=True)

    loaded = pl.read_parquet(output_path)
    n_out = loaded.height
    n_sym_out = loaded["symbol"].n_unique()

    if n_in != n_out:
        raise RuntimeError(f"passthrough row mismatch: source {n_in:,} vs output {n_out:,}")
    if n_sym_in != n_sym_out:
        raise RuntimeError(f"passthrough symbol mismatch: source {n_sym_in} vs output {n_sym_out}")

    logger.info(
        "Passthrough complete (no re-adjustment): {} rows / {} symbols -> {}",
        n_out,
        n_sym_out,
        output_path,
    )

    return {
        "rows": int(n_out),
        "symbols": int(n_sym_out),
        "min_date": loaded["date"].min(),
        "max_date": loaded["date"].max(),
    }
