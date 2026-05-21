"""Minute OHLCV ingestion from /Data6/db/minute/.

One CSV per symbol (``{SYMBOL}_minute-data.csv``). Each worker parses a single
file and writes its own hive partitions (``symbol=SYM/year=YYYY/data.parquet``)
under the output root — no shared state, no write contention.

The full NSE session is 375 minute bars (09:15–15:29 IST inclusive). Short days
are flagged via ``partial_day_flag`` but retained. Specific dates (e.g. the
trailing 2021-06-24) can be excluded entirely via ``drop_dates``.
"""

from __future__ import annotations

import re
import shutil
from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger

# Bars in a full regular NSE equity session (09:15–15:29 inclusive, 1-min bars).
FULL_SESSION_BARS: int = 375

_FILENAME_RE = re.compile(r"^(?P<symbol>[A-Z0-9&\-]+)_minute-data\.csv$")

_MINUTE_SCHEMA = {
    "date": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}

# Output columns (in order), excluding the hive-partition columns symbol/year.
_DATA_COLS = [
    "timestamp",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "partial_day_flag",
]


def _symbol_from_filename(name: str) -> str | None:
    m = _FILENAME_RE.match(name)
    return m.group("symbol") if m else None


def discover_minute_csvs(raw_dir: Path) -> list[Path]:
    """Return sorted list of per-symbol minute CSV paths under ``raw_dir``."""
    found: list[Path] = []
    for p in sorted(raw_dir.glob("*_minute-data.csv")):
        sym = _symbol_from_filename(p.name)
        if sym is None:
            logger.warning("Skipping non-conforming filename: {}", p.name)
            continue
        found.append(p)
    return found


def validate_symbol_universe(minute_paths: list[Path], expected_symbols: set[str]) -> set[str]:
    """Assert minute file symbol set matches ``expected_symbols`` exactly.

    Returns the set of symbols (derived from filenames). Raises ``RuntimeError``
    on any mismatch with a loud error message naming the offending symbols.
    """
    found = {_symbol_from_filename(p.name) for p in minute_paths}
    found.discard(None)  # type: ignore[arg-type]
    found_set: set[str] = found  # type: ignore[assignment]

    missing = expected_symbols - found_set
    extra = found_set - expected_symbols
    if missing or extra:
        logger.error(
            "Minute universe mismatch — missing={} extra={}",
            sorted(missing)[:10],
            sorted(extra)[:10],
        )
        raise RuntimeError(
            f"Minute symbol set does not match daily universe: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    logger.info("Symbol universe matches daily: {} symbols", len(found_set))
    return found_set


def parse_minute_csv(
    path: Path,
    *,
    drop_dates: frozenset[date] = frozenset(),
) -> pl.DataFrame | None:
    """Parse one minute CSV, clean it, and add flags. Returns ``None`` on error.

    Pipeline:
      1. Read raw CSV (string ``date`` + numeric OHLCV).
      2. Parse the IST-offset string into a tz-aware ``timestamp`` (Asia/Kolkata).
      3. Derive calendar ``date`` and ``year`` from the IST wall clock.
      4. Drop rows with null prices / negative prices.
      5. Drop rows in ``drop_dates`` (e.g. the trailing partial 2021-06-24).
      6. Deduplicate on (timestamp), keeping last; sort ascending.
      7. Compute per-day bar counts and set ``partial_day_flag`` < 375.
    """
    symbol = _symbol_from_filename(path.name)
    if symbol is None:
        logger.error("Cannot derive symbol from {}", path.name)
        return None

    try:
        df = pl.read_csv(
            path,
            schema_overrides=_MINUTE_SCHEMA,
            null_values=["", "NA", "NaN", "null"],
        )
    except Exception as exc:
        logger.error("{}: failed to read CSV: {}", symbol, exc)
        return None

    if df.is_empty():
        logger.warning("{}: empty file, skipping", symbol)
        return None

    # 2. Parse timezone-aware timestamp (IST). The +05:30 offset in the source
    # is preserved by keeping the column in the Asia/Kolkata zone. Non-strict so
    # rare malformed rows (e.g. day-of-month "97") become null and are dropped
    # below rather than killing the whole symbol.
    try:
        df = df.with_columns(
            pl.col("date")
            .str.to_datetime(
                "%Y-%m-%d %H:%M:%S%:z",
                time_zone="Asia/Kolkata",
                strict=False,
            )
            .alias("timestamp")
        )
    except Exception as exc:
        logger.error("{}: timestamp parse failed: {}", symbol, exc)
        return None

    n_bad_ts = df.filter(pl.col("timestamp").is_null()).height
    if n_bad_ts:
        logger.warning("{}: dropping {} rows with unparseable timestamps", symbol, n_bad_ts)
        df = df.filter(pl.col("timestamp").is_not_null())

    # 3. Derive calendar date + year from IST wall clock
    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("date"),
        pl.col("timestamp").dt.year().cast(pl.Int32).alias("year"),
    )

    initial_rows = df.height

    # 4. Drop null / negative prices
    price_cols = ["open", "high", "low", "close"]
    null_mask = pl.any_horizontal(*[pl.col(c).is_null() for c in price_cols])
    n_null = df.filter(null_mask).height
    if n_null:
        logger.warning("{}: dropping {} rows with null prices", symbol, n_null)
        df = df.filter(~null_mask)

    neg_mask = pl.any_horizontal(*[pl.col(c) < 0 for c in price_cols])
    n_neg = df.filter(neg_mask).height
    if n_neg:
        logger.warning("{}: dropping {} rows with negative prices", symbol, n_neg)
        df = df.filter(~neg_mask)

    # 5. Drop configured dates (e.g. trailing partial day)
    n_dropped_dates = 0
    if drop_dates:
        # Build a literal date series to filter against
        drop_list = list(drop_dates)
        before = df.height
        df = df.filter(~pl.col("date").is_in(drop_list))
        n_dropped_dates = before - df.height
        if n_dropped_dates:
            logger.info(
                "{}: dropped {} rows in configured drop_dates {}",
                symbol,
                n_dropped_dates,
                sorted(drop_list),
            )

    if df.is_empty():
        logger.warning("{}: no rows remain after cleaning", symbol)
        return None

    # Volume: null → 0 (keep as Int64)
    df = df.with_columns(pl.col("volume").fill_null(0))

    # 6. Deduplicate on timestamp, keep last; sort
    n_before_dedup = df.height
    df = df.unique(subset=["timestamp"], keep="last").sort("timestamp")
    n_dup = n_before_dedup - df.height
    if n_dup:
        logger.debug("{}: removed {} duplicate timestamps", symbol, n_dup)

    # 7. Partial-day flag: bars-per-(symbol, day) < 375.
    bars_per_day = df.group_by("date").len().rename({"len": "bars_in_day"})
    df = (
        df.join(bars_per_day, on="date", how="left")
        .with_columns((pl.col("bars_in_day") < FULL_SESSION_BARS).alias("partial_day_flag"))
        .drop("bars_in_day")
    )

    # Reorder to canonical layout. ``symbol`` is added below; we keep ``year``
    # for the hive partition layout.
    df = df.select(["year", *_DATA_COLS])

    rows_kept = df.height
    logger.debug(
        "{}: ingested {}/{} rows ({} null prices, {} neg, {} dup, {} dropped_dates)",
        symbol,
        rows_kept,
        initial_rows,
        n_null,
        n_neg,
        n_dup,
        n_dropped_dates,
    )
    return df


def _strictly_increasing(timestamps: pl.Series) -> bool:
    """True if the series is strictly increasing (no duplicates, no out-of-order)."""
    if timestamps.len() < 2:
        return True
    # Convert tz-aware datetime → integer microseconds for a portable < comparison.
    micros = timestamps.dt.timestamp("us")
    diffs = micros.diff().drop_nulls()
    return bool((diffs > 0).all())


def _write_symbol_partitions(df: pl.DataFrame, output_root: Path, symbol: str) -> int:
    """Write one ``year=YYYY`` partition file per year under
    ``output_root/symbol=SYM/``. Returns the number of partitions written.
    """
    sym_dir = output_root / f"symbol={symbol}"
    sym_dir.mkdir(parents=True, exist_ok=True)

    n_partitions = 0
    for (yr,), part in df.group_by("year", maintain_order=True):
        year_dir = sym_dir / f"year={yr}"
        year_dir.mkdir(parents=True, exist_ok=True)
        # Drop partition columns from the data file itself — they're encoded in
        # the directory names per the Hive convention.
        part.drop("year").write_parquet(year_dir / "data.parquet", compression="zstd")
        n_partitions += 1
    return n_partitions


def ingest_one_symbol(
    path: Path,
    output_root: Path,
    *,
    drop_dates: frozenset[date] = frozenset(),
) -> dict:
    """Parse + write a single symbol. Returns a stats dict for aggregation.

    Stats keys: ``symbol``, ``rows``, ``partitions``, ``partial_days``,
    ``min_ts``, ``max_ts``, ``strictly_increasing``, ``error``.
    """
    symbol = _symbol_from_filename(path.name) or path.stem
    df = parse_minute_csv(path, drop_dates=drop_dates)
    if df is None or df.is_empty():
        return {
            "symbol": symbol,
            "rows": 0,
            "partitions": 0,
            "partial_days": 0,
            "min_ts": None,
            "max_ts": None,
            "strictly_increasing": True,
            "error": "empty_or_unparseable",
        }

    partial_days = df.filter(pl.col("partial_day_flag")).select(pl.col("date").n_unique()).item()
    n_partitions = _write_symbol_partitions(df, output_root, symbol)

    return {
        "symbol": symbol,
        "rows": df.height,
        "partitions": n_partitions,
        "partial_days": int(partial_days),
        "min_ts": df["timestamp"].min(),
        "max_ts": df["timestamp"].max(),
        "strictly_increasing": _strictly_increasing(df["timestamp"]),
        "error": None,
    }


def ingest_all_minute(
    raw_dir: Path,
    output_root: Path,
    *,
    expected_symbols: set[str],
    drop_dates: frozenset[date] = frozenset(),
    n_jobs: int = -1,
) -> list[dict]:
    """Ingest every minute CSV under ``raw_dir`` into a hive-partitioned dataset.

    Idempotent: ``output_root`` is removed and recreated before writing.

    Args:
        raw_dir: directory of ``*_minute-data.csv`` files (read-only).
        output_root: target directory; will contain ``symbol=…/year=…/data.parquet``.
        expected_symbols: the daily-universe symbol set; minute files must match.
        drop_dates: calendar dates (IST) to exclude entirely.
        n_jobs: joblib worker count (-1 = all cores).

    Returns:
        List of per-symbol stats dicts (see ``ingest_one_symbol``).
    """
    from apt.utils.parallel import parallel_map

    csv_paths = discover_minute_csvs(raw_dir)
    logger.info("Discovered {} minute CSV files in {}", len(csv_paths), raw_dir)

    validate_symbol_universe(csv_paths, expected_symbols)

    # Idempotent reset of the output dataset directory.
    if output_root.exists():
        logger.info("Clearing existing dataset directory {}", output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    def _worker(p: Path) -> dict:
        return ingest_one_symbol(p, output_root, drop_dates=drop_dates)

    stats = parallel_map(
        _worker,
        csv_paths,
        n_jobs=n_jobs,
        desc="Ingesting minute CSVs",
        prefer="processes",
    )

    errored = [s for s in stats if s["error"]]
    if errored:
        logger.warning("{} symbols had errors", len(errored))
        for s in errored:
            logger.warning("  {}: {}", s["symbol"], s["error"])

    return stats
