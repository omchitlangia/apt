"""Daily OHLCV ingestion from raw CSVs in /Data6/db/.

Pure functions — no global state. All I/O is explicit via passed paths.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
from loguru import logger

# Columns present in every daily CSV
_DAILY_COLS = ["date", "open", "high", "low", "close", "volume"]

_DAILY_SCHEMA = {
    "date": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# Files to skip in the daily CSV directory
_SKIP_NAMES = frozenset({"sbin", "NSE_Instruments_23June2021"})

# Only filenames matching this pattern are considered daily equity CSVs
_SYMBOL_RE = re.compile(r"^[A-Z0-9&\-]+$")


def _symbol_from_path(p: Path) -> str:
    return p.stem.upper()


def discover_daily_csvs(raw_dir: Path) -> list[Path]:
    """Return sorted list of daily equity CSV paths, excluding known non-equity files."""
    found = []
    for p in sorted(raw_dir.glob("*.csv")):
        stem = p.stem
        if stem in _SKIP_NAMES:
            logger.debug("Skipping excluded file: {}", p.name)
            continue
        symbol = _symbol_from_path(p)
        if not _SYMBOL_RE.match(symbol):
            logger.warning("Skipping non-symbol filename: {}", p.name)
            continue
        found.append(p)
    return found


def _parse_date(s: str) -> str:
    """Normalise 'YYYY-MM-DD HH:MM:SS+TZ' → 'YYYY-MM-DD' UTC date string."""
    return s[:10]


def parse_daily_csv(path: Path) -> pl.DataFrame | None:
    """Parse a single daily OHLCV CSV. Returns None on unrecoverable error.

    - Timestamps are stripped to UTC date (date component only).
    - Rows with NaN prices or negative prices are dropped and logged.
    - Zero-volume rows are flagged but retained.
    - Duplicate dates (after date normalisation) are deduplicated, keeping last.
    """
    symbol = _symbol_from_path(path)
    try:
        df = pl.read_csv(
            path,
            schema_overrides=_DAILY_SCHEMA,
            null_values=["", "NA", "NaN", "null"],
        )
    except Exception as exc:
        logger.error("Failed to read {}: {}", path.name, exc)
        return None

    if df.is_empty():
        logger.warning("{}: empty file, skipping", symbol)
        return None

    # Normalise date: strip time + timezone component
    df = df.with_columns(pl.col("date").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date"))

    initial_rows = len(df)

    # Drop rows where any price column is null
    price_cols = ["open", "high", "low", "close"]
    null_mask = pl.any_horizontal(*[pl.col(c).is_null() for c in price_cols])
    null_count = df.filter(null_mask).height
    if null_count:
        logger.warning("{}: dropping {} rows with null prices", symbol, null_count)
        df = df.filter(~null_mask)

    # Drop rows with negative prices
    neg_mask = pl.any_horizontal(*[pl.col(c) < 0 for c in price_cols])
    neg_count = df.filter(neg_mask).height
    if neg_count:
        logger.warning("{}: dropping {} rows with negative prices", symbol, neg_count)
        df = df.filter(~neg_mask)

    if df.is_empty():
        logger.warning("{}: no valid rows remain after cleaning", symbol)
        return None

    # Volume: fill null with 0, flag zero-volume rows
    df = df.with_columns(pl.col("volume").fill_null(0.0))
    df = df.with_columns((pl.col("volume") == 0).alias("zero_volume_flag"))

    # Deduplicate on date, keep last occurrence
    n_before = len(df)
    df = df.unique(subset=["date"], keep="last").sort("date")
    n_dup = n_before - len(df)
    if n_dup:
        logger.debug("{}: removed {} duplicate date rows", symbol, n_dup)

    # Add symbol column
    df = df.with_columns(pl.lit(symbol).alias("symbol"))

    # Reorder to canonical schema
    df = df.select(["symbol", "date", "open", "high", "low", "close", "volume", "zero_volume_flag"])

    rows_kept = len(df)
    rows_dropped = initial_rows - rows_kept
    logger.debug(
        "{}: {} rows ingested ({} dropped from {} raw)",
        symbol,
        rows_kept,
        rows_dropped,
        initial_rows,
    )
    return df


def ingest_all_daily(
    raw_dir: Path,
    output_path: Path,
    *,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """Ingest all daily CSVs and write a long-format parquet.

    Uses joblib for parallelism. Returns the combined DataFrame.
    Idempotent: overwrites existing output_path.
    """
    from apt.utils.parallel import parallel_map

    csv_paths = discover_daily_csvs(raw_dir)
    logger.info("Discovered {} daily CSV files in {}", len(csv_paths), raw_dir)

    frames = parallel_map(
        parse_daily_csv,
        csv_paths,
        n_jobs=n_jobs,
        desc="Ingesting daily CSVs",
        prefer="threads",
    )

    valid = [f for f in frames if f is not None]
    logger.info("Successfully parsed {}/{} files", len(valid), len(csv_paths))

    if not valid:
        raise RuntimeError("No valid daily CSVs found — aborting ingest")

    combined = pl.concat(valid, how="vertical")

    # Partition by year for fast filtering later
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.with_columns(pl.col("date").dt.year().alias("year")).write_parquet(
        output_path,
        use_pyarrow=True,
    )

    logger.info(
        "Wrote {} rows × {} symbols → {}",
        len(combined),
        combined["symbol"].n_unique(),
        output_path,
    )
    return combined
