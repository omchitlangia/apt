#!/usr/bin/env python3
"""Script 02: Ingest minute OHLCV CSVs → hive-partitioned parquet dataset.

Output layout::

    data/interim/minute_raw/
        symbol=RELIANCE/
            year=2015/data.parquet
            year=2016/data.parquet
            ...
        symbol=TCS/
            ...

The trailing partial trading day 2021-06-24 is dropped from the written
dataset; other short days (Muhurat, half-days, gaps) are retained and flagged
via ``partial_day_flag``.

Idempotent: re-running wipes and rewrites the output directory.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from loguru import logger

from apt.config import settings
from apt.data.ingest_minute import FULL_SESSION_BARS, ingest_all_minute
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim

# Trailing partial day to exclude per the Day-2 plan.
_DROP_DATES = frozenset({date(2021, 6, 24)})


def _dataset_size_bytes(root) -> int:
    return sum(p.stat().st_size for p in root.rglob("*.parquet"))


def _validate(stats, output_root, expected_symbols):
    """Run hard validation gates and log each result. Returns dict of results."""
    results: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Read the full dataset back via hive-aware scan for sanity checks.
    # ------------------------------------------------------------------
    lf = pl.scan_parquet(str(output_root / "**/*.parquet"), hive_partitioning=True)

    # Gate 1: RELIANCE 2020-01-02 has exactly 375 bars
    rel = (
        lf.filter((pl.col("symbol") == "RELIANCE") & (pl.col("date") == date(2020, 1, 2)))
        .select(pl.len().alias("n"))
        .collect()
    )
    n_rel = int(rel["n"][0])
    gate1 = n_rel == FULL_SESSION_BARS
    results["gate1_reliance_2020_01_02_bars"] = {"value": n_rel, "pass": gate1}
    logger.info("Gate 1 — RELIANCE 2020-01-02 bars == 375: {} (got {})", gate1, n_rel)
    assert gate1, f"RELIANCE 2020-01-02 expected 375 bars, got {n_rel}"

    # Gate 2: written symbol set == expected (492)
    syms_written = set(lf.select(pl.col("symbol").unique()).collect()["symbol"].to_list())
    gate2 = syms_written == expected_symbols
    results["gate2_symbol_set_matches"] = {
        "n_written": len(syms_written),
        "n_expected": len(expected_symbols),
        "missing": sorted(expected_symbols - syms_written),
        "extra": sorted(syms_written - expected_symbols),
        "pass": gate2,
    }
    logger.info(
        "Gate 2 — symbol set matches daily ({} written, {} expected): {}",
        len(syms_written),
        len(expected_symbols),
        gate2,
    )
    assert gate2, "Written symbol set does not match daily universe"

    # Gate 3: min ts is 2015-02-02 09:15 IST, max ts falls on 2021-06-23
    tmin = lf.select(pl.col("timestamp").min()).collect()["timestamp"][0]
    tmax = lf.select(pl.col("timestamp").max()).collect()["timestamp"][0]
    gate3a = tmin.date() == date(2015, 2, 2) and (tmin.hour, tmin.minute) == (9, 15)
    gate3b = tmax.date() == date(2021, 6, 23)
    gate3 = gate3a and gate3b
    results["gate3_timestamp_range"] = {
        "min": str(tmin),
        "max": str(tmax),
        "pass": gate3,
    }
    logger.info(
        "Gate 3 — timestamp range min={} max={} → 09:15 on 2015-02-02 & 2021-06-23: {}",
        tmin,
        tmax,
        gate3,
    )
    assert gate3, f"Timestamp range invalid: min={tmin} max={tmax}"

    # Gate 4: no duplicate (symbol, timestamp) pairs.
    # Stream-aggregate: any (symbol, timestamp) with count > 1 fails. We do this
    # in two passes — group on the full lazy frame and check max count.
    dup_check = (
        lf.group_by(["symbol", "timestamp"])
        .agg(pl.len().alias("n"))
        .select(pl.col("n").max().alias("max_n"))
        .collect()
    )
    max_count = int(dup_check["max_n"][0])
    gate4 = max_count == 1
    results["gate4_no_duplicate_symbol_timestamp"] = {
        "max_count": max_count,
        "pass": gate4,
    }
    logger.info(
        "Gate 4 — no duplicate (symbol, timestamp) pairs: {} (max group size {})",
        gate4,
        max_count,
    )
    assert gate4, f"Found duplicate (symbol, timestamp) — max group size {max_count}"

    # Gate 5: per-symbol timestamps strictly increasing (already verified
    # in-worker; aggregate across all stats here).
    bad = [s["symbol"] for s in stats if not s["strictly_increasing"]]
    gate5 = len(bad) == 0
    results["gate5_strictly_increasing"] = {
        "n_violating_symbols": len(bad),
        "violating_symbols": bad[:10],
        "pass": gate5,
    }
    logger.info(
        "Gate 5 — per-symbol timestamps strictly increasing: {} ({} violators)",
        gate5,
        len(bad),
    )
    assert gate5, f"Non-monotonic symbols: {bad[:10]}"

    return results


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "02_ingest_minute.log")
    ensure_dirs()

    output_root = interim("minute_raw")

    # Load daily universe as ground truth.
    daily_path = interim("daily_raw.parquet")
    if not daily_path.exists():
        raise FileNotFoundError(
            f"daily_raw.parquet not found at {daily_path} — run scripts/01_ingest_daily.py first"
        )
    expected_symbols = set(
        pl.read_parquet(daily_path, columns=["symbol"])["symbol"].unique().to_list()
    )
    logger.info("Loaded {} expected symbols from daily universe", len(expected_symbols))

    raw_dir = settings.paths.raw_data_dir / "minute"
    stats = ingest_all_minute(
        raw_dir=raw_dir,
        output_root=output_root,
        expected_symbols=expected_symbols,
        drop_dates=_DROP_DATES,
        n_jobs=settings.parallel.n_jobs,
    )

    total_rows = sum(s["rows"] for s in stats)
    total_partitions = sum(s["partitions"] for s in stats)
    total_partial_days = sum(s["partial_days"] for s in stats)
    written_symbols = sum(1 for s in stats if s["rows"] > 0)

    # Run hard validation gates against the written dataset.
    gate_results = _validate(stats, output_root, expected_symbols)

    on_disk = _dataset_size_bytes(output_root)

    print("\n=== 02_ingest_minute complete ===")
    print(f"  Total rows         : {total_rows:,}")
    print(f"  Symbols written    : {written_symbols} / 492")
    print(f"  Partitions written : {total_partitions:,}")
    print(f"  Partial-day count  : {total_partial_days:,}  (sum across symbols)")
    print(f"  On-disk size       : {on_disk / 1e9:.2f} GB")
    print(f"  Output             : {output_root}")
    print("  --- Validation gates ---")
    for gate, res in gate_results.items():
        verdict = "PASS" if res["pass"] else "FAIL"
        print(f"  [{verdict}] {gate}: {res}")


if __name__ == "__main__":
    main()
