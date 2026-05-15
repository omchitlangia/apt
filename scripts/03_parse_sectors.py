#!/usr/bin/env python3
"""Script 03: Parse sector mapping from xlsx → data/interim/sectors.parquet

Idempotent: re-running overwrites the output file.
Requires daily_raw.parquet to exist (for universe symbol set).
"""

import polars as pl

from apt.config import settings
from apt.data.sectors import build_sector_mapping
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim

if __name__ == "__main__":
    setup_logging(log_file=settings.paths.logs_dir / "03_parse_sectors.log")
    ensure_dirs()

    xlsx = settings.paths.raw_data_dir / "Merge_21May2021.xlsx"
    out = interim("sectors.parquet")

    # Load universe symbols if daily_raw already exists
    daily_raw = interim("daily_raw.parquet")
    universe_symbols: set[str] | None = None
    if daily_raw.exists():
        universe_symbols = set(
            pl.read_parquet(daily_raw, columns=["symbol"])["symbol"].unique().to_list()
        )
        print(f"  Universe symbols loaded: {len(universe_symbols)}")
    else:
        print("  daily_raw.parquet not found — skipping unmapped-symbol report")

    df = build_sector_mapping(
        xlsx_path=xlsx,
        output_path=out,
        universe_symbols=universe_symbols,
    )

    mapped = df["symbol"].to_list()
    industries = df["industry"].drop_nulls().unique().sort().to_list()

    print("\n=== 03_parse_sectors complete ===")
    print(f"  Symbols mapped : {len(mapped)}")
    print(f"  Industries     : {len(industries)}")
    print(f"  Output         : {out}")
    if universe_symbols:
        missing = sorted(universe_symbols - set(mapped))
        print(f"  Universe without mapping: {len(missing)}")
        if missing:
            print(f"    {missing[:20]}{'...' if len(missing) > 20 else ''}")
