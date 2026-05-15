#!/usr/bin/env python3
"""Script 01: Ingest all daily OHLCV CSVs → data/interim/daily_raw.parquet

Idempotent: re-running overwrites the output file.
"""

from apt.config import settings
from apt.data.ingest import ingest_all_daily
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim

if __name__ == "__main__":
    setup_logging(log_file=settings.paths.logs_dir / "01_ingest_daily.log")
    ensure_dirs()

    out = interim("daily_raw.parquet")
    df = ingest_all_daily(
        raw_dir=settings.paths.raw_data_dir,
        output_path=out,
        n_jobs=settings.parallel.n_jobs,
    )

    print("\n=== 01_ingest_daily complete ===")
    print(f"  Rows    : {len(df):,}")
    print(f"  Symbols : {df['symbol'].n_unique()}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Output  : {out}")
