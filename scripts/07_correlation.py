#!/usr/bin/env python3
"""Script 07: Phase 1 Day 5 — fold-aware correlation pre-filter.

Demonstrates :func:`apt.signals.correlation.screen_pairs` on a single
window: the most recent 504 trading days of ``daily_clean.parquet``.
The function itself is windowed — Phase 2 calls it per walk-forward fold
on each fold's *training* window, never globally.

Outputs:
  * ``data/pairs/correlated_pairs.parquet`` — surviving pair list
    (sym1, sym2, corr, sector) for this demonstration window.
  * ``plots/phase1/pairs/correlation_heatmap.png`` — sector-clustered
    correlation matrix of the eligible symbols.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from apt.config import settings
from apt.plots.pairs import plot_sector_clustered_correlation
from apt.signals.correlation import (
    compute_window_correlation,
    screen_pairs,
)
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, processed
from apt.utils.paths import pairs as pairs_path


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "07_correlation.log")
    ensure_dirs()

    daily_path = processed("daily_clean.parquet")
    sectors_path = interim("sectors.parquet")
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing — run scripts/05 first")
    if not sectors_path.exists():
        raise FileNotFoundError(f"{sectors_path} missing — run scripts/03 first")

    daily = pl.read_parquet(daily_path)
    sectors = pl.read_parquet(sectors_path)
    logger.info(
        "Loaded daily_clean ({:,} rows, {} symbols) and sectors ({} rows)",
        daily.height,
        daily["symbol"].n_unique(),
        sectors.height,
    )

    # Demonstration window: most recent 504 trading days.
    trading_days = daily.select(pl.col("date").unique()).sort("date")["date"].to_list()
    n_corr_days = settings.screening.n_corr_days
    if len(trading_days) < n_corr_days:
        raise RuntimeError(
            f"Only {len(trading_days)} trading days available — need >= {n_corr_days}"
        )
    start = trading_days[-n_corr_days]
    end = trading_days[-1]
    logger.info("Demo window: {} → {} ({} trading days)", start, end, n_corr_days)

    # ------------------------------------------------------------------
    # Windowed correlation matrix (for the heatmap; reused inside
    # screen_pairs but recomputed cheaply here for the plot).
    # ------------------------------------------------------------------
    win_corr = compute_window_correlation(
        daily,
        start=start,
        end=end,
        max_internal_gap_days=settings.cleaning.contiguity_max_gap_days,
    )
    logger.info(
        "Eligibility: {} of {} symbols cover the window with no internal gap; "
        "{} of {} window days survive the common-dates intersection",
        len(win_corr.eligible_symbols),
        daily["symbol"].n_unique(),
        win_corr.n_used_dates,
        win_corr.n_window_days,
    )

    # ------------------------------------------------------------------
    # Windowed pair screen
    # ------------------------------------------------------------------
    pairs_df = screen_pairs(
        daily,
        sectors,
        start=start,
        end=end,
        corr_threshold=settings.screening.correlation_threshold,
        max_internal_gap_days=settings.cleaning.contiguity_max_gap_days,
    )
    out_pairs = pairs_path("correlated_pairs.parquet")
    pairs_df.write_parquet(out_pairs)
    logger.info("Wrote {} pair rows → {}", pairs_df.height, out_pairs)

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    heatmap_path = settings.paths.plots_dir / "phase1" / "pairs" / "correlation_heatmap.png"
    heatmap_stats = plot_sector_clustered_correlation(
        win_corr.corr_matrix,
        win_corr.eligible_symbols,
        sectors,
        heatmap_path,
        title=f"Correlation heatmap  [{start} → {end}]",
    )

    # ------------------------------------------------------------------
    # Guardrails (assert + log)
    # ------------------------------------------------------------------
    if not pairs_df.is_empty():
        # Same-sector only (no cross-sector pairs)
        sym_to_sector = dict(zip(sectors["symbol"], sectors["industry"], strict=True))
        for r in pairs_df.iter_rows(named=True):
            s1_sec = sym_to_sector.get(r["sym1"])
            s2_sec = sym_to_sector.get(r["sym2"])
            assert s1_sec == s2_sec == r["sector"], (
                f"Cross-sector pair leaked: {r['sym1']}({s1_sec}) / "
                f"{r['sym2']}({s2_sec}); pair sector {r['sector']}"
            )
        assert pairs_df["corr"].min() > settings.screening.correlation_threshold

    # ------------------------------------------------------------------
    # Per-sector breakdown
    # ------------------------------------------------------------------
    per_sector = (
        pairs_df.group_by("sector")
        .len()
        .rename({"len": "n_pairs"})
        .sort("n_pairs", descending=True)
    )

    print("\n=== 07_correlation complete ===")
    print(f"  Window         : {start} → {end} ({n_corr_days} trading days)")
    print(
        f"  Eligible       : {len(win_corr.eligible_symbols)} / "
        f"{daily['symbol'].n_unique()} symbols pass the window gap guard"
    )
    print(
        f"  Common dates   : {win_corr.n_used_dates} / {win_corr.n_window_days} after intersection"
    )
    print(
        f"  Threshold      : Pearson(log-returns) > "
        f"{settings.screening.correlation_threshold}, same-sector only"
    )
    print(f"  Surviving pairs: {pairs_df.height}")
    print(f"  Output         : {out_pairs}")
    print(f"  Heatmap        : {heatmap_path}")
    print()
    print("  --- Per-sector survivor count ---")
    if per_sector.is_empty():
        print("    (none)")
    else:
        for r in per_sector.iter_rows(named=True):
            print(f"    {r['sector']:<32} {r['n_pairs']:>6}")
    print()
    if pairs_df.is_empty():
        print("  No surviving pairs.")
    else:
        print("  Top 10 by correlation:")
        top = pairs_df.sort("corr", descending=True).head(10)
        for r in top.iter_rows(named=True):
            print(f"    {r['sym1']:<12} {r['sym2']:<12} corr={r['corr']:.4f}  sector={r['sector']}")

    # Final sanity: heatmap was produced with the eligible-symbol set
    assert heatmap_stats["n_symbols"] == len(win_corr.eligible_symbols)


if __name__ == "__main__":
    main()
