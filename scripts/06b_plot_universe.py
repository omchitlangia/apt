#!/usr/bin/env python3
"""Script 07: Phase 1 Day 4B — universe-EDA plots.

Reads ``data/processed/daily_clean.parquet`` and ``data/interim/sectors.parquet``
and writes five sanity-check PNGs to ``plots/phase1/universe/``:

  1. ``01_symbols_per_sector.png``    — bar chart, thin sectors flagged
  2. ``02_history_length_distribution.png`` — histogram + 756d floor
  3. ``03_coverage_heatmap.png``      — symbol × year availability
  4. ``04_return_distributions.png``  — aggregate + 5 named overlays
  5. ``05_adv_distribution.png``      — log-x with ₹1cr floor

These are diagnostics — flag anything off in the console summary.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from apt.config import settings
from apt.plots.universe import (
    plot_adv_distribution,
    plot_coverage_heatmap,
    plot_history_length_distribution,
    plot_return_distributions,
    plot_symbols_per_sector,
)
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, processed


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "07_plot_universe.log")
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

    out_dir = settings.paths.plots_dir / "phase1" / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_symbols_per_sector(daily, sectors, out_dir / "01_symbols_per_sector.png")
    p2 = plot_history_length_distribution(daily, out_dir / "02_history_length_distribution.png")
    p3 = plot_coverage_heatmap(daily, out_dir / "03_coverage_heatmap.png")
    p4 = plot_return_distributions(daily, out_dir / "04_return_distributions.png")
    p5 = plot_adv_distribution(daily, out_dir / "05_adv_distribution.png")

    print("\n=== 07_plot_universe complete ===")
    print(f"  output dir: {out_dir}")
    print()
    print("  --- 1. Symbols per sector ---")
    print(
        f"    sectors          : {p1['n_sectors']}  "
        f"(median {p1['median_per_sector']} symbols/sector)"
    )
    if p1["thin_sectors"]:
        thin_names = ", ".join(f"{t['industry']}({t['n_symbols']})" for t in p1["thin_sectors"])
        print(f"    FLAG thin (< 5) : {thin_names}")
    else:
        print("    thin sectors    : none")
    print()
    print("  --- 2. History-length distribution ---")
    print(
        f"    n_symbols={p2['n_symbols']}, "
        f"median={p2['median_days']:,}d, "
        f"min={p2['min_days']:,}d, max={p2['max_days']:,}d  "
        f"(floor=756)"
    )
    print(f"    just above floor (<=900): {p2['n_at_or_below_900']}")
    print()
    print("  --- 3. Coverage heatmap ---")
    print(
        f"    {p3['n_symbols']} symbols × {p3['n_years']} years ({p3['year_min']}–{p3['year_max']})"
    )
    print(f"    started 2003-2004    : {p3['n_symbols_starting_2003_04']}")
    print(f"    started 2015+        : {p3['n_symbols_starting_2015_plus']}")
    print()
    print("  --- 4. Return distributions ---")
    print(
        f"    n_rets={p4['n_rets']:,}, σ={p4['std']:.4f}, min={p4['min']:.4f}, max={p4['max']:.4f}"
    )
    print(f"    |logret| > 0.20      : {p4['n_abs_gt_0.2']}")
    if p4["top5_extreme"]:
        print("    top 5 by |logret|:")
        for r in p4["top5_extreme"]:
            print(
                f"      {r['symbol']:<12} {r['date']}  "
                f"logret={r['logret']:+.4f}  close={r['close']}"
            )
    print()
    print("  --- 5. ADV distribution ---")
    print(
        f"    n={p5['n_symbols']}; median ₹{p5['median_inr'] / 1e7:.1f} cr; "
        f"p10 ₹{p5['p10_inr'] / 1e7:.2f} cr; p90 ₹{p5['p90_inr'] / 1e7:.1f} cr"
    )
    print(f"    below ₹1cr floor    : {p5['n_below_floor']}")


if __name__ == "__main__":
    main()
