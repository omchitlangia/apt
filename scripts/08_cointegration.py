#!/usr/bin/env python3
"""Script 08: Phase 1 Day 6 — fold-aware cointegration with robustness selection.

Demonstrates :func:`apt.signals.cointegration.cointegrate_pairs` on the most
recent ~1008 trading days of ``daily_clean.parquet`` (falling back to 756 if
that's all the universe affords). The function itself is fully windowed —
Phase 2's walk-forward loop calls it per fold on the fold's training window
only, never globally.

The script also runs the same screen on the *prior adjacent window* (same
length, ending the trading day before ``start``) and passes that to the
cointegration call so the cross-window stability check can fire.

Outputs:
  * ``data/pairs/cointegrated_pairs.parquet`` — the full candidate diagnostic
    frame (one row per pair that made it past the share-class filter and the
    sufficient-overlap check). Boolean flag columns define the tradeable
    subset; downstream code filters with
    ``fdr_pass & half_life_in_band & hurst_pass & stable_prior_window``.
  * ``plots/phase1/pairs/spread_<sym1>_<sym2>.png`` — two spread diagnostics
    for the top tradeable pairs (mean ± 1σ / ±2σ bands).
  * ``reports/cointegrated_pairs_ranked.csv`` — human-readable ranking.

The full funnel (candidates → sufficient-overlap → ADF p<0.05 → FDR survivors
→ half-life band → Hurst<0.5 → cross-window stable) is printed to stdout.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from apt.config import settings
from apt.plots.pairs import plot_pair_spread
from apt.signals.cointegration import cointegrate_pairs
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, pairs, processed, reports


def _resolve_windows(trading_days: list, n_target: int, n_min: int) -> tuple:
    """Return ((train_start, train_end), (prior_start, prior_end), n_train).

    Falls back from ``n_target`` to ``n_min`` if the universe doesn't have
    enough trading days. The prior window is the same length as the train
    window, ending one trading day before ``train_start`` (and starting one
    train-window-length earlier, clamped to ``trading_days[0]``).
    """
    if len(trading_days) < n_min:
        raise RuntimeError(f"Only {len(trading_days)} trading days available — need >= {n_min}")
    n_train = min(n_target, len(trading_days))
    if n_train < n_min:
        raise RuntimeError(f"After capping by available history, n_train={n_train} < n_min={n_min}")

    end = trading_days[-1]
    start = trading_days[-n_train]

    # Prior adjacent window — same length, ending the trading day before start.
    if len(trading_days) > n_train:
        prior_end = trading_days[-n_train - 1]
        if len(trading_days) >= 2 * n_train:
            prior_start = trading_days[-2 * n_train]
        else:
            prior_start = trading_days[0]
        return (start, end), (prior_start, prior_end), n_train

    # No room for any prior window at all
    return (start, end), (None, None), n_train


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "08_cointegration.log")
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

    trading_days = daily.select(pl.col("date").unique()).sort("date")["date"].to_list()
    (start, end), (prior_start, prior_end), n_train = _resolve_windows(
        trading_days,
        n_target=settings.cointegration.n_train_days,
        n_min=settings.cointegration.min_train_days,
    )
    logger.info("Train window: {} → {} ({} trading days)", start, end, n_train)
    if prior_start is None:
        logger.warning("No prior adjacent window available — cross-window stability disabled")
    else:
        logger.info("Prior window: {} → {}", prior_start, prior_end)

    # ------------------------------------------------------------------
    # Run the windowed cointegration screen
    # ------------------------------------------------------------------
    res = cointegrate_pairs(
        daily,
        sectors,
        start=start,
        end=end,
        prior_start=prior_start,
        prior_end=prior_end,
        corr_threshold=settings.screening.correlation_threshold,
        fdr_alpha=settings.cointegration.fdr_alpha,
        raw_alpha=settings.cointegration.max_pvalue,
        half_life_min_days=settings.cointegration.half_life_min_days,
        half_life_max_days=settings.cointegration.half_life_max_days,
        hurst_max=settings.cointegration.hurst_max,
        hurst_max_lag=settings.cointegration.hurst_max_lag,
        min_overlap_days=settings.cointegration.min_train_days,
        max_internal_gap_days=settings.cleaning.contiguity_max_gap_days,
    )

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    out_pairs = pairs("cointegrated_pairs.parquet")
    res.pairs.write_parquet(out_pairs)
    logger.info("Wrote {} pair rows → {}", res.pairs.height, out_pairs)

    tradeable_mask = pl.col("fdr_pass") & pl.col("half_life_in_band") & pl.col("hurst_pass")
    if prior_start is not None:
        tradeable_mask = tradeable_mask & pl.col("stable_prior_window")
    tradeable = res.pairs.filter(tradeable_mask).sort("adf_pvalue")

    ranked_csv = reports("cointegrated_pairs_ranked.csv")
    tradeable.write_csv(ranked_csv)
    logger.info("Wrote {} tradeable rows → {}", tradeable.height, ranked_csv)

    # ------------------------------------------------------------------
    # Spread diagnostic plots — top 4 tradeable pairs by ADF p-value
    # ------------------------------------------------------------------
    plot_dir = settings.paths.plots_dir / "phase1" / "pairs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_targets = tradeable.head(4)
    plot_paths: list = []
    for r in plot_targets.iter_rows(named=True):
        fname = f"spread_{r['y_sym']}_{r['x_sym']}.png".replace("&", "AND")
        out = plot_dir / fname
        extra = (
            f"ADF p={r['adf_pvalue']:.4g}  |  half-life={r['half_life']:.1f}d  "
            f"|  Hurst={r['hurst']:.2f}"
        )
        plot_pair_spread(
            daily,
            y_sym=r["y_sym"],
            x_sym=r["x_sym"],
            alpha=r["alpha"],
            beta=r["beta"],
            start=start,
            end=end,
            out_path=out,
            sector=r["sector"],
            extra_title=extra,
        )
        plot_paths.append(out)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    f = res.funnel
    print("\n=== 08_cointegration complete ===")
    print(f"  Train window     : {start} → {end} ({n_train} trading days)")
    if prior_start is not None:
        print(f"  Prior window     : {prior_start} → {prior_end}")
    else:
        print("  Prior window     : (none — stability check disabled)")
    print()
    print("  --- Funnel ---")
    print(
        f"    eligible symbols (post share-class filter)  : {f.eligible_symbols:>5}  "
        f"(dropped {f.excluded_share_class} share-class duplicate(s))"
    )
    print(
        f"    same-sector candidates @ corr > {settings.screening.correlation_threshold:.2f} : "
        f"{f.candidates_after_corr:>5}"
    )
    print(
        f"    sufficient pair-wise overlap (>= {settings.cointegration.min_train_days}d): "
        f"{f.sufficient_overlap:>5}"
    )
    print(
        f"    raw ADF p < {settings.cointegration.max_pvalue:.2f}                      "
        f": {f.raw_p_below_alpha:>5}"
    )
    print(
        f"    BH FDR survivors @ α = {settings.cointegration.fdr_alpha:.2f}            "
        f": {f.fdr_survivors:>5}   "
        f"(multiple-comparisons inflation gap: {f.raw_p_below_alpha - f.fdr_survivors})"
    )
    print(
        f"    half-life ∈ [{settings.cointegration.half_life_min_days},"
        f"{settings.cointegration.half_life_max_days}] trading days     "
        f": {f.half_life_in_band:>5}"
    )
    print(
        f"    Hurst < {settings.cointegration.hurst_max:.2f}                          "
        f": {f.hurst_below_max:>5}"
    )
    if f.cross_window_stable is not None:
        print(
            f"    ALSO cointegrated in prior window (raw p < {settings.cointegration.max_pvalue:.2f})"
            f" : {f.cross_window_stable:>5}"
        )
    print(f"    -> tradeable pair universe                : {f.tradeable:>5}")
    print(f"       of which structural (holdco/subsidiary)    : {f.structural_in_tradeable:>5}")
    print()
    print(f"  Outputs          : {out_pairs}")
    print(f"                     {ranked_csv}")
    for p in plot_paths:
        print(f"                     {p}")
    print()
    if tradeable.is_empty():
        print("  No tradeable pairs in this window.")
    else:
        print("  --- Tradeable pairs (sorted by ADF p-value) ---")
        for r in tradeable.iter_rows(named=True):
            struct = " [STRUCTURAL]" if r["is_structural_pair"] else ""
            print(
                f"    {r['sym1']:<12} / {r['sym2']:<12}  "
                f"β={r['beta']:+.4f}  p={r['adf_pvalue']:.4g}  "
                f"hl={r['half_life']:.1f}d  H={r['hurst']:.2f}  "
                f"[{r['sector']}]{struct}"
            )

    # Sanity assertions
    if not tradeable.is_empty():
        assert (tradeable["fdr_pass"]).all()
        assert (tradeable["half_life_in_band"]).all()
        assert (tradeable["hurst_pass"]).all()
        if prior_start is not None:
            assert (tradeable["stable_prior_window"]).all()
        assert tradeable["half_life"].min() >= settings.cointegration.half_life_min_days
        assert tradeable["half_life"].max() <= settings.cointegration.half_life_max_days
        assert tradeable["hurst"].max() < settings.cointegration.hurst_max


if __name__ == "__main__":
    main()
