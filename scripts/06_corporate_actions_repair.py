#!/usr/bin/env python3
"""Script 06: Day-4C corporate-action repair.

Reads ``data/processed/daily_clean.parquet`` (produced by Day-4A), classifies
each surviving >40% discontinuity, applies ADJUST (back-adjustment) and TRIM
(KEEP-guarded left + right cutoffs), re-applies the min-history filter, and
overwrites ``daily_clean.parquet`` in place. Day-4A is reproducible from its
own script if Day-4C needs to be re-run from scratch.

Final validation gate re-scans with the tightened COVID window
(2020-02-24..2020-04-03) excused, in addition to the existing KEEP_EVENTS
and cached dividend ex-dates. Target: zero unexplained survivors.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from apt.config import settings
from apt.data.ca_repair import (
    COVID_WINDOW,
    apply_repair,
    classify_survivors,
)
from apt.data.clean import (
    KEEP_EVENTS,
    apply_min_history,
    validation_gate,
)
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, processed, reports


def main() -> None:
    setup_logging(
        log_file=settings.paths.logs_dir / "06_corporate_actions_repair.log"
    )
    ensure_dirs()

    src = processed("daily_clean.parquet")
    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run scripts/05_clean_align.py first")
    df = pl.read_parquet(src)
    init_rows, init_syms = df.height, df["symbol"].n_unique()
    logger.info("Loaded daily_clean: {:,} rows, {} symbols", init_rows, init_syms)

    actions = pl.read_parquet(interim("corporate_actions.parquet"))
    cov_csv = pl.read_csv(reports("corporate_actions_coverage.csv"))
    yf_failed: set[str] = set(
        cov_csv.filter(pl.col("status") == "failed")["symbol"].to_list()
    )

    # ------------------------------------------------------------------
    # Recompute the Day-4A baseline survivors (no COVID excuse yet).
    # ------------------------------------------------------------------
    baseline = validation_gate(
        df,
        actions,
        start_date=settings.cleaning.validation_start,
        threshold=settings.cleaning.validation_jump_threshold,
        keep=KEEP_EVENTS,
        yfinance_failed_symbols=yf_failed,
    )
    survivors = pl.DataFrame(baseline["survivors"])
    logger.info("Baseline survivors (Day-4A gate): {}", survivors.height)
    if survivors.is_empty():
        logger.info("Gate already green; nothing to repair.")
        return

    # ------------------------------------------------------------------
    # Classify and persist the per-survivor decision.
    # ------------------------------------------------------------------
    classified = classify_survivors(survivors)
    cls_path = reports("ca_repair_classification.csv")
    classified.with_columns(
        pl.col("symbol").is_in(yf_failed).alias("yfinance_blind")
    ).sort(["category", "symbol", "date"]).write_csv(cls_path)
    cat_counts = (
        classified.group_by("category").len().sort("len", descending=True).to_dicts()
    )

    # ------------------------------------------------------------------
    # Apply ADJUST + TRIM (left/right, KEEP-guarded).
    # ------------------------------------------------------------------
    df, repair_summary = apply_repair(df, classified)
    logger.info(
        "Repair: {} ADJUSTs, {} symbols left-trimmed, {} symbols right-trimmed",
        repair_summary["n_adjust_events"],
        repair_summary["n_trim_left_symbols"],
        repair_summary["n_trim_right_symbols"],
    )

    # ------------------------------------------------------------------
    # Re-apply min-history. TRIM can shorten symbols below the threshold.
    # ------------------------------------------------------------------
    pre_mh_syms = df["symbol"].n_unique()
    df, mh_report = apply_min_history(df, min_days=settings.universe.min_history_days)
    newly_dropped_minhist = pre_mh_syms - df["symbol"].n_unique()
    logger.info(
        "Min-history (>= {} days): {} → {} symbols ({} newly dropped)",
        settings.universe.min_history_days,
        pre_mh_syms,
        df["symbol"].n_unique(),
        newly_dropped_minhist,
    )

    # ------------------------------------------------------------------
    # Overwrite daily_clean.parquet.
    # ------------------------------------------------------------------
    df.sort(["symbol", "date"]).write_parquet(src, use_pyarrow=True)
    logger.info(
        "Wrote daily_clean: {:,} rows, {} symbols → {}",
        df.height,
        df["symbol"].n_unique(),
        src,
    )

    # ------------------------------------------------------------------
    # Final validation gate, now with COVID window excused.
    # ------------------------------------------------------------------
    gate = validation_gate(
        df,
        actions,
        start_date=settings.cleaning.validation_start,
        threshold=settings.cleaning.validation_jump_threshold,
        keep=KEEP_EVENTS,
        keep_windows=[COVID_WINDOW],
        yfinance_failed_symbols=yf_failed,
    )

    print("\n=== 06_corporate_actions_repair complete ===")
    print(
        f"  Input  : {init_rows:>10,} rows × {init_syms} symbols  (daily_clean before repair)"
    )
    print(
        f"  Output : {df.height:>10,} rows × {df['symbol'].n_unique()} symbols  "
        f"(daily_clean after repair)"
    )
    print()
    print("  --- Classification of Day-4A survivors ---")
    print(f"    total survivors classified: {survivors.height}")
    for r in cat_counts:
        print(f"      {r['category']:<16} {r['len']:>4}")
    print(f"    classification csv → {cls_path}")
    print()
    print("  --- ADJUST events (back-adjust pre-ex-date OHLC) ---")
    for a in repair_summary["adjusts"]:
        print(
            f"    {a['symbol']:<12} {a['ex_date']}  factor={a['adjust_factor']:.3f}  "
            f"ret={a['ret']:+.4f}  (implied split={a['implied_split_ratio']:.3f}:1)"
        )
    print()
    print("  --- TRIM events ---")
    print(f"    left-trim symbols  : {repair_summary['n_trim_left_symbols']}")
    print(f"    right-trim symbols : {repair_summary['n_trim_right_symbols']}")
    if repair_summary["trims_right"]:
        for t in repair_summary["trims_right"]:
            print(
                f"      right-trim {t['symbol']:<12} drop_from = {t['cutoff']}  "
                f"({t['n_trim_events']} event)"
            )
    print()
    print(
        f"  --- Min-history re-apply (>= {settings.universe.min_history_days} days) ---"
    )
    print(f"    newly dropped : {newly_dropped_minhist}")
    if mh_report["symbols_dropped"]:
        names = [
            f"{d['symbol']}({d['n_days']})" for d in mh_report["symbols_dropped"][:20]
        ]
        more = (
            ""
            if len(mh_report["symbols_dropped"]) <= 20
            else f" … +{len(mh_report['symbols_dropped']) - 20} more"
        )
        print(f"      {', '.join(names)}{more}")
    print()
    print("  --- Final validation gate (COVID window excused) ---")
    print(f"    threshold        : > {gate['threshold']:.2f} from {gate['start_date']}")
    print(f"    big moves found  : {gate['n_big_moves']}")
    print(f"    excused by window: {gate['n_excused_by_window']}")
    print(f"    UNEXPLAINED      : {gate['n_survivors']}")
    if gate["survivors"]:
        for s in sorted(
            gate["survivors"], key=lambda r: abs(r["ret"]), reverse=True
        ):
            print(
                f"      {s['symbol']:<12} {s['date']}  close={s['close']}  "
                f"return={s['ret']:+.4f}"
            )
    print(
        f"    yfinance-blind universe: {gate['n_yfinance_blind_total']} symbols; "
        f"survivors among them: {gate['n_yfinance_blind_survivors']}"
    )
    print(f"    GATE: {'PASS' if gate['pass'] else 'FAIL'}")

    assert gate["pass"], (
        f"Day-4C gate FAILED — {gate['n_survivors']} unexplained survivors remain"
    )


if __name__ == "__main__":
    main()
