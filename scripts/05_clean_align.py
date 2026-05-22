#!/usr/bin/env python3
"""Script 05: clean and align daily_adjusted → daily_clean.

Day-4A. Six ordered rules; final validation gate confirms no unexplained
discontinuity ≥ 40% from 2011-01-01 onward.
"""

from __future__ import annotations

import json
from datetime import date

import polars as pl
from loguru import logger

from apt.config import settings
from apt.data.clean import (
    KEEP_EVENTS,
    RESIDUAL_SPLITS,
    STRUCTURAL_EVENTS,
    apply_calendar_filter,
    apply_contiguity_filter,
    apply_liquidity_filter,
    apply_min_history,
    apply_residual_splits,
    apply_structural_events,
    build_trading_calendar,
    max_internal_gap_per_symbol,
    trim_phantom_history,
    validation_gate,
    verify_split_smoothness,
)
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, processed, reports


def _write_report(report: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "05_clean_align.log")
    ensure_dirs()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    src_path = processed("daily_adjusted.parquet")
    actions_path = interim("corporate_actions.parquet")
    coverage_path = reports("corporate_actions_coverage.csv")

    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} missing — run scripts/04_corporate_actions.py first")
    df = pl.read_parquet(src_path)
    logger.info(
        "Loaded daily_adjusted: {:,} rows, {} symbols, {} → {}",
        df.height,
        df["symbol"].n_unique(),
        df["date"].min(),
        df["date"].max(),
    )

    actions = (
        pl.read_parquet(actions_path)
        if actions_path.exists()
        else pl.DataFrame(
            {"symbol": [], "action_type": [], "ex_date": [], "value": []},
            schema={
                "symbol": pl.Utf8,
                "action_type": pl.Utf8,
                "ex_date": pl.Date,
                "value": pl.Float64,
            },
        )
    )
    yf_failed: set[str] = set()
    if coverage_path.exists():
        cov = pl.read_csv(coverage_path)
        yf_failed = set(cov.filter(pl.col("status") == "failed")["symbol"].to_list())
    logger.info("Loaded {} action rows; {} yfinance-blind symbols", actions.height, len(yf_failed))

    initial_rows = df.height
    initial_syms = df["symbol"].n_unique()

    # ------------------------------------------------------------------
    # Rule 1 — trading-calendar filter
    # ------------------------------------------------------------------
    calendar = build_trading_calendar(df, start=date(2003, 1, 1), end=date(2021, 6, 30))
    df, r1 = apply_calendar_filter(df, calendar)

    # ------------------------------------------------------------------
    # Rule 2 — patch residual splits
    # ------------------------------------------------------------------
    df, r2 = apply_residual_splits(df, RESIDUAL_SPLITS)
    smoothness = verify_split_smoothness(df, RESIDUAL_SPLITS, tolerance=0.15)
    r2["smoothness_check"] = smoothness

    # ------------------------------------------------------------------
    # Rule 3 — trim phantom history
    # ------------------------------------------------------------------
    df, r3 = trim_phantom_history(
        df,
        threshold=settings.cleaning.phantom_jump_threshold,
        splits=RESIDUAL_SPLITS,
        structural=STRUCTURAL_EVENTS,
        keep=KEEP_EVENTS,
    )

    # ------------------------------------------------------------------
    # Rule 4 — structural-event windowing
    # ------------------------------------------------------------------
    df, r4 = apply_structural_events(df, STRUCTURAL_EVENTS)

    # ------------------------------------------------------------------
    # Rule 5 — liquidity floor
    # ------------------------------------------------------------------
    df, r5 = apply_liquidity_filter(
        df,
        min_adv_inr=settings.liquidity.min_adv_inr,
        window=settings.liquidity.rolling_window,
        min_periods=settings.liquidity.rolling_min_periods,
    )

    # ------------------------------------------------------------------
    # Rule 7 — contiguity filter (between ADV floor and min-history)
    # ------------------------------------------------------------------
    df, r7 = apply_contiguity_filter(
        df,
        max_gap_days=settings.cleaning.contiguity_max_gap_days,
        prefer_overlap_after=settings.cleaning.contiguity_prefer_overlap_after,
    )

    # ------------------------------------------------------------------
    # Rule 6 — minimum history
    # ------------------------------------------------------------------
    df, r6 = apply_min_history(df, min_days=settings.universe.min_history_days)

    # ------------------------------------------------------------------
    # Write daily_clean
    # ------------------------------------------------------------------
    out_path = processed("daily_clean.parquet")
    df.sort(["symbol", "date"]).write_parquet(out_path, use_pyarrow=True)
    logger.info(
        "Wrote daily_clean: {:,} rows, {} symbols → {}",
        df.height,
        df["symbol"].n_unique(),
        out_path,
    )

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------
    gate = validation_gate(
        df,
        actions,
        start_date=settings.cleaning.validation_start,
        threshold=settings.cleaning.validation_jump_threshold,
        keep=KEEP_EVENTS,
        yfinance_failed_symbols=yf_failed,
    )

    # Magnitude buckets for the validation-gate survivors (very useful for
    # triage: >100% is almost always an unrecorded corporate action; 40-100%
    # is a mix of real extreme moves + unrecorded CA).
    survivors = gate["survivors"]
    big_100 = [s for s in survivors if abs(s["ret"]) > 1.0]
    mid = [s for s in survivors if 0.40 < abs(s["ret"]) <= 1.0]
    gate["survivors_over_100pct"] = len(big_100)
    gate["survivors_40_to_100pct"] = len(mid)

    full_report = {
        "input": {
            "path": str(src_path),
            "rows": initial_rows,
            "symbols": initial_syms,
        },
        "output": {
            "path": str(out_path),
            "rows": df.height,
            "symbols": df["symbol"].n_unique(),
            "date_min": df["date"].min(),
            "date_max": df["date"].max(),
        },
        "rules": [r1, r2, r3, r4, r5, r7, r6],
        "validation_gate": gate,
    }
    report_path = reports("daily_clean_report.json")
    _write_report(full_report, report_path)
    logger.info("Cleaning report written → {}", report_path)

    # CSV of survivors for easy triage in spreadsheets.
    if survivors:
        surv_df = pl.DataFrame(survivors).with_columns(
            pl.col("symbol").is_in(yf_failed).alias("yfinance_blind")
        )
        surv_path = reports("daily_clean_validation_survivors.csv")
        surv_df.sort([pl.col("ret").abs()], descending=True).write_csv(surv_path)
        logger.info("Survivors CSV → {}", surv_path)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n=== 05_clean_align complete ===")
    print(f"  Input  : {initial_rows:,} rows × {initial_syms} symbols  ({src_path})")
    print(f"  Output : {df.height:,} rows × {df['symbol'].n_unique()} symbols  ({out_path})")
    print(f"           date span: {df['date'].min()} → {df['date'].max()}")
    print()
    print("  --- Rule 1 calendar filter ---")
    print(f"    rows_dropped     : {r1['rows_dropped']:,}")
    print(f"    weekend_dropped  : {r1['weekend_rows_dropped']:,}")
    print(f"    jan1_dropped     : {r1['jan1_rows_dropped']:,}")
    print(f"    calendar         : {r1['calendar_days']:,} days  (~{r1['days_per_year_avg']}/yr)")
    print()
    print("  --- Rule 2 residual-split patch ---")
    print(f"    total_rows_adjusted: {r2['rows_adjusted']:,}")
    for s in r2["splits"]:
        print(
            f"      {s['symbol']:<12} {s['ex_date']}  {int(s['split_ratio'])}:1  "
            f"adjusted_rows={s['rows_adjusted']:,}"
        )
    print("    smoothness check (post-patch close return at ex-date, target |ret|<0.15):")
    for s in r2["smoothness_check"]:
        ret = s["post_patch_return"]
        ret_s = f"{ret:+.4f}" if ret is not None else "n/a"
        print(
            f"      {s['symbol']:<12} {s['ex_date']}  return={ret_s}  "
            f"smooth={s['smooth']} {s['note']}"
        )
    print()
    print(f"  --- Rule 3 phantom-trim (threshold {settings.cleaning.phantom_jump_threshold}) ---")
    print(f"    rows_dropped    : {r3['rows_dropped']:,}")
    print(f"    symbols_trimmed : {len(r3['trimmed_symbols'])}")
    for t in r3["trimmed_symbols"]:
        print(f"      {t['symbol']:<12} new_start = {t['new_start_date']}")
    print()
    print("  --- Rule 4 structural events ---")
    print(f"    rows_dropped : {r4['rows_dropped']:,}")
    for e in r4["events"]:
        print(f"      {e['symbol']:<12} event = {e['event_date']}  (keep date > event)")
    print()
    print(f"  --- Rule 5 liquidity floor (ADV >= ₹{settings.liquidity.min_adv_inr:,.0f}) ---")
    print(f"    rows_dropped   : {r5['rows_dropped']:,}")
    print(f"    symbols_lost   : {len(r5['symbols_dropped_entirely'])}")
    if r5["symbols_dropped_entirely"]:
        sample = ", ".join(r5["symbols_dropped_entirely"][:15])
        more = "" if len(r5["symbols_dropped_entirely"]) <= 15 else " ..."
        print(f"      {sample}{more}")
    print()
    print(
        f"  --- Rule 7 contiguity (gap > {r7['max_gap_days']} calendar days "
        f"splits; prefer overlap >= {r7['prefer_overlap_after']}) ---"
    )
    print(f"    rows_dropped         : {r7['rows_dropped']:,}")
    print(f"    symbols_segmented    : {r7['n_symbols_segmented']}")
    print(f"    segments_total/dropped: {r7['n_segments_total']} / {r7['n_segments_dropped']}")
    # Sanity: every kept symbol's max gap must now be <= threshold
    gap_summary = max_internal_gap_per_symbol(df)
    print(
        f"    post-Rule-7 max internal gap (any symbol): "
        f"{int(gap_summary['max_internal_gap_days'].max())} days"
    )
    print()
    print(f"  --- Rule 6 min-history (>= {settings.universe.min_history_days} days) ---")
    print(f"    symbols_kept    : {r6['symbols_kept']}")
    print(f"    symbols_dropped : {len(r6['symbols_dropped'])}")
    if r6["symbols_dropped"]:
        sample = ", ".join(f"{d['symbol']}({d['n_days']})" for d in r6["symbols_dropped"][:15])
        more = "" if len(r6["symbols_dropped"]) <= 15 else " ..."
        print(f"      {sample}{more}")
    print()
    print("  --- Validation gate ---")
    print(
        f"    Re-scan: |close[t]/close[t-1] - 1| > {gate['threshold']:.2f} "
        f"from {gate['start_date']}"
    )
    print(f"    big moves              : {gate['n_big_moves']}")
    print(f"    survivors              : {gate['n_survivors']}")
    print(
        f"      >100% magnitude      : {gate['survivors_over_100pct']}"
        "   (almost certainly unrecorded corporate actions)"
    )
    print(
        f"      40-100% magnitude    : {gate['survivors_40_to_100pct']}"
        "   (mix of real extreme moves + unrecorded CA)"
    )
    if survivors:
        top = sorted(survivors, key=lambda r: abs(r["ret"]), reverse=True)[:25]
        print("    top 25 by magnitude:")
        for s in top:
            print(
                f"      {s['symbol']:<12} {s['date']}  close={s['close']:>10}  "
                f"return={s['ret']:+.4f}"
            )
        if len(survivors) > 25:
            print(f"      … {len(survivors) - 25} more — see survivors CSV")
    print(
        f"    yfinance-blind universe: {gate['n_yfinance_blind_total']} symbols; "
        f"survivors among them: {gate['n_yfinance_blind_survivors']}"
    )
    print(f"    GATE: {'PASS' if gate['pass'] else 'FAIL'}")
    print(f"    report json            : {report_path}")

    assert gate["pass"], (
        f"Validation gate FAILED — {gate['n_survivors']} unexplained "
        f"discontinuit{'y' if gate['n_survivors'] == 1 else 'ies'} remain"
    )


if __name__ == "__main__":
    main()
