#!/usr/bin/env python3
"""Script 04: corporate-action cache + adjustment verification + passthrough.

Day-3 of the project. The daily and minute datasets are already split- and
bonus-adjusted; this script verifies that fact, caches yfinance's corporate-
action timeline for the universe, diagnoses dividend-flavor for a few
high-dividend names, and writes a validated passthrough copy of
``daily_raw.parquet`` to ``data/processed/daily_adjusted.parquet``.

No re-adjustment is applied to prices.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from loguru import logger

from apt.config import settings
from apt.data.corporate_actions import (
    DIVIDEND_FLAVOR_PROBES,
    audit_split_residuals,
    classify_adjustment_flavor,
    fetch_all_corporate_actions,
    passthrough_daily_adjusted,
    write_corporate_actions_cache,
)
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, processed, reports


def _format_audit_failure_rows(failures: pl.DataFrame, n: int = 25) -> str:
    if failures.is_empty():
        return "  (none)"
    head = failures.sort("ex_date").head(n)
    out_lines = []
    for row in head.to_dicts():
        out_lines.append(
            f"  {row['symbol']:<12} {row['ex_date']} "
            f"split={row['split_ratio']:>5.2f} "
            f"observed={row['observed_ratio']:.4f} "
            f"expected_unadj={row['expected_unadjusted_ratio']:.4f}"
        )
    if failures.height > n:
        out_lines.append(f"  … {failures.height - n} more")
    return "\n".join(out_lines)


def _yesbank_gate(daily: pl.DataFrame) -> dict:
    """YESBANK 2017-09-25 must be smooth (no residual split jump).

    Looks up the close on 2017-09-25 and the prior trading day; reports the
    return. With an already-adjusted series the return should be small in
    magnitude (well within ±15%).
    """
    target = date(2017, 9, 25)
    sym = (
        daily.filter(pl.col("symbol") == "YESBANK")
        .sort("date")
        .with_columns(pl.col("close").shift(1).alias("prev_close"))
        .filter(pl.col("date") == target)
    )
    if sym.is_empty():
        return {"date": target, "found": False, "pass": False, "note": "no row in daily_raw"}
    row = sym.to_dicts()[0]
    if row["prev_close"] is None or row["prev_close"] == 0:
        return {"date": target, "found": True, "pass": False, "note": "no prev_close"}
    obs = row["close"] / row["prev_close"]
    ret = obs - 1.0
    # A residual 5:1 split would show observed_ratio ≈ 0.2 (return -80%).
    # We require |return| < 0.15 for "smooth".
    return {
        "date": target,
        "found": True,
        "close": row["close"],
        "prev_close": row["prev_close"],
        "observed_ratio": obs,
        "return": ret,
        "pass": abs(ret) < 0.15,
        "note": "" if abs(ret) < 0.15 else "return out of bounds",
    }


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "04_corporate_actions.log")
    ensure_dirs()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    daily_raw_path = interim("daily_raw.parquet")
    if not daily_raw_path.exists():
        raise FileNotFoundError(
            f"daily_raw.parquet not found at {daily_raw_path} — run scripts/01_ingest_daily.py first"
        )
    daily = pl.read_parquet(daily_raw_path)
    symbols = sorted(daily["symbol"].unique().to_list())
    logger.info("Loaded daily_raw: {} rows, {} symbols", daily.height, len(symbols))

    # ------------------------------------------------------------------
    # Step 1 — fetch + cache yfinance corporate actions
    # ------------------------------------------------------------------
    logger.info("Step 1 — fetching yfinance corporate actions for {} symbols", len(symbols))
    actions, coverage = fetch_all_corporate_actions(symbols, n_jobs=8, max_retries=3)
    n_ok = int(coverage.filter(pl.col("status") == "ok").height)
    n_fail = int(coverage.filter(pl.col("status") == "failed").height)
    n_with_splits = int(coverage.filter(pl.col("n_splits") > 0).height)
    n_with_divs = int(coverage.filter(pl.col("n_dividends") > 0).height)
    logger.info(
        "Step 1 result — coverage {}/{} ({} failed); splits in {} symbols, dividends in {}",
        n_ok,
        len(symbols),
        n_fail,
        n_with_splits,
        n_with_divs,
    )

    cache_path = interim("corporate_actions.parquet")
    write_corporate_actions_cache(actions, cache_path)
    coverage_path = reports("corporate_actions_coverage.csv")
    coverage.write_csv(coverage_path)
    logger.info("Cached → {} ; coverage → {}", cache_path, coverage_path)

    # ------------------------------------------------------------------
    # Step 2 — universe split/bonus audit
    # ------------------------------------------------------------------
    logger.info("Step 2 — auditing residual split jumps against daily_raw")
    audit = audit_split_residuals(daily, actions, ratio_tol=0.05)
    audit_path = reports("corporate_actions_audit.csv")
    audit.write_csv(audit_path)

    status_counts = audit.group_by("status").len().sort("len", descending=True).to_dicts()
    failures = audit.filter(pl.col("status") == "residual_jump")
    logger.info("Step 2 result — audit row counts by status: {}", status_counts)
    logger.info("Step 2 result — residual_jump failures: {}", failures.height)

    # ------------------------------------------------------------------
    # Step 3 — dividend-flavor diagnosis
    # ------------------------------------------------------------------
    logger.info("Step 3 — dividend-flavor diagnosis for {}", list(DIVIDEND_FLAVOR_PROBES))
    flavor_results = []
    for sym in DIVIDEND_FLAVOR_PROBES:
        verdict = classify_adjustment_flavor(sym, daily)
        flavor_results.append(verdict)
        logger.info(
            "Step 3 — {}: n_overlap={}, cv_vs_close={:.6f}, cv_vs_adj_close={:.6f}, "
            "verdict={} ({})",
            verdict["symbol"],
            verdict["n_overlap"],
            verdict["cv_vs_close"] if verdict["cv_vs_close"] == verdict["cv_vs_close"] else 0.0,
            verdict["cv_vs_adj_close"]
            if verdict["cv_vs_adj_close"] == verdict["cv_vs_adj_close"]
            else 0.0,
            verdict["verdict"],
            verdict["confidence"],
        )

    flavor_df = pl.DataFrame(flavor_results)
    flavor_path = reports("dividend_flavor_diagnosis.csv")
    flavor_df.write_csv(flavor_path)

    # ------------------------------------------------------------------
    # Step 4 — validated passthrough
    # ------------------------------------------------------------------
    logger.info("Step 4 — passthrough daily_raw → daily_adjusted (no re-adjustment)")
    out_path = processed("daily_adjusted.parquet")
    pt = passthrough_daily_adjusted(daily_raw_path, out_path)

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    yes_gate = _yesbank_gate(daily)
    pt_rows_match = pt["rows"] == daily.height
    pt_sym_match = pt["symbols"] == daily["symbol"].n_unique() == 492

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n=== 04_corporate_actions complete ===")
    print(f"  Universe              : {len(symbols)} symbols")
    print(
        f"  Step 1 coverage       : {n_ok}/{len(symbols)} fetched ok ({n_fail} failed); "
        f"{n_with_splits} have splits, {n_with_divs} have dividends"
    )
    print(f"  Step 1 cache          : {cache_path}  ({actions.height:,} action rows)")
    print(f"  Step 1 coverage csv   : {coverage_path}")
    print()
    print("  Step 2 audit status counts:")
    for r in status_counts:
        print(f"    {r['status']:<14} {r['len']:>5}")
    print(f"  Step 2 audit csv      : {audit_path}")
    print("  Step 2 residual-jump failures (expected empty):")
    print(_format_audit_failure_rows(failures))
    print()
    print("  Step 3 dividend-flavor verdicts:")
    for v in flavor_results:
        cvc = v["cv_vs_close"]
        cva = v["cv_vs_adj_close"]
        cvc_s = f"{cvc:.6f}" if cvc == cvc else "nan"
        cva_s = f"{cva:.6f}" if cva == cva else "nan"
        print(
            f"    {v['symbol']:<12} verdict={v['verdict']:<14} confidence={v['confidence']:<6} "
            f"cv_close={cvc_s} cv_adj_close={cva_s} n_overlap={v['n_overlap']}"
        )
    print(f"  Step 3 csv            : {flavor_path}")
    print()
    print("  Step 4 passthrough:")
    print(f"    Rows     : {pt['rows']:,}  (source {daily.height:,})  → match={pt_rows_match}")
    print(f"    Symbols  : {pt['symbols']}  → match==492={pt_sym_match}")
    print(f"    Date span: {pt['min_date']} → {pt['max_date']}")
    print(f"    Output   : {out_path}")
    print()
    print("  --- Gates ---")
    print(f"  [{'PASS' if yes_gate['pass'] else 'FAIL'}] YESBANK 2017-09-25 smooth: {yes_gate}")
    print(f"  [{'PASS' if pt_rows_match else 'FAIL'}] daily_adjusted rowcount == daily_raw")
    print(f"  [{'PASS' if pt_sym_match else 'FAIL'}] daily_adjusted symbols == 492")

    # Hard assertions so the script exits non-zero on a gate failure.
    assert yes_gate["pass"], f"YESBANK gate failed: {yes_gate}"
    assert pt_rows_match, "passthrough row count mismatch"
    assert pt_sym_match, "passthrough symbol count != 492"


if __name__ == "__main__":
    main()
