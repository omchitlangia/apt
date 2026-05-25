#!/usr/bin/env python3
"""Script 10: Phase 2A — walk-forward classical backtest with first P&L.

Wires :func:`apt.backtest.walkforward.run_walkforward` to the APT pipeline:
  * ``select_pairs_fn``  → cointegrate_pairs on (prior, train) windows, filter
    to ``fdr_pass & half_life_in_band & hurst_pass & stable_prior_window``.
  * ``get_prices_fn``    → aligned close prices from ``daily_clean.parquet``.

Fixed equal-notional sizing per active pair (Phase 2A baseline). Inverse-vol
and z-proportional sizing come in 2B.

Outputs:
  * ``reports/backtest_portfolio_daily.csv``   — date, gross_log_ret, net_log_ret, n_active
  * ``reports/backtest_trades.csv``            — one row per round-trip
  * ``reports/backtest_per_pair_metrics.csv``  — per pair, lifetime metrics
  * ``reports/backtest_portfolio_metrics.csv`` — portfolio gross/net metrics (single row)
  * ``reports/backtest_caveats.txt``           — the non-negotiable honesty list
  * ``plots/phase2/backtest/equity_curve.png`` — gross/net + drawdown + active-pair count
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

import numpy as np
import polars as pl
from loguru import logger

from apt.backtest import Pair, build_folds, compute_metrics, run_walkforward
from apt.config import settings
from apt.plots.backtest import plot_equity_curve
from apt.signals.cointegration import cointegrate_pairs
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, processed, reports

CAVEATS = [
    "(1) MULTI-DAY SHORTING ON INDIAN CASH EQUITY IS NOT EXECUTABLE — this is "
    "an upper-bound POC P&L, not a live-tradeable strategy. The real venue for "
    "this engine is crypto / US futures / FX, where shorts settle normally.",
    "(2) Heavy shared-leg / financials concentration (HDFCBANK alone touches "
    "~⅓ of distinct tradeable pairs across the backtest) — pair returns are "
    "NOT independent and Sharpe is overstated when treated as if they were.",
    "(3) Thin per-fold pair count (typically 0–7 tradeable pairs) → wide error "
    "bars on every metric. A single bad fold can dominate the lifetime number.",
    "(4) Dividends not modeled. Equity prices are NOT dividend-adjusted (only "
    "splits/bonuses are). A short leg PAYS net of dividend on ex-date; a long "
    "leg RECEIVES it. Net effect for a balanced spread averages near zero but "
    "individual pairs can drift ~1–2 %/year. State the bias; don't capitalise it.",
    "(5) Cost model is a STATIC per-leg round-trip bps figure; real slippage "
    "spikes under stress (gap opens, illiquid stocks). The 25 bps default is "
    "mid-range — stress this in 2B once the sizing logic is in.",
]


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "10_backtest.log")
    ensure_dirs()

    daily_path = processed("daily_clean.parquet")
    sectors_path = interim("sectors.parquet")
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing — run scripts/05 first")
    if not sectors_path.exists():
        raise FileNotFoundError(f"{sectors_path} missing — run scripts/03 first")

    daily = pl.read_parquet(daily_path)
    sectors = pl.read_parquet(sectors_path)
    trading_days = daily.select(pl.col("date").unique()).sort("date")["date"].to_list()
    logger.info(
        "Loaded daily_clean ({:,} rows, {} symbols, {} trading days)",
        daily.height,
        daily["symbol"].n_unique(),
        len(trading_days),
    )

    folds = build_folds(
        trading_days,
        prior_days=settings.cointegration.n_train_days,
        train_days=settings.cointegration.n_train_days,
        test_days=settings.backtest.test_days_per_fold,
        step_days=settings.backtest.step_days,
    )
    logger.info("Built {} fold(s)", len(folds))

    # ------------------------------------------------------------------
    # APT-specific callbacks
    # ------------------------------------------------------------------
    def select_pairs_fn(
        prior_start: date,
        prior_end: date,
        train_start: date,
        train_end: date,
    ) -> list[Pair]:
        res = cointegrate_pairs(
            daily,
            sectors,
            start=train_start,
            end=train_end,
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
        tradeable = res.pairs.filter(
            pl.col("fdr_pass")
            & pl.col("half_life_in_band")
            & pl.col("hurst_pass")
            & pl.col("stable_prior_window")
        )
        return [
            Pair(
                y_sym=r["y_sym"],
                x_sym=r["x_sym"],
                alpha=float(r["alpha"]),
                beta=float(r["beta"]),
                half_life=float(r["half_life"]),
                sector=r["sector"],
                is_structural=bool(r["is_structural_pair"]),
            )
            for r in tradeable.iter_rows(named=True)
        ]

    def get_prices_fn(
        y_sym: str, x_sym: str, start: date, end: date
    ) -> tuple[list[date], np.ndarray, np.ndarray]:
        win = (
            daily.filter(
                pl.col("symbol").is_in([y_sym, x_sym])
                & (pl.col("date") >= start)
                & (pl.col("date") <= end)
            )
            .select(["symbol", "date", "close"])
            .sort("date")
        )
        wide = win.pivot(index="date", on="symbol", values="close").sort("date").drop_nulls()
        if wide.is_empty() or y_sym not in wide.columns or x_sym not in wide.columns:
            return [], np.empty(0), np.empty(0)
        return wide["date"].to_list(), wide[y_sym].to_numpy(), wide[x_sym].to_numpy()

    # ------------------------------------------------------------------
    # Run the engine
    # ------------------------------------------------------------------
    result = run_walkforward(
        folds,
        trading_days,
        select_pairs_fn=select_pairs_fn,
        get_prices_fn=get_prices_fn,
        rolling_window=settings.spread.rolling_window,
        entry_z=settings.signal.entry_z,
        exit_z=settings.signal.exit_z,
        stop_z=settings.signal.stop_z,
        max_holding_cap_days=settings.signal.max_holding_cap_days,
        max_holding_half_life_multiplier=settings.signal.max_holding_half_life_multiplier,
        cost_bps_per_leg=settings.backtest.cost_bps_per_leg,
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    port = result.portfolio_daily
    metrics_gross = (
        compute_metrics(port["gross_log_ret"].to_numpy())
        if not port.is_empty()
        else compute_metrics([])
    )
    metrics_net = (
        compute_metrics(port["net_log_ret"].to_numpy())
        if not port.is_empty()
        else compute_metrics([])
    )

    per_pair_rows: list[dict] = []
    pair_trades_count: Counter[str] = Counter()
    for t in result.trades:
        pair_trades_count[t.pair_key] += 1
    for pkey, daily_df in result.per_pair_daily.items():
        m_gross = compute_metrics(daily_df["gross_log_ret"].to_numpy())
        m_net = compute_metrics(daily_df["net_log_ret"].to_numpy())
        per_pair_rows.append(
            {
                "pair": pkey,
                "n_trades": int(pair_trades_count.get(pkey, 0)),
                "n_obs": int(daily_df.height),
                "gross_total_return_pct": m_gross["total_return_pct"],
                "net_total_return_pct": m_net["total_return_pct"],
                "gross_ann_return_pct": m_gross["ann_return_pct"],
                "net_ann_return_pct": m_net["ann_return_pct"],
                "gross_sharpe": m_gross["sharpe"],
                "net_sharpe": m_net["sharpe"],
                "net_max_drawdown_pct": m_net["max_drawdown_pct"],
            }
        )
    per_pair_df = (
        pl.DataFrame(per_pair_rows).sort("net_sharpe", descending=True, nulls_last=True)
        if per_pair_rows
        else pl.DataFrame()
    )

    # Trade-level stats
    n_trades = len(result.trades)
    n_winners_net = sum(1 for t in result.trades if t.net_log_pnl > 0)
    n_winners_gross = sum(1 for t in result.trades if t.gross_log_pnl > 0)
    n_stops = sum(1 for t in result.trades if t.exit_reason == "stop")
    n_mean_rev = sum(1 for t in result.trades if t.exit_reason == "mean_revert")
    n_time = sum(1 for t in result.trades if t.exit_reason == "time")
    n_boundary = sum(1 for t in result.trades if t.exit_reason == "fold_boundary")
    avg_gross_pnl_pct = (
        float(np.mean([np.expm1(t.gross_log_pnl) * 100 for t in result.trades]))
        if n_trades
        else 0.0
    )
    avg_net_pnl_pct = (
        float(np.mean([np.expm1(t.net_log_pnl) * 100 for t in result.trades])) if n_trades else 0.0
    )
    avg_hold = float(np.mean([t.days_held for t in result.trades])) if n_trades else 0.0
    win_rate_net = n_winners_net / n_trades if n_trades else 0.0
    win_rate_gross = n_winners_gross / n_trades if n_trades else 0.0

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    port_csv = reports("backtest_portfolio_daily.csv")
    if not port.is_empty():
        port.write_csv(port_csv)

    trades_rows = [
        {
            "fold_id": t.fold_id,
            "pair": t.pair_key,
            "direction": t.direction,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_z": t.entry_z,
            "exit_z": t.exit_z,
            "days_held": t.days_held,
            "gross_log_pnl": t.gross_log_pnl,
            "cost_log": t.cost_log,
            "net_log_pnl": t.net_log_pnl,
            "gross_pct": float(np.expm1(t.gross_log_pnl) * 100),
            "net_pct": float(np.expm1(t.net_log_pnl) * 100),
            "exit_reason": t.exit_reason,
        }
        for t in result.trades
    ]
    trades_csv = reports("backtest_trades.csv")
    pl.DataFrame(trades_rows).write_csv(trades_csv) if trades_rows else None

    pp_csv = reports("backtest_per_pair_metrics.csv")
    if not per_pair_df.is_empty():
        per_pair_df.write_csv(pp_csv)

    port_metrics_csv = reports("backtest_portfolio_metrics.csv")
    port_metrics_rows = [
        {
            "name": k,
            "gross": metrics_gross[k],
            "net": metrics_net[k],
        }
        for k in (
            "n_obs",
            "n_years",
            "total_return_pct",
            "ann_return_pct",
            "ann_vol_pct",
            "sharpe",
            "max_drawdown_pct",
        )
    ]
    pl.DataFrame(port_metrics_rows).write_csv(port_metrics_csv)

    funnel_path = reports("backtest_funnel.json")
    with funnel_path.open("w") as fh:
        json.dump(result.funnel, fh, indent=2, default=str)

    caveats_path = reports("backtest_caveats.txt")
    with caveats_path.open("w") as fh:
        fh.write("Walk-forward backtest caveats (Phase 2A)\n")
        fh.write("=" * 60 + "\n\n")
        for c in CAVEATS:
            fh.write(c + "\n\n")

    plot_path = settings.paths.plots_dir / "phase2" / "backtest" / "equity_curve.png"
    plot_stats = plot_equity_curve(
        port,
        plot_path,
        title=(
            "Walk-forward portfolio  |  cointegration pairs, fixed equal notional  |  "
            f"cost = {settings.backtest.cost_bps_per_leg:g} bps per leg round-trip"
        ),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== 10_backtest complete ===")
    print(f"  Folds (test windows)  : {len(folds)}")
    if folds:
        print(
            f"  Test span             : {folds[0].test_start} → {folds[-1].test_end}  "
            f"({sum(1 for _ in folds) * settings.backtest.test_days_per_fold} td)"
        )
    print(
        f"  Cost per leg roundtrip: {settings.backtest.cost_bps_per_leg:g} bps  "
        f"(⇒ {2 * settings.backtest.cost_bps_per_leg:g} bps total per trade)"
    )
    print()
    print("  --- Trade funnel ---")
    avg_pairs_per_fold = result.funnel["n_pair_selections"] / max(1, result.funnel["n_folds"])
    print(f"    avg tradeable pairs per fold : {avg_pairs_per_fold:.2f}")
    print(f"    pair × fold units processed  : {result.funnel['n_pair_fold_units']}")
    print(f"    total round-trips            : {n_trades}")
    print(
        f"      └── exit_reason breakdown  : "
        f"mean_revert={n_mean_rev}  stop={n_stops}  "
        f"time={n_time}  fold_boundary={n_boundary}"
    )
    print(
        f"    win rate (gross / net)       : "
        f"{win_rate_gross * 100:.1f}% / {win_rate_net * 100:.1f}%"
    )
    print(f"    avg P&L per trade (g / n)    : {avg_gross_pnl_pct:+.3f}% / {avg_net_pnl_pct:+.3f}%")
    print(f"    avg holding days             : {avg_hold:.1f}")
    print()
    print("  --- Portfolio metrics ---")
    print(f"    {'metric':<26}  {'gross':>10}  {'net':>10}  {'delta':>9}")
    for k, label in (
        ("n_obs", "trading days"),
        ("n_years", "years"),
        ("total_return_pct", "total return (%)"),
        ("ann_return_pct", "ann return (%)"),
        ("ann_vol_pct", "ann vol (%)"),
        ("sharpe", "Sharpe"),
        ("max_drawdown_pct", "max drawdown (%)"),
    ):
        g = metrics_gross[k]
        n = metrics_net[k]
        if k in ("sharpe",):
            delta = (g - n) if (np.isfinite(g) and np.isfinite(n)) else float("nan")
        else:
            delta = g - n
        print(f"    {label:<26}  {g:>10.3f}  {n:>10.3f}  {delta:>9.3f}")
    print()
    print("  --- Per-pair (sorted by net Sharpe) ---")
    if per_pair_df.is_empty():
        print("    (no pairs)")
    else:
        print(
            f"    {'pair':<26}  {'n_trd':>5}  {'gross_ret':>10}  {'net_ret':>10}  "
            f"{'g_sharpe':>8}  {'n_sharpe':>8}  {'maxDD':>7}"
        )
        for r in per_pair_df.iter_rows(named=True):
            print(
                f"    {r['pair']:<26}  {r['n_trades']:>5}  "
                f"{r['gross_total_return_pct']:>9.2f}%  "
                f"{r['net_total_return_pct']:>9.2f}%  "
                f"{r['gross_sharpe']:>8.2f}  {r['net_sharpe']:>8.2f}  "
                f"{r['net_max_drawdown_pct']:>6.2f}%"
            )
    print()
    print(f"  Plot   : {plot_path}   stats={plot_stats}")
    print(f"  Outputs: {port_csv}")
    print(f"           {trades_csv}")
    print(f"           {pp_csv}")
    print(f"           {port_metrics_csv}")
    print(f"           {funnel_path}")
    print(f"           {caveats_path}")
    print()
    print("  --- CAVEATS (non-negotiable honesty) ---")
    for c in CAVEATS:
        print("  • " + c)
    print()
    print("  --- Frank read ---")
    if not n_trades:
        print("    No trades executed. Engine ran but the universe produced nothing tradeable.")
    else:
        g_sh = metrics_gross["sharpe"]
        n_sh = metrics_net["sharpe"]
        g_ret = metrics_gross["ann_return_pct"]
        n_ret = metrics_net["ann_return_pct"]
        cost_drag = g_ret - n_ret
        edge_signal = (
            "YES"
            if (n_sh > 0.5 and n_ret > 5.0)
            else ("MARGINAL" if (n_sh > 0.0 and n_ret > 0.0) else "NO")
        )
        print(
            f"    Cost-surviving edge?  {edge_signal}.  "
            f"Gross Sharpe = {g_sh:.2f}, net Sharpe = {n_sh:.2f};  "
            f"gross ann ret = {g_ret:+.2f}%, net = {n_ret:+.2f}% "
            f"(cost drag = {cost_drag:.2f} pp/year)."
        )
        if edge_signal == "NO":
            print(
                "    Costs eat the gross edge entirely. The classical strategy with "
                "fixed sizing does not survive realistic Indian-equity costs in this "
                "universe."
            )
        elif edge_signal == "MARGINAL":
            print(
                "    Net edge is positive but thin. Whether it survives sizing / "
                "regime stress is what Phase 2B measures."
            )
        else:
            print(
                "    Edge is present and survives costs at the default cost level. "
                "Phase 2B sizing baselines will set the headroom for RL."
            )


if __name__ == "__main__":
    main()
