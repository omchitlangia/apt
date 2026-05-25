#!/usr/bin/env python3
"""Script 09: Phase 1 Day 7 — spread + rolling z-score + signal diagnostics.

Loads the Day-6 ``cointegrated_pairs.parquet`` survivors, filters down to the
tradeable subset (``fdr_pass & half_life_in_band & hurst_pass &
stable_prior_window``), and for each one:

  1. aligns the two legs' closing prices on the train window;
  2. computes the OLS-residual spread via :func:`apt.signals.spread.compute_spread`
     using the (alpha, beta) Day-6 carried forward — NEVER re-fit here;
  3. computes the trailing rolling z-score via :func:`rolling_zscore`;
  4. generates entry/exit/stop signals via :func:`generate_signals` with
     per-pair ``max_holding = min(cap, ceil(half_life * multiplier))``;
  5. summarises via :func:`signal_diagnostics`;
  6. saves a two-panel diagnostic plot (spread on top, z + bands + position
     shading on the bottom).

Outputs:
  * ``reports/spread_signal_diagnostics.csv`` — one row per tradeable pair.
  * ``plots/phase1/pairs/signal_<y_sym>_<x_sym>.png`` — per-pair plot.

No backtester. No P&L. The point is to size the signal layer — trade count,
% time in position, stop frequency — so we know how much data Phase 2 sees
per pair.
"""

from __future__ import annotations

import math

import polars as pl
from loguru import logger

from apt.config import settings
from apt.plots.pairs import plot_spread_zscore_signal
from apt.signals.spread import (
    compute_spread,
    generate_signals,
    rolling_zscore,
    signal_diagnostics,
)
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, pairs, processed, reports


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "09_spread_signals.log")
    ensure_dirs()

    pairs_path = pairs("cointegrated_pairs.parquet")
    daily_path = processed("daily_clean.parquet")
    if not pairs_path.exists():
        raise FileNotFoundError(f"{pairs_path} missing — run scripts/08 first")
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing — run scripts/05 first")

    cp = pl.read_parquet(pairs_path)
    daily = pl.read_parquet(daily_path)
    logger.info(
        "Loaded {} candidate pair rows; daily has {:,} rows, {} symbols",
        cp.height,
        daily.height,
        daily["symbol"].n_unique(),
    )

    # Tradeable subset — Day-6 boolean flags
    tradeable = cp.filter(
        pl.col("fdr_pass")
        & pl.col("half_life_in_band")
        & pl.col("hurst_pass")
        & pl.col("stable_prior_window")
    ).sort("adf_pvalue")
    logger.info("Tradeable pairs: {}", tradeable.height)

    if tradeable.is_empty():
        print("\n=== 09_spread_signals: nothing to do (zero tradeable pairs) ===")
        return

    # The train window the Day-6 betas were fit on. We diagnose on the SAME
    # window (this is the in-sample signal-engine sanity check, not OOS P&L).
    trading_days = daily.select(pl.col("date").unique()).sort("date")["date"].to_list()
    n_train = min(settings.cointegration.n_train_days, len(trading_days))
    train_end = trading_days[-1]
    train_start = trading_days[-n_train]
    logger.info("Train window: {} → {} ({} trading days)", train_start, train_end, n_train)

    plot_dir = settings.paths.plots_dir / "phase1" / "pairs"
    plot_dir.mkdir(parents=True, exist_ok=True)

    diag_rows: list[dict] = []
    plot_paths: list = []

    for r in tradeable.iter_rows(named=True):
        y_sym = r["y_sym"]
        x_sym = r["x_sym"]
        alpha = r["alpha"]
        beta = r["beta"]
        half_life = r["half_life"]

        max_holding = max(
            2,
            min(
                settings.signal.max_holding_cap_days,
                int(math.ceil(half_life * settings.signal.max_holding_half_life_multiplier)),
            ),
        )

        win = (
            daily.filter(
                pl.col("symbol").is_in([y_sym, x_sym])
                & (pl.col("date") >= train_start)
                & (pl.col("date") <= train_end)
            )
            .select(["symbol", "date", "close"])
            .sort("date")
        )
        wide = win.pivot(index="date", on="symbol", values="close").sort("date").drop_nulls()
        if wide.is_empty() or y_sym not in wide.columns or x_sym not in wide.columns:
            logger.warning("Skipping {}/{} — could not align prices", y_sym, x_sym)
            continue

        dates = wide["date"].to_list()
        p_y = wide[y_sym].to_numpy()
        p_x = wide[x_sym].to_numpy()

        spread = compute_spread(p_y, p_x, beta=beta, intercept=alpha)
        z = rolling_zscore(spread, window=settings.spread.rolling_window)
        sig = generate_signals(
            z,
            entry=settings.signal.entry_z,
            exit=settings.signal.exit_z,
            stop=settings.signal.stop_z,
            max_holding=max_holding,
        )
        diag = signal_diagnostics(sig)

        fname = f"signal_{y_sym}_{x_sym}.png".replace("&", "AND")
        plot_path = plot_dir / fname
        plot_spread_zscore_signal(
            dates,
            spread,
            z,
            sig.position,
            y_sym=y_sym,
            x_sym=x_sym,
            out_path=plot_path,
            entry=settings.signal.entry_z,
            exit_threshold=settings.signal.exit_z,
            stop=settings.signal.stop_z,
            sector=r["sector"],
            extra_title=(
                f"β={beta:+.4f}  half-life={half_life:.1f}d  "
                f"max-hold={max_holding}d  roll-win={settings.spread.rolling_window}d"
            ),
        )
        plot_paths.append(plot_path)

        diag_rows.append(
            {
                "pair": f"{y_sym}/{x_sym}",
                "sector": r["sector"],
                "beta": beta,
                "half_life": half_life,
                "max_holding": max_holding,
                "rolling_window": settings.spread.rolling_window,
                **diag,
            }
        )

    diag_df = pl.DataFrame(diag_rows)
    out_csv = reports("spread_signal_diagnostics.csv")
    diag_df.write_csv(out_csv)
    logger.info("Wrote {} diagnostic rows → {}", diag_df.height, out_csv)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== 09_spread_signals complete ===")
    print(f"  Train window     : {train_start} → {train_end} ({n_train} trading days)")
    print(
        f"  Thresholds       : entry |z|={settings.signal.entry_z}  "
        f"exit |z|={settings.signal.exit_z}  stop |z|={settings.signal.stop_z}  "
        f"roll-win={settings.spread.rolling_window}d"
    )
    print(
        f"  max_holding rule : min({settings.signal.max_holding_cap_days}d, "
        f"⌈{settings.signal.max_holding_half_life_multiplier:g}·half_life⌉)"
    )
    print(f"  Tradeable pairs  : {tradeable.height}")
    print(f"  Output (CSV)     : {out_csv}")
    for p in plot_paths:
        print(f"  Plot             : {p}")
    print()
    print("  --- Per-pair signal diagnostics ---")
    header = (
        f"  {'pair':<26}  {'trades':>6}  {'avg_hold':>9}  {'%inpos':>7}  "
        f"{'rev':>3}  {'stop':>4}  {'time':>4}  {'open?':>5}  {'max_hold':>8}"
    )
    print(header)
    for r in diag_rows:
        avg = r["avg_holding_days"]
        avg_s = f"{avg:.1f}d" if avg is not None else "  —  "
        print(
            f"  {r['pair']:<26}  {r['n_round_trips']:>6}  {avg_s:>9}  "
            f"{r['pct_time_in_position'] * 100:>6.1f}%  "
            f"{r['n_exits_mean_revert']:>3}  {r['n_exits_stop']:>4}  "
            f"{r['n_exits_time']:>4}  {'yes' if r['n_open_at_end'] else 'no':>5}  "
            f"{r['max_holding']:>7}d"
        )
    print()
    # Roll-ups for the whole tradeable universe
    total_trades = sum(r["n_round_trips"] for r in diag_rows)
    total_stops = sum(r["n_exits_stop"] for r in diag_rows)
    total_time = sum(r["n_exits_time"] for r in diag_rows)
    total_rev = sum(r["n_exits_mean_revert"] for r in diag_rows)
    avg_pct_inpos = sum(r["pct_time_in_position"] for r in diag_rows) / max(1, len(diag_rows))
    print(
        f"  Universe roll-up: {total_trades} round-trips across {len(diag_rows)} pairs  "
        f"(mean-rev={total_rev}, stop={total_stops}, time={total_time})  "
        f"avg %-time-in-position={avg_pct_inpos * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
