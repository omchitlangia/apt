#!/usr/bin/env python3
"""Script 11: Phase 2A per-pair backtest diagnostics.

Re-runs the walk-forward backtest in memory (re-using the same callbacks as
``scripts/10_backtest.py``) and produces drill-down artefacts for every pair
that ever traded:

  * ``plots/phase2/pairs/per_pair_overview.png``
      A small-multiples grid: one mini-panel per pair of cumulative net P&L,
      sorted best → worst by total net return. The at-a-glance "who carries
      the strategy and who broke."
  * ``plots/phase2/pairs/card_<Y>_<X>.png``
      Three-panel diagnostic card per pair:
        (a) the two legs' prices normalised to the pair's first active date;
        (b) the spread (piecewise per fold) with the rolling-z mean ± 2σ
            bands, fold/test windows shaded, and trade markers (long entry,
            short entry, mean-revert exit, stop, time-stop, fold-boundary);
        (c) cumulative net P&L over the pair's active dates.
  * ``reports/phase2_per_pair.csv``
      One row per pair with: n_trades, gross/net total & ann return,
      gross/net Sharpe, win-rate, avg holding days, exit-reason breakdown,
      max drawdown, top-1/top-2 trade share of positive P&L, first/last
      active date, sector.

Does NOT re-fit anything — uses the same leakage-free engine as script 10.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from loguru import logger

from apt.backtest import Pair, build_folds, compute_metrics, run_walkforward
from apt.config import settings
from apt.plots.backtest import plot_pair_card, plot_per_pair_overview
from apt.signals.cointegration import cointegrate_pairs
from apt.signals.spread import compute_spread
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, processed, reports


def _segments_for_pair(
    *,
    pair_key: str,
    daily: pl.DataFrame,
    selected_pairs_per_fold: dict[int, list[Pair]],
    folds: list,
    trading_days: list[date],
    rolling_window: int,
) -> list[dict]:
    """Per fold the pair was selected in, compute the spread + rolling bands
    over the test slice (using that fold's frozen alpha/beta and a per-fold
    warm-up so the band is defined at test_start)."""
    day_to_idx = {d: i for i, d in enumerate(trading_days)}
    y_sym, x_sym = pair_key.split("/")
    segments: list[dict] = []

    for fold in folds:
        pairs_in_fold = selected_pairs_per_fold.get(fold.fold_id, [])
        match = next((p for p in pairs_in_fold if p.key == pair_key), None)
        if match is None:
            continue
        test_start_idx = day_to_idx[fold.test_start]
        warmup_start_idx = max(0, test_start_idx - rolling_window + 1)
        warmup_start = trading_days[warmup_start_idx]

        win = (
            daily.filter(
                pl.col("symbol").is_in([y_sym, x_sym])
                & (pl.col("date") >= warmup_start)
                & (pl.col("date") <= fold.test_end)
            )
            .select(["symbol", "date", "close"])
            .sort("date")
        )
        wide = win.pivot(index="date", on="symbol", values="close").sort("date").drop_nulls()
        if wide.is_empty() or y_sym not in wide.columns or x_sym not in wide.columns:
            continue

        dates_full = wide["date"].to_list()
        p_y = wide[y_sym].to_numpy()
        p_x = wide[x_sym].to_numpy()
        spread_full = compute_spread(p_y, p_x, beta=match.beta, intercept=match.alpha)
        s_series = pl.Series("s", spread_full)
        roll_mean = s_series.rolling_mean(
            window_size=rolling_window, min_samples=rolling_window
        ).to_numpy()
        roll_std = s_series.rolling_std(
            window_size=rolling_window, min_samples=rolling_window
        ).to_numpy()
        band_upper = roll_mean + 2.0 * roll_std
        band_lower = roll_mean - 2.0 * roll_std

        # Slice to test
        try:
            test_first = next(i for i, d in enumerate(dates_full) if d >= fold.test_start)
        except StopIteration:
            continue
        segments.append(
            {
                "fold_id": fold.fold_id,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "dates": dates_full[test_first:],
                "spread": spread_full[test_first:],
                "roll_mean": roll_mean[test_first:],
                "band_upper": band_upper[test_first:],
                "band_lower": band_lower[test_first:],
            }
        )
    return segments


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "11_backtest_per_pair.log")
    ensure_dirs()
    card_dir = settings.paths.plots_dir / "phase2" / "pairs"
    card_dir.mkdir(parents=True, exist_ok=True)

    daily = pl.read_parquet(processed("daily_clean.parquet"))
    sectors = pl.read_parquet(interim("sectors.parquet"))
    trading_days = daily.select(pl.col("date").unique()).sort("date")["date"].to_list()
    sector_map = dict(zip(sectors["symbol"], sectors["industry"], strict=True))

    folds = build_folds(
        trading_days,
        prior_days=settings.cointegration.n_train_days,
        train_days=settings.cointegration.n_train_days,
        test_days=settings.backtest.test_days_per_fold,
        step_days=settings.backtest.step_days,
    )

    def select_pairs_fn(prior_start, prior_end, train_start, train_end):
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

    def get_prices_fn(y_sym, x_sym, start, end):
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
    # Per-pair stats
    # ------------------------------------------------------------------
    pair_stats: list[dict] = []
    for pkey, daily_df in result.per_pair_daily.items():
        pair_trades = [t for t in result.trades if t.pair_key == pkey]
        gross_arr = daily_df["gross_log_ret"].to_numpy()
        net_arr = daily_df["net_log_ret"].to_numpy()
        m_gross = compute_metrics(gross_arr)
        m_net = compute_metrics(net_arr)
        n_trades = len(pair_trades)
        n_winners_net = sum(1 for t in pair_trades if t.net_log_pnl > 0)
        win_rate_net = n_winners_net / n_trades if n_trades else 0.0
        avg_hold = float(np.mean([t.days_held for t in pair_trades])) if n_trades else 0.0
        n_mr = sum(1 for t in pair_trades if t.exit_reason == "mean_revert")
        n_stop = sum(1 for t in pair_trades if t.exit_reason == "stop")
        n_time = sum(1 for t in pair_trades if t.exit_reason == "time")
        n_fb = sum(1 for t in pair_trades if t.exit_reason == "fold_boundary")
        first_active = daily_df["date"].min()
        last_active = daily_df["date"].max()

        # Top-trade concentration: top1/top2 share of all POSITIVE net trade P&L
        pos_pnls = sorted([t.net_log_pnl for t in pair_trades if t.net_log_pnl > 0], reverse=True)
        total_pos = sum(pos_pnls) if pos_pnls else 0.0
        top1_share = (pos_pnls[0] / total_pos) if total_pos > 0 else 0.0
        top2_share = (
            sum(pos_pnls[:2]) / total_pos if total_pos > 0 and len(pos_pnls) >= 2 else top1_share
        )

        y_sym = pkey.split("/")[0]
        sec = sector_map.get(y_sym, "")

        pair_stats.append(
            {
                "pair": pkey,
                "sector": sec,
                "n_trades": n_trades,
                "first_active": first_active,
                "last_active": last_active,
                "n_obs": int(daily_df.height),
                "gross_total_pct": m_gross["total_return_pct"],
                "net_total_pct": m_net["total_return_pct"],
                "gross_ann_pct": m_gross["ann_return_pct"],
                "net_ann_pct": m_net["ann_return_pct"],
                "gross_sharpe": m_gross["sharpe"],
                "net_sharpe": m_net["sharpe"],
                "net_max_drawdown_pct": m_net["max_drawdown_pct"],
                "win_rate_net": win_rate_net,
                "avg_holding_days": avg_hold,
                "n_exits_mean_revert": n_mr,
                "n_exits_stop": n_stop,
                "n_exits_time": n_time,
                "n_exits_fold_boundary": n_fb,
                "top1_pos_pnl_share": top1_share,
                "top2_pos_pnl_share": top2_share,
            }
        )

    pair_stats.sort(key=lambda r: -r["net_total_pct"])

    csv_path = reports("phase2_per_pair.csv")
    pl.DataFrame(pair_stats).write_csv(csv_path)
    logger.info("Wrote {} per-pair rows → {}", len(pair_stats), csv_path)

    # ------------------------------------------------------------------
    # Overview small-multiples grid
    # ------------------------------------------------------------------
    overview_curves: list[dict] = []
    for s in pair_stats:
        d = result.per_pair_daily[s["pair"]].sort("date")
        cum_pct = np.expm1(np.cumsum(d["net_log_ret"].to_numpy())) * 100
        overview_curves.append(
            {
                "pair": s["pair"],
                "dates": d["date"].to_list(),
                "cum_net_pct": cum_pct,
                "n_trades": s["n_trades"],
                "net_total_pct": s["net_total_pct"],
                "net_sharpe": s["net_sharpe"],
            }
        )
    overview_path = card_dir / "per_pair_overview.png"
    plot_per_pair_overview(
        overview_curves,
        overview_path,
        title=(f"Per-pair cumulative net P&L  ({len(pair_stats)} pairs, sorted best → worst)"),
        cols=5,
    )
    logger.info("Wrote overview grid → {}", overview_path)

    # ------------------------------------------------------------------
    # Per-pair detail cards
    # ------------------------------------------------------------------
    card_paths: list = []
    for s in pair_stats:
        pkey = s["pair"]
        y_sym, x_sym = pkey.split("/")
        first_active: date = s["first_active"]
        last_active: date = s["last_active"]

        # Full leg prices over [first_active, last_active] for panel (a)
        win = (
            daily.filter(
                pl.col("symbol").is_in([y_sym, x_sym])
                & (pl.col("date") >= first_active)
                & (pl.col("date") <= last_active)
            )
            .select(["symbol", "date", "close"])
            .sort("date")
        )
        wide = win.pivot(index="date", on="symbol", values="close").sort("date").drop_nulls()
        if wide.is_empty() or y_sym not in wide.columns or x_sym not in wide.columns:
            logger.warning("Pair {}: could not assemble price panel", pkey)
            continue
        full_dates = wide["date"].to_list()
        py = wide[y_sym].to_numpy()
        px = wide[x_sym].to_numpy()
        py_norm = py / py[0]
        px_norm = px / px[0]

        # Per-fold segments for panel (b)
        segments = _segments_for_pair(
            pair_key=pkey,
            daily=daily,
            selected_pairs_per_fold=result.selected_pairs_per_fold,
            folds=folds,
            trading_days=trading_days,
            rolling_window=settings.spread.rolling_window,
        )
        test_window_spans = [(seg["test_start"], seg["test_end"]) for seg in segments]

        # Cumulative net P&L for panel (c)
        d = result.per_pair_daily[pkey].sort("date")
        cum_net_pct = np.expm1(np.cumsum(d["net_log_ret"].to_numpy())) * 100

        pair_trades = [t for t in result.trades if t.pair_key == pkey]

        safe = pkey.replace("/", "_").replace("&", "AND")
        out_path = card_dir / f"card_{safe}.png"
        plot_pair_card(
            pair_key=pkey,
            sector=s["sector"] or None,
            y_sym=y_sym,
            x_sym=x_sym,
            full_dates=full_dates,
            py_norm=py_norm,
            px_norm=px_norm,
            fold_segments=segments,
            trades=pair_trades,
            cum_dates=d["date"].to_list(),
            cum_net_pct=cum_net_pct,
            stats=s,
            out_path=out_path,
            test_window_spans=test_window_spans,
        )
        card_paths.append(out_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== 11_backtest_per_pair complete ===")
    print(f"  Pairs analysed   : {len(pair_stats)}")
    print(f"  Folds            : {len(folds)}")
    print(f"  CSV              : {csv_path}")
    print(f"  Overview         : {overview_path}")
    print(f"  Cards            : {len(card_paths)} → {card_dir}/card_*.png")
    print()
    print(
        f"  {'pair':<26}  {'n_t':>3}  {'net%':>7}  {'Sh':>6}  "
        f"{'win%':>5}  {'top1%':>6}  {'stops/n':>8}  "
        f"{'first→last active':>28}"
    )
    for s in pair_stats:
        stops_ratio = (s["n_exits_stop"] / s["n_trades"]) if s["n_trades"] else 0.0
        print(
            f"  {s['pair']:<26}  {s['n_trades']:>3}  "
            f"{s['net_total_pct']:>+7.1f}  {s['net_sharpe']:>+6.2f}  "
            f"{s['win_rate_net'] * 100:>4.0f}%  "
            f"{s['top1_pos_pnl_share'] * 100:>5.0f}%  "
            f"{stops_ratio * 100:>6.0f}%  "
            f"  {str(s['first_active']):>10} → {str(s['last_active']):>10}"
        )


if __name__ == "__main__":
    main()
