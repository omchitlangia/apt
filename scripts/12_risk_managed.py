#!/usr/bin/env python3
"""Script 12: Phase 2B — risk-managed walk-forward ablation ladder.

Runs the SAME folds, selection and signals as ``scripts/10_backtest.py``,
varying ONLY sizing/risk. Five rungs (R0..R4) form the ablation ladder;
two sweeps probe the survivability knobs on R4:

  * cluster-cap sweep  (R4 + cluster_cap ∈ {none, 5%, 3%})
  * kill-aggressiveness sweep  (R4 + kill_mode ∈ {none, loss_only, relationship})

Per-rung and per-sweep-arm we report: gross & net total/ann return, Sharpe,
ann vol, max drawdown, turnover, n trades, the 2018-19 fold-specific net
P&L (the "rescue" measure), the carriers' (PFC/SBIN, ONGC/OIL) per-rung net,
and premature-cut counts/foregone P&L.

Outputs:
  * reports/risk_managed_ladder.csv     — one row per rung, full metrics
  * reports/risk_managed_sweeps.csv     — one row per sweep arm
  * reports/risk_managed_premature_cuts.csv — log of every cut + reverted flag
  * reports/risk_managed_kill_events.csv    — every kill event with reason
  * plots/phase2/risk_managed/*.png     — equity, DD, cluster exposure, carriers
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
from loguru import logger

from apt.backtest import (
    Pair,
    RiskConfig,
    build_folds,
    compute_metrics,
    run_walkforward,
    run_walkforward_risk_managed,
)
from apt.config import settings
from apt.plots.backtest import (
    plot_carriers_r0_vs_r4,
    plot_cluster_cap_sweep,
    plot_cluster_exposure,
    plot_drawdown_per_rung,
    plot_ladder_equity,
)
from apt.signals.cointegration import cointegrate_pairs
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, interim, processed, reports

PHASE2B_PLOT_DIR = settings.paths.plots_dir / "phase2" / "risk_managed"
BREAKDOWN_FOLD_PERIOD = (date(2018, 10, 11), date(2019, 10, 24))  # fold 6 in our build
CARRIER_KEYS = ("PFC/SBIN", "ONGC/OIL")


def _fold_period_metric(portfolio_daily: pl.DataFrame, start: date, end: date) -> dict:
    """Metrics on the slice ``[start, end]`` of the portfolio_daily frame."""
    if portfolio_daily.is_empty():
        return compute_metrics([])
    slice_df = portfolio_daily.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    if slice_df.is_empty():
        return compute_metrics([])
    return compute_metrics(slice_df["net_log_ret"].to_numpy())


def _carrier_metrics(per_pair_daily: dict[str, pl.DataFrame], pair_key: str) -> dict:
    df = per_pair_daily.get(pair_key)
    if df is None or df.is_empty():
        return compute_metrics([])
    return compute_metrics(df["net_log_ret"].to_numpy())


def _select_pairs_cb(daily: pl.DataFrame, sectors: pl.DataFrame):
    def fn(prior_start, prior_end, train_start, train_end) -> list[Pair]:
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

    return fn


def _get_prices_cb(daily: pl.DataFrame):
    def fn(y_sym, x_sym, start, end):
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

    return fn


def _rung_config(rung: int, **kwargs) -> RiskConfig:
    """Build a RiskConfig for rung `rung` with sensible defaults overridable
    via kwargs."""
    rm = settings.risk_managed
    defaults = {
        "rung": rung,
        "risk_frac": rm.risk_frac,
        "per_pair_cap": rm.per_pair_cap,
        "cluster_cap": rm.cluster_cap,
        "total_cap": rm.total_cap,
        "gross_cap": rm.gross_cap,
        "kill_mode": "relationship" if rung >= 4 else "none",
        "kill_K": rm.kill_K,
        "kill_cap": rm.kill_cap,
        "kill_check_interval_days": rm.kill_check_interval_days,
        "kill_relationship_window_days": rm.kill_relationship_window_days,
        "kill_relationship_adf_alpha": rm.kill_relationship_adf_alpha,
        "kill_relationship_halflife_max_days": rm.kill_relationship_halflife_max_days,
        "kill_relationship_vol_ratio_max": rm.kill_relationship_vol_ratio_max,
    }
    defaults.update(kwargs)
    return RiskConfig(**defaults)


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "12_risk_managed.log")
    ensure_dirs()
    PHASE2B_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    daily = pl.read_parquet(processed("daily_clean.parquet"))
    sectors = pl.read_parquet(interim("sectors.parquet"))
    trading_days = daily.select(pl.col("date").unique()).sort("date")["date"].to_list()

    folds = build_folds(
        trading_days,
        prior_days=settings.cointegration.n_train_days,
        train_days=settings.cointegration.n_train_days,
        test_days=settings.backtest.test_days_per_fold,
        step_days=settings.backtest.step_days,
    )
    logger.info("Built {} folds", len(folds))

    select_fn = _select_pairs_cb(daily, sectors)
    prices_fn = _get_prices_cb(daily)

    # ------------------------------------------------------------------
    # Phase 2A baseline: run once to capture the selection per fold + cache
    # ------------------------------------------------------------------
    logger.info("Running Phase 2A baseline (R0 anchor) to capture pair selection ...")
    baseline_2a = run_walkforward(
        folds,
        trading_days,
        select_pairs_fn=select_fn,
        get_prices_fn=prices_fn,
        rolling_window=settings.spread.rolling_window,
        entry_z=settings.signal.entry_z,
        exit_z=settings.signal.exit_z,
        stop_z=settings.signal.stop_z,
        max_holding_cap_days=settings.signal.max_holding_cap_days,
        max_holding_half_life_multiplier=settings.signal.max_holding_half_life_multiplier,
        cost_bps_per_leg=settings.backtest.cost_bps_per_leg,
    )
    selected_pairs_per_fold = baseline_2a.selected_pairs_per_fold

    # ------------------------------------------------------------------
    # Ladder R0..R4
    # ------------------------------------------------------------------
    rungs: dict[str, dict] = {}
    for r in range(5):
        cfg = _rung_config(r)
        logger.info("Running rung R{} (kill_mode={}) ...", r, cfg.kill_mode)
        res = run_walkforward_risk_managed(
            folds,
            trading_days,
            get_prices_fn=prices_fn,
            pre_selected_pairs=selected_pairs_per_fold,
            rolling_window=settings.spread.rolling_window,
            entry_z=settings.signal.entry_z,
            exit_z=settings.signal.exit_z,
            stop_z=settings.signal.stop_z,
            max_holding_cap_days=settings.signal.max_holding_cap_days,
            max_holding_half_life_multiplier=settings.signal.max_holding_half_life_multiplier,
            cost_bps_per_leg=settings.backtest.cost_bps_per_leg,
            config=cfg,
        )
        rungs[f"R{r}"] = {"result": res, "config": cfg}

    # ------------------------------------------------------------------
    # R0 must reproduce 2A (sanity gate)
    # ------------------------------------------------------------------
    r0_port = rungs["R0"]["result"].portfolio_daily.sort("date")
    p2a_port = baseline_2a.portfolio_daily.sort("date")
    repro_ok = False
    repro_max_abs_diff = float("nan")
    if not r0_port.is_empty() and not p2a_port.is_empty():
        merged = r0_port.join(
            p2a_port.rename({"net_log_ret": "p2a_net", "gross_log_ret": "p2a_gross"}),
            on="date",
            how="inner",
        )
        max_abs_diff = float(
            np.abs(merged["net_log_ret"].to_numpy() - merged["p2a_net"].to_numpy()).max()
        )
        repro_max_abs_diff = max_abs_diff
        repro_ok = max_abs_diff < 1e-9
        logger.info(
            "R0 ↔ 2A reproducibility: max |net_diff| = {:.3e}  ({})",
            max_abs_diff,
            "OK" if repro_ok else "MISMATCH",
        )

    # ------------------------------------------------------------------
    # Cluster-cap sweep on R4
    # ------------------------------------------------------------------
    cluster_arms = {
        "R4 cluster=none": _rung_config(4, kill_mode="relationship", cluster_cap=1e9),
        "R4 cluster=5%": _rung_config(4, kill_mode="relationship", cluster_cap=0.05),
        "R4 cluster=3%": _rung_config(4, kill_mode="relationship", cluster_cap=0.03),
    }
    cluster_sweep: dict[str, dict] = {}
    for label, cfg in cluster_arms.items():
        if label == "R4 cluster=5%":
            res = rungs["R4"]["result"]  # already computed
        else:
            logger.info("Cluster sweep arm: {}", label)
            res = run_walkforward_risk_managed(
                folds,
                trading_days,
                get_prices_fn=prices_fn,
                pre_selected_pairs=selected_pairs_per_fold,
                rolling_window=settings.spread.rolling_window,
                entry_z=settings.signal.entry_z,
                exit_z=settings.signal.exit_z,
                stop_z=settings.signal.stop_z,
                max_holding_cap_days=settings.signal.max_holding_cap_days,
                max_holding_half_life_multiplier=settings.signal.max_holding_half_life_multiplier,
                cost_bps_per_leg=settings.backtest.cost_bps_per_leg,
                config=cfg,
            )
        cluster_sweep[label] = {"result": res, "config": cfg}

    # ------------------------------------------------------------------
    # Kill-aggressiveness sweep on R4
    # ------------------------------------------------------------------
    kill_arms = {
        "R4 kill=none": _rung_config(4, kill_mode="none"),
        "R4 kill=loss_only": _rung_config(4, kill_mode="loss_only"),
        "R4 kill=relationship": _rung_config(4, kill_mode="relationship"),
    }
    kill_sweep: dict[str, dict] = {}
    for label, cfg in kill_arms.items():
        if label == "R4 kill=relationship":
            res = rungs["R4"]["result"]
        else:
            logger.info("Kill sweep arm: {}", label)
            res = run_walkforward_risk_managed(
                folds,
                trading_days,
                get_prices_fn=prices_fn,
                pre_selected_pairs=selected_pairs_per_fold,
                rolling_window=settings.spread.rolling_window,
                entry_z=settings.signal.entry_z,
                exit_z=settings.signal.exit_z,
                stop_z=settings.signal.stop_z,
                max_holding_cap_days=settings.signal.max_holding_cap_days,
                max_holding_half_life_multiplier=settings.signal.max_holding_half_life_multiplier,
                cost_bps_per_leg=settings.backtest.cost_bps_per_leg,
                config=cfg,
            )
        kill_sweep[label] = {"result": res, "config": cfg}

    # ------------------------------------------------------------------
    # Build per-rung / per-arm metric table
    # ------------------------------------------------------------------
    def _row(label: str, res, cfg: RiskConfig) -> dict:
        port = res.portfolio_daily
        m_g = (
            compute_metrics(port["gross_log_ret"].to_numpy())
            if not port.is_empty()
            else compute_metrics([])
        )
        m_n = (
            compute_metrics(port["net_log_ret"].to_numpy())
            if not port.is_empty()
            else compute_metrics([])
        )
        breakdown = _fold_period_metric(port, *BREAKDOWN_FOLD_PERIOD)
        carriers_net = {
            pkey: _carrier_metrics(res.per_pair_daily, pkey)["total_return_pct"]
            for pkey in CARRIER_KEYS
        }
        n_trades = len(res.trades)
        n_premature = sum(1 for p in res.premature_cuts if p.reverted_within_horizon)
        prem_foregone = sum(
            max(0.0, p.foregone_gross_log_pnl)
            for p in res.premature_cuts
            if p.reverted_within_horizon
        )
        turnover = (
            (n_trades * 2.0) / max(1.0, m_n["n_years"])  # 2 transactions/trade per year
        )
        return {
            "label": label,
            "rung": cfg.rung,
            "kill_mode": cfg.kill_mode,
            "cluster_cap": cfg.cluster_cap,
            "n_trades": n_trades,
            "n_kill_events": len(res.kill_events),
            "n_premature_cuts": n_premature,
            "premature_foregone_pct": float(math.expm1(prem_foregone) * 100),
            "gross_total_pct": m_g["total_return_pct"],
            "net_total_pct": m_n["total_return_pct"],
            "gross_ann_pct": m_g["ann_return_pct"],
            "net_ann_pct": m_n["ann_return_pct"],
            "ann_vol_pct": m_n["ann_vol_pct"],
            "gross_sharpe": m_g["sharpe"],
            "net_sharpe": m_n["sharpe"],
            "max_drawdown_pct": m_n["max_drawdown_pct"],
            "turnover_per_year": turnover,
            "fold_2018_19_net_pct": breakdown["total_return_pct"],
            "carrier_PFC_SBIN_net_pct": carriers_net.get("PFC/SBIN", 0.0),
            "carrier_ONGC_OIL_net_pct": carriers_net.get("ONGC/OIL", 0.0),
        }

    ladder_rows = [
        _row(label, payload["result"], payload["config"]) for label, payload in rungs.items()
    ]
    sweep_rows = [_row(label, p["result"], p["config"]) for label, p in cluster_sweep.items()] + [
        _row(label, p["result"], p["config"]) for label, p in kill_sweep.items()
    ]

    pl.DataFrame(ladder_rows).write_csv(reports("risk_managed_ladder.csv"))
    pl.DataFrame(sweep_rows).write_csv(reports("risk_managed_sweeps.csv"))

    # Premature cuts + kill events flat tables
    prem_rows = []
    for label, payload in rungs.items():
        for pc in payload["result"].premature_cuts:
            prem_rows.append(
                {
                    "rung": label,
                    "fold_id": pc.fold_id,
                    "pair": pc.pair_key,
                    "cut_date": pc.cut_date,
                    "cut_reason": pc.cut_reason,
                    "direction": pc.direction,
                    "horizon_days": pc.horizon_days,
                    "reverted_within_horizon": pc.reverted_within_horizon,
                    "foregone_gross_log_pnl": pc.foregone_gross_log_pnl,
                    "foregone_gross_pct": float(math.expm1(pc.foregone_gross_log_pnl) * 100),
                }
            )
    pl.DataFrame(prem_rows).write_csv(
        reports("risk_managed_premature_cuts.csv")
    ) if prem_rows else None

    kill_rows = []
    for label, payload in rungs.items():
        for ke in payload["result"].kill_events:
            kill_rows.append(
                {
                    "rung": label,
                    "fold_id": ke.fold_id,
                    "pair": ke.pair_key,
                    "date": ke.date,
                    "reason": ke.reason,
                    "detail": str(ke.detail),
                }
            )
    pl.DataFrame(kill_rows).write_csv(
        reports("risk_managed_kill_events.csv")
    ) if kill_rows else None

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_ladder_equity(
        {k: v["result"].portfolio_daily for k, v in rungs.items()},
        PHASE2B_PLOT_DIR / "ladder_equity.png",
        annotate_period=BREAKDOWN_FOLD_PERIOD,
    )
    plot_drawdown_per_rung(
        {k: v["result"].portfolio_daily for k, v in rungs.items()},
        PHASE2B_PLOT_DIR / "drawdown_per_rung.png",
        annotate_period=BREAKDOWN_FOLD_PERIOD,
    )
    plot_cluster_exposure(
        rungs["R4"]["result"].cluster_exposure_daily,
        cluster_cap=rungs["R4"]["config"].cluster_cap,
        out_path=PHASE2B_PLOT_DIR / "cluster_exposure_R4.png",
        highlight_period=BREAKDOWN_FOLD_PERIOD,
    )
    plot_carriers_r0_vs_r4(
        {
            pkey: {
                "R0": rungs["R0"]["result"].per_pair_daily.get(pkey, pl.DataFrame()),
                "R4": rungs["R4"]["result"].per_pair_daily.get(pkey, pl.DataFrame()),
            }
            for pkey in CARRIER_KEYS
        },
        PHASE2B_PLOT_DIR / "carriers_R0_vs_R4.png",
    )
    plot_cluster_cap_sweep(
        {
            label: {
                "portfolio_daily": p["result"].portfolio_daily,
                "config": p["config"],
            }
            for label, p in cluster_sweep.items()
        },
        PHASE2B_PLOT_DIR / "cluster_cap_sweep.png",
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n=== 12_risk_managed complete ===")
    print(
        f"  R0 ↔ Phase-2A reproducibility: max |net_diff| = {repro_max_abs_diff:.3e} "
        f"({'OK' if repro_ok else 'MISMATCH'})"
    )
    print(f"  Folds (test windows)         : {len(folds)}")
    print(
        f"  Breakdown fold annotated     : {BREAKDOWN_FOLD_PERIOD[0]} → {BREAKDOWN_FOLD_PERIOD[1]}"
    )
    print()
    hdr = (
        f"  {'rung':<22}  {'trades':>6}  {'gross%':>7}  {'net%':>7}  "
        f"{'Sh g':>5}  {'Sh n':>5}  {'maxDD':>6}  "
        f"{'fold18_19':>9}  {'PFC/SBIN':>9}  {'ONGC/OIL':>9}  "
        f"{'kills':>5}  {'premature':>9}"
    )
    print("  --- Ladder (R0..R4) ---")
    print(hdr)
    for r in ladder_rows:
        print(
            f"  {r['label']:<22}  {r['n_trades']:>6}  "
            f"{r['gross_total_pct']:>+7.1f}  {r['net_total_pct']:>+7.1f}  "
            f"{r['gross_sharpe']:>+5.2f}  {r['net_sharpe']:>+5.2f}  "
            f"{r['max_drawdown_pct']:>+6.1f}  "
            f"{r['fold_2018_19_net_pct']:>+9.1f}  "
            f"{r['carrier_PFC_SBIN_net_pct']:>+9.1f}  "
            f"{r['carrier_ONGC_OIL_net_pct']:>+9.1f}  "
            f"{r['n_kill_events']:>5}  {r['n_premature_cuts']:>9}"
        )
    print()
    print("  --- Cluster-cap sweep (R4) ---")
    print(hdr)
    for r in [_row(lab, p["result"], p["config"]) for lab, p in cluster_sweep.items()]:
        print(
            f"  {r['label']:<22}  {r['n_trades']:>6}  "
            f"{r['gross_total_pct']:>+7.1f}  {r['net_total_pct']:>+7.1f}  "
            f"{r['gross_sharpe']:>+5.2f}  {r['net_sharpe']:>+5.2f}  "
            f"{r['max_drawdown_pct']:>+6.1f}  "
            f"{r['fold_2018_19_net_pct']:>+9.1f}  "
            f"{r['carrier_PFC_SBIN_net_pct']:>+9.1f}  "
            f"{r['carrier_ONGC_OIL_net_pct']:>+9.1f}  "
            f"{r['n_kill_events']:>5}  {r['n_premature_cuts']:>9}"
        )
    print()
    print("  --- Kill-aggressiveness sweep (R4) ---")
    print(hdr)
    for r in [_row(lab, p["result"], p["config"]) for lab, p in kill_sweep.items()]:
        print(
            f"  {r['label']:<22}  {r['n_trades']:>6}  "
            f"{r['gross_total_pct']:>+7.1f}  {r['net_total_pct']:>+7.1f}  "
            f"{r['gross_sharpe']:>+5.2f}  {r['net_sharpe']:>+5.2f}  "
            f"{r['max_drawdown_pct']:>+6.1f}  "
            f"{r['fold_2018_19_net_pct']:>+9.1f}  "
            f"{r['carrier_PFC_SBIN_net_pct']:>+9.1f}  "
            f"{r['carrier_ONGC_OIL_net_pct']:>+9.1f}  "
            f"{r['n_kill_events']:>5}  {r['n_premature_cuts']:>9}"
        )
    print()
    print("  Outputs:")
    print("    reports/risk_managed_ladder.csv")
    print("    reports/risk_managed_sweeps.csv")
    print("    reports/risk_managed_premature_cuts.csv")
    print("    reports/risk_managed_kill_events.csv")
    print("    plots/phase2/risk_managed/*.png")


if __name__ == "__main__":
    main()
