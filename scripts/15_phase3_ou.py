#!/usr/bin/env python3
"""Script 15: Phase 3 OU/Bertram optimal-threshold backtest.

Reuses Phase 3 v2 (script 13) for fold construction, daily pair
selection, and intraday liquidity gating. Replaces the rolling-z entry/
exit/stop bands with Ornstein-Uhlenbeck (Bertram 2010) optimal
thresholds in Z-OU coordinates.

Grid (default):
  bar_freq    in {1, 5, 15} minutes
  cost_bps    in {1, 3, 5, 8}  (re-fit Bertram per cost level)
  regime      in {A, B}
  stop_mode   in {"none"} (full sweep) + {"hard", K=4} at bars x cost=3 x A/B
  Additional rolling_z coarse-bar baselines at bars {5, 15} x regimes {A, B}.

CLI:
  --mode smoke    : single cell (bar=5, cost=3, A, stop=none, ~17 pair-folds)
  --mode full     : 24 OU cells + 6 stop-ablation cells + 4 rolling_z baselines
  --mode reduced  : reduced grid bars{5,15} x cost{3,8} x A/B x stop=none = 8

Outputs (reports/phase3_ou/):
  metrics_ou.csv               — per-cell aggregated metrics, gross+net
  ou_pair_fold_diag.csv        — per-pair-fold OU diagnostics
  trades_ou.csv                — all trades from all cells (large)
  exclusions_ou.csv            — pair-fold exclusion accounting per cell
  caveats_ou.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from apt.backtest import Pair, compute_metrics
from apt.config import settings
from apt.intraday import (
    NSE_BARS_PER_SESSION,
    load_minute_pair,
    resample_within_session,
)
from apt.intraday.backtest import run_pair_fold
from apt.intraday.costs import TRADE_CSV_SCHEMA_VERSION, CostBreakdown
from apt.intraday.signals import generate_signals_ou
from apt.stats.ou import bertram_threshold, fit_ou_params
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, processed

# Reuse helpers from the v2 script — these are pure functions defined at
# module scope on scripts/13_phase3_intraday.py. Loaded via importlib to
# avoid renaming the file (it starts with a digit).
_SCRIPT_DIR = Path(__file__).parent
_V2_PATH = _SCRIPT_DIR / "13_phase3_intraday.py"
_spec = importlib.util.spec_from_file_location("script_13", _V2_PATH)
assert _spec is not None and _spec.loader is not None
_v2 = importlib.util.module_from_spec(_spec)
sys.modules["script_13"] = _v2
_spec.loader.exec_module(_v2)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MINUTE_ROOT = _v2.MINUTE_ROOT
MINUTE_PANEL_START = _v2.MINUTE_PANEL_START
MINUTE_PANEL_END = _v2.MINUTE_PANEL_END

REPORTS_DIR = Path("reports/phase3_ou")

FULL_OU_FREQS = (1, 5, 15)
FULL_COSTS_BPS = (1, 3, 5, 8)
FULL_REGIMES = ("A", "B")

REDUCED_FREQS = (5, 15)
REDUCED_COSTS_BPS = (3, 8)

STOP_ABLATION_COST_BPS = 3
STOP_K_DEFAULT = 4.0


# ----------------------------------------------------------------------
# Grid cell descriptor
# ----------------------------------------------------------------------
class Cell(NamedTuple):
    engine: str  # 'ou' or 'rolling_z'
    freq_min: int
    cost_bps: int
    regime: str  # 'A' or 'B'
    stop_mode: str  # 'none' or 'hard'
    stop_k_sigma: float


def _smoke_cells() -> list[Cell]:
    return [Cell("ou", 5, 3, "A", "none", 0.0)]


def _full_cells() -> list[Cell]:
    cells: list[Cell] = []
    for f in FULL_OU_FREQS:
        for c in FULL_COSTS_BPS:
            for r in FULL_REGIMES:
                cells.append(Cell("ou", f, c, r, "none", 0.0))
        for r in FULL_REGIMES:
            cells.append(Cell("ou", f, STOP_ABLATION_COST_BPS, r, "hard", STOP_K_DEFAULT))
    # Coarse-bar rolling_z baselines (addendum #5)
    for f in (5, 15):
        for c in FULL_COSTS_BPS:
            for r in FULL_REGIMES:
                cells.append(Cell("rolling_z", f, c, r, "none", 0.0))
    return cells


def _reduced_cells() -> list[Cell]:
    cells: list[Cell] = []
    for f in REDUCED_FREQS:
        for c in REDUCED_COSTS_BPS:
            for r in FULL_REGIMES:
                cells.append(Cell("ou", f, c, r, "none", 0.0))
    return cells


# ----------------------------------------------------------------------
# Per-pair-fold cache: (fold_id, pair_key, freq_min) -> dict
# ----------------------------------------------------------------------
def _ou_fit_for_pair_fold(
    pair: Pair,
    fold,
    *,
    freq_min: int,
    min_obs: int,
) -> dict | None:
    """Load minute panel, resample, fit OU on train slice, compute Z-OU on test.

    Returns dict with: ou_fit, train_mask, test_mask, ts_test, sids_test,
    spread_test, z_ou_test, tradeable_test, drift_mean (test-slice Z-OU mean).
    None on insufficient data.
    """
    test_start = max(fold.test_start, MINUTE_PANEL_START)
    test_end = min(fold.test_end, MINUTE_PANEL_END)
    aligned_full = load_minute_pair(
        pair.y_sym, pair.x_sym, fold.train_start, test_end, root=MINUTE_ROOT
    )
    if aligned_full.n_bars == 0:
        return None
    rs = resample_within_session(aligned_full, freq_minutes=freq_min)
    if rs.n_bars == 0:
        return None

    # Compute log spread on the resampled grid (NaN where any leg non-tradeable)
    with np.errstate(invalid="ignore", divide="ignore"):
        py = np.where(rs.tradeable, rs.close_y, np.nan)
        px = np.where(rs.tradeable, rs.close_x, np.nan)
        spread_full = np.log(py) - pair.beta * np.log(px) - pair.alpha

    ts_full = pd.DatetimeIndex(rs.timestamps)
    train_mask = (ts_full.date >= fold.train_start) & (ts_full.date <= fold.train_end)
    test_mask = (ts_full.date >= test_start) & (ts_full.date <= test_end)

    # Fit OU on train slice
    fit = fit_ou_params(spread_full[train_mask], freq_minutes=freq_min, min_obs=min_obs)
    if not fit.fit_ok:
        return {"ou_fit": fit, "reject_reason": fit.reason}

    # Compute Z-OU on test slice (NaN-safe)
    spread_test = spread_full[test_mask]
    z_ou_test = (spread_test - fit.mu) / fit.sigma_eq

    # Test-slice mean (drift diagnostic)
    finite = np.isfinite(z_ou_test)
    drift_mean = float(np.mean(z_ou_test[finite])) if finite.any() else float("nan")

    # Test slice metadata
    ts_test = ts_full[test_mask]
    sids_test = rs.session_id[test_mask]
    # Re-rank session ids to dense 0..S-1 within test slice
    _, sids_test = np.unique(sids_test, return_inverse=True)
    sids_test = sids_test.astype(np.int32)
    tradeable_test = rs.tradeable[test_mask]

    return {
        "ou_fit": fit,
        "spread_test": spread_test,
        "z_ou_test": z_ou_test,
        "ts_test": ts_test,
        "sids_test": sids_test,
        "tradeable_test": tradeable_test,
        "drift_mean": drift_mean,
        "reject_reason": "",
    }


def _hl_band_for_regime(regime: str) -> tuple[float, float]:
    band = settings.signal.ou.half_life_band
    if regime == "A":
        return float(band.A_min_minutes), float(band.A_max_minutes)
    return float(band.B_min_minutes), float(band.B_max_minutes)


def _max_holding_bars(pair: Pair, freq_min: int) -> int:
    """v2's max_holding = max(window_bars, min(cap_days*375, ceil(HL*375*mult))).

    Then expressed in bars at the active frequency.
    """
    # window in minutes
    window_min = int(
        np.clip(
            round(pair.half_life * NSE_BARS_PER_SESSION),
            NSE_BARS_PER_SESSION,
            5 * NSE_BARS_PER_SESSION,
        )
    )
    max_min = max(
        window_min,
        min(
            settings.signal.max_holding_cap_days * NSE_BARS_PER_SESSION,
            round(
                pair.half_life
                * NSE_BARS_PER_SESSION
                * settings.signal.max_holding_half_life_multiplier
            ),
        ),
    )
    # Convert to bars at active freq
    bars = max(int(round(max_min / freq_min)), 1)
    return bars


# ----------------------------------------------------------------------
# Per-cell runner
# ----------------------------------------------------------------------
def _run_ou_cell(
    *,
    cell: Cell,
    pair_fold_cache: dict[tuple[int, str, int], dict],
    pair_fold_keep: dict[tuple[int, str], Pair],
    folds_by_id: dict,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Run one OU cell over all surviving pair-folds.

    Returns (trade_rows, pair_session_rows, exclusion_counts).
    """
    trade_rows: list[dict] = []
    pair_session_rows: list[dict] = []
    exclusions = defaultdict(int)

    hl_min, hl_max = _hl_band_for_regime(cell.regime)
    cb = CostBreakdown(total_spread_bps=cell.cost_bps)

    for (fold_id, pair_key), pair in pair_fold_keep.items():
        cache_key = (fold_id, pair_key, cell.freq_min)
        if cache_key not in pair_fold_cache:
            exclusions["no_cache"] += 1
            continue
        c = pair_fold_cache[cache_key]
        fit = c["ou_fit"]
        if not fit.fit_ok:
            exclusions["no_reversion"] += 1
            continue
        if not (hl_min <= fit.half_life_minutes <= hl_max):
            exclusions["hl_band"] += 1
            continue

        # β-aware cost: (1 + pair.beta) × per-leg log cost. The Bertram
        # solver MUST receive this same β-aware c so the entry threshold
        # is optimal under the actual billing.
        cost_log = cb.billed_cost_log_per_pair_round_trip(beta=float(pair.beta))

        # Solve Bertram threshold at this β-aware cost
        th = bertram_threshold(fit, cost_log_per_round_trip=cost_log)
        if not th.fit_ok:
            exclusions["infeasible_at_cost"] += 1
            continue

        max_h = _max_holding_bars(pair, cell.freq_min)
        sig = generate_signals_ou(
            c["z_ou_test"],
            c["sids_test"],
            c["tradeable_test"],
            regime=cell.regime,
            a_entry_z=th.a_entry_z,
            stop_mode=cell.stop_mode,
            stop_k_sigma=cell.stop_k_sigma,
            max_holding=max_h,
        )
        res = run_pair_fold(
            fold_id=fold_id,
            pair_key=pair_key,
            timestamps=c["ts_test"],
            session_id=c["sids_test"],
            spread=c["spread_test"],
            z=c["z_ou_test"],
            signals=sig,
            cost_log_per_round_trip=cost_log,
            finalize_fold_boundary=(cell.regime == "B"),
            pair_beta=float(pair.beta),
        )

        # Per-session aggregation
        df = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(res.timestamps).date,
                "gross_log_ret": res.gross_log_ret,
                "net_log_ret": res.net_log_ret,
            }
        )
        per_sess = df.groupby("date", as_index=False)[["gross_log_ret", "net_log_ret"]].sum()
        per_sess["active"] = (
            df.assign(active=(df["gross_log_ret"] != 0) | (df["net_log_ret"] != 0))
            .groupby("date")["active"]
            .any()
            .reindex(per_sess["date"])
            .values
        )

        for _, row in per_sess.iterrows():
            pair_session_rows.append(
                {
                    "engine": cell.engine,
                    "freq_min": cell.freq_min,
                    "regime": cell.regime,
                    "spread_bps": cell.cost_bps,
                    "stop_mode": cell.stop_mode,
                    "stop_k_sigma": cell.stop_k_sigma,
                    "fold_id": fold_id,
                    "pair": pair_key,
                    "date": row["date"],
                    "gross_log_ret": float(row["gross_log_ret"]),
                    "net_log_ret": float(row["net_log_ret"]),
                    "active": bool(row["active"]),
                }
            )

        for tr in res.trades:
            trade_rows.append(
                {
                    "engine": cell.engine,
                    "freq_min": cell.freq_min,
                    "regime": cell.regime,
                    "spread_bps": cell.cost_bps,
                    "stop_mode": cell.stop_mode,
                    "stop_k_sigma": cell.stop_k_sigma,
                    "fold_id": tr.fold_id,
                    "pair": tr.pair_key,
                    "side": "long_spread" if tr.direction == +1 else "short_spread",
                    "entry_ts": tr.entry_ts.isoformat(),
                    "exit_ts": tr.exit_ts.isoformat(),
                    "z_entry": tr.entry_z,
                    "z_exit": tr.exit_z,
                    "bars_held": tr.bars_held,
                    "sessions_held": tr.sessions_held,
                    "gross_log_pnl": tr.gross_log_pnl,
                    "cost_log": tr.cost_log,
                    "net_log_pnl": tr.net_log_pnl,
                    "exit_reason": tr.exit_reason,
                    "a_entry_z": th.a_entry_z,
                    "pair_beta": tr.pair_beta,
                    "cost_log_per_pair_rt": tr.cost_log,
                    "n_legs": 2,  # DEPRECATED — see schema_version; carried one cycle
                    "schema_version": TRADE_CSV_SCHEMA_VERSION,
                }
            )

    return trade_rows, pair_session_rows, dict(exclusions)


def _aggregate_cell_metrics(
    pair_session_rows: list[dict],
    trade_rows: list[dict],
    cell: Cell,
    exclusions: dict[str, int],
) -> dict | None:
    """Produce one metrics row per (engine, freq_min, regime, cost, stop_mode)."""
    if not pair_session_rows:
        return {
            "engine": cell.engine,
            "freq_min": cell.freq_min,
            "regime": cell.regime,
            "spread_bps": cell.cost_bps,
            "stop_mode": cell.stop_mode,
            "stop_k_sigma": cell.stop_k_sigma,
            "n_pairs": 0,
            "n_trades": 0,
            "n_excluded_no_reversion": exclusions.get("no_reversion", 0),
            "n_excluded_hl_band": exclusions.get("hl_band", 0),
            "n_excluded_infeasible": exclusions.get("infeasible_at_cost", 0),
            "n_excluded_other": exclusions.get("no_cache", 0),
        }
    df = pd.DataFrame(pair_session_rows)
    port = (
        df.groupby("date", as_index=False)
        .agg(
            gross_log_ret=("gross_log_ret", "mean"),
            net_log_ret=("net_log_ret", "mean"),
            n_pairs=("pair", "nunique"),
            n_active=("active", "sum"),
        )
        .sort_values("date")
    )
    m_gross = compute_metrics(port["gross_log_ret"].to_numpy())
    m_net = compute_metrics(port["net_log_ret"].to_numpy())

    # Exit-type breakdown
    exit_split: dict[str, int] = defaultdict(int)
    bars_total = 0
    n_trades = len(trade_rows)
    for tr in trade_rows:
        exit_split[tr["exit_reason"]] += 1
        bars_total += int(tr["bars_held"])
    avg_bars = (bars_total / n_trades) if n_trades else 0.0

    return {
        "engine": cell.engine,
        "freq_min": cell.freq_min,
        "regime": cell.regime,
        "spread_bps": cell.cost_bps,
        "stop_mode": cell.stop_mode,
        "stop_k_sigma": cell.stop_k_sigma,
        "n_sessions": int(port.shape[0]),
        "n_pairs": int(df["pair"].nunique()),
        "n_trades": n_trades,
        "n_excluded_no_reversion": exclusions.get("no_reversion", 0),
        "n_excluded_hl_band": exclusions.get("hl_band", 0),
        "n_excluded_infeasible": exclusions.get("infeasible_at_cost", 0),
        "n_excluded_other": exclusions.get("no_cache", 0),
        "gross_total_pct": m_gross["total_return_pct"],
        "net_total_pct": m_net["total_return_pct"],
        "gross_ann_pct": m_gross["ann_return_pct"],
        "net_ann_pct": m_net["ann_return_pct"],
        "gross_sharpe": m_gross["sharpe"],
        "net_sharpe": m_net["sharpe"],
        "net_max_drawdown_pct": m_net["max_drawdown_pct"],
        "avg_bars_held": avg_bars,
        "exit_mean_revert": exit_split.get("mean_revert", 0),
        "exit_z_stop": exit_split.get("z_stop", 0),
        "exit_time_stop": exit_split.get("time_stop", 0),
        "exit_eod_squareoff": exit_split.get("session_close", 0),
        "exit_fold_close": exit_split.get("fold_boundary", 0),
    }


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def _emit_pair_fold_diagnostics(
    pair_fold_cache: dict[tuple[int, str, int], dict],
    pair_fold_keep: dict[tuple[int, str], Pair],
) -> None:
    rows = []
    for (fold_id, pair_key, freq_min), c in pair_fold_cache.items():
        pair = pair_fold_keep.get((fold_id, pair_key))
        if pair is None:
            continue
        fit = c["ou_fit"]
        hl_minutes = fit.half_life_minutes if fit.fit_ok else float("nan")
        # HL ratio = intraday HL minutes / (daily HL days * 375 minutes per day)
        daily_hl_min = pair.half_life * NSE_BARS_PER_SESSION
        hl_ratio = (
            (hl_minutes / daily_hl_min) if (fit.fit_ok and daily_hl_min > 0) else float("nan")
        )
        rows.append(
            {
                "fold_id": fold_id,
                "pair": pair_key,
                "freq_min": freq_min,
                "fit_ok": fit.fit_ok,
                "reject_reason": c.get("reject_reason", ""),
                "phi": fit.phi if fit.fit_ok else float("nan"),
                "kappa_per_bar": fit.kappa if fit.fit_ok else float("nan"),
                "mu_log_spread": fit.mu if fit.fit_ok else float("nan"),
                "sigma_eq": fit.sigma_eq if fit.fit_ok else float("nan"),
                "half_life_bars": fit.half_life_bars if fit.fit_ok else float("nan"),
                "half_life_minutes": hl_minutes,
                "half_life_sessions": (hl_minutes / NSE_BARS_PER_SESSION)
                if np.isfinite(hl_minutes)
                else float("nan"),
                "daily_inherited_hl_days": pair.half_life,
                "intraday_to_daily_hl_ratio": hl_ratio,
                "z_ou_test_mean": c.get("drift_mean", float("nan")),
                "z_ou_drift_flag": bool(
                    np.isfinite(c.get("drift_mean", float("nan")))
                    and abs(c["drift_mean"]) > settings.signal.ou.drift_flag_abs_mean
                ),
                "pair_beta": pair.beta,
                "one_plus_beta_over_2": (1.0 + pair.beta) / 2.0,
                "sector": pair.sector,
                "is_structural": pair.is_structural,
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "ou_pair_fold_diag.csv", index=False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("smoke", "reduced", "full"),
        default="smoke",
        help="Grid scope: smoke=1 cell, reduced=8 cells, full=34 cells.",
    )
    args = parser.parse_args()

    setup_logging(log_file=settings.paths.logs_dir / "15_phase3_ou.log")
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "=== Phase 3 OU — mode={} (engine config: {}) ===", args.mode, settings.signal.engine
    )

    if args.mode == "smoke":
        cells = _smoke_cells()
    elif args.mode == "reduced":
        cells = _reduced_cells()
    else:
        cells = _full_cells()
    # Limit to engine=ou cells in this script. (rolling_z baselines will need
    # script 13 invocation at the same coarser freq -- a [TODO] follow-up unit.)
    cells = [c for c in cells if c.engine == "ou"]
    logger.info("Cells to run (engine=ou only): {}", len(cells))

    # Reuse v2 helpers
    daily = pl.read_parquet(processed("daily_clean.parquet"))
    sectors = pl.read_parquet(Path("data/interim/sectors.parquet"))
    folds = _v2._build_overlap_folds(daily)
    folds_by_id = {f.fold_id: f for f in folds}
    logger.info("Built {} overlap folds", len(folds))

    # Daily selection per fold + intraday liquidity gate (reuse v2)
    pairs_by_fold: dict[int, list[Pair]] = {}
    for fold in folds:
        pairs_by_fold[fold.fold_id] = _v2._select_pairs_for_fold(fold, daily, sectors)
        logger.info(
            "Fold {}: {} pairs after daily selection",
            fold.fold_id,
            len(pairs_by_fold[fold.fold_id]),
        )

    pair_fold_keep: dict[tuple[int, str], Pair] = {}
    for fold in folds:
        from datetime import timedelta as _td

        liq_train_start = max(fold.train_end - _td(days=365), fold.train_start)
        liq_train_end = fold.train_end
        if liq_train_end < MINUTE_PANEL_START:
            liq_train_start = MINUTE_PANEL_START
            liq_train_end = min(fold.test_end, MINUTE_PANEL_START + _td(days=180))
        for p in pairs_by_fold[fold.fold_id]:
            ok, _, _ = _v2._passes_intraday_floor(p, liq_train_start, liq_train_end)
            if ok:
                pair_fold_keep[(fold.fold_id, p.key)] = p
    logger.info("After intraday liquidity gate: {} pair-folds", len(pair_fold_keep))

    # Build per-(pair-fold, freq) cache
    freqs_needed = sorted({c.freq_min for c in cells})
    pair_fold_cache: dict[tuple[int, str, int], dict] = {}

    t0 = time.time()
    n_fit = 0
    for (fold_id, pair_key), pair in pair_fold_keep.items():
        fold = folds_by_id[fold_id]
        for freq_min in freqs_needed:
            try:
                c = _ou_fit_for_pair_fold(
                    pair, fold, freq_min=freq_min, min_obs=settings.signal.ou.min_obs
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  fit failed: fold={} pair={} freq={} ({})", fold_id, pair_key, freq_min, exc
                )
                continue
            if c is not None:
                pair_fold_cache[(fold_id, pair_key, freq_min)] = c
                n_fit += 1
    t_fit = time.time() - t0
    logger.info("OU fit pass: {} (pair-fold, freq) computed in {:.1f}s", n_fit, t_fit)

    # Run cells
    all_metrics: list[dict] = []
    all_trades: list[dict] = []
    all_pair_sessions: list[dict] = []
    cell_times = []
    for i, cell in enumerate(cells, 1):
        cell_t0 = time.time()
        trades, pair_sessions, exclusions = _run_ou_cell(
            cell=cell,
            pair_fold_cache=pair_fold_cache,
            pair_fold_keep=pair_fold_keep,
            folds_by_id=folds_by_id,
        )
        cell_dt = time.time() - cell_t0
        cell_times.append(cell_dt)
        m = _aggregate_cell_metrics(pair_sessions, trades, cell, exclusions)
        if m is not None:
            all_metrics.append(m)
        all_trades.extend(trades)
        all_pair_sessions.extend(pair_sessions)
        logger.info(
            "[{}/{}] cell={} freq={} regime={} cost={} stop={} -> "
            "{} trades, {} pair-folds, {:.1f}s",
            i,
            len(cells),
            cell.engine,
            cell.freq_min,
            cell.regime,
            cell.cost_bps,
            cell.stop_mode,
            len(trades),
            m.get("n_pairs", 0),
            cell_dt,
        )

    # Emit CSVs
    pd.DataFrame(all_metrics).to_csv(REPORTS_DIR / "metrics_ou.csv", index=False)
    pd.DataFrame(all_trades).to_csv(REPORTS_DIR / "trades_ou.csv", index=False)
    pd.DataFrame(all_pair_sessions).to_csv(REPORTS_DIR / "pair_sessions_ou.csv", index=False)
    _emit_pair_fold_diagnostics(pair_fold_cache, pair_fold_keep)

    total_t = time.time() - t0
    logger.info(
        "=== Phase 3 OU complete: {} cells in {:.1f}s (mean {:.1f}s/cell) ===",
        len(cells),
        total_t,
        float(np.mean(cell_times)) if cell_times else 0.0,
    )
    print(
        f"\n=== 15_phase3_ou complete (mode={args.mode}, cells={len(cells)}, total_time={total_t:.1f}s) ===\n"
    )


if __name__ == "__main__":
    main()
