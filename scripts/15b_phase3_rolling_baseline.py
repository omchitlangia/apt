#!/usr/bin/env python3
"""Script 15b: coarse-bar rolling_z baselines (addendum #5, deferred from 15).

Runs the 4 deferred rolling_z baseline cells at bars {5, 15} × regimes
{A, B}, with cost levels {1, 3, 5, 8} bps re-stamped from a single base
gross run per (freq, regime).

- Thresholds: entry=2.0, exit=0.5, stop=3.5 (Stage-2 baseline, untouched).
- Window per pair: clip(round(half_life_days*375), [375, 1875]) MINUTES,
  then converted to bars at the active freq: max(round(window_min/freq), 1).
- max_holding: minute-equivalent same as v2, converted to bars likewise.
- Cost levels: re-stamped via `_net_pnl_for_cost` (same helper used by v2).

Outputs (reports/phase3_ou/):
  metrics_rolling_baseline.csv     - one row per (freq, regime, cost)
  trades_rolling_baseline.csv      - all trades at all cost levels
  pair_sessions_rolling_baseline.csv  - per (freq, regime, cost, pair, session)
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import defaultdict
from datetime import timedelta as _td
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from apt.backtest import Pair, compute_metrics
from apt.config import settings
from apt.intraday import (
    NSE_BARS_PER_SESSION,
    intraday_rolling_zscore,
    load_minute_pair,
    resample_within_session,
)
from apt.intraday.backtest import run_pair_fold
from apt.intraday.costs import CostBreakdown
from apt.intraday.signals import generate_signals_two_regime
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, processed

# Reuse v2 (script 13) helpers for fold construction and liquidity gate.
_SCRIPT_DIR = Path(__file__).parent
_V2_PATH = _SCRIPT_DIR / "13_phase3_intraday.py"
_spec = importlib.util.spec_from_file_location("script_13_b", _V2_PATH)
assert _spec is not None and _spec.loader is not None
_v2 = importlib.util.module_from_spec(_spec)
sys.modules["script_13_b"] = _v2
_spec.loader.exec_module(_v2)

MINUTE_ROOT = _v2.MINUTE_ROOT
MINUTE_PANEL_START = _v2.MINUTE_PANEL_START
MINUTE_PANEL_END = _v2.MINUTE_PANEL_END
SESSION_WARMUP_BARS = _v2.SESSION_WARMUP_BARS
MIN_ROLLING_WINDOW_MIN = _v2.MIN_ROLLING_WINDOW_MIN
MAX_ROLLING_WINDOW_MIN = _v2.MAX_ROLLING_WINDOW_MIN
_net_pnl_for_cost = _v2._net_pnl_for_cost

REPORTS_DIR = Path("reports/phase3_ou")
FREQS = (5, 15)
REGIMES = ("A", "B")
COSTS_BPS = (1, 3, 5, 8)


def _window_bars_minute_equiv(pair: Pair, freq_min: int) -> int:
    """Window per addendum #5: clip(round(HL*375), [375, 1875]) min, then /freq bars."""
    window_min = int(
        np.clip(
            round(pair.half_life * NSE_BARS_PER_SESSION),
            MIN_ROLLING_WINDOW_MIN,
            MAX_ROLLING_WINDOW_MIN,
        )
    )
    return max(int(round(window_min / freq_min)), 2)


def _max_holding_bars(pair: Pair, freq_min: int) -> int:
    """Match script 15 + script 13 semantics, expressed in bars at active freq."""
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
    return max(int(round(max_min / freq_min)), 1)


def _compute_rolling_pair_fold(
    pair: Pair,
    fold,
    *,
    freq_min: int,
    base_cost_log: float,
):
    """Load minute panel, resample, build rolling z, run both regimes.

    Returns dict keyed by regime with the cached IntradayPairFoldResult,
    plus window_bars and max_holding_bars metadata. None on insufficient data.
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

    # Log spread on the resampled grid (NaN where non-tradeable)
    with np.errstate(invalid="ignore", divide="ignore"):
        py = np.where(rs.tradeable, rs.close_y, np.nan)
        px = np.where(rs.tradeable, rs.close_x, np.nan)
        spread_full = np.log(py) - pair.beta * np.log(px) - pair.alpha

    ts_full = pd.DatetimeIndex(rs.timestamps)
    test_mask = (ts_full.date >= test_start) & (ts_full.date <= test_end)

    window = _window_bars_minute_equiv(pair, freq_min)
    if test_mask.sum() < window:
        return None

    z_full = intraday_rolling_zscore(
        spread_full,
        rs.session_id,
        window=window,
        session_warmup_bars=SESSION_WARMUP_BARS,
    )
    z_test = z_full[test_mask]
    spread_test = spread_full[test_mask]
    ts_test = ts_full[test_mask]
    sids_test = rs.session_id[test_mask]
    _, sids_test = np.unique(sids_test, return_inverse=True)
    sids_test = sids_test.astype(np.int32)
    tradeable_test = rs.tradeable[test_mask]

    max_holding = _max_holding_bars(pair, freq_min)

    out: dict = {"window_bars": window, "max_holding_bars": max_holding}
    for regime in REGIMES:
        sig = generate_signals_two_regime(
            z_test,
            sids_test,
            tradeable_test,
            regime=regime,
            entry=settings.signal.entry_z,
            exit=settings.signal.exit_z,
            stop=settings.signal.stop_z,
            max_holding=max_holding,
        )
        res = run_pair_fold(
            fold_id=fold.fold_id,
            pair_key=pair.key,
            timestamps=ts_test,
            session_id=sids_test,
            spread=spread_test,
            z=z_test,
            signals=sig,
            cost_log_per_round_trip=base_cost_log,
            finalize_fold_boundary=(regime == "B"),
        )
        out[regime] = res
    return out


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "15b_phase3_rolling_baseline.log")
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 3 rolling_z coarse-bar baseline (addendum #5) ===")

    daily = pl.read_parquet(processed("daily_clean.parquet"))
    sectors = pl.read_parquet(Path("data/interim/sectors.parquet"))
    folds = _v2._build_overlap_folds(daily)
    folds_by_id = {f.fold_id: f for f in folds}
    logger.info("Built {} overlap folds", len(folds))

    pairs_by_fold: dict[int, list[Pair]] = {}
    for fold in folds:
        pairs_by_fold[fold.fold_id] = _v2._select_pairs_for_fold(fold, daily, sectors)

    pair_fold_keep: dict[tuple[int, str], Pair] = {}
    for fold in folds:
        liq_train_start = max(fold.train_end - _td(days=365), fold.train_start)
        liq_train_end = fold.train_end
        if liq_train_end < MINUTE_PANEL_START:
            liq_train_start = MINUTE_PANEL_START
            liq_train_end = min(fold.test_end, MINUTE_PANEL_START + _td(days=180))
        for p in pairs_by_fold[fold.fold_id]:
            ok, _, _ = _v2._passes_intraday_floor(p, liq_train_start, liq_train_end)
            if ok:
                pair_fold_keep[(fold.fold_id, p.key)] = p
    logger.info("After liquidity gate: {} pair-folds", len(pair_fold_keep))

    base_cost = CostBreakdown(total_spread_bps=3)
    base_cost_log = base_cost.cost_log_per_pair_round_trip

    # Cache: (fold_id, pair_key, freq_min) -> {A: result, B: result, window_bars, max_holding_bars}
    cache: dict[tuple[int, str, int], dict] = {}
    t0 = time.time()
    for (fold_id, pair_key), pair in pair_fold_keep.items():
        fold = folds_by_id[fold_id]
        for freq_min in FREQS:
            try:
                res = _compute_rolling_pair_fold(
                    pair, fold, freq_min=freq_min, base_cost_log=base_cost_log
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  fit failed: fold={} pair={} freq={} ({})",
                    fold_id,
                    pair_key,
                    freq_min,
                    exc,
                )
                continue
            if res is not None:
                cache[(fold_id, pair_key, freq_min)] = res
    logger.info(
        "Computed {} (pair-fold, freq) base runs in {:.1f}s",
        len(cache),
        time.time() - t0,
    )

    # Cell loop: (freq, regime, cost) — restamp cost via _net_pnl_for_cost.
    metric_rows: list[dict] = []
    trade_rows: list[dict] = []
    pair_session_rows: list[dict] = []

    for freq_min in FREQS:
        for regime in REGIMES:
            # Window stats per (freq, regime) for diagnostics
            windows = [
                cache[(fid, pk, freq_min)]["window_bars"]
                for (fid, pk) in pair_fold_keep
                if (fid, pk, freq_min) in cache
            ]
            for cb in (CostBreakdown(total_spread_bps=c) for c in COSTS_BPS):
                new_cost_log = cb.cost_log_per_pair_round_trip
                pair_session_records: list[dict] = []
                cell_trades: list = []
                bars_held_total = 0
                exit_split: dict[str, int] = defaultdict(int)

                for (fold_id, pair_key), _pair in pair_fold_keep.items():
                    key = (fold_id, pair_key, freq_min)
                    if key not in cache:
                        continue
                    pf_res = cache[key][regime]
                    net_arr, new_trades = _net_pnl_for_cost(
                        pf_res, new_cost_log=new_cost_log, base_cost_log=base_cost_log
                    )

                    df = pd.DataFrame(
                        {
                            "date": pd.DatetimeIndex(pf_res.timestamps).date,
                            "gross_log_ret": pf_res.gross_log_ret,
                            "net_log_ret": net_arr,
                        }
                    )
                    per_sess = df.groupby("date", as_index=False)[
                        ["gross_log_ret", "net_log_ret"]
                    ].sum()
                    per_sess["active"] = (
                        df.assign(active=(df["gross_log_ret"] != 0) | (df["net_log_ret"] != 0))
                        .groupby("date")["active"]
                        .any()
                        .reindex(per_sess["date"])
                        .values
                    )
                    for _, row in per_sess.iterrows():
                        rec = {
                            "engine": "rolling_z",
                            "freq_min": freq_min,
                            "regime": regime,
                            "spread_bps": cb.total_spread_bps,
                            "fold_id": fold_id,
                            "pair": pair_key,
                            "date": row["date"],
                            "gross_log_ret": float(row["gross_log_ret"]),
                            "net_log_ret": float(row["net_log_ret"]),
                            "active": bool(row["active"]),
                        }
                        pair_session_records.append(rec)
                        pair_session_rows.append(rec)
                    cell_trades.extend(new_trades)
                    for tr in new_trades:
                        exit_split[tr.exit_reason] += 1
                        bars_held_total += int(tr.bars_held)
                        trade_rows.append(
                            {
                                "engine": "rolling_z",
                                "freq_min": freq_min,
                                "regime": regime,
                                "spread_bps": cb.total_spread_bps,
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
                            }
                        )

                if not pair_session_records:
                    metric_rows.append(
                        {
                            "engine": "rolling_z",
                            "freq_min": freq_min,
                            "regime": regime,
                            "spread_bps": cb.total_spread_bps,
                            "n_pairs": 0,
                            "n_trades": 0,
                            "window_bars_p50": int(np.median(windows)) if windows else 0,
                            "window_bars_min": int(np.min(windows)) if windows else 0,
                            "window_bars_max": int(np.max(windows)) if windows else 0,
                        }
                    )
                    continue

                df_all = pd.DataFrame(pair_session_records)
                port = (
                    df_all.groupby("date", as_index=False)
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
                n_trades = len(cell_trades)
                avg_bars = (bars_held_total / n_trades) if n_trades else 0.0
                metric_rows.append(
                    {
                        "engine": "rolling_z",
                        "freq_min": freq_min,
                        "regime": regime,
                        "spread_bps": cb.total_spread_bps,
                        "n_sessions": int(port.shape[0]),
                        "n_pairs": int(df_all["pair"].nunique()),
                        "n_trades": n_trades,
                        "window_bars_p50": int(np.median(windows)) if windows else 0,
                        "window_bars_min": int(np.min(windows)) if windows else 0,
                        "window_bars_max": int(np.max(windows)) if windows else 0,
                        "gross_total_pct": m_gross["total_return_pct"],
                        "net_total_pct": m_net["total_return_pct"],
                        "gross_ann_pct": m_gross["ann_return_pct"],
                        "net_ann_pct": m_net["ann_return_pct"],
                        "gross_sharpe": m_gross["sharpe"],
                        "net_sharpe": m_net["sharpe"],
                        "net_max_drawdown_pct": m_net["max_drawdown_pct"],
                        "avg_bars_held": avg_bars,
                        "exit_mean_revert": exit_split.get("mean_revert", 0),
                        "exit_stop": exit_split.get("stop", 0),
                        "exit_time": exit_split.get("time", 0),
                        "exit_session_close": exit_split.get("session_close", 0),
                        "exit_fold_boundary": exit_split.get("fold_boundary", 0),
                    }
                )
                logger.info(
                    "freq={} regime={} cost={} -> {} trades, {} pair-folds, "
                    "net_ann={:.2f}% net_sharpe={:.3f}",
                    freq_min,
                    regime,
                    cb.total_spread_bps,
                    n_trades,
                    int(df_all["pair"].nunique()),
                    m_net["ann_return_pct"],
                    m_net["sharpe"],
                )

    pd.DataFrame(metric_rows).to_csv(REPORTS_DIR / "metrics_rolling_baseline.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(REPORTS_DIR / "trades_rolling_baseline.csv", index=False)
    pd.DataFrame(pair_session_rows).to_csv(
        REPORTS_DIR / "pair_sessions_rolling_baseline.csv", index=False
    )
    logger.info("=== rolling_z baseline complete: {} metric rows ===", len(metric_rows))
    print(f"\n=== 15b_phase3_rolling_baseline complete (cells={len(metric_rows)}) ===\n")


if __name__ == "__main__":
    main()
