#!/usr/bin/env python3
"""Script 17: Unit K — adaptive-equilibrium (per-session local-level μ_t).

Implements the locked rulings in ``docs/kalman_design.md`` Decision Log:

* μ-only state; β frozen (daily EG fit). TOD ignored.
* Per-session causal local-level filter (``apt.stats.kalman``): μ updates
  at session close, applied from next session open. Re-anchor half-life in
  SESSIONS, one GLOBAL value selected on TRAIN only.
* Signal centre Z_k = (X − μ_t) / σ_eq_resid; (κ, σ_eq_resid) fit on TRAIN
  residuals; Bertram a* re-solved per cost level under main's (1+β) billing.
* Static-HL gate (same survivors as the OU run). Regime B only.
* 8 cells: kalman_mu × OU-Bertram × freq{5,15} × cost{1,3,5,8} × B (the
  rolling_z-with-kalman arm is SKIPPED — would make 16; see Decision Log §6).

Selection protocol (TRAIN only):
  grid H ∈ {∞, 20, 10, 5} sessions. For each H, per surviving pair-fold:
  run filter on train, fit residual OU, ABSORPTION GUARD (residual HL within
  [0.5×, 1.5×] of frozen-μ HL); admissible configs scored by the analytic
  Bertram net-return-per-unit-time on the train residual fit (β-aware 3 bps
  reference). One global H (max mean train criterion among admissible) → test.

Outputs (reports/phase3_kalman/):
  selection_table.csv          — criterion + guard per (H, freq, pair-fold)
  selection_summary.csv        — aggregate per H, the chosen H flagged
  metrics_kalman.csv           — 16-cell metrics, gross + net
  trades_kalman.csv            — all trades (v2-cost-beta schema)
  pair_sessions_kalman.csv     — per (cell, pair-fold, session)
  drift_before_after.csv       — frozen Z-OU mean vs adaptive Z_k mean per pf
  mu_overlay_<pair>_fold<k>.csv — spread + frozen μ + μ_t for the money figure
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from datetime import timedelta as _td
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from apt.backtest import Pair, compute_metrics
from apt.config import settings
from apt.intraday import NSE_BARS_PER_SESSION, load_minute_pair, resample_within_session
from apt.intraday.backtest import run_pair_fold
from apt.intraday.costs import TRADE_CSV_SCHEMA_VERSION, CostBreakdown
from apt.intraday.signals import generate_signals_ou
from apt.stats.kalman import run_local_level_mu
from apt.stats.ou import bertram_threshold, fit_ou_params
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, processed

# Reuse v2 (script 13) helpers for fold construction, selection, liquidity gate.
_SCRIPT_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("script_13_k", _SCRIPT_DIR / "13_phase3_intraday.py")
assert _spec is not None and _spec.loader is not None
_v2 = importlib.util.module_from_spec(_spec)
sys.modules["script_13_k"] = _v2
_spec.loader.exec_module(_v2)

MINUTE_ROOT = _v2.MINUTE_ROOT
MINUTE_PANEL_START = _v2.MINUTE_PANEL_START
MINUTE_PANEL_END = _v2.MINUTE_PANEL_END

REPORTS_DIR = Path("reports/phase3_kalman")

FREQS = (5, 15)
COSTS_BPS = (1, 3, 5, 8)
REGIME = "B"
HALF_LIFE_GRID_SESSIONS = (math.inf, 20.0, 10.0, 5.0)
ABSORPTION_GUARD = (0.5, 1.5)  # residual HL must be within [0.5x, 1.5x] frozen HL
SELECTION_REF_COST_BPS = 3  # reference cost for the train selection criterion


class Cell(NamedTuple):
    freq_min: int
    cost_bps: int


# ----------------------------------------------------------------------
# Per-(pair-fold, freq) cache — frozen OU fit + train/test spread + sids
# ----------------------------------------------------------------------
def _kalman_fit_for_pair_fold(pair: Pair, fold, *, freq_min: int, min_obs: int) -> dict | None:
    """Frozen OU fit on train + full-window spread/sids for the filter."""
    test_start = max(fold.test_start, MINUTE_PANEL_START)
    test_end = min(fold.test_end, MINUTE_PANEL_END)
    aligned = load_minute_pair(pair.y_sym, pair.x_sym, fold.train_start, test_end, root=MINUTE_ROOT)
    if aligned.n_bars == 0:
        return None
    rs = resample_within_session(aligned, freq_minutes=freq_min)
    if rs.n_bars == 0:
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        py = np.where(rs.tradeable, rs.close_y, np.nan)
        px = np.where(rs.tradeable, rs.close_x, np.nan)
        spread_full = np.log(py) - pair.beta * np.log(px) - pair.alpha

    ts_full = pd.DatetimeIndex(rs.timestamps)
    train_mask = (ts_full.date >= fold.train_start) & (ts_full.date <= fold.train_end)
    test_mask = (ts_full.date >= test_start) & (ts_full.date <= test_end)
    if test_mask.sum() < min_obs or train_mask.sum() < min_obs:
        return None

    frozen = fit_ou_params(spread_full[train_mask], freq_minutes=freq_min, min_obs=min_obs)
    if not frozen.fit_ok:
        return {"frozen_fit": frozen, "reject_reason": frozen.reason}

    # Dense-rank session ids over the FULL window so the filter runs train→test.
    _, sids_full = np.unique(rs.session_id, return_inverse=True)
    sids_full = sids_full.astype(np.int32)

    return {
        "frozen_fit": frozen,
        "spread_full": spread_full,
        "sids_full": sids_full,
        "ts_full": ts_full,
        "tradeable_full": rs.tradeable,
        "train_mask": np.asarray(train_mask),
        "test_mask": np.asarray(test_mask),
        "mu_init": float(frozen.mu),
        "hl_minutes_frozen": float(frozen.half_life_minutes),
        "beta": float(pair.beta),
        "reject_reason": "",
    }


def _hl_band_b() -> tuple[float, float]:
    band = settings.signal.ou.half_life_band
    return float(band.B_min_minutes), float(band.B_max_minutes)


def _max_holding_bars(pair: Pair, freq_min: int) -> int:
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


# ----------------------------------------------------------------------
# Residual OU fit under a given half-life (train-only)
# ----------------------------------------------------------------------
def _residual_ou_on_train(cache: dict, half_life_sessions: float, freq_min: int) -> dict:
    """Run the filter over TRAIN, fit OU on the train residual.

    Returns {fit_ok, kappa, sigma_eq_resid, hl_minutes_resid, guard_ratio,
             admissible}.
    """
    tm = cache["train_mask"]
    spread_tr = cache["spread_full"][tm]
    sids_tr = cache["sids_full"][tm]
    trd_tr = cache["tradeable_full"][tm]
    res = run_local_level_mu(
        spread_tr,
        sids_tr,
        mu_init=cache["mu_init"],
        half_life_sessions=half_life_sessions,
        tradeable=trd_tr,
    )
    if not res.fit_ok:
        return {"fit_ok": False, "reason": res.reason}
    rfit = fit_ou_params(res.residual, freq_minutes=freq_min, min_obs=settings.signal.ou.min_obs)
    if not rfit.fit_ok:
        return {"fit_ok": False, "reason": rfit.reason}
    ratio = rfit.half_life_minutes / cache["hl_minutes_frozen"]
    admissible = ABSORPTION_GUARD[0] <= ratio <= ABSORPTION_GUARD[1]
    return {
        "fit_ok": True,
        "kappa": rfit.kappa,
        "sigma_eq_resid": rfit.sigma_eq,
        "hl_minutes_resid": rfit.half_life_minutes,
        "guard_ratio": float(ratio),
        "admissible": bool(admissible),
        "residual_fit": rfit,
    }


# ----------------------------------------------------------------------
# TRAIN-only selection of the global half-life
# ----------------------------------------------------------------------
def _select_half_life(
    pair_fold_cache: dict, pair_fold_keep: dict, survivors: dict
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Grid over HALF_LIFE_GRID_SESSIONS; criterion = analytic Bertram
    net-return-per-unit-time on TRAIN residuals at the β-aware reference cost,
    subject to the absorption guard. Returns (chosen_H, full_table, summary)."""
    cb_ref = CostBreakdown(total_spread_bps=SELECTION_REF_COST_BPS)
    rows: list[dict] = []
    for H in HALF_LIFE_GRID_SESSIONS:
        for fold_id, pair_key, freq_min in survivors:
            cache = pair_fold_cache[(fold_id, pair_key, freq_min)]
            r = _residual_ou_on_train(cache, H, freq_min)
            row = {
                "half_life_sessions": (np.inf if math.isinf(H) else H),
                "fold_id": fold_id,
                "pair": pair_key,
                "freq_min": freq_min,
                "fit_ok": r["fit_ok"],
            }
            if r["fit_ok"]:
                c_ref = cb_ref.billed_cost_log_per_pair_round_trip(beta=cache["beta"])
                th = bertram_threshold(r["residual_fit"], cost_log_per_round_trip=c_ref)
                row.update(
                    {
                        "hl_minutes_resid": r["hl_minutes_resid"],
                        "guard_ratio": r["guard_ratio"],
                        "admissible": r["admissible"],
                        "a_entry_z": th.a_entry_z if th.fit_ok else np.nan,
                        "ret_per_unit_time": th.expected_return_per_unit_time
                        if th.fit_ok
                        else np.nan,
                        "bertram_ok": bool(th.fit_ok),
                    }
                )
            else:
                row.update(
                    {
                        "hl_minutes_resid": np.nan,
                        "guard_ratio": np.nan,
                        "admissible": False,
                        "a_entry_z": np.nan,
                        "ret_per_unit_time": np.nan,
                        "bertram_ok": False,
                    }
                )
            rows.append(row)
    table = pd.DataFrame(rows)

    # Aggregate per H over ADMISSIBLE + bertram_ok survivors.
    summ_rows: list[dict] = []
    for H in HALF_LIFE_GRID_SESSIONS:
        Hval = np.inf if math.isinf(H) else H
        sub = table[(table.half_life_sessions == Hval) & table.admissible & table.bertram_ok]
        n_adm = int(sub.shape[0])
        n_total = int((table.half_life_sessions == Hval).sum())
        crit = float(sub.ret_per_unit_time.mean()) if n_adm else float("nan")
        summ_rows.append(
            {
                "half_life_sessions": Hval,
                "n_admissible": n_adm,
                "n_total": n_total,
                "mean_ret_per_unit_time": crit,
                "globally_admissible": n_adm > 0,
            }
        )
    summary = pd.DataFrame(summ_rows)

    adm = summary[summary.globally_admissible & summary.mean_ret_per_unit_time.notna()]
    if adm.empty:
        chosen = np.inf  # fall back to frozen
    else:
        chosen = float(
            adm.sort_values("mean_ret_per_unit_time", ascending=False).iloc[0]["half_life_sessions"]
        )
    summary["chosen"] = summary.half_life_sessions == chosen
    return chosen, table, summary


# ----------------------------------------------------------------------
# Run one cell (kalman_mu × OU-Bertram) at the selected half-life
# ----------------------------------------------------------------------
def _run_cell(
    cell: Cell, half_life: float, pair_fold_cache: dict, pair_fold_keep: dict, survivors: dict
) -> tuple[list[dict], list[dict]]:
    cb = CostBreakdown(total_spread_bps=cell.cost_bps)
    trade_rows: list[dict] = []
    pair_session_rows: list[dict] = []
    for fold_id, pair_key, freq_min in survivors:
        if freq_min != cell.freq_min:
            continue
        cache = pair_fold_cache[(fold_id, pair_key, freq_min)]
        pair = pair_fold_keep[(fold_id, pair_key)]
        # Residual OU on train (frozen for test) under the selected H.
        rinfo = _residual_ou_on_train(cache, half_life, freq_min)
        if not rinfo["fit_ok"]:
            continue
        residual_fit = rinfo["residual_fit"]
        sigma_eq_resid = rinfo["sigma_eq_resid"]
        cost_log = cb.billed_cost_log_per_pair_round_trip(beta=cache["beta"])
        th = bertram_threshold(residual_fit, cost_log_per_round_trip=cost_log)
        if not th.fit_ok:
            continue

        # Run filter over FULL window, slice μ_path to test (causal carry).
        full = run_local_level_mu(
            cache["spread_full"],
            cache["sids_full"],
            mu_init=cache["mu_init"],
            half_life_sessions=half_life,
            tradeable=cache["tradeable_full"],
        )
        tmask = cache["test_mask"]
        spread_test = cache["spread_full"][tmask]
        mu_test = full.mu_path[tmask]
        z_k = (spread_test - mu_test) / sigma_eq_resid
        ts_test = cache["ts_full"][tmask]
        sids_test_raw = cache["sids_full"][tmask]
        _, sids_test = np.unique(sids_test_raw, return_inverse=True)
        sids_test = sids_test.astype(np.int32)
        trd_test = cache["tradeable_full"][tmask]

        max_h = _max_holding_bars(pair, freq_min)
        sig = generate_signals_ou(
            z_k,
            sids_test,
            trd_test,
            regime=REGIME,
            a_entry_z=th.a_entry_z,
            stop_mode="none",
            stop_k_sigma=0.0,
            max_holding=max_h,
        )
        result = run_pair_fold(
            fold_id=fold_id,
            pair_key=pair_key,
            timestamps=ts_test,
            session_id=sids_test,
            spread=spread_test,
            z=z_k,
            signals=sig,
            cost_log_per_round_trip=cost_log,
            finalize_fold_boundary=True,
            pair_beta=cache["beta"],
        )

        df = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(result.timestamps).date,
                "gross_log_ret": result.gross_log_ret,
                "net_log_ret": result.net_log_ret,
            }
        )
        per = df.groupby("date", as_index=False)[["gross_log_ret", "net_log_ret"]].sum()
        per["active"] = (
            df.assign(a=(df.gross_log_ret != 0) | (df.net_log_ret != 0))
            .groupby("date")["a"]
            .any()
            .reindex(per["date"])
            .values
        )
        for _, r in per.iterrows():
            pair_session_rows.append(
                {
                    "engine": "kalman_mu",
                    "freq_min": cell.freq_min,
                    "regime": REGIME,
                    "spread_bps": cell.cost_bps,
                    "stop_mode": "none",
                    "fold_id": fold_id,
                    "pair": pair_key,
                    "date": r["date"],
                    "gross_log_ret": r.gross_log_ret,
                    "net_log_ret": r.net_log_ret,
                    "active": bool(r.active),
                }
            )
        for t in result.trades:
            trade_rows.append(
                {
                    "engine": "kalman_mu",
                    "freq_min": cell.freq_min,
                    "regime": REGIME,
                    "spread_bps": cell.cost_bps,
                    "stop_mode": "none",
                    "fold_id": t.fold_id,
                    "pair": t.pair_key,
                    "side": "long_spread" if t.direction == 1 else "short_spread",
                    "entry_ts": t.entry_ts.isoformat(),
                    "exit_ts": t.exit_ts.isoformat(),
                    "z_entry": t.entry_z,
                    "z_exit": t.exit_z,
                    "bars_held": t.bars_held,
                    "sessions_held": t.sessions_held,
                    "gross_log_pnl": t.gross_log_pnl,
                    "cost_log": t.cost_log,
                    "net_log_pnl": t.net_log_pnl,
                    "exit_reason": t.exit_reason,
                    "a_entry_z": th.a_entry_z,
                    "half_life_sessions": half_life,
                    "pair_beta": t.pair_beta,
                    "cost_log_per_pair_rt": t.cost_log,
                    "n_legs": 2,
                    "schema_version": TRADE_CSV_SCHEMA_VERSION,
                }
            )
    return trade_rows, pair_session_rows


def _aggregate(
    pair_sessions: list[dict], trades: list[dict], cell: Cell, half_life: float
) -> dict | None:
    if not pair_sessions:
        return None
    df = pd.DataFrame(pair_sessions)
    port = (
        df.groupby("date", as_index=False)
        .agg(
            gross_log_ret=("gross_log_ret", "mean"),
            net_log_ret=("net_log_ret", "mean"),
            n_pairs=("pair", "nunique"),
        )
        .sort_values("date")
    )
    mg = compute_metrics(port.gross_log_ret.to_numpy())
    mn = compute_metrics(port.net_log_ret.to_numpy())
    return {
        "engine": "kalman_mu",
        "freq_min": cell.freq_min,
        "regime": REGIME,
        "spread_bps": cell.cost_bps,
        "stop_mode": "none",
        "half_life_sessions": half_life,
        "n_pairs": int(df.pair.nunique()),
        "n_trades": len(trades),
        "n_sessions": int(port.shape[0]),
        "gross_total_pct": mg["total_return_pct"],
        "net_total_pct": mn["total_return_pct"],
        "gross_ann_pct": mg["ann_return_pct"],
        "net_ann_pct": mn["ann_return_pct"],
        "gross_sharpe": mg["sharpe"],
        "net_sharpe": mn["sharpe"],
        "net_max_drawdown_pct": mn["max_drawdown_pct"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full"], default="full")
    ap.parse_args()
    setup_logging(log_file=settings.paths.logs_dir / "17_kalman_equilibrium.log")
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=== Unit K — adaptive-equilibrium (per-session local-level mu) ===")

    daily = pl.read_parquet(processed("daily_clean.parquet"))
    sectors = pl.read_parquet(Path("data/interim/sectors.parquet"))
    folds = _v2._build_overlap_folds(daily)
    folds_by_id = {f.fold_id: f for f in folds}

    pairs_by_fold = {f.fold_id: _v2._select_pairs_for_fold(f, daily, sectors) for f in folds}
    pair_fold_keep: dict[tuple[int, str], Pair] = {}
    for fold in folds:
        lo = max(fold.train_end - _td(days=365), fold.train_start)
        hi = fold.train_end
        if hi < MINUTE_PANEL_START:
            lo = MINUTE_PANEL_START
            hi = min(fold.test_end, MINUTE_PANEL_START + _td(days=180))
        for p in pairs_by_fold[fold.fold_id]:
            ok, _, _ = _v2._passes_intraday_floor(p, lo, hi)
            if ok:
                pair_fold_keep[(fold.fold_id, p.key)] = p
    logger.info("After intraday liquidity gate: {} pair-folds", len(pair_fold_keep))

    # Build cache + static HL-band survivors (Regime B band on frozen fit).
    hl_min, hl_max = _hl_band_b()
    cache: dict[tuple[int, str, int], dict] = {}
    survivors: dict[tuple[int, str, int], bool] = {}
    t0 = time.time()
    for (fold_id, pair_key), pair in pair_fold_keep.items():
        for freq_min in FREQS:
            try:
                c = _kalman_fit_for_pair_fold(
                    pair,
                    folds_by_id[fold_id],
                    freq_min=freq_min,
                    min_obs=settings.signal.ou.min_obs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "  fit failed fold={} pair={} freq={}: {}", fold_id, pair_key, freq_min, exc
                )
                continue
            if c is None or not c["frozen_fit"].fit_ok:
                continue
            cache[(fold_id, pair_key, freq_min)] = c
            if hl_min <= c["hl_minutes_frozen"] <= hl_max:
                survivors[(fold_id, pair_key, freq_min)] = True
    logger.info(
        "Cache: {} (pf,freq); static-HL-band survivors: {} in {:.1f}s",
        len(cache),
        len(survivors),
        time.time() - t0,
    )

    # ---- TRAIN-only selection of the global half-life ----
    chosen_H, sel_table, sel_summary = _select_half_life(cache, pair_fold_keep, survivors)
    sel_table.to_csv(REPORTS_DIR / "selection_table.csv", index=False)
    sel_summary.to_csv(REPORTS_DIR / "selection_summary.csv", index=False)
    logger.info("Selected global re-anchor half-life: {} sessions", chosen_H)
    logger.info("\n{}", sel_summary.to_string(index=False))

    # ---- Run the 16 cells at the selected H ----
    cells = [Cell(f, c) for f in FREQS for c in COSTS_BPS]
    all_metrics, all_trades, all_ps = [], [], []
    for cell in cells:
        tr, ps = _run_cell(cell, chosen_H, cache, pair_fold_keep, survivors)
        m = _aggregate(ps, tr, cell, chosen_H)
        if m is not None:
            all_metrics.append(m)
        all_trades.extend(tr)
        all_ps.extend(ps)
        logger.info(
            "cell freq={} cost={} -> {} trades, {} pf",
            cell.freq_min,
            cell.cost_bps,
            len(tr),
            m["n_pairs"] if m else 0,
        )
    pd.DataFrame(all_metrics).to_csv(REPORTS_DIR / "metrics_kalman.csv", index=False)
    pd.DataFrame(all_trades).to_csv(REPORTS_DIR / "trades_kalman.csv", index=False)
    pd.DataFrame(all_ps).to_csv(REPORTS_DIR / "pair_sessions_kalman.csv", index=False)

    # ---- Drift before/after + μ-overlay artifacts ----
    _emit_drift_and_overlays(cache, survivors, chosen_H)

    print(f"\n=== 17_kalman_equilibrium complete: H={chosen_H}, {len(cells)} cells ===\n")
    return 0


def _emit_drift_and_overlays(cache: dict, survivors: dict, half_life: float) -> None:
    """Drift before (frozen Z-OU) vs after (adaptive Z_k) per pair-fold, plus
    spread + frozen μ + μ_t overlays for INDUSINDBK folds 4 and 6."""
    drift_rows: list[dict] = []
    overlays = {
        (4, "INDUSINDBK/HDFCBANK", 5),
        (6, "INDUSINDBK/HDFCBANK", 5),
        (4, "INDUSINDBK/HDFCBANK", 15),
        (6, "INDUSINDBK/HDFCBANK", 15),
    }
    for fold_id, pair_key, freq_min in survivors:
        cache_e = cache[(fold_id, pair_key, freq_min)]
        rinfo = _residual_ou_on_train(cache_e, half_life, freq_min)
        if not rinfo["fit_ok"]:
            continue
        sig_eq = rinfo["sigma_eq_resid"]
        frozen = cache_e["frozen_fit"]
        tmask = cache_e["test_mask"]
        spread_test = cache_e["spread_full"][tmask]
        # before: frozen Z-OU
        z_frozen = (spread_test - frozen.mu) / frozen.sigma_eq
        # after: adaptive
        full = run_local_level_mu(
            cache_e["spread_full"],
            cache_e["sids_full"],
            mu_init=cache_e["mu_init"],
            half_life_sessions=half_life,
            tradeable=cache_e["tradeable_full"],
        )
        mu_test = full.mu_path[tmask]
        z_k = (spread_test - mu_test) / sig_eq
        fb = np.isfinite(z_frozen)
        fa = np.isfinite(z_k)
        drift_rows.append(
            {
                "fold_id": fold_id,
                "pair": pair_key,
                "freq_min": freq_min,
                "drift_frozen_sigma_eq": float(np.mean(z_frozen[fb])) if fb.any() else np.nan,
                "drift_kalman_sigma_eq": float(np.mean(z_k[fa])) if fa.any() else np.nan,
                "half_life_sessions": half_life,
            }
        )
        if (fold_id, pair_key, freq_min) in overlays:
            ts_test = cache_e["ts_full"][tmask]
            ov = pd.DataFrame(
                {
                    "ts": pd.DatetimeIndex(ts_test),
                    "spread": spread_test,
                    "mu_frozen": frozen.mu,
                    "mu_kalman": mu_test,
                }
            )
            tag = pair_key.replace("/", "_")
            ov.to_csv(REPORTS_DIR / f"mu_overlay_{tag}_fold{fold_id}_f{freq_min}.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(REPORTS_DIR / "drift_before_after.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
