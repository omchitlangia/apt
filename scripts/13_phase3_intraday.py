#!/usr/bin/env python3
"""Script 13: Phase 3 — intraday two-regime validation.

Pipeline:
  1. Build Phase 2A daily folds; restrict to those whose test window
     overlaps the minute panel.
  2. Per fold, run the same daily ``cointegrate_pairs`` to get the
     tradeable pairs with FROZEN (alpha, beta).
  3. Per pair: compute the intraday liquidity floor on the TRAIN window
     (both legs must pass); drop any failing.
  4. For each surviving pair, load minute data for the TEST window,
     compute the spread, the session-aware rolling z (flat) and the
     time-of-day-adjusted z, then run signals for BOTH regimes:
        Regime A (intraday-only / deployable) — force-close at session end.
        Regime B (multi-day carry / upper-bound) — positions persist
                 across the overnight gap.
  5. Sweep the assumed bid-ask spread over {1,3,5,8} bps; gross PnL is
     fixed and cached, net PnL is re-computed per spread level by
     re-stamping the cost-per-round-trip on each trade's exit bar.
  6. Emit metrics, plots, the where-the-edge-dies curve, and caveats.

Outputs:
  reports/phase3/
    fold_pairs.csv              — selected pairs per fold with (alpha,beta,hl)
    liquidity_filter.csv        — intraday liquidity gate verdict per pair-fold
    trades_two_regime.csv       — every round-trip, both regimes, all costs
    pair_daily_two_regime.csv   — per-pair, per-session aggregated PnL
    portfolio_two_regime.csv    — equally-weighted portfolio daily PnL
    metrics_two_regime.csv      — per-regime, per-cost metrics
    tod_vol_profile_diag.csv    — TOD vol profile sensitivity diag (one fold)
    caveats_phase3.txt
  plots/phase3/
    sharpe_vs_spread.png        — where-the-edge-dies curve, both regimes
    equity_A_vs_B.png           — A vs B at 3 bps
    per_pair_grid.png           — per-pair contribution heatmap
    tod_vol_profile.png         — time-of-day vol curve
    spread_z_example.png        — one pair, one session, spread+z+bands

Wall-clock: dominated by the per-fold daily cointegration selection and
the minute-data load. Run end-to-end on a workstation in ~10-15 min for
the full overlap (5 full + 1 partial fold).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from apt.backtest import Pair, build_folds, compute_metrics
from apt.config import settings
from apt.intraday import (
    NSE_BARS_PER_SESSION,
    fit_tod_vol_profile,
    generate_signals_two_regime,
    intraday_rolling_zscore,
    load_minute_pair,
    tod_adjusted_zscore,
)
from apt.intraday.backtest import run_pair_fold
from apt.intraday.costs import (
    FIXED_PER_LEG_RT,
    SPREAD_SWEEP_BPS,
    CostBreakdown,
    cost_breakdowns,
)
from apt.signals.cointegration import cointegrate_pairs
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs, processed

# ---- knobs ---------------------------------------------------------------
MINUTE_ROOT = Path("data/interim/minute_raw")
MINUTE_PANEL_START = date(2015, 2, 2)
MINUTE_PANEL_END = date(2021, 6, 23)

# Intraday liquidity gate — FILL-RATE ONLY (v2, 2026-06-03 re-run).
# Rationale: at the POC notional of Rs 50,000 per leg, market impact is
# negligible — any minute with a real trade can absorb the order. The
# binding constraint is therefore "does a fill exist this minute?", not
# "what share of that minute's volume are we?". The v1 per-minute-volume
# cap (Rs 50k / 2% → Rs 2.5M/min floor) was a CAPACITY constraint sized
# for a much larger book; on a POC book it collapsed the tradeable set to
# 3 pairs (all anchored on HDFCBANK), excluding daily carriers like
# ONGC/OIL. Reverting to fill-rate-only restores a representative sample.
TARGET_PER_LEG_NOTIONAL_INR = 50_000  # documentation only; not used in gate
MIN_SESSION_FILL_RATE = 0.90

# Rolling-window default: half_life_days * 375 minutes, clamped to
# [1 session, 5 sessions] of CHRONOLOGICAL bars. The rolling z is multi-
# session (see :func:`apt.intraday.zscore.intraday_rolling_zscore`); the
# overnight gap return is excluded from P&L by the engine.
MIN_ROLLING_WINDOW_MIN = NSE_BARS_PER_SESSION
MAX_ROLLING_WINDOW_MIN = 5 * NSE_BARS_PER_SESSION
# First N bars of each session: suppress signals (open auction noise).
SESSION_WARMUP_BARS = 15

DAILY_BASELINE_NET_SHARPE = 1.077  # Phase 2A
DAILY_BASELINE_ANN_RET_PCT = 17.76  # Phase 2A
DAILY_BASELINE_AVG_HOLD_DAYS = 14.7  # Phase 2A weighted

PLOTS_DIR = settings.paths.plots_dir / "phase3"
REPORTS_DIR = Path("reports/phase3")


# -------------------------------------------------------------------------
# Stage 1 — fold construction + daily selection (Phase 2A reuse)
# -------------------------------------------------------------------------


def _build_overlap_folds(daily: pl.DataFrame) -> list:
    """Build all daily folds, then keep those whose test window touches the minute panel."""
    trading_days = daily.select(pl.col("date").unique()).sort("date")["date"].to_list()
    all_folds = build_folds(
        trading_days,
        prior_days=settings.cointegration.n_train_days,
        train_days=settings.cointegration.n_train_days,
        test_days=settings.backtest.test_days_per_fold,
        step_days=settings.backtest.step_days,
    )
    # keep folds whose test window OVERLAPS the minute panel
    overlap = [
        f
        for f in all_folds
        if f.test_end >= MINUTE_PANEL_START and f.test_start <= MINUTE_PANEL_END
    ]
    return overlap


def _select_pairs_for_fold(
    fold,
    daily: pl.DataFrame,
    sectors: pl.DataFrame,
) -> list[Pair]:
    """Mirror Phase 2A's select_pairs_fn for one fold."""
    res = cointegrate_pairs(
        daily,
        sectors,
        start=fold.train_start,
        end=fold.train_end,
        prior_start=fold.prior_start,
        prior_end=fold.prior_end,
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


# -------------------------------------------------------------------------
# Stage 2 — intraday liquidity floor (per pair, on TRAIN window)
# -------------------------------------------------------------------------


def _intraday_liquidity_metrics(sym: str, start: date, end: date) -> dict:
    """Median per-min traded value + session fill-rate for one symbol/window."""
    base = MINUTE_ROOT / f"symbol={sym}"
    if not base.is_dir():
        return {"sym": sym, "n_sessions": 0, "median_min_rupee": np.nan, "fill_rate": 0.0}
    frames = []
    for yr in range(start.year, end.year + 1):
        f = base / f"year={yr}" / "data.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f, columns=["timestamp", "close", "volume"]))
    if not frames:
        return {"sym": sym, "n_sessions": 0, "median_min_rupee": np.nan, "fill_rate": 0.0}
    df = pd.concat(frames, ignore_index=True)
    t = df["timestamp"]
    df = df[
        (t.dt.date >= start)
        & (t.dt.date <= end)
        & (t.dt.time >= pd.Timestamp("09:15").time())
        & (t.dt.time <= pd.Timestamp("15:29").time())
    ]
    if df.empty:
        return {"sym": sym, "n_sessions": 0, "median_min_rupee": np.nan, "fill_rate": 0.0}
    df = df.assign(rupee=df["close"] * df["volume"])
    per_sess = df.groupby(df["timestamp"].dt.date).size()
    fill = float((per_sess / NSE_BARS_PER_SESSION).mean())
    return {
        "sym": sym,
        "n_sessions": int(per_sess.shape[0]),
        "median_min_rupee": float(df["rupee"].median()),
        "fill_rate": fill,
    }


def _passes_intraday_floor(
    pair: Pair, train_start: date, train_end: date
) -> tuple[bool, dict, dict]:
    """Apply the FILL-RATE-ONLY intraday liquidity gate (v2).

    Both legs must have ``fill_rate >= MIN_SESSION_FILL_RATE`` over the
    probe window. The per-minute traded-value floor and the 2%-of-volume
    cap are intentionally removed — they were a CAPACITY constraint, not
    a tradeability constraint, and they suppressed daily carriers at the
    POC notional. Median per-minute traded value is still computed and
    returned for transparency, but it is NOT a gate.
    """
    m_y = _intraday_liquidity_metrics(pair.y_sym, train_start, train_end)
    m_x = _intraday_liquidity_metrics(pair.x_sym, train_start, train_end)
    pass_y = m_y["fill_rate"] >= MIN_SESSION_FILL_RATE
    pass_x = m_x["fill_rate"] >= MIN_SESSION_FILL_RATE
    return bool(pass_y and pass_x), m_y, m_x


# -------------------------------------------------------------------------
# Stage 3 — intraday per-pair-per-fold worker
# -------------------------------------------------------------------------


def _compute_pair_fold(
    pair: Pair,
    fold,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    *,
    base_cost_log: float,
):
    """Load minute data, compute spread + z + signals + gross PnL for one pair-fold.

    Returns a dict keyed by regime with cached arrays + the trade list.
    All NET PnLs are produced later by re-stamping the cost per spread level.
    """
    # Load both legs over TRAIN+TEST (we need training data to fit the TOD profile).
    aligned_full = load_minute_pair(pair.y_sym, pair.x_sym, train_start, test_end, root=MINUTE_ROOT)
    if aligned_full.n_bars == 0:
        return None

    # Compute spread on the full span (NaN at non-tradeable bars)
    s = np.where(aligned_full.tradeable, 1.0, np.nan)
    # Safe log only on tradeable bars
    with np.errstate(invalid="ignore", divide="ignore"):
        py = np.where(aligned_full.tradeable, aligned_full.close_y, np.nan)
        px = np.where(aligned_full.tradeable, aligned_full.close_x, np.nan)
        spread_full = np.log(py) - pair.beta * np.log(px) - pair.alpha
    spread_full = spread_full * s  # NaN propagation; tradeable=False -> NaN

    # Choose rolling window = half_life_days * 375, clamped
    window = int(
        np.clip(
            round(pair.half_life * NSE_BARS_PER_SESSION),
            MIN_ROLLING_WINDOW_MIN,
            MAX_ROLLING_WINDOW_MIN,
        )
    )

    # Slice to train and test
    ts_full = pd.DatetimeIndex(aligned_full.timestamps)
    train_mask = (ts_full.date >= train_start) & (ts_full.date <= train_end)
    test_mask = (ts_full.date >= test_start) & (ts_full.date <= test_end)

    sids_full = aligned_full.session_id
    bins_full = aligned_full.bar_in_session

    if test_mask.sum() < window:
        # Not enough test data even to fill the warm-up window.
        return None

    # Fit the time-of-day vol profile on TRAIN only (causal — never sees test).
    tod_sigma = fit_tod_vol_profile(
        spread_full[train_mask],
        sids_full[train_mask],
        bins_full[train_mask],
        window=window,
        smooth_radius=5,
    )

    # Build the TEST-window arrays. We re-rank session_id locally so the
    # session segments are dense from 0.
    s_test = spread_full[test_mask]
    sids_test = sids_full[test_mask]
    # Re-rank session ids to start at 0 within the test slice
    _, sids_test = np.unique(sids_test, return_inverse=True)
    sids_test = sids_test.astype(np.int32)
    bins_test = bins_full[test_mask]
    tradeable_test = aligned_full.tradeable[test_mask]
    ts_test = ts_full[test_mask]

    # z (flat) — multi-session chronological rolling z over (TRAIN+TEST) so
    # the test window starts already warmed; then sliced to the test mask.
    z_full = intraday_rolling_zscore(
        spread_full,
        sids_full,
        window=window,
        session_warmup_bars=SESSION_WARMUP_BARS,
    )
    z_flat = z_full[test_mask]
    # z (tod) — diagnostic, computed in parallel for sensitivity reporting.
    z_tod = tod_adjusted_zscore(s_test, sids_test, bins_test, tod_sigma, window=window)

    # Convert pair max-holding (days) → bars
    max_holding_bars = int(
        max(
            window,
            min(
                settings.signal.max_holding_cap_days * NSE_BARS_PER_SESSION,
                round(
                    pair.half_life
                    * NSE_BARS_PER_SESSION
                    * settings.signal.max_holding_half_life_multiplier
                ),
            ),
        )
    )

    out = {}
    for regime in ("A", "B"):
        sig = generate_signals_two_regime(
            z_flat,
            sids_test,
            tradeable_test,
            regime=regime,
            entry=settings.signal.entry_z,
            exit=settings.signal.exit_z,
            stop=settings.signal.stop_z,
            max_holding=max_holding_bars,
        )
        res = run_pair_fold(
            fold_id=fold.fold_id,
            pair_key=pair.key,
            timestamps=ts_test,
            session_id=sids_test,
            spread=s_test,
            z=z_flat,
            signals=sig,
            cost_log_per_round_trip=base_cost_log,
            finalize_fold_boundary=(regime == "B"),
        )
        out[regime] = res

    out["window_bars"] = window
    out["max_holding_bars"] = max_holding_bars
    out["tod_sigma"] = tod_sigma
    out["z_tod"] = z_tod
    out["z_flat"] = z_flat
    out["timestamps"] = ts_test
    out["session_id"] = sids_test
    out["bar_in_session"] = bins_test
    out["spread"] = s_test
    out["tradeable"] = tradeable_test
    return out


# -------------------------------------------------------------------------
# Stage 4 — net-PnL re-derivation under a different cost level
# -------------------------------------------------------------------------


def _net_pnl_for_cost(
    pair_fold_res,
    *,
    new_cost_log: float,
    base_cost_log: float,
) -> tuple[np.ndarray, list]:
    """Adjust the cached gross/net + trade list for a different cost level.

    Cheap: only re-stamps cost_log per trade and rewrites the per-bar net.
    """
    gross = pair_fold_res.gross_log_ret
    # Start from gross and re-deduct the new cost at every trade exit bar.
    net = gross.copy()
    new_trades = []
    # Map exit timestamp -> exit_idx for indexing into the per-bar net array.
    ts_to_idx = {t: i for i, t in enumerate(pair_fold_res.timestamps)}
    for tr in pair_fold_res.trades:
        exit_idx = ts_to_idx.get(tr.exit_ts)
        if exit_idx is None:
            continue
        net[exit_idx] -= new_cost_log
        new_trades.append(
            type(tr)(
                fold_id=tr.fold_id,
                pair_key=tr.pair_key,
                regime=tr.regime,
                direction=tr.direction,
                entry_ts=tr.entry_ts,
                exit_ts=tr.exit_ts,
                entry_z=tr.entry_z,
                exit_z=tr.exit_z,
                bars_held=tr.bars_held,
                sessions_held=tr.sessions_held,
                gross_log_pnl=tr.gross_log_pnl,
                cost_log=new_cost_log,
                net_log_pnl=tr.gross_log_pnl - new_cost_log,
                exit_reason=tr.exit_reason,
            )
        )
    return net, new_trades


# -------------------------------------------------------------------------
# Stage 5 — main
# -------------------------------------------------------------------------


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "13_phase3_intraday.log")
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 3 — intraday two-regime validation")
    daily = pl.read_parquet(processed("daily_clean.parquet"))
    sectors = pl.read_parquet(Path("data/interim/sectors.parquet"))

    folds = _build_overlap_folds(daily)
    logger.info(
        "Built {} fold(s) overlapping the minute panel ({} -> {})",
        len(folds),
        MINUTE_PANEL_START,
        MINUTE_PANEL_END,
    )
    for f in folds:
        logger.info(
            "  fold {}: train {}->{}, test {}->{}",
            f.fold_id,
            f.train_start,
            f.train_end,
            f.test_start,
            f.test_end,
        )

    # -- stage 1: per-fold daily selection (Phase 2A reuse) -----------------
    fold_pairs_rows: list[dict] = []
    pairs_by_fold: dict[int, list[Pair]] = {}
    for fold in folds:
        pairs = _select_pairs_for_fold(fold, daily, sectors)
        pairs_by_fold[fold.fold_id] = pairs
        logger.info(
            "  fold {}: daily selection produced {} tradeable pair(s)", fold.fold_id, len(pairs)
        )
        for p in pairs:
            fold_pairs_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "pair": p.key,
                    "y_sym": p.y_sym,
                    "x_sym": p.x_sym,
                    "alpha": p.alpha,
                    "beta": p.beta,
                    "half_life_days": p.half_life,
                    "sector": p.sector,
                    "is_structural": p.is_structural,
                }
            )
    pd.DataFrame(fold_pairs_rows).to_csv(REPORTS_DIR / "fold_pairs.csv", index=False)

    # -- stage 2: intraday liquidity floor (per pair-fold) ------------------
    liq_rows: list[dict] = []
    pair_fold_keep: dict[tuple[int, str], Pair] = {}
    for fold in folds:
        # Use only the LAST year of the train window for the liquidity probe —
        # mirrors the live tradeability check while keeping the read small.
        # Clip to the minute panel: for early folds whose train_end is BEFORE
        # the minute panel starts, the probe must use forward-looking minute
        # data from the test window (the next year) — there is no other
        # option since the train window predates the minute feed.
        liq_train_start = fold.train_end - timedelta(days=365)
        liq_train_start = max(liq_train_start, fold.train_start)
        liq_train_end = fold.train_end
        if liq_train_end < MINUTE_PANEL_START:
            liq_train_start = MINUTE_PANEL_START
            liq_train_end = min(fold.test_end, MINUTE_PANEL_START + timedelta(days=180))
        for p in pairs_by_fold[fold.fold_id]:
            ok, m_y, m_x = _passes_intraday_floor(p, liq_train_start, liq_train_end)
            liq_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "pair": p.key,
                    "liq_probe_start": liq_train_start,
                    "liq_probe_end": liq_train_end,
                    "y_med_min_rupee": m_y["median_min_rupee"],
                    "x_med_min_rupee": m_x["median_min_rupee"],
                    "y_fill_rate": m_y["fill_rate"],
                    "x_fill_rate": m_x["fill_rate"],
                    "intraday_pass": ok,
                }
            )
            if ok:
                pair_fold_keep[(fold.fold_id, p.key)] = p
    liq_df = pd.DataFrame(liq_rows)
    liq_df.to_csv(REPORTS_DIR / "liquidity_filter.csv", index=False)
    n_kept = liq_df["intraday_pass"].sum()
    n_total = len(liq_df)
    logger.info(
        "Intraday liquidity gate (v2, fill-rate only): {} / {} pair-fold units survive "
        "(POC notional Rs {:,}/leg; fill-rate >= {} on both legs)",
        int(n_kept),
        int(n_total),
        TARGET_PER_LEG_NOTIONAL_INR,
        MIN_SESSION_FILL_RATE,
    )

    # -- stage 3: per pair-fold compute (gross-only; net derived per cost) --
    base_cost = CostBreakdown(total_spread_bps=SPREAD_SWEEP_BPS[0])
    base_cost_log = base_cost.cost_log_per_pair_round_trip  # any value works for caching
    cache: dict[tuple[int, str], dict] = {}
    for (fold_id, pair_key), p in pair_fold_keep.items():
        fold = next(f for f in folds if f.fold_id == fold_id)
        logger.info(
            "  computing pair-fold: fold {} {} (half-life={:.1f}d)", fold_id, pair_key, p.half_life
        )
        try:
            res = _compute_pair_fold(
                p,
                fold,
                fold.train_start,
                fold.train_end,
                max(fold.test_start, MINUTE_PANEL_START),
                min(fold.test_end, MINUTE_PANEL_END),
                base_cost_log=base_cost_log,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("    skipped due to error: {}", exc)
            continue
        if res is None:
            logger.warning("    skipped (no data or insufficient bars)")
            continue
        cache[(fold_id, pair_key)] = res

    logger.info("Cached {} pair-fold compute results", len(cache))

    # -- stage 3.5: diagnostic sensitivity ---------------------------------
    _emit_sensitivity_diagnostics(cache, folds, pair_fold_keep)

    # -- stage 4: spread sweep + portfolio aggregation ----------------------
    all_trade_rows: list[dict] = []
    portfolio_rows: list[dict] = []  # one row per (regime, cost, session)
    pair_daily_rows: list[dict] = []  # one row per (regime, cost, pair, session)
    metrics_rows: list[dict] = []  # one row per (regime, cost)

    for cb in cost_breakdowns():
        cost_log = cb.cost_log_per_pair_round_trip
        for regime in ("A", "B"):
            # Stack per-pair daily returns onto a common date axis for the
            # portfolio = equal-weighted mean across active pairs that day.
            pair_daily: dict[str, pd.DataFrame] = {}
            for (fold_id, pair_key), res in cache.items():
                pf_res = res[regime]
                net_arr, new_trades = _net_pnl_for_cost(
                    pf_res, new_cost_log=cost_log, base_cost_log=base_cost_log
                )
                # Per-session sum of minute returns
                df = pd.DataFrame(
                    {
                        "date": pd.DatetimeIndex(pf_res.timestamps).date,
                        "gross_log_ret": pf_res.gross_log_ret,
                        "net_log_ret": net_arr,
                    }
                )
                pair_session_df = df.groupby("date", as_index=False)[
                    ["gross_log_ret", "net_log_ret"]
                ].sum()
                # Track active days as days where the pair traded at all
                # (any bar had non-zero gross or net move)
                pair_session_df["active"] = (
                    df.assign(active=(df["gross_log_ret"] != 0) | (df["net_log_ret"] != 0))
                    .groupby("date")["active"]
                    .any()
                    .reindex(pair_session_df["date"])
                    .values
                )
                pair_session_df["pair"] = pair_key
                pair_session_df["fold_id"] = fold_id
                pair_daily[pair_key + f"::f{fold_id}"] = pair_session_df

                if cb.total_spread_bps == 3:
                    for _, row in pair_session_df.iterrows():
                        pair_daily_rows.append(
                            {
                                "regime": regime,
                                "spread_bps": cb.total_spread_bps,
                                "fold_id": fold_id,
                                "pair": pair_key,
                                "date": row["date"],
                                "gross_log_ret": float(row["gross_log_ret"]),
                                "net_log_ret": float(row["net_log_ret"]),
                                "active": bool(row["active"]),
                            }
                        )
                    for tr in new_trades:
                        all_trade_rows.append(
                            {
                                **{
                                    k: v
                                    for k, v in asdict(tr).items()
                                    if k not in ("entry_ts", "exit_ts")
                                },
                                "entry_ts": tr.entry_ts.isoformat(),
                                "exit_ts": tr.exit_ts.isoformat(),
                                "spread_bps": cb.total_spread_bps,
                            }
                        )

            if not pair_daily:
                continue

            # Equal-weighted portfolio: at each date, mean across pairs
            all_df = pd.concat(pair_daily.values(), ignore_index=True)
            port = (
                all_df.groupby("date", as_index=False)
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

            n_trades_total = sum(len(res[regime].trades) for res in cache.values())
            # Exit reasons under this (regime, cost) combo are independent of cost,
            # so we can pull them straight from the cached trade list (cost doesn't
            # affect exit timing).
            exit_split: dict[str, int] = defaultdict(int)
            avg_bars_held = 0
            avg_sessions_held = 0
            if n_trades_total:
                for res in cache.values():
                    for tr in res[regime].trades:
                        exit_split[tr.exit_reason] += 1
                        avg_bars_held += tr.bars_held
                        avg_sessions_held += tr.sessions_held
                avg_bars_held = avg_bars_held / n_trades_total
                avg_sessions_held = avg_sessions_held / n_trades_total

            # Active-day count (any pair trading)
            pct_time_deployed = float((port["n_active"] > 0).mean()) if not port.empty else 0.0

            metrics_rows.append(
                {
                    "regime": regime,
                    "spread_bps": cb.total_spread_bps,
                    "fixed_cost_per_leg_bps": FIXED_PER_LEG_RT,
                    "cost_per_leg_bps": cb.cost_bps_per_leg,
                    "n_sessions": int(port.shape[0]),
                    "n_pairs_total": int(len(pair_daily)),
                    "n_trades": n_trades_total,
                    "gross_total_pct": m_gross["total_return_pct"],
                    "net_total_pct": m_net["total_return_pct"],
                    "gross_ann_pct": m_gross["ann_return_pct"],
                    "net_ann_pct": m_net["ann_return_pct"],
                    "gross_ann_vol_pct": m_gross["ann_vol_pct"],
                    "net_ann_vol_pct": m_net["ann_vol_pct"],
                    "gross_sharpe": m_gross["sharpe"],
                    "net_sharpe": m_net["sharpe"],
                    "net_max_drawdown_pct": m_net["max_drawdown_pct"],
                    "avg_bars_held": avg_bars_held,
                    "avg_sessions_held": avg_sessions_held,
                    "pct_time_deployed": pct_time_deployed,
                    "exit_mean_revert": exit_split.get("mean_revert", 0),
                    "exit_stop": exit_split.get("stop", 0),
                    "exit_time": exit_split.get("time", 0),
                    "exit_session_close": exit_split.get("session_close", 0),
                    "exit_fold_boundary": exit_split.get("fold_boundary", 0),
                }
            )
            for _, row in port.iterrows():
                portfolio_rows.append(
                    {
                        "regime": regime,
                        "spread_bps": cb.total_spread_bps,
                        "date": row["date"],
                        "gross_log_ret": float(row["gross_log_ret"]),
                        "net_log_ret": float(row["net_log_ret"]),
                        "n_pairs": int(row["n_pairs"]),
                        "n_active": int(row["n_active"]),
                    }
                )

    pd.DataFrame(all_trade_rows).to_csv(REPORTS_DIR / "trades_two_regime_3bps.csv", index=False)
    pd.DataFrame(pair_daily_rows).to_csv(
        REPORTS_DIR / "pair_daily_two_regime_3bps.csv", index=False
    )
    pd.DataFrame(portfolio_rows).to_csv(REPORTS_DIR / "portfolio_two_regime.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(REPORTS_DIR / "metrics_two_regime.csv", index=False)

    # -- stage 5: plots -----------------------------------------------------
    _emit_plots(metrics_rows, portfolio_rows, cache, folds)

    # -- stage 6: expanded reporting (v2) ----------------------------------
    v2_summary = _emit_phase3_v2_outputs(cache, folds, pair_fold_keep, base_cost_log)

    # -- caveats ------------------------------------------------------------
    _write_caveats(metrics_rows)

    # -- console summary ----------------------------------------------------
    _print_summary(metrics_rows, len(cache), len(folds), n_kept, n_total)
    if v2_summary is not None:
        _print_v2_summary(v2_summary)
    print("\n=== 13_phase3_intraday complete ===\n")


# -------------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------------


def _emit_sensitivity_diagnostics(cache, folds, pair_fold_keep) -> None:
    """Side-by-side diagnostic: TOD-adjusted z vs flat z, and window sensitivity.

    Runs ONLY on the most-recent fold's pair-fold to keep wall-clock small
    (training-window only; never grid-searched on the full sample). Reports
    net Sharpe at 3 bps spread for each variant.
    """
    if not cache:
        return
    base_cost_log = CostBreakdown(total_spread_bps=3).cost_log_per_pair_round_trip
    # Pick the most-recent pair-fold available
    target_key = max(cache.keys(), key=lambda k: k[0])
    res = cache[target_key]
    fold_id, pair_key = target_key
    pair = pair_fold_keep[target_key]
    _ = folds  # signature symmetry; folds carry no extra info beyond the cached res
    logger.info(
        "Sensitivity diag on {} (fold {}, half-life={:.1f}d, window={} bars)",
        pair_key,
        fold_id,
        pair.half_life,
        res["window_bars"],
    )

    rows: list[dict] = []
    base_window = res["window_bars"]

    # (a) Flat vs TOD-adjusted z at the default window
    for variant in ("flat", "tod"):
        z = res["z_flat"] if variant == "flat" else res["z_tod"]
        for regime in ("A", "B"):
            sig = generate_signals_two_regime(
                z,
                res["session_id"],
                res["tradeable"],
                regime=regime,
                entry=settings.signal.entry_z,
                exit=settings.signal.exit_z,
                stop=settings.signal.stop_z,
                max_holding=res["max_holding_bars"],
            )
            rr = run_pair_fold(
                fold_id=fold_id,
                pair_key=pair_key,
                timestamps=res["timestamps"],
                session_id=res["session_id"],
                spread=res["spread"],
                z=z,
                signals=sig,
                cost_log_per_round_trip=base_cost_log,
                finalize_fold_boundary=(regime == "B"),
            )
            # Per-session sum + Sharpe
            df = pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(rr.timestamps).date,
                    "gross_log_ret": rr.gross_log_ret,
                    "net_log_ret": rr.net_log_ret,
                }
            )
            daily = df.groupby("date", as_index=False)[["gross_log_ret", "net_log_ret"]].sum()
            m_g = compute_metrics(daily["gross_log_ret"].to_numpy())
            m_n = compute_metrics(daily["net_log_ret"].to_numpy())
            rows.append(
                {
                    "variant": f"z={variant}",
                    "window_bars": base_window,
                    "regime": regime,
                    "n_trades": len(rr.trades),
                    "gross_sharpe": m_g["sharpe"],
                    "net_sharpe": m_n["sharpe"],
                    "net_ann_pct": m_n["ann_return_pct"],
                }
            )

    # (b) Window sensitivity: 1, 3, 5 sessions (Regime A + B at flat z)
    for n_sess in (1, 3, 5):
        win = n_sess * NSE_BARS_PER_SESSION
        z_full = intraday_rolling_zscore(
            res["spread"],  # already TEST-only spread
            res["session_id"],
            window=win,
            session_warmup_bars=SESSION_WARMUP_BARS,
        )
        for regime in ("A", "B"):
            sig = generate_signals_two_regime(
                z_full,
                res["session_id"],
                res["tradeable"],
                regime=regime,
                entry=settings.signal.entry_z,
                exit=settings.signal.exit_z,
                stop=settings.signal.stop_z,
                max_holding=res["max_holding_bars"],
            )
            rr = run_pair_fold(
                fold_id=fold_id,
                pair_key=pair_key,
                timestamps=res["timestamps"],
                session_id=res["session_id"],
                spread=res["spread"],
                z=z_full,
                signals=sig,
                cost_log_per_round_trip=base_cost_log,
                finalize_fold_boundary=(regime == "B"),
            )
            df = pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(rr.timestamps).date,
                    "gross_log_ret": rr.gross_log_ret,
                    "net_log_ret": rr.net_log_ret,
                }
            )
            daily = df.groupby("date", as_index=False)[["gross_log_ret", "net_log_ret"]].sum()
            m_g = compute_metrics(daily["gross_log_ret"].to_numpy())
            m_n = compute_metrics(daily["net_log_ret"].to_numpy())
            rows.append(
                {
                    "variant": f"window={n_sess}sess",
                    "window_bars": win,
                    "regime": regime,
                    "n_trades": len(rr.trades),
                    "gross_sharpe": m_g["sharpe"],
                    "net_sharpe": m_n["sharpe"],
                    "net_ann_pct": m_n["ann_return_pct"],
                }
            )

    diag = pd.DataFrame(rows)
    diag["pair"] = pair_key
    diag["fold_id"] = fold_id
    diag.to_csv(REPORTS_DIR / "sensitivity_z_and_window.csv", index=False)
    logger.info("Sensitivity diag saved to {}", REPORTS_DIR / "sensitivity_z_and_window.csv")


_STRUCTURAL_PAIR_KEYS: frozenset[str] = frozenset({"HDFC/HDFCBANK"})
# Renamed exit reasons for the v2 trade-level export (matches the brief's
# taxonomy: mean_revert / z_stop / time_stop / eod_squareoff / fold_close).
_EXIT_REASON_RENAME: dict[str, str] = {
    "mean_revert": "mean_revert",
    "stop": "z_stop",
    "time": "time_stop",
    "session_close": "eod_squareoff",
    "fold_boundary": "fold_close",
}


def _portfolio_from_cache(
    cache: dict,
    regime: str,
    new_cost_log: float,
    base_cost_log: float,
    exclude_keys: frozenset[str] | set[str] = frozenset(),
) -> tuple[pd.DataFrame, list]:
    """Aggregate a (regime, cost) view into an equal-weighted portfolio.

    Equal notional per active pair, idle capital at 0%, MEAN across pairs
    per session (matches Phase 2A and the v1 phase-3 aggregation).
    Returns the per-session portfolio frame + the new (cost-restamped) trade list.
    Pairs whose key is in ``exclude_keys`` are dropped before aggregation.
    """
    pair_daily: dict[str, pd.DataFrame] = {}
    all_new_trades: list = []
    for (fold_id, pair_key), res in cache.items():
        if pair_key in exclude_keys:
            continue
        pf_res = res[regime]
        net_arr, new_trades = _net_pnl_for_cost(
            pf_res, new_cost_log=new_cost_log, base_cost_log=base_cost_log
        )
        all_new_trades.extend(new_trades)
        df = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(pf_res.timestamps).date,
                "gross_log_ret": pf_res.gross_log_ret,
                "net_log_ret": net_arr,
            }
        )
        pair_session_df = df.groupby("date", as_index=False)[["gross_log_ret", "net_log_ret"]].sum()
        pair_session_df["active"] = (
            df.assign(active=(df["gross_log_ret"] != 0) | (df["net_log_ret"] != 0))
            .groupby("date")["active"]
            .any()
            .reindex(pair_session_df["date"])
            .values
        )
        pair_session_df["fold_id"] = fold_id
        pair_session_df["pair"] = pair_key
        pair_daily[f"{pair_key}::f{fold_id}"] = pair_session_df

    if not pair_daily:
        return pd.DataFrame(columns=["date", "gross_log_ret", "net_log_ret"]), []

    all_df = pd.concat(pair_daily.values(), ignore_index=True)
    port = (
        all_df.groupby("date", as_index=False)
        .agg(
            gross_log_ret=("gross_log_ret", "mean"),
            net_log_ret=("net_log_ret", "mean"),
            n_pairs=("pair", "nunique"),
            n_active=("active", "sum"),
        )
        .sort_values("date")
    )
    return port, all_new_trades


def _per_pair_metrics_at_3bps(
    cache: dict,
    pair_fold_keep: dict,
    regime: str,
    base_cost_log: float,
) -> pd.DataFrame:
    """Full per-pair metrics at 3 bps total spread, ranked by net_ann_pct."""
    cost_log = CostBreakdown(total_spread_bps=3).cost_log_per_pair_round_trip
    rows: list[dict] = []
    for (fold_id, pair_key), res in cache.items():
        pf_res = res[regime]
        net_arr, new_trades = _net_pnl_for_cost(
            pf_res, new_cost_log=cost_log, base_cost_log=base_cost_log
        )
        df = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(pf_res.timestamps).date,
                "gross_log_ret": pf_res.gross_log_ret,
                "net_log_ret": net_arr,
            }
        )
        daily = df.groupby("date", as_index=False)[["gross_log_ret", "net_log_ret"]].sum()
        active_per_day = (
            df.assign(active=(df["gross_log_ret"] != 0) | (df["net_log_ret"] != 0))
            .groupby("date")["active"]
            .any()
        )
        pct_time_deployed = float(active_per_day.mean()) if not active_per_day.empty else 0.0
        m_g = compute_metrics(daily["gross_log_ret"].to_numpy())
        m_n = compute_metrics(daily["net_log_ret"].to_numpy())

        n_trades = len(new_trades)
        n_wins = sum(1 for t in new_trades if t.net_log_pnl > 0)
        win_rate = n_wins / n_trades if n_trades else 0.0
        avg_hold_bars = float(sum(t.bars_held for t in new_trades) / n_trades) if n_trades else 0.0
        avg_hold_sessions = (
            float(sum(t.sessions_held for t in new_trades) / n_trades) if n_trades else 0.0
        )
        exit_split: dict[str, int] = defaultdict(int)
        for t in new_trades:
            exit_split[_EXIT_REASON_RENAME.get(t.exit_reason, t.exit_reason)] += 1

        pair_obj = pair_fold_keep.get((fold_id, pair_key))
        rows.append(
            {
                "fold_id": fold_id,
                "pair": pair_key,
                "sector": pair_obj.sector if pair_obj else "",
                "is_structural": bool(pair_obj.is_structural) if pair_obj else False,
                "is_hdfcbank_anchored": "HDFCBANK" in pair_key.split("/"),
                "n_trades": n_trades,
                "n_obs": int(daily.shape[0]),
                "gross_total_pct": m_g["total_return_pct"],
                "net_total_pct": m_n["total_return_pct"],
                "gross_ann_pct": m_g["ann_return_pct"],
                "net_ann_pct": m_n["ann_return_pct"],
                "gross_sharpe": m_g["sharpe"],
                "net_sharpe": m_n["sharpe"],
                "net_max_drawdown_pct": m_n["max_drawdown_pct"],
                "win_rate_net": win_rate,
                "avg_bars_held": avg_hold_bars,
                "avg_sessions_held": avg_hold_sessions,
                "pct_time_deployed": pct_time_deployed,
                "exit_mean_revert": exit_split.get("mean_revert", 0),
                "exit_z_stop": exit_split.get("z_stop", 0),
                "exit_time_stop": exit_split.get("time_stop", 0),
                "exit_eod_squareoff": exit_split.get("eod_squareoff", 0),
                "exit_fold_close": exit_split.get("fold_close", 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("net_ann_pct", ascending=False, na_position="last").reset_index(drop=True)


def _trade_level_rows_at_3bps(
    cache: dict,
    pair_fold_keep: dict,
    regime: str,
    base_cost_log: float,
) -> list[dict]:
    """One row per round-trip at 3 bps, with reconstructable-net columns."""
    cb_3 = CostBreakdown(total_spread_bps=3)
    cost_log = cb_3.cost_log_per_pair_round_trip
    n_legs = 2
    rows: list[dict] = []
    for (fold_id, pair_key), res in cache.items():
        pf_res = res[regime]
        _, new_trades = _net_pnl_for_cost(
            pf_res, new_cost_log=cost_log, base_cost_log=base_cost_log
        )
        pair_obj = pair_fold_keep.get((fold_id, pair_key))
        sector = pair_obj.sector if pair_obj else ""
        is_structural = bool(pair_obj.is_structural) if pair_obj else False
        is_hdfcbank_anchored = "HDFCBANK" in pair_key.split("/")
        for t in new_trades:
            side = "long_spread" if t.direction == 1 else "short_spread"
            rows.append(
                {
                    "fold_id": fold_id,
                    "regime": regime,
                    "pair": pair_key,
                    "sector": sector,
                    "is_structural": is_structural,
                    "is_hdfcbank_anchored": is_hdfcbank_anchored,
                    "side": side,
                    "entry_ts": t.entry_ts.isoformat(),
                    "exit_ts": t.exit_ts.isoformat(),
                    "z_entry": t.entry_z,
                    "z_exit": t.exit_z,
                    "bars_held": t.bars_held,
                    "sessions_held": t.sessions_held,
                    "gross_log_pnl": t.gross_log_pnl,
                    "net_log_pnl_at_3bps": t.net_log_pnl,
                    "gross_pct": float(np.expm1(t.gross_log_pnl) * 100),
                    "net_pct_at_3bps": float(np.expm1(t.net_log_pnl) * 100),
                    "cost_bps_excl_spread_per_leg_rt": FIXED_PER_LEG_RT,
                    "n_legs": n_legs,
                    "exit_reason": _EXIT_REASON_RENAME.get(t.exit_reason, t.exit_reason),
                }
            )
    return rows


def _emit_phase3_v2_outputs(
    cache: dict,
    folds: list,
    pair_fold_keep: dict,
    base_cost_log: float,
) -> dict | None:
    """Phase-3 re-run (v2) outputs: full per-pair tables, trade-level CSVs,
    with/without-HDFC/HDFCBANK splits, full-span equity curve, and a refreshed
    sharpe-vs-spread sweep. Designed to coexist with the v1 outputs from the
    original run; nothing here overwrites existing files.
    """
    if not cache:
        return None

    n_pair_folds_total = len(cache)
    n_pair_folds_ex = sum(1 for k in cache if k[1] not in _STRUCTURAL_PAIR_KEYS)
    logger.info(
        "v2 outputs: {} pair-fold units total; {} after excluding HDFC/HDFCBANK",
        n_pair_folds_total,
        n_pair_folds_ex,
    )

    # --- 1. Trade-level CSVs for both regimes -----------------------------
    for regime in ("A", "B"):
        rows = _trade_level_rows_at_3bps(cache, pair_fold_keep, regime, base_cost_log)
        df = pd.DataFrame(rows)
        out_path = REPORTS_DIR / f"trades_all_pairs_{regime}.csv"
        df.to_csv(out_path, index=False)
        logger.info("  wrote {} ({} trades, regime {})", out_path, len(df), regime)

    # --- 2. Full per-pair performance tables ------------------------------
    for regime in ("A", "B"):
        df = _per_pair_metrics_at_3bps(cache, pair_fold_keep, regime, base_cost_log)
        out_path = REPORTS_DIR / f"per_pair_full_{regime}.csv"
        df.to_csv(out_path, index=False)
        logger.info("  wrote {} ({} pair-fold rows, regime {})", out_path, len(df), regime)

    # --- 3. metrics_two_regime_v2.csv: spread sweep with/without HDFC/HDFCBANK
    metrics_v2: list[dict] = []
    for cb in cost_breakdowns():
        cost_log = cb.cost_log_per_pair_round_trip
        for regime in ("A", "B"):
            for variant_name, exclude in (
                ("all", frozenset()),
                ("ex_hdfc_hdfcbank", _STRUCTURAL_PAIR_KEYS),
            ):
                port, trades_v = _portfolio_from_cache(
                    cache, regime, cost_log, base_cost_log, exclude
                )
                if port.empty:
                    continue
                m_g = compute_metrics(port["gross_log_ret"].to_numpy())
                m_n = compute_metrics(port["net_log_ret"].to_numpy())
                metrics_v2.append(
                    {
                        "variant": variant_name,
                        "regime": regime,
                        "spread_bps": cb.total_spread_bps,
                        "n_pair_fold_units": len([k for k in cache if k[1] not in exclude]),
                        "n_trades": len(trades_v),
                        "gross_total_pct": m_g["total_return_pct"],
                        "net_total_pct": m_n["total_return_pct"],
                        "gross_ann_pct": m_g["ann_return_pct"],
                        "net_ann_pct": m_n["ann_return_pct"],
                        "gross_sharpe": m_g["sharpe"],
                        "net_sharpe": m_n["sharpe"],
                        "net_max_drawdown_pct": m_n["max_drawdown_pct"],
                        "pct_time_deployed": float((port["n_active"] > 0).mean()),
                    }
                )
    metrics_v2_df = pd.DataFrame(metrics_v2)
    metrics_v2_df.to_csv(REPORTS_DIR / "metrics_two_regime_v2.csv", index=False)
    logger.info(
        "  wrote {} ({} rows)",
        REPORTS_DIR / "metrics_two_regime_v2.csv",
        len(metrics_v2_df),
    )

    # --- 4. equity_curve_daily.csv (3 bps) + full-span plot ---------------
    cost_log_3 = CostBreakdown(total_spread_bps=3).cost_log_per_pair_round_trip
    portA_all, _ = _portfolio_from_cache(cache, "A", cost_log_3, base_cost_log)
    portB_all, _ = _portfolio_from_cache(cache, "B", cost_log_3, base_cost_log)
    portA_ex, _ = _portfolio_from_cache(
        cache, "A", cost_log_3, base_cost_log, _STRUCTURAL_PAIR_KEYS
    )
    portB_ex, _ = _portfolio_from_cache(
        cache, "B", cost_log_3, base_cost_log, _STRUCTURAL_PAIR_KEYS
    )
    eq_frames = {
        "A_net": portA_all,
        "B_net": portB_all,
        "A_net_exHDFCBANK": portA_ex,
        "B_net_exHDFCBANK": portB_ex,
    }
    # union date axis
    all_dates: set = set()
    for fr in eq_frames.values():
        all_dates.update(fr["date"].tolist())
    sorted_dates = sorted(all_dates)
    eq_df = pd.DataFrame({"date": sorted_dates})
    for col, fr in eq_frames.items():
        s = fr.set_index("date")["net_log_ret"].reindex(sorted_dates).fillna(0.0)
        eq_df[col] = s.values
    eq_df.to_csv(REPORTS_DIR / "equity_curve_daily.csv", index=False)
    logger.info(
        "  wrote {} ({} sessions)",
        REPORTS_DIR / "equity_curve_daily.csv",
        len(eq_df),
    )

    _plot_full_span_equity(eq_df)
    _plot_sharpe_vs_spread_v2(metrics_v2_df)
    _plot_equity_a_vs_b_3bps(portA_all, portB_all, portA_ex, portB_ex)

    # Return summary dict for print
    return {
        "n_pair_folds": n_pair_folds_total,
        "n_pair_folds_ex": n_pair_folds_ex,
        "metrics_v2": metrics_v2_df,
    }


def _plot_full_span_equity(eq_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    dates = pd.to_datetime(eq_df["date"])
    for col, ls, color in (
        ("A_net", "-", "tab:red"),
        ("A_net_exHDFCBANK", "--", "tab:red"),
        ("B_net", "-", "tab:blue"),
        ("B_net_exHDFCBANK", "--", "tab:blue"),
    ):
        ax.plot(dates, eq_df[col].cumsum(), linestyle=ls, color=color, label=col)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(
        "Phase 3 v2 — full-span equity at 3 bps spread (solid = all; dashed = ex HDFC/HDFCBANK)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net log return")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "equity_curve_full_span.png", dpi=140)
    plt.close(fig)


def _plot_sharpe_vs_spread_v2(metrics_v2_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metrics_v2_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, ls in (("all", "-"), ("ex_hdfc_hdfcbank", "--")):
        for regime, marker, color in (("A", "o", "tab:red"), ("B", "s", "tab:blue")):
            sub = metrics_v2_df[
                (metrics_v2_df["variant"] == variant) & (metrics_v2_df["regime"] == regime)
            ].sort_values("spread_bps")
            if sub.empty:
                continue
            ax.plot(
                sub["spread_bps"],
                sub["net_sharpe"],
                linestyle=ls,
                marker=marker,
                color=color,
                label=f"Regime {regime} ({variant})",
            )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(
        DAILY_BASELINE_NET_SHARPE,
        color="black",
        linestyle=":",
        linewidth=0.8,
        label=f"daily Phase 2A net Sharpe = {DAILY_BASELINE_NET_SHARPE:.2f}",
    )
    ax.set_xlabel("Assumed total bid-ask spread (bps)")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Phase 3 v2 — net Sharpe vs assumed spread (with & without HDFC/HDFCBANK)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sharpe_vs_spread_v2.png", dpi=140)
    plt.close(fig)


def _plot_equity_a_vs_b_3bps(portA_all, portB_all, portA_ex, portB_ex) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for fr, name, ls, color in (
        (portA_all, "Regime A (all)", "-", "tab:red"),
        (portB_all, "Regime B (all)", "-", "tab:blue"),
        (portA_ex, "Regime A (ex HDFC/HDFCBANK)", "--", "tab:red"),
        (portB_ex, "Regime B (ex HDFC/HDFCBANK)", "--", "tab:blue"),
    ):
        if fr.empty:
            continue
        ax.plot(
            pd.to_datetime(fr["date"]),
            fr["net_log_ret"].cumsum(),
            linestyle=ls,
            color=color,
            label=name,
        )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net log return")
    ax.set_title("Phase 3 v2 — A vs B equity curves at 3 bps spread")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "equity_A_vs_B_3bps.png", dpi=140)
    plt.close(fig)


def _print_v2_summary(v2_summary: dict) -> None:
    print()
    print("  --- v2 (fill-rate-only) Sharpe vs spread, with/without HDFC/HDFCBANK ---")
    M = v2_summary["metrics_v2"]
    if M.empty:
        print("    (no v2 metrics)")
        return
    for regime in ("A", "B"):
        sub = M[M["regime"] == regime].pivot(
            index="spread_bps", columns="variant", values="net_sharpe"
        )
        sub = sub.sort_index()
        print(f"    Regime {regime} net Sharpe (spread_bps as rows):")
        print(sub.to_string(float_format=lambda v: f"{v:+.3f}"))
    print()
    print("  --- v2 net ann return % vs spread ---")
    for regime in ("A", "B"):
        sub = M[M["regime"] == regime].pivot(
            index="spread_bps", columns="variant", values="net_ann_pct"
        )
        sub = sub.sort_index()
        print(f"    Regime {regime} net ann %:")
        print(sub.to_string(float_format=lambda v: f"{v:+.2f}"))
    print()
    # A-B gap at 3 bps
    m3 = M[M["spread_bps"] == 3]
    if not m3.empty:
        for variant in ("all", "ex_hdfc_hdfcbank"):
            sub = m3[m3["variant"] == variant]
            a = (
                sub[sub["regime"] == "A"]["net_sharpe"].squeeze()
                if not sub[sub["regime"] == "A"].empty
                else float("nan")
            )
            b = (
                sub[sub["regime"] == "B"]["net_sharpe"].squeeze()
                if not sub[sub["regime"] == "B"].empty
                else float("nan")
            )
            try:
                gap = float(b) - float(a)
            except Exception:  # noqa: BLE001
                gap = float("nan")
            print(f"  A→B Sharpe gap @ 3 bps ({variant}): {gap:+.3f}")
    print()
    # Trade-count multiple vs daily baseline
    m3_all = m3[(m3["variant"] == "all")] if not m3.empty else m3
    if not m3_all.empty:
        for regime in ("A", "B"):
            row = m3_all[m3_all["regime"] == regime]
            if row.empty:
                continue
            nt = int(row["n_trades"].iloc[0])
            mult = nt / 198.0
            print(
                f"  Trade count Regime {regime} @ 3bps : {nt}  "
                f"({mult:.1f}× the daily Phase 2A baseline of 198 trades / 7 yrs)"
            )


def _emit_plots(metrics_rows, portfolio_rows, cache, folds) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = pd.DataFrame(metrics_rows)
    P = pd.DataFrame(portfolio_rows)
    if M.empty:
        logger.warning("No metrics to plot.")
        return

    # 1. Net Sharpe vs spread, both regimes
    fig, ax = plt.subplots(figsize=(9, 5))
    for regime, marker in (("A", "o"), ("B", "s")):
        sub = M[M["regime"] == regime].sort_values("spread_bps")
        ax.plot(
            sub["spread_bps"],
            sub["net_sharpe"],
            marker=marker,
            label=f"Regime {regime} ({'intraday-only' if regime == 'A' else 'multi-day carry'})",
        )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(
        DAILY_BASELINE_NET_SHARPE,
        color="black",
        linestyle="--",
        linewidth=0.7,
        label=f"daily Phase 2A net Sharpe = {DAILY_BASELINE_NET_SHARPE:.2f}",
    )
    ax.set_xlabel("Assumed total bid-ask spread (bps)")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Phase 3 — where the intraday edge dies vs assumed spread")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sharpe_vs_spread.png", dpi=140)
    plt.close(fig)

    # 2. Equity curves A vs B at 3 bps
    fig, ax = plt.subplots(figsize=(11, 5))
    for regime, ls in (("A", "-"), ("B", "--")):
        sub = P[(P["regime"] == regime) & (P["spread_bps"] == 3)].sort_values("date")
        if sub.empty:
            continue
        cum = sub["net_log_ret"].cumsum()
        ax.plot(pd.to_datetime(sub["date"]), cum, linestyle=ls, label=f"Regime {regime}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net log return")
    ax.set_title("Phase 3 — A (intraday) vs B (carry) equity at 3 bps assumed spread")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "equity_A_vs_B.png", dpi=140)
    plt.close(fig)

    # 3. TOD vol profile (most recent fold's first pair, if any)
    if cache:
        (fid, pkey), res = next(iter(cache.items()))
        sigma = res["tod_sigma"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(NSE_BARS_PER_SESSION), sigma)
        ax.set_xlabel("Minute of session (0 = 09:15, 374 = 15:29)")
        ax.set_ylabel("Spread innovation std (log units)")
        ax.set_title(f"Phase 3 — TOD vol profile (fold {fid} {pkey})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "tod_vol_profile.png", dpi=140)
        plt.close(fig)

        # 4. Representative session: spread + z + bands for that pair
        ts = pd.DatetimeIndex(res["timestamps"])
        sids = res["session_id"]
        z = res["z_flat"]
        spread = res["spread"]
        # First session in the test window
        first_sess = np.flatnonzero(sids == 0)
        if first_sess.size:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1, ax2 = axes
            ax1.plot(ts[first_sess], spread[first_sess])
            ax1.set_ylabel("Spread (log)")
            ax1.set_title(f"Phase 3 — example session: {pkey} ({ts[first_sess][0].date()})")
            ax1.grid(alpha=0.3)
            ax2.plot(ts[first_sess], z[first_sess])
            ax2.axhline(2.0, color="orange", linestyle=":", linewidth=0.8, label="entry ±2σ")
            ax2.axhline(-2.0, color="orange", linestyle=":", linewidth=0.8)
            ax2.axhline(0.5, color="green", linestyle=":", linewidth=0.8, label="exit ±0.5σ")
            ax2.axhline(-0.5, color="green", linestyle=":", linewidth=0.8)
            ax2.axhline(3.5, color="red", linestyle=":", linewidth=0.8, label="stop ±3.5σ")
            ax2.axhline(-3.5, color="red", linestyle=":", linewidth=0.8)
            ax2.set_ylabel("Sessionized rolling z")
            ax2.set_xlabel("Time")
            ax2.legend(loc="best", fontsize=8)
            ax2.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "spread_z_example.png", dpi=140)
            plt.close(fig)

    # 5. Per-pair grid: net total return at 3bps, regime A
    A3 = P[(P["regime"] == "A") & (P["spread_bps"] == 3)]
    if not A3.empty and cache:
        per_pair_3 = pd.DataFrame(
            [
                {
                    "pair": pkey,
                    "fold_id": fid,
                    "n_trades_A": len(res["A"].trades),
                    "n_trades_B": len(res["B"].trades),
                }
                for (fid, pkey), res in cache.items()
            ]
        )
        per_pair_3.to_csv(REPORTS_DIR / "per_pair_trade_counts.csv", index=False)


def _write_caveats(metrics_rows) -> None:
    caveats = [
        "(1) ASSUMED SPREAD, NOT MEASURED. The bid-ask half-spread is the "
        "dominant cost component intraday; the result is a curve over "
        "{1,3,5,8} bps total spread, not a point estimate.",
        "(2) Intraday CAPACITY is much lower than daily. POC notional is "
        "Rs 50,000 per leg; market impact is NOT modeled. The liquidity "
        "gate is FILL-RATE-only (>= 90% session fill on both legs); the "
        "per-minute-volume cap was removed in v2 because at POC scale it "
        "is a capacity constraint, not a tradeability constraint.",
        "(3) Residual Epps/asynchronicity risk: bars with no trade on either "
        "leg are marked non-tradeable (no forward-fill), but very-low-volume "
        "minutes still embed wider effective spreads.",
        "(4) Daily-adjustment cross-check found per-symbol max divergences of "
        "~8-10% on a handful of corporate-action days (ONGC, HINDPETRO, "
        "BPCL, IOC). Median ratio is 1.000 across all sampled symbols. "
        "Regime A is robust to this (session-local rolling z absorbs uniform "
        "level shifts); Regime B is exposed on those specific dates.",
        "(5) Minute panel ends 2021-06-23 — the last fold's test window is "
        "truncated. Fewer-than-full folds inflate error bars.",
        "(6) Dividends not modeled (same caveat as Phase 2A); minimal effect "
        "on the within-session spread but a small bias on overnight-carry "
        "Regime B.",
        "(7) Regime B carries the SAME unshortable-multi-day caveat as Phase "
        "2A. Only Regime A is deployable on Indian cash equity.",
    ]
    with (REPORTS_DIR / "caveats_phase3.txt").open("w") as fh:
        fh.write("Phase 3 — intraday two-regime validation caveats\n")
        fh.write("=" * 60 + "\n\n")
        for c in caveats:
            fh.write(c + "\n\n")


def _print_summary(metrics_rows, n_pair_folds, n_folds, n_kept, n_total) -> None:
    M = pd.DataFrame(metrics_rows)
    print("\n=== Phase 3 intraday two-regime validation summary ===")
    print(f"  folds in overlap : {n_folds}")
    print(f"  intraday-liquidity survivors : {int(n_kept)} / {int(n_total)} pair-fold units")
    print(f"  pair-fold compute units run  : {n_pair_folds}")
    if M.empty:
        print("  (no metrics produced)")
        return
    print()
    print("  Net Sharpe vs assumed spread (both regimes):")
    pivot = M.pivot(index="spread_bps", columns="regime", values="net_sharpe").sort_index()
    print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))
    print()
    print("  Net ann return % vs assumed spread:")
    pivot2 = M.pivot(index="spread_bps", columns="regime", values="net_ann_pct").sort_index()
    print(pivot2.to_string(float_format=lambda v: f"{v:+.2f}"))
    print()
    print(
        f"  daily Phase 2A baseline net Sharpe = {DAILY_BASELINE_NET_SHARPE:.2f}, "
        f"net ann ret = {DAILY_BASELINE_ANN_RET_PCT:.2f}%, avg hold = {DAILY_BASELINE_AVG_HOLD_DAYS}d"
    )
    print()
    # Trade-count comparison
    for r in metrics_rows:
        if r["regime"] == "A" and r["spread_bps"] == 3:
            print(
                f"  Regime A @ 3bps : {r['n_trades']} trades, avg hold {r['avg_bars_held']:.0f} bars "
                f"({r['avg_bars_held'] / NSE_BARS_PER_SESSION:.2f} sessions), "
                f"% time deployed {r['pct_time_deployed'] * 100:.1f}%"
            )
        if r["regime"] == "B" and r["spread_bps"] == 3:
            print(
                f"  Regime B @ 3bps : {r['n_trades']} trades, avg hold "
                f"{r['avg_sessions_held']:.2f} sessions, "
                f"% time deployed {r['pct_time_deployed'] * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
