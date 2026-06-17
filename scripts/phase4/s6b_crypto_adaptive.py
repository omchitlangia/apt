#!/usr/bin/env python3
"""Phase 4 final — Part 2 (A10): adaptive engines on crypto — the verdict experiment.

Wires the existing rolling_z (control), frozen-OU, and kalman μ-only engines
through the crypto intraday pipeline (UTC-day sessions). EG-FDR pair selection
(leakage-free, per fold); (1+β) billing; train-only global half-life selection
with the absorption guard; per-cost Bertram refit. Regime A (intraday
square-off, funding-CLEAN, authoritative) AND Regime B (multi-day carry,
funding-UNPRICED [TODO]). Bar-freq sweep {1,5,15}; cost = taker + spread.

Outputs (reports/phase4/crypto_adaptive/):
  selection_kalman.csv, metrics.csv (gross+net, all cells),
  per_session_kalman.csv, trades_kalman.csv, trades_rolling.csv,
  pairfolds.csv, beta_residual_watch.csv
"""

from __future__ import annotations

import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller

from apt.backtest import Pair, build_folds, compute_metrics
from apt.crypto import intraday as ci
from apt.crypto.costs import CryptoCostBreakdown
from apt.intraday import generate_signals_ou, generate_signals_two_regime, intraday_rolling_zscore
from apt.intraday.backtest import run_pair_fold
from apt.intraday.costs import TRADE_CSV_SCHEMA_VERSION
from apt.stats.kalman import run_local_level_mu
from apt.stats.ou import bertram_threshold, fit_ou_params

PANEL = Path("data/interim/crypto_daily.parquet")
OUT = Path("reports/phase4/crypto_adaptive")
MIN_QV = 10_000_000.0
FREQS = (1, 5, 15)
COSTS = (1, 3, 5, 8)
REGIMES = ("A", "B")
H_GRID = (math.inf, 20.0, 10.0, 5.0)
GUARD = (0.5, 1.5)
PRIOR, TRAIN, TEST = 270, 365, 180
EG_ALPHA = 0.05
MIN_OBS = 60
ROLL_WIN = 60
MAX_PAIRS_PER_FOLD = 8  # cap strongest EG-FDR pairs/fold (bounds 1-min runtime)
WINDOW_START, WINDOW_END = date(2022, 1, 1), date(2026, 4, 30)


def _eg(ly, lx):
    fit = sm.OLS(ly, sm.add_constant(lx)).fit()
    a, b = float(fit.params[0]), float(fit.params[1])
    resid = np.asarray(fit.resid)
    try:
        p = float(adfuller(resid, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        p = np.nan
    dr, lag = np.diff(resid), resid[:-1]
    k = -float(sm.OLS(dr, sm.add_constant(lag)).fit().params[1])
    hl = math.log(2) / k if k > 0 else 30.0
    return p, a, b, float(np.clip(hl, 2.0, 120.0))


def select_pairs(wide, prior_start, train_end):
    win = wide.loc[(wide.index >= prior_start) & (wide.index <= train_end)]
    syms = list(win.columns)
    cand = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            sy, sx = syms[i], syms[j]
            sub = win[[sy, sx]].dropna()
            if sub.shape[0] < 250:
                continue
            p, a, b, hl = _eg(sub[sy].to_numpy(), sub[sx].to_numpy())
            if np.isfinite(p) and b > 0:
                cand.append({"y": sy, "x": sx, "p": p, "a": a, "b": b, "hl": hl})
    if not cand:
        return []
    cdf = pd.DataFrame(cand)
    rej, _, _, _ = multipletests(cdf.p, alpha=EG_ALPHA, method="fdr_bh")
    cdf = cdf[rej].sort_values("p").head(MAX_PAIRS_PER_FOLD)  # strongest first, capped
    return [
        Pair(y_sym=r.y, x_sym=r.x, alpha=r.a, beta=r.b, half_life=r.hl) for _, r in cdf.iterrows()
    ]


def _max_holding(hl_days, freq):
    bps = 1440 // freq  # bars per UTC-day session
    return int(np.clip(round(hl_days * bps * 3), bps, 5 * bps))


def build_cache(wide, pairfolds):
    cache = {}
    for fold, p in pairfolds:
        for freq in FREQS:
            bars = ci.load_pair_resampled(p.y_sym, p.x_sym, fold.train_start, fold.test_end, freq)
            if bars is None:
                continue
            spread = bars.log_y - p.beta * bars.log_x - p.alpha
            ts = bars.timestamps
            day = pd.Timedelta(days=1)
            train_mask = np.asarray(
                (ts >= pd.Timestamp(fold.train_start)) & (ts < pd.Timestamp(fold.train_end) + day)
            )
            test_mask = np.asarray(
                (ts >= pd.Timestamp(fold.test_start)) & (ts < pd.Timestamp(fold.test_end) + day)
            )
            if train_mask.sum() < MIN_OBS or test_mask.sum() < MIN_OBS:
                continue
            ou = fit_ou_params(spread[train_mask], freq_minutes=freq, min_obs=MIN_OBS)
            if not ou.fit_ok:
                continue
            cache[(fold.fold_id, p.key, freq)] = {
                "fold": fold,
                "pair": p,
                "freq": freq,
                "spread": spread,
                "sids": bars.session_id,
                "ts": ts,
                "trd": bars.tradeable,
                "train": train_mask,
                "test": test_mask,
                "ou": ou,
                "hl_frozen": ou.half_life_minutes,
            }
    return cache


def _residual_fit(c, H):
    tr = c["train"]
    res = run_local_level_mu(
        c["spread"][tr],
        c["sids"][tr],
        mu_init=float(c["ou"].mu),
        half_life_sessions=H,
        tradeable=c["trd"][tr],
    )
    if not res.fit_ok:
        return None
    rfit = fit_ou_params(res.residual, freq_minutes=c["freq"], min_obs=MIN_OBS)
    if not rfit.fit_ok:
        return None
    ratio = rfit.half_life_minutes / c["hl_frozen"]
    return {"rfit": rfit, "ratio": ratio, "admissible": GUARD[0] <= ratio <= GUARD[1]}


def select_global_h(cache):
    rows = []
    for key, c in cache.items():
        for H in H_GRID:
            r = _residual_fit(c, H)
            Hv = np.inf if math.isinf(H) else H
            if r is None:
                rows.append({"key": str(key), "H": Hv, "admissible": False, "crit": np.nan})
                continue
            cost = CryptoCostBreakdown(total_spread_bps=3).billed_cost_log_per_pair_round_trip(
                c["pair"].beta
            )
            th = bertram_threshold(r["rfit"], cost_log_per_round_trip=cost)
            rows.append(
                {
                    "key": str(key),
                    "H": Hv,
                    "admissible": bool(r["admissible"] and th.fit_ok),
                    "ratio": r["ratio"],
                    "crit": th.expected_return_per_unit_time if th.fit_ok else np.nan,
                }
            )
    tab = pd.DataFrame(rows)
    summ = (
        tab[tab.admissible]
        .groupby("H")
        .crit.mean()
        .reset_index()
        .rename(columns={"crit": "mean_crit"})
    )
    summ["n_admissible"] = tab[tab.admissible].groupby("H").size().reindex(summ.H).values
    chosen = (
        float(summ.sort_values("mean_crit", ascending=False).iloc[0].H) if len(summ) else np.inf
    )
    tab.to_csv(OUT / "selection_kalman.csv", index=False)
    summ.to_csv(OUT / "selection_summary.csv", index=False)
    return chosen, summ


def run_cell(c, H):
    """Return per-(engine,regime,cost) trades + per-session for one pair-fold-freq."""
    out = {"trades": [], "sessions": []}
    rf = _residual_fit(c, H)
    if rf is None:
        return out
    sig_eq_resid = rf["rfit"].sigma_eq
    test = c["test"]
    spread_test = c["spread"][test]
    ts_test = c["ts"][test]
    _, sid_test = np.unique(c["sids"][test], return_inverse=True)
    sid_test = sid_test.astype(np.int32)
    trd_test = c["trd"][test]
    freq, pair, fold = c["freq"], c["pair"], c["fold"]
    maxh = _max_holding(pair.half_life, freq)

    # engine z-series on the TEST slice
    full = run_local_level_mu(
        c["spread"], c["sids"], mu_init=float(c["ou"].mu), half_life_sessions=H, tradeable=c["trd"]
    )
    z_kalman = (spread_test - full.mu_path[test]) / sig_eq_resid
    z_frozen = (spread_test - c["ou"].mu) / c["ou"].sigma_eq
    z_roll = intraday_rolling_zscore(spread_test, sid_test, window=ROLL_WIN)

    for cost in COSTS:
        cb = CryptoCostBreakdown(total_spread_bps=cost)
        cost_log = cb.billed_cost_log_per_pair_round_trip(pair.beta)
        # Bertram a* on the residual fit (kalman) and frozen fit
        th_k = bertram_threshold(rf["rfit"], cost_log_per_round_trip=cost_log)
        th_f = bertram_threshold(c["ou"], cost_log_per_round_trip=cost_log)
        for regime in REGIMES:
            engines = []
            if th_k.fit_ok:
                engines.append(
                    (
                        "kalman_mu",
                        z_kalman,
                        generate_signals_ou(
                            z_kalman,
                            sid_test,
                            trd_test,
                            regime=regime,
                            a_entry_z=th_k.a_entry_z,
                            stop_mode="none",
                            stop_k_sigma=0.0,
                            max_holding=maxh,
                        ),
                    )
                )
            if th_f.fit_ok:
                engines.append(
                    (
                        "frozen_ou",
                        z_frozen,
                        generate_signals_ou(
                            z_frozen,
                            sid_test,
                            trd_test,
                            regime=regime,
                            a_entry_z=th_f.a_entry_z,
                            stop_mode="none",
                            stop_k_sigma=0.0,
                            max_holding=maxh,
                        ),
                    )
                )
            engines.append(
                (
                    "rolling_z",
                    z_roll,
                    generate_signals_two_regime(
                        z_roll,
                        sid_test,
                        trd_test,
                        regime=regime,
                        entry=2.0,
                        exit=0.5,
                        stop=3.5,
                        max_holding=maxh,
                    ),
                )
            )
            for eng, zz, sig in engines:
                r = run_pair_fold(
                    fold_id=fold.fold_id,
                    pair_key=pair.key,
                    timestamps=ts_test,
                    session_id=sid_test,
                    spread=spread_test,
                    z=zz,
                    signals=sig,
                    cost_log_per_round_trip=cost_log,
                    finalize_fold_boundary=True,
                    pair_beta=pair.beta,
                )
                df = pd.DataFrame(
                    {
                        "date": pd.DatetimeIndex(r.timestamps).date,
                        "g": r.gross_log_ret,
                        "n": r.net_log_ret,
                    }
                )
                per = df.groupby("date", as_index=False)[["g", "n"]].sum()
                for _, rr in per.iterrows():
                    out["sessions"].append(
                        {
                            "engine": eng,
                            "regime": regime,
                            "freq_min": freq,
                            "spread_bps": cost,
                            "fold_id": fold.fold_id,
                            "pair": pair.key,
                            "date": rr["date"],
                            "gross_log_ret": rr.g,
                            "net_log_ret": rr.n,
                        }
                    )
                for t in r.trades:
                    out["trades"].append(
                        {
                            "engine": eng,
                            "regime": regime,
                            "freq_min": freq,
                            "spread_bps": cost,
                            "fold_id": t.fold_id,
                            "pair": t.pair_key,
                            "side": "long_spread" if t.direction == 1 else "short_spread",
                            "entry_ts": t.entry_ts.isoformat(),
                            "exit_ts": t.exit_ts.isoformat(),
                            "bars_held": t.bars_held,
                            "sessions_held": t.sessions_held,
                            "gross_log_pnl": t.gross_log_pnl,
                            "net_log_pnl": t.net_log_pnl,
                            "exit_reason": t.exit_reason,
                            "pair_beta": t.pair_beta,
                            "cost_log_per_pair_rt": t.cost_log,
                            "schema_version": TRADE_CSV_SCHEMA_VERSION,
                        }
                    )
    return out


def aggregate(sessions: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eng, reg, freq, cost), g in sessions.groupby(
        ["engine", "regime", "freq_min", "spread_bps"]
    ):
        port = g.groupby("date").agg(gr=("gross_log_ret", "mean"), nr=("net_log_ret", "mean"))
        mg, mn = compute_metrics(port.gr.to_numpy()), compute_metrics(port.nr.to_numpy())
        nt = trades[
            (trades.engine == eng)
            & (trades.regime == reg)
            & (trades.freq_min == freq)
            & (trades.spread_bps == cost)
        ].shape[0]
        rows.append(
            {
                "engine": eng,
                "regime": reg,
                "freq_min": freq,
                "spread_bps": cost,
                "n_pairs": g.pair.nunique(),
                "n_trades": int(nt),
                "n_sessions": port.shape[0],
                "gross_total_pct": mg["total_return_pct"],
                "net_total_pct": mn["total_return_pct"],
                "gross_sharpe": mg["sharpe"],
                "net_sharpe": mn["sharpe"],
                "net_ann_pct": mn["ann_return_pct"],
                "net_max_drawdown_pct": mn["max_drawdown_pct"],
                "funding_status": "clean" if reg == "A" else "UNPRICED_TODO",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ci.set_window_months("2021-09", "2026-04")
    t0 = time.time()
    panel = pl.read_parquet(PANEL)
    agg = panel.group_by("symbol").agg(mqv=pl.col("quote_volume").median()).to_pandas()
    liquid = list(agg[agg.mqv >= MIN_QV].symbol)
    pdf = panel.filter(pl.col("symbol").is_in(liquid)).to_pandas()
    wide = np.log(pdf.pivot(index="date", columns="symbol", values="close").sort_index())
    wide.index = pd.to_datetime(wide.index)
    wide = wide.loc[
        (wide.index >= pd.Timestamp(WINDOW_START)) & (wide.index <= pd.Timestamp(WINDOW_END))
    ]
    wide = wide.loc[wide.notna().sum(axis=1) >= 6]
    days = list(wide.index)
    folds = build_folds(days, prior_days=PRIOR, train_days=TRAIN, test_days=TEST)
    pairfolds = []
    for f in folds:
        for p in select_pairs(wide, f.prior_start, f.train_end):
            pairfolds.append((f, p))
    pd.DataFrame(
        [
            {
                "fold_id": f.fold_id,
                "pair": p.key,
                "beta": p.beta,
                "hl_days": p.half_life,
                "test_start": f.test_start,
                "test_end": f.test_end,
            }
            for f, p in pairfolds
        ]
    ).to_csv(OUT / "pairfolds.csv", index=False)
    print(
        f"[A10] {len(folds)} folds, {len(pairfolds)} EG-FDR pair-folds; building intraday cache..."
    )
    cache = build_cache(wide, pairfolds)
    print(f"[A10] cache: {len(cache)} (pair-fold,freq) cells in {time.time() - t0:.0f}s")
    chosen_H, summ = select_global_h(cache)
    print(f"[A10] global kalman H = {chosen_H} sessions\n{summ.to_string(index=False)}")
    all_sess, all_trades = [], []
    for i, c in enumerate(cache.values()):
        o = run_cell(c, chosen_H)
        all_sess.extend(o["sessions"])
        all_trades.extend(o["trades"])
        if (i + 1) % 20 == 0:
            print(f"  ran {i + 1}/{len(cache)} cells ({time.time() - t0:.0f}s)")
    sessions = pd.DataFrame(all_sess)
    trades = pd.DataFrame(all_trades)
    sessions.to_csv(OUT / "per_session_all.csv", index=False)
    trades.to_csv(OUT / "trades_all.csv", index=False)
    metrics = aggregate(sessions, trades)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    print(f"\n[A10] done in {time.time() - t0:.0f}s. Headline (Regime A, cost 3):")
    print(
        metrics[(metrics.regime == "A") & (metrics.spread_bps == 3)]
        .sort_values(["engine", "freq_min"])
        .round(2)
        .to_string(index=False)
    )
    print("\n=== s6b_crypto_adaptive complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
