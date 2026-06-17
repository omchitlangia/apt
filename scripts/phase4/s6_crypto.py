#!/usr/bin/env python3
"""Phase 4 — Section 6: Crypto port (scaffold + first results).

Pipeline mirroring the NSE methodology AS FAR AS THE DATA PERMITS:
  - daily panel from Binance 1m klines (apt.crypto), minimal cleaning
  - liquidity/turnover gate (labeled default → A_LIQ)
  - cointegration selection: EG+BH-FDR AND Johansen (Section 5 confound applies)
  - leakage-free walk-forward (apt.backtest.run_walkforward) — rolling-z engine
    (OU/Bertram + Kalman crypto engines are [TODO scope])
  - Regime B (multi-day carry); Regime A (intraday squareoff) needs intraday
    bars -> [TODO]. Risk-management = labeled default stop_z (flagged, not tuned).
  - crypto cost = taker + spread; funding [TODO] (A8). Gross AND net.
  - Section-2 DSR/PBO on the crypto results.

LABELED AS SCAFFOLD + FIRST RESULTS — not claimed validated.

Outputs (reports/phase4/crypto/):
  inventory.csv, liquidity_gate.csv, cointegration.csv, metrics.csv,
  per_pair.csv, dsr_pbo.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

from apt.backtest import Pair, build_folds, compute_metrics, run_walkforward  # noqa: E402
from apt.crypto.costs import CRYPTO_SPREAD_SWEEP_BPS, CryptoCostBreakdown  # noqa: E402
from apt.phase4 import savefig_with_csv  # noqa: E402
from apt.stats.dsr import deflated_sharpe_ratio  # noqa: E402
from apt.stats.johansen import is_cointegrated, johansen_pair_test  # noqa: E402
from apt.stats.pbo import pbo_cscv  # noqa: E402

PANEL = Path("data/interim/crypto_daily.parquet")
OUT = Path("reports/phase4/crypto")
FIG = Path("plots/phase4/crypto")

# Labeled defaults (→ ASSUMPTIONS).
MIN_MEDIAN_QUOTE_VOL_USD = 10_000_000.0  # A_LIQ liquidity/turnover gate ($/day)
EG_ALPHA = 0.05
STOP_Z = 3.5  # A9 risk-management default (flagged, NOT tuned)
ROLLING_WINDOW = 60
ENTRY_Z, EXIT_Z = 2.0, 0.5
PRIOR_DAYS, TRAIN_DAYS, TEST_DAYS = 365, 365, 180


def liquidity_gate(panel: pl.DataFrame) -> tuple[list[str], pd.DataFrame]:
    agg = (
        panel.group_by("symbol")
        .agg(
            median_quote_vol=pl.col("quote_volume").median(),
            n_days=pl.len(),
            first=pl.col("date").min(),
            last=pl.col("date").max(),
        )
        .sort("median_quote_vol", descending=True)
        .to_pandas()
    )
    agg["passes"] = agg.median_quote_vol >= MIN_MEDIAN_QUOTE_VOL_USD
    agg.to_csv(OUT / "liquidity_gate.csv", index=False)
    return list(agg[agg.passes].symbol), agg


def _wide_log_close(panel: pl.DataFrame, symbols: list[str]) -> pd.DataFrame:
    pdf = panel.filter(pl.col("symbol").is_in(symbols)).to_pandas()
    wide = pdf.pivot(index="date", columns="symbol", values="close").sort_index()
    return np.log(wide)


def _eg(ly: np.ndarray, lx: np.ndarray) -> tuple[float, float, float, float]:
    """EG: returns (adf_p, alpha, beta, half_life_days)."""
    X = sm.add_constant(lx)
    fit = sm.OLS(ly, X).fit()
    a, b = float(fit.params[0]), float(fit.params[1])
    resid = np.asarray(fit.resid)
    try:
        p = float(adfuller(resid, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        p = np.nan
    # half-life from AR(1) on the residual
    dr = np.diff(resid)
    lag = resid[:-1]
    phi_fit = sm.OLS(dr, sm.add_constant(lag)).fit()
    k = -float(phi_fit.params[1])
    hl = math.log(2) / k if k > 0 else 30.0
    return p, a, b, float(np.clip(hl, 2.0, 120.0))


def cointegration_selection(wide: pd.DataFrame) -> pd.DataFrame:
    """EG+BH-FDR AND Johansen on all pairs over the full liquid window."""
    syms = list(wide.columns)
    rows = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            sy, sx = syms[i], syms[j]
            sub = wide[[sy, sx]].dropna()
            if sub.shape[0] < 250:
                continue
            ly, lx = sub[sy].to_numpy(), sub[sx].to_numpy()
            adf_p, a, b, hl = _eg(ly, lx)
            jt = johansen_pair_test(ly, lx)
            rows.append(
                {
                    "y_sym": sy,
                    "x_sym": sx,
                    "n_obs": sub.shape[0],
                    "eg_adf_pvalue": adf_p,
                    "alpha": a,
                    "beta": b,
                    "half_life_days": hl,
                    "johansen_coint": bool(is_cointegrated(jt)),
                    "johansen_rank95": jt.rank_95,
                }
            )
    df = pd.DataFrame(rows)
    valid = df.eg_adf_pvalue.notna()
    df["eg_fdr_pass"] = False
    if valid.any():
        rej, _, _, _ = multipletests(
            df.loc[valid, "eg_adf_pvalue"], alpha=EG_ALPHA, method="fdr_bh"
        )
        df.loc[valid, "eg_fdr_pass"] = rej
    df.to_csv(OUT / "cointegration.csv", index=False)
    return df


def _make_callbacks(wide: pd.DataFrame):
    """Leakage-free select-pairs (EG+BH-FDR on train+prior only) + price getter."""

    def select_pairs_fn(prior_start, prior_end, train_start, train_end):
        win = wide.loc[(wide.index >= prior_start) & (wide.index <= train_end)]
        syms = list(win.columns)
        cand = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                sy, sx = syms[i], syms[j]
                sub = win[[sy, sx]].dropna()
                if sub.shape[0] < 250:
                    continue
                adf_p, a, b, hl = _eg(sub[sy].to_numpy(), sub[sx].to_numpy())
                if np.isfinite(adf_p) and b > 0:
                    cand.append({"y": sy, "x": sx, "p": adf_p, "a": a, "b": b, "hl": hl})
        if not cand:
            return []
        cdf = pd.DataFrame(cand)
        rej, _, _, _ = multipletests(cdf.p, alpha=EG_ALPHA, method="fdr_bh")
        cdf = cdf[rej]
        return [
            Pair(y_sym=r.y, x_sym=r.x, alpha=r.a, beta=r.b, half_life=r.hl)
            for _, r in cdf.iterrows()
        ]

    def get_prices_fn(y_sym, x_sym, start, end):
        sub = wide.loc[(wide.index >= start) & (wide.index <= end), [y_sym, x_sym]].dropna()
        return list(sub.index), np.exp(sub[y_sym].to_numpy()), np.exp(sub[x_sym].to_numpy())

    return select_pairs_fn, get_prices_fn


def backtest(wide: pd.DataFrame, coint: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trading_days = list(wide.index)
    folds = build_folds(
        trading_days, prior_days=PRIOR_DAYS, train_days=TRAIN_DAYS, test_days=TEST_DAYS
    )
    select_fn, price_fn = _make_callbacks(wide)
    rows, per_pair_frames = [], {}
    for cost in CRYPTO_SPREAD_SWEEP_BPS:
        cb = CryptoCostBreakdown(total_spread_bps=cost)
        res = run_walkforward(
            folds,
            trading_days,
            select_pairs_fn=select_fn,
            get_prices_fn=price_fn,
            rolling_window=ROLLING_WINDOW,
            entry_z=ENTRY_Z,
            exit_z=EXIT_Z,
            stop_z=STOP_Z,
            cost_bps_per_leg=cb.cost_bps_per_leg,
        )
        pdf = res.portfolio_daily.to_pandas() if res.portfolio_daily.height else pd.DataFrame()
        if pdf.empty:
            continue
        mg = compute_metrics(pdf.gross_log_ret.to_numpy())
        mn = compute_metrics(pdf.net_log_ret.to_numpy())
        rows.append(
            {
                "engine": "rolling_z",
                "regime": "B",
                "cost_bps": cost,
                "taker_bps_per_side": cb.taker_fee_bps_per_side,
                "n_trades": res.funnel["n_trades"],
                "n_pair_fold_units": res.funnel["n_pair_fold_units"],
                "gross_total_pct": mg["total_return_pct"],
                "net_total_pct": mn["total_return_pct"],
                "gross_sharpe": mg["sharpe"],
                "net_sharpe": mn["sharpe"],
                "net_ann_pct": mn["ann_return_pct"],
                "net_max_drawdown_pct": mn["max_drawdown_pct"],
                "n_sessions": pdf.shape[0],
            }
        )
        if cost == 3:  # keep per-pair return matrix at the reference cost
            for k, v in res.per_pair_daily.items():
                vp = v.to_pandas()
                per_pair_frames[k] = vp.set_index("date").net_log_ret
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    if per_pair_frames:
        mat = pd.DataFrame(per_pair_frames).fillna(0.0)
        mat.to_csv(OUT / "per_pair_returns_c3.csv")
    else:
        mat = pd.DataFrame()
    return metrics, mat, res.funnel


def dsr_pbo_crypto(metrics: pd.DataFrame, mat: pd.DataFrame) -> pd.DataFrame:
    """Section-2 machinery on the crypto results."""
    rows = []
    if not metrics.empty:
        best = metrics.loc[metrics.net_sharpe.idxmax()]
        # portfolio per-period series at the best cost (reconstruct sharpe via metrics)
        # trial variance across the crypto cost cells (de-annualized)
        v = (
            float(np.var(metrics.net_sharpe.to_numpy() / np.sqrt(252), ddof=1))
            if metrics.shape[0] > 1
            else 0.001
        )
        # use the per-pair matrix portfolio (mean across pairs) as the candidate series
        if not mat.empty:
            port = mat.mean(axis=1).to_numpy()
            res = deflated_sharpe_ratio(
                port, n_trials=int(metrics.shape[0]), trial_sharpe_var=max(v, 1e-6)
            )
            rows.append(
                {
                    "candidate": "crypto rolling_z portfolio (c3)",
                    "n_pairs_cols": mat.shape[1],
                    "per_period_sharpe": res.sr_hat,
                    "sample_length_T": res.sample_length,
                    "n_trials": res.n_trials,
                    "DSR": res.dsr,
                    "DSR_pvalue": res.p_value,
                    "best_cell_net_sharpe": float(best.net_sharpe),
                }
            )
    pbo_val = np.nan
    if not mat.empty and mat.shape[1] >= 2 and mat.shape[0] >= 16:
        pbo_val = pbo_cscv(mat.to_numpy(), n_blocks=16).pbo
    df = pd.DataFrame(rows)
    df["PBO_across_pairs"] = pbo_val
    df.to_csv(OUT / "dsr_pbo.csv", index=False)
    return df


def write_inventory() -> pd.DataFrame:
    from apt.crypto import discover_symbols

    panel = pl.read_parquet(PANEL)
    cov = (
        panel.group_by("symbol")
        .agg(first=pl.col("date").min(), last=pl.col("date").max(), n_days=pl.len())
        .sort("symbol")
        .to_pandas()
    )
    inv = pd.DataFrame(
        [
            {
                "n_symbols": len(discover_symbols()),
                "bar_freq": "1m (Binance klines)",
                "columns": "12-col headerless (open_time ms/us UTC, OHLCV, quote_vol, n_trades, taker_buy_*)",
                "tz": "UTC",
                "date_range": f"{cov['first'].min()} -> {cov['last'].max()}",
                "funding_series_present": False,
                "panel_rows": panel.shape[0],
            }
        ]
    )
    inv.to_csv(OUT / "inventory.csv", index=False)
    cov.to_csv(OUT / "symbol_coverage.csv", index=False)
    return inv


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if not PANEL.exists():
        print("[s6] PANEL not cached — run build_daily_panel first")
        return 1
    inv = write_inventory()
    print("[6a] inventory:\n", inv.T.to_string(header=False))
    panel = pl.read_parquet(PANEL)
    liquid, agg = liquidity_gate(panel)
    print(
        f"\n[6b] liquidity gate (median quote vol >= ${MIN_MEDIAN_QUOTE_VOL_USD:,.0f}): "
        f"{len(liquid)}/{agg.shape[0]} symbols -> {liquid}"
    )
    wide = _wide_log_close(panel, liquid)
    # restrict to the common window where >= 6 liquid symbols have data
    counts = wide.notna().sum(axis=1)
    wide = wide.loc[counts >= 6]
    print(f"[6b] panel window: {wide.index.min()} -> {wide.index.max()} ({wide.shape[0]} days)")
    coint = cointegration_selection(wide)
    print(
        f"[6b] cointegration: {coint.eg_fdr_pass.sum()} EG-FDR, {coint.johansen_coint.sum()} Johansen "
        f"of {coint.shape[0]} pairs"
    )
    metrics, mat, funnel = backtest(wide, coint)
    print(f"\n[6e] walk-forward (rolling_z, Regime B), funnel={funnel}")
    print(metrics.round(2).to_string(index=False))
    dsr = dsr_pbo_crypto(metrics, mat)
    print("\n[6e] crypto DSR/PBO:")
    print(dsr.round(4).to_string(index=False))

    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(metrics.cost_bps, metrics.net_sharpe, "o-", label="net Sharpe")
        ax.plot(metrics.cost_bps, metrics.gross_sharpe, "s--", label="gross Sharpe")
        ax.set_xlabel("spread cost (bps)")
        ax.set_ylabel("Sharpe")
        ax.set_title("crypto rolling_z cost ladder (Regime B; taker+spread, funding [TODO])")
        ax.legend()
        savefig_with_csv(fig, FIG / "crypto_cost_ladder.png", metrics)
        plt.close(fig)
    print("\n=== s6_crypto complete (SCAFFOLD + FIRST RESULTS) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
