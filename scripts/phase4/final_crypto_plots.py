#!/usr/bin/env python3
"""Phase 4 final — Part 3: crypto consolidated plots (from Part 2 A10 artifacts).

To plots/phase4/final/crypto/, each PNG + same-basename data CSV:
  portfolio_equity          — crypto kalman NAV (gross/net/DD), Regime A & B
                              (B labeled funding-[TODO]); rolling_z + frozen-OU
  per_pair_equity           — small-multiples grid per traded crypto pair
  trade_return_distribution — per-trade net P&L, kalman vs rolling_z
  per_pair_frequency        — n_trades, trades/day, holding, best freq per pair
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from apt.phase4 import savefig_with_csv  # noqa: E402
from apt.phase4.equity import compute_nav_drawdown, plot_equity_panels  # noqa: E402

SRC = Path("reports/phase4/crypto_adaptive")
FIG = Path("plots/phase4/final/crypto")


def _best_kalman_cell(metrics: pd.DataFrame) -> tuple[int, int]:
    """Best kalman cell by net Sharpe in Regime A (the funding-clean, authoritative regime)."""
    k = metrics[(metrics.engine == "kalman_mu") & (metrics.regime == "A")]
    if k.empty or k.net_sharpe.isna().all():
        k = metrics[metrics.engine == "kalman_mu"]
    best = k.loc[k.net_sharpe.idxmax()]
    return int(best.freq_min), int(best.spread_bps)


def _portfolio(sess: pd.DataFrame, engine, regime, freq, cost):
    s = sess[
        (sess.engine == engine)
        & (sess.regime == regime)
        & (sess.freq_min == freq)
        & (sess.spread_bps == cost)
    ]
    p = s.groupby("date").agg(g=("gross_log_ret", "mean"), n=("net_log_ret", "mean")).sort_index()
    return pd.to_datetime(p.index), p.g.to_numpy(), p.n.to_numpy()


def portfolio_equity(sess, freq, cost):
    panels = {}
    for label, eng, reg in (
        (f"kalman_mu  RegimeA  f{freq}/c{cost}  (funding-clean)", "kalman_mu", "A"),
        (f"kalman_mu  RegimeB  f{freq}/c{cost}  (funding-[TODO])", "kalman_mu", "B"),
        (f"frozen-OU  RegimeA  f{freq}/c{cost}", "frozen_ou", "A"),
        (f"rolling_z  RegimeA  f{freq}/c{cost}", "rolling_z", "A"),
    ):
        d, g, n = _portfolio(sess, eng, reg, freq, cost)
        if len(d) == 0:
            continue
        panels[label] = compute_nav_drawdown(d, g, n)
    fig, data = plot_equity_panels(
        panels,
        FIG / "portfolio_equity.png",
        suptitle=f"Crypto portfolio NAV — gross/net/drawdown (best kalman cell f{freq}/c{cost})",
    )
    savefig_with_csv(fig, FIG / "portfolio_equity.png", data)
    plt.close(fig)
    print("[part3] portfolio_equity done")


def per_pair_equity(sess, freq, cost):
    s = sess[
        (sess.engine == "kalman_mu")
        & (sess.regime == "A")
        & (sess.freq_min == freq)
        & (sess.spread_bps == cost)
    ]
    keys = list(s.groupby(["fold_id", "pair"]).groups.keys())[:12]  # cap small-multiples
    panels = {}
    for fid, pair in keys:
        sub = s[(s.fold_id == fid) & (s.pair == pair)]
        p = (
            sub.groupby("date")
            .agg(g=("gross_log_ret", "sum"), n=("net_log_ret", "sum"))
            .sort_index()
        )
        panels[f"f{fid} {pair}"] = compute_nav_drawdown(
            pd.to_datetime(p.index), p.g.to_numpy(), p.n.to_numpy()
        )
    if not panels:
        print("[part3] per_pair_equity: no kalman pairs traded")
        return
    fig, data = plot_equity_panels(
        panels,
        FIG / "per_pair_equity.png",
        suptitle=f"Crypto per-pair NAV (kalman RegimeA f{freq}/c{cost}) — gross/net/drawdown",
    )
    savefig_with_csv(fig, FIG / "per_pair_equity.png", data)
    plt.close(fig)
    print("[part3] per_pair_equity done")


def trade_return_distribution(trades, freq, cost):
    kt = trades[
        (trades.engine == "kalman_mu")
        & (trades.regime == "A")
        & (trades.freq_min == freq)
        & (trades.spread_bps == cost)
    ]
    rt = trades[
        (trades.engine == "rolling_z")
        & (trades.regime == "A")
        & (trades.freq_min == freq)
        & (trades.spread_bps == cost)
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    allv = np.concatenate([kt.net_log_pnl.to_numpy(), rt.net_log_pnl.to_numpy()])
    lo, hi = np.percentile(allv, [1, 99]) if allv.size else (-0.05, 0.05)
    bins = np.linspace(lo, hi, 50)
    ax.hist(
        kt.net_log_pnl,
        bins=bins,
        alpha=0.55,
        color="C3",
        label=f"kalman_mu (n={kt.shape[0]})",
        density=True,
    )
    ax.hist(
        rt.net_log_pnl,
        bins=bins,
        alpha=0.45,
        color="C0",
        label=f"rolling_z (n={rt.shape[0]})",
        density=True,
    )
    rows = []
    for nm, t, col in (("kalman", kt, "C3"), ("rolling_z", rt, "C0")):
        if t.empty:
            continue
        v = t.net_log_pnl.to_numpy()
        for k, val in {
            "mean": v.mean(),
            "median": np.median(v),
            "p5": np.percentile(v, 5),
            "p95": np.percentile(v, 95),
            "worst": v.min(),
        }.items():
            ax.axvline(val, color=col, lw=1.0, ls="--", alpha=0.6)
            rows.append({"engine": nm, "stat": k, "net_log_pnl": val, "bps": val * 1e4})
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("per-trade net log P&L")
    ax.set_ylabel("density")
    ax.set_title(f"Crypto per-trade net P&L — kalman vs rolling_z (RegimeA f{freq}/c{cost})")
    ax.legend()
    savefig_with_csv(fig, FIG / "trade_return_distribution.png", pd.DataFrame(rows))
    plt.close(fig)
    print("[part3] trade_return_distribution done")


def per_pair_frequency(trades):
    kt = trades[(trades.engine == "kalman_mu") & (trades.regime == "A")]
    rows = []
    for (pair, freq), g in kt.groupby(["pair", "freq_min"]):
        sess_sum = max(g.sessions_held.sum(), 1)
        rows.append(
            {
                "pair": pair,
                "freq_min": freq,
                "n_trades": g.shape[0],
                "mean_hold_bars": float(g.bars_held.mean()),
                "median_hold_bars": float(g.bars_held.median()),
                "mean_hold_min": float(g.bars_held.mean() * freq),
                "median_hold_min": float(g.bars_held.median() * freq),
                "trades_per_day": float(g.shape[0] / sess_sum),
                "net_total_log": float(g.net_log_pnl.sum()),
            }
        )
    df = pd.DataFrame(rows)
    df["best_freq_for_pair"] = False
    for _key, g in df.groupby("pair"):
        df.loc[g.net_total_log.idxmax(), "best_freq_for_pair"] = True
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df.groupby("freq_min").n_trades.sum().plot(kind="bar", ax=axes[0])
    axes[0].set_title("kalman total n_trades by freq (RegimeA)")
    axes[0].set_ylabel("n_trades")
    df.boxplot(column="median_hold_min", by="freq_min", ax=axes[1])
    axes[1].set_title("median holding (min) by freq")
    axes[1].set_ylabel("minutes")
    fig.suptitle("Crypto kalman per-pair frequency / holding")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_with_csv(fig, FIG / "per_pair_frequency.png", df)
    plt.close(fig)
    print("[part3] per_pair_frequency done")
    return df


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(SRC / "metrics.csv")
    sess = pd.read_csv(SRC / "per_session_all.csv")
    trades = pd.read_csv(SRC / "trades_all.csv")
    freq, cost = _best_kalman_cell(metrics)
    print(f"[part3] best kalman cell (RegimeA, net Sharpe): f{freq}/c{cost}")
    portfolio_equity(sess, freq, cost)
    per_pair_equity(sess, freq, cost)
    trade_return_distribution(trades, freq, cost)
    freqdf = per_pair_frequency(trades)
    freqdf.to_csv(SRC / "per_pair_frequency_summary.csv", index=False)
    print("=== final_crypto_plots complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
