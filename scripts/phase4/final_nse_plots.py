#!/usr/bin/env python3
"""Phase 4 final — Part 1: NSE consolidated plots (persisted artifacts, NO re-runs).

To plots/phase4/final/nse/, each PNG + same-basename data CSV:
  portfolio_equity          — kalman f5/c1 NAV (gross/net/DD) + frozen-OU f5/c3 + rolling_z
  per_pair_equity           — one panel per traded pair-fold (gross/net/DD)
  trade_return_distribution — per-trade net log-P&L, kalman vs frozen-OU
  per_pair_frequency        — n_trades, trades/session, holding (bars+min), best freq
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

KALMAN = Path("reports/phase3_kalman")
OU = Path("reports/phase3_ou")
FIG = Path("plots/phase4/final/nse")
MATCHED = {(4, "INDUSINDBK/HDFCBANK"), (6, "KOTAKBANK/HDFCBANK")}


def _matched(ps: pd.DataFrame, freq, cost, stop=None) -> pd.DataFrame:
    m = (ps.freq_min == freq) & (ps.regime == "B") & (ps.spread_bps == cost)
    if stop is not None:
        m &= ps.stop_mode == stop
    sub = ps[m]
    sub = sub[sub.apply(lambda r: (r.fold_id, r.pair) in MATCHED, axis=1)]
    return sub


def _port(sub: pd.DataFrame):
    p = sub.groupby("date").agg(g=("gross_log_ret", "mean"), n=("net_log_ret", "mean")).sort_index()
    return pd.to_datetime(p.index), p.g.to_numpy(), p.n.to_numpy()


def portfolio_equity():
    psk = pd.read_csv(KALMAN / "pair_sessions_kalman.csv")
    pso = pd.read_csv(OU / "pair_sessions_ou.csv")
    psr = pd.read_csv(OU / "pair_sessions_rolling_baseline.csv")
    panels = {}
    for label, sub in (
        ("kalman_mu  f5/c1/B", _matched(psk, 5, 1)),
        ("frozen-OU  f5/c3/B", _matched(pso, 5, 3, stop="none")),
        ("rolling_z  f5/c3/B", _matched(psr, 5, 3)),
    ):
        d, g, n = _port(sub)
        panels[label] = compute_nav_drawdown(d, g, n)
    fig, data = plot_equity_panels(
        panels,
        FIG / "portfolio_equity.png",
        suptitle="NSE matched portfolio NAV — gross / net / drawdown (n=2 pair-folds)",
    )
    savefig_with_csv(fig, FIG / "portfolio_equity.png", data)
    plt.close(fig)
    print("[part1] portfolio_equity done")


def per_pair_equity():
    psk = pd.read_csv(KALMAN / "pair_sessions_kalman.csv")
    panels = {}
    for fid, pair in [(4, "INDUSINDBK/HDFCBANK"), (6, "KOTAKBANK/HDFCBANK")]:
        sub = psk[
            (psk.freq_min == 5) & (psk.spread_bps == 1) & (psk.fold_id == fid) & (psk.pair == pair)
        ]
        p = (
            sub.groupby("date")
            .agg(g=("gross_log_ret", "sum"), n=("net_log_ret", "sum"))
            .sort_index()
        )
        panels[f"fold{fid} {pair}"] = compute_nav_drawdown(
            pd.to_datetime(p.index), p.g.to_numpy(), p.n.to_numpy()
        )
    fig, data = plot_equity_panels(
        panels,
        FIG / "per_pair_equity.png",
        suptitle="NSE per-pair-fold NAV (kalman f5/c1) — gross / net / drawdown",
    )
    savefig_with_csv(fig, FIG / "per_pair_equity.png", data)
    plt.close(fig)
    print("[part1] per_pair_equity done")


def trade_return_distribution():
    kt = pd.read_csv(KALMAN / "trades_kalman.csv")
    kt = kt[(kt.freq_min == 5) & (kt.spread_bps == 1)]
    ft = pd.read_csv(OU / "trades_ou.csv")
    ft = ft[
        (ft.freq_min == 5) & (ft.regime == "B") & (ft.spread_bps == 3) & (ft.stop_mode == "none")
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-0.06, 0.13, 45)
    ax.hist(
        kt.net_log_pnl, bins=bins, alpha=0.55, color="C3", label=f"kalman f5/c1 (n={kt.shape[0]})"
    )
    ax.hist(
        ft.net_log_pnl,
        bins=bins,
        alpha=0.55,
        color="C0",
        label=f"frozen-OU f5/c3 (n={ft.shape[0]})",
    )
    rows = []
    for nm, t, col in (("kalman", kt, "C3"), ("frozen_ou", ft, "C0")):
        v = t.net_log_pnl.to_numpy()
        stats = {
            "mean": v.mean(),
            "median": np.median(v),
            "p5": np.percentile(v, 5),
            "p95": np.percentile(v, 95),
            "worst": v.min(),
        }
        for k, val in stats.items():
            ax.axvline(val, color=col, lw=1.0, ls="--", alpha=0.7)
            rows.append({"engine": nm, "stat": k, "net_log_pnl": val, "bps": val * 1e4})
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("per-trade net log P&L")
    ax.set_ylabel("count")
    ax.set_title("NSE per-trade net P&L — kalman vs frozen-OU (mean/median/p5/p95/worst marked)")
    ax.legend()
    savefig_with_csv(fig, FIG / "trade_return_distribution.png", pd.DataFrame(rows))
    plt.close(fig)
    print("[part1] trade_return_distribution done")


def per_pair_frequency():
    kt = pd.read_csv(KALMAN / "trades_kalman.csv")
    rows = []
    for fid, pair in [(4, "INDUSINDBK/HDFCBANK"), (6, "KOTAKBANK/HDFCBANK")]:
        for freq in (5, 15):
            sub = kt[
                (kt.freq_min == freq)
                & (kt.spread_bps == 1)
                & (kt.fold_id == fid)
                & (kt.pair == pair)
            ]
            if sub.empty:
                continue
            rows.append(
                {
                    "fold_id": fid,
                    "pair": pair,
                    "freq_min": freq,
                    "engine": "kalman_mu",
                    "n_trades": sub.shape[0],
                    "mean_hold_bars": float(sub.bars_held.mean()),
                    "median_hold_bars": float(sub.bars_held.median()),
                    "mean_hold_min": float(sub.bars_held.mean() * freq),
                    "median_hold_min": float(sub.bars_held.median() * freq),
                    "mean_sessions_held": float(sub.sessions_held.mean()),
                    "trades_per_session": float(sub.shape[0] / max(sub.sessions_held.sum(), 1)),
                    "net_total_log": float(sub.net_log_pnl.sum()),
                }
            )
    df = pd.DataFrame(rows)
    # best freq per pair-fold by net_total_log
    df["best_freq_for_pairfold"] = False
    for _key, g in df.groupby(["fold_id", "pair"]):
        bi = g.net_total_log.idxmax()
        df.loc[bi, "best_freq_for_pairfold"] = True
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    piv_n = df.pivot_table(index=["fold_id", "pair"], columns="freq_min", values="n_trades")
    piv_h = df.pivot_table(index=["fold_id", "pair"], columns="freq_min", values="median_hold_min")
    piv_n.plot(kind="bar", ax=axes[0])
    axes[0].set_title("n_trades by freq")
    axes[0].set_ylabel("n_trades")
    piv_h.plot(kind="bar", ax=axes[1])
    axes[1].set_title("median holding (min) by freq")
    axes[1].set_ylabel("minutes")
    for ax in axes:
        ax.set_xticklabels(
            [f"f{i}\n{p.split('/')[0]}" for i, p in piv_n.index], rotation=0, fontsize=8
        )
    fig.suptitle("NSE kalman per-pair-fold frequency/holding (freq 1 not a kalman survivor)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_with_csv(fig, FIG / "per_pair_frequency.png", df)
    plt.close(fig)
    print("[part1] per_pair_frequency done")
    print(df.to_string(index=False))


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    portfolio_equity()
    per_pair_equity()
    trade_return_distribution()
    per_pair_frequency()
    print("=== final_nse_plots complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
