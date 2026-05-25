"""Backtest plots: equity curve + drawdown + active-pair count."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from apt.plots.style import apply_style


def plot_equity_curve(
    portfolio_daily: pl.DataFrame,
    out_path: Path,
    *,
    title: str = "Walk-forward portfolio",
) -> dict:
    """Three-panel plot: gross/net cumulative log return, drawdown, active-pair count.

    Expects ``portfolio_daily`` with columns ``date``, ``gross_log_ret``,
    ``net_log_ret``, ``n_active_pairs``.
    """
    apply_style()
    if portfolio_daily.is_empty():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no portfolio data", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return {"n_obs": 0}

    df = portfolio_daily.sort("date")
    dates = df["date"].to_list()
    gross = df["gross_log_ret"].to_numpy()
    net = df["net_log_ret"].to_numpy()
    n_active = df["n_active_pairs"].to_numpy()

    cum_gross = np.cumsum(gross)
    cum_net = np.cumsum(net)
    cum_gross_pct = (np.expm1(cum_gross)) * 100
    cum_net_pct = (np.expm1(cum_net)) * 100

    running_max_net = np.maximum.accumulate(cum_net)
    dd_log = cum_net - running_max_net
    dd_pct = (np.expm1(dd_log)) * 100

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True, height_ratios=[2.2, 1.0, 0.8]
    )
    # Equity
    ax1.plot(dates, cum_gross_pct, color="#3B8E5C", lw=1.1, label="cumulative gross")
    ax1.plot(dates, cum_net_pct, color="#2E86AB", lw=1.1, label="cumulative net (after costs)")
    ax1.axhline(0, color="#7E8083", lw=0.5, ls=":")
    ax1.set_ylabel("cumulative return (%)")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    # Drawdown
    ax2.fill_between(dates, dd_pct, 0, color="#C73E1D", alpha=0.4, step="pre")
    ax2.plot(dates, dd_pct, color="#C73E1D", lw=0.6)
    ax2.axhline(0, color="#7E8083", lw=0.5, ls=":")
    ax2.set_ylabel("drawdown (%) — net")

    # Active pairs
    ax3.fill_between(dates, n_active, 0, color="#6B5B95", alpha=0.4, step="pre")
    ax3.plot(dates, n_active, color="#6B5B95", lw=0.6)
    ax3.set_ylabel("active pairs")
    ax3.set_xlabel("date")
    ax3.set_ylim(bottom=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "n_obs": int(df.height),
        "final_gross_pct": float(cum_gross_pct[-1]),
        "final_net_pct": float(cum_net_pct[-1]),
        "max_drawdown_pct": float(dd_pct.min()),
    }
