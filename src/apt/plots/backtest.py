"""Backtest plots: equity curve + drawdown + active-pair count + per-pair diagnostics."""

from __future__ import annotations

import math
from datetime import date
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


# ---------------------------------------------------------------------------
# Per-pair diagnostics (Phase 2A drill-down)
# ---------------------------------------------------------------------------


def plot_per_pair_overview(
    pair_curves: list[dict],
    out_path: Path,
    *,
    title: str = "Per-pair cumulative net P&L (sorted best → worst)",
    cols: int = 5,
) -> dict:
    """Small-multiples grid of per-pair cumulative net P&L curves.

    Each item in ``pair_curves`` must have:
        ``pair``           : str
        ``dates``          : list[date]
        ``cum_net_pct``    : np.ndarray (same length as dates)
        ``n_trades``       : int
        ``net_total_pct``  : float
        ``net_sharpe``     : float
    The list is rendered in the order given — caller is expected to sort.
    """
    apply_style()
    n = len(pair_curves)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no pairs to plot", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return {"n_pairs": 0}

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.3))
    axes_flat = np.atleast_1d(np.array(axes)).flatten()

    for i, curve in enumerate(pair_curves):
        ax = axes_flat[i]
        cum_pct = np.asarray(curve["cum_net_pct"], dtype=float)
        color = "#3B8E5C" if cum_pct[-1] > 0 else "#C73E1D"
        ax.plot(curve["dates"], cum_pct, color=color, lw=1.0)
        ax.fill_between(curve["dates"], cum_pct, 0, color=color, alpha=0.12)
        ax.axhline(0, color="#7E8083", lw=0.5, ls=":")
        ax.set_title(
            f"{curve['pair']}\n"
            f"n={curve['n_trades']}  net={curve['net_total_pct']:+.0f}%  "
            f"Sh={curve['net_sharpe']:+.2f}",
            fontsize=8,
        )
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"n_pairs": n, "rows": rows, "cols": cols}


_MARKER_BY_REASON = {
    "mean_revert": ("v", "#2E86AB", "mean-revert exit"),
    "stop": ("X", "#C73E1D", "stop"),
    "time": ("s", "#A23B72", "time stop"),
    "fold_boundary": ("D", "#F18F01", "fold-boundary close"),
}


def plot_pair_card(
    *,
    pair_key: str,
    sector: str | None,
    y_sym: str,
    x_sym: str,
    full_dates: list[date],
    py_norm: np.ndarray,
    px_norm: np.ndarray,
    fold_segments: list[dict],
    trades: list,
    cum_dates: list[date],
    cum_net_pct: np.ndarray,
    stats: dict,
    out_path: Path,
    test_window_spans: list[tuple[date, date]] | None = None,
) -> dict:
    """Three-panel diagnostic card for one pair.

    Panel (a) — both legs' prices normalized to the first active date.
    Panel (b) — spread (piecewise per fold) with rolling-z mean ±2σ bands,
                test windows shaded, and entry/exit markers (one shape per
                exit reason).
    Panel (c) — cumulative net P&L over active dates.

    ``fold_segments`` is a list of dicts (one per fold the pair was in), each
    with: ``dates`` (list), ``spread``, ``roll_mean``, ``band_upper``,
    ``band_lower`` (np.ndarrays of equal length).

    ``trades`` is a list of objects with attrs:
        entry_date, exit_date, direction, exit_reason.
    """
    apply_style()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(12, 8.5), sharex=True, height_ratios=[1.0, 1.5, 1.0]
    )

    # --- Panel A: normalized leg prices ---
    ax_a.plot(full_dates, py_norm, label=y_sym, color="#2E86AB", lw=1.0)
    ax_a.plot(full_dates, px_norm, label=x_sym, color="#A23B72", lw=1.0)
    ax_a.axhline(1.0, color="#7E8083", lw=0.5, ls=":")
    ax_a.set_ylabel("price (norm. to first active)")
    ax_a.legend(loc="upper left", fontsize=8)
    ax_a.grid(True, alpha=0.3)

    # --- Panel B: spread + bands + markers + fold shading ---
    if test_window_spans:
        for start, end in test_window_spans:
            ax_b.axvspan(start, end, color="#E8E8E8", alpha=0.4)
    date_to_spread: dict[date, float] = {}
    for seg in fold_segments:
        seg_dates = seg["dates"]
        seg_spread = np.asarray(seg["spread"], dtype=float)
        seg_mean = np.asarray(seg["roll_mean"], dtype=float)
        seg_up = np.asarray(seg["band_upper"], dtype=float)
        seg_lo = np.asarray(seg["band_lower"], dtype=float)
        ax_b.fill_between(
            seg_dates,
            seg_lo,
            seg_up,
            color="#3B8E5C",
            alpha=0.12,
            step="pre",
            linewidth=0,
        )
        ax_b.plot(seg_dates, seg_mean, color="#7E8083", lw=0.6, ls="--")
        ax_b.plot(seg_dates, seg_spread, color="#2E86AB", lw=0.9)
        for d, s in zip(seg_dates, seg_spread, strict=False):
            date_to_spread[d] = float(s)

    # Trade markers
    for t in trades:
        # Entry marker: green triangle up (long) or red triangle down (short)
        es = date_to_spread.get(t.entry_date)
        if es is not None:
            marker = "^" if t.direction > 0 else "v"
            ax_b.scatter(
                [t.entry_date],
                [es],
                marker=marker,
                facecolor="#3B8E5C" if t.direction > 0 else "#C73E1D",
                edgecolor="black",
                s=45,
                linewidths=0.5,
                zorder=5,
            )
        xs = date_to_spread.get(t.exit_date)
        if xs is not None and t.exit_reason in _MARKER_BY_REASON:
            mshape, mcolor, _label = _MARKER_BY_REASON[t.exit_reason]
            ax_b.scatter(
                [t.exit_date],
                [xs],
                marker=mshape,
                facecolor=mcolor,
                edgecolor="black",
                s=55,
                linewidths=0.5,
                zorder=5,
            )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#3B8E5C",
            markeredgecolor="black",
            label="long entry",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="#C73E1D",
            markeredgecolor="black",
            label="short entry",
            markersize=8,
        ),
    ]
    for shape, c, lbl in _MARKER_BY_REASON.values():
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=shape,
                color="w",
                markerfacecolor=c,
                markeredgecolor="black",
                label=lbl,
                markersize=8,
            )
        )
    ax_b.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=3)
    ax_b.set_ylabel("spread (log)  |  rolling μ ± 2σ")
    ax_b.grid(True, alpha=0.3)

    # --- Panel C: cumulative net P&L ---
    cum_arr = np.asarray(cum_net_pct, dtype=float)
    final_color = "#3B8E5C" if cum_arr.size and cum_arr[-1] > 0 else "#C73E1D"
    ax_c.plot(cum_dates, cum_arr, color=final_color, lw=1.0)
    ax_c.fill_between(cum_dates, cum_arr, 0, color=final_color, alpha=0.15)
    ax_c.axhline(0, color="#7E8083", lw=0.5, ls=":")
    ax_c.set_ylabel("cumulative net P&L (%)")
    ax_c.set_xlabel("date")
    ax_c.grid(True, alpha=0.3)

    # Title
    sector_str = f"  [{sector}]" if sector else ""
    exits_str = (
        f"MR={stats['n_exits_mean_revert']}  STP={stats['n_exits_stop']}  "
        f"TM={stats['n_exits_time']}  FB={stats['n_exits_fold_boundary']}"
    )
    sharpe = stats["net_sharpe"]
    sharpe_str = f"{sharpe:+.2f}" if math.isfinite(sharpe) else "n/a"
    fig.suptitle(
        f"{pair_key}{sector_str}    "
        f"trades={stats['n_trades']}  "
        f"net={stats['net_total_pct']:+.1f}%  "
        f"net Sh={sharpe_str}  "
        f"win={stats['win_rate_net'] * 100:.0f}%  "
        f"avg-hold={stats['avg_holding_days']:.1f}d   |   exits: {exits_str}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "pair": pair_key,
        "n_segments": len(fold_segments),
        "n_trades": len(trades),
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Phase 2B — risk-managed ablation plots
# ---------------------------------------------------------------------------


_RUNG_COLORS = {
    "R0": "#7E8083",
    "R1": "#F18F01",
    "R2": "#3B8E5C",
    "R3": "#2E86AB",
    "R4": "#A23B72",
}


def plot_ladder_equity(
    rung_results: dict[str, pl.DataFrame],
    out_path: Path,
    *,
    title: str = "Risk-management ladder — cumulative net equity (gross of costs already applied)",
    annotate_period: tuple[date, date] | None = None,
    annotate_label: str = "2018-19 PSU-bank breakdown fold",
) -> dict:
    """Overlay R0..R4 net cumulative log-return curves on one chart.

    ``rung_results`` maps rung label (e.g. ``'R0'``) → portfolio_daily frame
    with columns ``date`` and ``net_log_ret``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for rung_label, df in rung_results.items():
        if df.is_empty():
            continue
        d_sorted = df.sort("date")
        dates = d_sorted["date"].to_list()
        cum_net_pct = (np.expm1(np.cumsum(d_sorted["net_log_ret"].to_numpy()))) * 100
        color = _RUNG_COLORS.get(rung_label, "#7E8083")
        ax.plot(dates, cum_net_pct, label=rung_label, color=color, lw=1.2)
    ax.axhline(0, color="#7E8083", lw=0.5, ls=":")
    if annotate_period is not None:
        ax.axvspan(annotate_period[0], annotate_period[1], color="#E8E8E8", alpha=0.6)
        ax.annotate(
            annotate_label,
            xy=(annotate_period[0], ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 0),
            fontsize=8,
            color="#7E8083",
        )
    ax.set_ylabel("cumulative net return (%)")
    ax.set_xlabel("date")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"n_rungs": len(rung_results)}


def plot_drawdown_per_rung(
    rung_results: dict[str, pl.DataFrame],
    out_path: Path,
    *,
    annotate_period: tuple[date, date] | None = None,
    title: str = "Drawdown per rung (net of costs)",
) -> dict:
    """Drawdown curves R0..R4, with the breakdown fold optionally annotated."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for rung_label, df in rung_results.items():
        if df.is_empty():
            continue
        d_sorted = df.sort("date")
        dates = d_sorted["date"].to_list()
        cum_log = np.cumsum(d_sorted["net_log_ret"].to_numpy())
        running_max = np.maximum.accumulate(cum_log)
        dd_pct = (np.expm1(cum_log - running_max)) * 100
        color = _RUNG_COLORS.get(rung_label, "#7E8083")
        ax.plot(dates, dd_pct, label=rung_label, color=color, lw=1.0)
    if annotate_period is not None:
        ax.axvspan(annotate_period[0], annotate_period[1], color="#E8E8E8", alpha=0.6)
    ax.axhline(0, color="#7E8083", lw=0.5, ls=":")
    ax.set_ylabel("drawdown (%)")
    ax.set_xlabel("date")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"n_rungs": len(rung_results)}


def plot_cluster_exposure(
    cluster_df: pl.DataFrame,
    cluster_cap: float | None,
    out_path: Path,
    *,
    title: str = "Per-cluster open exposure (capital fraction)",
    highlight_period: tuple[date, date] | None = None,
) -> dict:
    """Stacked-area / line plot of per-cluster open exposure over time."""
    apply_style()
    if cluster_df.is_empty():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "no cluster exposure data", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return {"n_clusters": 0}
    sectors = sorted(cluster_df["sector"].unique().to_list())
    pivot = (
        cluster_df.pivot(index="date", on="sector", values="exposure").sort("date").fill_null(0.0)
    )
    dates = pivot["date"].to_list()
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = ["#2E86AB", "#A23B72", "#F18F01", "#3B8E5C", "#6B5B95", "#C73E1D", "#7E8083"]
    for i, sec in enumerate(sectors):
        if sec not in pivot.columns:
            continue
        ax.plot(dates, pivot[sec].to_numpy(), label=sec[:20], color=cmap[i % len(cmap)], lw=1.0)
    if cluster_cap is not None and cluster_cap > 0:
        ax.axhline(
            cluster_cap,
            color="#C73E1D",
            lw=1.0,
            ls="--",
            label=f"cluster cap = {cluster_cap * 100:.0f}%",
        )
    if highlight_period is not None:
        ax.axvspan(highlight_period[0], highlight_period[1], color="#E8E8E8", alpha=0.5)
    ax.set_ylabel("exposure (fraction of capital)")
    ax.set_xlabel("date")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"n_clusters": len(sectors)}


def plot_carriers_r0_vs_r4(
    pair_curves: dict[str, dict],
    out_path: Path,
    *,
    title: str = "Carriers (PFC/SBIN, ONGC/OIL) — R0 vs R4 cumulative net P&L",
) -> dict:
    """Per-pair cumulative net log-return: R0 (equal-notional) vs R4 (full RM).

    ``pair_curves`` maps pair_key → {"R0": pl.DataFrame, "R4": pl.DataFrame}.
    Each DataFrame has columns ``date`` and ``net_log_ret``.
    """
    apply_style()
    n = len(pair_curves)
    if n == 0:
        return {"n_pairs": 0}
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    axes = axes.flatten()
    for i, (pkey, curves) in enumerate(pair_curves.items()):
        ax = axes[i]
        for rung_label, df in curves.items():
            if df.is_empty():
                continue
            d_sorted = df.sort("date")
            dates = d_sorted["date"].to_list()
            cum_pct = (np.expm1(np.cumsum(d_sorted["net_log_ret"].to_numpy()))) * 100
            color = _RUNG_COLORS.get(rung_label, "#7E8083")
            ax.plot(dates, cum_pct, label=rung_label, color=color, lw=1.2)
        ax.axhline(0, color="#7E8083", lw=0.5, ls=":")
        ax.set_title(pkey, fontsize=11, fontweight="bold")
        ax.set_ylabel("cum net P&L (%)")
        ax.set_xlabel("date")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"n_pairs": n}


def plot_cluster_cap_sweep(
    sweep_results: dict[str, dict],
    out_path: Path,
    *,
    title: str = "Cluster-cap sweep — equity curve + max drawdown",
) -> dict:
    """Two-panel sweep plot. ``sweep_results`` maps arm label →
    {"portfolio_daily": df, "metrics": {"max_drawdown_pct": ..., "ann_return_pct": ...}}.
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[1.3, 1.0])
    colors = ["#3B8E5C", "#2E86AB", "#A23B72", "#F18F01"]
    for i, (label, payload) in enumerate(sweep_results.items()):
        df = payload["portfolio_daily"]
        if df.is_empty():
            continue
        d_sorted = df.sort("date")
        dates = d_sorted["date"].to_list()
        cum_pct = (np.expm1(np.cumsum(d_sorted["net_log_ret"].to_numpy()))) * 100
        c = colors[i % len(colors)]
        ax1.plot(dates, cum_pct, color=c, lw=1.2, label=label)
        cum_log = np.cumsum(d_sorted["net_log_ret"].to_numpy())
        dd_pct = (np.expm1(cum_log - np.maximum.accumulate(cum_log))) * 100
        ax2.plot(dates, dd_pct, color=c, lw=1.0, label=label)
    ax1.axhline(0, color="#7E8083", lw=0.5, ls=":")
    ax1.set_ylabel("cumulative net (%)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax2.axhline(0, color="#7E8083", lw=0.5, ls=":")
    ax2.set_ylabel("drawdown (%)")
    ax2.set_xlabel("date")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"n_arms": len(sweep_results)}
