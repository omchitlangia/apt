"""Phase-3 plotting helpers — Phase-2 visual style adapted to intraday outputs.

Two reusable plot functions:

* :func:`plot_phase3_per_pair_card` — one figure per distinct tradeable pair.
  Panel 1 (top): cumulative NET equity over the full minute span at 3 bps,
  Regime A vs B on the same axes, with fold-boundary markers + idle
  stretches (where the pair was not selected) visible.
  Panel 2 (bottom): one representative fold's minute spread + rolling z
  with entry / exit / stop bands and trade entry/exit markers overlaid.

* :func:`plot_phase3_portfolio_equity` — three-panel Phase-2-style overall
  portfolio chart: cumulative gross + net per regime + variant, drawdown
  on net, and active-pair count. Used to refresh the full-span plot.

Both helpers take pre-aggregated arrays/Frames (no recompute of trades or
P&L). Spread + z for the per-pair lower panel ARE recomputed
deterministically from the loaded minute prices and the frozen (alpha,
beta) — that is plotting, not strategy logic.
"""

from __future__ import annotations

import contextlib
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apt.plots.style import APT_PALETTE, apply_style

# Marker / colour scheme (intraday taxonomy after rename in the v2 trade CSV)
_EXIT_MARKERS: dict[str, tuple[str, str, str]] = {
    "mean_revert": ("v", "#2E86AB", "mean-revert exit"),
    "z_stop": ("X", "#C73E1D", "z-stop"),
    "time_stop": ("s", "#A23B72", "time-stop"),
    "eod_squareoff": ("D", "#F18F01", "eod squareoff (Regime A)"),
    "fold_close": ("P", "#6B5B95", "fold-close"),
}

_REGIME_COLOR: dict[str, str] = {"A": "#C73E1D", "B": "#2E86AB"}


# ---------------------------------------------------------------------------
# Per-pair card
# ---------------------------------------------------------------------------


def plot_phase3_per_pair_card(
    *,
    pair_key: str,
    sector: str,
    is_structural: bool,
    is_hdfcbank_anchored: bool,
    daily_dates: list[date],
    cum_net_A_pct: np.ndarray,
    cum_net_B_pct: np.ndarray,
    fold_spans: list[tuple[int, date, date]],
    rep_fold_id: int | None,
    rep_timestamps: pd.DatetimeIndex | None,
    rep_spread: np.ndarray | None,
    rep_z: np.ndarray | None,
    trades_in_rep_fold_A: pd.DataFrame | None,
    trades_in_rep_fold_B: pd.DataFrame | None,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    out_path: Path,
    stats_line: str = "",
) -> dict:
    """Two-panel diagnostic card for one distinct intraday pair.

    Parameters are pre-computed by the caller; this function only draws.
    A ``rep_fold_id`` of ``None`` skips the bottom panel and emits a
    full-figure equity-only card (useful when no fold's minute data is
    available).
    """
    apply_style()

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8.5), height_ratios=[1.2, 1.0], gridspec_kw={"hspace": 0.32}
    )
    ax_eq, ax_sz = axes

    # ---- Top: full-span cumulative NET equity, A vs B -------------------
    ax_eq.plot(
        daily_dates,
        cum_net_A_pct,
        color=_REGIME_COLOR["A"],
        lw=1.1,
        label="Regime A (intraday-only)",
    )
    ax_eq.plot(
        daily_dates,
        cum_net_B_pct,
        color=_REGIME_COLOR["B"],
        lw=1.1,
        label="Regime B (carry)",
    )
    ax_eq.axhline(0, color="#7E8083", lw=0.5, ls=":")
    # Fold-boundary markings: shade the pair's active fold windows
    for fid, s, e in fold_spans:
        ax_eq.axvspan(s, e, color="#3B8E5C", alpha=0.06)
        ax_eq.axvline(s, color="#3B8E5C", lw=0.4, ls="--", alpha=0.5)
        ax_eq.axvline(e, color="#3B8E5C", lw=0.4, ls="--", alpha=0.5)
        # Fold number label at the top — matplotlib silently no-ops on
        # any axis edge case (e.g., empty axis limits), but suppress
        # belt-and-braces to keep the loop robust across pairs.
        mid = s + (e - s) / 2
        with contextlib.suppress(Exception):
            ax_eq.annotate(
                f"f{fid}",
                xy=(mid, ax_eq.get_ylim()[1]),
                xytext=(0, -2),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                color="#3B8E5C",
            )
    ax_eq.set_ylabel("cumulative NET P&L (%) @ 3 bps")
    ax_eq.legend(loc="upper left", fontsize=8)

    # ---- Bottom: representative fold's minute spread + z + markers ------
    if rep_fold_id is None or rep_timestamps is None or rep_spread is None or rep_z is None:
        ax_sz.text(
            0.5,
            0.5,
            "no representative fold available",
            ha="center",
            va="center",
            transform=ax_sz.transAxes,
        )
        ax_sz.axis("off")
    else:
        ax_sz2 = ax_sz.twinx()
        # Spread (left y-axis)
        ax_sz.plot(rep_timestamps, rep_spread, color="#2E86AB", lw=0.4, label="spread", alpha=0.8)
        ax_sz.set_ylabel("spread (log)", color="#2E86AB")
        ax_sz.tick_params(axis="y", labelcolor="#2E86AB")
        # Z (right y-axis)
        ax_sz2.plot(rep_timestamps, rep_z, color="#A23B72", lw=0.3, alpha=0.8, label="rolling z")
        ax_sz2.axhline(+entry_z, color="#C73E1D", lw=0.5, ls="--", alpha=0.7)
        ax_sz2.axhline(-entry_z, color="#C73E1D", lw=0.5, ls="--", alpha=0.7)
        ax_sz2.axhline(+exit_z, color="#3B8E5C", lw=0.5, ls=":", alpha=0.7)
        ax_sz2.axhline(-exit_z, color="#3B8E5C", lw=0.5, ls=":", alpha=0.7)
        ax_sz2.axhline(+stop_z, color="#6B5B95", lw=0.5, ls="-.", alpha=0.7)
        ax_sz2.axhline(-stop_z, color="#6B5B95", lw=0.5, ls="-.", alpha=0.7)
        ax_sz2.set_ylabel("rolling z", color="#A23B72")
        ax_sz2.tick_params(axis="y", labelcolor="#A23B72")

        finite_z = rep_z[np.isfinite(rep_z)]
        if finite_z.size:
            zlim = max(stop_z * 1.05, float(np.nanmax(np.abs(finite_z))) * 1.05)
            ax_sz2.set_ylim(-zlim, +zlim)

        # Trade markers: on the spread series
        ts_index = pd.DatetimeIndex(rep_timestamps)
        spread_lookup = pd.Series(rep_spread, index=ts_index)

        def _scatter_markers(trades_df: pd.DataFrame, edge: str, regime_tag: str) -> None:
            for row in trades_df.itertuples():
                # Match exit timestamp to the nearest bar in the fold panel
                try:
                    entry = pd.Timestamp(row.entry_ts)
                    exit_ = pd.Timestamp(row.exit_ts)
                except Exception:  # noqa: BLE001
                    continue
                if entry not in spread_lookup.index or exit_ not in spread_lookup.index:
                    continue
                direction = +1 if row.side == "long_spread" else -1
                entry_marker = "^" if direction > 0 else "v"
                entry_color = "#3B8E5C" if direction > 0 else "#C73E1D"
                ax_sz.scatter(
                    [entry],
                    [spread_lookup.loc[entry]],
                    marker=entry_marker,
                    facecolor=entry_color,
                    edgecolor=edge,
                    s=22,
                    linewidths=0.4,
                    zorder=5,
                    label=f"_{regime_tag}_entry",
                )
                exit_info = _EXIT_MARKERS.get(row.exit_reason)
                if exit_info is None:
                    continue
                shape, mcolor, _label = exit_info
                ax_sz.scatter(
                    [exit_],
                    [spread_lookup.loc[exit_]],
                    marker=shape,
                    facecolor=mcolor,
                    edgecolor=edge,
                    s=26,
                    linewidths=0.4,
                    zorder=5,
                    label=f"_{regime_tag}_exit",
                )

        if trades_in_rep_fold_A is not None and not trades_in_rep_fold_A.empty:
            _scatter_markers(trades_in_rep_fold_A, edge="black", regime_tag="A")
        if trades_in_rep_fold_B is not None and not trades_in_rep_fold_B.empty:
            _scatter_markers(trades_in_rep_fold_B, edge="white", regime_tag="B")

        # Bottom-panel title (small)
        ax_sz.set_title(
            f"Fold {rep_fold_id} spread + rolling z  "
            f"(entry ±{entry_z:g}, exit ±{exit_z:g}, stop ±{stop_z:g}; "
            "Regime A entries: black-edged; Regime B entries: white-edged)",
            fontsize=9,
            fontweight="normal",
        )
        ax_sz.set_xlabel("session timestamp (IST)")

    # ---- Top-level title ------------------------------------------------
    tags: list[str] = []
    if is_structural:
        tags.append("STRUCTURAL")
    if is_hdfcbank_anchored:
        tags.append("HDFCBANK-anchored")
    tag_str = "  ".join(f"[{t}]" for t in tags)
    title = f"{pair_key}    [{sector}]"
    if tag_str:
        title = f"{title}    {tag_str}"
    if stats_line:
        title = f"{title}\n{stats_line}"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "pair": pair_key,
        "rep_fold_id": rep_fold_id,
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Overall portfolio equity (Phase-2 style adapted)
# ---------------------------------------------------------------------------


def plot_phase3_portfolio_equity(
    portfolio_daily: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Phase 3 portfolio — gross + net, Regime A vs B (3 bps)",
) -> dict:
    """Three-panel Phase-2-style equity chart for the v2 portfolio.

    ``portfolio_daily`` must have columns:
        date, A_gross, A_net, B_gross, B_net,
              A_gross_ex, A_net_ex, B_gross_ex, B_net_ex,
              n_active_pairs   (optional)

    Suffix ``_ex`` denotes the ex-HDFC/HDFCBANK variant. Missing columns
    are skipped silently. The third panel is omitted if
    ``n_active_pairs`` is not present.
    """
    apply_style()
    if portfolio_daily.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no portfolio data", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return {"n_obs": 0}

    df = portfolio_daily.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).tolist()

    def _cum_pct(col: str) -> np.ndarray | None:
        if col not in df.columns:
            return None
        arr = df[col].to_numpy(dtype=float)
        return (np.expm1(np.cumsum(arr))) * 100

    n_panels = 3 if "n_active_pairs" in df.columns else 2
    heights = [2.4, 1.0, 0.8] if n_panels == 3 else [2.4, 1.0]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12, 4.5 + 2.0 * n_panels), sharex=True, height_ratios=heights
    )
    ax_eq = axes[0]
    ax_dd = axes[1]
    ax_act = axes[2] if n_panels == 3 else None

    # Equity panel — solid = all, dashed = ex HDFC/HDFCBANK
    plotted_any = False
    for regime, ls in (("A", "-"), ("B", "-")):
        for variant_suffix, ls_inner, label_suffix in (
            ("", ls, "all"),
            ("_ex", "--" if ls == "-" else ls, "ex HDFC/HDFCBANK"),
        ):
            for series_kind, lw in (("gross", 1.4), ("net", 1.0)):
                col = f"{regime}_{series_kind}{variant_suffix}"
                cum = _cum_pct(col)
                if cum is None:
                    continue
                style = "-" if series_kind == "gross" else ls_inner
                ax_eq.plot(
                    dates,
                    cum,
                    color=_REGIME_COLOR[regime],
                    lw=lw,
                    linestyle=style,
                    alpha=0.95 if series_kind == "gross" else 0.85,
                    label=f"Regime {regime} {series_kind} ({label_suffix})",
                )
                plotted_any = True
    if plotted_any:
        ax_eq.axhline(0, color="#7E8083", lw=0.5, ls=":")
        ax_eq.set_ylabel("cumulative return (%)")
        ax_eq.set_title(title, fontsize=12, fontweight="bold")
        ax_eq.legend(loc="best", fontsize=7, ncol=2)

    # Drawdown panel — on NET, both regimes, full sample
    for regime, color in (("A", _REGIME_COLOR["A"]), ("B", _REGIME_COLOR["B"])):
        col = f"{regime}_net"
        if col not in df.columns:
            continue
        cum_log = np.cumsum(df[col].to_numpy(dtype=float))
        rm = np.maximum.accumulate(cum_log)
        dd_pct = (np.expm1(cum_log - rm)) * 100
        ax_dd.fill_between(dates, dd_pct, 0, color=color, alpha=0.25, step="pre")
        ax_dd.plot(dates, dd_pct, color=color, lw=0.7, label=f"Regime {regime} net DD")
    ax_dd.axhline(0, color="#7E8083", lw=0.5, ls=":")
    ax_dd.set_ylabel("drawdown (%) — net")
    ax_dd.legend(loc="lower left", fontsize=8)

    if ax_act is not None:
        ax_act.fill_between(
            dates,
            df["n_active_pairs"].to_numpy(dtype=float),
            0,
            color="#6B5B95",
            alpha=0.35,
            step="pre",
        )
        ax_act.plot(dates, df["n_active_pairs"], color="#6B5B95", lw=0.6)
        ax_act.set_ylabel("active pairs")
        ax_act.set_xlabel("date")
        ax_act.set_ylim(bottom=0)
    else:
        ax_dd.set_xlabel("date")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {"n_obs": int(df.shape[0])}


__all__ = [
    "plot_phase3_per_pair_card",
    "plot_phase3_portfolio_equity",
    "APT_PALETTE",
]
