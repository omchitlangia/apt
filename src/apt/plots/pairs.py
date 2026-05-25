"""Pair-level plots (correlation heatmaps, spread series, etc.).

Day-5 added the sector-clustered correlation heatmap; Day-6 adds a spread
diagnostic plot used to eyeball the most promising cointegrated pairs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from apt.plots.style import apply_style


def plot_sector_clustered_correlation(
    corr_matrix: np.ndarray,
    symbols: list[str],
    sectors: pl.DataFrame,
    out_path: Path,
    *,
    title: str = "Same-sector correlation matrix",
) -> dict:
    """Draw a sector-clustered correlation heatmap to ``out_path``.

    Symbols are reordered so all members of a sector are adjacent, with
    thin vertical/horizontal lines separating sectors so within-sector
    blocks are visually obvious.
    """
    apply_style()
    if len(symbols) == 0 or corr_matrix.size == 0:
        # Still write a placeholder PNG so the script's contract holds.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no eligible symbols", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return {"n_symbols": 0, "n_sectors": 0}

    sym_to_sector = dict(zip(sectors["symbol"], sectors["industry"], strict=True))
    decorated = [(sym_to_sector.get(s, "UNKNOWN"), s) for s in symbols]
    # Sort by (sector, symbol)
    order = sorted(range(len(symbols)), key=lambda i: decorated[i])
    perm = np.array(order)
    ordered_secs = [decorated[i][0] for i in order]

    M = corr_matrix[np.ix_(perm, perm)]

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(
        M,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(f"{title}  (n={len(symbols)} symbols, sector-clustered)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw sector boundaries
    boundaries: list[int] = []
    midpoints: list[tuple[int, str]] = []
    last = ordered_secs[0]
    seg_start = 0
    for i, s in enumerate(ordered_secs + [None]):
        if s != last or i == len(ordered_secs):
            mid = (seg_start + i - 1) // 2
            midpoints.append((mid, last))
            if i < len(ordered_secs):
                boundaries.append(i - 0.5)
            seg_start = i
            last = s
    for b in boundaries:
        ax.axhline(b, color="black", lw=0.6, alpha=0.5)
        ax.axvline(b, color="black", lw=0.6, alpha=0.5)

    # Label sectors along the y-axis at their midpoints
    ax.set_yticks([m for m, _ in midpoints])
    ax.set_yticklabels([s if s else "" for _, s in midpoints], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Pearson correlation of daily log-returns")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "n_symbols": len(symbols),
        "n_sectors": len(set(ordered_secs)),
    }


def plot_pair_spread(
    daily: pl.DataFrame,
    *,
    y_sym: str,
    x_sym: str,
    alpha: float,
    beta: float,
    start: date,
    end: date,
    out_path: Path,
    bands: tuple[float, ...] = (1.0, 2.0),
    sector: str | None = None,
    extra_title: str | None = None,
) -> dict:
    """Plot the OLS residual spread for one cointegrated pair to ``out_path``.

    Spread is computed as ``log(close[y_sym]) - alpha - beta * log(close[x_sym])``
    over ``[start, end]``, aligned by date. ±k·σ bands (with σ the in-window
    standard deviation of the spread) are drawn for each ``k`` in ``bands``.
    """
    apply_style()

    win = (
        daily.filter(
            pl.col("symbol").is_in([y_sym, x_sym])
            & (pl.col("date") >= start)
            & (pl.col("date") <= end)
        )
        .select(["symbol", "date", "close"])
        .with_columns(pl.col("close").log().alias("log_close"))
    )
    wide = win.pivot(index="date", on="symbol", values="log_close").sort("date").drop_nulls()
    if wide.is_empty() or y_sym not in wide.columns or x_sym not in wide.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, f"no data for {y_sym}/{x_sym}", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return {"y_sym": y_sym, "x_sym": x_sym, "n_obs": 0}

    dates = wide["date"].to_list()
    y = wide[y_sym].to_numpy()
    x = wide[x_sym].to_numpy()
    spread = y - alpha - beta * x
    mu = float(np.mean(spread))
    sigma = float(np.std(spread))

    title_main = f"Spread  {y_sym} = α + β·{x_sym} + ε   (β={beta:+.4f})"
    if sector:
        title_main = f"{title_main}   [{sector}]"
    if extra_title:
        title_main = f"{title_main}\n{extra_title}"

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, spread, lw=0.9, color="#2E86AB", label="spread")
    ax.axhline(mu, color="#7E8083", lw=0.8, ls="--", label=f"mean = {mu:+.4f}")
    band_colors = ["#3B8E5C", "#C73E1D", "#A23B72", "#6B5B95"]
    for i, k in enumerate(bands):
        c = band_colors[i % len(band_colors)]
        ax.axhline(mu + k * sigma, color=c, lw=0.7, ls=":", label=f"±{k:g}σ")
        ax.axhline(mu - k * sigma, color=c, lw=0.7, ls=":")
    ax.set_xlim(dates[0], dates[-1])
    ax.set_title(title_main)
    ax.set_xlabel("date")
    ax.set_ylabel("residual spread (log scale)")
    ax.legend(loc="upper left", ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "y_sym": y_sym,
        "x_sym": x_sym,
        "n_obs": len(spread),
        "mean": mu,
        "sigma": sigma,
    }


def plot_spread_zscore_signal(
    dates: list,
    spread: np.ndarray,
    z: np.ndarray,
    position: np.ndarray,
    *,
    y_sym: str,
    x_sym: str,
    out_path: Path,
    entry: float = 2.0,
    exit_threshold: float = 0.5,
    stop: float = 3.5,
    sector: str | None = None,
    extra_title: str | None = None,
) -> dict:
    """Two-panel plot: spread (top) + rolling z-score with bands + signal shading (bottom).

    ``position`` is a length-N array in ``{-1, 0, +1}`` produced by
    :func:`apt.signals.spread.generate_signals`. Long-spread bars are
    shaded blue, short-spread bars red, on the z-score panel.
    """
    apply_style()
    spread = np.asarray(spread, dtype=float)
    z = np.asarray(z, dtype=float)
    position = np.asarray(position, dtype=np.int8)

    fig, (ax_s, ax_z) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[1.0, 1.3])
    # ---- top: spread ----
    ax_s.plot(dates, spread, color="#2E86AB", lw=0.9, label="spread")
    if np.isfinite(spread).any():
        mu_full = float(np.nanmean(spread))
        ax_s.axhline(mu_full, color="#7E8083", lw=0.7, ls="--", label="full-window mean")
    ax_s.set_ylabel("spread (log)")
    ax_s.legend(loc="upper left", fontsize=8)

    # ---- bottom: z-score + bands + position shading ----
    ax_z.plot(dates, z, color="#2E86AB", lw=0.8, label="rolling z")
    ax_z.axhline(0, color="#7E8083", lw=0.6, ls=":")
    for k, c, label in (
        (entry, "#C73E1D", f"±{entry:g} entry"),
        (exit_threshold, "#3B8E5C", f"±{exit_threshold:g} exit"),
        (stop, "#A23B72", f"±{stop:g} stop"),
    ):
        ax_z.axhline(+k, color=c, lw=0.6, ls="--", label=label)
        ax_z.axhline(-k, color=c, lw=0.6, ls="--")

    finite_z = z[np.isfinite(z)]
    if finite_z.size:
        zmax = max(float(np.max(finite_z)) * 1.05, stop * 1.05)
        zmin = min(float(np.min(finite_z)) * 1.05, -stop * 1.05)
        ax_z.set_ylim(zmin, zmax)
    longs = position == 1
    shorts = position == -1
    if longs.any():
        ax_z.fill_between(
            dates, *ax_z.get_ylim(), where=longs, color="#2E86AB", alpha=0.13, step="pre"
        )
    if shorts.any():
        ax_z.fill_between(
            dates, *ax_z.get_ylim(), where=shorts, color="#C73E1D", alpha=0.13, step="pre"
        )
    ax_z.set_ylabel("rolling z-score")
    ax_z.set_xlabel("date")
    ax_z.legend(loc="upper left", ncol=4, fontsize=8)

    title = f"Spread + z-signal:  {y_sym} − β·{x_sym}"
    if sector:
        title = f"{title}   [{sector}]"
    if extra_title:
        title = f"{title}\n{extra_title}"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "y_sym": y_sym,
        "x_sym": x_sym,
        "n_obs": int(spread.size),
        "n_long_bars": int(longs.sum()),
        "n_short_bars": int(shorts.sum()),
    }
