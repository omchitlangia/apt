"""Pair-level plots (correlation heatmaps, spread series, etc.).

Day-5 deliverable: a sector-clustered correlation heatmap visualising the
single-window screen output. Future days (cointegration, spread plots) add
to this module.
"""

from __future__ import annotations

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

    sym_to_sector = dict(
        zip(sectors["symbol"], sectors["industry"], strict=True)
    )
    decorated = [
        (sym_to_sector.get(s, "UNKNOWN"), s) for s in symbols
    ]
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
    ax.set_yticklabels(
        [s if s else "" for _, s in midpoints], fontsize=7
    )
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
