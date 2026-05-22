"""Universe-EDA plots for Phase 1 Day 4B.

Each function takes the cleaned daily frame (plus sectors where needed)
and a destination ``Path``; writes a PNG and returns a tiny dict of stats
so the calling script can flag anomalies in its console summary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from apt.plots.style import APT_PALETTE, apply_style

# Sectors with fewer than this many surviving symbols are too thin for
# within-sector pairs trading.
THIN_SECTOR_FLOOR: int = 5


def plot_symbols_per_sector(
    daily: pl.DataFrame,
    sectors: pl.DataFrame,
    out_path: Path,
    *,
    thin_floor: int = THIN_SECTOR_FLOOR,
) -> dict:
    """Bar chart: symbol count per sector among the cleaned universe."""
    apply_style()
    syms = daily["symbol"].unique().to_list()
    sec = (
        sectors.filter(pl.col("symbol").is_in(syms))
        .group_by("industry")
        .len()
        .rename({"len": "n_symbols"})
        .sort("n_symbols", descending=True)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * sec.height)))
    industries = sec["industry"].to_list()
    counts = sec["n_symbols"].to_list()
    colors = [
        APT_PALETTE[3] if c < thin_floor else APT_PALETTE[0] for c in counts
    ]
    bars = ax.barh(industries, counts, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("symbols in cleaned universe")
    ax.set_title(f"Symbols per sector  (n={len(syms)}, thin <{thin_floor} in red)")
    for rect, c in zip(bars, counts, strict=True):
        ax.text(
            rect.get_width() + 0.4,
            rect.get_y() + rect.get_height() / 2,
            str(c),
            va="center",
            fontsize=8,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    thin = sec.filter(pl.col("n_symbols") < thin_floor).to_dicts()
    return {
        "n_sectors": sec.height,
        "n_thin_sectors": len(thin),
        "thin_sectors": thin,
        "median_per_sector": int(sec["n_symbols"].median() or 0),
    }


def plot_history_length_distribution(
    daily: pl.DataFrame,
    out_path: Path,
    *,
    min_days_floor: int = 756,
) -> dict:
    """Histogram of per-symbol row counts with the 756-day floor marked."""
    apply_style()
    counts = daily.group_by("symbol").len().rename({"len": "n_days"})
    nd = np.array(counts["n_days"].to_list())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        nd, bins=40, color=APT_PALETTE[0], edgecolor="#1F5775", alpha=0.85
    )
    ax.axvline(min_days_floor, color=APT_PALETTE[3], lw=2, ls="--",
               label="756-day floor (Rule 6)")
    ax.set_xlabel("trading days per symbol")
    ax.set_ylabel("symbols")
    ax.set_title(
        f"History-length distribution  (n={len(nd)}, "
        f"median {int(np.median(nd))}, min {int(nd.min())}, max {int(nd.max())})"
    )
    ax.legend(loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "n_symbols": len(nd),
        "median_days": int(np.median(nd)),
        "min_days": int(nd.min()),
        "max_days": int(nd.max()),
        "n_at_or_below_900": int((nd <= 900).sum()),
    }


def plot_coverage_heatmap(daily: pl.DataFrame, out_path: Path) -> dict:
    """Heatmap of (symbol × year) trading-day counts.

    Symbols are sorted by first appearance ascending so the staggered
    start dates and Rule-7 internal trims are visible at a glance.
    """
    apply_style()
    enriched = daily.with_columns(pl.col("date").dt.year().alias("year"))
    counts = (
        enriched.group_by(["symbol", "year"]).len().rename({"len": "n_days"})
    )
    first_year = (
        enriched.group_by("symbol").agg(pl.col("date").min().alias("first_d"))
    )
    syms = (
        first_year.sort("first_d")["symbol"].to_list()
    )
    years = sorted(set(counts["year"].to_list()))
    sym_to_idx = {s: i for i, s in enumerate(syms)}
    yr_to_idx = {y: i for i, y in enumerate(years)}

    grid = np.zeros((len(syms), len(years)), dtype=float)
    for row in counts.to_dicts():
        grid[sym_to_idx[row["symbol"]], yr_to_idx[row["year"]]] = row["n_days"]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.025 * len(syms))))
    im = ax.imshow(grid, aspect="auto", cmap="viridis",
                   interpolation="nearest", vmax=255)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("year")
    ax.set_ylabel(f"symbols  (n={len(syms)}, sorted by first trading date)")
    ax.set_title(
        "Coverage heatmap: trading days per symbol per year "
        "(viridis: dark=missing, bright=full)"
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("trading days in year")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "n_symbols": len(syms),
        "n_years": len(years),
        "year_min": min(years),
        "year_max": max(years),
        "n_symbols_starting_2003_04": int(
            first_year.filter(pl.col("first_d").dt.year() <= 2004).height
        ),
        "n_symbols_starting_2015_plus": int(
            first_year.filter(pl.col("first_d").dt.year() >= 2015).height
        ),
    }


def plot_return_distributions(
    daily: pl.DataFrame,
    out_path: Path,
    *,
    sample_symbols: tuple[str, ...] = (
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ITC",
    ),
    extreme_threshold: float = 0.20,
) -> dict:
    """Aggregate log-return histogram + a few named symbols overlaid.

    Returns the count of returns whose absolute log-return exceeds
    ``extreme_threshold`` — should be small and concentrated around
    known KEEP events / dividend ex-dates.
    """
    apply_style()
    rets = (
        daily.sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol"))
            .log()
            .alias("logret")
        )
        .filter(pl.col("logret").is_not_null())
    )
    agg = np.array(rets["logret"].to_list())

    fig, (ax_agg, ax_sample) = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    # Left: aggregate
    bins = np.linspace(-0.5, 0.5, 121)
    ax_agg.hist(agg, bins=bins, color=APT_PALETTE[0],
                edgecolor="#1F5775", alpha=0.85)
    ax_agg.set_yscale("log")
    ax_agg.set_xlabel("daily log-return")
    ax_agg.set_ylabel("count (log scale)")
    ax_agg.set_title(
        f"Aggregate log-return histogram  (n={len(agg):,}, "
        f"|σ|≈{agg.std():.4f})"
    )
    ax_agg.axvline(-0.40, color=APT_PALETTE[3], lw=1.2, ls="--",
                   label="±0.40 gate threshold")
    ax_agg.axvline(0.40, color=APT_PALETTE[3], lw=1.2, ls="--")
    ax_agg.legend(loc="upper right")

    # Right: per-symbol density-style overlay
    for i, sym in enumerate(sample_symbols):
        sub = rets.filter(pl.col("symbol") == sym)["logret"].to_list()
        if not sub:
            continue
        sub = np.array(sub)
        # Smooth-ish histogram drawn as line (density)
        h, edges = np.histogram(sub, bins=60, range=(-0.2, 0.2), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax_sample.plot(
            centers, h, lw=1.5,
            color=APT_PALETTE[i % len(APT_PALETTE)],
            label=f"{sym}  (n={len(sub):,})",
        )
    ax_sample.set_xlabel("daily log-return")
    ax_sample.set_ylabel("density")
    ax_sample.set_title("Selected names")
    ax_sample.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    n_extreme = int((np.abs(agg) > extreme_threshold).sum())
    extreme_rows = (
        rets.filter(pl.col("logret").abs() > extreme_threshold)
        .sort(pl.col("logret").abs(), descending=True)
        .head(5)
        .to_dicts()
    )
    return {
        "n_rets": len(agg),
        "std": float(agg.std()),
        "min": float(agg.min()),
        "max": float(agg.max()),
        f"n_abs_gt_{extreme_threshold}": n_extreme,
        "top5_extreme": extreme_rows,
    }


def plot_adv_distribution(
    daily: pl.DataFrame,
    out_path: Path,
    *,
    window: int = 60,
    min_periods: int = 20,
    floor_inr: float = 10_000_000.0,
) -> dict:
    """Histogram (log-x) of each symbol's median 60-day rolling-median ADV.

    For each symbol: compute the rolling-window median of (close × volume),
    then take the median of that rolling series. One scalar per symbol.
    """
    apply_style()
    per_sym = (
        daily.sort(["symbol", "date"])
        .with_columns((pl.col("close") * pl.col("volume")).alias("_adv"))
        .with_columns(
            pl.col("_adv")
            .rolling_median(window_size=window, min_samples=min_periods)
            .over("symbol")
            .alias("_adv_roll")
        )
        .filter(pl.col("_adv_roll").is_not_null())
        .group_by("symbol")
        .agg(pl.col("_adv_roll").median().alias("median_rolling_adv"))
    )
    vals = np.array(per_sym["median_rolling_adv"].to_list())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(6, 11, 40)
    ax.hist(vals, bins=bins, color=APT_PALETTE[0],
            edgecolor="#1F5775", alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(floor_inr, color=APT_PALETTE[3], lw=2, ls="--",
               label=f"₹{floor_inr/1e7:.0f} crore floor (Rule 5)")
    ax.set_xlabel("median 60-day rolling-median ADV (INR, log)")
    ax.set_ylabel("symbols")
    ax.set_title(
        f"Per-symbol median ADV distribution  (n={len(vals)}, "
        f"median ₹{np.median(vals)/1e7:.1f} cr)"
    )
    ax.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "n_symbols": len(vals),
        "median_inr": float(np.median(vals)),
        "p10_inr": float(np.quantile(vals, 0.10)),
        "p90_inr": float(np.quantile(vals, 0.90)),
        "n_below_floor": int((vals < floor_inr).sum()),
    }
