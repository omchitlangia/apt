"""Fold-aware correlation pair screening.

The defining requirement: **never** compute a single global correlation
matrix on "the last N days" and freeze the survivor list — that leaks
future information into early walk-forward folds. The screen is a windowed
function instead. Phase 2 calls it per fold on each fold's *training*
window only.

Public API:
    * :func:`window_eligible_symbols` — symbols whose contiguous history
      covers ``[start, end]`` with no internal gap > the threshold.
    * :func:`compute_window_correlation` — Pearson correlation of daily
      log-returns over ``[start, end]`` for those symbols; returns the
      matrix, the symbol ordering, and dataframe diagnostics.
    * :func:`screen_pairs` — the windowed pair list. Returns
      ``(sym1, sym2, corr, sector)`` for same-sector pairs above the
      correlation threshold.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from loguru import logger


@dataclass(frozen=True)
class WindowCorrelation:
    """Output of :func:`compute_window_correlation`."""

    start: date
    end: date
    eligible_symbols: list[str]
    corr_matrix: np.ndarray  # shape (n_eligible, n_eligible), float64
    n_window_days: int  # number of distinct trading days in [start, end]
    n_used_dates: int   # number of dates surviving the intersection drop_nulls


# ---------------------------------------------------------------------------
# Step 1 — window-level eligibility (the gap guard)
# ---------------------------------------------------------------------------


def window_eligible_symbols(
    daily: pl.DataFrame,
    *,
    start: date,
    end: date,
    max_internal_gap_days: int = 10,
) -> list[str]:
    """Symbols whose history fully covers ``[start, end]`` contiguously.

    A symbol passes iff:
      * it has a row on ``start`` (the exact left boundary trading day);
      * it has a row on ``end`` (the exact right boundary trading day);
      * the max consecutive-row gap within ``[start, end]`` is
        ``<= max_internal_gap_days`` calendar days.

    ``start`` and ``end`` are themselves expected to be trading days
    (the caller usually derives them from the universe's trading-day list).
    """
    win = daily.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    if win.is_empty():
        return []

    win_sorted = win.sort(["symbol", "date"]).with_columns(
        pl.col("date")
        .diff()
        .over("symbol")
        .dt.total_days()
        .fill_null(0)
        .alias("_gap")
    )
    per_sym = win_sorted.group_by("symbol").agg(
        [
            (pl.col("date") == start).any().alias("has_start"),
            (pl.col("date") == end).any().alias("has_end"),
            pl.col("_gap").max().alias("max_gap"),
            pl.len().alias("n_rows"),
        ]
    )
    eligible = per_sym.filter(
        pl.col("has_start")
        & pl.col("has_end")
        & (pl.col("max_gap") <= max_internal_gap_days)
    )
    return sorted(eligible["symbol"].to_list())


# ---------------------------------------------------------------------------
# Step 2 — windowed correlation matrix
# ---------------------------------------------------------------------------


def compute_window_correlation(
    daily: pl.DataFrame,
    *,
    start: date,
    end: date,
    max_internal_gap_days: int = 10,
    symbols: Iterable[str] | None = None,
) -> WindowCorrelation:
    """Pearson correlation matrix of daily log-returns over ``[start, end]``.

    Pipeline:
      1. Determine eligible symbols via :func:`window_eligible_symbols`
         (unless ``symbols`` is supplied, in which case those are used and
         the gap guard is *also* applied for safety).
      2. Compute log-returns within the window per symbol.
      3. Pivot to a (date × symbol) wide frame and drop any dates with a
         missing observation (intersection on dates). This guarantees a
         dense matrix and a clean Pearson computation.
      4. ``np.corrcoef`` on the column-wise data → returns the symmetric
         ``(n, n)`` correlation matrix in the order of ``eligible_symbols``.
    """
    if symbols is None:
        eligible = window_eligible_symbols(
            daily,
            start=start,
            end=end,
            max_internal_gap_days=max_internal_gap_days,
        )
    else:
        # Verify the requested symbols pass the gap guard too — fail loud
        # rather than silently returning a stale/leaky matrix.
        requested = set(symbols)
        passes = set(
            window_eligible_symbols(
                daily,
                start=start,
                end=end,
                max_internal_gap_days=max_internal_gap_days,
            )
        )
        missing = sorted(requested - passes)
        if missing:
            raise ValueError(
                f"Requested {len(missing)} symbol(s) that fail the window "
                f"gap guard for [{start}, {end}]: {missing[:5]}"
            )
        eligible = sorted(requested)

    n_window_days = (
        daily.filter((pl.col("date") >= start) & (pl.col("date") <= end))[
            "date"
        ]
        .unique()
        .len()
    )
    if not eligible:
        return WindowCorrelation(
            start=start,
            end=end,
            eligible_symbols=[],
            corr_matrix=np.zeros((0, 0)),
            n_window_days=n_window_days,
            n_used_dates=0,
        )

    win = (
        daily.filter(
            pl.col("symbol").is_in(eligible)
            & (pl.col("date") >= start)
            & (pl.col("date") <= end)
        )
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol"))
            .log()
            .alias("logret")
        )
        .filter(pl.col("logret").is_not_null())
    )

    wide = win.pivot(index="date", on="symbol", values="logret").sort("date")
    wide_clean = wide.drop_nulls()

    n_used = wide_clean.height
    if n_used < 0.8 * n_window_days:
        logger.warning(
            "Window [{}, {}]: only {} of {} trading days survive the "
            "common-dates intersection — pivot has too many NaNs",
            start,
            end,
            n_used,
            n_window_days,
        )

    # Column order: explicit, matching the order of eligible symbols.
    sym_cols = [c for c in wide_clean.columns if c != "date"]
    # Sort to match `eligible` (alphabetical from window_eligible_symbols).
    sym_cols_sorted = sorted(sym_cols)
    if sym_cols_sorted != eligible:
        # Defensive: in pathological cases pivot might drop a column whose
        # entire returns series got squeezed by drop_nulls. Recompute.
        eligible = sym_cols_sorted

    arr = wide_clean.select(eligible).to_numpy()
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        corr = np.full((arr.shape[1], arr.shape[1]), np.nan)
    else:
        corr = np.corrcoef(arr, rowvar=False)

    return WindowCorrelation(
        start=start,
        end=end,
        eligible_symbols=eligible,
        corr_matrix=corr,
        n_window_days=n_window_days,
        n_used_dates=n_used,
    )


# ---------------------------------------------------------------------------
# Step 3 — same-sector pair screen
# ---------------------------------------------------------------------------


_PAIR_SCHEMA = {
    "sym1": pl.Utf8,
    "sym2": pl.Utf8,
    "corr": pl.Float64,
    "sector": pl.Utf8,
}


def screen_pairs(
    daily: pl.DataFrame,
    sectors: pl.DataFrame,
    *,
    start: date,
    end: date,
    corr_threshold: float = 0.85,
    max_internal_gap_days: int = 10,
) -> pl.DataFrame:
    """Return same-sector pairs with correlation > ``corr_threshold``.

    The single windowed entry point. Phase 2's walk-forward loop will call
    this per fold with that fold's training window — no global call.

    Returns a frame with columns ``sym1, sym2, corr, sector`` sorted by
    sector then descending correlation. Empty (matching schema) if no
    pairs survive.
    """
    if not 0 <= corr_threshold <= 1:
        raise ValueError(f"corr_threshold must be in [0,1], got {corr_threshold}")
    if start >= end:
        raise ValueError(f"start {start} must be < end {end}")

    win_corr = compute_window_correlation(
        daily,
        start=start,
        end=end,
        max_internal_gap_days=max_internal_gap_days,
    )
    eligible = win_corr.eligible_symbols
    if not eligible:
        logger.warning("Window [{}, {}]: no symbols passed the gap guard", start, end)
        return pl.DataFrame(schema=_PAIR_SCHEMA)

    sym_to_sector = dict(
        zip(sectors["symbol"], sectors["industry"], strict=True)
    )

    by_sector: dict[str, list[int]] = {}
    for i, s in enumerate(eligible):
        sec = sym_to_sector.get(s)
        if sec is None:
            continue
        by_sector.setdefault(sec, []).append(i)

    corr = win_corr.corr_matrix
    rows: list[dict] = []
    for sector, idxs in by_sector.items():
        if len(idxs) < 2:
            continue
        for ai in range(len(idxs)):
            i = idxs[ai]
            for bi in range(ai + 1, len(idxs)):
                j = idxs[bi]
                c = float(corr[i, j])
                if not np.isfinite(c):
                    continue
                if c > corr_threshold:
                    rows.append(
                        {
                            "sym1": eligible[i],
                            "sym2": eligible[j],
                            "corr": c,
                            "sector": sector,
                        }
                    )

    if not rows:
        return pl.DataFrame(schema=_PAIR_SCHEMA)

    return pl.DataFrame(rows, schema=_PAIR_SCHEMA).sort(
        ["sector", "corr"], descending=[False, True]
    )
