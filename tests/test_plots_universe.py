"""Smoke tests for apt.plots.universe — verify each plot function writes a
non-empty PNG without raising. Visual correctness isn't tested here."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from apt.plots.universe import (
    plot_adv_distribution,
    plot_coverage_heatmap,
    plot_history_length_distribution,
    plot_return_distributions,
    plot_symbols_per_sector,
)


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _synthetic_daily(n_symbols: int = 7, n_days: int = 800) -> pl.DataFrame:
    dates = _weekdays(date(2018, 1, 1), n_days)
    frames = []
    for i in range(n_symbols):
        sym = f"S{i:03d}"
        # mild drifting close, decent volume
        closes = [100.0 + 0.05 * j + (i * 5) for j in range(n_days)]
        vols = [1_000_000] * n_days
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [sym] * n_days,
                    "date": dates,
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": vols,
                }
            )
        )
    return pl.concat(frames)


def _synthetic_sectors(n_symbols: int = 7) -> pl.DataFrame:
    # n_symbols-2 in SECTOR_A (≥5 → not thin), 2 in SECTOR_B (thin).
    return pl.DataFrame(
        {
            "symbol": [f"S{i:03d}" for i in range(n_symbols)],
            "company_name": [f"Co {i}" for i in range(n_symbols)],
            "industry": (["SECTOR_A"] * (n_symbols - 2) + ["SECTOR_B"] * 2),
            "isin": [f"IN{i:010d}" for i in range(n_symbols)],
            "bse_industry": [""] * n_symbols,
        }
    )


def test_plot_symbols_per_sector_writes_png(tmp_path: Path):
    daily = _synthetic_daily()
    sectors = _synthetic_sectors()
    out = tmp_path / "01_sectors.png"
    stats = plot_symbols_per_sector(daily, sectors, out)
    assert out.exists() and out.stat().st_size > 1000
    assert stats["n_sectors"] == 2
    assert stats["n_thin_sectors"] == 1  # SECTOR_B has 2 < 5


def test_plot_history_length_distribution_writes_png(tmp_path: Path):
    daily = _synthetic_daily(n_symbols=4, n_days=900)
    out = tmp_path / "02_hist.png"
    stats = plot_history_length_distribution(daily, out, min_days_floor=756)
    assert out.exists() and out.stat().st_size > 1000
    assert stats["median_days"] == 900


def test_plot_coverage_heatmap_writes_png(tmp_path: Path):
    daily = _synthetic_daily(n_symbols=5, n_days=1200)
    out = tmp_path / "03_heatmap.png"
    stats = plot_coverage_heatmap(daily, out)
    assert out.exists() and out.stat().st_size > 1000
    assert stats["n_symbols"] == 5


def test_plot_return_distributions_writes_png(tmp_path: Path):
    daily = _synthetic_daily(n_symbols=3, n_days=500)
    out = tmp_path / "04_returns.png"
    # Use the synthetic symbol names so the per-symbol overlay has data.
    stats = plot_return_distributions(daily, out, sample_symbols=("S000", "S001", "S002"))
    assert out.exists() and out.stat().st_size > 1000
    # Smooth drifting closes → no extreme returns
    assert stats["n_abs_gt_0.2"] == 0


def test_plot_adv_distribution_writes_png(tmp_path: Path):
    daily = _synthetic_daily(n_symbols=4, n_days=200)
    out = tmp_path / "05_adv.png"
    stats = plot_adv_distribution(daily, out, floor_inr=1e7)
    assert out.exists() and out.stat().st_size > 1000
    assert stats["n_symbols"] == 4
