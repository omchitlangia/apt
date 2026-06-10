"""Smoke tests for the Phase-3 plotting helpers.

These verify the import contract and that each helper writes a non-empty
PNG when given a minimal synthetic input — they do not check the visual
output. Visual diffs are reviewed manually.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from apt.plots.intraday import (
    plot_phase3_per_pair_card,
    plot_phase3_portfolio_equity,
)


def test_portfolio_equity_writes_png(tmp_path: Path) -> None:
    n = 200
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": dates,
            "A_gross": rng.normal(0.0005, 0.01, n),
            "A_net": rng.normal(0.0001, 0.01, n),
            "B_gross": rng.normal(0.0007, 0.012, n),
            "B_net": rng.normal(0.0003, 0.012, n),
            "A_gross_ex": rng.normal(0.0005, 0.01, n),
            "A_net_ex": rng.normal(0.0001, 0.01, n),
            "B_gross_ex": rng.normal(0.0007, 0.012, n),
            "B_net_ex": rng.normal(0.0003, 0.012, n),
            "n_active_pairs": rng.integers(0, 5, n),
        }
    )
    out = tmp_path / "portfolio.png"
    stats = plot_phase3_portfolio_equity(df, out)
    assert out.exists() and out.stat().st_size > 1000
    assert stats["n_obs"] == n


def test_per_pair_card_writes_png_without_rep_fold(tmp_path: Path) -> None:
    n = 100
    start = date(2019, 6, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(1)
    cum_a = np.cumsum(rng.normal(0.05, 0.5, n))
    cum_b = np.cumsum(rng.normal(0.10, 0.6, n))
    out = tmp_path / "pair.png"
    res = plot_phase3_per_pair_card(
        pair_key="FOO/BAR",
        sector="TEST",
        is_structural=False,
        is_hdfcbank_anchored=False,
        daily_dates=dates,
        cum_net_A_pct=cum_a,
        cum_net_B_pct=cum_b,
        fold_spans=[(7, dates[10], dates[40])],
        rep_fold_id=None,
        rep_timestamps=None,
        rep_spread=None,
        rep_z=None,
        trades_in_rep_fold_A=None,
        trades_in_rep_fold_B=None,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        out_path=out,
    )
    assert out.exists() and out.stat().st_size > 1000
    assert res["pair"] == "FOO/BAR"
    assert res["rep_fold_id"] is None


def test_per_pair_card_writes_png_with_rep_fold(tmp_path: Path) -> None:
    n = 100
    start = date(2019, 6, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    cum_a = np.cumsum(np.random.default_rng(2).normal(0, 0.5, n))
    cum_b = np.cumsum(np.random.default_rng(3).normal(0, 0.5, n))

    # Synthetic intraday spread + z covering one session (375 bars)
    rep_ts = pd.date_range("2020-06-15 09:15", periods=375, freq="1min", tz="Asia/Kolkata")
    rep_spread = np.cumsum(np.random.default_rng(4).normal(0, 0.001, 375))
    rep_z = rep_spread / np.std(rep_spread)
    out = tmp_path / "pair_with_fold.png"
    plot_phase3_per_pair_card(
        pair_key="FOO/BAR",
        sector="TEST",
        is_structural=True,
        is_hdfcbank_anchored=True,
        daily_dates=dates,
        cum_net_A_pct=cum_a,
        cum_net_B_pct=cum_b,
        fold_spans=[(6, dates[10], dates[60])],
        rep_fold_id=6,
        rep_timestamps=rep_ts,
        rep_spread=rep_spread,
        rep_z=rep_z,
        trades_in_rep_fold_A=pd.DataFrame(
            {
                "entry_ts": [rep_ts[20].isoformat(), rep_ts[80].isoformat()],
                "exit_ts": [rep_ts[40].isoformat(), rep_ts[120].isoformat()],
                "side": ["long_spread", "short_spread"],
                "exit_reason": ["mean_revert", "eod_squareoff"],
            }
        ),
        trades_in_rep_fold_B=pd.DataFrame(),
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        out_path=out,
    )
    assert out.exists() and out.stat().st_size > 1000
