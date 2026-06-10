"""Zero-price-move round-trip must realise net P&L exactly -cost.

This pins the v2 cost-deduction plumbing. If the OU engine's accounting
ever drifts from v2's (e.g. a 2x scaling, an off-by-one bar, or a
direction-sign flip), this test catches it.

The notional convention (equal-notional, n_legs=2) is documented in
docs/ou_thresholds_design.md §8.1; this test does NOT pin the
convention itself — just the plumbing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apt.intraday.backtest import run_pair_fold
from apt.intraday.signals import IntradaySignalSeries


@pytest.mark.parametrize("regime", ["A", "B"])
def test_zero_move_round_trip_realises_negative_cost(regime: str) -> None:
    """Force a synthetic round-trip on a flat spread and check the deduction."""
    n = 8
    timestamps = pd.date_range("2020-06-15 09:15", periods=n, freq="1min", tz="Asia/Kolkata")
    sids = np.zeros(n, dtype=np.int32)
    # FLAT spread: entry at bar 1, exit at bar 3, no price movement.
    spread = np.full(n, 0.5, dtype=float)
    z = spread.copy()  # placeholder; not used for accounting

    pos = np.zeros(n, dtype=np.int8)
    held = np.zeros(n, dtype=np.int32)
    er = [None] * n
    # Short entry at bar 1; close at bar 3 with mean_revert
    pos[1] = -1
    pos[2] = -1
    held[1] = 1
    held[2] = 2
    pos[3] = 0
    er[3] = "mean_revert"

    sig = IntradaySignalSeries(position=pos, days_in_trade=held, exit_reason=er, regime=regime)
    cost = 0.0015  # 3 bps total spread, 15 bps round-trip in log units
    res = run_pair_fold(
        fold_id=0,
        pair_key="TEST/PAIR",
        timestamps=timestamps,
        session_id=sids,
        spread=spread,
        z=z,
        signals=sig,
        cost_log_per_round_trip=cost,
        finalize_fold_boundary=(regime == "B"),
    )
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.gross_log_pnl == pytest.approx(0.0, abs=1e-15)
    assert tr.cost_log == pytest.approx(cost, abs=1e-15)
    assert tr.net_log_pnl == pytest.approx(-cost, abs=1e-15)
    # Net per-bar at exit deducted EXACTLY cost (gross at exit = 0 since no move)
    exit_idx = 3
    assert res.gross_log_ret[exit_idx] == pytest.approx(0.0, abs=1e-15)
    assert res.net_log_ret[exit_idx] == pytest.approx(-cost, abs=1e-15)


def test_zero_move_long_round_trip() -> None:
    """Mirror test: long-spread round-trip on flat spread."""
    n = 8
    timestamps = pd.date_range("2020-06-15 09:15", periods=n, freq="1min", tz="Asia/Kolkata")
    sids = np.zeros(n, dtype=np.int32)
    spread = np.full(n, -0.3, dtype=float)
    z = spread.copy()

    pos = np.zeros(n, dtype=np.int8)
    held = np.zeros(n, dtype=np.int32)
    er = [None] * n
    pos[2] = +1
    pos[3] = +1
    pos[4] = +1
    held[2] = 1
    held[3] = 2
    held[4] = 3
    pos[5] = 0
    er[5] = "mean_revert"

    sig = IntradaySignalSeries(position=pos, days_in_trade=held, exit_reason=er, regime="B")
    cost = 0.0025  # 8 bps spread
    res = run_pair_fold(
        fold_id=0,
        pair_key="TEST/PAIR",
        timestamps=timestamps,
        session_id=sids,
        spread=spread,
        z=z,
        signals=sig,
        cost_log_per_round_trip=cost,
        finalize_fold_boundary=False,
    )
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.gross_log_pnl == pytest.approx(0.0, abs=1e-15)
    assert tr.net_log_pnl == pytest.approx(-cost, abs=1e-15)


def test_zero_move_costs_compose_additively() -> None:
    """Two sequential zero-move trades realize -2c (one cost per round-trip)."""
    n = 12
    timestamps = pd.date_range("2020-06-15 09:15", periods=n, freq="1min", tz="Asia/Kolkata")
    sids = np.zeros(n, dtype=np.int32)
    spread = np.full(n, 0.0, dtype=float)
    z = spread.copy()

    pos = np.zeros(n, dtype=np.int8)
    held = np.zeros(n, dtype=np.int32)
    er = [None] * n
    # Trade 1: short entry at bar 1, exit at bar 3
    pos[1] = pos[2] = -1
    held[1] = 1
    held[2] = 2
    pos[3] = 0
    er[3] = "mean_revert"
    # Trade 2: short entry at bar 5, exit at bar 7
    pos[5] = pos[6] = -1
    held[5] = 1
    held[6] = 2
    pos[7] = 0
    er[7] = "mean_revert"

    sig = IntradaySignalSeries(position=pos, days_in_trade=held, exit_reason=er, regime="B")
    cost = 0.0015
    res = run_pair_fold(
        fold_id=0,
        pair_key="TEST/PAIR",
        timestamps=timestamps,
        session_id=sids,
        spread=spread,
        z=z,
        signals=sig,
        cost_log_per_round_trip=cost,
        finalize_fold_boundary=False,
    )
    assert len(res.trades) == 2
    for tr in res.trades:
        assert tr.net_log_pnl == pytest.approx(-cost, abs=1e-15)
    # Sum of per-bar net = -2c
    assert float(res.net_log_ret.sum()) == pytest.approx(-2.0 * cost, abs=1e-15)
