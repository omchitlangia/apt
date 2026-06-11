"""Zero-price-move round-trip must realise net P&L exactly −(1+β)·per_leg_cost.

Under the **(1+β) billing** convention in :mod:`apt.intraday.costs`, the
single central function ``CostBreakdown.billed_cost_log_per_pair_round_trip``
computes pair-fold round-trip cost as ``(1 + β) × cost_log_per_leg``.
``β = 1`` reproduces the legacy 2× equal-notional value exactly (the
continuity pin replacing the old ``-c`` test).

The tests below cover:

* The β-grid asked for by Unit C: ``{0.057, 0.872, 1.0, 1.643}``.
* Long + short sides on a flat spread.
* Two sequential round-trips composing to −2·(1+β)·per_leg.
* Mismatched ``pair_beta`` arguments to ``run_pair_fold`` are passed
  through to the emitted ``IntradayTrade`` rows verbatim (so re-stamps
  can join β safely).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from apt.intraday.backtest import run_pair_fold
from apt.intraday.costs import CostBreakdown
from apt.intraday.signals import IntradaySignalSeries

# Per-leg log cost used for every test below (3 bps total spread ⇒ 7.5 bps per leg).
PER_LEG_LOG_COST = CostBreakdown(total_spread_bps=3).cost_log_per_leg


@pytest.mark.parametrize("beta", [0.057, 0.872, 1.0, 1.643])
@pytest.mark.parametrize("regime", ["A", "B"])
def test_zero_move_round_trip_realises_negative_one_plus_beta_cost(
    beta: float, regime: str
) -> None:
    """A flat-spread short round-trip on a pair-fold with β realises net = −(1+β)·per_leg."""
    n = 8
    timestamps = pd.date_range("2020-06-15 09:15", periods=n, freq="1min", tz="Asia/Kolkata")
    sids = np.zeros(n, dtype=np.int32)
    spread = np.full(n, 0.5, dtype=float)
    z = spread.copy()

    pos = np.zeros(n, dtype=np.int8)
    held = np.zeros(n, dtype=np.int32)
    er: list = [None] * n
    pos[1] = -1
    pos[2] = -1
    held[1] = 1
    held[2] = 2
    pos[3] = 0
    er[3] = "mean_revert"

    sig = IntradaySignalSeries(position=pos, days_in_trade=held, exit_reason=er, regime=regime)
    cb = CostBreakdown(total_spread_bps=3)
    cost = cb.billed_cost_log_per_pair_round_trip(beta=beta)
    expected = (1.0 + beta) * PER_LEG_LOG_COST
    assert cost == pytest.approx(expected, abs=1e-15)

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
        pair_beta=beta,
    )
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.gross_log_pnl == pytest.approx(0.0, abs=1e-15)
    assert tr.cost_log == pytest.approx(cost, abs=1e-15)
    assert tr.net_log_pnl == pytest.approx(-cost, abs=1e-15)
    assert tr.pair_beta == pytest.approx(beta, abs=1e-15)

    # Net per-bar at exit deducted EXACTLY cost (gross at exit = 0)
    exit_idx = 3
    assert res.gross_log_ret[exit_idx] == pytest.approx(0.0, abs=1e-15)
    assert res.net_log_ret[exit_idx] == pytest.approx(-cost, abs=1e-15)


def test_continuity_pin_beta_one_reproduces_legacy_2x() -> None:
    """β=1 must reproduce the legacy 2 × per_leg value bit-exactly."""
    cb = CostBreakdown(total_spread_bps=3)
    billed = cb.billed_cost_log_per_pair_round_trip(beta=1.0)
    legacy_2x = 2.0 * cb.cost_log_per_leg
    assert billed == pytest.approx(legacy_2x, abs=0)  # exact equality
    assert billed == 2.0 * cb.cost_log_per_leg  # bit-exact


def test_zero_move_long_round_trip_at_beta() -> None:
    """Mirror test: long-spread round-trip on flat spread with β = 1.643."""
    n = 8
    timestamps = pd.date_range("2020-06-15 09:15", periods=n, freq="1min", tz="Asia/Kolkata")
    sids = np.zeros(n, dtype=np.int32)
    spread = np.full(n, -0.3, dtype=float)
    z = spread.copy()

    pos = np.zeros(n, dtype=np.int8)
    held = np.zeros(n, dtype=np.int32)
    er: list = [None] * n
    pos[2] = +1
    pos[3] = +1
    pos[4] = +1
    held[2] = 1
    held[3] = 2
    held[4] = 3
    pos[5] = 0
    er[5] = "mean_revert"

    sig = IntradaySignalSeries(position=pos, days_in_trade=held, exit_reason=er, regime="B")
    cb = CostBreakdown(total_spread_bps=8)  # 8 bps spread; 12.5 bps per leg
    beta = 1.643
    cost = cb.billed_cost_log_per_pair_round_trip(beta=beta)
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
        pair_beta=beta,
    )
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.gross_log_pnl == pytest.approx(0.0, abs=1e-15)
    assert tr.net_log_pnl == pytest.approx(-cost, abs=1e-15)
    assert tr.pair_beta == pytest.approx(beta, abs=1e-15)


def test_zero_move_costs_compose_additively_at_beta() -> None:
    """Two sequential zero-move trades realize -2·(1+β)·per_leg."""
    n = 12
    timestamps = pd.date_range("2020-06-15 09:15", periods=n, freq="1min", tz="Asia/Kolkata")
    sids = np.zeros(n, dtype=np.int32)
    spread = np.full(n, 0.0, dtype=float)
    z = spread.copy()

    pos = np.zeros(n, dtype=np.int8)
    held = np.zeros(n, dtype=np.int32)
    er: list = [None] * n
    # Two short round-trips
    pos[1] = pos[2] = -1
    held[1] = 1
    held[2] = 2
    pos[3] = 0
    er[3] = "mean_revert"
    pos[5] = pos[6] = -1
    held[5] = 1
    held[6] = 2
    pos[7] = 0
    er[7] = "mean_revert"

    sig = IntradaySignalSeries(position=pos, days_in_trade=held, exit_reason=er, regime="B")
    cb = CostBreakdown(total_spread_bps=3)
    beta = 0.872
    cost = cb.billed_cost_log_per_pair_round_trip(beta=beta)
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
        pair_beta=beta,
    )
    assert len(res.trades) == 2
    for tr in res.trades:
        assert tr.net_log_pnl == pytest.approx(-cost, abs=1e-15)
        assert tr.pair_beta == pytest.approx(beta, abs=1e-15)
    assert float(res.net_log_ret.sum()) == pytest.approx(-2.0 * cost, abs=1e-15)


def test_billing_rejects_invalid_beta() -> None:
    cb = CostBreakdown(total_spread_bps=3)
    with pytest.raises(ValueError):
        cb.billed_cost_log_per_pair_round_trip(beta=-0.1)
    with pytest.raises(ValueError):
        cb.billed_cost_log_per_pair_round_trip(beta=float("nan"))
    with pytest.raises(ValueError):
        cb.billed_cost_log_per_pair_round_trip(beta=math.inf)
