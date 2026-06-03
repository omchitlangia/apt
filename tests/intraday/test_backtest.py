"""Tests for the intraday PnL accounting + trade extraction."""

import numpy as np
import pandas as pd

from apt.intraday.backtest import run_pair_fold
from apt.intraday.signals import generate_signals_two_regime
from apt.intraday.zscore import sessionized_rolling_zscore


def _synthetic_two_session(n_per: int = 60) -> dict:
    """Two sessions of a known mean-reverting spread that should produce trades."""
    rng = np.random.default_rng(0)
    half = n_per // 2
    rest = n_per - half
    s1 = np.concatenate([np.linspace(0, -3.0, half), np.linspace(-3.0, 0.0, rest)])
    s2 = np.concatenate([np.linspace(0, +3.0, half), np.linspace(+3.0, 0.0, rest)])
    s1 += rng.normal(0, 0.05, n_per)
    s2 += rng.normal(0, 0.05, n_per)
    spread = np.concatenate([s1, s2])
    sids = np.concatenate([np.zeros(n_per, dtype=np.int32), np.ones(n_per, dtype=np.int32)])
    ts = pd.date_range("2020-01-02 09:15", periods=2 * n_per, freq="1min", tz="Asia/Kolkata")
    tradeable = np.ones(2 * n_per, dtype=bool)
    return {"spread": spread, "sids": sids, "ts": ts, "tradeable": tradeable}


def test_regime_a_session_open_bars_contribute_zero_return() -> None:
    d = _synthetic_two_session(n_per=50)
    z = sessionized_rolling_zscore(d["spread"], d["sids"], window=10)
    sig = generate_signals_two_regime(
        z, d["sids"], d["tradeable"], regime="A", entry=1.0, exit=0.5, stop=3.5, max_holding=999
    )
    res = run_pair_fold(
        fold_id=0,
        pair_key="A/B",
        timestamps=d["ts"],
        session_id=d["sids"],
        spread=d["spread"],
        z=z,
        signals=sig,
        cost_log_per_round_trip=0.0,
        finalize_fold_boundary=False,
    )
    # session_open indices in two-session layout: 0 and n_per (=50)
    assert res.gross_log_ret[0] == 0.0
    assert res.gross_log_ret[50] == 0.0
    # Total log PnL (Regime A) ignores the overnight move
    # No costs -> gross == net
    np.testing.assert_array_equal(res.gross_log_ret, res.net_log_ret)


def test_regime_b_realizes_overnight_gap() -> None:
    d = _synthetic_two_session(n_per=50)
    z = sessionized_rolling_zscore(d["spread"], d["sids"], window=10)
    sigB = generate_signals_two_regime(
        z, d["sids"], d["tradeable"], regime="B", entry=1.0, exit=0.5, stop=3.5, max_holding=999
    )
    resB = run_pair_fold(
        fold_id=0,
        pair_key="A/B",
        timestamps=d["ts"],
        session_id=d["sids"],
        spread=d["spread"],
        z=z,
        signals=sigB,
        cost_log_per_round_trip=0.0,
        finalize_fold_boundary=True,
    )
    # If any position was open at session 1 close, the bar at session 2 open
    # captures the gap return = pos * (spread[50] - spread[49]).
    if int(resB.position[49]) != 0:
        expected = float(resB.position[49]) * (d["spread"][50] - d["spread"][49])
        assert resB.gross_log_ret[50] == expected


def test_trade_cost_deducted_at_exit() -> None:
    d = _synthetic_two_session(n_per=40)
    z = sessionized_rolling_zscore(d["spread"], d["sids"], window=8)
    sig = generate_signals_two_regime(
        z, d["sids"], d["tradeable"], regime="A", entry=1.0, exit=0.3, stop=3.5, max_holding=999
    )
    res = run_pair_fold(
        fold_id=0,
        pair_key="A/B",
        timestamps=d["ts"],
        session_id=d["sids"],
        spread=d["spread"],
        z=z,
        signals=sig,
        cost_log_per_round_trip=0.001,
        finalize_fold_boundary=False,
    )
    # Each trade's net = gross - cost; sum of (gross - net) across bars = sum of costs
    if res.trades:
        per_trade_cost = sum(t.cost_log for t in res.trades)
        bar_cost = float((res.gross_log_ret - res.net_log_ret).sum())
        assert abs(bar_cost - per_trade_cost) < 1e-9


def test_no_lookahead_truncation_invariance() -> None:
    """Slicing the future cannot change a past bar's gross return."""
    d = _synthetic_two_session(n_per=40)
    z = sessionized_rolling_zscore(d["spread"], d["sids"], window=8)
    sig = generate_signals_two_regime(
        z, d["sids"], d["tradeable"], regime="A", entry=1.0, exit=0.3, stop=3.5, max_holding=999
    )
    res_full = run_pair_fold(
        fold_id=0,
        pair_key="A/B",
        timestamps=d["ts"],
        session_id=d["sids"],
        spread=d["spread"],
        z=z,
        signals=sig,
        cost_log_per_round_trip=0.0,
        finalize_fold_boundary=False,
    )
    # Truncate one session worth and re-run on the truncated slice
    n = 40
    z_t = sessionized_rolling_zscore(d["spread"][:n], d["sids"][:n], window=8)
    sig_t = generate_signals_two_regime(
        z_t,
        d["sids"][:n],
        d["tradeable"][:n],
        regime="A",
        entry=1.0,
        exit=0.3,
        stop=3.5,
        max_holding=999,
    )
    res_t = run_pair_fold(
        fold_id=0,
        pair_key="A/B",
        timestamps=d["ts"][:n],
        session_id=d["sids"][:n],
        spread=d["spread"][:n],
        z=z_t,
        signals=sig_t,
        cost_log_per_round_trip=0.0,
        finalize_fold_boundary=False,
    )
    # Regime A force-closes at each session's last bar.  Removing session 2
    # means the session-1 force-close bar (index 39) is now also the LAST
    # bar of the truncated test window — in the full series that bar is just
    # an end-of-session-1 close and the position then resets and may re-enter
    # in session 2.  Either way, the GROSS returns up to index 38 must agree.
    np.testing.assert_allclose(res_t.gross_log_ret[:39], res_full.gross_log_ret[:39])
