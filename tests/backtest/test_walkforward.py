"""Tests for apt.backtest.walkforward — fold boundaries (no leakage), trade
extraction + cost application, open-trade handoff, metric math, causal
selection, and the asset-agnostic callback interface."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from apt.backtest.walkforward import (
    Pair,
    _identify_trades_and_returns,
    build_folds,
    compute_metrics,
    run_walkforward,
)
from apt.signals.spread import SignalSeries


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# build_folds
# ---------------------------------------------------------------------------


def test_build_folds_correct_count_and_ordering():
    days = _weekdays(date(2005, 1, 3), 3000)
    folds = build_folds(days, prior_days=500, train_days=500, test_days=100)
    assert len(folds) > 0
    for f in folds:
        # Strict ordering: prior < train < test, all contiguous
        assert f.prior_start < f.prior_end
        assert f.prior_end < f.train_start
        assert f.train_start < f.train_end
        assert f.train_end < f.test_start
        assert f.test_start <= f.test_end


def test_build_folds_no_train_test_overlap():
    days = _weekdays(date(2005, 1, 3), 3000)
    folds = build_folds(days, prior_days=500, train_days=500, test_days=100)
    for f in folds:
        train_set = {d for d in days if f.train_start <= d <= f.train_end}
        test_set = {d for d in days if f.test_start <= d <= f.test_end}
        assert train_set.isdisjoint(test_set), f"fold {f.fold_id}: train and test overlap"


def test_build_folds_train_immediately_precedes_test():
    days = _weekdays(date(2005, 1, 3), 3000)
    folds = build_folds(days, prior_days=500, train_days=500, test_days=100)
    day_idx = {d: i for i, d in enumerate(days)}
    for f in folds:
        # No gap: train_end is exactly the trading day before test_start
        assert day_idx[f.test_start] == day_idx[f.train_end] + 1


def test_build_folds_non_overlapping_test_windows_default_step():
    days = _weekdays(date(2005, 1, 3), 3000)
    folds = build_folds(days, prior_days=500, train_days=500, test_days=100)
    for i in range(len(folds) - 1):
        # test_end of fold i < test_start of fold i+1 (or equal if step=test_days exactly)
        assert folds[i].test_end < folds[i + 1].test_start
        # And test windows are contiguous (no missed days) when step == test_days
        day_idx = {d: j for j, d in enumerate(days)}
        assert day_idx[folds[i + 1].test_start] == day_idx[folds[i].test_end] + 1


def test_build_folds_insufficient_history_raises():
    days = _weekdays(date(2005, 1, 3), 100)
    with pytest.raises(ValueError, match="need"):
        build_folds(days, prior_days=500, train_days=500, test_days=100)


def test_build_folds_bad_args_raise():
    days = _weekdays(date(2005, 1, 3), 2000)
    with pytest.raises(ValueError):
        build_folds(days, prior_days=0, train_days=500, test_days=100)
    with pytest.raises(ValueError):
        build_folds(days, prior_days=500, train_days=500, test_days=100, step_days=0)


def test_build_folds_step_smaller_than_test_overlaps():
    """If step < test_days, consecutive test windows overlap (rolling-cv style)."""
    days = _weekdays(date(2005, 1, 3), 2000)
    folds = build_folds(days, prior_days=500, train_days=500, test_days=100, step_days=50)
    # Folds 0 and 1 must overlap by 50 days
    assert folds[0].test_end > folds[1].test_start


# ---------------------------------------------------------------------------
# _identify_trades_and_returns
# ---------------------------------------------------------------------------


def _make_signal_series(positions: list[int], exit_reasons: list[str | None]):
    """Build a SignalSeries given a position path; compute days_in_trade for it."""
    pos = np.array(positions, dtype=np.int8)
    diit = np.zeros(len(pos), dtype=np.int32)
    held = 0
    for i, p in enumerate(pos):
        if p == 0:
            held = 0
        else:
            # Same logic as generate_signals: entry day starts at 1
            prev = 0 if i == 0 else pos[i - 1]
            held = 1 if prev == 0 else held + 1
        diit[i] = held
    return SignalSeries(position=pos, days_in_trade=diit, exit_reason=list(exit_reasons))


def test_trade_extraction_single_long_round_trip():
    # Long entry at i=1, exit at i=3 via mean_revert
    dates = _weekdays(date(2020, 1, 6), 5)
    spread = np.array([0.0, -0.5, -0.3, -0.1, 0.0])
    z = np.array([0.0, -2.5, -1.0, +0.3, 0.0])
    positions = [0, 1, 1, 0, 0]
    reasons = [None, None, None, "mean_revert", None]
    sig = _make_signal_series(positions, reasons)
    trades, gross, net = _identify_trades_and_returns(
        fold_id=0,
        pair_key="A/B",
        dates_test=dates,
        spread_test=spread,
        z_test=z,
        position=sig.position,
        exit_reason=sig.exit_reason,
        days_in_trade=sig.days_in_trade,
        cost_log_per_round_trip=0.005,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == +1
    assert t.entry_date == dates[1]
    assert t.exit_date == dates[3]
    # Gross = direction * (spread[3] - spread[1]) = 1 * (-0.1 - (-0.5)) = 0.4
    assert t.gross_log_pnl == pytest.approx(0.4, abs=1e-12)
    assert t.cost_log == pytest.approx(0.005, abs=1e-12)
    assert t.net_log_pnl == pytest.approx(0.395, abs=1e-12)
    assert t.exit_reason == "mean_revert"


def test_daily_returns_match_position_times_spread_diff():
    dates = _weekdays(date(2020, 1, 6), 5)
    spread = np.array([0.0, 0.10, 0.05, -0.05, -0.10])
    z = np.zeros(5)
    positions = [0, 1, 1, 0, 0]  # entry day 1, exit day 3
    reasons = [None, None, None, "mean_revert", None]
    sig = _make_signal_series(positions, reasons)
    _, gross, _ = _identify_trades_and_returns(
        fold_id=0,
        pair_key="A/B",
        dates_test=dates,
        spread_test=spread,
        z_test=z,
        position=sig.position,
        exit_reason=sig.exit_reason,
        days_in_trade=sig.days_in_trade,
        cost_log_per_round_trip=0.0,
    )
    # day 0: no carry → 0
    # day 1: pos[0]=0 * (spread[1]-spread[0]) = 0
    # day 2: pos[1]=1 * (spread[2]-spread[1]) = -0.05
    # day 3: pos[2]=1 * (spread[3]-spread[2]) = -0.10
    # day 4: pos[3]=0 → 0
    expected = np.array([0.0, 0.0, -0.05, -0.10, 0.0])
    np.testing.assert_allclose(gross, expected, atol=1e-12)


def test_cost_deducted_on_exit_day_only():
    dates = _weekdays(date(2020, 1, 6), 5)
    spread = np.array([0.0, 0.1, 0.2, 0.0, 0.0])
    z = np.zeros(5)
    positions = [0, 1, 1, 0, 0]
    reasons = [None, None, None, "mean_revert", None]
    sig = _make_signal_series(positions, reasons)
    _, gross, net = _identify_trades_and_returns(
        fold_id=0,
        pair_key="A/B",
        dates_test=dates,
        spread_test=spread,
        z_test=z,
        position=sig.position,
        exit_reason=sig.exit_reason,
        days_in_trade=sig.days_in_trade,
        cost_log_per_round_trip=0.005,
    )
    # net differs from gross only at exit day index 3
    diff = net - gross
    for i, d in enumerate(diff):
        if i == 3:
            assert d == pytest.approx(-0.005, abs=1e-12)
        else:
            assert d == pytest.approx(0.0, abs=1e-12)


def test_open_position_force_closed_at_fold_boundary():
    """Position still on at the last test bar → emit Trade with reason 'fold_boundary'."""
    dates = _weekdays(date(2020, 1, 6), 5)
    spread = np.array([0.0, -0.5, -0.6, -0.7, -0.4])  # never reverts
    z = np.array([0.0, -2.5, -2.6, -2.7, -1.5])
    positions = [0, 1, 1, 1, 1]
    reasons = [None, None, None, None, None]  # no signal-driven exit
    sig = _make_signal_series(positions, reasons)
    trades, _, _ = _identify_trades_and_returns(
        fold_id=0,
        pair_key="A/B",
        dates_test=dates,
        spread_test=spread,
        z_test=z,
        position=sig.position,
        exit_reason=sig.exit_reason,
        days_in_trade=sig.days_in_trade,
        cost_log_per_round_trip=0.005,
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "fold_boundary"
    assert trades[0].exit_date == dates[-1]
    # Gross = 1 * (spread[-1] - spread[1]) = 1 * (-0.4 - (-0.5)) = 0.1
    assert trades[0].gross_log_pnl == pytest.approx(0.1, abs=1e-12)


def test_short_round_trip_direction():
    dates = _weekdays(date(2020, 1, 6), 5)
    spread = np.array([0.0, 0.5, 0.3, 0.1, 0.0])
    z = np.array([0.0, +2.5, +1.0, +0.3, 0.0])
    positions = [0, -1, -1, 0, 0]
    reasons = [None, None, None, "mean_revert", None]
    sig = _make_signal_series(positions, reasons)
    trades, _, _ = _identify_trades_and_returns(
        fold_id=0,
        pair_key="A/B",
        dates_test=dates,
        spread_test=spread,
        z_test=z,
        position=sig.position,
        exit_reason=sig.exit_reason,
        days_in_trade=sig.days_in_trade,
        cost_log_per_round_trip=0.0,
    )
    assert len(trades) == 1
    assert trades[0].direction == -1
    # Gross = -1 * (spread[3] - spread[1]) = -1 * (0.1 - 0.5) = +0.4
    assert trades[0].gross_log_pnl == pytest.approx(0.4, abs=1e-12)


def test_multiple_trades_within_one_fold():
    dates = _weekdays(date(2020, 1, 6), 9)
    spread = np.array([0.0, -0.5, 0.0, 0.0, +0.5, 0.0, -0.5, 0.0, 0.0])
    z = np.array([0.0, -2.5, +0.0, 0.0, +2.5, 0.0, -2.5, 0.0, 0.0])
    # Two long entries (i=1, i=6) and one short entry (i=4); all close immediately
    positions = [0, 1, 0, 0, -1, 0, 1, 0, 0]
    reasons = [None, None, "mean_revert", None, None, "mean_revert", None, "mean_revert", None]
    sig = _make_signal_series(positions, reasons)
    trades, _, _ = _identify_trades_and_returns(
        fold_id=0,
        pair_key="A/B",
        dates_test=dates,
        spread_test=spread,
        z_test=z,
        position=sig.position,
        exit_reason=sig.exit_reason,
        days_in_trade=sig.days_in_trade,
        cost_log_per_round_trip=0.0,
    )
    assert len(trades) == 3


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_metrics_empty_series():
    m = compute_metrics([])
    assert m["n_obs"] == 0
    assert m["total_return_pct"] == 0.0
    assert np.isnan(m["sharpe"])


def test_metrics_constant_zero_return_series():
    m = compute_metrics([0.0] * 252)
    assert m["total_return_pct"] == pytest.approx(0.0)
    assert m["ann_return_pct"] == pytest.approx(0.0)
    assert np.isnan(m["sharpe"])  # zero variance → undefined Sharpe
    assert m["max_drawdown_pct"] == pytest.approx(0.0)


def test_metrics_positive_drift_gives_positive_sharpe():
    # Steady positive drift with small daily noise — Sharpe should be very high
    rng = np.random.default_rng(123)
    r = 0.0005 + rng.normal(scale=0.005, size=1000)
    m = compute_metrics(r)
    assert m["total_return_pct"] > 0
    assert m["sharpe"] > 1.0  # consistent positive drift dominates the noise


def test_metrics_known_total_return_log_to_arith():
    r = np.array([0.01, 0.01, 0.01])
    m = compute_metrics(r)
    # cum log = 0.03 → arith = exp(0.03) - 1 ≈ 0.030455 → 3.045%
    assert m["total_return_pct"] == pytest.approx(3.0455, abs=0.01)


def test_metrics_max_drawdown_negative_when_series_falls():
    # +5% then -10% (in log): peak cum_log = 0.05, trough = -0.05,
    # peak-to-trough = -0.10 in log → arith = exp(-0.10) - 1 ≈ -9.52%
    r = np.array([0.05, -0.10])
    m = compute_metrics(r)
    assert m["max_drawdown_pct"] < 0
    assert m["max_drawdown_pct"] == pytest.approx(-9.516, abs=0.05)


# ---------------------------------------------------------------------------
# run_walkforward — asset-agnostic end-to-end on mock callbacks
# ---------------------------------------------------------------------------


def _make_mock_walkforward(seed: int = 0, n_pairs: int = 2):
    """A self-contained mock universe for testing run_walkforward without
    touching daily_clean.parquet — pure asset-agnostic exercise."""
    rng = np.random.default_rng(seed)
    days = _weekdays(date(2010, 1, 4), 1500)
    # Synthetic cointegrated price series: shared log-stochastic-trend + AR(1) noise
    common = rng.normal(scale=0.01, size=len(days)).cumsum()
    pair_prices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    pairs_meta: list[Pair] = []
    for k in range(n_pairs):
        beta_true = 0.7 + 0.1 * k
        noise_y = np.zeros(len(days))
        noise_x = np.zeros(len(days))
        for i in range(1, len(days)):
            noise_y[i] = 0.5 * noise_y[i - 1] + rng.normal(scale=0.005)
            noise_x[i] = 0.5 * noise_x[i - 1] + rng.normal(scale=0.005)
        log_py = 4.0 + common + noise_y
        log_px = 3.0 + beta_true * common + noise_x
        py = np.exp(log_py)
        px = np.exp(log_px)
        sym_y = f"Y{k}"
        sym_x = f"X{k}"
        pair_prices[(sym_y, sym_x)] = (py, px)
        pairs_meta.append(
            Pair(
                y_sym=sym_y,
                x_sym=sym_x,
                alpha=1.0,
                beta=beta_true,
                half_life=15.0,
                sector="SYNTH",
                is_structural=False,
            )
        )

    select_calls: list[tuple[date, date, date, date]] = []
    price_calls: list[tuple[str, str, date, date]] = []

    def select_pairs_fn(prior_start, prior_end, train_start, train_end):
        select_calls.append((prior_start, prior_end, train_start, train_end))
        return pairs_meta

    def get_prices_fn(y_sym, x_sym, start, end):
        price_calls.append((y_sym, x_sym, start, end))
        # Find dates in the window
        idx_mask = np.array([(d >= start) and (d <= end) for d in days])
        idxs = np.where(idx_mask)[0]
        py, px = pair_prices[(y_sym, x_sym)]
        return [days[i] for i in idxs], py[idxs], px[idxs]

    return days, pairs_meta, select_pairs_fn, get_prices_fn, select_calls, price_calls


def test_run_walkforward_end_to_end_mock_asset_agnostic():
    days, _, select_fn, prices_fn, sel_calls, _ = _make_mock_walkforward(seed=1)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    assert len(folds) >= 2
    result = run_walkforward(
        folds,
        days,
        select_pairs_fn=select_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=10.0,
    )
    # selection called once per fold
    assert len(sel_calls) == len(folds)
    # Portfolio has n_test rows per fold
    expected_rows = sum(sum(1 for d in days if f.test_start <= d <= f.test_end) for f in folds)
    assert result.portfolio_daily.height == expected_rows
    # Funnel sanity
    assert result.funnel["n_folds"] == len(folds)
    assert result.funnel["n_pair_fold_units"] > 0


def test_run_walkforward_selection_never_sees_test_dates():
    """LEAKAGE TEST: select_pairs_fn is called with (prior, train) only.
    Sentinel test: capture every (start, end) passed to select_fn; assert that
    no test_start..test_end date is in the (prior_start..train_end) range that
    the selector sees."""
    days, _, _, _, _, _ = _make_mock_walkforward(seed=2)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)

    captured: list[tuple[date, date]] = []

    def select_fn(prior_start, prior_end, train_start, train_end):
        captured.append((prior_start, train_end))
        return []  # no pairs

    def prices_fn(y, x, start, end):
        return [], np.empty(0), np.empty(0)

    run_walkforward(
        folds,
        days,
        select_pairs_fn=select_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=0,
    )
    for (cap_start, cap_end), f in zip(captured, folds, strict=True):
        # The captured window must end strictly before the fold's test_start
        assert cap_end < f.test_start, (
            f"Leakage: selector saw {cap_end} >= test_start {f.test_start}"
        )
        assert cap_start == f.prior_start
        assert cap_end == f.train_end


def test_run_walkforward_zero_cost_makes_gross_equal_net():
    days, _, select_fn, prices_fn, _, _ = _make_mock_walkforward(seed=3)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    result = run_walkforward(
        folds,
        days,
        select_pairs_fn=select_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=0.0,
    )
    if not result.portfolio_daily.is_empty():
        gross = result.portfolio_daily["gross_log_ret"].to_numpy()
        net = result.portfolio_daily["net_log_ret"].to_numpy()
        np.testing.assert_allclose(gross, net, atol=1e-12)


def test_run_walkforward_higher_cost_lowers_net_return():
    days, _, select_fn, prices_fn, _, _ = _make_mock_walkforward(seed=4)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    res_lo = run_walkforward(
        folds,
        days,
        select_pairs_fn=select_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=5.0,
    )
    res_hi = run_walkforward(
        folds,
        days,
        select_pairs_fn=select_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=50.0,
    )
    # Same selection + same prices ⇒ same gross
    if not res_lo.portfolio_daily.is_empty():
        np.testing.assert_allclose(
            res_lo.portfolio_daily["gross_log_ret"].to_numpy(),
            res_hi.portfolio_daily["gross_log_ret"].to_numpy(),
            atol=1e-12,
        )
        # But higher cost ⇒ lower net
        m_lo = compute_metrics(res_lo.portfolio_daily["net_log_ret"].to_numpy())
        m_hi = compute_metrics(res_hi.portfolio_daily["net_log_ret"].to_numpy())
        assert m_hi["total_return_pct"] <= m_lo["total_return_pct"]


def test_run_walkforward_open_position_handoff_no_carry_across_folds():
    """A trade open at the end of fold k is force-closed; fold k+1 starts flat.

    Engine guarantees fold-level independence — we verify by ensuring every
    exit_reason on the FIRST bar of any fold is None and position[0] is
    governed only by that fold's z at test_start (not the prior fold's
    state). The strongest available black-box check: every Trade emitted in
    fold k has entry_date >= fold k's test_start.
    """
    days, _, select_fn, prices_fn, _, _ = _make_mock_walkforward(seed=5)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    fold_test_starts = {f.fold_id: f.test_start for f in folds}
    res = run_walkforward(
        folds,
        days,
        select_pairs_fn=select_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=5.0,
    )
    for t in res.trades:
        assert t.entry_date >= fold_test_starts[t.fold_id], (
            f"Trade entered at {t.entry_date} which is before its fold's "
            f"test_start {fold_test_starts[t.fold_id]} — position carried across folds"
        )


def test_run_walkforward_invalid_cost_raises():
    days = _weekdays(date(2010, 1, 4), 1500)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    with pytest.raises(ValueError, match="cost"):
        run_walkforward(
            folds,
            days,
            select_pairs_fn=lambda *a: [],
            get_prices_fn=lambda *a: ([], np.array([]), np.array([])),
            cost_bps_per_leg=-1.0,
        )


def test_run_walkforward_empty_pair_selection_produces_empty_portfolio():
    days = _weekdays(date(2010, 1, 4), 1500)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    res = run_walkforward(
        folds,
        days,
        select_pairs_fn=lambda *a: [],
        get_prices_fn=lambda *a: ([], np.array([]), np.array([])),
        cost_bps_per_leg=10.0,
    )
    assert res.portfolio_daily.is_empty()
    assert res.trades == []
    assert res.funnel["n_pair_selections"] == 0
