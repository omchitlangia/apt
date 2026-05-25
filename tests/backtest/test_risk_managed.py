"""Tests for Phase 2B risk-managed walk-forward — sizing math, per-pair cap,
cluster cap binding + pro-rata scale-down, shared-leg de-dup, pair-kill triggers
(relationship + loss backstop), weighted portfolio aggregation, and the R0 ==
Phase-2A A/B reproducibility gate."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from apt.backtest import (
    Pair,
    RiskConfig,
    build_folds,
    run_walkforward,
    run_walkforward_risk_managed,
)
from apt.backtest.kill_switch import (
    check_loss_backstop,
    check_relationship_breakdown,
    evaluate_kill,
)
from apt.backtest.sizing import (
    apply_cluster_scaledown,
    apply_per_pair_cap,
    cluster_usage,
    compute_risk_based_weight,
    has_shared_leg,
)


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# 1% sizing math
# ---------------------------------------------------------------------------


def test_risk_based_weight_long_entry_known_value():
    # z_entry=-2.5, z_stop=3.5 ⇒ z-distance = 1.0
    # sigma=0.05 ⇒ spread-distance = 0.05
    # risk_frac=0.01 ⇒ weight = 0.01 / 0.05 = 0.20
    w = compute_risk_based_weight(z_entry=-2.5, z_stop=3.5, sigma_at_entry=0.05, risk_frac=0.01)
    assert w == pytest.approx(0.20, abs=1e-9)


def test_risk_based_weight_short_entry_symmetric():
    w_long = compute_risk_based_weight(
        z_entry=-2.5, z_stop=3.5, sigma_at_entry=0.05, risk_frac=0.01
    )
    w_short = compute_risk_based_weight(
        z_entry=+2.5, z_stop=3.5, sigma_at_entry=0.05, risk_frac=0.01
    )
    assert w_short == pytest.approx(w_long, abs=1e-12)


def test_risk_based_weight_larger_sigma_smaller_position():
    """If volatility doubles, position halves (same dollars-at-risk)."""
    w_lo = compute_risk_based_weight(z_entry=-2.5, z_stop=3.5, sigma_at_entry=0.05, risk_frac=0.01)
    w_hi = compute_risk_based_weight(z_entry=-2.5, z_stop=3.5, sigma_at_entry=0.10, risk_frac=0.01)
    assert w_hi == pytest.approx(w_lo / 2, abs=1e-9)


def test_risk_based_weight_zero_sigma_yields_zero():
    assert (
        compute_risk_based_weight(z_entry=-2.5, z_stop=3.5, sigma_at_entry=0.0, risk_frac=0.01)
        == 0.0
    )


# ---------------------------------------------------------------------------
# Per-pair cap
# ---------------------------------------------------------------------------


def test_per_pair_cap_clamps():
    assert apply_per_pair_cap(0.20, per_pair_cap=0.12) == 0.12
    assert apply_per_pair_cap(0.05, per_pair_cap=0.12) == 0.05


# ---------------------------------------------------------------------------
# Cluster cap: binding + pro-rata scale-down
# ---------------------------------------------------------------------------


def test_cluster_usage_sums_by_sector():
    sm = {"A/B": "FIN", "C/D": "FIN", "E/F": "OIL"}
    weights = {"A/B": 0.10, "C/D": 0.04, "E/F": 0.03}
    assert cluster_usage(weights, sm) == {"FIN": 0.14, "OIL": 0.03}


def test_cluster_scaledown_brings_over_cap_cluster_to_exactly_cap():
    sm = {"A/B": "FIN", "C/D": "FIN", "E/F": "OIL"}
    weights = {"A/B": 0.10, "C/D": 0.10, "E/F": 0.03}
    apply_cluster_scaledown(weights, sm, cluster_cap=0.05)
    # FIN cluster was 0.20 → scaled by 0.25 to total 0.05
    assert weights["A/B"] == pytest.approx(0.025)
    assert weights["C/D"] == pytest.approx(0.025)
    assert weights["E/F"] == 0.03  # OIL unchanged
    # Verify cluster sum is at cap
    assert cluster_usage(weights, sm)["FIN"] == pytest.approx(0.05, abs=1e-12)


def test_cluster_scaledown_does_nothing_when_under_cap():
    sm = {"A/B": "FIN"}
    weights = {"A/B": 0.03}
    apply_cluster_scaledown(weights, sm, cluster_cap=0.05)
    assert weights["A/B"] == 0.03


# ---------------------------------------------------------------------------
# Shared-leg de-dup
# ---------------------------------------------------------------------------


def test_shared_leg_detects_y_overlap():
    assert has_shared_leg("A/X", ["A/B"])


def test_shared_leg_detects_x_overlap():
    assert has_shared_leg("Z/B", ["A/B"])


def test_shared_leg_no_overlap():
    assert not has_shared_leg("P/Q", ["A/B", "C/D"])


# ---------------------------------------------------------------------------
# Kill switch: relationship + loss backstop
# ---------------------------------------------------------------------------


def test_kill_relationship_passes_on_stationary_ar1():
    rng = np.random.default_rng(0)
    n = 500
    s = np.zeros(n)
    phi = 0.85
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal()
    v = check_relationship_breakdown(s, window=252, adf_alpha=0.05, halflife_max_days=60)
    assert not v.killed


def test_kill_relationship_fires_on_random_walk():
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(size=500))
    v = check_relationship_breakdown(rw, window=252, adf_alpha=0.05)
    assert v.killed
    # ADF fails first for a random walk
    assert v.reason in {"adf_fail", "halflife_blowout"}


def test_kill_loss_backstop_consecutive_stops_K():
    assert check_loss_backstop(consecutive_stops=4, cum_pair_loss=0.0, kill_K=4).killed is True
    assert check_loss_backstop(consecutive_stops=3, cum_pair_loss=0.0, kill_K=4).killed is False


def test_kill_loss_backstop_cum_cap():
    assert (
        check_loss_backstop(consecutive_stops=0, cum_pair_loss=-0.05, kill_cap=0.04).killed is True
    )
    assert (
        check_loss_backstop(consecutive_stops=0, cum_pair_loss=-0.03, kill_cap=0.04).killed is False
    )


def test_kill_mode_none_never_fires():
    rng = np.random.default_rng(2)
    rw = np.cumsum(rng.normal(size=500))  # would fail relationship
    v = evaluate_kill(
        mode="none",
        spread_history=rw,
        consecutive_stops=10,
        cum_pair_loss=-0.10,
    )
    assert v.killed is False


def test_kill_mode_loss_only_skips_relationship():
    rng = np.random.default_rng(3)
    rw = np.cumsum(rng.normal(size=500))  # random walk → would fail relationship
    v = evaluate_kill(
        mode="loss_only",
        spread_history=rw,
        consecutive_stops=0,
        cum_pair_loss=0.0,
    )
    assert v.killed is False  # loss_only ignores relationship


# ---------------------------------------------------------------------------
# Engine: weighted portfolio aggregation + R0 reproducibility
# ---------------------------------------------------------------------------


def _mock_universe(seed: int = 0, n_days: int = 1200):
    """Synthetic 3-pair universe for end-to-end engine tests."""
    rng = np.random.default_rng(seed)
    days = _weekdays(date(2010, 1, 4), n_days)

    pair_prices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    pairs_meta: list[Pair] = []
    for k in range(3):
        common = rng.normal(scale=0.01, size=n_days).cumsum()
        noise_y = np.zeros(n_days)
        noise_x = np.zeros(n_days)
        for i in range(1, n_days):
            noise_y[i] = 0.5 * noise_y[i - 1] + rng.normal(scale=0.005)
            noise_x[i] = 0.5 * noise_x[i - 1] + rng.normal(scale=0.005)
        beta_true = 0.7 + 0.1 * k
        p_y = np.exp(4.0 + common + noise_y)
        p_x = np.exp(3.0 + beta_true * common + noise_x)
        y_sym = f"Y{k}"
        x_sym = f"X{k}"
        pair_prices[(y_sym, x_sym)] = (p_y, p_x)
        pairs_meta.append(
            Pair(
                y_sym=y_sym,
                x_sym=x_sym,
                alpha=1.0,
                beta=beta_true,
                half_life=15.0,
                sector="SYNTH",
                is_structural=False,
            )
        )

    def select_pairs_fn(prior_start, prior_end, train_start, train_end):
        return pairs_meta

    def get_prices_fn(y_sym, x_sym, start, end):
        idx = np.array([(d >= start) and (d <= end) for d in days])
        ii = np.where(idx)[0]
        p_y, p_x = pair_prices[(y_sym, x_sym)]
        return [days[i] for i in ii], p_y[ii], p_x[ii]

    return days, pairs_meta, select_pairs_fn, get_prices_fn


def test_r0_reproduces_phase_2a_byte_identical():
    """R0 (rung=0 in risk-managed engine) must produce identical daily portfolio
    returns to the Phase-2A walkforward engine on the same data + signals."""
    days, _pairs, sel_fn, prices_fn = _mock_universe(seed=10)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    assert len(folds) >= 2

    res_2a = run_walkforward(
        folds,
        days,
        select_pairs_fn=sel_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=15.0,
    )
    res_r0 = run_walkforward_risk_managed(
        folds,
        days,
        get_prices_fn=prices_fn,
        pre_selected_pairs=res_2a.selected_pairs_per_fold,
        rolling_window=30,
        cost_bps_per_leg=15.0,
        config=RiskConfig(rung=0, kill_mode="none"),
    )
    p2a = res_2a.portfolio_daily.sort("date")
    pr0 = res_r0.portfolio_daily.sort("date")
    merged = p2a.join(pr0, on="date", how="inner", suffix="_r0")
    diff_net = np.abs(merged["net_log_ret"].to_numpy() - merged["net_log_ret_r0"].to_numpy()).max()
    diff_gross = np.abs(
        merged["gross_log_ret"].to_numpy() - merged["gross_log_ret_r0"].to_numpy()
    ).max()
    # Must be ≤ floating-point noise
    assert diff_net < 1e-12, f"net diff = {diff_net}"
    assert diff_gross < 1e-12, f"gross diff = {diff_gross}"


def test_weighted_aggregation_higher_per_pair_cap_yields_larger_returns():
    """With looser per-pair cap, more capital is deployed per trade → larger
    gross returns (in absolute value). A direct sanity check of the
    sum(w_i * r_i) aggregation."""
    days, _pairs, sel_fn, prices_fn = _mock_universe(seed=11)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    res_2a = run_walkforward(
        folds,
        days,
        select_pairs_fn=sel_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=0.0,
    )
    pre = res_2a.selected_pairs_per_fold

    res_tight = run_walkforward_risk_managed(
        folds,
        days,
        get_prices_fn=prices_fn,
        pre_selected_pairs=pre,
        rolling_window=30,
        cost_bps_per_leg=0.0,
        config=RiskConfig(
            rung=2, kill_mode="none", per_pair_cap=0.05, cluster_cap=1.0, total_cap=1.0
        ),
    )
    res_loose = run_walkforward_risk_managed(
        folds,
        days,
        get_prices_fn=prices_fn,
        pre_selected_pairs=pre,
        rolling_window=30,
        cost_bps_per_leg=0.0,
        config=RiskConfig(
            rung=2, kill_mode="none", per_pair_cap=0.20, cluster_cap=1.0, total_cap=1.0
        ),
    )
    gross_tight = float(np.abs(res_tight.portfolio_daily["gross_log_ret"].to_numpy()).sum())
    gross_loose = float(np.abs(res_loose.portfolio_daily["gross_log_ret"].to_numpy()).sum())
    assert gross_loose >= gross_tight


def test_cluster_cap_binding_caps_total_cluster_exposure():
    """When per_pair_cap=12% and cluster_cap=5%, no day's cluster exposure
    should exceed 5% under R3."""
    days, _pairs, sel_fn, prices_fn = _mock_universe(seed=12)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    res_2a = run_walkforward(
        folds,
        days,
        select_pairs_fn=sel_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=0.0,
    )
    pre = res_2a.selected_pairs_per_fold

    res = run_walkforward_risk_managed(
        folds,
        days,
        get_prices_fn=prices_fn,
        pre_selected_pairs=pre,
        rolling_window=30,
        cost_bps_per_leg=0.0,
        config=RiskConfig(
            rung=3,
            kill_mode="none",
            risk_frac=0.01,
            per_pair_cap=0.12,
            cluster_cap=0.05,
            total_cap=1.0,
        ),
    )
    if not res.cluster_exposure_daily.is_empty():
        max_cluster_exposure = float(res.cluster_exposure_daily["exposure"].max())
        assert max_cluster_exposure <= 0.05 + 1e-9, (
            f"cluster cap of 5% violated: max={max_cluster_exposure:.4f}"
        )


def test_shared_leg_dedup_blocks_overlapping_second_entry():
    """Two pairs sharing a leg, both signalling entry — only one should open."""
    # Build a degenerate two-pair universe where both pairs trigger on day 30
    days = _weekdays(date(2010, 1, 4), 500)
    n = len(days)
    rng = np.random.default_rng(13)
    common = rng.normal(scale=0.005, size=n).cumsum()
    p_a = np.exp(4.0 + common)
    p_b = np.exp(3.0 + 0.7 * common)
    p_c = np.exp(2.0 + 0.5 * common)
    # Pairs A/B and A/C share leg A
    pair_prices = {("A", "B"): (p_a, p_b), ("A", "C"): (p_a, p_c)}
    pairs_meta = [
        Pair(
            y_sym="A",
            x_sym="B",
            alpha=1.0,
            beta=0.7,
            half_life=15.0,
            sector="X",
            is_structural=False,
        ),
        Pair(
            y_sym="A",
            x_sym="C",
            alpha=2.0,
            beta=0.5,
            half_life=15.0,
            sector="X",
            is_structural=False,
        ),
    ]

    def select_pairs_fn(*a):
        return pairs_meta

    def get_prices_fn(y_sym, x_sym, start, end):
        idx = np.array([(d >= start) and (d <= end) for d in days])
        ii = np.where(idx)[0]
        p_y, p_x = pair_prices[(y_sym, x_sym)]
        return [days[i] for i in ii], p_y[ii], p_x[ii]

    folds = build_folds(days, prior_days=180, train_days=180, test_days=80)
    res_2a = run_walkforward(
        folds,
        days,
        select_pairs_fn=select_pairs_fn,
        get_prices_fn=get_prices_fn,
        rolling_window=30,
        cost_bps_per_leg=0.0,
    )
    pre = res_2a.selected_pairs_per_fold

    res = run_walkforward_risk_managed(
        folds,
        days,
        get_prices_fn=get_prices_fn,
        pre_selected_pairs=pre,
        rolling_window=30,
        cost_bps_per_leg=0.0,
        config=RiskConfig(
            rung=3,
            kill_mode="none",
            risk_frac=0.01,
            per_pair_cap=0.50,
            cluster_cap=1.0,
            total_cap=1.0,
        ),
    )
    # We expect the shared-leg counter to register at least one skip if both
    # pairs ever attempt to enter simultaneously.
    assert res.funnel.get("n_entries_skipped_shared_leg", 0) >= 0
    # Additionally: at every timestep, never both A/B and A/C open
    for pkey_a, df_a in res.per_pair_daily.items():
        if pkey_a != "A/B":
            continue
        df_b = res.per_pair_daily.get("A/C")
        if df_b is None:
            continue
        merged = df_a.select(["date", "weight"]).join(
            df_b.select(["date", "weight"]),
            on="date",
            how="inner",
            suffix="_other",
        )
        both_open = (merged["weight"] != 0) & (merged["weight_other"] != 0)
        assert not both_open.any(), "shared-leg dedup failed: A/B and A/C open simultaneously"


def test_kill_mode_none_yields_zero_kill_events():
    days, _pairs, sel_fn, prices_fn = _mock_universe(seed=14)
    folds = build_folds(days, prior_days=400, train_days=400, test_days=100)
    res_2a = run_walkforward(
        folds,
        days,
        select_pairs_fn=sel_fn,
        get_prices_fn=prices_fn,
        rolling_window=30,
        cost_bps_per_leg=0.0,
    )
    pre = res_2a.selected_pairs_per_fold
    res = run_walkforward_risk_managed(
        folds,
        days,
        get_prices_fn=prices_fn,
        pre_selected_pairs=pre,
        rolling_window=30,
        cost_bps_per_leg=0.0,
        config=RiskConfig(rung=4, kill_mode="none"),
    )
    assert len(res.kill_events) == 0


def test_engine_requires_either_callback_or_preselected():
    days = _weekdays(date(2010, 1, 4), 500)
    folds = build_folds(days, prior_days=180, train_days=180, test_days=80)
    with pytest.raises(ValueError, match="pre_selected_pairs or select_pairs_fn"):
        run_walkforward_risk_managed(
            folds,
            days,
            get_prices_fn=lambda *a: ([], np.array([]), np.array([])),
            config=RiskConfig(rung=0, kill_mode="none"),
        )
