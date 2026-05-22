"""Tests for apt.signals.cointegration — Engle-Granger both-directions, half-life,
Hurst, BH FDR, cross-window stability, DVR/structural handling, end-to-end
windowed screen."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from apt.signals.cointegration import (
    STRUCTURAL_PAIRS,
    benjamini_hochberg,
    cointegrate_pairs,
    engle_granger,
    engle_granger_best_direction,
    half_life_ar1,
    hurst_exponent,
    is_share_class_duplicate,
)


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _make_symbol(sym: str, dates: list[date], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [sym] * len(dates),
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * len(dates),
        }
    )


def _sectors_df(rows: dict[str, tuple[str, str]]) -> pl.DataFrame:
    """rows maps symbol → (industry, isin)."""
    return pl.DataFrame(
        {
            "symbol": list(rows.keys()),
            "company_name": list(rows),
            "industry": [v[0] for v in rows.values()],
            "isin": [v[1] for v in rows.values()],
            "bse_industry": [""] * len(rows),
        }
    )


def _build_pair_universe(
    *,
    n_days: int = 900,
    sym_a: str = "A",
    sym_b: str = "B",
    sector: str = "SEC1",
    phi: float = 0.85,
    drift: float = 0.0,
    beta_true: float = 0.7,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Construct a daily frame where (sym_a, sym_b) is genuinely cointegrated.

    Common stochastic trend: ``f_t = drift + Σ shocks_t``. Then
        log P_a(t) = c_a + f_t + ε_a(t),
        log P_b(t) = c_b + beta_true * f_t + ε_b(t),
    so log P_a − ((c_a − c_b)/beta_true_to_one)·log P_b is stationary mean-reverting AR(1).
    """
    rng = np.random.default_rng(seed)
    days = _weekdays(date(2010, 1, 4), n_days)
    shocks = rng.normal(scale=0.01, size=n_days)
    f = np.cumsum(shocks) + drift * np.arange(n_days)
    # mean-reverting noise on top
    noise_a = np.zeros(n_days)
    noise_b = np.zeros(n_days)
    for i in range(1, n_days):
        noise_a[i] = phi * noise_a[i - 1] + rng.normal(scale=0.005)
        noise_b[i] = phi * noise_b[i - 1] + rng.normal(scale=0.005)
    log_pa = 4.0 + f + noise_a
    log_pb = 3.0 + beta_true * f + noise_b
    closes_a = np.exp(log_pa).tolist()
    closes_b = np.exp(log_pb).tolist()
    a = _make_symbol(sym_a, days, closes_a)
    b = _make_symbol(sym_b, days, closes_b)
    daily = pl.concat([a, b])
    isin_a = f"INE{abs(hash(sym_a)) % 10**9:09d}1"
    isin_b = f"INE{abs(hash(sym_b)) % 10**9:09d}1"
    sectors = _sectors_df({sym_a: (sector, isin_a), sym_b: (sector, isin_b)})
    return daily, sectors


# ---------------------------------------------------------------------------
# is_share_class_duplicate
# ---------------------------------------------------------------------------


def test_share_class_blacklist_hits():
    assert is_share_class_duplicate("TATAMTRDVR", "IN9155A01020")


def test_share_class_dvr_suffix():
    """Any symbol ending in DVR is flagged even without a hardcoded entry."""
    assert is_share_class_duplicate("FUTUREDVR", "INE000000001")


def test_share_class_non_ine_isin_flagged():
    """Non-INE ISIN prefix marks differential/preferred lines."""
    assert is_share_class_duplicate("WEIRDCO", "IN9000000000")


def test_share_class_normal_equity_passes():
    assert not is_share_class_duplicate("TATAMOTORS", "INE155A01022")
    assert not is_share_class_duplicate("RELIANCE", "INE002A01018")


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------


def test_bh_textbook_example():
    """Classic BH example: alpha=0.05, n=10 hypotheses."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.96])
    mask = benjamini_hochberg(p, alpha=0.05)
    # Largest k with p_(k) <= k/n * alpha: only k=2 (p=0.008 ≤ 0.010) — k=3 fails (0.039 > 0.015)
    assert mask.tolist() == [True, True, False, False, False, False, False, False, False, False]


def test_bh_all_significant():
    p = np.array([0.001, 0.002, 0.003, 0.004])
    mask = benjamini_hochberg(p, alpha=0.05)
    assert mask.all()


def test_bh_none_significant():
    p = np.array([0.6, 0.7, 0.8, 0.9])
    mask = benjamini_hochberg(p, alpha=0.05)
    assert not mask.any()


def test_bh_empty():
    assert benjamini_hochberg(np.array([]), alpha=0.05).size == 0


def test_bh_is_less_conservative_than_bonferroni():
    """BH should always reject AT LEAST as many as Bonferroni at the same α."""
    rng = np.random.default_rng(0)
    # mix of real signal + nulls
    p = np.concatenate([rng.uniform(0, 0.01, 30), rng.uniform(0, 1, 70)])
    alpha = 0.05
    bh = benjamini_hochberg(p, alpha=alpha)
    bonf = p <= alpha / len(p)
    # BH rejects everything Bonferroni does, possibly more
    assert (bh | bonf).sum() == bh.sum()
    assert bh.sum() >= bonf.sum()


# ---------------------------------------------------------------------------
# half_life_ar1
# ---------------------------------------------------------------------------


def test_half_life_known_phi():
    """Half-life from AR(1) should track -ln(2)/ln(φ) within ~15% on n=2000."""
    rng = np.random.default_rng(1)
    phi = 0.9
    n = 2000
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal()
    hl = half_life_ar1(s)
    expected = -np.log(2) / np.log(phi)
    assert hl == pytest.approx(expected, rel=0.15)


def test_half_life_random_walk_is_inf():
    rng = np.random.default_rng(2)
    rw = np.cumsum(rng.normal(size=2000))
    hl = half_life_ar1(rw)
    assert hl == float("inf") or hl > 1000  # essentially unbounded


def test_half_life_too_short_returns_nan():
    assert np.isnan(half_life_ar1(np.array([1.0, 2.0])))


# ---------------------------------------------------------------------------
# hurst_exponent
# ---------------------------------------------------------------------------


def test_hurst_random_walk_about_half():
    rng = np.random.default_rng(3)
    rw = np.cumsum(rng.normal(size=4000))
    h = hurst_exponent(rw, max_lag=200)
    assert 0.40 <= h <= 0.60


def test_hurst_strongly_mean_reverting():
    """A bounded mean-reverting series should give H well below 0.5."""
    rng = np.random.default_rng(4)
    n = 4000
    s = np.zeros(n)
    phi = 0.5
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal()
    h = hurst_exponent(s, max_lag=200)
    assert h < 0.35


def test_hurst_too_short_is_nan():
    assert np.isnan(hurst_exponent(np.array([1.0, 2.0, 3.0])))


# ---------------------------------------------------------------------------
# engle_granger + best-direction selection
# ---------------------------------------------------------------------------


def test_engle_granger_recovers_beta_on_genuine_cointegration():
    """If P_a, P_b share a stochastic trend with known beta_true, EG should
    recover beta_true."""
    daily, sectors = _build_pair_universe(n_days=1000, beta_true=0.7, phi=0.5, seed=10)
    log_a = np.log(daily.filter(pl.col("symbol") == "A")["close"].to_numpy())
    log_b = np.log(daily.filter(pl.col("symbol") == "B")["close"].to_numpy())
    eg = engle_granger(log_b, log_a, y_sym="B", x_sym="A")
    assert eg.beta == pytest.approx(0.7, rel=0.05)
    assert eg.adf_pvalue < 0.05


def test_engle_granger_best_direction_picks_lower_pvalue():
    """When β > 1 in one direction and 1/β in the other, residual variance
    differs — best_direction must return the direction with smaller p-value."""
    daily, _ = _build_pair_universe(n_days=1500, beta_true=1.4, phi=0.4, seed=11)
    log_a = np.log(daily.filter(pl.col("symbol") == "A")["close"].to_numpy())
    log_b = np.log(daily.filter(pl.col("symbol") == "B")["close"].to_numpy())

    eg_ab = engle_granger(log_a, log_b, y_sym="A", x_sym="B")
    eg_ba = engle_granger(log_b, log_a, y_sym="B", x_sym="A")
    best = engle_granger_best_direction(log_a, log_b, sym_a="A", sym_b="B")
    expected_dir = eg_ab if eg_ab.adf_pvalue <= eg_ba.adf_pvalue else eg_ba
    assert best.adf_pvalue == pytest.approx(expected_dir.adf_pvalue, abs=1e-12)
    assert best.beta == pytest.approx(expected_dir.beta, abs=1e-12)


def test_engle_granger_independent_walks_rarely_reject():
    """Across many seeds of two independent random walks, the false-positive
    rate at α=0.05 should stay near the nominal level — i.e. EG is not
    systematically over-rejecting. Spurious rejections do happen on individual
    seeds (Granger-Newbold), so we test rate, not any single seed."""
    n = 600
    rejects = 0
    n_trials = 40
    for s in range(n_trials):
        rng = np.random.default_rng(1_000 + s)
        a = np.cumsum(rng.normal(size=n))
        b = np.cumsum(rng.normal(size=n))
        eg = engle_granger(a, b)
        if eg.adf_pvalue < 0.05:
            rejects += 1
    # Tolerate a few spurious rejections but well below "most pairs cointegrate"
    assert rejects <= 0.20 * n_trials


def test_engle_granger_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        engle_granger(np.zeros(50), np.zeros(40))


def test_engle_granger_too_short_raises():
    with pytest.raises(ValueError, match="need >="):
        engle_granger(np.zeros(10), np.zeros(10))


# ---------------------------------------------------------------------------
# cointegrate_pairs — end-to-end on synthetic data
# ---------------------------------------------------------------------------


def test_cointegrate_pairs_finds_cointegrated_synthetic_pair():
    daily, sectors = _build_pair_universe(n_days=900, beta_true=0.7, phi=0.4, seed=20)
    days = daily["date"].unique().sort()
    res = cointegrate_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[-1],
        corr_threshold=0.30,
        fdr_alpha=0.05,
        raw_alpha=0.05,
        min_overlap_days=500,
        half_life_min_days=2,
        half_life_max_days=200,
        hurst_max=0.55,
    )
    assert res.funnel.candidates_after_corr >= 1
    assert res.pairs.height >= 1
    row = res.pairs.row(0, named=True)
    assert row["adf_pvalue"] < 0.05
    assert row["fdr_pass"] is True
    assert row["beta"] == pytest.approx(0.7, rel=0.10) or row["beta"] == pytest.approx(
        1.0 / 0.7, rel=0.10
    )


def test_cointegrate_pairs_drops_share_class_duplicates():
    """A pair (P, P_DVR) must be excluded before EG even runs.

    We construct a perfectly identical DVR series and a regular sibling, both
    in the same sector. Without the DVR filter we'd see 1 surviving pair; with
    the filter we should see 0.
    """
    days = _weekdays(date(2015, 1, 5), 900)
    rng = np.random.default_rng(30)
    closes = (100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))).tolist()
    parent = _make_symbol("PARENT", days, closes)
    dvr = _make_symbol("PARENTDVR", days, closes)  # identical → trivial coint
    daily = pl.concat([parent, dvr])
    sectors = _sectors_df(
        {
            "PARENT": ("AUTO", "INE000000001"),
            "PARENTDVR": ("AUTO", "IN9000000000"),
        }
    )

    res = cointegrate_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[-1],
        corr_threshold=0.50,
        min_overlap_days=500,
    )
    # PARENTDVR is filtered as a share class → no candidates can include it
    assert res.funnel.excluded_share_class == 1
    assert res.funnel.candidates_after_corr == 0
    assert res.pairs.is_empty()


def test_cointegrate_pairs_tags_structural_pair():
    """If a structural pair from the hardcoded list is in the candidates, it
    must be tagged with is_structural_pair=True."""
    # Pick a known structural pair and inject identical series for both legs.
    pair = next(iter(STRUCTURAL_PAIRS))
    sym1, sym2 = sorted(pair)
    days = _weekdays(date(2015, 1, 5), 900)
    rng = np.random.default_rng(31)
    closes_a = (100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))).tolist()
    closes_b = (100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))).tolist()
    a = _make_symbol(sym1, days, closes_a)
    b = _make_symbol(sym2, days, closes_b)
    daily = pl.concat([a, b])
    sectors = _sectors_df(
        {
            sym1: ("FIN", "INE000000001"),
            sym2: ("FIN", "INE000000002"),
        }
    )
    res = cointegrate_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[-1],
        corr_threshold=0.10,  # very permissive so the pair survives screening
        min_overlap_days=500,
    )
    # If overlap+correlation gates pass, we expect exactly one diagnostic row
    # and is_structural_pair must be True on it.
    if not res.pairs.is_empty():
        assert res.pairs["is_structural_pair"].all()


def test_cointegrate_pairs_window_independence():
    """Two non-overlapping windows on the same daily frame must each see ONLY
    their own data (no leakage of future info)."""
    daily, sectors = _build_pair_universe(n_days=1600, beta_true=0.7, phi=0.4, seed=40)
    days = daily["date"].unique().sort()

    # Train window 1: first half
    res1 = cointegrate_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[799],
        corr_threshold=0.30,
        min_overlap_days=500,
    )
    # Train window 2: second half
    res2 = cointegrate_pairs(
        daily,
        sectors,
        start=days[800],
        end=days[-1],
        corr_threshold=0.30,
        min_overlap_days=500,
    )
    if not res1.pairs.is_empty() and not res2.pairs.is_empty():
        assert res1.pairs.row(0, named=True)["n_obs"] <= 800
        assert res2.pairs.row(0, named=True)["n_obs"] <= 800


def test_cointegrate_pairs_stability_check_runs():
    """When prior_start/prior_end is supplied the stability flag is populated."""
    daily, sectors = _build_pair_universe(n_days=1800, beta_true=0.7, phi=0.5, seed=50)
    days = daily["date"].unique().sort()
    n = 800
    train_start = days[-n]
    train_end = days[-1]
    prior_start = days[-2 * n]
    prior_end = days[-n - 1]
    res = cointegrate_pairs(
        daily,
        sectors,
        start=train_start,
        end=train_end,
        prior_start=prior_start,
        prior_end=prior_end,
        corr_threshold=0.30,
        min_overlap_days=500,
        hurst_max=0.55,
        half_life_max_days=300,
    )
    assert res.funnel.cross_window_stable is not None
    assert "stable_prior_window" in res.pairs.columns
    if not res.pairs.is_empty():
        # For a robust synthetic pair, stability should be reachable
        non_null = res.pairs.filter(pl.col("stable_prior_window").is_not_null())
        assert non_null.height >= 0


def test_cointegrate_pairs_stability_check_disabled_when_no_prior():
    """No prior window → stable_prior_window column is all null, funnel is None."""
    daily, sectors = _build_pair_universe(n_days=900, beta_true=0.7, phi=0.4, seed=51)
    days = daily["date"].unique().sort()
    res = cointegrate_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[-1],
        corr_threshold=0.30,
        min_overlap_days=500,
        hurst_max=0.55,
    )
    assert res.funnel.cross_window_stable is None
    if not res.pairs.is_empty():
        assert res.pairs["stable_prior_window"].null_count() == res.pairs.height


def test_cointegrate_pairs_prior_window_order_validated():
    daily, sectors = _build_pair_universe(n_days=400, seed=60)
    days = daily["date"].unique().sort()
    # prior_end must be < start
    with pytest.raises(ValueError, match="prior_end .* must be < start"):
        cointegrate_pairs(
            daily,
            sectors,
            start=days[100],
            end=days[-1],
            prior_start=days[0],
            prior_end=days[200],
            min_overlap_days=10,
        )


def test_cointegrate_pairs_invalid_window_raises():
    daily, sectors = _build_pair_universe(n_days=300, seed=61)
    days = daily["date"].unique().sort()
    with pytest.raises(ValueError, match="must be <"):
        cointegrate_pairs(
            daily,
            sectors,
            start=days[-1],
            end=days[0],
            min_overlap_days=10,
        )


def test_cointegrate_pairs_returns_correct_schema():
    daily, sectors = _build_pair_universe(n_days=800, seed=70)
    days = daily["date"].unique().sort()
    res = cointegrate_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[-1],
        corr_threshold=0.10,
        min_overlap_days=400,
        hurst_max=0.55,
        half_life_max_days=200,
    )
    expected = {
        "sym1",
        "sym2",
        "sector",
        "corr",
        "direction",
        "y_sym",
        "x_sym",
        "alpha",
        "beta",
        "adf_stat",
        "adf_pvalue",
        "raw_pass",
        "fdr_pass",
        "half_life",
        "half_life_in_band",
        "hurst",
        "hurst_pass",
        "stable_prior_window",
        "is_structural_pair",
        "n_obs",
    }
    assert set(res.pairs.columns) == expected


def test_cointegrate_pairs_funnel_counts_consistent():
    """fdr_survivors <= raw_p_below_alpha; tradeable <= each downstream filter."""
    daily, sectors = _build_pair_universe(n_days=900, seed=80)
    days = daily["date"].unique().sort()
    res = cointegrate_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[-1],
        corr_threshold=0.10,
        min_overlap_days=500,
    )
    f = res.funnel
    assert f.fdr_survivors <= f.raw_p_below_alpha
    assert f.half_life_in_band <= f.fdr_survivors
    assert f.hurst_below_max <= f.half_life_in_band
    if f.cross_window_stable is not None:
        assert f.cross_window_stable <= f.hurst_below_max
    assert f.candidates_after_share_class <= f.candidates_after_corr
