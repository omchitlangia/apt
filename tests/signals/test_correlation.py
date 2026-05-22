"""Tests for apt.signals.correlation — windowed, gap-guarded, same-sector
correlation pre-filter."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from apt.signals.correlation import (
    WindowCorrelation,
    compute_window_correlation,
    screen_pairs,
    window_eligible_symbols,
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


def _sectors_df(rows: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": list(rows.keys()),
            "company_name": list(rows),
            "industry": list(rows.values()),
            "isin": [f"IN{i:010d}" for i in range(len(rows))],
            "bse_industry": [""] * len(rows),
        }
    )


# ---------------------------------------------------------------------------
# window_eligible_symbols
# ---------------------------------------------------------------------------


def test_eligibility_passes_full_coverage_symbols():
    days = _weekdays(date(2020, 1, 1), 60)
    a = _make_symbol("A", days, [10.0 + i * 0.1 for i in range(60)])
    b = _make_symbol("B", days, [20.0 + i * 0.2 for i in range(60)])
    df = pl.concat([a, b])
    el = window_eligible_symbols(df, start=days[0], end=days[-1])
    assert set(el) == {"A", "B"}


def test_eligibility_excludes_missing_start():
    days = _weekdays(date(2020, 1, 1), 60)
    # Symbol B starts later — won't have a row on `start`
    a = _make_symbol("A", days, [10.0] * 60)
    b = _make_symbol("B", days[5:], [20.0] * 55)
    df = pl.concat([a, b])
    el = window_eligible_symbols(df, start=days[0], end=days[-1])
    assert el == ["A"]


def test_eligibility_excludes_missing_end():
    days = _weekdays(date(2020, 1, 1), 60)
    a = _make_symbol("A", days, [10.0] * 60)
    b = _make_symbol("B", days[:-5], [20.0] * 55)  # ends early
    df = pl.concat([a, b])
    el = window_eligible_symbols(df, start=days[0], end=days[-1])
    assert el == ["A"]


def test_eligibility_excludes_internal_gap():
    days = _weekdays(date(2020, 1, 1), 60)
    a = _make_symbol("A", days, [10.0] * 60)
    # Symbol B has a 30-calendar-day hole in the middle
    days_hole = days[:20] + days[40:]
    b = _make_symbol("B", days_hole, [20.0] * len(days_hole))
    df = pl.concat([a, b])
    el = window_eligible_symbols(
        df, start=days[0], end=days[-1], max_internal_gap_days=10
    )
    assert el == ["A"]


def test_eligibility_normal_weekend_gap_under_threshold():
    """A 3-calendar-day Fri→Mon gap is below threshold; symbol passes."""
    days = _weekdays(date(2020, 1, 6), 20)  # starts Monday
    a = _make_symbol("A", days, [10.0] * 20)
    el = window_eligible_symbols(a, start=days[0], end=days[-1])
    assert el == ["A"]


# ---------------------------------------------------------------------------
# compute_window_correlation
# ---------------------------------------------------------------------------


def test_correlation_two_identical_series_is_one():
    days = _weekdays(date(2020, 1, 1), 50)
    rng = np.random.default_rng(0)
    closes_a = 100 * np.cumprod(1 + rng.normal(0, 0.01, 50))
    a = _make_symbol("A", days, closes_a.tolist())
    b = _make_symbol("B", days, closes_a.tolist())  # identical → corr = 1
    df = pl.concat([a, b])
    wc = compute_window_correlation(df, start=days[0], end=days[-1])
    assert isinstance(wc, WindowCorrelation)
    assert wc.eligible_symbols == ["A", "B"]
    assert wc.corr_matrix.shape == (2, 2)
    assert wc.corr_matrix[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_correlation_anti_correlated_series():
    days = _weekdays(date(2020, 1, 1), 50)
    rng = np.random.default_rng(1)
    closes_a = 100 * np.cumprod(1 + rng.normal(0, 0.01, 50))
    # B has exactly the inverse log-returns: 1/ratio for each step
    rets_a = np.diff(np.log(closes_a))
    closes_b = 100 * np.exp(np.concatenate([[0], -rets_a]).cumsum())
    a = _make_symbol("A", days, closes_a.tolist())
    b = _make_symbol("B", days, closes_b.tolist())
    df = pl.concat([a, b])
    wc = compute_window_correlation(df, start=days[0], end=days[-1])
    assert wc.corr_matrix[0, 1] == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# screen_pairs — same-sector + threshold + windowing
# ---------------------------------------------------------------------------


def _three_symbol_universe(
    *,
    a_b_corr: float = 0.95,
    a_c_corr_target: float = 0.50,
    n_days: int = 80,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Three symbols: A and B highly correlated, A and C weakly correlated."""
    rng = np.random.default_rng(seed)
    days = _weekdays(date(2020, 1, 1), n_days)
    rets_a = rng.normal(0, 0.01, n_days)
    rets_b = a_b_corr * rets_a + np.sqrt(1 - a_b_corr**2) * rng.normal(
        0, 0.01, n_days
    )
    rets_c = a_c_corr_target * rets_a + np.sqrt(
        max(0, 1 - a_c_corr_target**2)
    ) * rng.normal(0, 0.01, n_days)

    def _series_from_rets(rets: np.ndarray) -> list[float]:
        # Start at 100, build closes from compounded returns.
        log = np.concatenate([[np.log(100)], np.log(100) + rets.cumsum()])
        return np.exp(log[: len(rets)]).tolist()

    a = _make_symbol("A", days, _series_from_rets(rets_a))
    b = _make_symbol("B", days, _series_from_rets(rets_b))
    c = _make_symbol("C", days, _series_from_rets(rets_c))
    daily = pl.concat([a, b, c])
    sectors = _sectors_df({"A": "SEC1", "B": "SEC1", "C": "SEC1"})
    return daily, sectors


def test_screen_pairs_keeps_above_threshold_same_sector():
    daily, sectors = _three_symbol_universe(a_b_corr=0.97, a_c_corr_target=0.40)
    days = daily["date"].unique().sort()
    pairs = screen_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[-1],
        corr_threshold=0.85,
    )
    # Exactly the AB pair survives.
    syms = sorted({(r["sym1"], r["sym2"]) for r in pairs.iter_rows(named=True)})
    assert syms == [("A", "B")]


def test_screen_pairs_drops_below_threshold():
    daily, sectors = _three_symbol_universe(a_b_corr=0.50, a_c_corr_target=0.30)
    days = daily["date"].unique().sort()
    pairs = screen_pairs(
        daily, sectors, start=days[0], end=days[-1], corr_threshold=0.85
    )
    assert pairs.is_empty()


def test_screen_pairs_excludes_cross_sector_high_corr():
    """Identical series in DIFFERENT sectors must not be paired."""
    days = _weekdays(date(2020, 1, 1), 60)
    closes = [100.0 + 0.1 * i for i in range(60)]
    a = _make_symbol("A", days, closes)
    b = _make_symbol("B", days, closes)  # identical → corr = 1
    daily = pl.concat([a, b])
    # Different sectors
    sectors = _sectors_df({"A": "SEC1", "B": "SEC2"})
    pairs = screen_pairs(
        daily, sectors, start=days[0], end=days[-1], corr_threshold=0.85
    )
    assert pairs.is_empty()


def test_screen_pairs_gap_guard_drops_symbol():
    """A symbol with a >threshold internal gap must be excluded entirely."""
    days = _weekdays(date(2020, 1, 1), 60)
    closes = np.linspace(100, 110, 60).tolist()
    a = _make_symbol("A", days, closes)
    b = _make_symbol("B", days, closes)
    # Symbol C has a huge mid-window hole
    days_c = days[:20] + days[40:]
    c = _make_symbol("C", days_c, np.linspace(100, 110, len(days_c)).tolist())
    daily = pl.concat([a, b, c])
    sectors = _sectors_df({"A": "S", "B": "S", "C": "S"})
    pairs = screen_pairs(
        daily,
        sectors,
        start=days[0],
        end=days[-1],
        corr_threshold=0.85,
        max_internal_gap_days=10,
    )
    # Only the AB pair appears (C excluded by gap guard)
    found = {(r["sym1"], r["sym2"]) for r in pairs.iter_rows(named=True)}
    assert found == {("A", "B")}


def test_screen_pairs_window_independence():
    """Different windows over the same daily produce different pair lists."""
    days = _weekdays(date(2020, 1, 1), 200)
    rng = np.random.default_rng(7)
    # First half: A & B move together; second half: they decouple
    rets_a = rng.normal(0, 0.01, 200)
    rets_b = np.concatenate(
        [
            0.98 * rets_a[:100]
            + 0.2 * rng.normal(0, 0.01, 100),  # high corr
            rng.normal(0, 0.01, 100),  # uncorrelated
        ]
    )
    a_closes = np.exp(np.concatenate([[np.log(100)], np.log(100) + rets_a.cumsum()]))
    b_closes = np.exp(np.concatenate([[np.log(100)], np.log(100) + rets_b.cumsum()]))
    a = _make_symbol("A", days, a_closes[:200].tolist())
    b = _make_symbol("B", days, b_closes[:200].tolist())
    daily = pl.concat([a, b])
    sectors = _sectors_df({"A": "S", "B": "S"})

    pairs_first = screen_pairs(
        daily, sectors, start=days[0], end=days[99], corr_threshold=0.85
    )
    pairs_second = screen_pairs(
        daily, sectors, start=days[100], end=days[-1], corr_threshold=0.85
    )
    assert not pairs_first.is_empty()
    assert pairs_second.is_empty()


def test_screen_pairs_thin_sector_yields_nothing():
    """A sector with a single eligible member can't produce within-sector pairs."""
    days = _weekdays(date(2020, 1, 1), 60)
    closes = [100.0 + 0.1 * i for i in range(60)]
    a = _make_symbol("A", days, closes)
    b = _make_symbol("B", days, closes)
    daily = pl.concat([a, b])
    sectors = _sectors_df({"A": "SOLO", "B": "OTHER"})
    pairs = screen_pairs(
        daily, sectors, start=days[0], end=days[-1], corr_threshold=0.85
    )
    assert pairs.is_empty()


def test_screen_pairs_invalid_window_raises():
    daily = _make_symbol(
        "A", _weekdays(date(2020, 1, 1), 10), [10.0] * 10
    )
    sectors = _sectors_df({"A": "S"})
    with pytest.raises(ValueError, match="must be <"):
        screen_pairs(daily, sectors, start=date(2020, 6, 1), end=date(2020, 1, 1))


def test_screen_pairs_returns_correct_schema():
    daily, sectors = _three_symbol_universe(a_b_corr=0.95)
    days = daily["date"].unique().sort()
    pairs = screen_pairs(daily, sectors, start=days[0], end=days[-1])
    assert pairs.columns == ["sym1", "sym2", "corr", "sector"]
    assert pairs.schema["corr"] == pl.Float64
    assert pairs.schema["sector"] == pl.Utf8
