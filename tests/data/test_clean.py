"""Tests for apt.data.clean — every rule + the validation gate."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from apt.data.clean import (
    KEEP_EVENTS,
    RESIDUAL_SPLITS,
    STRUCTURAL_EVENTS,
    apply_calendar_filter,
    apply_contiguity_filter,
    apply_liquidity_filter,
    apply_min_history,
    apply_residual_splits,
    apply_structural_events,
    build_trading_calendar,
    max_internal_gap_per_symbol,
    trim_phantom_history,
    validation_gate,
    verify_split_smoothness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weekdays(start: date, n: int) -> list[date]:
    """Return n consecutive weekdays starting from ``start`` (skipping Sat/Sun)."""
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _make_daily(
    symbol: str, dates: list[date], closes: list[float], *, volume: list[float] | None = None
) -> pl.DataFrame:
    vol = volume if volume is not None else [1_000_000] * len(dates)
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": vol,
        }
    )


def _empty_actions() -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [], "action_type": [], "ex_date": [], "value": []},
        schema={
            "symbol": pl.Utf8,
            "action_type": pl.Utf8,
            "ex_date": pl.Date,
            "value": pl.Float64,
        },
    )


# ---------------------------------------------------------------------------
# Rule 1 — calendar
# ---------------------------------------------------------------------------


def test_build_calendar_excludes_weekend_and_jan1():
    # Build a daily frame across 30 days with all 492 symbols on every day
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(30)]
    frames = []
    for sym_idx in range(150):
        frames.append(_make_daily(f"S{sym_idx:03d}", days, [100.0] * len(days)))
    df = pl.concat(frames)

    calendar = build_trading_calendar(
        df,
        start=date(2020, 1, 1),
        end=date(2020, 1, 30),
        min_symbols_per_day=100,
    )
    # Jan 1 is a national holiday → excluded
    assert date(2020, 1, 1) not in calendar
    # Jan 26 (Republic Day) is excluded too — within window for this test
    assert date(2020, 1, 26) not in calendar
    # All weekends excluded
    assert all(d.weekday() < 5 for d in calendar)


def test_calendar_filter_drops_off_calendar_rows():
    days = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]  # all weekdays
    df = _make_daily("AAA", days, [100, 101, 102])
    # Add an off-calendar row (Saturday)
    extra = _make_daily("AAA", [date(2020, 1, 11)], [999])  # Saturday
    df = pl.concat([df, extra])

    out, report = apply_calendar_filter(df, days)
    assert out.height == 3
    assert report["rows_dropped"] == 1


# ---------------------------------------------------------------------------
# Rule 2 — residual splits
# ---------------------------------------------------------------------------


def test_apply_residual_split_back_adjusts_pre_dates():
    days = _weekdays(date(2020, 1, 1), 10)
    # Unadjusted: 200 pre-split, 100 post-split (a 2:1)
    closes = [200.0] * 5 + [100.0] * 5
    df = _make_daily("XSP", days, closes)
    splits = [("XSP", days[5], 2.0)]
    out, report = apply_residual_splits(df, splits)
    out = out.sort("date")
    closes_out = out["close"].to_list()
    # Pre-split divided by 2 → 100 each
    assert closes_out[:5] == [pytest.approx(100.0)] * 5
    # Post-split unchanged
    assert closes_out[5:] == [pytest.approx(100.0)] * 5
    assert report["rows_adjusted"] == 5


def test_verify_split_smoothness_after_patch():
    days = _weekdays(date(2020, 1, 1), 6)
    closes = [200.0] * 3 + [100.0] * 3  # unadjusted 2:1 on day 3
    df = _make_daily("XSP", days, closes)
    splits = [("XSP", days[3], 2.0)]
    patched, _ = apply_residual_splits(df, splits)
    checks = verify_split_smoothness(patched, splits, tolerance=0.05)
    assert checks[0]["smooth"] is True
    assert checks[0]["post_patch_return"] == pytest.approx(0.0, abs=1e-9)


def test_residual_splits_handles_two_overlapping_splits_same_symbol():
    """Two pre-existing splits compound the back-adjustment multiplier."""
    days = _weekdays(date(2020, 1, 1), 12)
    # Pre 2:1 (day 4), pre 5:1 (day 8). Real unadjusted timeline:
    closes = [1000.0] * 4 + [500.0] * 4 + [100.0] * 4  # naive: 1000, then 500 (÷2), then 100 (÷5)
    df = _make_daily("X2", days, closes)
    splits = [("X2", days[4], 2.0), ("X2", days[8], 5.0)]
    out, _ = apply_residual_splits(df, splits)
    out = out.sort("date")
    closes_out = out["close"].to_list()
    # Pre-day 4: ÷(2 * 5) = ÷10 → 100; day 4-7: ÷5 → 100; day 8-11: 100.
    assert closes_out[0:4] == [pytest.approx(100.0)] * 4
    assert closes_out[4:8] == [pytest.approx(100.0)] * 4
    assert closes_out[8:12] == [pytest.approx(100.0)] * 4


# ---------------------------------------------------------------------------
# Rule 3 — phantom trim
# ---------------------------------------------------------------------------


def test_trim_phantom_drops_pre_seam_rows():
    days = _weekdays(date(2020, 1, 1), 20)
    closes = [10.0] * 10 + [200.0] + [201.0] * 9  # 20x jump on day 10 (no split)
    df = _make_daily("PH", days, closes)
    out, report = trim_phantom_history(df, threshold=0.65, splits=[], structural=[], keep=[])
    assert out["date"].min() == days[10]
    assert report["trimmed_symbols"][0]["new_start_date"] == days[10]
    assert report["rows_dropped"] == 10


def test_trim_phantom_keeps_excused_keep_event():
    days = _weekdays(date(2020, 1, 1), 10)
    closes = [100.0] * 5 + [20.0] + [21.0] * 4  # crash day 5 → -80% return
    df = _make_daily("YES", days, closes)
    keep = [("YES", days[5])]
    out, report = trim_phantom_history(
        df, threshold=0.65, splits=[], structural=[], keep=keep
    )
    assert out.height == 10
    assert report["trimmed_symbols"] == []
    assert report["rows_dropped"] == 0


def test_trim_phantom_ignores_split_dates():
    days = _weekdays(date(2020, 1, 1), 10)
    closes = [200.0] * 5 + [100.0] * 5  # unadjusted 2:1 on day 5 → -50% return (under 65%)
    df = _make_daily("XSP", days, closes)
    # Even if it WERE over threshold, the split date is excused
    out, report = trim_phantom_history(
        df, threshold=0.30, splits=[("XSP", days[5], 2.0)], structural=[], keep=[]
    )
    assert out.height == 10
    assert report["rows_dropped"] == 0


# ---------------------------------------------------------------------------
# Rule 4 — structural events
# ---------------------------------------------------------------------------


def test_structural_events_keeps_only_post_event():
    days = _weekdays(date(2020, 1, 1), 10)
    df = _make_daily("IDE", days, [100.0] * 10)
    events = [("IDE", days[4])]
    out, report = apply_structural_events(df, events)
    assert out["date"].min() == days[5]
    assert report["rows_dropped"] == 5  # days 0..4 inclusive (strictly AFTER event)


def test_structural_events_unaffected_symbols_pass_through():
    days = _weekdays(date(2020, 1, 1), 6)
    df = pl.concat([_make_daily("A", days, [1.0] * 6), _make_daily("B", days, [1.0] * 6)])
    out, _ = apply_structural_events(df, [("A", days[3])])
    assert out.filter(pl.col("symbol") == "B").height == 6
    assert out.filter(pl.col("symbol") == "A").height == 2  # days 4, 5


# ---------------------------------------------------------------------------
# Rule 5 — liquidity floor
# ---------------------------------------------------------------------------


def test_liquidity_drops_low_adv_rows():
    days = _weekdays(date(2020, 1, 1), 60)
    # Constant price 100, volume varies: first 30 days vol=100 → ADV 10_000;
    # next 30 days vol=1_000_000 → ADV 100M.
    vol = [100] * 30 + [1_000_000] * 30
    df = _make_daily("XLIQ", days, [100.0] * 60, volume=vol)
    out, report = apply_liquidity_filter(
        df, min_adv_inr=1_000_000, window=20, min_periods=5
    )
    # First days mostly below floor, last days above. We expect at least
    # the later block to survive.
    assert out["date"].min() >= days[5]  # min_periods warmup
    assert out["date"].max() == days[-1]
    assert report["rows_dropped"] > 0
    assert report["floor_inr"] == 1_000_000


# ---------------------------------------------------------------------------
# Rule 7 — contiguity
# ---------------------------------------------------------------------------


def test_contiguity_no_gaps_is_noop():
    days = _weekdays(date(2020, 1, 1), 30)
    df = _make_daily("X", days, [100.0] * 30)
    out, report = apply_contiguity_filter(df, max_gap_days=10)
    assert out.height == 30
    assert report["n_symbols_segmented"] == 0
    assert report["rows_dropped"] == 0


def test_contiguity_normal_weekend_is_under_threshold():
    """A Fri→Mon transition is 3 calendar days — under the default 10-day threshold."""
    # First 5 weekdays (Mon-Fri), then next 5 weekdays (Mon-Fri) — 3-day weekend gap.
    days = _weekdays(date(2020, 1, 6), 10)
    df = _make_daily("X", days, [100.0] * 10)
    out, report = apply_contiguity_filter(df, max_gap_days=10)
    assert out.height == 10
    assert report["n_segments_total"] == 1


def test_contiguity_splits_on_large_gap_keeps_longest():
    """20-day stretch, then 30-calendar-day gap, then 5-day stretch → keep 20."""
    a = _weekdays(date(2020, 1, 1), 20)
    b = _weekdays(date(2020, 4, 1), 5)  # ~70 calendar days after a's end
    df = _make_daily("G", a + b, [100.0] * 25)
    out, report = apply_contiguity_filter(
        df, max_gap_days=10, prefer_overlap_after=date(1900, 1, 1)
    )
    assert out.height == 20
    assert out["date"].min() == a[0]
    assert out["date"].max() == a[-1]
    assert report["n_symbols_segmented"] == 1
    assert report["n_segments_total"] == 2
    assert report["rows_dropped"] == 5


def test_contiguity_prefers_2015_overlap_over_longer_pre_segment():
    """A long pre-2015 segment loses to a shorter 2015+ segment."""
    # 100-day pre-2015 segment
    pre = _weekdays(date(2010, 1, 4), 100)
    # 30-day 2015+ segment after a large gap
    post = _weekdays(date(2016, 1, 4), 30)
    df = _make_daily("J", pre + post, [10.0] * 130)
    out, report = apply_contiguity_filter(
        df, max_gap_days=10, prefer_overlap_after=date(2015, 1, 1)
    )
    # Should keep the shorter 2015+ segment
    assert out.height == 30
    assert out["date"].min() >= date(2015, 1, 1)
    assert report["rows_dropped"] == 100


def test_contiguity_falls_back_to_longest_when_no_2015_overlap():
    """If no segment overlaps the preference date, longest wins."""
    a = _weekdays(date(2008, 1, 1), 50)
    b = _weekdays(date(2012, 1, 2), 20)
    df = _make_daily("K", a + b, [10.0] * 70)
    out, report = apply_contiguity_filter(
        df, max_gap_days=10, prefer_overlap_after=date(2015, 1, 1)
    )
    assert out.height == 50
    assert out["date"].max() <= a[-1]
    assert report["rows_dropped"] == 20


def test_contiguity_multi_symbol_independent_decisions():
    """Symbol A is contiguous; symbol B is split. Filter affects only B."""
    days_a = _weekdays(date(2018, 1, 1), 40)
    a_df = _make_daily("A", days_a, [10.0] * 40)
    pre_b = _weekdays(date(2018, 1, 1), 5)
    post_b = _weekdays(date(2018, 4, 1), 30)
    b_df = _make_daily("B", pre_b + post_b, [10.0] * 35)
    df = pl.concat([a_df, b_df])
    out, _ = apply_contiguity_filter(df, max_gap_days=10)
    a_after = out.filter(pl.col("symbol") == "A").height
    b_after = out.filter(pl.col("symbol") == "B").height
    assert a_after == 40  # untouched
    assert b_after == 30  # kept the longer post segment (also 2015+)


def test_contiguity_post_filter_max_gap_at_or_below_threshold():
    """Every kept symbol's max internal gap must be ≤ threshold."""
    days_a = _weekdays(date(2020, 1, 1), 50)
    a_df = _make_daily("A", days_a, [10.0] * 50)
    # Symbol B with two segments
    pre_b = _weekdays(date(2020, 1, 1), 5)
    post_b = _weekdays(date(2020, 5, 1), 25)
    b_df = _make_daily("B", pre_b + post_b, [10.0] * 30)
    df = pl.concat([a_df, b_df])
    out, _ = apply_contiguity_filter(df, max_gap_days=10)
    gaps = max_internal_gap_per_symbol(out)
    assert int(gaps["max_internal_gap_days"].max()) <= 10


def test_contiguity_single_row_symbol_passes():
    """Edge case: symbol with only one row should pass through unchanged."""
    df = _make_daily("S", [date(2020, 6, 1)], [100.0])
    out, _ = apply_contiguity_filter(df, max_gap_days=10)
    assert out.height == 1


# ---------------------------------------------------------------------------
# Rule 6 — min history
# ---------------------------------------------------------------------------


def test_min_history_drops_short_symbols():
    days = _weekdays(date(2020, 1, 1), 100)
    long_sym = _make_daily("LONG", days, [1.0] * 100)
    short_sym = _make_daily("SHORT", days[:10], [1.0] * 10)
    df = pl.concat([long_sym, short_sym])
    out, report = apply_min_history(df, min_days=50)
    syms = set(out["symbol"].unique().to_list())
    assert syms == {"LONG"}
    assert any(d["symbol"] == "SHORT" for d in report["symbols_dropped"])


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------


def test_validation_gate_passes_on_smooth_series():
    days = _weekdays(date(2011, 1, 3), 100)
    df = _make_daily("OK", days, [100.0 + 0.05 * i for i in range(100)])
    gate = validation_gate(df, _empty_actions(), start_date=date(2011, 1, 1), threshold=0.40)
    assert gate["pass"] is True
    assert gate["n_survivors"] == 0


def test_validation_gate_excuses_dividend_ex_date():
    days = _weekdays(date(2011, 1, 3), 20)
    closes = [100.0] * 10 + [50.0] + [50.05] * 9  # -50% on day 10 = "huge special div"
    df = _make_daily("SPDIV", days, closes)
    actions = pl.DataFrame(
        {
            "symbol": ["SPDIV"],
            "action_type": ["dividend"],
            "ex_date": [days[10]],
            "value": [50.0],
        }
    )
    gate = validation_gate(df, actions, start_date=date(2011, 1, 1), threshold=0.40)
    assert gate["pass"] is True
    assert gate["n_big_moves"] == 1
    assert gate["n_survivors"] == 0


def test_validation_gate_excuses_keep_event():
    days = _weekdays(date(2011, 1, 3), 20)
    closes = [100.0] * 10 + [20.0] + [21.0] * 9  # -80% on day 10
    df = _make_daily("KEEPER", days, closes)
    keep = [("KEEPER", days[10])]
    gate = validation_gate(
        df, _empty_actions(), start_date=date(2011, 1, 1), threshold=0.40, keep=keep
    )
    assert gate["pass"] is True
    assert gate["n_survivors"] == 0


def test_validation_gate_fails_on_unexplained_survivor():
    days = _weekdays(date(2011, 1, 3), 20)
    closes = [100.0] * 10 + [40.0] + [40.5] * 9  # -60% with no excuse
    df = _make_daily("BAD", days, closes)
    gate = validation_gate(df, _empty_actions(), start_date=date(2011, 1, 1), threshold=0.40)
    assert gate["pass"] is False
    assert gate["n_survivors"] == 1
    assert gate["survivors"][0]["symbol"] == "BAD"


def test_validation_gate_tracks_yfinance_blind_survivors():
    days = _weekdays(date(2011, 1, 3), 20)
    closes = [100.0] * 10 + [40.0] + [40.5] * 9
    df = _make_daily("BLIND", days, closes)
    gate = validation_gate(
        df,
        _empty_actions(),
        start_date=date(2011, 1, 1),
        threshold=0.40,
        yfinance_failed_symbols={"BLIND"},
    )
    assert gate["pass"] is False
    assert gate["n_yfinance_blind_total"] == 1
    assert gate["n_yfinance_blind_survivors"] == 1


# ---------------------------------------------------------------------------
# Sanity: the hardcoded event lists are well-formed
# ---------------------------------------------------------------------------


def test_event_constants_are_well_formed():
    for sym, d, ratio in RESIDUAL_SPLITS:
        assert isinstance(sym, str) and len(sym) > 0
        assert isinstance(d, date)
        assert ratio > 1.0  # all our residual splits are forward splits
    for sym, d in STRUCTURAL_EVENTS:
        assert isinstance(sym, str) and len(sym) > 0
        assert isinstance(d, date)
    for sym, d in KEEP_EVENTS:
        assert isinstance(sym, str) and len(sym) > 0
        assert isinstance(d, date)
