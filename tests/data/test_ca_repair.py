"""Tests for apt.data.ca_repair — classifier + ADJUST + KEEP-guarded TRIM."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from apt.data.ca_repair import (
    ADJUST_FACTORS,
    COVID_WINDOW,
    _classify_one,
    apply_repair,
    classify_survivors,
)
from apt.data.clean import validation_gate


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _make_daily(sym: str, dates: list[date], closes: list[float]) -> pl.DataFrame:
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
# _classify_one + classify_survivors
# ---------------------------------------------------------------------------


def test_classify_keep_hardcoded():
    cat, fac, _ = _classify_one(
        "YESBANK",
        date(2020, 3, 6),
        -0.5,
        keep_set={("YESBANK", date(2020, 3, 6))},
    )
    assert cat == "KEEP_HARDCODED"
    assert fac is None


def test_classify_covid_window_inclusive_endpoints():
    """Both 02-24 and 04-03 are inclusive endpoints."""
    for d in (COVID_WINDOW[0], COVID_WINDOW[1]):
        cat, _, _ = _classify_one("X", d, -0.5, keep_set=set())
        assert cat == "KEEP_COVID"


def test_classify_outside_covid_is_not_keep():
    """2020-04-30 (VTL date) is OUTSIDE the tightened window."""
    cat, _, _ = _classify_one("VTL", date(2020, 4, 30), -0.4069, keep_set=set())
    assert cat == "TRIM"


def test_classify_adjust_clean_factor_within_tol():
    cat, fac, note = _classify_one(
        "MAHSEAMLES", date(2020, 6, 22), -0.5107, keep_set=set()
    )
    assert cat == "ADJUST"
    assert fac == 0.50
    assert "0.4893" in note  # ratio


@pytest.mark.parametrize(
    "ret,expected_factor",
    [
        (-0.5107, 0.50),    # MAHSEAMLES-style
        (-0.66667, 0.333),  # 2:1 bonus / 3:1 effective
        (-0.7659, 0.25),    # NETWORK18 2012 style (ratio 0.2341)
        (-0.7989, 0.20),    # 4:1 bonus
        (-0.9, 0.10),       # 9:1 bonus
    ],
)
def test_classify_adjust_each_factor(ret: float, expected_factor: float):
    cat, fac, _ = _classify_one("X", date(2018, 1, 1), ret, keep_set=set())
    assert cat == "ADJUST"
    assert fac == expected_factor


def test_classify_trim_down_no_clean_factor():
    # SHRIRAMCIT-style ratio 0.4606 — 0.04 from 0.50, too far for tol=0.03
    cat, _, _ = _classify_one("SHRIRAMCIT", date(2020, 4, 30), -0.5394, keep_set=set())
    assert cat == "TRIM"


def test_classify_trim_up_move():
    cat, _, _ = _classify_one("PAGEIND", date(2011, 8, 30), +5.34, keep_set=set())
    assert cat == "TRIM"


def test_classify_survivors_frame_end_to_end():
    survivors = pl.DataFrame(
        {
            "symbol": ["MAH", "VTL", "YESBANK", "EIHOTEL", "PAGEIND"],
            "date": [
                date(2020, 6, 22),
                date(2020, 4, 30),
                date(2020, 3, 6),
                date(2020, 3, 20),
                date(2011, 8, 30),
            ],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "ret": [-0.5107, -0.4069, -0.5, -0.43, 5.34],
        }
    )
    out = classify_survivors(survivors, keep_events=[("YESBANK", date(2020, 3, 6))])
    got = dict(zip(out["symbol"], out["category"], strict=True))
    assert got == {
        "MAH": "ADJUST",
        "VTL": "TRIM",
        "YESBANK": "KEEP_HARDCODED",
        "EIHOTEL": "KEEP_COVID",
        "PAGEIND": "TRIM",
    }


# ---------------------------------------------------------------------------
# apply_repair — ADJUST
# ---------------------------------------------------------------------------


def test_apply_repair_back_adjusts_pre_ex_date():
    days = _weekdays(date(2020, 1, 1), 10)
    closes = [200.0] * 5 + [100.0] * 5  # naive 2:1 bonus on day 5
    df = _make_daily("XSP", days, closes)
    cl = pl.DataFrame(
        {
            "symbol": ["XSP"],
            "date": [days[5]],
            "close": [100.0],
            "ret": [-0.5],
            "category": ["ADJUST"],
            "adjust_factor": [0.5],
            "classification_note": [""],
        }
    )
    out, summary = apply_repair(df, cl)
    out = out.sort("date")
    cs = out["close"].to_list()
    assert cs[:5] == [pytest.approx(100.0)] * 5
    assert cs[5:] == [pytest.approx(100.0)] * 5
    assert summary["n_adjust_events"] == 1


def test_apply_repair_compounds_two_adjusts_same_symbol():
    """Two ADJUSTs on the same symbol multiply correctly (matches Rule 2)."""
    days = _weekdays(date(2020, 1, 1), 12)
    # Naive unadjusted: 1000 (×10 deflated), then 500 (×5), then 100 (normal)
    closes = [1000.0] * 4 + [500.0] * 4 + [100.0] * 4
    df = _make_daily("X2", days, closes)
    cl = pl.DataFrame(
        {
            "symbol": ["X2", "X2"],
            "date": [days[4], days[8]],
            "close": [500.0, 100.0],
            "ret": [-0.5, -0.8],
            "category": ["ADJUST", "ADJUST"],
            "adjust_factor": [0.5, 0.2],
            "classification_note": ["", ""],
        }
    )
    out, _ = apply_repair(df, cl)
    out = out.sort("date")
    cs = out["close"].to_list()
    assert cs[:4] == [pytest.approx(100.0)] * 4
    assert cs[4:8] == [pytest.approx(100.0)] * 4
    assert cs[8:] == [pytest.approx(100.0)] * 4


# ---------------------------------------------------------------------------
# apply_repair — TRIM (left + right + KEEP-guard)
# ---------------------------------------------------------------------------


def test_apply_repair_left_trim_drops_pre_cutoff():
    days = _weekdays(date(2020, 1, 1), 10)
    df = _make_daily("PH", days, [100.0] * 10)
    cl = pl.DataFrame(
        {
            "symbol": ["PH"],
            "date": [days[5]],
            "close": [100.0],
            "ret": [5.0],
            "category": ["TRIM"],
            "adjust_factor": [None],
            "classification_note": [""],
        }
    )
    out, summary = apply_repair(df, cl)
    assert out["date"].min() == days[5]
    assert summary["n_trim_left_symbols"] == 1
    assert summary["n_trim_right_symbols"] == 0


def test_apply_repair_keep_guard_blocks_left_trim_past_keep():
    """KEEP before all TRIMs → no left trim (no pre-keep TRIMs)."""
    days = _weekdays(date(2020, 1, 1), 20)
    df = _make_daily("IFB", days, [100.0] * 20)
    cl = pl.DataFrame(
        {
            "symbol": ["IFB", "IFB"],
            "date": [days[5], days[15]],
            "close": [100.0, 100.0],
            "ret": [-0.5, 1.0],
            "category": ["KEEP_COVID", "TRIM"],
            "adjust_factor": [None, None],
            "classification_note": ["", ""],
        }
    )
    out, summary = apply_repair(df, cl)
    # No left trim — KEEP precedes the only TRIM, which becomes a right trim.
    assert out["date"].min() == days[0]
    assert out["date"].max() < days[15]
    assert summary["n_trim_left_symbols"] == 0
    assert summary["n_trim_right_symbols"] == 1


def test_apply_repair_both_left_and_right_trim_around_keep():
    days = _weekdays(date(2020, 1, 1), 20)
    df = _make_daily("X", days, [100.0] * 20)
    cl = pl.DataFrame(
        {
            "symbol": ["X", "X", "X"],
            "date": [days[3], days[10], days[15]],
            "close": [100.0, 100.0, 100.0],
            "ret": [5.0, -0.5, 5.0],
            "category": ["TRIM", "KEEP_COVID", "TRIM"],
            "adjust_factor": [None, None, None],
            "classification_note": ["", "", ""],
        }
    )
    out, summary = apply_repair(df, cl)
    # Left-trim to day 3; right-trim drops day 15 onwards.
    assert out["date"].min() == days[3]
    assert out["date"].max() < days[15]
    assert summary["n_trim_left_symbols"] == 1
    assert summary["n_trim_right_symbols"] == 1


def test_apply_repair_empty_classified_is_noop():
    days = _weekdays(date(2020, 1, 1), 5)
    df = _make_daily("Z", days, [10.0] * 5)
    cl = pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "date": pl.Date,
            "close": pl.Float64,
            "ret": pl.Float64,
            "category": pl.Utf8,
            "adjust_factor": pl.Float64,
            "classification_note": pl.Utf8,
        }
    )
    out, summary = apply_repair(df, cl)
    assert out.height == 5
    assert summary == {
        "n_adjust_events": 0,
        "n_trim_left_symbols": 0,
        "n_trim_right_symbols": 0,
        "adjusts": [],
        "trims_left": [],
        "trims_right": [],
    }


# ---------------------------------------------------------------------------
# validation_gate keep_windows kwarg
# ---------------------------------------------------------------------------


def test_validation_gate_keep_windows_excuses_in_window():
    days = _weekdays(date(2020, 2, 17), 30)
    closes = [100.0] * 5 + [55.0] + [60.0] * 24  # -45% in window
    df = _make_daily("Q", days, closes)
    gate = validation_gate(
        df,
        _empty_actions(),
        start_date=date(2011, 1, 1),
        threshold=0.40,
        keep_windows=[COVID_WINDOW],
    )
    assert gate["pass"] is True
    assert gate["n_excused_by_window"] == 1


def test_validation_gate_keep_windows_does_not_excuse_outside():
    days = _weekdays(date(2020, 4, 6), 30)
    closes = [100.0] * 5 + [55.0] + [60.0] * 24  # -45% on 2020-04-13, outside window
    df = _make_daily("Q", days, closes)
    gate = validation_gate(
        df,
        _empty_actions(),
        start_date=date(2011, 1, 1),
        threshold=0.40,
        keep_windows=[COVID_WINDOW],
    )
    assert gate["pass"] is False
    assert gate["n_survivors"] == 1


# ---------------------------------------------------------------------------
# Sanity: the ADJUST_FACTORS list is well-formed
# ---------------------------------------------------------------------------


def test_adjust_factors_are_in_unit_interval_and_unique():
    assert all(0 < f < 1 for f in ADJUST_FACTORS)
    assert len(set(ADJUST_FACTORS)) == len(ADJUST_FACTORS)
