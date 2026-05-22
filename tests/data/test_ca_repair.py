"""Tests for apt.data.ca_repair — classifier + KEEP-guarded TRIM (TRIM-only)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from apt.data.ca_repair import (
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
    cat, _ = _classify_one(
        "YESBANK",
        date(2020, 3, 6),
        -0.5,
        keep_set={("YESBANK", date(2020, 3, 6))},
    )
    assert cat == "KEEP_HARDCODED"


def test_classify_covid_window_inclusive_endpoints():
    """Both 02-24 and 04-03 are inclusive endpoints."""
    for d in (COVID_WINDOW[0], COVID_WINDOW[1]):
        cat, _ = _classify_one("X", d, -0.5, keep_set=set())
        assert cat == "KEEP_COVID"


def test_classify_outside_covid_is_not_keep():
    """2020-04-30 (VTL date) is OUTSIDE the tightened window."""
    cat, _ = _classify_one("VTL", date(2020, 4, 30), -0.4069, keep_set=set())
    assert cat == "TRIM"


def test_classify_clean_ratio_down_move_is_now_trim_not_adjust():
    """Removed ratio-snap ADJUST: a clean ratio (e.g. exactly 0.50) routes to TRIM."""
    cat, _ = _classify_one("MAHSEAMLES", date(2020, 6, 22), -0.5107, keep_set=set())
    assert cat == "TRIM"


@pytest.mark.parametrize(
    "ret",
    [-0.5107, -0.6667, -0.7659, -0.7989, -0.90, +5.34, +1.0],
)
def test_classify_all_non_keep_route_to_trim(ret: float):
    cat, _ = _classify_one("X", date(2018, 1, 1), ret, keep_set=set())
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
        "MAH": "TRIM",  # was ADJUST under the removed rule; now TRIM
        "VTL": "TRIM",
        "YESBANK": "KEEP_HARDCODED",
        "EIHOTEL": "KEEP_COVID",
        "PAGEIND": "TRIM",
    }


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
            "classification_note": [""],
        }
    )
    out, summary = apply_repair(df, cl)
    assert out["date"].min() == days[5]
    assert summary["n_trim_left_symbols"] == 1
    assert summary["n_trim_right_symbols"] == 0


def test_apply_repair_keep_guard_blocks_left_trim_past_keep():
    """KEEP before the only TRIM → that TRIM becomes a right trim, not a left."""
    days = _weekdays(date(2020, 1, 1), 20)
    df = _make_daily("IFB", days, [100.0] * 20)
    cl = pl.DataFrame(
        {
            "symbol": ["IFB", "IFB"],
            "date": [days[5], days[15]],
            "close": [100.0, 100.0],
            "ret": [-0.5, 1.0],
            "category": ["KEEP_COVID", "TRIM"],
            "classification_note": ["", ""],
        }
    )
    out, summary = apply_repair(df, cl)
    assert out["date"].min() == days[0]  # no left-trim
    assert out["date"].max() < days[15]  # right-trim from day 15
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
            "classification_note": ["", "", ""],
        }
    )
    out, summary = apply_repair(df, cl)
    assert out["date"].min() == days[3]
    assert out["date"].max() < days[15]
    assert summary["n_trim_left_symbols"] == 1
    assert summary["n_trim_right_symbols"] == 1


def test_apply_repair_multiple_trims_picks_latest_left_cutoff():
    """Two TRIM events on one symbol with no KEEP → cutoff is the later date."""
    days = _weekdays(date(2020, 1, 1), 20)
    df = _make_daily("Z", days, [100.0] * 20)
    cl = pl.DataFrame(
        {
            "symbol": ["Z", "Z"],
            "date": [days[3], days[10]],
            "close": [100.0, 100.0],
            "ret": [+5.0, +5.0],
            "category": ["TRIM", "TRIM"],
            "classification_note": ["", ""],
        }
    )
    out, summary = apply_repair(df, cl)
    assert out["date"].min() == days[10]
    assert summary["trims_left"][0]["n_trim_events"] == 2


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
            "classification_note": pl.Utf8,
        }
    )
    out, summary = apply_repair(df, cl)
    assert out.height == 5
    assert summary == {
        "n_trim_left_symbols": 0,
        "n_trim_right_symbols": 0,
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
