"""Tests for apt.data.corporate_actions — fetch, audit, flavor classifier,
passthrough. Network is fully stubbed unless a test is marked ``slow``."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from apt.data.corporate_actions import (
    _coefficient_of_variation,
    audit_split_residuals,
    classify_adjustment_flavor,
    fetch_all_corporate_actions,
    fetch_corporate_actions_for_symbol,
    passthrough_daily_adjusted,
    write_corporate_actions_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trading_dates(start: date, n: int) -> list[date]:
    """Simple monotonically increasing calendar days (good enough for tests)."""
    return [start + timedelta(days=i) for i in range(n)]


def _make_daily(symbol: str, dates: list[date], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * len(dates),
        }
    )


def _make_actions(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {"symbol": [], "action_type": [], "ex_date": [], "value": []},
            schema={
                "symbol": pl.Utf8,
                "action_type": pl.Utf8,
                "ex_date": pl.Date,
                "value": pl.Float64,
            },
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# fetch_corporate_actions_for_symbol — stubbed
# ---------------------------------------------------------------------------


def _fake_actions_df(
    splits: dict[str, float] | None = None, divs: dict[str, float] | None = None
) -> pd.DataFrame:
    """Build a pandas DataFrame matching yfinance Ticker.actions shape."""
    splits = splits or {}
    divs = divs or {}
    all_dates = sorted(set(splits) | set(divs))
    if not all_dates:
        return pd.DataFrame(columns=["Dividends", "Stock Splits"])
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="Asia/Kolkata") for d in all_dates], name="Date")
    return pd.DataFrame(
        {
            "Dividends": [divs.get(d, 0.0) for d in all_dates],
            "Stock Splits": [splits.get(d, 0.0) for d in all_dates],
        },
        index=idx,
    )


def test_fetch_success_returns_actions():
    def stub(sym: str) -> pd.DataFrame:
        assert sym == "AAA"
        return _fake_actions_df(splits={"2020-01-15": 2.0}, divs={"2020-06-01": 1.5})

    out = fetch_corporate_actions_for_symbol("AAA", ticker_fn=stub)
    assert out["error"] is None
    assert out["splits"] == [(date(2020, 1, 15), 2.0)]
    assert out["dividends"] == [(date(2020, 6, 1), 1.5)]


def test_fetch_empty_actions_counts_as_success():
    out = fetch_corporate_actions_for_symbol("BBB", ticker_fn=lambda s: _fake_actions_df())
    assert out["error"] is None
    assert out["splits"] == []
    assert out["dividends"] == []


def test_fetch_retries_then_fails():
    calls = {"n": 0}

    def stub(_sym: str):
        calls["n"] += 1
        raise RuntimeError("simulated rate limit")

    out = fetch_corporate_actions_for_symbol("CCC", ticker_fn=stub, max_retries=3, base_delay=0.0)
    assert out["error"] is not None
    assert "rate limit" in out["error"].lower()
    assert calls["n"] == 3


def test_fetch_recovers_on_retry():
    calls = {"n": 0}

    def stub(_sym: str):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return _fake_actions_df(splits={"2018-08-08": 5.0})

    out = fetch_corporate_actions_for_symbol("DDD", ticker_fn=stub, max_retries=3, base_delay=0.0)
    assert out["error"] is None
    assert out["splits"] == [(date(2018, 8, 8), 5.0)]


def test_fetch_all_writes_long_frame_and_coverage(tmp_path: Path):
    def stub(sym: str):
        if sym == "AAA":
            return _fake_actions_df(splits={"2017-09-21": 5.0})
        if sym == "BBB":
            return _fake_actions_df(divs={"2019-05-05": 1.25})
        return _fake_actions_df()

    actions, coverage = fetch_all_corporate_actions(["AAA", "BBB", "CCC"], n_jobs=1, ticker_fn=stub)
    assert actions.height == 2
    assert set(actions["symbol"].unique().to_list()) == {"AAA", "BBB"}
    assert coverage.height == 3
    assert (coverage["status"] == "ok").all()

    path = tmp_path / "out.parquet"
    write_corporate_actions_cache(actions, path)
    loaded = pl.read_parquet(path)
    assert loaded.height == 2
    assert loaded.schema["ex_date"] == pl.Date


# ---------------------------------------------------------------------------
# audit_split_residuals
# ---------------------------------------------------------------------------


def test_audit_flags_unadjusted_2for1():
    dates = _trading_dates(date(2020, 1, 1), 10)
    closes = [100.0] * 5 + [50.0] * 5  # split-day drop on dates[5]
    daily = _make_daily("XSPLIT", dates, closes)
    actions = _make_actions(
        [
            {
                "symbol": "XSPLIT",
                "action_type": "split",
                "ex_date": dates[5],
                "value": 2.0,
            }
        ]
    )
    audit = audit_split_residuals(daily, actions, ratio_tol=0.05)
    assert audit.height == 1
    row = audit.to_dicts()[0]
    assert row["status"] == "residual_jump"
    assert row["observed_ratio"] == pytest.approx(0.5, abs=1e-9)


def test_audit_passes_adjusted_close():
    dates = _trading_dates(date(2020, 1, 1), 10)
    closes = [100.0 + 0.1 * i for i in range(10)]  # smooth, no jump
    daily = _make_daily("XADJ", dates, closes)
    actions = _make_actions(
        [
            {
                "symbol": "XADJ",
                "action_type": "split",
                "ex_date": dates[5],
                "value": 2.0,
            }
        ]
    )
    audit = audit_split_residuals(daily, actions, ratio_tol=0.05)
    assert audit.height == 1
    row = audit.to_dicts()[0]
    assert row["status"] == "ok"


def test_audit_marks_out_of_window():
    dates = _trading_dates(date(2020, 1, 1), 5)
    daily = _make_daily("XOOW", dates, [100.0] * 5)
    # ex-date well after the daily window
    actions = _make_actions(
        [
            {
                "symbol": "XOOW",
                "action_type": "split",
                "ex_date": date(2025, 1, 1),
                "value": 2.0,
            }
        ]
    )
    audit = audit_split_residuals(daily, actions, ratio_tol=0.05)
    assert audit.to_dicts()[0]["status"] == "out_of_window"


def test_audit_marks_no_prev_close():
    dates = _trading_dates(date(2020, 1, 1), 5)
    daily = _make_daily("XFIRST", dates, [100.0] * 5)
    # ex-date is first day of history → no prev_close
    actions = _make_actions(
        [
            {
                "symbol": "XFIRST",
                "action_type": "split",
                "ex_date": dates[0],
                "value": 2.0,
            }
        ]
    )
    audit = audit_split_residuals(daily, actions, ratio_tol=0.05)
    assert audit.to_dicts()[0]["status"] == "no_prev_close"


def test_audit_marks_trivial_split_near_one():
    """A split ratio of ~1.0 (yfinance no-op) must not be flagged as a residual jump."""
    dates = _trading_dates(date(2020, 1, 1), 10)
    # Normal smooth prices — no actual split jump.
    closes = [100.0 + 0.5 * i for i in range(10)]
    daily = _make_daily("XNOOP", dates, closes)
    actions = _make_actions(
        [
            {
                "symbol": "XNOOP",
                "action_type": "split",
                "ex_date": dates[5],
                "value": 1.0008,
            }
        ]
    )
    audit = audit_split_residuals(daily, actions, ratio_tol=0.05)
    assert audit.to_dicts()[0]["status"] == "trivial_split"


def test_audit_ignores_dividends_only():
    dates = _trading_dates(date(2020, 1, 1), 5)
    daily = _make_daily("XDIV", dates, [100.0] * 5)
    actions = _make_actions(
        [
            {
                "symbol": "XDIV",
                "action_type": "dividend",
                "ex_date": dates[2],
                "value": 2.5,
            }
        ]
    )
    audit = audit_split_residuals(daily, actions)
    assert audit.is_empty()


# ---------------------------------------------------------------------------
# classify_adjustment_flavor
# ---------------------------------------------------------------------------


def _fake_history_split_only(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily.close mirrors yf.Close exactly; Adj Close differs (dividend-laden)."""
    dates = pd.date_range(start=start, end=end - timedelta(days=1), freq="D")
    close = np.linspace(100.0, 120.0, len(dates))
    # adj close: divergent cumulative product mimicking dividends
    adj_close = close * np.linspace(0.5, 0.9, len(dates))
    return pd.DataFrame({"Close": close, "Adj Close": adj_close}, index=dates).rename_axis("Date")


def _fake_history_total_return(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily.close mirrors yf.Adj Close exactly; yf.Close differs."""
    dates = pd.date_range(start=start, end=end - timedelta(days=1), freq="D")
    # Use a constant ratio so close is total-return adjusted relative to yf.Close
    close = np.linspace(100.0, 120.0, len(dates))
    adj_close = close * 0.75
    return pd.DataFrame({"Close": close, "Adj Close": adj_close}, index=dates).rename_axis("Date")


def test_flavor_classifier_detects_split_only():
    # daily.close == yf.Close → cv_vs_close ≈ 0
    dates = _trading_dates(date(2020, 1, 1), 200)
    closes = list(np.linspace(100.0, 120.0, 200))
    daily = _make_daily("XYZ", dates, closes)

    def stub(symbol: str, start: date, end: date) -> pd.DataFrame:
        return _fake_history_split_only(symbol, start, end)

    verdict = classify_adjustment_flavor("XYZ", daily, history_fn=stub)
    assert verdict["verdict"] == "split_only"
    assert verdict["cv_vs_close"] < verdict["cv_vs_adj_close"]


def test_flavor_classifier_detects_total_return():
    # daily.close == 0.75 * yf.Close so daily.close / yf.AdjClose is constant = 1.
    dates = _trading_dates(date(2020, 1, 1), 200)
    yf_close = np.linspace(100.0, 120.0, 200)
    # match the stub's adj_close = yf_close * 0.75
    closes = list(yf_close * 0.75)
    daily = _make_daily("ABC", dates, closes)

    def stub(symbol: str, start: date, end: date) -> pd.DataFrame:
        return _fake_history_total_return(symbol, start, end)

    verdict = classify_adjustment_flavor("ABC", daily, history_fn=stub)
    assert verdict["verdict"] == "total_return"
    assert verdict["cv_vs_adj_close"] < verdict["cv_vs_close"]


def test_flavor_classifier_insufficient_overlap():
    dates = _trading_dates(date(2020, 1, 1), 5)  # tiny window
    daily = _make_daily("TINY", dates, [100.0] * 5)

    def stub(symbol: str, start: date, end: date) -> pd.DataFrame:
        idx = pd.date_range(start=start, end=end, freq="D")
        return pd.DataFrame(
            {"Close": [100.0] * len(idx), "Adj Close": [100.0] * len(idx)}, index=idx
        ).rename_axis("Date")

    verdict = classify_adjustment_flavor("TINY", daily, history_fn=stub)
    assert verdict["verdict"] == "insufficient_overlap"


def test_cv_zero_mean_returns_inf():
    s = pl.Series([0.0, 0.0, 0.0])
    assert _coefficient_of_variation(s) == float("inf")


# ---------------------------------------------------------------------------
# passthrough_daily_adjusted
# ---------------------------------------------------------------------------


def test_passthrough_preserves_row_count_and_values(tmp_path: Path):
    src = tmp_path / "in.parquet"
    dst = tmp_path / "out.parquet"
    dates = _trading_dates(date(2020, 1, 1), 5)
    df = _make_daily("AAA", dates, [10.0, 11.0, 12.0, 13.0, 14.0])
    df.write_parquet(src)

    summary = passthrough_daily_adjusted(src, dst)
    assert summary["rows"] == 5
    assert summary["symbols"] == 1

    loaded = pl.read_parquet(dst)
    assert loaded.height == df.height
    assert loaded["close"].to_list() == df["close"].to_list()


def test_passthrough_rejects_missing_columns(tmp_path: Path):
    src = tmp_path / "in.parquet"
    # Missing "volume" column → should fail validation
    bad = pl.DataFrame(
        {
            "symbol": ["A"],
            "date": [date(2020, 1, 1)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        }
    )
    bad.write_parquet(src)
    with pytest.raises(RuntimeError, match="missing columns"):
        passthrough_daily_adjusted(src, tmp_path / "out.parquet")


def test_passthrough_rejects_empty(tmp_path: Path):
    src = tmp_path / "in.parquet"
    empty = pl.DataFrame(
        {
            "symbol": [],
            "date": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        },
        schema={
            "symbol": pl.Utf8,
            "date": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )
    empty.write_parquet(src)
    with pytest.raises(RuntimeError, match="empty source"):
        passthrough_daily_adjusted(src, tmp_path / "out.parquet")


# ---------------------------------------------------------------------------
# Slow: live yfinance probe (default-deselected)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_live_yfinance_reliance_has_2017_split():
    out = fetch_corporate_actions_for_symbol("RELIANCE", max_retries=2)
    if out["error"] is not None:
        pytest.skip(f"yfinance unavailable: {out['error']}")
    split_dates = [d for d, _ in out["splits"]]
    assert date(2017, 9, 7) in split_dates
