"""Tests for apt.data.ingest — daily CSV ingestion."""

from __future__ import annotations

import textwrap
from pathlib import Path

import polars as pl
import pytest

from apt.data.ingest import (
    discover_daily_csvs,
    ingest_all_daily,
    parse_daily_csv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    """Write a small set of test CSVs mimicking /Data6/db/ layout."""
    good = tmp_path / "RELIANCE.csv"
    good.write_text(
        textwrap.dedent("""\
        date,open,high,low,close,volume
        2020-01-01 00:00:00+05:30,100.0,110.0,95.0,105.0,1000000
        2020-01-02 00:00:00+05:30,105.0,115.0,100.0,110.0,1200000
        2020-01-03 00:00:00+05:30,110.0,120.0,105.0,115.0,900000
    """)
    )

    # File with zero-volume rows
    zvol = tmp_path / "TCS.csv"
    zvol.write_text(
        textwrap.dedent("""\
        date,open,high,low,close,volume
        2020-01-01 00:00:00+05:30,500.0,510.0,495.0,505.0,0
        2020-01-02 00:00:00+05:30,505.0,515.0,500.0,510.0,500000
    """)
    )

    # Should be excluded (lowercase stem)
    (tmp_path / "sbin.csv").write_text("date,open,high,low,close,volume\n")
    # Should be excluded (instrument master)
    (tmp_path / "NSE_Instruments_23June2021.csv").write_text("a,b\n1,2\n")

    return tmp_path


@pytest.fixture()
def good_csv(tmp_path: Path) -> Path:
    p = tmp_path / "INFY.csv"
    p.write_text(
        textwrap.dedent("""\
        date,open,high,low,close,volume
        2021-06-01 00:00:00+05:30,1500.0,1520.0,1490.0,1510.0,2000000
        2021-06-02 00:00:00+05:30,1510.0,1530.0,1500.0,1520.0,1800000
        2021-06-03 00:00:00+05:30,1520.0,1540.0,1510.0,1530.0,2100000
    """)
    )
    return p


@pytest.fixture()
def dup_date_csv(tmp_path: Path) -> Path:
    """CSV with a duplicate date entry."""
    p = tmp_path / "WIPRO.csv"
    p.write_text(
        textwrap.dedent("""\
        date,open,high,low,close,volume
        2021-06-01 00:00:00+05:30,400.0,410.0,395.0,405.0,500000
        2021-06-01 00:00:00+05:30,401.0,411.0,396.0,406.0,600000
        2021-06-02 00:00:00+05:30,406.0,416.0,401.0,411.0,700000
    """)
    )
    return p


@pytest.fixture()
def neg_price_csv(tmp_path: Path) -> Path:
    """CSV with a row having a negative price."""
    p = tmp_path / "HDFCBANK.csv"
    p.write_text(
        textwrap.dedent("""\
        date,open,high,low,close,volume
        2021-06-01 00:00:00+05:30,1200.0,1220.0,-10.0,1210.0,1000000
        2021-06-02 00:00:00+05:30,1210.0,1230.0,1200.0,1220.0,900000
    """)
    )
    return p


@pytest.fixture()
def null_price_csv(tmp_path: Path) -> Path:
    """CSV with a row having a null close price."""
    p = tmp_path / "ICICI.csv"
    p.write_text(
        textwrap.dedent("""\
        date,open,high,low,close,volume
        2021-06-01 00:00:00+05:30,700.0,710.0,695.0,,800000
        2021-06-02 00:00:00+05:30,710.0,720.0,705.0,715.0,750000
    """)
    )
    return p


# ---------------------------------------------------------------------------
# discover_daily_csvs
# ---------------------------------------------------------------------------


def test_discover_excludes_sbin(csv_dir):
    paths = discover_daily_csvs(csv_dir)
    names = [p.stem for p in paths]
    assert "sbin" not in names


def test_discover_excludes_nse_instruments(csv_dir):
    paths = discover_daily_csvs(csv_dir)
    names = [p.stem for p in paths]
    assert "NSE_Instruments_23June2021" not in names


def test_discover_finds_equity_csvs(csv_dir):
    paths = discover_daily_csvs(csv_dir)
    names = sorted(p.stem for p in paths)
    assert "RELIANCE" in names
    assert "TCS" in names


def test_discover_count(csv_dir):
    paths = discover_daily_csvs(csv_dir)
    assert len(paths) == 2


# ---------------------------------------------------------------------------
# parse_daily_csv — schema
# ---------------------------------------------------------------------------


def test_schema_columns(good_csv):
    df = parse_daily_csv(good_csv)
    assert df is not None
    expected = {"symbol", "date", "open", "high", "low", "close", "volume", "zero_volume_flag"}
    assert set(df.columns) == expected


def test_symbol_column_correct(good_csv):
    df = parse_daily_csv(good_csv)
    assert df is not None
    assert df["symbol"].unique().to_list() == ["INFY"]


def test_date_dtype_is_date(good_csv):
    df = parse_daily_csv(good_csv)
    assert df is not None
    assert df["date"].dtype == pl.Date


def test_row_count_matches_csv(good_csv):
    df = parse_daily_csv(good_csv)
    assert df is not None
    assert len(df) == 3


# ---------------------------------------------------------------------------
# parse_daily_csv — zero-volume flagging
# ---------------------------------------------------------------------------


def test_zero_volume_flagged(tmp_path):
    p = tmp_path / "ABC.csv"
    p.write_text(
        textwrap.dedent("""\
        date,open,high,low,close,volume
        2020-01-01 00:00:00+05:30,10.0,11.0,9.0,10.5,0
        2020-01-02 00:00:00+05:30,10.5,11.5,10.0,11.0,5000
    """)
    )
    df = parse_daily_csv(p)
    assert df is not None
    flags = df.sort("date")["zero_volume_flag"].to_list()
    assert flags == [True, False]


def test_zero_volume_rows_retained(tmp_path):
    p = tmp_path / "XYZ.csv"
    p.write_text(
        textwrap.dedent("""\
        date,open,high,low,close,volume
        2020-01-01 00:00:00+05:30,10.0,11.0,9.0,10.5,0
    """)
    )
    df = parse_daily_csv(p)
    assert df is not None
    assert len(df) == 1  # row retained, just flagged


# ---------------------------------------------------------------------------
# parse_daily_csv — data cleaning
# ---------------------------------------------------------------------------


def test_negative_price_rows_dropped(neg_price_csv):
    df = parse_daily_csv(neg_price_csv)
    assert df is not None
    # Row with low=-10 should be dropped
    assert len(df) == 1


def test_null_price_rows_dropped(null_price_csv):
    df = parse_daily_csv(null_price_csv)
    assert df is not None
    assert len(df) == 1


def test_duplicate_dates_deduplicated(dup_date_csv):
    df = parse_daily_csv(dup_date_csv)
    assert df is not None
    assert len(df) == 2
    # Last row's values should be kept for 2021-06-01
    row = df.filter(pl.col("date") == pl.date(2021, 6, 1))
    assert row["open"][0] == pytest.approx(401.0)


# ---------------------------------------------------------------------------
# parse_daily_csv — timestamp handling
# ---------------------------------------------------------------------------


def test_timezone_stripped_from_date(good_csv):
    df = parse_daily_csv(good_csv)
    assert df is not None
    import datetime

    dates = df["date"].to_list()
    assert all(isinstance(d, datetime.date) for d in dates)


# ---------------------------------------------------------------------------
# parse_daily_csv — special cases
# ---------------------------------------------------------------------------


def test_returns_none_on_empty_file(tmp_path):
    p = tmp_path / "EMPTY.csv"
    p.write_text("date,open,high,low,close,volume\n")
    result = parse_daily_csv(p)
    assert result is None


def test_returns_none_on_corrupt_file(tmp_path):
    p = tmp_path / "CORRUPT.csv"
    p.write_bytes(b"\xff\xfe garbage data that is not a CSV")
    # Should not raise — returns None
    result = parse_daily_csv(p)
    # Either None or an empty-parse result — key thing: no exception
    assert result is None or len(result) == 0


# ---------------------------------------------------------------------------
# ingest_all_daily — integration
# ---------------------------------------------------------------------------


def test_ingest_all_daily_writes_parquet(csv_dir, tmp_path):
    out = tmp_path / "output.parquet"
    df = ingest_all_daily(csv_dir, out, n_jobs=1)
    assert out.exists()
    assert len(df) > 0
    assert "symbol" in df.columns


def test_ingest_all_daily_excludes_sbin(csv_dir, tmp_path):
    out = tmp_path / "output.parquet"
    df = ingest_all_daily(csv_dir, out, n_jobs=1)
    assert "SBIN" not in df["symbol"].unique().to_list()


def test_ingest_idempotent(csv_dir, tmp_path):
    out = tmp_path / "output.parquet"
    df1 = ingest_all_daily(csv_dir, out, n_jobs=1)
    df2 = ingest_all_daily(csv_dir, out, n_jobs=1)
    assert len(df1) == len(df2)


def test_ingest_parquet_has_year_column(csv_dir, tmp_path):
    out = tmp_path / "output.parquet"
    ingest_all_daily(csv_dir, out, n_jobs=1)
    loaded = pl.read_parquet(out)
    assert "year" in loaded.columns
