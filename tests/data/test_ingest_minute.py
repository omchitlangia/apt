"""Tests for apt.data.ingest_minute — minute CSV ingestion + hive partitioning."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from apt.data.ingest_minute import (
    FULL_SESSION_BARS,
    discover_minute_csvs,
    ingest_all_minute,
    parse_minute_csv,
    validate_symbol_universe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _session_bars(d: date, n: int = FULL_SESSION_BARS) -> list[str]:
    """Generate ``n`` IST minute timestamps starting at 09:15 for date ``d``."""
    rows = []
    for offset in range(n):
        hh = 9 + (15 + offset) // 60
        mm = (15 + offset) % 60
        rows.append(f"{d.isoformat()} {hh:02d}:{mm:02d}:00+05:30")
    return rows


def _make_csv(path: Path, ts_rows: list[str], *, base_price: float = 100.0) -> None:
    """Write a minute CSV with the given timestamps and trivial flat OHLC."""
    lines = ["date,open,high,low,close,volume"]
    for i, ts in enumerate(ts_rows):
        p = base_price + 0.01 * i
        lines.append(f"{ts},{p:.2f},{p + 0.5:.2f},{p - 0.5:.2f},{p + 0.1:.2f},1000")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture()
def full_day_csv(tmp_path: Path) -> Path:
    """A symbol with one complete 375-bar session."""
    p = tmp_path / "RELIANCE_minute-data.csv"
    _make_csv(p, _session_bars(date(2020, 1, 2)))
    return p


@pytest.fixture()
def short_day_csv(tmp_path: Path) -> Path:
    """A symbol whose only day has 100 bars (partial)."""
    p = tmp_path / "SHORTY_minute-data.csv"
    _make_csv(p, _session_bars(date(2020, 1, 2), n=100))
    return p


@pytest.fixture()
def two_day_csv(tmp_path: Path) -> Path:
    """A symbol with one full day + one short day."""
    p = tmp_path / "MIXED_minute-data.csv"
    rows = _session_bars(date(2020, 1, 2)) + _session_bars(date(2020, 1, 3), n=50)
    _make_csv(p, rows)
    return p


@pytest.fixture()
def dup_csv(tmp_path: Path) -> Path:
    """A symbol whose CSV contains a duplicated row."""
    p = tmp_path / "DUPSYM_minute-data.csv"
    base = _session_bars(date(2020, 1, 2), n=3)
    # duplicate the second row
    rows = [base[0], base[1], base[1], base[2]]
    _make_csv(p, rows)
    return p


@pytest.fixture()
def trailing_drop_csv(tmp_path: Path) -> Path:
    """Full day on 2021-06-23 plus a short trailing 2021-06-24 to be dropped."""
    p = tmp_path / "DROP_minute-data.csv"
    rows = _session_bars(date(2021, 6, 23)) + _session_bars(date(2021, 6, 24), n=297)
    _make_csv(p, rows)
    return p


# ---------------------------------------------------------------------------
# discover_minute_csvs + filename validation
# ---------------------------------------------------------------------------


def test_discover_finds_conforming_files(tmp_path: Path):
    (tmp_path / "RELIANCE_minute-data.csv").write_text("date,open,high,low,close,volume\n")
    (tmp_path / "TCS_minute-data.csv").write_text("date,open,high,low,close,volume\n")
    (tmp_path / "random.csv").write_text("foo\n")
    paths = discover_minute_csvs(tmp_path)
    names = [p.name for p in paths]
    assert "RELIANCE_minute-data.csv" in names
    assert "TCS_minute-data.csv" in names
    assert "random.csv" not in names


# ---------------------------------------------------------------------------
# parse_minute_csv — schema and dtypes
# ---------------------------------------------------------------------------


def test_schema_columns(full_day_csv: Path):
    df = parse_minute_csv(full_day_csv)
    assert df is not None
    expected = {
        "year",
        "timestamp",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "partial_day_flag",
    }
    assert set(df.columns) == expected


def test_dtypes(full_day_csv: Path):
    df = parse_minute_csv(full_day_csv)
    assert df is not None
    assert df["timestamp"].dtype == pl.Datetime(time_unit="us", time_zone="Asia/Kolkata")
    assert df["date"].dtype == pl.Date
    assert df["year"].dtype == pl.Int32
    for c in ("open", "high", "low", "close"):
        assert df[c].dtype == pl.Float64
    assert df["volume"].dtype == pl.Int64
    assert df["partial_day_flag"].dtype == pl.Boolean


# ---------------------------------------------------------------------------
# parse_minute_csv — timezone handling
# ---------------------------------------------------------------------------


def test_timezone_preserved_as_ist(full_day_csv: Path):
    df = parse_minute_csv(full_day_csv)
    assert df is not None
    first = df["timestamp"].min()
    # tz-aware datetime in IST. First bar of a regular session is 09:15.
    assert first.tzinfo is not None
    assert first.hour == 9 and first.minute == 15


def test_calendar_date_derived_from_ist(full_day_csv: Path):
    df = parse_minute_csv(full_day_csv)
    assert df is not None
    assert df["date"].unique().to_list() == [date(2020, 1, 2)]


# ---------------------------------------------------------------------------
# parse_minute_csv — 375-bar full-session count
# ---------------------------------------------------------------------------


def test_full_session_375_bars(full_day_csv: Path):
    df = parse_minute_csv(full_day_csv)
    assert df is not None
    assert df.height == FULL_SESSION_BARS


def test_full_session_not_flagged_partial(full_day_csv: Path):
    df = parse_minute_csv(full_day_csv)
    assert df is not None
    assert df["partial_day_flag"].sum() == 0


# ---------------------------------------------------------------------------
# parse_minute_csv — partial-day flagging
# ---------------------------------------------------------------------------


def test_short_day_flagged(short_day_csv: Path):
    df = parse_minute_csv(short_day_csv)
    assert df is not None
    assert df.height == 100
    assert bool(df["partial_day_flag"].all())


def test_mixed_days_flag_only_short(two_day_csv: Path):
    df = parse_minute_csv(two_day_csv)
    assert df is not None
    flags_by_day = (
        df.group_by("date").agg(pl.col("partial_day_flag").any().alias("partial")).sort("date")
    )
    rows = flags_by_day.to_dicts()
    assert rows[0]["date"] == date(2020, 1, 2)
    assert rows[0]["partial"] is False
    assert rows[1]["date"] == date(2020, 1, 3)
    assert rows[1]["partial"] is True


# ---------------------------------------------------------------------------
# parse_minute_csv — duplicates
# ---------------------------------------------------------------------------


def test_duplicates_removed(dup_csv: Path):
    df = parse_minute_csv(dup_csv)
    assert df is not None
    # 4 rows in CSV, one is an exact duplicate of another timestamp → 3 unique
    assert df.height == 3
    n_unique = df["timestamp"].n_unique()
    assert n_unique == 3


def test_timestamps_strictly_increasing(dup_csv: Path):
    df = parse_minute_csv(dup_csv)
    assert df is not None
    ts = df["timestamp"].to_list()
    assert ts == sorted(ts)
    assert len(set(ts)) == len(ts)


# ---------------------------------------------------------------------------
# parse_minute_csv — cleaning + drop_dates
# ---------------------------------------------------------------------------


def test_null_prices_dropped(tmp_path: Path):
    p = tmp_path / "NULLP_minute-data.csv"
    p.write_text(
        textwrap.dedent(
            """\
        date,open,high,low,close,volume
        2020-01-02 09:15:00+05:30,100.0,101.0,99.0,,500
        2020-01-02 09:16:00+05:30,100.1,101.1,99.1,100.5,500
    """
        )
    )
    df = parse_minute_csv(p)
    assert df is not None
    assert df.height == 1


def test_negative_prices_dropped(tmp_path: Path):
    p = tmp_path / "NEGP_minute-data.csv"
    p.write_text(
        textwrap.dedent(
            """\
        date,open,high,low,close,volume
        2020-01-02 09:15:00+05:30,100.0,101.0,-1.0,100.5,500
        2020-01-02 09:16:00+05:30,100.1,101.1,99.1,100.5,500
    """
        )
    )
    df = parse_minute_csv(p)
    assert df is not None
    assert df.height == 1


def test_drop_dates_excludes_rows(trailing_drop_csv: Path):
    df = parse_minute_csv(trailing_drop_csv, drop_dates=frozenset({date(2021, 6, 24)}))
    assert df is not None
    assert date(2021, 6, 24) not in df["date"].unique().to_list()
    assert date(2021, 6, 23) in df["date"].unique().to_list()


def test_empty_csv_returns_none(tmp_path: Path):
    p = tmp_path / "EMPTY_minute-data.csv"
    p.write_text("date,open,high,low,close,volume\n")
    assert parse_minute_csv(p) is None


# ---------------------------------------------------------------------------
# validate_symbol_universe
# ---------------------------------------------------------------------------


def test_universe_matches(tmp_path: Path):
    (tmp_path / "AAA_minute-data.csv").write_text("date,open,high,low,close,volume\n")
    (tmp_path / "BBB_minute-data.csv").write_text("date,open,high,low,close,volume\n")
    paths = discover_minute_csvs(tmp_path)
    assert validate_symbol_universe(paths, {"AAA", "BBB"}) == {"AAA", "BBB"}


def test_universe_mismatch_raises(tmp_path: Path):
    (tmp_path / "AAA_minute-data.csv").write_text("date,open,high,low,close,volume\n")
    paths = discover_minute_csvs(tmp_path)
    with pytest.raises(RuntimeError, match="does not match"):
        validate_symbol_universe(paths, {"AAA", "BBB"})


# ---------------------------------------------------------------------------
# ingest_all_minute — hive-partitioned dataset round-trip
# ---------------------------------------------------------------------------


def test_ingest_all_minute_round_trip(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"

    # Two symbols, two years each.
    _make_csv(raw / "AAA_minute-data.csv", _session_bars(date(2020, 1, 2)))
    _make_csv(raw / "BBB_minute-data.csv", _session_bars(date(2021, 6, 23)))

    stats = ingest_all_minute(
        raw_dir=raw,
        output_root=out,
        expected_symbols={"AAA", "BBB"},
        n_jobs=1,
    )

    assert len(stats) == 2
    assert all(s["error"] is None for s in stats)
    assert all(s["strictly_increasing"] for s in stats)

    # Hive layout check
    assert (out / "symbol=AAA" / "year=2020" / "data.parquet").exists()
    assert (out / "symbol=BBB" / "year=2021" / "data.parquet").exists()

    # Round-trip through hive-aware scan
    df = pl.scan_parquet(str(out / "**/*.parquet"), hive_partitioning=True).collect()
    assert set(df.columns) >= {
        "symbol",
        "year",
        "timestamp",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "partial_day_flag",
    }
    assert set(df["symbol"].unique().to_list()) == {"AAA", "BBB"}
    assert df.filter(pl.col("symbol") == "AAA").height == FULL_SESSION_BARS


def test_ingest_all_minute_drops_trailing_day(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"

    rows = _session_bars(date(2021, 6, 23)) + _session_bars(date(2021, 6, 24), n=297)
    _make_csv(raw / "AAA_minute-data.csv", rows)

    ingest_all_minute(
        raw_dir=raw,
        output_root=out,
        expected_symbols={"AAA"},
        drop_dates=frozenset({date(2021, 6, 24)}),
        n_jobs=1,
    )

    df = pl.scan_parquet(str(out / "**/*.parquet"), hive_partitioning=True).collect()
    dates = set(df["date"].unique().to_list())
    assert date(2021, 6, 24) not in dates
    assert date(2021, 6, 23) in dates


def test_ingest_idempotent(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    _make_csv(raw / "AAA_minute-data.csv", _session_bars(date(2020, 1, 2)))

    stats1 = ingest_all_minute(raw_dir=raw, output_root=out, expected_symbols={"AAA"}, n_jobs=1)
    stats2 = ingest_all_minute(raw_dir=raw, output_root=out, expected_symbols={"AAA"}, n_jobs=1)
    assert stats1[0]["rows"] == stats2[0]["rows"]


# ---------------------------------------------------------------------------
# Marker: slow tests hitting real /Data6/db (skipped by default in CI fast lane)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reliance_real_data_2020_01_02_375_bars():
    path = Path("/Data6/db/minute/RELIANCE_minute-data.csv")
    if not path.exists():
        pytest.skip("Real minute data not available")
    df = parse_minute_csv(path, drop_dates=frozenset({date(2021, 6, 24)}))
    assert df is not None
    n = df.filter(pl.col("date") == date(2020, 1, 2)).height
    assert n == FULL_SESSION_BARS
