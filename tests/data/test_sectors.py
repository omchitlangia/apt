"""Tests for apt.data.sectors — sector mapping from xlsx."""

from __future__ import annotations

from pathlib import Path

import pytest

from apt.data.sectors import _normalise_symbol, build_sector_mapping, parse_nifty500_sectors

# ---------------------------------------------------------------------------
# _normalise_symbol
# ---------------------------------------------------------------------------


def test_normalise_symbol_uppercase():
    assert _normalise_symbol("reliance") == "RELIANCE"


def test_normalise_symbol_strips_spaces():
    assert _normalise_symbol("  TCS  ") == "TCS"


def test_normalise_symbol_none():
    assert _normalise_symbol(None) is None


def test_normalise_symbol_mixed():
    assert _normalise_symbol(" Infy ") == "INFY"


# ---------------------------------------------------------------------------
# parse_nifty500_sectors — uses real xlsx (fast, read-only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nifty500_df():
    xlsx = Path("/Data6/db/Merge_21May2021.xlsx")
    if not xlsx.exists():
        pytest.skip("xlsx not available")
    return parse_nifty500_sectors(xlsx)


def test_nifty500_has_rows(nifty500_df):
    assert len(nifty500_df) > 400


def test_nifty500_columns(nifty500_df):
    expected = {"symbol", "company_name", "industry", "isin"}
    assert expected.issubset(set(nifty500_df.columns))


def test_nifty500_symbols_uppercase(nifty500_df):
    symbols = nifty500_df["symbol"].to_list()
    for s in symbols:
        assert s == s.upper(), f"Symbol not uppercase: {s}"


def test_nifty500_no_null_symbols(nifty500_df):
    assert nifty500_df["symbol"].null_count() == 0


def test_nifty500_known_symbol_present(nifty500_df):
    symbols = set(nifty500_df["symbol"].to_list())
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert "INFY" in symbols


def test_nifty500_industry_not_all_null(nifty500_df):
    non_null = nifty500_df["industry"].drop_nulls()
    assert len(non_null) > 400


# ---------------------------------------------------------------------------
# build_sector_mapping — missing-symbol reporting
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_build_sector_mapping_writes_parquet(tmp_path):
    xlsx = Path("/Data6/db/Merge_21May2021.xlsx")
    if not xlsx.exists():
        pytest.skip("xlsx not available")
    out = tmp_path / "sectors.parquet"
    df = build_sector_mapping(xlsx, out)
    assert out.exists()
    assert len(df) > 0


def test_missing_symbols_reported(nifty500_df, capsys, tmp_path):
    """Symbols in universe but not in mapping should be logged (loguru → stderr)."""
    from loguru import logger

    xlsx = Path("/Data6/db/Merge_21May2021.xlsx")
    if not xlsx.exists():
        pytest.skip("xlsx not available")

    universe = {"RELIANCE", "TCS", "__NOSUCHSYMBOL__"}
    out = tmp_path / "sectors.parquet"

    messages: list[str] = []

    def _capture(msg):
        messages.append(msg)

    sink_id = logger.add(_capture, format="{message}", level="WARNING")
    try:
        build_sector_mapping(xlsx, out, universe_symbols=universe)
    finally:
        logger.remove(sink_id)

    assert any("__NOSUCHSYMBOL__" in m for m in messages), (
        f"Expected warning not found in: {messages}"
    )


def test_all_symbols_mapped_no_warning(nifty500_df, tmp_path):
    """When universe is a subset of the mapping, no 'without sector mapping' warning."""
    from loguru import logger

    xlsx = Path("/Data6/db/Merge_21May2021.xlsx")
    if not xlsx.exists():
        pytest.skip("xlsx not available")

    universe = {"RELIANCE", "TCS"}
    out = tmp_path / "sectors.parquet"

    warnings: list[str] = []

    def _capture(msg):
        warnings.append(msg)

    sink_id = logger.add(_capture, format="{message}", level="WARNING")
    try:
        build_sector_mapping(xlsx, out, universe_symbols=universe)
    finally:
        logger.remove(sink_id)

    assert not any("without sector mapping" in m for m in warnings)
