"""Day-4A daily cleaning pipeline.

Six ordered rules transform ``daily_adjusted`` → ``daily_clean``:

  1. Trading-calendar filter (drop weekends, national holidays, off-calendar bars).
  2. Patch the three known residual splits (back-adjust pre-ex-date OHLC).
  3. Trim phantom / seam history (drop rows before last unexplained >65% jump).
  4. Apply structural-event windows (keep only post-event history).
  5. Liquidity floor (drop rows where 60-day rolling-median ADV < floor).
  6. Minimum-history filter (drop symbols with fewer than N valid days).

Then a hard validation gate re-scans the cleaned frame for any unexplained
>40% single-day moves on/after 2011-01-01. Survivors (those not on a known
dividend ex-date and not in the KEEP list) fail the gate.

Each rule is a pure function returning ``(df_out, report_dict)``. The script
glues them together; tests cover each rule and the gate individually.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import polars as pl
from loguru import logger

# ---------------------------------------------------------------------------
# Known events (data, not config — these are project-historical decisions)
# ---------------------------------------------------------------------------

# Rule 2 — residual splits surfaced by Day-3's audit (FEDERALBNK + VINATIORGA).
# Each entry is (symbol, ex_date, split_ratio).
RESIDUAL_SPLITS: list[tuple[str, date, float]] = [
    ("FEDERALBNK", date(2004, 11, 29), 3.0),
    ("FEDERALBNK", date(2013, 10, 17), 5.0),
    ("VINATIORGA", date(2009, 10, 30), 5.0),
]

# Rule 4 — structural events; keep only history strictly AFTER each date.
STRUCTURAL_EVENTS: list[tuple[str, date]] = [
    ("ADANIENT", date(2015, 6, 3)),
    ("JSL", date(2015, 11, 19)),
    ("TATACHEM", date(2020, 3, 4)),
    ("CENTURYTEX", date(2019, 10, 11)),
    ("IDEA", date(2018, 8, 31)),  # Vodafone-Idea merger
]

# Validation-gate excused crashes (real one-day moves we explicitly preserve).
KEEP_EVENTS: list[tuple[str, date]] = [
    ("YESBANK", date(2020, 3, 6)),
    ("INFIBEAM", date(2018, 9, 28)),
    ("SUZLON", date(2008, 10, 24)),
]


# ---------------------------------------------------------------------------
# Rule 1 — Trading-calendar filter
# ---------------------------------------------------------------------------


def _weekdays_in_range(start: date, end: date) -> list[date]:
    """Inclusive Monday-through-Friday list."""
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _national_holidays_in_range(start: date, end: date) -> set[date]:
    """Universal NSE-closed dates that recur every year regardless of weekday.

    These are the six anchors the exchange always observes. Other NSE holidays
    (Diwali, Holi, Eid, Maha Shivratri, etc.) are caught data-driven via the
    presence threshold — they shift with the lunar calendar so a hardcoded
    list ages badly.
    """
    out: set[date] = set()
    for yr in range(start.year, end.year + 1):
        out.update(
            {
                date(yr, 1, 1),  # New Year
                date(yr, 1, 26),  # Republic Day
                date(yr, 5, 1),  # Labour Day
                date(yr, 8, 15),  # Independence Day
                date(yr, 10, 2),  # Gandhi Jayanti
                date(yr, 12, 25),  # Christmas
            }
        )
    return out


def build_trading_calendar(
    daily: pl.DataFrame,
    *,
    start: date = date(2003, 1, 1),
    end: date = date(2021, 6, 30),
    min_symbols_per_day: int = 100,
) -> list[date]:
    """Canonical NSE trading-day list, returned sorted ascending.

    A date is a trading day iff:
      * it is a weekday in ``[start, end]``;
      * it is not a hardcoded national holiday;
      * at least ``min_symbols_per_day`` symbols have a row on that date in
        ``daily`` (data-driven — catches moving lunar-calendar NSE holidays).
    """
    weekdays = set(_weekdays_in_range(start, end))
    national = _national_holidays_in_range(start, end)
    base = weekdays - national

    presence = daily.group_by("date").len().filter(pl.col("len") >= min_symbols_per_day)
    present = set(presence["date"].to_list())
    return sorted(base & present)


def apply_calendar_filter(df: pl.DataFrame, calendar: Iterable[date]) -> tuple[pl.DataFrame, dict]:
    """Rule 1 — keep only rows whose date is in ``calendar``."""
    cal_list = list(calendar)
    cal_set = set(cal_list)

    n_before = df.height
    weekend_mask = pl.col("date").dt.weekday() > 5
    weekend_rows = df.filter(weekend_mask).height
    jan1_rows = df.filter((pl.col("date").dt.month() == 1) & (pl.col("date").dt.day() == 1)).height

    out = df.filter(pl.col("date").is_in(cal_list))
    n_after = out.height

    # Validate ~250 trading days/year over the calendar's span
    if cal_list:
        years = max(1, cal_list[-1].year - cal_list[0].year)
        per_year = len(cal_list) / years
    else:
        per_year = 0.0

    report = {
        "rule": "calendar_filter",
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_before - n_after,
        "calendar_days": len(cal_set),
        "days_per_year_avg": round(per_year, 2),
        "weekend_rows_dropped": weekend_rows,
        "jan1_rows_dropped": jan1_rows,
    }
    logger.info(
        "Rule 1 calendar — kept {:,}/{:,} rows; {} weekend, {} Jan-1 dropped; "
        "calendar has {} days ({:.1f}/yr)",
        n_after,
        n_before,
        weekend_rows,
        jan1_rows,
        len(cal_set),
        per_year,
    )
    return out, report


# ---------------------------------------------------------------------------
# Rule 2 — Patch known residual splits
# ---------------------------------------------------------------------------


def _splits_as_frame(splits: list[tuple[str, date, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [s[0] for s in splits],
            "ex_date": [s[1] for s in splits],
            "split_ratio": [float(s[2]) for s in splits],
        }
    )


def apply_residual_splits(
    df: pl.DataFrame,
    splits: list[tuple[str, date, float]] = RESIDUAL_SPLITS,
) -> tuple[pl.DataFrame, dict]:
    """Rule 2 — back-adjust pre-ex-date OHLC by 1/ratio for each split."""
    if not splits:
        return df, {"rule": "residual_splits", "splits": [], "rows_adjusted": 0}

    splits_df = _splits_as_frame(splits)

    # Per row, multiplier = ∏ (1/ratio) over splits where ex_date > row.date.
    multipliers = (
        df.select(["symbol", "date"])
        .join(splits_df, on="symbol", how="inner")
        .filter(pl.col("ex_date") > pl.col("date"))
        .with_columns((1.0 / pl.col("split_ratio")).alias("step_mult"))
        .group_by(["symbol", "date"])
        .agg(pl.col("step_mult").product().alias("multiplier"))
    )

    out = df.join(multipliers, on=["symbol", "date"], how="left").with_columns(
        pl.coalesce(pl.col("multiplier"), pl.lit(1.0)).alias("multiplier")
    )

    out = out.with_columns(
        [(pl.col(c) * pl.col("multiplier")).alias(c) for c in ("open", "high", "low", "close")]
    ).drop("multiplier")

    # Per-split adjustment counts: rows with date < ex_date for the symbol.
    counts_by_split: list[dict] = []
    for sym, ex_d, ratio in splits:
        n = df.filter((pl.col("symbol") == sym) & (pl.col("date") < ex_d)).height
        counts_by_split.append(
            {"symbol": sym, "ex_date": ex_d, "split_ratio": ratio, "rows_adjusted": n}
        )

    total_adjusted = sum(c["rows_adjusted"] for c in counts_by_split)
    logger.info(
        "Rule 2 residual splits — {} adjusted rows across {} splits", total_adjusted, len(splits)
    )
    return out, {
        "rule": "residual_splits",
        "splits": counts_by_split,
        "rows_adjusted": total_adjusted,
    }


def verify_split_smoothness(
    df: pl.DataFrame,
    splits: list[tuple[str, date, float]] = RESIDUAL_SPLITS,
    *,
    tolerance: float = 0.15,
) -> list[dict]:
    """Per residual-split row, return the post-patch close-to-close return.

    Used after :func:`apply_residual_splits` to verify each patched ex-date is
    now within ±tolerance — i.e. inside normal daily-move bounds.
    """
    rows = df.sort(["symbol", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret")
    )
    out = []
    for sym, ex_d, ratio in splits:
        r = rows.filter((pl.col("symbol") == sym) & (pl.col("date") == ex_d))
        if r.is_empty():
            out.append(
                {
                    "symbol": sym,
                    "ex_date": ex_d,
                    "split_ratio": ratio,
                    "post_patch_return": None,
                    "smooth": False,
                    "note": "ex-date missing from frame",
                }
            )
            continue
        ret_val = r["ret"][0]
        out.append(
            {
                "symbol": sym,
                "ex_date": ex_d,
                "split_ratio": ratio,
                "post_patch_return": ret_val,
                "smooth": ret_val is not None and abs(ret_val) < tolerance,
                "note": "" if ret_val is not None and abs(ret_val) < tolerance else "outside bound",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Rule 3 — Trim phantom / seam history
# ---------------------------------------------------------------------------


def _excused_event_set(
    splits: list[tuple[str, date, float]],
    structural: list[tuple[str, date]],
    keep: list[tuple[str, date]],
) -> dict[str, set[date]]:
    out: dict[str, set[date]] = {}
    for s, d, _ in splits:
        out.setdefault(s, set()).add(d)
    for s, d in structural:
        out.setdefault(s, set()).add(d)
    for s, d in keep:
        out.setdefault(s, set()).add(d)
    return out


def trim_phantom_history(
    df: pl.DataFrame,
    *,
    threshold: float = 0.65,
    splits: list[tuple[str, date, float]] = RESIDUAL_SPLITS,
    structural: list[tuple[str, date]] = STRUCTURAL_EVENTS,
    keep: list[tuple[str, date]] = KEEP_EVENTS,
) -> tuple[pl.DataFrame, dict]:
    """Rule 3 — drop rows before last unexplained ``>threshold`` day-over-day move.

    "Unexplained" means the jump is NOT on a residual-split ex-date, a
    structural-event date, or a KEEP-list date for that symbol.
    """
    excused = _excused_event_set(splits, structural, keep)

    enriched = df.sort(["symbol", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret")
    )
    big_moves = enriched.filter(pl.col("ret").abs() > threshold).select(["symbol", "date", "ret"])

    unexplained_rows = [
        r for r in big_moves.to_dicts() if r["date"] not in excused.get(r["symbol"], set())
    ]
    if not unexplained_rows:
        logger.info("Rule 3 phantom-trim — no unexplained big moves found")
        return df, {
            "rule": "trim_phantom_history",
            "threshold": threshold,
            "trimmed_symbols": [],
            "rows_dropped": 0,
        }

    # Per symbol, the latest unexplained jump date is the seam to keep onward.
    trims: dict[str, date] = {}
    for r in unexplained_rows:
        prev = trims.get(r["symbol"])
        if prev is None or r["date"] > prev:
            trims[r["symbol"]] = r["date"]

    trims_df = pl.DataFrame({"symbol": list(trims.keys()), "seam_date": list(trims.values())})
    out = df.join(trims_df, on="symbol", how="left").filter(
        pl.col("seam_date").is_null() | (pl.col("date") >= pl.col("seam_date"))
    )

    rows_dropped = df.height - out.height
    trim_report = [{"symbol": s, "new_start_date": d} for s, d in sorted(trims.items())]
    logger.info(
        "Rule 3 phantom-trim — trimmed {} symbol(s); {} rows dropped",
        len(trim_report),
        rows_dropped,
    )
    return out.drop("seam_date"), {
        "rule": "trim_phantom_history",
        "threshold": threshold,
        "trimmed_symbols": trim_report,
        "rows_dropped": rows_dropped,
    }


# ---------------------------------------------------------------------------
# Rule 4 — Structural-event windowing
# ---------------------------------------------------------------------------


def apply_structural_events(
    df: pl.DataFrame, events: list[tuple[str, date]] = STRUCTURAL_EVENTS
) -> tuple[pl.DataFrame, dict]:
    """Rule 4 — for each event, keep only rows strictly AFTER ``event_date``."""
    if not events:
        return df, {"rule": "structural_events", "events": [], "rows_dropped": 0}

    events_df = pl.DataFrame(
        {"symbol": [e[0] for e in events], "event_date": [e[1] for e in events]}
    )

    n_before = df.height
    out = (
        df.join(events_df, on="symbol", how="left")
        .filter(pl.col("event_date").is_null() | (pl.col("date") > pl.col("event_date")))
        .drop("event_date")
    )
    n_after = out.height

    logger.info(
        "Rule 4 structural events — {} events; {} rows dropped",
        len(events),
        n_before - n_after,
    )
    return out, {
        "rule": "structural_events",
        "events": [{"symbol": s, "event_date": d} for s, d in events],
        "rows_dropped": n_before - n_after,
    }


# ---------------------------------------------------------------------------
# Rule 5 — Liquidity floor
# ---------------------------------------------------------------------------


def apply_liquidity_filter(
    df: pl.DataFrame,
    *,
    min_adv_inr: float = 10_000_000.0,
    window: int = 60,
    min_periods: int = 20,
) -> tuple[pl.DataFrame, dict]:
    """Rule 5 — drop rows whose rolling-median ADV falls below the floor.

    ``adv = close * volume`` is computed per row, then a right-aligned rolling
    median of size ``window`` (with ``min_periods`` warm-up) per symbol. Rows
    with a null or sub-floor rolling median are dropped.
    """
    n_before = df.height
    enriched = (
        df.sort(["symbol", "date"])
        .with_columns((pl.col("close") * pl.col("volume")).alias("adv"))
        .with_columns(
            pl.col("adv")
            .rolling_median(window_size=window, min_samples=min_periods)
            .over("symbol")
            .alias("adv_rolling_median")
        )
    )
    kept = enriched.filter(
        pl.col("adv_rolling_median").is_not_null() & (pl.col("adv_rolling_median") >= min_adv_inr)
    ).drop(["adv", "adv_rolling_median"])

    n_after = kept.height
    # Symbols that lost ALL their rows in the filter
    surviving_syms = set(kept["symbol"].unique().to_list())
    pre_syms = set(df["symbol"].unique().to_list())
    lost_syms = sorted(pre_syms - surviving_syms)
    logger.info(
        "Rule 5 liquidity (floor ₹{:,.0f}) — kept {:,}/{:,} rows; {} symbols lost entirely",
        min_adv_inr,
        n_after,
        n_before,
        len(lost_syms),
    )
    return kept, {
        "rule": "liquidity",
        "floor_inr": min_adv_inr,
        "window": window,
        "min_periods": min_periods,
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_before - n_after,
        "symbols_dropped_entirely": lost_syms,
    }


# ---------------------------------------------------------------------------
# Rule 7 — Contiguity filter (applied between Rule 5 and Rule 6)
# ---------------------------------------------------------------------------


def apply_contiguity_filter(
    df: pl.DataFrame,
    *,
    max_gap_days: int = 10,
    prefer_overlap_after: date = date(2015, 1, 1),
) -> tuple[pl.DataFrame, dict]:
    """Rule 7 — keep only the longest contiguous segment per symbol.

    A segment is a maximal run of consecutive rows for one symbol where
    no two adjacent dates are more than ``max_gap_days`` calendar days
    apart. After Rule 5's liquidity filter, ADV-dipping mid-history holes
    can leave a symbol's data as several disjoint segments; rolling and
    cointegration windows would silently span those holes.

    Preference: segments with ``seg_end >= prefer_overlap_after`` win over
    segments that don't (otherwise a long pre-backtest segment would beat
    a short but backtest-relevant one). Within each class, longest wins,
    with ``seg_end`` as the final tiebreaker.
    """
    n_before = df.height
    if df.is_empty():
        empty_report: dict = {
            "rule": "contiguity",
            "max_gap_days": max_gap_days,
            "prefer_overlap_after": prefer_overlap_after,
            "rows_before": 0,
            "rows_after": 0,
            "rows_dropped": 0,
            "n_symbols_segmented": 0,
            "n_segments_total": 0,
            "n_segments_kept": 0,
            "n_segments_dropped": 0,
            "segment_detail": [],
        }
        return df, empty_report

    enriched = (
        df.sort(["symbol", "date"])
        .with_columns(
            pl.col("date").diff().over("symbol").dt.total_days().fill_null(0).alias("_gap_days")
        )
        .with_columns(
            (pl.col("_gap_days") > max_gap_days)
            .cast(pl.Int64)
            .cum_sum()
            .over("symbol")
            .alias("_segment_id")
        )
    )

    seg_stats = (
        enriched.group_by(["symbol", "_segment_id"])
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("date").min().alias("seg_start"),
                pl.col("date").max().alias("seg_end"),
                pl.col("_gap_days").max().alias("max_gap_within"),
            ]
        )
        .with_columns((pl.col("seg_end") >= prefer_overlap_after).alias("overlaps_target"))
    )

    # Priority: any 2015+-overlapping segment beats any non-overlapping;
    # within each class, larger n_rows wins; final tiebreak on seg_end.
    sorted_segs = seg_stats.sort(
        ["symbol", "overlaps_target", "n_rows", "seg_end"],
        descending=[False, True, True, True],
    )
    best = sorted_segs.unique(subset=["symbol"], keep="first").select(["symbol", "_segment_id"])

    out = enriched.join(best, on=["symbol", "_segment_id"], how="inner").drop(
        ["_gap_days", "_segment_id"]
    )
    n_after = out.height

    n_segs_per_sym = seg_stats.group_by("symbol").len().rename({"len": "n_segments"})
    segmented_syms = n_segs_per_sym.filter(pl.col("n_segments") > 1)

    detail = (
        seg_stats.join(segmented_syms.select("symbol"), on="symbol", how="inner")
        .join(
            best.with_columns(pl.lit(True).alias("kept")),
            on=["symbol", "_segment_id"],
            how="left",
        )
        .with_columns(pl.col("kept").fill_null(False))
        .sort(["symbol", "_segment_id"])
        .rename({"_segment_id": "segment_id"})
        .select(
            [
                "symbol",
                "segment_id",
                "n_rows",
                "seg_start",
                "seg_end",
                "max_gap_within",
                "overlaps_target",
                "kept",
            ]
        )
    )

    logger.info(
        "Rule 7 contiguity (gap > {}d): {}/{} symbols had multi-segment history; {:,} rows dropped",
        max_gap_days,
        segmented_syms.height,
        df["symbol"].n_unique(),
        n_before - n_after,
    )

    return out, {
        "rule": "contiguity",
        "max_gap_days": max_gap_days,
        "prefer_overlap_after": prefer_overlap_after,
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_before - n_after,
        "n_symbols_segmented": segmented_syms.height,
        "n_segments_total": seg_stats.height,
        "n_segments_kept": best.height,
        "n_segments_dropped": seg_stats.height - best.height,
        "segment_detail": detail.to_dicts(),
    }


def max_internal_gap_per_symbol(df: pl.DataFrame) -> pl.DataFrame:
    """Helper: ``symbol -> max calendar-day gap`` between consecutive rows.

    Used post-Rule-7 to confirm every kept symbol's max gap is now ≤ the
    threshold.
    """
    return (
        df.sort(["symbol", "date"])
        .with_columns(
            pl.col("date").diff().over("symbol").dt.total_days().fill_null(0).alias("gap")
        )
        .group_by("symbol")
        .agg(pl.col("gap").max().alias("max_internal_gap_days"))
        .sort("max_internal_gap_days", descending=True)
    )


# ---------------------------------------------------------------------------
# Rule 6 — Minimum history
# ---------------------------------------------------------------------------


def apply_min_history(df: pl.DataFrame, *, min_days: int = 756) -> tuple[pl.DataFrame, dict]:
    """Rule 6 — drop symbols with fewer than ``min_days`` cleaned rows."""
    counts = df.group_by("symbol").len().rename({"len": "n_days"})
    keepers = counts.filter(pl.col("n_days") >= min_days).select("symbol")
    dropped = counts.filter(pl.col("n_days") < min_days).sort("n_days")

    out = df.join(keepers, on="symbol", how="inner")
    logger.info(
        "Rule 6 min-history (>= {} days) — {} symbols kept, {} dropped",
        min_days,
        keepers.height,
        dropped.height,
    )
    return out, {
        "rule": "min_history",
        "min_days": min_days,
        "symbols_kept": keepers.height,
        "symbols_dropped": dropped.to_dicts(),
    }


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------


def validation_gate(
    df: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    start_date: date = date(2011, 1, 1),
    threshold: float = 0.40,
    keep: list[tuple[str, date]] = KEEP_EVENTS,
    keep_windows: list[tuple[date, date]] | None = None,
    yfinance_failed_symbols: set[str] | None = None,
) -> dict:
    """Re-scan ``df`` for unexplained ``>threshold`` single-day moves.

    A jump on ``date`` is *explained* if (symbol, date) is in the cached
    dividend ex-date set, in ``keep``, or lies inside one of the inclusive
    ``keep_windows`` (e.g. ``[(date(2020,2,24), date(2020,4,3))]`` for the
    COVID crash). Any other jump is a survivor and fails the gate.

    Returns a dict with the gate verdict, the survivor list, and the
    coverage check on the supplied ``yfinance_failed_symbols`` blind spot.
    """
    yfinance_failed_symbols = yfinance_failed_symbols or set()
    keep_windows = keep_windows or []

    scan = (
        df.sort(["symbol", "date"])
        .with_columns((pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret"))
        .filter((pl.col("date") >= start_date) & (pl.col("ret").abs() > threshold))
        .select(["symbol", "date", "close", "ret"])
    )

    dividend_keys: set[tuple[str, date]] = set()
    if not actions.is_empty():
        divs = actions.filter(pl.col("action_type") == "dividend")
        dividend_keys = {(r["symbol"], r["ex_date"]) for r in divs.to_dicts()}
    keep_keys: set[tuple[str, date]] = {(s, d) for s, d in keep}

    survivors: list[dict] = []
    excused_by_window = 0
    for row in scan.to_dicts():
        key = (row["symbol"], row["date"])
        if key in keep_keys or key in dividend_keys:
            continue
        if any(start <= row["date"] <= end for start, end in keep_windows):
            excused_by_window += 1
            continue
        survivors.append(row)

    yf_blind_survivors = [s for s in survivors if s["symbol"] in yfinance_failed_symbols]

    return {
        "start_date": start_date,
        "threshold": threshold,
        "n_big_moves": scan.height,
        "n_survivors": len(survivors),
        "n_excused_by_window": excused_by_window,
        "survivors": survivors,
        "n_yfinance_blind_total": len(yfinance_failed_symbols),
        "n_yfinance_blind_survivors": len(yf_blind_survivors),
        "yfinance_blind_survivors": yf_blind_survivors,
        "pass": len(survivors) == 0,
    }
