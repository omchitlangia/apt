"""Day-4C corporate-action repair.

Drives the Day-4A validation gate to green by classifying every surviving
``>40%`` single-day discontinuity into one of three buckets and acting on it:

  * ``KEEP_HARDCODED`` — listed in :data:`apt.data.clean.KEEP_EVENTS`
    (YESBANK 2020-03-06, INFIBEAM 2018-09-28, SUZLON 2008-10-24).
  * ``KEEP_COVID`` — falls inside the tightened COVID crash window
    2020-02-24 .. 2020-04-03 (regardless of direction; market-wide volatility
    moves both legs of a pair together so the spread stays stationary).
  * ``TRIM`` — every other survivor (single-name CAs and phantoms). A
    single-name move only shifts one leg of any pair, breaking the spread,
    so the segment before the event is removed.

No data values are ever changed. We previously had a ratio-snap ``ADJUST``
class that back-adjusted DOWN moves matching a clean bonus factor, but a
±10-day check against the cached corporate-action timeline corroborated
**none** of those seven events with a split or dividend — and a clean ratio
alone cannot distinguish an adjustable split/bonus from a non-adjustable
special dividend (MAHSEAMLES 2020-06-22 is a likely special-dividend false
positive). Adjusting any of them risks silent corruption, so they all route
to TRIM instead.

The TRIM step is *KEEP-guarded*: it never trims away a KEEP day. When a
symbol has TRIM events before its earliest KEEP, those become a *left* trim
(drop ``date < cutoff``); TRIM events after its latest KEEP become a *right*
trim (drop ``date >= cutoff``). The KEEP-anchored middle segment survives.

Cached dividend ex-dates are already excused by
:func:`apt.data.clean.validation_gate` — they don't appear in survivors and
need no action here.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl
from loguru import logger

from apt.data.clean import KEEP_EVENTS

# Tightened COVID crash window. The previous 04-30 bound wrongly swept in
# late-April single-name CAs (VTL, SHRIRAMCIT) that we want to TRIM.
COVID_WINDOW_START: date = date(2020, 2, 24)
COVID_WINDOW_END: date = date(2020, 4, 3)
COVID_WINDOW: tuple[date, date] = (COVID_WINDOW_START, COVID_WINDOW_END)

Category = Literal["KEEP_COVID", "KEEP_HARDCODED", "TRIM"]


def _classify_one(
    symbol: str,
    d: date,
    ret: float,
    *,
    keep_set: set[tuple[str, date]],
    covid_window: tuple[date, date] = COVID_WINDOW,
) -> tuple[Category, str]:
    """Return ``(category, note)`` for a single survivor."""
    if (symbol, d) in keep_set:
        return "KEEP_HARDCODED", "in KEEP_EVENTS"
    if covid_window[0] <= d <= covid_window[1]:
        return "KEEP_COVID", "in COVID window"
    # Any other survivor (UP, DOWN, regardless of clean-ratio match) is TRIM.
    # A clean price ratio alone can't distinguish a split/bonus from a
    # special dividend, so we never back-adjust here.
    _ = ret  # kept in signature for readability/future use
    return "TRIM", ""


def classify_survivors(
    survivors: pl.DataFrame,
    *,
    keep_events: list[tuple[str, date]] = KEEP_EVENTS,
    covid_window: tuple[date, date] = COVID_WINDOW,
) -> pl.DataFrame:
    """Add ``category`` and ``classification_note`` columns."""
    keep_set = {(s, d) for s, d in keep_events}
    out_rows: list[dict] = []
    for r in survivors.to_dicts():
        cat, note = _classify_one(
            r["symbol"],
            r["date"],
            float(r["ret"]),
            keep_set=keep_set,
            covid_window=covid_window,
        )
        out_rows.append(
            {
                **r,
                "category": cat,
                "classification_note": note,
            }
        )
    return pl.DataFrame(out_rows)


def apply_repair(df: pl.DataFrame, classified: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Apply KEEP-guarded left + right TRIMs to ``df``.

    Returns ``(df_out, summary)`` where ``summary`` reports the cutoff
    chosen for each affected symbol.
    """
    summary: dict = {
        "n_trim_left_symbols": 0,
        "n_trim_right_symbols": 0,
        "trims_left": [],
        "trims_right": [],
    }

    if classified.is_empty():
        return df, summary

    trims_by_sym: dict[str, list[date]] = {}
    keeps_by_sym: dict[str, list[date]] = {}
    for r in classified.to_dicts():
        if r["category"] == "TRIM":
            trims_by_sym.setdefault(r["symbol"], []).append(r["date"])
        elif r["category"].startswith("KEEP"):
            keeps_by_sym.setdefault(r["symbol"], []).append(r["date"])

    left_cutoffs: dict[str, date] = {}
    right_cutoffs: dict[str, date] = {}

    for sym, trims in trims_by_sym.items():
        keeps = keeps_by_sym.get(sym, [])
        if not keeps:
            left_cutoffs[sym] = max(trims)
            summary["trims_left"].append(
                {"symbol": sym, "cutoff": max(trims), "n_trim_events": len(trims)}
            )
            continue

        earliest_keep = min(keeps)
        latest_keep = max(keeps)
        pre_keep = [t for t in trims if t < earliest_keep]
        post_keep = [t for t in trims if t > latest_keep]
        between = [t for t in trims if earliest_keep <= t <= latest_keep]
        if between:
            logger.warning(
                "{}: {} TRIM event(s) lie between KEEP events ({} … {}); "
                "skipping to preserve KEEPs",
                sym,
                len(between),
                earliest_keep,
                latest_keep,
            )

        if pre_keep:
            left_cutoffs[sym] = max(pre_keep)
            summary["trims_left"].append(
                {
                    "symbol": sym,
                    "cutoff": max(pre_keep),
                    "n_trim_events": len(pre_keep),
                }
            )
        if post_keep:
            right_cutoffs[sym] = min(post_keep)
            summary["trims_right"].append(
                {
                    "symbol": sym,
                    "cutoff": min(post_keep),
                    "n_trim_events": len(post_keep),
                }
            )

    summary["n_trim_left_symbols"] = len(left_cutoffs)
    summary["n_trim_right_symbols"] = len(right_cutoffs)

    if left_cutoffs:
        lt = pl.DataFrame({"symbol": list(left_cutoffs), "_left": list(left_cutoffs.values())})
        df = (
            df.join(lt, on="symbol", how="left")
            .filter(pl.col("_left").is_null() | (pl.col("date") >= pl.col("_left")))
            .drop("_left")
        )

    if right_cutoffs:
        rt = pl.DataFrame({"symbol": list(right_cutoffs), "_right": list(right_cutoffs.values())})
        df = (
            df.join(rt, on="symbol", how="left")
            .filter(pl.col("_right").is_null() | (pl.col("date") < pl.col("_right")))
            .drop("_right")
        )

    return df, summary
