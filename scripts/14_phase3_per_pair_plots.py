#!/usr/bin/env python3
"""Script 14: Phase-3 per-pair + overall plots (Phase-2 style).

PLOTTING ONLY. Reads the committed v2 artifacts on phase/3-intraday
(`reports/phase3/per_pair_full_{A,B}.csv`, `trades_all_pairs_{A,B}.csv`,
`pair_daily_two_regime_3bps.csv`, `equity_curve_daily.csv`,
`fold_pairs.csv`); does NOT re-run the backtest, the selection, or any
trade logic.

Outputs:
  plots/phase3/per_pair/<PAIR>.png   one file per distinct tradeable
                                     pair (14 of them).
  plots/phase3/equity_curve_full_span.png
                                     refreshed Phase-2-style overall
                                     portfolio equity (gross + net,
                                     Regime A vs B, with & without
                                     HDFC/HDFCBANK).
  plots/phase3/spread_z_example.png  representative spread + z plot,
                                     recomputed on ONGC/OIL fold 7 (the
                                     carrier that cleared costs).

Spread + z for the per-pair lower panel and the spread_z_example plot
are recomputed deterministically from the loaded minute prices and the
frozen (alpha, beta) for the chosen fold — this is plotting, not
strategy logic. Trades, P&L, fold-pair selection all come from the
committed CSVs.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from apt.config import settings
from apt.intraday import (
    NSE_BARS_PER_SESSION,
    intraday_rolling_zscore,
    load_minute_pair,
)
from apt.plots.intraday import (
    plot_phase3_per_pair_card,
    plot_phase3_portfolio_equity,
)
from apt.plots.style import apply_style
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs

REPORTS_DIR = Path("reports/phase3")
PLOTS_DIR = Path("plots/phase3")
PER_PAIR_DIR = PLOTS_DIR / "per_pair"
MINUTE_ROOT = Path("data/interim/minute_raw")

ENTRY_Z = settings.signal.entry_z
EXIT_Z = settings.signal.exit_z
STOP_Z = settings.signal.stop_z

# Rolling-window logic identical to scripts/13: HL_days * 375 bars, clamped
# to [1 session, 5 sessions] of chronological minutes. Session warm-up 15 bars.
MIN_ROLLING_WINDOW_MIN = NSE_BARS_PER_SESSION
MAX_ROLLING_WINDOW_MIN = 5 * NSE_BARS_PER_SESSION
SESSION_WARMUP_BARS = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_rep_fold(
    pair_key: str,
    pair_folds: list[dict],
    trades_A: pd.DataFrame,
) -> dict | None:
    """Pick the fold to plot in the bottom panel.

    Criterion: the fold with the most Regime-A trades for this pair
    (the most active picture). Falls back to the first fold the pair
    was selected in.
    """
    if not pair_folds:
        return None
    counts_by_fold = trades_A[trades_A["pair"] == pair_key]["fold_id"].value_counts().to_dict()
    best_fold = max(
        pair_folds,
        key=lambda r: counts_by_fold.get(int(r["fold_id"]), 0),
        default=None,
    )
    if best_fold is None or counts_by_fold.get(int(best_fold["fold_id"]), 0) == 0:
        # No Regime-A trades anywhere for this pair → fall back to first fold
        best_fold = pair_folds[0]
    return best_fold


def _compute_spread_z_for_fold(
    pair_key: str,
    fold_row: dict,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray] | None:
    """Recompute the intraday spread + rolling z for one (pair, fold).

    Loads the minute pair for the fold's test window (clipped to the
    minute panel), then derives:
        spread[t] = log(p_y[t]) - beta * log(p_x[t]) - alpha
        z[t]      = intraday_rolling_zscore(spread, session_id,
                                             window, session_warmup)
    Deterministic given the inputs in ``fold_row``.
    """
    test_start = pd.to_datetime(fold_row["test_start"]).date()
    test_end = pd.to_datetime(fold_row["test_end"]).date()
    minute_start = max(test_start, date(2015, 2, 2))
    minute_end = min(test_end, date(2021, 6, 23))
    if minute_end < minute_start:
        return None

    aligned = load_minute_pair(
        fold_row["y_sym"],
        fold_row["x_sym"],
        minute_start,
        minute_end,
        root=MINUTE_ROOT,
    )
    if aligned.n_bars == 0:
        return None

    py = np.where(aligned.tradeable, aligned.close_y, np.nan)
    px = np.where(aligned.tradeable, aligned.close_x, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.log(py) - float(fold_row["beta"]) * np.log(px) - float(fold_row["alpha"])

    half_life_days = float(fold_row["half_life_days"])
    window = int(
        np.clip(
            round(half_life_days * NSE_BARS_PER_SESSION),
            MIN_ROLLING_WINDOW_MIN,
            MAX_ROLLING_WINDOW_MIN,
        )
    )
    z = intraday_rolling_zscore(
        spread,
        aligned.session_id,
        window=window,
        session_warmup_bars=SESSION_WARMUP_BARS,
    )
    return aligned.timestamps, spread, z


def _cum_pct_series(per_pair_daily: pd.DataFrame, pair: str, regime: str) -> pd.Series:
    """Cumulative NET % equity (per-session log-returns summed → expm1*100).

    Returns a Series indexed by date; rows with no data contribute 0.
    """
    sub = per_pair_daily[
        (per_pair_daily["pair"] == pair) & (per_pair_daily["regime"] == regime)
    ].sort_values("date")
    if sub.empty:
        return pd.Series(dtype=float)
    s = sub.set_index(pd.to_datetime(sub["date"]))["net_log_ret"].astype(float)
    return (np.expm1(s.cumsum())) * 100


def _portfolio_gross_net(
    per_pair_daily: pd.DataFrame, regime: str, exclude_keys: set[str]
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Equal-weighted MEAN-across-pairs portfolio gross/net log-return and active count.

    Mirrors the v2 orchestrator's aggregation (mean across pairs per
    session, idle bars contribute 0). Returns (gross, net, n_active)
    indexed by date.
    """
    sub = per_pair_daily[
        (per_pair_daily["regime"] == regime) & (~per_pair_daily["pair"].isin(exclude_keys))
    ].copy()
    if sub.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    agg = (
        sub.groupby("date", as_index=False)
        .agg(
            gross_log_ret=("gross_log_ret", "mean"),
            net_log_ret=("net_log_ret", "mean"),
            n_active=("active", "sum"),
        )
        .sort_values("date")
    )
    idx = pd.to_datetime(agg["date"])
    return (
        agg["gross_log_ret"].astype(float).set_axis(idx),
        agg["net_log_ret"].astype(float).set_axis(idx),
        agg["n_active"].astype(int).set_axis(idx),
    )


# ---------------------------------------------------------------------------
# Per-pair plotting
# ---------------------------------------------------------------------------


def _emit_per_pair_plots(
    per_pair_daily: pd.DataFrame,
    trades_A: pd.DataFrame,
    trades_B: pd.DataFrame,
    fold_pairs: pd.DataFrame,
    per_pair_A: pd.DataFrame,
    per_pair_B: pd.DataFrame,
) -> list[Path]:
    """Emit one figure per DISTINCT tradeable pair."""
    apply_style()
    PER_PAIR_DIR.mkdir(parents=True, exist_ok=True)

    distinct_pairs = sorted(per_pair_daily["pair"].dropna().unique().tolist())
    logger.info("Plotting {} distinct pairs", len(distinct_pairs))

    # Index the per-pair fold metadata by pair
    folds_by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in fold_pairs.to_dict(orient="records"):
        folds_by_pair[r["pair"]].append(r)

    out_paths: list[Path] = []
    for pair_key in distinct_pairs:
        pair_folds = folds_by_pair.get(pair_key, [])
        # Compose a single date axis across both regimes for this pair
        cum_A = _cum_pct_series(per_pair_daily, pair_key, "A")
        cum_B = _cum_pct_series(per_pair_daily, pair_key, "B")
        all_idx = cum_A.index.union(cum_B.index).sort_values()
        cum_A = cum_A.reindex(all_idx).ffill().fillna(0.0)
        cum_B = cum_B.reindex(all_idx).ffill().fillna(0.0)

        # Pick a representative fold and recompute spread+z for the bottom panel
        rep_fold = _pick_rep_fold(pair_key, pair_folds, trades_A)
        rep_id = None
        ts = spread = z = None
        trades_in_rep_A = trades_in_rep_B = None
        if rep_fold is not None:
            try:
                triple = _compute_spread_z_for_fold(pair_key, rep_fold)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  {}: spread/z recompute failed: {}", pair_key, exc)
                triple = None
            if triple is not None:
                rep_id = int(rep_fold["fold_id"])
                ts, spread, z = triple
                # Trades in the rep fold for this pair
                trades_in_rep_A = trades_A[
                    (trades_A["pair"] == pair_key) & (trades_A["fold_id"] == rep_id)
                ]
                trades_in_rep_B = trades_B[
                    (trades_B["pair"] == pair_key) & (trades_B["fold_id"] == rep_id)
                ]

        # Compose fold-span list for shading on the equity panel
        fold_spans = [
            (
                int(r["fold_id"]),
                pd.to_datetime(r["test_start"]).date(),
                pd.to_datetime(r["test_end"]).date(),
            )
            for r in pair_folds
        ]

        # Build the stats line (aggregated across pair-folds for the pair)
        a_rows = per_pair_A[per_pair_A["pair"] == pair_key]
        b_rows = per_pair_B[per_pair_B["pair"] == pair_key]
        a_trades = int(a_rows["n_trades"].sum()) if not a_rows.empty else 0
        b_trades = int(b_rows["n_trades"].sum()) if not b_rows.empty else 0
        a_net = float(cum_A.iloc[-1]) if not cum_A.empty else 0.0
        b_net = float(cum_B.iloc[-1]) if not cum_B.empty else 0.0
        stats_line = (
            f"Regime A: {a_trades} trades, cum net {a_net:+.1f}%   |   "
            f"Regime B: {b_trades} trades, cum net {b_net:+.1f}%   |   "
            f"selected in folds {sorted({fid for fid, _, _ in fold_spans})}"
        )

        # Tags
        sector = ""
        is_structural = False
        is_hdfcbank_anchored = "HDFCBANK" in pair_key.split("/")
        if not a_rows.empty:
            sector = str(a_rows.iloc[0].get("sector", ""))
            is_structural = bool(a_rows.iloc[0].get("is_structural", False))

        out_name = pair_key.replace("/", "_").replace("&", "AND")
        out_path = PER_PAIR_DIR / f"{out_name}.png"
        plot_phase3_per_pair_card(
            pair_key=pair_key,
            sector=sector,
            is_structural=is_structural,
            is_hdfcbank_anchored=is_hdfcbank_anchored,
            daily_dates=list(all_idx.date),
            cum_net_A_pct=cum_A.to_numpy(),
            cum_net_B_pct=cum_B.to_numpy(),
            fold_spans=fold_spans,
            rep_fold_id=rep_id,
            rep_timestamps=ts,
            rep_spread=spread,
            rep_z=z,
            trades_in_rep_fold_A=trades_in_rep_A,
            trades_in_rep_fold_B=trades_in_rep_B,
            entry_z=ENTRY_Z,
            exit_z=EXIT_Z,
            stop_z=STOP_Z,
            out_path=out_path,
            stats_line=stats_line,
        )
        out_paths.append(out_path)
        logger.info("  wrote {}", out_path)
    return out_paths


# ---------------------------------------------------------------------------
# Overall portfolio + ONGC/OIL spread example
# ---------------------------------------------------------------------------


def _emit_portfolio_plot(per_pair_daily: pd.DataFrame) -> Path:
    """Refresh equity_curve_full_span.png in Phase-2 style."""
    structural_keys = {"HDFC/HDFCBANK"}
    A_gross, A_net, A_active = _portfolio_gross_net(per_pair_daily, "A", set())
    B_gross, B_net, B_active = _portfolio_gross_net(per_pair_daily, "B", set())
    A_gross_ex, A_net_ex, _ = _portfolio_gross_net(per_pair_daily, "A", structural_keys)
    B_gross_ex, B_net_ex, _ = _portfolio_gross_net(per_pair_daily, "B", structural_keys)

    # Union date axis; missing series contribute NaN (and 0 in returns).
    all_idx = pd.DatetimeIndex(
        sorted(
            set(A_gross.index) | set(B_gross.index) | set(A_gross_ex.index) | set(B_gross_ex.index)
        )
    )

    def _align(s: pd.Series) -> pd.Series:
        return s.reindex(all_idx).fillna(0.0)

    # Use the max of the two regimes' active counts as a single "active pairs" series
    active_combined = (
        pd.concat([A_active.rename("a"), B_active.rename("b")], axis=1)
        .reindex(all_idx)
        .fillna(0)
        .max(axis=1)
        .astype(int)
    )

    df = pd.DataFrame(
        {
            "date": all_idx,
            "A_gross": _align(A_gross).values,
            "A_net": _align(A_net).values,
            "B_gross": _align(B_gross).values,
            "B_net": _align(B_net).values,
            "A_gross_ex": _align(A_gross_ex).values,
            "A_net_ex": _align(A_net_ex).values,
            "B_gross_ex": _align(B_gross_ex).values,
            "B_net_ex": _align(B_net_ex).values,
            "n_active_pairs": active_combined.values,
        }
    )

    out_path = PLOTS_DIR / "equity_curve_full_span.png"
    plot_phase3_portfolio_equity(df, out_path)
    logger.info("  wrote {}", out_path)
    return out_path


def _emit_ongc_oil_spread_example(
    fold_pairs: pd.DataFrame, trades_A: pd.DataFrame, trades_B: pd.DataFrame
) -> Path | None:
    """Regenerate spread_z_example.png on ONGC/OIL fold 7 — the carrier
    that cleared costs in Regime A (the only positive Regime-A pair-fold)."""
    apply_style()
    import matplotlib.pyplot as plt

    target = fold_pairs[(fold_pairs["pair"] == "ONGC/OIL") & (fold_pairs["fold_id"] == 7)]
    if target.empty:
        logger.warning("ONGC/OIL fold 7 not found in fold_pairs.csv")
        return None
    fold_row = target.iloc[0].to_dict()
    triple = _compute_spread_z_for_fold("ONGC/OIL", fold_row)
    if triple is None:
        logger.warning("Could not recompute spread/z for ONGC/OIL fold 7")
        return None
    ts, spread, z = triple

    # Zoom to the FIRST trading session of the fold for readability —
    # 375 bars at minute resolution is what a single chart can render
    # without becoming a smear.
    ts_index = pd.DatetimeIndex(ts)
    first_date = ts_index.date[0]
    mask = ts_index.date == first_date
    if mask.sum() == 0:
        return None
    ts_z = ts_index[mask]
    spread_z = spread[mask]
    z_z = z[mask]

    fig, (ax_s, ax_z) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[1.0, 1.3])
    ax_s.plot(ts_z, spread_z, color="#2E86AB", lw=0.7, label="spread")
    if np.isfinite(spread_z).any():
        ax_s.axhline(
            float(np.nanmean(spread_z)), color="#7E8083", lw=0.6, ls="--", label="session-mean"
        )
    ax_s.set_ylabel("spread (log)")
    ax_s.legend(loc="upper left", fontsize=8)

    ax_z.plot(ts_z, z_z, color="#2E86AB", lw=0.6, label="rolling z")
    for k, c, label in (
        (ENTRY_Z, "#C73E1D", f"±{ENTRY_Z:g} entry"),
        (EXIT_Z, "#3B8E5C", f"±{EXIT_Z:g} exit"),
        (STOP_Z, "#A23B72", f"±{STOP_Z:g} stop"),
    ):
        ax_z.axhline(+k, color=c, lw=0.5, ls="--", label=label)
        ax_z.axhline(-k, color=c, lw=0.5, ls="--")
    finite = z_z[np.isfinite(z_z)]
    if finite.size:
        zmax = max(STOP_Z * 1.05, float(np.nanmax(np.abs(finite))) * 1.05)
        ax_z.set_ylim(-zmax, +zmax)
    ax_z.set_ylabel("rolling z-score")
    ax_z.set_xlabel("time (IST)")
    ax_z.legend(loc="upper left", ncol=4, fontsize=8)

    fig.suptitle(
        f"ONGC/OIL — fold 7 representative session  ({first_date})  "
        f"|  the only Regime-A pair-fold to clear costs (+0.23 Sharpe @ 3 bps)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = PLOTS_DIR / "spread_z_example.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("  wrote {}", out_path)
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging(log_file=settings.paths.logs_dir / "14_phase3_per_pair_plots.log")
    ensure_dirs()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_PAIR_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading v2 artifacts")
    per_pair_daily = pd.read_csv(REPORTS_DIR / "pair_daily_two_regime_3bps.csv")
    trades_A = pd.read_csv(REPORTS_DIR / "trades_all_pairs_A.csv")
    trades_B = pd.read_csv(REPORTS_DIR / "trades_all_pairs_B.csv")
    fold_pairs = pd.read_csv(REPORTS_DIR / "fold_pairs.csv")
    per_pair_A = pd.read_csv(REPORTS_DIR / "per_pair_full_A.csv")
    per_pair_B = pd.read_csv(REPORTS_DIR / "per_pair_full_B.csv")

    logger.info(
        "  per_pair_daily: {} rows  |  trades A: {}  |  trades B: {}  |  fold-pair selections: {}",
        len(per_pair_daily),
        len(trades_A),
        len(trades_B),
        len(fold_pairs),
    )

    pair_paths = _emit_per_pair_plots(
        per_pair_daily=per_pair_daily,
        trades_A=trades_A,
        trades_B=trades_B,
        fold_pairs=fold_pairs,
        per_pair_A=per_pair_A,
        per_pair_B=per_pair_B,
    )
    portfolio_path = _emit_portfolio_plot(per_pair_daily)
    spread_z_path = _emit_ongc_oil_spread_example(fold_pairs, trades_A, trades_B)

    print()
    print("=== 14_phase3_per_pair_plots complete ===")
    print(f"  per-pair plots         : {len(pair_paths)} files in {PER_PAIR_DIR}")
    print(f"  overall portfolio plot : {portfolio_path}")
    print(f"  spread/z example       : {spread_z_path}")


if __name__ == "__main__":
    main()
