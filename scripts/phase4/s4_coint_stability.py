#!/usr/bin/env python3
"""Phase 4 — Section 4: rolling cointegration-stability gate (NSE).

Rolling ADF p-value on the daily spread of each fold_pairs pair-fold over its
test window; blacklist on degradation. Machinery is the deliverable; on the
2 intraday-traded survivors the gate is EXPECTED near-null (they are
cointegrated by construction) — that result is marked [TODO]-pending-breadth.

Outputs (reports/phase4/coint_stability/):
  gate_summary.csv  — per pair-fold ADF degradation + gate decision (19 pf)
  gate_impact.csv   — net/DD with vs without gating on the kalman 5/3 survivors
  adf_paths_<pf>.csv — rolling p-value path (figure companions)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from apt.backtest import compute_metrics  # noqa: E402
from apt.phase4 import savefig_with_csv  # noqa: E402
from apt.stats.coint_stability import (  # noqa: E402
    ADF_PVALUE_THRESHOLD,
    CONSECUTIVE_WINDOWS_TO_GATE,
    gate_summary,
    rolling_adf_pvalues,
)

OUT = Path("reports/phase4/coint_stability")
FIG = Path("plots/phase4/coint_stability")
MATCHED = {(4, "INDUSINDBK/HDFCBANK"), (6, "KOTAKBANK/HDFCBANK")}


def _daily_close(daily: pl.DataFrame, sym: str) -> pd.Series:
    d = daily.filter(pl.col("symbol") == sym).select(["date", "close"]).to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date").close


def _test_window_idx(cy: pd.Series, cx: pd.Series, test_start, test_end) -> pd.DatetimeIndex:
    idx = cy.index.intersection(cx.index)
    return idx[(idx >= pd.Timestamp(test_start)) & (idx <= pd.Timestamp(test_end))]


def build_gate_summary(daily: pl.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in fp.iterrows():
        cy, cx = _daily_close(daily, r.y_sym), _daily_close(daily, r.x_sym)
        idx = _test_window_idx(cy, cx, r.test_start, r.test_end)
        if len(idx) < 90:
            continue
        spread = (np.log(cy.loc[idx]) - r.beta * np.log(cx.loc[idx]) - r.alpha).to_numpy()
        roll = rolling_adf_pvalues(spread)
        summ = gate_summary(roll)
        dates = list(idx)
        summ.update(
            {
                "fold_id": int(r.fold_id),
                "pair": r.pair,
                "traded_survivor": (int(r.fold_id), r.pair) in MATCHED,
                "gate_date": str(dates[summ["gate_end_idx"]]) if summ["gated"] else "",
            }
        )
        rows.append(summ)
        tag = r.pair.replace("/", "_")
        pd.DataFrame(
            {"end_date": [str(dates[e]) for e in roll.end_idx], "adf_pvalue": roll.pvalues}
        ).to_csv(OUT / f"adf_path_{tag}_fold{int(r.fold_id)}.csv", index=False)
    df = pd.DataFrame(rows)
    cols = [
        "fold_id",
        "pair",
        "traded_survivor",
        "n_windows",
        "n_windows_degraded",
        "frac_degraded",
        "mean_pvalue",
        "max_pvalue",
        "gated",
        "gate_date",
    ]
    df = df[cols].sort_values(["fold_id", "pair"])
    df.to_csv(OUT / "gate_summary.csv", index=False)
    return df


def gate_impact(daily: pl.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    """Net/DD with vs without the gate on the kalman 5/3 matched survivors."""
    ps = pd.read_csv("reports/phase3_kalman/pair_sessions_kalman.csv")
    ps = ps[(ps.freq_min == 5) & (ps.spread_bps == 3)].copy()
    ps = ps[ps.apply(lambda x: (x.fold_id, x.pair) in MATCHED, axis=1)]
    ps["date"] = pd.to_datetime(ps.date)

    gate_dates = {}
    for _, r in fp.iterrows():
        if (int(r.fold_id), r.pair) not in MATCHED:
            continue
        cy, cx = _daily_close(daily, r.y_sym), _daily_close(daily, r.x_sym)
        idx = _test_window_idx(cy, cx, r.test_start, r.test_end)
        spread = (np.log(cy.loc[idx]) - r.beta * np.log(cx.loc[idx]) - r.alpha).to_numpy()
        roll = rolling_adf_pvalues(spread)
        summ = gate_summary(roll)
        gate_dates[(int(r.fold_id), r.pair)] = (
            pd.Timestamp(list(idx)[summ["gate_end_idx"]]) if summ["gated"] else None
        )

    def _metrics(masked: bool) -> dict:
        sub = ps.copy()
        if masked:
            keep = []
            for _, row in sub.iterrows():
                gd = gate_dates.get((row.fold_id, row.pair))
                keep.append(gd is None or row.date <= gd)
            sub = sub[pd.Series(keep, index=sub.index)]
        port = sub.groupby("date").agg(g=("gross_log_ret", "mean"), n=("net_log_ret", "mean"))
        mn = compute_metrics(port.n.to_numpy())
        return {
            "net_total_pct": mn["total_return_pct"],
            "net_sharpe": mn["sharpe"],
            "net_maxDD_pct": mn["max_drawdown_pct"],
            "n_sessions": int(port.shape[0]),
        }

    base, gated = _metrics(False), _metrics(True)
    df = pd.DataFrame(
        [
            {"scenario": "no_gate", **base},
            {"scenario": "gated", **gated},
            {
                "scenario": "delta",
                "net_total_pct": gated["net_total_pct"] - base["net_total_pct"],
                "net_sharpe": gated["net_sharpe"] - base["net_sharpe"],
                "net_maxDD_pct": gated["net_maxDD_pct"] - base["net_maxDD_pct"],
                "n_sessions": 0,
            },
        ]
    )
    df["n_survivors_gated"] = sum(1 for v in gate_dates.values() if v is not None)
    df.to_csv(OUT / "gate_impact.csv", index=False)
    return df


def threshold_sensitivity(daily: pl.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    """How many of the 19 pair-folds gate out across threshold/window settings."""
    rows = []
    for window in (60, 120):
        for thr in (0.05, 0.10, 0.25, 0.50):
            for cons in (3,):
                n_gated = 0
                n_eval = 0
                for _, r in fp.iterrows():
                    cy, cx = _daily_close(daily, r.y_sym), _daily_close(daily, r.x_sym)
                    idx = _test_window_idx(cy, cx, r.test_start, r.test_end)
                    if len(idx) < window + 30:
                        continue
                    spread = (
                        np.log(cy.loc[idx]) - r.beta * np.log(cx.loc[idx]) - r.alpha
                    ).to_numpy()
                    roll = rolling_adf_pvalues(spread, window=window)
                    n_eval += 1
                    if gate_summary(roll, threshold=thr, consecutive=cons)["gated"]:
                        n_gated += 1
                rows.append(
                    {
                        "window": window,
                        "threshold": thr,
                        "consecutive": cons,
                        "n_eval": n_eval,
                        "n_gated": n_gated,
                        "frac_gated": n_gated / n_eval if n_eval else np.nan,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    daily = pl.read_parquet("data/processed/daily_clean.parquet")
    fp = pd.read_csv("reports/phase3/fold_pairs.csv")
    print(f"[s4] threshold p>{ADF_PVALUE_THRESHOLD}, consecutive={CONSECUTIVE_WINDOWS_TO_GATE}")
    gs = build_gate_summary(daily, fp)
    impact = gate_impact(daily, fp)
    sens = threshold_sensitivity(daily, fp)
    print("\n[4b] threshold/window sensitivity (n_gated / n_eval):")
    print(sens.to_string(index=False))

    # figure: ADF paths for survivors + the most-degraded pair-fold
    fig, ax = plt.subplots(figsize=(9, 4))
    show = gs.sort_values("mean_pvalue", ascending=False).head(3)
    for _, r in (
        pd.concat([gs[gs.traded_survivor], show]).drop_duplicates(["fold_id", "pair"]).iterrows()
    ):
        f = OUT / f"adf_path_{r.pair.replace('/', '_')}_fold{int(r.fold_id)}.csv"
        if f.exists():
            d = pd.read_csv(f)
            ax.plot(
                range(len(d)),
                d.adf_pvalue,
                marker=".",
                ms=3,
                label=f"f{int(r.fold_id)} {r.pair}{' (surv)' if r.traded_survivor else ''}",
            )
    ax.axhline(
        ADF_PVALUE_THRESHOLD, color="r", ls="--", lw=1, label=f"threshold {ADF_PVALUE_THRESHOLD}"
    )
    ax.set_xlabel("rolling window #")
    ax.set_ylabel("ADF p-value (spread level)")
    ax.set_title("rolling cointegration-stability — ADF p-value paths")
    ax.legend(fontsize=7)
    savefig_with_csv(fig, FIG / "adf_pvalue_paths.png", gs)
    plt.close(fig)

    print(f"\n[4b] {gs.gated.sum()}/{len(gs)} pair-folds gate out (broad daily universe)")
    print(gs.to_string(index=False))
    print("\n[4b] gate impact on kalman 5/3 matched survivors:")
    print(impact.to_string(index=False))
    print("\n=== s4_coint_stability complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
