#!/usr/bin/env python3
"""Script 18: figures for Unit K (adaptive equilibrium).

Reads reports/phase3_kalman/* (and the frozen-OU / rolling_z matched
metrics from reports/phase3_ou) and renders the report figures into
reports/phase3_kalman/figures/. Also generates the fold-6 INDUSINDBK
μ-overlay diagnostically (that pair-fold is NOT a traded survivor — its
HL is outside the B band — but it is the canonical −7σ hard case the
report calls out, so we render its overlay for the money figure).

No numbers are recomputed for the traded cells; the diagnostic overlay
re-runs only the filter (causal, train-only hyperparameters).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from apt.config import settings
from apt.plots.style import APT_PALETTE, apply_style
from apt.reporting import figures as rf
from apt.stats.kalman import run_local_level_mu
from apt.utils.paths import ensure_dirs

_SCRIPT_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("k17_fig", _SCRIPT_DIR / "17_kalman_equilibrium.py")
assert _spec is not None and _spec.loader is not None
_k = importlib.util.module_from_spec(_spec)
sys.modules["k17_fig"] = _k
_spec.loader.exec_module(_k)

KDIR = Path("reports/phase3_kalman")
OUT = KDIR / "figures"
SELECTED_H = 20.0


def _emit_fold6_overlay() -> None:
    """Diagnostic μ-overlay for (fold 6, INDUSINDBK/HDFCBANK) — not traded."""
    daily = pl.read_parquet("data/processed/daily_clean.parquet")
    sectors = pl.read_parquet("data/interim/sectors.parquet")
    folds = _k._v2._build_overlap_folds(daily)
    fb = {f.fold_id: f for f in folds}
    pairs = _k._v2._select_pairs_for_fold(fb[6], daily, sectors)
    target = next((p for p in pairs if p.key == "INDUSINDBK/HDFCBANK"), None)
    if target is None:
        return
    for freq in (5, 15):
        c = _k._kalman_fit_for_pair_fold(
            target, fb[6], freq_min=freq, min_obs=settings.signal.ou.min_obs
        )
        if c is None or not c["frozen_fit"].fit_ok:
            continue
        full = run_local_level_mu(
            c["spread_full"],
            c["sids_full"],
            mu_init=c["mu_init"],
            half_life_sessions=SELECTED_H,
            tradeable=c["tradeable_full"],
        )
        tm = c["test_mask"]
        ov = pd.DataFrame(
            {
                "ts": pd.DatetimeIndex(c["ts_full"][tm]),
                "spread": c["spread_full"][tm],
                "mu_frozen": c["frozen_fit"].mu,
                "mu_kalman": full.mu_path[tm],
            }
        )
        ov.to_csv(KDIR / f"mu_overlay_INDUSINDBK_HDFCBANK_fold6_f{freq}.csv", index=False)


def _fig_mu_overlay(path_csv: Path, title: str, out_name: str) -> None:
    """The money figure: spread + frozen μ + adaptive μ_t."""
    apply_style()
    df = pd.read_csv(path_csv, parse_dates=["ts"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(df.ts, df.spread, color="#9ca3af", lw=0.5, label="log-spread X")
    ax.plot(df.ts, df.mu_frozen, color=APT_PALETTE[3], lw=1.6, label="frozen μ_OU (train)")
    ax.plot(df.ts, df.mu_kalman, color=APT_PALETTE[0], lw=1.4, label="adaptive μ_t (H=20 sessions)")
    ax.set_ylabel("log-spread")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)


def _fig_drift_before_after() -> None:
    apply_style()
    d = pd.read_csv(KDIR / "drift_before_after.csv").sort_values(["freq_min", "fold_id", "pair"])
    labels = [
        f"f{int(r.freq_min)}·fold{int(r.fold_id)}·{r['pair'].split('/')[0]}"
        for _, r in d.iterrows()
    ]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(
        y - 0.2, d.drift_frozen_sigma_eq, height=0.38, color=APT_PALETTE[3], label="frozen μ_OU"
    )
    ax.barh(
        y + 0.2, d.drift_kalman_sigma_eq, height=0.38, color=APT_PALETTE[0], label="adaptive μ_t"
    )
    ax.axvline(0, color="#444", lw=0.7)
    for x in (-0.5, 0.5):
        ax.axvline(x, color="#888", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("test-slice Z mean (σ_eq units)")
    ax.set_title("Drift before (frozen) vs after (adaptive) — ±0.5σ guides")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "drift_before_after.png")
    plt.close(fig)
    d.to_csv(OUT / "drift_before_after.csv", index=False)


def _fig_selection() -> None:
    apply_style()
    s = pd.read_csv(KDIR / "selection_summary.csv")
    s["hl_label"] = s.half_life_sessions.map(lambda v: "∞" if not np.isfinite(v) else f"{int(v)}")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [APT_PALETTE[4] if a else APT_PALETTE[6] for a in s.globally_admissible]
    bars = ax.bar(s.hl_label, s.mean_ret_per_unit_time.fillna(0), color=colors)
    for b, row in zip(bars, s.itertuples(), strict=False):
        tag = "chosen" if row.chosen else ("inadmissible" if not row.globally_admissible else "")
        if tag:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                tag,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xlabel("re-anchor half-life (sessions)")
    ax.set_ylabel("mean train Bertram return/unit-time")
    ax.set_title("TRAIN-only selection (green=admissible, grey=guard-failed)")
    fig.tight_layout()
    fig.savefig(OUT / "selection.png")
    plt.close(fig)


def _fig_cost_ladder_3engine() -> None:
    """Kalman vs frozen-OU vs rolling_z net Sharpe + net total per cost, freq 5/15."""
    apply_style()
    k = pd.read_csv(KDIR / "metrics_kalman.csv")
    ou = pd.read_csv("reports/phase3_ou/metrics_ou.csv")
    ou = ou[(ou.engine == "ou") & (ou.regime == "B") & (ou.stop_mode == "none")]
    mm = pd.read_csv("reports/phase3_ou/figures/matched_universe/matched_metrics.csv")
    rz = mm[mm.engine == "rolling_z"]
    for freq in (5, 15):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for eng, df, col in [
            ("kalman_mu", k, APT_PALETTE[0]),
            ("frozen-OU", ou, APT_PALETTE[3]),
            ("rolling_z", rz, APT_PALETTE[2]),
        ]:
            sub = df[df.freq_min == freq].sort_values("spread_bps")
            axes[0].plot(sub.spread_bps, sub.net_sharpe, marker="o", color=col, label=eng, lw=1.4)
            axes[1].plot(
                sub.spread_bps, sub.net_total_pct, marker="o", color=col, label=eng, lw=1.4
            )
        for ax in axes:
            ax.axhline(0, color="#888", lw=0.5, ls="--")
            ax.set_xlabel("cost (bps)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("net Sharpe")
        axes[1].set_ylabel("net total return (%)")
        fig.suptitle(f"Cost ladder — freq={freq}-min · Regime B · matched universe (n=2)", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / f"cost_ladder_3engine_f{freq}.png")
        plt.close(fig)


def main() -> int:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    _emit_fold6_overlay()

    # Money figure: μ-overlay for INDUSINDBK folds 4 and 6 (freq 5).
    _fig_mu_overlay(
        KDIR / "mu_overlay_INDUSINDBK_HDFCBANK_fold4_f5.csv",
        "μ overlay — INDUSINDBK/HDFCBANK fold 4 (β=1.64, traded; drift −3.29σ → −0.92σ)",
        "mu_overlay_fold4.png",
    )
    f6 = KDIR / "mu_overlay_INDUSINDBK_HDFCBANK_fold6_f5.csv"
    if f6.exists():
        _fig_mu_overlay(
            f6,
            "μ overlay — INDUSINDBK/HDFCBANK fold 6 (β=1.14, NOT traded; canonical −7.0σ → −1.8σ)",
            "mu_overlay_fold6.png",
        )

    _fig_drift_before_after()
    _fig_selection()
    _fig_cost_ladder_3engine()

    # Standard (b) NAV + (h) exit-reason via the reporting module.
    ps = pd.read_csv(KDIR / "pair_sessions_kalman.csv")
    tr = pd.read_csv(KDIR / "trades_kalman.csv")
    cell = ps[(ps.freq_min == 5) & (ps.spread_bps == 3)]
    rf.fig_b_portfolio_nav(
        cell,
        out_dir=OUT,
        name="b_kalman_best_nav",
        engine="kalman_mu",
        freq_min=5,
        regime="B",
        spread_bps=3,
        stop_mode="none",
    )
    rf.fig_h_exit_reason_stacked(
        tr, out_dir=OUT, name="h_kalman_exit_reasons", group_by=("freq_min", "spread_bps")
    )

    print(f"=== 18_kalman_figures complete: figures in {OUT} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
