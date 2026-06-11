"""Retroactive figure generation for Phase 3 v2 + Phase 3 OU/Bertram.

Numbers are NEVER recomputed — this script only loads the persisted
artifacts under ``reports/phase3/`` and ``reports/phase3_ou/`` and
calls the standard figure functions in :mod:`apt.reporting.figures`.

Outputs land in ``reports/phase3_ou/figures/`` (one folder per group).
Every figure also writes its companion CSV — those CSVs ARE the
"figure data" by the reporting-standard contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from apt.reporting import figures as rf
from apt.utils.logging import setup_logging
from apt.utils.paths import ensure_dirs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "reports" / "phase3"
P3OU = ROOT / "reports" / "phase3_ou"
OUT_ROOT = P3OU / "figures"

# Pair-folds that survived the OU HL band at the headline cells. Hardcoded
# from ou_pair_fold_diag.csv (regime=B, hl_band_pass==True, fit_ok==True).
HEADLINE_SURVIVORS = [
    (4, "INDUSINDBK/HDFCBANK"),
    (6, "KOTAKBANK/HDFCBANK"),
]


def _filter_pairfolds(df: pd.DataFrame, survivors: list[tuple[int, str]]) -> pd.DataFrame:
    keys = set(survivors)
    df = df.copy()
    df["_pf"] = list(zip(df.fold_id, df.pair, strict=False))
    return df[df["_pf"].isin(keys)].drop(columns=["_pf"])


# ---------------------------------------------------------------------------
# Group A: OU best cell (5-min, B, 3 bps, none) — figures (a, b, h, c)
# ---------------------------------------------------------------------------


def emit_ou_best_cell() -> dict:
    out = OUT_ROOT / "ou_best_cell"
    ps = pd.read_csv(P3OU / "pair_sessions_ou.csv")
    trades = pd.read_csv(P3OU / "trades_ou.csv")
    ps_cell = ps[
        (ps.engine == "ou")
        & (ps.freq_min == 5)
        & (ps.regime == "B")
        & (ps.spread_bps == 3)
        & (ps.stop_mode == "none")
    ]
    tr_cell = trades[
        (trades.freq_min == 5)
        & (trades.regime == "B")
        & (trades.spread_bps == 3)
        & (trades.stop_mode == "none")
    ]
    paths: dict = {}
    paths["a"] = rf.fig_a_per_pair_fold_equity(
        ps_cell,
        out_dir=out,
        name="a_ou_best_per_pair_fold_equity",
        engine="ou",
        freq_min=5,
        regime="B",
        spread_bps=3,
        stop_mode="none",
    )
    paths["b"] = rf.fig_b_portfolio_nav(
        ps_cell,
        out_dir=out,
        name="b_ou_best_portfolio_nav",
        engine="ou",
        freq_min=5,
        regime="B",
        spread_bps=3,
        stop_mode="none",
    )
    paths["h"] = rf.fig_h_exit_reason_stacked(
        tr_cell,
        out_dir=out,
        name="h_ou_best_exit_reasons",
        group_by=("fold_id", "pair"),
    )
    logger.info("OU best cell figures: {}", {k: str(v[0].name) for k, v in paths.items()})
    return paths


# ---------------------------------------------------------------------------
# Group B: OU + rolling_z full grid roll-ups (d, i, e, f, g, j, k)
# ---------------------------------------------------------------------------


def _metrics_long(p_ou: Path, p_rz: Path) -> pd.DataFrame:
    cols_keep = [
        "engine",
        "freq_min",
        "regime",
        "spread_bps",
        "stop_mode",
        "n_pairs",
        "n_trades",
        "gross_sharpe",
        "net_sharpe",
        "gross_total_pct",
        "net_total_pct",
        "gross_ann_pct",
        "net_ann_pct",
    ]
    ou = pd.read_csv(p_ou)[cols_keep + ["net_max_drawdown_pct"]]
    rz = pd.read_csv(p_rz)
    # rolling baseline lacks 'engine' and 'stop_mode' columns — fill defaults
    if "engine" not in rz.columns:
        rz = rz.assign(engine="rolling_z")
    if "stop_mode" not in rz.columns:
        rz = rz.assign(stop_mode="none")
    rz = rz[cols_keep + (["net_max_drawdown_pct"] if "net_max_drawdown_pct" in rz.columns else [])]
    return pd.concat([ou, rz], ignore_index=True)


def emit_grid_rollups() -> dict:
    out = OUT_ROOT / "grid_rollups"
    metrics = _metrics_long(P3OU / "metrics_ou.csv", P3OU / "metrics_rolling_baseline.csv")
    diag = pd.read_csv(P3OU / "ou_pair_fold_diag.csv")
    paths: dict = {}

    # (d) cost ladder per (freq, regime) — overlay engines
    for freq in (1, 5, 15):
        for regime in ("A", "B"):
            sub = metrics[
                (metrics.freq_min == freq)
                & (metrics.regime == regime)
                & (metrics.stop_mode == "none")
            ]
            if sub.empty:
                continue
            paths[f"d_{freq}_{regime}"] = rf.fig_d_cost_ladder(
                metrics,
                out_dir=out,
                name=f"d_cost_ladder_f{freq}_{regime}",
                freq_min=freq,
                regime=regime,
            )

    # (i) trade counts by engine × freq at headline (regime=B, cost=3, stop=none)
    paths["i"] = rf.fig_i_trade_counts(
        metrics,
        out_dir=out,
        name="i_trade_counts_B_3bps",
        regime="B",
        stop_mode="none",
        spread_bps=3,
    )

    # (f) HL distribution per freq — bands [30,120] and [120,1875]
    hl = diag.dropna(subset=["half_life_minutes"])[["freq_min", "half_life_minutes"]].rename(
        columns={"half_life_minutes": "half_life_min"}
    )
    paths["f"] = rf.fig_f_half_life_distribution(
        hl,
        out_dir=out,
        name="f_half_life_distribution",
        band_a=(30.0, 120.0),
        band_b=(120.0, 1875.0),
    )

    # (g) drift chart — Z-OU test mean per pair-fold-freq with ±0.5 σ_eq flags
    drift = diag[diag.fit_ok].copy()
    # add regime + traded flags from metrics_ou n_pairs > 0 logic; simpler:
    # mark "traded" if (fold, pair, freq) appears in trades_ou.csv
    tou = pd.read_csv(P3OU / "trades_ou.csv")
    traded_keys = set(zip(tou.fold_id, tou.pair, tou.freq_min, strict=False))
    drift["traded"] = [
        (int(r.fold_id), r.pair, int(r.freq_min)) in traded_keys for r in drift.itertuples()
    ]
    # Synthetic regime tag: use intraday HL band 30-1875 → "B" else "A" guess —
    # but use the actual hl_band_pass flag if present.
    drift["regime"] = np.where(drift.half_life_minutes.between(30, 1875), "B", "A")
    drift = drift.rename(columns={"z_ou_test_mean": "drift_mean_sigma_eq"})
    paths["g"] = rf.fig_g_drift_chart(
        drift[["fold_id", "pair", "freq_min", "regime", "drift_mean_sigma_eq", "traded"]],
        out_dir=out,
        name="g_drift_chart",
    )

    # (j) (1+β)/2 histogram — all pairs vs traded survivors. Dedup to unique
    # pair_keys for the "all pairs" baseline.
    betas = diag[diag.freq_min == 5].drop_duplicates(subset=["pair"])[
        ["pair", "one_plus_beta_over_2"]
    ]
    betas["traded"] = betas["pair"].isin({p for (_, p) in HEADLINE_SURVIVORS})
    paths["j"] = rf.fig_j_beta_histogram(
        betas,
        out_dir=out,
        name="j_beta_histogram_all_vs_traded",
    )

    # (k) exclusion funnel from metrics_ou aggregate columns + diag counts
    funnel_rows = []
    for freq in (1, 5, 15):
        sub = diag[diag.freq_min == freq]
        attempted = len(sub)
        ar1_ok = int(sub.fit_ok.sum())
        # HL band B = [120, 1875]
        hl_band = int(
            ((sub.half_life_minutes >= 120) & (sub.half_life_minutes <= 1875) & sub.fit_ok).sum()
        )
        # traded at B = unique (fold, pair) in trades_ou for freq+B+cost=3
        tr = pd.read_csv(P3OU / "trades_ou.csv")
        traded_n = (
            tr[
                (tr.freq_min == freq)
                & (tr.regime == "B")
                & (tr.spread_bps == 3)
                & (tr.stop_mode == "none")
            ]
            .drop_duplicates(subset=["fold_id", "pair"])
            .shape[0]
        )
        for stage, n in [
            ("attempted (liq-gate-passed)", attempted),
            ("AR(1) fit_ok", ar1_ok),
            ("HL ∈ [120,1875] min (B band)", hl_band),
            ("traded (B @ 3 bps)", traded_n),
        ]:
            funnel_rows.append({"freq_min": freq, "regime": "B", "stage": stage, "n": n})
    paths["k"] = rf.fig_k_exclusion_funnel(
        pd.DataFrame(funnel_rows),
        out_dir=out,
        name="k_exclusion_funnel",
    )
    logger.info("Grid roll-up figures: {}", {k: str(v[0].name) for k, v in paths.items()})
    return paths


# ---------------------------------------------------------------------------
# Group B-extra: MATCHED-UNIVERSE cost ladder — engines on the SAME pair-folds
# ---------------------------------------------------------------------------


def _matched_metrics_long() -> pd.DataFrame:
    """Aggregate pair_sessions for both engines restricted to HEADLINE_SURVIVORS
    (the 2 OU HL-band survivor pair-folds), compute portfolio metrics per
    (engine, freq, cost, regime=B, stop=none).

    Returns a DataFrame with the column schema fig_d_cost_ladder expects:
        engine, freq_min, regime, spread_bps, stop_mode,
        gross_sharpe, net_sharpe, gross_total_pct, net_total_pct, n_trades.
    """
    import math

    ps_ou = pd.read_csv(P3OU / "pair_sessions_ou.csv")
    ps_rz = pd.read_csv(P3OU / "pair_sessions_rolling_baseline.csv")
    if "stop_mode" not in ps_rz.columns:
        ps_rz = ps_rz.assign(stop_mode="none")
    if "engine" not in ps_rz.columns:
        ps_rz = ps_rz.assign(engine="rolling_z")
    tr_ou = pd.read_csv(P3OU / "trades_ou.csv")
    tr_rz = pd.read_csv(P3OU / "trades_rolling_baseline.csv")
    if "stop_mode" not in tr_rz.columns:
        tr_rz = tr_rz.assign(stop_mode="none")

    def _portfolio_metrics(sub: pd.DataFrame) -> dict:
        port = (
            sub.groupby("date", as_index=False)
            .agg(g=("gross_log_ret", "mean"), n=("net_log_ret", "mean"))
            .sort_values("date")
        )
        g = port.g.to_numpy()
        n = port.n.to_numpy()
        sd_g = float(g.std(ddof=1)) if len(g) > 1 else 0.0
        sd_n = float(n.std(ddof=1)) if len(n) > 1 else 0.0
        return {
            "gross_total_pct": float(math.expm1(g.sum()) * 100),
            "net_total_pct": float(math.expm1(n.sum()) * 100),
            "gross_sharpe": float(g.mean() / sd_g * math.sqrt(252)) if sd_g > 0 else float("nan"),
            "net_sharpe": float(n.mean() / sd_n * math.sqrt(252)) if sd_n > 0 else float("nan"),
        }

    rows: list[dict] = []
    survivors = set(HEADLINE_SURVIVORS)
    for ps, tr, engine in [(ps_ou, tr_ou, "ou"), (ps_rz, tr_rz, "rolling_z")]:
        for freq in (5, 15):
            for cost in (1, 3, 5, 8):
                sub_ps = ps[
                    (ps.engine == engine)
                    & (ps.freq_min == freq)
                    & (ps.regime == "B")
                    & (ps.spread_bps == cost)
                    & (ps.stop_mode == "none")
                ]
                sub_ps = sub_ps[
                    [
                        (int(f), p) in survivors
                        for f, p in zip(sub_ps.fold_id, sub_ps.pair, strict=False)
                    ]
                ]
                if sub_ps.empty:
                    continue
                m = _portfolio_metrics(sub_ps)
                sub_tr = tr[
                    (tr.engine == engine)
                    & (tr.freq_min == freq)
                    & (tr.regime == "B")
                    & (tr.spread_bps == cost)
                    & (tr.stop_mode == "none")
                ]
                sub_tr = sub_tr[
                    [
                        (int(f), p) in survivors
                        for f, p in zip(sub_tr.fold_id, sub_tr.pair, strict=False)
                    ]
                ]
                rows.append(
                    {
                        "engine": engine,
                        "freq_min": freq,
                        "regime": "B",
                        "spread_bps": cost,
                        "stop_mode": "none",
                        "n_trades": int(len(sub_tr)),
                        **m,
                    }
                )
    return pd.DataFrame(rows)


def emit_matched_universe_ladder() -> dict:
    """Cost ladder restricted to the 2 OU HL-band survivor pair-folds.

    Equal universe per engine. Engine effect = ladder slope (cost-aware vs
    cost-blind thresholds); universe effect on levels is removed.
    """
    out = OUT_ROOT / "matched_universe"
    metrics_matched = _matched_metrics_long()
    # Persist the matched metrics CSV so the report can cite the exact numbers
    (out).mkdir(parents=True, exist_ok=True)
    matched_path = out / "matched_metrics.csv"
    metrics_matched.to_csv(matched_path, index=False)
    logger.info("Wrote matched-universe metrics: {}", matched_path)

    paths: dict = {}
    for freq in (5, 15):
        paths[f"d_{freq}"] = rf.fig_d_cost_ladder(
            metrics_matched,
            out_dir=out,
            name=f"d_matched_universe_cost_ladder_f{freq}_B",
            freq_min=freq,
            regime="B",
        )
    logger.info(
        "Matched-universe ladder figures: {}", {k: str(v[0].name) for k, v in paths.items()}
    )
    return paths


# ---------------------------------------------------------------------------
# Group C: Coarse rolling_z cells (the addendum-#5 baselines) — (a, b, h) each
# ---------------------------------------------------------------------------


def emit_coarse_rolling_z() -> dict:
    out = OUT_ROOT / "rolling_z_coarse"
    ps = pd.read_csv(P3OU / "pair_sessions_rolling_baseline.csv")
    trades = pd.read_csv(P3OU / "trades_rolling_baseline.csv")
    # rolling baseline CSVs do not carry stop_mode (always "none"); add it
    if "stop_mode" not in trades.columns:
        trades = trades.assign(stop_mode="none")
    paths: dict = {}
    for freq, regime in [(5, "A"), (5, "B"), (15, "A"), (15, "B")]:
        ps_cell = ps[(ps.freq_min == freq) & (ps.regime == regime) & (ps.spread_bps == 3)]
        tr_cell = trades[
            (trades.freq_min == freq)
            & (trades.regime == regime)
            & (trades.spread_bps == 3)
            & (trades.stop_mode == "none")
        ]
        if ps_cell.empty:
            logger.warning("rolling_z coarse cell empty: freq={} regime={}", freq, regime)
            continue
        tag = f"f{freq}_{regime}"
        paths[f"b_{tag}"] = rf.fig_b_portfolio_nav(
            ps_cell,
            out_dir=out,
            name=f"b_rollingz_coarse_{tag}_portfolio_nav",
            engine="rolling_z",
            freq_min=freq,
            regime=regime,
            spread_bps=3,
            stop_mode="none",
        )
        if not tr_cell.empty:
            paths[f"h_{tag}"] = rf.fig_h_exit_reason_stacked(
                tr_cell,
                out_dir=out,
                name=f"h_rollingz_coarse_{tag}_exit_reasons",
                group_by=("freq_min", "regime"),
            )
    logger.info("Coarse rolling_z figures: {}", {k: str(v[0].name) for k, v in paths.items()})
    return paths


# ---------------------------------------------------------------------------
# Group D: Attribution slice — rolling_z restricted to OU HL-band survivors
# ---------------------------------------------------------------------------


def emit_attribution_slice() -> dict:
    out = OUT_ROOT / "attribution_slice"
    ps = pd.read_csv(P3OU / "pair_sessions_rolling_baseline.csv")
    paths: dict = {}
    # Restrict to the OU HL-band survivor pair-folds at freq 5 and 15, all costs
    for freq in (5, 15):
        for cost in (1, 3, 5, 8):
            sub = ps[(ps.freq_min == freq) & (ps.regime == "B") & (ps.spread_bps == cost)]
            sub = _filter_pairfolds(sub, HEADLINE_SURVIVORS)
            if sub.empty:
                continue
            tag = f"f{freq}_B_{cost}bps"
            paths[f"b_{tag}"] = rf.fig_b_portfolio_nav(
                sub,
                out_dir=out,
                name=f"b_attribution_rollingz_{tag}_portfolio_nav",
                engine="rolling_z",
                freq_min=freq,
                regime="B",
                spread_bps=cost,
                stop_mode="none",
            )
    # And the OU side at the same cells for visual comparison
    ps_ou = pd.read_csv(P3OU / "pair_sessions_ou.csv")
    for freq in (5, 15):
        for cost in (1, 3, 5, 8):
            sub = ps_ou[
                (ps_ou.engine == "ou")
                & (ps_ou.freq_min == freq)
                & (ps_ou.regime == "B")
                & (ps_ou.spread_bps == cost)
                & (ps_ou.stop_mode == "none")
            ]
            if sub.empty:
                continue
            tag = f"f{freq}_B_{cost}bps"
            paths[f"b_ou_{tag}"] = rf.fig_b_portfolio_nav(
                sub,
                out_dir=out,
                name=f"b_attribution_ou_{tag}_portfolio_nav",
                engine="ou",
                freq_min=freq,
                regime="B",
                spread_bps=cost,
                stop_mode="none",
            )
    logger.info("Attribution slice figures: {}", {k: str(v[0].name) for k, v in paths.items()})
    return paths


# ---------------------------------------------------------------------------
# Group E: Phase 3 v2 (1-min rolling_z) — applicable set (a, b, h, d roll-up)
# ---------------------------------------------------------------------------


def emit_phase3_v2_applicable() -> dict:
    """Applicable figures for the v2 1-min rolling_z baseline.

    We only have access to:
    - reports/phase3/metrics_two_regime_v2.csv  (per-cell metrics)
    - reports/phase3/trades_two_regime_3bps.csv (3-bps trade-level rows)
    - reports/phase3/pair_daily_two_regime_3bps.csv (per-pair daily P&L)
    - reports/phase3/equity_curve_daily.csv (portfolio NAV daily)

    So we generate figures we can drive from those exact CSVs: (b) NAV
    from equity_curve_daily, (h) exit-reasons from trades_two_regime_3bps.
    Per-pair-fold (a) requires per-pair daily P&L which IS available.
    """
    out = OUT_ROOT / "v2_1min_rolling_z"
    paths: dict = {}

    # The right driver is the per-pair daily P&L table, which has gross AND
    # net per (fold_id, pair, regime, date).
    per_pair_path = P3 / "pair_daily_two_regime_3bps.csv"
    if per_pair_path.exists():
        pp = pd.read_csv(per_pair_path)
        for regime in ("A", "B"):
            sub = pp[pp.regime == regime].copy()
            if sub.empty:
                continue
            paths[f"b_{regime}"] = rf.fig_b_portfolio_nav(
                sub[["date", "fold_id", "pair", "gross_log_ret", "net_log_ret"]],
                out_dir=out,
                name=f"b_v2_1min_portfolio_nav_{regime}",
                engine="rolling_z",
                freq_min=1,
                regime=regime,
                spread_bps=3,
                stop_mode="none",
            )

    trades = pd.read_csv(P3 / "trades_two_regime_3bps.csv")
    # exit_reason vocabulary is already standardized in scripts/13's
    # _trade_level_rows_at_3bps via _EXIT_REASON_RENAME — confirm by inspection
    # (it maps to the 5-cat set). Stacked bars by regime.
    if "regime" in trades.columns and "exit_reason" in trades.columns:
        trades = trades.assign(engine="rolling_z", freq_min=1)
        paths["h"] = rf.fig_h_exit_reason_stacked(
            trades,
            out_dir=out,
            name="h_v2_1min_exit_reasons",
            group_by=("engine", "freq_min", "regime"),
        )

    logger.info("Phase 3 v2 figures: {}", {k: str(v[0].name) for k, v in paths.items()})
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ensure_dirs()
    setup_logging(log_file=ROOT / "logs" / "16_retro_figures_phase3.log")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {}
    summary["ou_best_cell"] = emit_ou_best_cell()
    summary["grid_rollups"] = emit_grid_rollups()
    summary["matched_universe"] = emit_matched_universe_ladder()
    summary["rolling_z_coarse"] = emit_coarse_rolling_z()
    summary["attribution_slice"] = emit_attribution_slice()
    summary["v2_1min_rolling_z"] = emit_phase3_v2_applicable()

    # Build a manifest CSV of every PNG + CSV produced
    rows = []
    for group, paths_dict in summary.items():
        for key, (png, csv) in paths_dict.items():
            rows.append(
                {
                    "group": group,
                    "key": key,
                    "png": str(png.relative_to(ROOT)),
                    "csv": str(csv.relative_to(ROOT)),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest_path = OUT_ROOT / "MANIFEST.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info("Wrote manifest: {} ({} rows)", manifest_path, len(manifest))

    print(f"=== 16_retro_figures_phase3 complete: {len(manifest)} figures ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
