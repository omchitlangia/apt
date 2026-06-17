#!/usr/bin/env python3
"""Phase 4 — Section 2: DSR / PBO gate (retroactive; no strategy re-runs).

Builds the per-period (daily) NET return matrix of the evaluated NSE
configurations on the MATCHED 2 pair-folds from the persisted per-session
CSVs, then:

  2a honest trial count N           -> 2a_trial_ledger.csv
  2b Deflated Sharpe per engine best -> 2b_dsr.csv
  2c PBO via CSCV over the grid       -> 2c_pbo.csv + logit plot
  2d honesty caveat                   -> printed + report

All Sharpes are de-annualized (per-period) inside the DSR/PBO machinery.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from apt.phase4 import savefig_with_csv  # noqa: E402
from apt.stats.dsr import deflated_sharpe_ratio  # noqa: E402
from apt.stats.pbo import pbo_cscv  # noqa: E402

KALMAN = Path("reports/phase3_kalman")
OU = Path("reports/phase3_ou")
OUT = Path("reports/phase4/dsr_pbo")
FIG = Path("plots/phase4/dsr_pbo")
PPY = 252  # periods per year (daily sessions)
MATCHED = {(4, "INDUSINDBK/HDFCBANK"), (6, "KOTAKBANK/HDFCBANK")}


def _cell_series(ps: pd.DataFrame, freq: int, cost: int, date_axis=None) -> pd.Series:
    """Matched-portfolio daily NET log-return series for one cell."""
    sub = ps[(ps.freq_min == freq) & (ps.spread_bps == cost)]
    sub = sub[sub.apply(lambda r: (r.fold_id, r.pair) in MATCHED, axis=1)]
    port = sub.groupby("date").net_log_ret.mean()
    port.index = pd.to_datetime(port.index)
    if date_axis is not None:
        port = port.reindex(date_axis, fill_value=0.0)
    return port


def build_matrix() -> tuple[pd.DataFrame, dict]:
    """T×N daily NET-return matrix over the 24 none-stop candidate cells."""
    ps_k = pd.read_csv(KALMAN / "pair_sessions_kalman.csv")
    ps_o = pd.read_csv(OU / "pair_sessions_ou.csv")
    ps_o = ps_o[ps_o.stop_mode == "none"]
    ps_r = pd.read_csv(OU / "pair_sessions_rolling_baseline.csv")

    # union daily axis across all cells
    dates = set()
    for ps in (ps_k, ps_o, ps_r):
        d = ps[ps.apply(lambda r: (r.fold_id, r.pair) in MATCHED, axis=1)].date
        dates |= set(pd.to_datetime(d))
    axis = pd.DatetimeIndex(sorted(dates))

    cols = {}
    for eng, ps in (("kalman", ps_k), ("ou", ps_o), ("rz", ps_r)):
        for freq in (5, 15):
            for cost in (1, 3, 5, 8):
                cols[f"{eng}_f{freq}_c{cost}"] = _cell_series(ps, freq, cost, axis)
    mat = pd.DataFrame(cols, index=axis)
    return mat, {"n_dates": len(axis)}


def trial_ledger() -> pd.DataFrame:
    rows = [
        {
            "source": "OU grid",
            "detail": "freq{1,5,15}×regime{A,B}×cost{1,3,5,8}×stop{none}=24 + cost3×stop{hard}=6",
            "count": 30,
        },
        {"source": "rolling_z coarse grid", "detail": "coarse search cells", "count": 4},
        {"source": "kalman cells", "detail": "freq{5,15}×cost{1,3,5,8}", "count": 8},
        {
            "source": "kalman re-anchor H-grid",
            "detail": "{inf,20,10,5} train-only selection knob",
            "count": 4,
        },
    ]
    df = pd.DataFrame(rows)
    explicit = int(df["count"].sum())
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {"source": "TOTAL explicit N", "detail": "sum of the above", "count": explicit},
                    {
                        "source": "implicit (HL bands {A,B,C}, best-of-cell reporting)",
                        "detail": "compounds further; not added to headline N",
                        "count": 0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    df.to_csv(OUT / "2a_trial_ledger.csv", index=False)
    print(f"[2a] honest explicit trial count N = {explicit}")
    return df, explicit


def _trial_sharpe_variance() -> tuple[float, int, int]:
    """V = variance of per-period NET Sharpe across non-degenerate evaluated cells."""
    sh = []
    om = pd.read_csv(OU / "metrics_ou.csv")
    sh += list(om.net_sharpe.dropna())
    km = pd.read_csv(KALMAN / "metrics_kalman.csv")
    sh += list(km.net_sharpe.dropna())
    mm = pd.read_csv(OU / "figures/matched_universe/matched_metrics.csv")
    sh += list(mm[mm.engine == "rolling_z"].net_sharpe.dropna())
    sh = np.array(sh, dtype=float)
    n_all = sh.size
    # de-annualize and drop degenerate blow-ups (annualized Sharpe < -2)
    keep = sh[sh > -2.0] / np.sqrt(PPY)
    return float(np.var(keep, ddof=1)), int(keep.size), int(n_all - keep.size)


def dsr_table(mat: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    v, n_kept, n_dropped = _trial_sharpe_variance()
    km = pd.read_csv(KALMAN / "metrics_kalman.csv")
    om = pd.read_csv(OU / "metrics_ou.csv")
    om = om[(om.regime == "B") & (om.stop_mode == "none")]
    mm = pd.read_csv(OU / "figures/matched_universe/matched_metrics.csv")
    rz = mm[mm.engine == "rolling_z"]

    # selected = argmax net Sharpe within each engine's candidate set
    k_best = km.loc[km.net_sharpe.idxmax()]
    o_best = om.loc[om.net_sharpe.idxmax()]
    r_best = rz.loc[rz.net_sharpe.idxmax()]
    candidates = [
        ("kalman", int(k_best.freq_min), int(k_best.spread_bps), float(k_best.net_sharpe)),
        ("ou", int(o_best.freq_min), int(o_best.spread_bps), float(o_best.net_sharpe)),
        ("rz", int(r_best.freq_min), int(r_best.spread_bps), float(r_best.net_sharpe)),
    ]
    rows = []
    for eng, freq, cost, ann_sh in candidates:
        col = f"{eng}_f{freq}_c{cost}"
        series = mat[col].to_numpy()
        res = deflated_sharpe_ratio(series, n_trials=n_trials, trial_sharpe_var=v)
        rows.append(
            {
                "engine": eng,
                "selected_cell": f"f{freq}/c{cost}",
                "ann_net_sharpe": ann_sh,
                "per_period_sharpe": res.sr_hat,
                "sample_length_T": res.sample_length,
                "skew": res.skew,
                "kurtosis_nonexcess": res.kurtosis_nonexcess,
                "n_trials_N": n_trials,
                "trial_sharpe_var_V": v,
                "SR0_deflator": res.sr_benchmark,
                "PSR_vs_zero": res.psr_vs_zero,
                "DSR": res.dsr,
                "DSR_pvalue": res.p_value,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "2b_dsr.csv", index=False)
    print(f"[2b] V={v:.5f} (per-period, n={n_kept} cells, {n_dropped} degenerate dropped)")
    return df


def dsr_sensitivity(mat: pd.DataFrame) -> pd.DataFrame:
    """DSR of the kalman best cell vs the assumed trial count N."""
    v, _, _ = _trial_sharpe_variance()
    series = mat["kalman_f5_c1"].to_numpy()
    rows = []
    for n in (1, 30, 46, 100, 200):
        res = deflated_sharpe_ratio(series, n_trials=n, trial_sharpe_var=v)
        rows.append(
            {
                "n_trials": n,
                "SR0_deflator": res.sr_benchmark,
                "DSR": res.dsr,
                "p_value": res.p_value,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "2b_dsr_sensitivity_to_N.csv", index=False)
    print("[2b] DSR(kalman 5/1) sensitivity to N:")
    print(df.round(4).to_string(index=False))
    return df


def pbo_table(mat: pd.DataFrame) -> pd.DataFrame:
    m = mat.to_numpy()
    res = pbo_cscv(m, n_blocks=16)
    df = pd.DataFrame(
        [
            {
                "n_strategies": res.n_strategies,
                "n_blocks_S": res.n_splits_blocks,
                "n_combinations": res.n_combinations,
                "PBO": res.pbo,
                "frac_is_best_also_oos_best": res.frac_is_best_also_oos_best,
                "logit_median": float(np.median(res.logits)),
            }
        ]
    )
    df.to_csv(OUT / "2c_pbo.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(res.logits, bins=40, color="C0", alpha=0.8)
    ax.axvline(0, color="r", lw=1.2, label=f"PBO = {res.pbo:.3f}")
    ax.set_xlabel("logit λ = ln(ω/(1-ω))  [<0 ⇒ IS-best below OOS median]")
    ax.set_ylabel("count")
    ax.set_title(f"CSCV logit distribution — {res.n_strategies} NSE cells, S={res.n_splits_blocks}")
    ax.legend()
    savefig_with_csv(fig, FIG / "pbo_logit_distribution.png", pd.DataFrame({"logit": res.logits}))
    plt.close(fig)
    print(
        f"[2c] PBO = {res.pbo:.4f} over {res.n_combinations} CSCV splits, {res.n_strategies} cells"
    )
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    mat, meta = build_matrix()
    print(f"[matrix] {mat.shape[0]} dates x {mat.shape[1]} cells")
    ledger, n_trials = trial_ledger()
    dsr = dsr_table(mat, n_trials)
    dsr_sensitivity(mat)
    pbo = pbo_table(mat)
    print("\n=== 2b DSR ===")
    print(dsr.round(4).to_string(index=False))
    print("\n=== 2c PBO ===")
    print(pbo.round(4).to_string(index=False))
    print("\n=== s2_dsr_pbo complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
