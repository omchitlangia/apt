#!/usr/bin/env python3
"""Phase 4 — Section 5: Johansen pair selection (NSE).

Runs Johansen (order-independent) AND a fresh Engle-Granger+BH-FDR pass on the
SAME 234 correlation-screened candidate pairs over a common window from
daily_clean — an apples-to-apples selection-path comparison. Johansen CHANGES
the selected universe, so its pairs are NOT merged into any matched table; the
section reports the universe difference and flags non-comparability.

Outputs (reports/phase4/johansen/):
  johansen_selection.csv  — per-pair EG vs Johansen verdicts
  comparison_summary.csv  — set overlap (EG-FDR vs Johansen), new/lost counts
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

from apt.phase4 import savefig_with_csv  # noqa: E402
from apt.stats.johansen import is_cointegrated, johansen_pair_test  # noqa: E402

OUT = Path("reports/phase4/johansen")
FIG = Path("plots/phase4/johansen")
WINDOW = 1008  # trailing common trading days (matches EG n_obs=1008)
MIN_OBS = 250
EG_ALPHA = 0.05


def _log_close_map(daily: pl.DataFrame, symbols: set[str]) -> dict[str, pd.Series]:
    out = {}
    for sym in symbols:
        d = daily.filter(pl.col("symbol") == sym).select(["date", "close"]).to_pandas()
        d["date"] = pd.to_datetime(d["date"])
        out[sym] = np.log(d.set_index("date").close)
    return out


def _eg_test(log_y: np.ndarray, log_x: np.ndarray) -> tuple[float, float]:
    X = sm.add_constant(log_x)
    fit = sm.OLS(log_y, X).fit()
    resid = np.asarray(fit.resid)
    try:
        p = float(adfuller(resid, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        p = np.nan
    return p, float(fit.params[1])


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pl.read_parquet("data/processed/daily_clean.parquet")
    cand = pl.read_parquet("data/pairs/cointegrated_pairs.parquet").to_pandas()
    symbols = set(cand.sym1) | set(cand.sym2)
    lc = _log_close_map(daily, symbols)

    rows = []
    for _, r in cand.iterrows():
        sy, sx = lc.get(r.sym1), lc.get(r.sym2)
        if sy is None or sx is None:
            continue
        idx = sy.index.intersection(sx.index)
        if len(idx) < MIN_OBS:
            continue
        idx = idx[-WINDOW:]
        ly, lx = sy.loc[idx].to_numpy(), sx.loc[idx].to_numpy()
        adf_p, eg_beta = _eg_test(ly, lx)
        jt = johansen_pair_test(ly, lx)
        rows.append(
            {
                "sym1": r.sym1,
                "sym2": r.sym2,
                "sector": r.sector,
                "n_obs": len(idx),
                "eg_adf_pvalue": adf_p,
                "eg_beta": eg_beta,
                "eg_persisted_fdr_pass": bool(r.fdr_pass),
                "johansen_trace_r0": jt.trace_stat_r0,
                "johansen_crit95": jt.trace_crit_r0[1],
                "johansen_rank95": jt.rank_95,
                "johansen_beta": jt.beta,
                "johansen_coint": bool(is_cointegrated(jt)),
            }
        )
    df = pd.DataFrame(rows)
    # fresh BH-FDR on the EG p-values
    valid = df.eg_adf_pvalue.notna()
    df["eg_fdr_pass"] = False
    if valid.any():
        rej, _, _, _ = multipletests(
            df.loc[valid, "eg_adf_pvalue"], alpha=EG_ALPHA, method="fdr_bh"
        )
        df.loc[valid, "eg_fdr_pass"] = rej
    df.to_csv(OUT / "johansen_selection.csv", index=False)

    df["eg_raw_pass"] = df.eg_adf_pvalue < EG_ALPHA
    eg_set = set(zip(df[df.eg_fdr_pass].sym1, df[df.eg_fdr_pass].sym2, strict=True))
    jo_set = set(zip(df[df.johansen_coint].sym1, df[df.johansen_coint].sym2, strict=True))
    both = eg_set & jo_set
    summary = pd.DataFrame(
        [
            {
                "n_candidates_tested": int(df.shape[0]),
                "n_eg_raw_p05_selected": int(df.eg_raw_pass.sum()),
                "n_eg_fdr_selected": len(eg_set),
                "n_johansen_selected": len(jo_set),
                "n_both": len(both),
                "n_johansen_only_NEW": len(jo_set - eg_set),
                "n_eg_only_LOST": len(eg_set - jo_set),
                "jaccard": len(both) / len(eg_set | jo_set) if (eg_set | jo_set) else np.nan,
                "fresh_eg_vs_persisted_fdr_agreement": float(
                    (df.eg_fdr_pass == df.eg_persisted_fdr_pass).mean()
                ),
            }
        ]
    )
    summary.to_csv(OUT / "comparison_summary.csv", index=False)
    return df, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df, summary = run()

    fig, ax = plt.subplots(figsize=(7, 5))
    sel = {
        (True, True): ("both", "C2"),
        (True, False): ("EG-only", "C0"),
        (False, True): ("Johansen-only", "C1"),
        (False, False): ("neither", "0.8"),
    }
    for (eg, jo), (lab, col) in sel.items():
        s = df[(df.eg_fdr_pass == eg) & (df.johansen_coint == jo)]
        ax.scatter(s.eg_adf_pvalue, s.johansen_trace_r0, s=14, c=col, label=f"{lab} ({len(s)})")
    ax.axvline(EG_ALPHA, color="C0", ls="--", lw=0.8)
    ax.axhline(df.johansen_crit95.median(), color="C1", ls="--", lw=0.8)
    ax.set_xlabel("EG ADF p-value")
    ax.set_ylabel("Johansen trace stat (r=0)")
    ax.set_title("EG vs Johansen selection — NSE candidate pairs (CONFOUND: different universes)")
    ax.legend(fontsize=8)
    savefig_with_csv(fig, FIG / "eg_vs_johansen.png", df)
    plt.close(fig)

    print("=== Section 5 — Johansen vs EG selection ===")
    print(summary.to_string(index=False))
    print("\nsample Johansen-only (NEW) pairs:")
    new = df[df.johansen_coint & ~df.eg_fdr_pass][
        ["sym1", "sym2", "sector", "eg_adf_pvalue", "johansen_rank95"]
    ]
    print(new.head(10).to_string(index=False))
    print("\n=== s5_johansen complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
