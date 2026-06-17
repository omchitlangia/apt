#!/usr/bin/env python3
"""Phase 4 final — Part 2: DSR / PBO on the crypto kalman arm (the thesis test).

Reads the A10 artifacts (reports/phase4/crypto_adaptive/) and runs the
Section-2 machinery on the kalman μ-only arm: honest trial count, Deflated
Sharpe Ratio + p-value vs the multiple-testing luck bar, and PBO via CSCV over
the per-pair return matrix. Regime A (funding-clean) is authoritative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from apt.stats.dsr import deflated_sharpe_ratio
from apt.stats.pbo import pbo_cscv

SRC = Path("reports/phase4/crypto_adaptive")
PPY = 365  # crypto trades 24/7 -> 365 sessions/year


def _kalman_trial_count(metrics: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    k = metrics[metrics.engine == "kalman_mu"]
    ledger = pd.DataFrame(
        [
            {
                "source": "kalman cells (freq{1,5,15}×cost{1,3,5,8}×regime{A,B})",
                "count": int(k.shape[0]),
            },
            {"source": "re-anchor H-grid {inf,20,10,5} (train-only selection)", "count": 4},
        ]
    )
    n = int(ledger["count"].sum())
    ledger = pd.concat(
        [ledger, pd.DataFrame([{"source": "TOTAL explicit N", "count": n}])], ignore_index=True
    )
    return n, ledger


def main() -> int:
    metrics = pd.read_csv(SRC / "metrics.csv")
    sess = pd.read_csv(SRC / "per_session_all.csv")
    n_trials, ledger = _kalman_trial_count(metrics)
    ledger.to_csv(SRC / "dsr_trial_ledger.csv", index=False)

    # trial Sharpe variance: per-period net Sharpe across all kalman cells
    ksh = metrics[metrics.engine == "kalman_mu"].net_sharpe.dropna().to_numpy() / np.sqrt(PPY)
    v = float(np.var(ksh, ddof=1)) if ksh.size > 1 else 1e-4

    rows = []
    for regime in ("A", "B"):
        k = metrics[(metrics.engine == "kalman_mu") & (metrics.regime == regime)]
        if k.empty or k.net_sharpe.isna().all():
            continue
        best = k.loc[k.net_sharpe.idxmax()]
        freq, cost = int(best.freq_min), int(best.spread_bps)
        s = sess[
            (sess.engine == "kalman_mu")
            & (sess.regime == regime)
            & (sess.freq_min == freq)
            & (sess.spread_bps == cost)
        ]
        port = s.groupby("date").net_log_ret.mean().to_numpy()
        res = deflated_sharpe_ratio(port, n_trials=n_trials, trial_sharpe_var=max(v, 1e-6))
        # PBO over per-pair return matrix at this cell
        mat = s.pivot_table(
            index="date", columns="pair", values="net_log_ret", aggfunc="mean"
        ).fillna(0.0)
        pbo = np.nan
        if mat.shape[1] >= 2 and mat.shape[0] >= 16:
            pbo = pbo_cscv(mat.to_numpy(), n_blocks=16).pbo
        rows.append(
            {
                "regime": regime,
                "funding": "clean" if regime == "A" else "UNPRICED_TODO",
                "cell": f"f{freq}/c{cost}",
                "ann_net_sharpe": float(best.net_sharpe),
                "per_period_sharpe": res.sr_hat,
                "sample_T": res.sample_length,
                "n_pairs": mat.shape[1],
                "n_trials_N": n_trials,
                "SR0_luck_bar": res.sr_benchmark,
                "DSR": res.dsr,
                "DSR_pvalue": res.p_value,
                "PBO": pbo,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(SRC / "dsr_pbo_kalman.csv", index=False)
    print(f"[crypto DSR/PBO] N={n_trials}, V={v:.6f}")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
