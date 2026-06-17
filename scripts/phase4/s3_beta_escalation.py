#!/usr/bin/env python3
"""Phase 4 — Section 3: β-escalation (joint β+μ Kalman).

Runs the joint (β, c) per-session causal filter (`apt.stats.kalman_beta`) on
the 2 traded survivors + the fold-6 INDUSINDBK −1.79σ diagnostic, loading ONLY
those pair-folds from minute data (Pair/Fold reconstructed from
`reports/phase3/fold_pairs.csv`; no pair-selection re-run).

Outputs (reports/phase4/beta_escalation/):
  selection_beta.csv     — train-only H_β selection (guard + criterion)
  beta_collapse.csv      — per pair-fold β_t path summary + collapse flag
  drift_3way.csv         — frozen vs μ-only vs β+μ test-slice drift (σ_eq)
  metrics_beta_mu.csv    — 8-cell β+μ metrics (gross + net)
  compare_3engine.csv    — β+μ vs μ-only vs frozen, matched cells
  beta_path_<pf>.csv      — per-session β_s path (figure companion)
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from apt.backtest import Fold, Pair, compute_metrics  # noqa: E402
from apt.config import settings  # noqa: E402
from apt.intraday import (  # noqa: E402
    NSE_BARS_PER_SESSION,
    load_minute_pair,
    resample_within_session,
)
from apt.intraday.backtest import run_pair_fold  # noqa: E402
from apt.intraday.costs import CostBreakdown  # noqa: E402
from apt.intraday.signals import generate_signals_ou  # noqa: E402
from apt.phase4 import savefig_with_csv  # noqa: E402
from apt.stats.kalman import run_local_level_mu  # noqa: E402
from apt.stats.kalman_beta import (  # noqa: E402
    beta_collapse_flag,
    beta_stable_on_train,
    run_joint_beta_mu,
)
from apt.stats.ou import bertram_threshold, fit_ou_params  # noqa: E402

_SD = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s13b", _SD / "13_phase3_intraday.py")
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
MINUTE_ROOT, MPS, MPE = _v2.MINUTE_ROOT, _v2.MINUTE_PANEL_START, _v2.MINUTE_PANEL_END

OUT = Path("reports/phase4/beta_escalation")
FIG = Path("plots/phase4/beta_escalation")
FREQS = (5, 15)
COSTS = (1, 3, 5, 8)
H_C = 20.0  # inherited from Unit K (mu-only selection)
H_BETA_GRID = (math.inf, 40.0, 20.0, 10.0)
GUARD = (0.5, 1.5)
SURVIVORS = [(4, "INDUSINDBK/HDFCBANK"), (6, "KOTAKBANK/HDFCBANK")]
DIAG = (6, "INDUSINDBK/HDFCBANK")  # fold-6 -7sigma diagnostic (not traded)
ALL_PF = SURVIVORS + [DIAG]


def _load_pairs() -> dict:
    fp = pd.read_csv("reports/phase3/fold_pairs.csv")
    out = {}
    for _, r in fp.iterrows():
        key = (int(r.fold_id), r.pair)
        if key not in ALL_PF:
            continue
        out[key] = {
            "pair": Pair(
                y_sym=r.y_sym,
                x_sym=r.x_sym,
                alpha=float(r.alpha),
                beta=float(r.beta),
                half_life=float(r.half_life_days),
            ),
            "fold": Fold(
                fold_id=int(r.fold_id),
                prior_start=pd.to_datetime(r.train_start).date(),
                prior_end=pd.to_datetime(r.train_start).date(),
                train_start=pd.to_datetime(r.train_start).date(),
                train_end=pd.to_datetime(r.train_end).date(),
                test_start=pd.to_datetime(r.test_start).date(),
                test_end=pd.to_datetime(r.test_end).date(),
            ),
        }
    return out


def _build_cache(pair: Pair, fold: Fold, freq: int) -> dict | None:
    test_start = max(fold.test_start, MPS)
    test_end = min(fold.test_end, MPE)
    aligned = load_minute_pair(pair.y_sym, pair.x_sym, fold.train_start, test_end, root=MINUTE_ROOT)
    if aligned.n_bars == 0:
        return None
    rs = resample_within_session(aligned, freq_minutes=freq)
    if rs.n_bars == 0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        log_y = np.where(rs.tradeable, np.log(rs.close_y), np.nan)
        log_x = np.where(rs.tradeable, np.log(rs.close_x), np.nan)
        spread = log_y - pair.beta * log_x - pair.alpha
    ts = pd.DatetimeIndex(rs.timestamps)
    train_mask = np.asarray((ts.date >= fold.train_start) & (ts.date <= fold.train_end))
    test_mask = np.asarray((ts.date >= test_start) & (ts.date <= test_end))
    mo = settings.signal.ou.min_obs
    if test_mask.sum() < mo or train_mask.sum() < mo:
        return None
    frozen = fit_ou_params(spread[train_mask], freq_minutes=freq, min_obs=mo)
    if not frozen.fit_ok:
        return None
    _, sids = np.unique(rs.session_id, return_inverse=True)
    return {
        "log_y": log_y,
        "log_x": log_x,
        "spread": spread,
        "sids": sids.astype(np.int32),
        "ts": ts,
        "tradeable": np.asarray(rs.tradeable),
        "train_mask": train_mask,
        "test_mask": test_mask,
        "frozen": frozen,
        "beta": pair.beta,
        "alpha": pair.alpha,
        "mu_init": float(frozen.mu),
        "hl_frozen": float(frozen.half_life_minutes),
        "pair": pair,
    }


def _max_holding(pair: Pair, freq: int) -> int:
    window_min = int(
        np.clip(
            round(pair.half_life * NSE_BARS_PER_SESSION),
            NSE_BARS_PER_SESSION,
            5 * NSE_BARS_PER_SESSION,
        )
    )
    max_min = max(
        window_min,
        min(
            settings.signal.max_holding_cap_days * NSE_BARS_PER_SESSION,
            round(
                pair.half_life
                * NSE_BARS_PER_SESSION
                * settings.signal.max_holding_half_life_multiplier
            ),
        ),
    )
    return max(int(round(max_min / freq)), 1)


def _train_residual_fit(cache: dict, h_beta: float, freq: int) -> dict:
    tm = cache["train_mask"]
    res = run_joint_beta_mu(
        cache["log_y"][tm],
        cache["log_x"][tm],
        cache["sids"][tm],
        beta_init=cache["beta"],
        c_init=cache["alpha"] + cache["mu_init"],
        half_life_beta_sessions=h_beta,
        half_life_c_sessions=H_C,
        tradeable=cache["tradeable"][tm],
    )
    if not res.fit_ok:
        return {"ok": False}
    rfit = fit_ou_params(res.residual, freq_minutes=freq, min_obs=settings.signal.ou.min_obs)
    if not rfit.fit_ok:
        return {"ok": False}
    ratio = rfit.half_life_minutes / cache["hl_frozen"]
    return {
        "ok": True,
        "rfit": rfit,
        "sigma_eq_resid": rfit.sigma_eq,
        "guard_ratio": float(ratio),
        "hl_guard": GUARD[0] <= ratio <= GUARD[1],
        "beta_stable": beta_stable_on_train(res),
        "train_residual_var": float(np.var(res.residual[np.isfinite(res.residual)])),
    }


def select_h_beta(cache: dict) -> tuple[float, pd.DataFrame]:
    cb = CostBreakdown(total_spread_bps=3)
    rows = []
    for h in H_BETA_GRID:
        for fold_id, pk in SURVIVORS:
            for freq in FREQS:
                c = cache.get((fold_id, pk, freq))
                if c is None:
                    continue
                tr = _train_residual_fit(c, h, freq)
                row = {
                    "half_life_beta": (np.inf if math.isinf(h) else h),
                    "fold_id": fold_id,
                    "pair": pk,
                    "freq_min": freq,
                    "fit_ok": tr["ok"],
                }
                if tr["ok"]:
                    cost = cb.billed_cost_log_per_pair_round_trip(beta=c["beta"])
                    th = bertram_threshold(tr["rfit"], cost_log_per_round_trip=cost)
                    admissible = tr["hl_guard"] and tr["beta_stable"] and th.fit_ok
                    row.update(
                        {
                            "hl_guard": tr["hl_guard"],
                            "beta_stable": tr["beta_stable"],
                            "guard_ratio": tr["guard_ratio"],
                            "admissible": bool(admissible),
                            "ret_per_unit_time": th.expected_return_per_unit_time
                            if th.fit_ok
                            else np.nan,
                        }
                    )
                else:
                    row.update(
                        {
                            "hl_guard": False,
                            "beta_stable": False,
                            "guard_ratio": np.nan,
                            "admissible": False,
                            "ret_per_unit_time": np.nan,
                        }
                    )
                rows.append(row)
    table = pd.DataFrame(rows)
    summ = []
    for h in H_BETA_GRID:
        hv = np.inf if math.isinf(h) else h
        sub = table[(table.half_life_beta == hv) & table.admissible]
        summ.append(
            (hv, int(sub.shape[0]), float(sub.ret_per_unit_time.mean()) if len(sub) else np.nan)
        )
    summ_df = pd.DataFrame(summ, columns=["half_life_beta", "n_admissible", "mean_crit"])
    adm = summ_df[(summ_df.n_admissible > 0) & summ_df.mean_crit.notna()]
    chosen = (
        float(adm.sort_values("mean_crit", ascending=False).iloc[0].half_life_beta)
        if len(adm)
        else np.inf
    )
    table["chosen_h_beta"] = chosen
    table.to_csv(OUT / "selection_beta.csv", index=False)
    summ_df.to_csv(OUT / "selection_beta_summary.csv", index=False)
    return chosen, table


def run_joint_full(cache: dict, h_beta: float):
    return run_joint_beta_mu(
        cache["log_y"],
        cache["log_x"],
        cache["sids"],
        beta_init=cache["beta"],
        c_init=cache["alpha"] + cache["mu_init"],
        half_life_beta_sessions=h_beta,
        half_life_c_sessions=H_C,
        tradeable=cache["tradeable"],
    )


def collapse_and_drift(cache_all: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep H_beta over the whole grid (β only tracks for FINITE H_β) so the
    collapse + fold-6 drift diagnostics actually exercise β-tracking."""
    coll_rows, drift_rows = [], []
    for fold_id, pk in ALL_PF:
        for freq in FREQS:
            c = cache_all.get((fold_id, pk, freq))
            if c is None:
                continue
            tm = c["test_mask"]
            frozen = c["frozen"]
            spread_test = c["spread"][tm]
            z_frozen = (spread_test - frozen.mu) / frozen.sigma_eq
            muonly = run_local_level_mu(
                c["spread"],
                c["sids"],
                mu_init=c["mu_init"],
                half_life_sessions=H_C,
                tradeable=c["tradeable"],
            )
            # mu-only's OWN sigma_eq (from its train residual) — constant in H_β,
            # so the mu-only drift column is the honest Unit-K baseline (NOT the
            # collapse-inflated β+μ sigma).
            mu_tr = run_local_level_mu(
                c["spread"][c["train_mask"]],
                c["sids"][c["train_mask"]],
                mu_init=c["mu_init"],
                half_life_sessions=H_C,
                tradeable=c["tradeable"][c["train_mask"]],
            )
            mu_fit = fit_ou_params(
                mu_tr.residual, freq_minutes=freq, min_obs=settings.signal.ou.min_obs
            )
            mu_sigeq = mu_fit.sigma_eq if mu_fit.fit_ok else frozen.sigma_eq
            z_mu = (spread_test - muonly.mu_path[tm]) / mu_sigeq
            for h in H_BETA_GRID:
                tr = _train_residual_fit(c, h, freq)
                if not tr["ok"]:
                    continue
                sig_eq = tr["sigma_eq_resid"]
                full = run_joint_full(c, h)
                resid_test = full.residual[tm]
                flag = beta_collapse_flag(full, train_residual_var=tr["train_residual_var"])
                test_sids = np.unique(c["sids"][tm])
                sb_test = full.session_beta[test_sids]
                hv = np.inf if math.isinf(h) else h
                coll_rows.append(
                    {
                        "fold_id": fold_id,
                        "pair": pk,
                        "freq_min": freq,
                        "h_beta": hv,
                        "traded": (fold_id, pk) in SURVIVORS,
                        "beta_init": c["beta"],
                        "beta_test_min": float(np.min(sb_test)),
                        "beta_test_terminal": float(sb_test[-1]),
                        "min_beta_ratio": float(np.min(sb_test) / c["beta"]),
                        "frac_unidentified": flag.get("frac_sessions_unidentified", np.nan),
                        "resid_var_ratio": flag.get("resid_var_ratio", np.nan),
                        "beta_toward_zero": flag.get("beta_toward_zero", False),
                        "resid_var_unstable": flag.get("resid_var_unstable", False),
                        "collapsed": flag.get("collapsed", False),
                        "hl_guard": tr["hl_guard"],
                        "beta_stable": tr["beta_stable"],
                    }
                )
                z_bm = resid_test / sig_eq  # β+μ residual in β+μ-σ units
                fb, fm, fbm = np.isfinite(z_frozen), np.isfinite(z_mu), np.isfinite(z_bm)
                drift_rows.append(
                    {
                        "fold_id": fold_id,
                        "pair": pk,
                        "freq_min": freq,
                        "h_beta": hv,
                        "traded": (fold_id, pk) in SURVIVORS,
                        "drift_frozen_sigeq": float(np.mean(z_frozen[fb])),
                        "drift_muonly_sigeq": float(np.mean(z_mu[fm])),
                        "drift_betamu_sigeq": float(np.mean(z_bm[fbm])),
                        "betamu_inside_1sigma": bool(abs(float(np.mean(z_bm[fbm]))) <= 1.0),
                        "beta_min_ratio": float(
                            np.min(full.session_beta[np.unique(c["sids"][tm])]) / c["beta"]
                        ),
                        "admissible": bool(tr["hl_guard"] and tr["beta_stable"]),
                    }
                )
                # persist β path at the most-aggressive H_β=10 (where tracking is visible)
                if hv == 10.0:
                    pd.DataFrame({"session_idx": test_sids, "beta_s": sb_test}).to_csv(
                        OUT / f"beta_path_{pk.replace('/', '_')}_fold{fold_id}_f{freq}.csv",
                        index=False,
                    )
    return pd.DataFrame(coll_rows), pd.DataFrame(drift_rows)


def eight_cell(cache_all: dict, h_beta: float) -> pd.DataFrame:
    rows = []
    for freq in FREQS:
        for cost in COSTS:
            cb = CostBreakdown(total_spread_bps=cost)
            ps_rows, n_trades = [], 0
            for fold_id, pk in SURVIVORS:
                c = cache_all.get((fold_id, pk, freq))
                if c is None:
                    continue
                tr = _train_residual_fit(c, h_beta, freq)
                if not tr["ok"]:
                    continue
                sig_eq = tr["sigma_eq_resid"]
                cost_log = cb.billed_cost_log_per_pair_round_trip(beta=c["beta"])
                th = bertram_threshold(tr["rfit"], cost_log_per_round_trip=cost_log)
                if not th.fit_ok:
                    continue
                full = run_joint_full(c, h_beta)
                tm = c["test_mask"]
                resid_test = full.residual[tm]
                z = resid_test / sig_eq
                # P&L base is the TRADED spread (market prices with the carried
                # per-session β), NOT the residual — passing the residual would
                # inject spurious μ/β-jump P&L at session boundaries. With β
                # frozen this equals the Unit-K spread X exactly.
                traded_spread = (c["log_y"] - full.beta_path * c["log_x"] - c["alpha"])[tm]
                ts_test = c["ts"][tm]
                _, sids_test = np.unique(c["sids"][tm], return_inverse=True)
                trd_test = c["tradeable"][tm]
                sig = generate_signals_ou(
                    z,
                    sids_test.astype(np.int32),
                    trd_test,
                    regime="B",
                    a_entry_z=th.a_entry_z,
                    stop_mode="none",
                    stop_k_sigma=0.0,
                    max_holding=_max_holding(c["pair"], freq),
                )
                result = run_pair_fold(
                    fold_id=fold_id,
                    pair_key=pk,
                    timestamps=ts_test,
                    session_id=sids_test.astype(np.int32),
                    spread=traded_spread,
                    z=z,
                    signals=sig,
                    cost_log_per_round_trip=cost_log,
                    finalize_fold_boundary=True,
                    pair_beta=c["beta"],
                )
                n_trades += len(result.trades)
                d = pd.DataFrame(
                    {
                        "date": pd.DatetimeIndex(result.timestamps).date,
                        "g": result.gross_log_ret,
                        "n": result.net_log_ret,
                    }
                )
                per = d.groupby("date", as_index=False)[["g", "n"]].sum()
                for _, rr in per.iterrows():
                    ps_rows.append({"date": rr["date"], "g": rr.g, "n": rr.n})
            if not ps_rows:
                continue
            df = (
                pd.DataFrame(ps_rows)
                .groupby("date", as_index=False)
                .agg(g=("g", "mean"), n=("n", "mean"))
            )
            mg, mn = compute_metrics(df.g.to_numpy()), compute_metrics(df.n.to_numpy())
            rows.append(
                {
                    "engine": "beta_mu",
                    "freq_min": freq,
                    "spread_bps": cost,
                    "n_trades": n_trades,
                    "gross_total_pct": mg["total_return_pct"],
                    "net_total_pct": mn["total_return_pct"],
                    "gross_sharpe": mg["sharpe"],
                    "net_sharpe": mn["sharpe"],
                    "net_ann_pct": mn["ann_return_pct"],
                    "net_max_drawdown_pct": mn["max_drawdown_pct"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_beta_mu.csv", index=False)
    return df


def compare_3engine(bm: pd.DataFrame) -> pd.DataFrame:
    km = pd.read_csv("reports/phase3_kalman/metrics_kalman.csv")
    mm = pd.read_csv("reports/phase3_ou/figures/matched_universe/matched_metrics.csv")
    ou = mm[mm.engine == "ou"]
    rows = []
    for freq in FREQS:
        for cost in COSTS:
            b = bm[(bm.freq_min == freq) & (bm.spread_bps == cost)]
            k = km[(km.freq_min == freq) & (km.spread_bps == cost)]
            o = ou[(ou.freq_min == freq) & (ou.spread_bps == cost)]
            rows.append(
                {
                    "freq_min": freq,
                    "cost_bps": cost,
                    "bm_net_total": float(b.net_total_pct.iloc[0]) if len(b) else np.nan,
                    "mu_net_total": float(k.net_total_pct.iloc[0]) if len(k) else np.nan,
                    "ou_net_total": float(o.net_total_pct.iloc[0]) if len(o) else np.nan,
                    "bm_net_sharpe": float(b.net_sharpe.iloc[0]) if len(b) else np.nan,
                    "mu_net_sharpe": float(k.net_sharpe.iloc[0]) if len(k) else np.nan,
                    "ou_net_sharpe": float(o.net_sharpe.iloc[0]) if len(o) else np.nan,
                    "bm_gross_total": float(b.gross_total_pct.iloc[0]) if len(b) else np.nan,
                    "mu_gross_total": float(k.gross_total_pct.iloc[0]) if len(k) else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "compare_3engine.csv", index=False)
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pairs = _load_pairs()
    print(f"[load] {len(pairs)} target pair-folds from fold_pairs.csv")
    cache = {}
    for (fold_id, pk), pf in pairs.items():
        for freq in FREQS:
            c = _build_cache(pf["pair"], pf["fold"], freq)
            if c is not None:
                cache[(fold_id, pk, freq)] = c
    print(f"[cache] built {len(cache)} (pair-fold,freq) caches")

    chosen_h, sel = select_h_beta(cache)
    print(f"[3a] selected global H_beta = {chosen_h} (H_c={H_C} inherited)")

    coll, drift = collapse_and_drift(cache)
    coll.to_csv(OUT / "beta_collapse.csv", index=False)
    drift.to_csv(OUT / "drift_3way.csv", index=False)
    bm = eight_cell(cache, chosen_h)
    cmp = compare_3engine(bm)

    # figures: beta-path overlay + drift comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    for fold_id, pk in ALL_PF:
        f = OUT / f"beta_path_{pk.replace('/', '_')}_fold{fold_id}_f5.csv"
        if f.exists():
            d = pd.read_csv(f)
            ax.plot(
                d.session_idx, d.beta_s, marker=".", ms=3, label=f"fold{fold_id} {pk.split('/')[0]}"
            )
    ax.axhline(0, color="r", lw=0.8, ls=":")
    ax.set_xlabel("test session index")
    ax.set_ylabel("filtered β_s")
    ax.set_title(f"β_s path (β+μ, H_β={chosen_h}, f5) — collapse diagnostic")
    ax.legend(fontsize=8)
    savefig_with_csv(fig, FIG / "beta_paths_f5.png", coll)
    plt.close(fig)

    print("\n=== 3b β-collapse (finite H_β, f5) ===")
    print(coll[(coll.freq_min == 5) & (coll.h_beta != np.inf)].round(3).to_string(index=False))
    print("\n=== 3c/3d fold-6 INDUSINDBK drift vs H_β (f5) — the success metric ===")
    d6 = drift[(drift.fold_id == 6) & (drift.pair == "INDUSINDBK/HDFCBANK") & (drift.freq_min == 5)]
    print(d6.round(3).to_string(index=False))
    any_inside = drift[
        (drift.fold_id == 6)
        & (drift.pair == "INDUSINDBK/HDFCBANK")
        & drift.betamu_inside_1sigma
        & drift.admissible
    ]
    print(
        f"\n[3c] fold-6 INDUSINDBK inside ±1σ at an ADMISSIBLE config? "
        f"{'YES' if len(any_inside) else 'NO'} ({len(any_inside)} configs)"
    )
    print("\n=== 3c compare (β+μ@selected vs μ-only vs frozen, f5) ===")
    print(cmp[cmp.freq_min == 5].round(2).to_string(index=False))
    print("\n=== s3_beta_escalation complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
