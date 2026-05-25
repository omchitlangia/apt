"""Phase 2B risk-managed walk-forward engine — sizing/risk ablation ladder.

Builds on top of :mod:`apt.backtest.walkforward`. Selection + signals are
held identical to Phase 2A; the only thing this engine varies is HOW MUCH
capital each pair gets per trade and WHEN to pull a pair entirely. The
rung parameter switches behaviour:

  R0  fixed equal-notional   (must reproduce :mod:`apt.backtest.walkforward`)
  R1  + §3.2 1%-risk sizing
  R2  + §3.3 per-pair cap
  R3  + §3.4 cluster cap + §3.5 shared-leg de-dup
  R4  + §3.7 pair-kill switch (relationship-breakdown primary; loose
            consecutive-stops backstop at K=4)

Loop is day-by-day rather than pair-by-pair because sizing decisions cross
pairs (cluster cap, total cap, shared-leg). On each day:
  1. mark-to-market open positions (today's spread move × held weight);
  2. process signal-driven exits (deduct cost on the exit day, reset state);
  3. (R4) periodic kill check; force-close any open trade on a killed pair;
  4. process new entries (sizing → caps → record state);
  5. pro-rata scale-down for any cluster over the cap (defensive);
  6. record portfolio gross/net daily log return + per-cluster exposure.

Hand-off at fold boundary: all open positions are force-closed mark-to-market.
The same pair may be re-selected in the next fold and re-enter normally.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl
from loguru import logger

from apt.backtest.kill_switch import KillMode, evaluate_kill
from apt.backtest.sizing import (
    apply_cluster_scaledown,
    apply_per_pair_cap,
    cluster_usage,
    compute_risk_based_weight,
    has_shared_leg,
)
from apt.backtest.walkforward import Fold, Pair
from apt.signals.spread import compute_spread, generate_signals, rolling_zscore

# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConfig:
    """All knobs for one ladder rung + sweep arm."""

    rung: int  # 0..4
    risk_frac: float = 0.01
    per_pair_cap: float = 0.12
    cluster_cap: float = 0.05
    total_cap: float = 0.10
    gross_cap: float = 2.0
    kill_mode: KillMode = "relationship"  # 'none' | 'loss_only' | 'relationship'
    kill_K: int = 4
    kill_cap: float = 0.04
    kill_check_interval_days: int = 21
    kill_relationship_window_days: int = 252
    kill_relationship_adf_alpha: float = 0.05
    kill_relationship_halflife_max_days: int = 60
    kill_relationship_vol_ratio_max: float = 2.0


@dataclass(frozen=True)
class RMTrade:
    """Risk-managed trade record — extends the Phase 2A Trade with sizing info."""

    fold_id: int
    pair_key: str
    direction: int
    entry_date: date
    exit_date: date
    entry_z: float
    exit_z: float
    entry_sigma: float
    weight: float  # set at entry, may be scaled later
    days_held: int
    gross_log_pnl: float  # in capital-fraction units (= weight × spread move × direction)
    cost_log: float  # in capital-fraction units (= weight × 2·bps/10000)
    net_log_pnl: float
    exit_reason: str  # mean_revert | stop | time | fold_boundary | kill_*


@dataclass(frozen=True)
class KillEvent:
    fold_id: int
    pair_key: str
    date: date
    reason: str
    detail: dict


@dataclass(frozen=True)
class PrematureCut:
    fold_id: int
    pair_key: str
    cut_date: date
    cut_reason: str  # the original cut: 'stop' | 'kill_relationship' | 'kill_loss_backstop' | 'kill_consecutive_stops'
    direction: int
    cut_spread: float
    cut_z: float
    horizon_days: int
    reverted_within_horizon: bool
    foregone_gross_log_pnl: float  # >0 if we missed gains by exiting too early


@dataclass(frozen=True)
class RiskManagedResult:
    config: RiskConfig
    folds: list[Fold]
    trades: list[RMTrade]
    kill_events: list[KillEvent]
    premature_cuts: list[PrematureCut]
    portfolio_daily: (
        pl.DataFrame
    )  # date, fold_id, gross_log_ret, net_log_ret, n_open, gross_exposure
    per_pair_daily: dict[str, pl.DataFrame]  # pair → daily series
    cluster_exposure_daily: pl.DataFrame  # date, fold_id, sector, exposure
    funnel: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-fold pre-compute (spread + z + signals once, used by the day loop)
# ---------------------------------------------------------------------------


def _precompute_pair_in_fold(
    *,
    pair: Pair,
    fold: Fold,
    daily_get_prices_fn: Callable,
    trading_days: list[date],
    rolling_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_holding_cap_days: int,
    max_holding_half_life_multiplier: float,
) -> dict | None:
    """Compute spread / z / rolling_sigma / signals for one pair over its fold.

    Returns ``None`` if the pair has no aligned prices in the test slice.
    """
    day_to_idx = {d: i for i, d in enumerate(trading_days)}
    test_start_idx = day_to_idx[fold.test_start]
    warmup_start_idx = max(0, test_start_idx - rolling_window + 1)
    warmup_start = trading_days[warmup_start_idx]

    dates_full, p_y, p_x = daily_get_prices_fn(pair.y_sym, pair.x_sym, warmup_start, fold.test_end)
    if len(dates_full) == 0:
        return None

    spread_full = compute_spread(p_y, p_x, beta=pair.beta, intercept=pair.alpha)
    s_series = pl.Series("s", spread_full)
    roll_sigma = s_series.rolling_std(
        window_size=rolling_window, min_samples=rolling_window
    ).to_numpy()
    z_full = rolling_zscore(spread_full, window=rolling_window)

    try:
        test_first = next(i for i, d in enumerate(dates_full) if d >= fold.test_start)
    except StopIteration:
        return None

    z_test = z_full[test_first:]
    sigma_test = roll_sigma[test_first:]
    spread_test = spread_full[test_first:]
    dates_test = dates_full[test_first:]

    import math

    max_holding = max(
        2,
        min(
            max_holding_cap_days,
            int(math.ceil(pair.half_life * max_holding_half_life_multiplier)),
        ),
    )
    sig = generate_signals(z_test, entry=entry_z, exit=exit_z, stop=stop_z, max_holding=max_holding)

    # Also keep enough spread HISTORY (including warmup) for relationship checks.
    return {
        "pair": pair,
        "dates_test": dates_test,
        "spread_test": spread_test,
        "z_test": z_test,
        "sigma_test": sigma_test,
        "signal_position": sig.position,
        "signal_exit_reason": sig.exit_reason,
        "signal_days_held": sig.days_in_trade,
        "max_holding": max_holding,
        # For the relationship test: history including warmup
        "spread_full": spread_full,
        "dates_full": dates_full,
    }


# ---------------------------------------------------------------------------
# Premature-cut diagnostic
# ---------------------------------------------------------------------------


def _check_premature_cut(
    *,
    pd_pair: dict,
    cut_idx_in_test: int,
    direction: int,
    horizon_days: int,
    exit_z: float,
) -> tuple[bool, float]:
    """Did the spread mean-revert (cross the |z| ≤ exit_z band) within
    ``horizon_days`` after the cut?

    Returns ``(reverted, foregone_gross_log_pnl)``. ``foregone`` is the gross
    spread P&L the trade would have captured had it been held until the
    mean-revert exit, in log units (>0 if we missed gains, <0 if exiting was
    correct on a divergence we avoided).
    """
    z_test = pd_pair["z_test"]
    spread_test = pd_pair["spread_test"]
    n = len(z_test)
    end = min(n, cut_idx_in_test + 1 + horizon_days)
    horizon_z = z_test[cut_idx_in_test + 1 : end]
    horizon_sp = spread_test[cut_idx_in_test + 1 : end]
    if horizon_z.size == 0:
        return False, 0.0
    # Mean-revert: long → z[t] >= -exit_z; short → z[t] <= +exit_z
    rev_mask = horizon_z >= -exit_z if direction > 0 else horizon_z <= +exit_z
    if not rev_mask.any():
        return False, 0.0
    rev_idx = int(np.argmax(rev_mask))
    foregone = direction * float(horizon_sp[rev_idx] - spread_test[cut_idx_in_test])
    return True, foregone


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def run_walkforward_risk_managed(
    folds: list[Fold],
    trading_days: Sequence[date],
    *,
    select_pairs_fn: Callable | None = None,
    get_prices_fn: Callable,
    pre_selected_pairs: dict[int, list[Pair]] | None = None,
    rolling_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_cap_days: int = 60,
    max_holding_half_life_multiplier: float = 3.0,
    cost_bps_per_leg: float = 25.0,
    config: RiskConfig,
) -> RiskManagedResult:
    """Run the risk-managed walk-forward at the given rung/config.

    ``pre_selected_pairs`` short-circuits the (expensive) cointegration
    selection — pass the cached per-fold pair list to reuse across all
    rungs. If ``None``, ``select_pairs_fn`` is called per fold.
    """
    if pre_selected_pairs is None and select_pairs_fn is None:
        raise ValueError("must supply either pre_selected_pairs or select_pairs_fn")
    if cost_bps_per_leg < 0:
        raise ValueError(f"cost_bps_per_leg must be >= 0, got {cost_bps_per_leg}")

    trading_days_list = list(trading_days)
    cost_log_per_round_trip = 2.0 * cost_bps_per_leg / 10_000.0

    all_trades: list[RMTrade] = []
    kill_events: list[KillEvent] = []
    premature_cuts: list[PrematureCut] = []
    portfolio_rows: list[dict] = []
    cluster_rows: list[dict] = []
    per_pair_daily: dict[str, list[dict]] = defaultdict(list)

    funnel = {
        "rung": config.rung,
        "kill_mode": config.kill_mode,
        "cluster_cap": config.cluster_cap,
        "n_folds": len(folds),
        "n_selected_pair_units": 0,
        "n_trades": 0,
        "n_entries_skipped_cluster": 0,
        "n_entries_skipped_shared_leg": 0,
        "n_entries_skipped_total_cap": 0,
        "n_kill_events": 0,
        "n_force_closed_kill": 0,
        "n_force_closed_fold_boundary": 0,
    }

    for fold in folds:
        if pre_selected_pairs is not None:
            pairs_in_fold = list(pre_selected_pairs.get(fold.fold_id, []))
        else:
            pairs_in_fold = list(
                select_pairs_fn(fold.prior_start, fold.prior_end, fold.train_start, fold.train_end)
            )
        funnel["n_selected_pair_units"] += len(pairs_in_fold)
        if not pairs_in_fold:
            continue

        # Pre-compute every pair's signal/spread/sigma series for this fold
        pair_data: dict[str, dict] = {}
        for pair in pairs_in_fold:
            pd_p = _precompute_pair_in_fold(
                pair=pair,
                fold=fold,
                daily_get_prices_fn=get_prices_fn,
                trading_days=trading_days_list,
                rolling_window=rolling_window,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=stop_z,
                max_holding_cap_days=max_holding_cap_days,
                max_holding_half_life_multiplier=max_holding_half_life_multiplier,
            )
            if pd_p is not None:
                pair_data[pair.key] = pd_p

        sector_map: dict[str, str] = {p.key: (p.sector or "OTHER") for p in pairs_in_fold}
        n_pairs_fold = len(pair_data)
        equal_w = 1.0 / max(1, n_pairs_fold)

        state: dict[str, dict] = {
            pkey: {
                "w": 0.0,
                "direction": 0,
                "entry_date": None,
                "entry_idx": -1,
                "entry_z": 0.0,
                "entry_sigma": 0.0,
                "entry_spread": 0.0,
                "is_killed": False,
                "consecutive_stops": 0,
                "cum_pair_loss": 0.0,
            }
            for pkey in pair_data
        }

        # Index per-pair test dates for O(1) lookup
        pair_date_idx: dict[str, dict[date, int]] = {
            pkey: {d: i for i, d in enumerate(pd_p["dates_test"])}
            for pkey, pd_p in pair_data.items()
        }

        # Canonical test calendar for the fold
        day_to_global_idx = {d: i for i, d in enumerate(trading_days_list)}
        ts_idx = day_to_global_idx[fold.test_start]
        te_idx = day_to_global_idx[fold.test_end]
        test_dates = trading_days_list[ts_idx : te_idx + 1]

        for day_idx, today in enumerate(test_dates):
            # 1. mark-to-market open positions (gross daily contribution per pair)
            daily_pair_gross: dict[str, float] = {}
            daily_pair_net: dict[str, float] = {}
            for pkey, st in state.items():
                if st["w"] == 0:
                    daily_pair_gross[pkey] = 0.0
                    daily_pair_net[pkey] = 0.0
                    continue
                i = pair_date_idx[pkey].get(today)
                if i is None or i == 0:
                    daily_pair_gross[pkey] = 0.0
                    daily_pair_net[pkey] = 0.0
                    continue
                delta = pair_data[pkey]["spread_test"][i] - pair_data[pkey]["spread_test"][i - 1]
                contrib = st["w"] * st["direction"] * float(delta)
                daily_pair_gross[pkey] = contrib
                daily_pair_net[pkey] = contrib

            # 2. signal-driven exits
            for pkey, st in state.items():
                if st["w"] == 0:
                    continue
                i = pair_date_idx[pkey].get(today)
                if i is None:
                    continue
                exit_reason = pair_data[pkey]["signal_exit_reason"][i]
                if exit_reason is None:
                    continue
                _close_position(
                    pkey,
                    st,
                    pair_data[pkey],
                    i,
                    today,
                    exit_reason,
                    fold,
                    cost_log_per_round_trip,
                    daily_pair_net,
                    all_trades,
                    premature_cuts,
                    exit_z,
                )

            # 3. (R4) periodic kill check — fold-boundary forces happen at step 6
            if config.rung >= 4 and day_idx > 0 and day_idx % config.kill_check_interval_days == 0:
                for pkey, st in state.items():
                    if st["is_killed"]:
                        continue
                    i = pair_date_idx[pkey].get(today)
                    if i is None:
                        continue
                    # spread history up to and including today, in original order
                    # (warmup + test, sliced to "as of today")
                    pd_p = pair_data[pkey]
                    today_full_idx = next(
                        (k for k, d in enumerate(pd_p["dates_full"]) if d == today), None
                    )
                    if today_full_idx is None:
                        continue
                    spread_history = pd_p["spread_full"][: today_full_idx + 1]
                    verdict = evaluate_kill(
                        mode=config.kill_mode,
                        spread_history=spread_history,
                        consecutive_stops=st["consecutive_stops"],
                        cum_pair_loss=st["cum_pair_loss"],
                        window=config.kill_relationship_window_days,
                        adf_alpha=config.kill_relationship_adf_alpha,
                        halflife_max_days=config.kill_relationship_halflife_max_days,
                        vol_ratio_max=config.kill_relationship_vol_ratio_max,
                        kill_K=config.kill_K,
                        kill_cap=config.kill_cap,
                    )
                    if not verdict.killed:
                        continue
                    funnel["n_kill_events"] += 1
                    kill_events.append(
                        KillEvent(
                            fold_id=fold.fold_id,
                            pair_key=pkey,
                            date=today,
                            reason=f"kill_{verdict.reason}",
                            detail=verdict.detail,
                        )
                    )
                    if st["w"] != 0:
                        funnel["n_force_closed_kill"] += 1
                        _close_position(
                            pkey,
                            st,
                            pair_data[pkey],
                            i,
                            today,
                            f"kill_{verdict.reason}",
                            fold,
                            cost_log_per_round_trip,
                            daily_pair_net,
                            all_trades,
                            premature_cuts,
                            exit_z,
                        )
                    st["is_killed"] = True

            # 4. new entries
            for pkey, st in state.items():
                if st["w"] != 0 or st["is_killed"]:
                    continue
                i = pair_date_idx[pkey].get(today)
                if i is None:
                    continue
                cur_pos = int(pair_data[pkey]["signal_position"][i])
                prev_pos = int(pair_data[pkey]["signal_position"][i - 1]) if i > 0 else 0
                if not (prev_pos == 0 and cur_pos != 0):
                    continue
                z_entry = float(pair_data[pkey]["z_test"][i])
                sigma_entry = float(pair_data[pkey]["sigma_test"][i])

                # Sizing
                if config.rung == 0:
                    proposed_w = equal_w
                else:
                    proposed_w = compute_risk_based_weight(
                        z_entry=z_entry,
                        z_stop=stop_z,
                        sigma_at_entry=sigma_entry,
                        risk_frac=config.risk_frac,
                    )
                if config.rung >= 2:
                    proposed_w = apply_per_pair_cap(proposed_w, config.per_pair_cap)

                # Caps (R3+) — shared-leg is a HARD skip; cluster/total caps CLIP
                # the new entry to remaining headroom (and only skip if room is 0).
                if config.rung >= 3:
                    open_w = {k: state[k]["w"] for k in state if state[k]["w"] != 0}
                    # Shared-leg de-dup (hard skip — §3.5)
                    if has_shared_leg(pkey, list(open_w.keys())):
                        funnel["n_entries_skipped_shared_leg"] += 1
                        continue
                    # Cluster cap (clip to remaining room)
                    if config.cluster_cap is not None and config.cluster_cap > 0:
                        usage = cluster_usage(open_w, sector_map)
                        sec = sector_map.get(pkey, "OTHER")
                        cluster_room = max(0.0, config.cluster_cap - usage.get(sec, 0.0))
                        if proposed_w > cluster_room:
                            funnel["n_entries_clipped_cluster"] = (
                                funnel.get("n_entries_clipped_cluster", 0) + 1
                            )
                            proposed_w = cluster_room
                    # Total open-risk cap (clip to remaining room)
                    total_used = sum(open_w.values())
                    total_room = max(0.0, config.total_cap - total_used)
                    if proposed_w > total_room:
                        funnel["n_entries_clipped_total"] = (
                            funnel.get("n_entries_clipped_total", 0) + 1
                        )
                        proposed_w = total_room
                    if proposed_w <= 1e-9:
                        funnel["n_entries_skipped_cluster"] += 1
                        continue

                if proposed_w <= 0:
                    continue

                st["w"] = proposed_w
                st["direction"] = cur_pos
                st["entry_date"] = today
                st["entry_idx"] = i
                st["entry_z"] = z_entry
                st["entry_sigma"] = sigma_entry
                st["entry_spread"] = float(pair_data[pkey]["spread_test"][i])

            # 5. defensive cluster scale-down
            if config.rung >= 3 and config.cluster_cap is not None and config.cluster_cap > 0:
                open_w = {k: state[k]["w"] for k in state if state[k]["w"] != 0}
                if open_w:
                    apply_cluster_scaledown(open_w, sector_map, config.cluster_cap)
                    for k, new_w in open_w.items():
                        state[k]["w"] = new_w

            # 6. fold-boundary force-close on the last test day
            if day_idx == len(test_dates) - 1:
                for pkey, st in state.items():
                    if st["w"] == 0:
                        continue
                    i = pair_date_idx[pkey].get(today)
                    if i is None:
                        continue
                    funnel["n_force_closed_fold_boundary"] += 1
                    _close_position(
                        pkey,
                        st,
                        pair_data[pkey],
                        i,
                        today,
                        "fold_boundary",
                        fold,
                        cost_log_per_round_trip,
                        daily_pair_net,
                        all_trades,
                        premature_cuts,
                        exit_z,
                    )

            # Portfolio aggregates
            portfolio_gross = float(sum(daily_pair_gross.values()))
            portfolio_net = float(sum(daily_pair_net.values()))
            open_w_today = {k: state[k]["w"] for k in state if state[k]["w"] != 0}
            n_open = len(open_w_today)
            gross_exposure = float(sum(open_w_today.values()))
            portfolio_rows.append(
                {
                    "date": today,
                    "fold_id": fold.fold_id,
                    "gross_log_ret": portfolio_gross,
                    "net_log_ret": portfolio_net,
                    "n_open": n_open,
                    "gross_exposure": gross_exposure,
                }
            )
            # Per-cluster exposure snapshot
            usage_today = cluster_usage(open_w_today, sector_map)
            for sec, used in usage_today.items():
                cluster_rows.append(
                    {
                        "date": today,
                        "fold_id": fold.fold_id,
                        "sector": sec,
                        "exposure": float(used),
                    }
                )
            # Per-pair daily record
            for pkey in pair_data:
                per_pair_daily[pkey].append(
                    {
                        "date": today,
                        "fold_id": fold.fold_id,
                        "gross_log_ret": float(daily_pair_gross.get(pkey, 0.0)),
                        "net_log_ret": float(daily_pair_net.get(pkey, 0.0)),
                        "weight": float(state[pkey]["w"]),
                        "direction": int(state[pkey]["direction"]),
                    }
                )

    funnel["n_trades"] = len(all_trades)

    portfolio_df = pl.DataFrame(portfolio_rows).sort("date") if portfolio_rows else pl.DataFrame()
    per_pair_df = {k: pl.DataFrame(v).sort("date") for k, v in per_pair_daily.items() if v}
    cluster_df = pl.DataFrame(cluster_rows).sort("date") if cluster_rows else pl.DataFrame()

    logger.info(
        "Rung R{} ({}): {} trades, {} kill events, {} cluster-cap skips, {} shared-leg skips",
        config.rung,
        config.kill_mode,
        funnel["n_trades"],
        funnel["n_kill_events"],
        funnel["n_entries_skipped_cluster"],
        funnel["n_entries_skipped_shared_leg"],
    )

    return RiskManagedResult(
        config=config,
        folds=folds,
        trades=all_trades,
        kill_events=kill_events,
        premature_cuts=premature_cuts,
        portfolio_daily=portfolio_df,
        per_pair_daily=per_pair_df,
        cluster_exposure_daily=cluster_df,
        funnel=funnel,
    )


def _close_position(
    pkey: str,
    st: dict,
    pd_pair: dict,
    today_idx: int,
    today: date,
    reason: str,
    fold: Fold,
    cost_log_per_round_trip: float,
    daily_pair_net: dict[str, float],
    all_trades: list[RMTrade],
    premature_cuts: list[PrematureCut],
    exit_z: float,
) -> None:
    """Close ``pkey``'s open position; deduct cost from today's net; record
    Trade and (if applicable) a premature-cut diagnostic."""
    direction = st["direction"]
    weight = st["w"]
    cost = weight * cost_log_per_round_trip
    daily_pair_net[pkey] = daily_pair_net.get(pkey, 0.0) - cost

    entry_idx = st["entry_idx"]
    gross_log = direction * (pd_pair["spread_test"][today_idx] - pd_pair["spread_test"][entry_idx])
    gross_weighted = weight * float(gross_log)
    net_weighted = gross_weighted - cost
    st["cum_pair_loss"] += float(net_weighted)

    # consecutive stops bookkeeping for kill backstop
    if reason == "stop":
        st["consecutive_stops"] += 1
    else:
        st["consecutive_stops"] = 0

    duration = int(pd_pair["signal_days_held"][today_idx - 1]) if today_idx > 0 else 0

    all_trades.append(
        RMTrade(
            fold_id=fold.fold_id,
            pair_key=pkey,
            direction=direction,
            entry_date=st["entry_date"],
            exit_date=today,
            entry_z=st["entry_z"],
            exit_z=float(pd_pair["z_test"][today_idx]),
            entry_sigma=st["entry_sigma"],
            weight=weight,
            days_held=duration,
            gross_log_pnl=float(gross_weighted),
            cost_log=float(cost),
            net_log_pnl=float(net_weighted),
            exit_reason=reason,
        )
    )

    # Premature-cut diagnostic — only for "cut" exits (stop or any kill_*)
    if reason == "stop" or reason.startswith("kill_"):
        horizon = pd_pair["max_holding"]
        reverted, foregone = _check_premature_cut(
            pd_pair=pd_pair,
            cut_idx_in_test=today_idx,
            direction=direction,
            horizon_days=horizon,
            exit_z=exit_z,
        )
        premature_cuts.append(
            PrematureCut(
                fold_id=fold.fold_id,
                pair_key=pkey,
                cut_date=today,
                cut_reason=reason,
                direction=direction,
                cut_spread=float(pd_pair["spread_test"][today_idx]),
                cut_z=float(pd_pair["z_test"][today_idx]),
                horizon_days=horizon,
                reverted_within_horizon=bool(reverted),
                foregone_gross_log_pnl=float(foregone),
            )
        )

    # Reset state for re-entry within the same fold (kill stays sticky;
    # signal-driven exits unset only the trade fields).
    st["w"] = 0.0
    st["direction"] = 0
    st["entry_date"] = None
    st["entry_idx"] = -1
    st["entry_z"] = 0.0
    st["entry_sigma"] = 0.0
    st["entry_spread"] = 0.0
