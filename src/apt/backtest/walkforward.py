"""Walk-forward, leakage-free, cost-realistic backtest engine.

Phase 2A. Asset-agnostic — operates through two callbacks (pair selection
and price retrieval) and a calendar of trading days. Knows nothing about
equities, polars schemas, or any specific data source. Drop-in replacement
of those two callbacks ports the engine to crypto / US futures / FX.

Fold layout (rolled forward annually by default):

    [ prior 1008d (stability) | train 1008d (selection + OLS) | test 252d (P&L) ]
                                                              ^
                                                              z-warmup
                                                              (the last
                                                              rolling_window
                                                              bars of train
                                                              are *only*
                                                              used to seed
                                                              the trailing z;
                                                              no trading there)

Leakage-free guarantees (CI-tested):
  * Pair selection (``select_pairs_fn``) sees ONLY ``prior`` + ``train`` data.
    The engine never passes test-window dates into it.
  * Hedge ratio (``Pair.beta``, ``Pair.alpha``) is fit on TRAIN and FROZEN.
    :func:`apt.signals.spread.compute_spread` applies it; never re-fits.
  * Rolling z-score is causal (trailing window only) and re-uses the
    end-of-train data for the warmup at the start of test — that data was
    already known at the test bar, so this is NOT leakage.
  * Signal state at test start is FLAT — warmup bars don't produce trades.

Cost model: ``cost_bps_per_leg`` is the per-LEG per-round-trip cost
(brokerage + STT + slippage). Under the **(1+β) billing convention**
(see ``apt.intraday.costs``), the total log-cost deducted per pair
round-trip is ``(1 + pair.beta) × cost_bps_per_leg / 10000``, where
β is the pair's frozen hedge ratio from the train-window EG fit. β=1
reproduces the legacy 2× equal-notional value exactly. Cost is
deducted on the EXIT day of each trade (entry day is cost-free).

Open-trade handoff at a fold boundary: any open position at ``test_end`` is
force-closed at the close of ``test_end`` (mark-to-market). The same pair
may be re-selected and re-entered in the next fold; the engine does NOT
carry positions across folds (positions reset to flat at each fold start).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl
from loguru import logger

from apt.signals.spread import (
    compute_spread,
    generate_signals,
    rolling_zscore,
)

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """Asset-agnostic pair specification. Carries the train-window OLS fit."""

    y_sym: str
    x_sym: str
    alpha: float
    beta: float
    half_life: float
    sector: str | None = None
    is_structural: bool = False

    @property
    def key(self) -> str:
        return f"{self.y_sym}/{self.x_sym}"


@dataclass(frozen=True)
class Fold:
    fold_id: int
    prior_start: date
    prior_end: date
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class Trade:
    """One round-trip in one fold for one pair.

    ``cost_log`` is the **billed** round-trip cost
    (``(1 + pair_beta) × cost_log_per_leg`` under the v2-cost-beta
    schema). ``pair_beta`` records the β used to bill so a re-stamp can
    re-derive net P&L without re-running.
    """

    fold_id: int
    pair_key: str
    direction: int  # +1 long spread / -1 short spread
    entry_date: date
    exit_date: date
    entry_z: float
    exit_z: float
    days_held: int
    gross_log_pnl: float
    cost_log: float
    net_log_pnl: float
    exit_reason: str  # mean_revert | stop | time | fold_boundary
    pair_beta: float = float("nan")


# Callback signatures (documented here; not enforced as runtime types).
#
# select_pairs_fn(prior_start, prior_end, train_start, train_end) -> list[Pair]
# get_prices_fn(y_sym, x_sym, start, end) -> tuple[list[date], np.ndarray, np.ndarray]
SelectPairsFn = Callable[[date, date, date, date], Iterable[Pair]]
GetPricesFn = Callable[[str, str, date, date], tuple[list[date], np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[Fold]
    selected_pairs_per_fold: dict[int, list[Pair]]
    trades: list[Trade]
    portfolio_daily: pl.DataFrame
    per_pair_daily: dict[str, pl.DataFrame] = field(default_factory=dict)
    funnel: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------


def build_folds(
    trading_days: Sequence[date],
    *,
    prior_days: int,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[Fold]:
    """Build the list of walk-forward folds from a trading-day calendar.

    The earliest fold's prior window starts at ``trading_days[0]`` (or later
    if there isn't enough history). Folds step forward by ``step_days``
    (defaults to ``test_days`` ⇒ non-overlapping test windows). The latest
    fold's test window ends at or before ``trading_days[-1]``.

    Strict ordering: ``prior_start < prior_end < train_start <= train_end
    < test_start <= test_end``. The train and test windows are CONTIGUOUS
    (``train_end`` is the trading day immediately before ``test_start``) so
    no information is wasted in a gap.
    """
    if prior_days < 1 or train_days < 1 or test_days < 1:
        raise ValueError("prior_days, train_days, test_days must each be >= 1")
    if step_days is None:
        step_days = test_days
    if step_days < 1:
        raise ValueError(f"step_days must be >= 1, got {step_days}")

    days = list(trading_days)
    n = len(days)
    if n < prior_days + train_days + test_days:
        raise ValueError(f"need >= {prior_days + train_days + test_days} trading days, got {n}")

    first_test_start_idx = prior_days + train_days  # 0-indexed
    folds: list[Fold] = []
    fid = 0
    test_start_idx = first_test_start_idx
    while test_start_idx + test_days - 1 < n:
        test_end_idx = test_start_idx + test_days - 1
        train_end_idx = test_start_idx - 1
        train_start_idx = train_end_idx - train_days + 1
        prior_end_idx = train_start_idx - 1
        prior_start_idx = prior_end_idx - prior_days + 1
        if prior_start_idx < 0:
            break
        folds.append(
            Fold(
                fold_id=fid,
                prior_start=days[prior_start_idx],
                prior_end=days[prior_end_idx],
                train_start=days[train_start_idx],
                train_end=days[train_end_idx],
                test_start=days[test_start_idx],
                test_end=days[test_end_idx],
            )
        )
        fid += 1
        test_start_idx += step_days
    return folds


# ---------------------------------------------------------------------------
# Per-pair-per-fold trade extraction
# ---------------------------------------------------------------------------


def _identify_trades_and_returns(
    *,
    fold_id: int,
    pair_key: str,
    dates_test: list[date],
    spread_test: np.ndarray,
    z_test: np.ndarray,
    position: np.ndarray,
    exit_reason: list,
    days_in_trade: np.ndarray,
    cost_log_per_round_trip: float,
    pair_beta: float = float("nan"),
) -> tuple[list[Trade], np.ndarray, np.ndarray]:
    """Walk through one pair's test slice; extract trades + daily returns.

    Daily strategy log return at index t (t >= 1):
        r[t] = position[t-1] * (spread[t] - spread[t-1])
    r[0] is 0 (no carried position at the start of test).

    Net daily return equals gross daily return except on exit days, where
    the full round-trip cost is deducted.

    An open position at the end of the slice is force-closed (mark-to-market)
    and emitted as a Trade with ``exit_reason='fold_boundary'``.
    """
    n_test = len(dates_test)
    gross_daily = np.zeros(n_test, dtype=float)
    for i in range(1, n_test):
        gross_daily[i] = float(position[i - 1]) * (spread_test[i] - spread_test[i - 1])
    net_daily = gross_daily.copy()

    trades: list[Trade] = []
    in_trade_entry: int | None = None
    in_trade_direction: int = 0
    for i in range(n_test):
        prev_pos = 0 if i == 0 else int(position[i - 1])
        cur_pos = int(position[i])
        # Entry detection
        if prev_pos == 0 and cur_pos != 0:
            in_trade_entry = i
            in_trade_direction = cur_pos
        # Signal-driven exit
        if exit_reason[i] is not None and in_trade_entry is not None:
            entry_idx = in_trade_entry
            exit_idx = i
            direction = in_trade_direction
            gross = direction * (spread_test[exit_idx] - spread_test[entry_idx])
            cost = cost_log_per_round_trip
            net = gross - cost
            net_daily[exit_idx] -= cost
            # days_in_trade at the bar BEFORE exit is the actual holding count
            duration = int(days_in_trade[exit_idx - 1]) if exit_idx > 0 else 0
            trades.append(
                Trade(
                    fold_id=fold_id,
                    pair_key=pair_key,
                    direction=direction,
                    entry_date=dates_test[entry_idx],
                    exit_date=dates_test[exit_idx],
                    entry_z=float(z_test[entry_idx]),
                    exit_z=float(z_test[exit_idx]),
                    days_held=duration,
                    gross_log_pnl=float(gross),
                    cost_log=float(cost),
                    net_log_pnl=float(net),
                    exit_reason=str(exit_reason[i]),
                    pair_beta=float(pair_beta),
                )
            )
            in_trade_entry = None
            in_trade_direction = 0

    # Force-close any open trade at the fold boundary
    if in_trade_entry is not None:
        entry_idx = in_trade_entry
        exit_idx = n_test - 1
        direction = in_trade_direction
        gross = direction * (spread_test[exit_idx] - spread_test[entry_idx])
        cost = cost_log_per_round_trip
        net = gross - cost
        net_daily[exit_idx] -= cost
        duration = int(days_in_trade[exit_idx])  # last bar's running count
        trades.append(
            Trade(
                fold_id=fold_id,
                pair_key=pair_key,
                direction=direction,
                entry_date=dates_test[entry_idx],
                exit_date=dates_test[exit_idx],
                entry_z=float(z_test[entry_idx]),
                exit_z=float(z_test[exit_idx]),
                days_held=duration,
                gross_log_pnl=float(gross),
                cost_log=float(cost),
                net_log_pnl=float(net),
                exit_reason="fold_boundary",
                pair_beta=float(pair_beta),
            )
        )

    return trades, gross_daily, net_daily


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    daily_log_returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: int = 252,
) -> dict:
    """Standard portfolio metrics on a daily log-return series.

    Returns
    -------
    dict with:
      n_obs, n_years, total_return_pct, ann_return_pct, ann_vol_pct,
      sharpe, max_drawdown_pct
    """
    arr = np.asarray(daily_log_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out = {
        "n_obs": n,
        "n_years": 0.0,
        "total_return_pct": 0.0,
        "ann_return_pct": 0.0,
        "ann_vol_pct": 0.0,
        "sharpe": float("nan"),
        "max_drawdown_pct": 0.0,
    }
    if n == 0:
        return out

    cum_log = np.cumsum(arr)
    total_log = float(cum_log[-1])
    n_years = n / periods_per_year
    ann_log = total_log / max(n_years, 1.0 / periods_per_year)
    std_daily = float(arr.std(ddof=1)) if n > 1 else 0.0
    sharpe = (
        float(arr.mean() / std_daily * math.sqrt(periods_per_year))
        if std_daily > 0
        else float("nan")
    )
    running_max = np.maximum.accumulate(cum_log)
    dd_log = cum_log - running_max
    max_dd_log = float(dd_log.min()) if dd_log.size else 0.0

    out.update(
        {
            "n_years": float(n_years),
            "total_return_pct": float(math.expm1(total_log) * 100),
            "ann_return_pct": float(math.expm1(ann_log) * 100),
            "ann_vol_pct": float(std_daily * math.sqrt(periods_per_year) * 100),
            "sharpe": sharpe,
            "max_drawdown_pct": float(math.expm1(max_dd_log) * 100),
        }
    )
    return out


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def run_walkforward(
    folds: list[Fold],
    trading_days: Sequence[date],
    *,
    select_pairs_fn: SelectPairsFn,
    get_prices_fn: GetPricesFn,
    rolling_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_holding_cap_days: int = 60,
    max_holding_half_life_multiplier: float = 3.0,
    cost_bps_per_leg: float = 25.0,
) -> WalkForwardResult:
    """Run the walk-forward backtest end-to-end.

    For each fold:
      1. Call ``select_pairs_fn`` on (prior, train) windows — selection sees
         NO test-window data.
      2. For each tradeable pair, fetch aligned prices for
         ``[warmup_start, test_end]`` (warmup = the trailing
         ``rolling_window-1`` train days needed to seed the causal z).
      3. ``compute_spread`` with the pair's frozen (alpha, beta).
      4. ``rolling_zscore`` (causal). Slice to test only.
      5. ``generate_signals`` (causal) on the test slice with per-pair
         ``max_holding = min(cap, ⌈mult · half_life⌉)``.
      6. Extract trades + daily gross/net log returns; force-close open
         trades at the fold boundary.

    Portfolio daily return = arithmetic mean across the fold's selected pairs
    (equal-notional sizing; flat pairs contribute 0 to the mean). Folds are
    concatenated into one continuous daily series across the full backtest.
    """
    if cost_bps_per_leg < 0:
        raise ValueError(f"cost_bps_per_leg must be >= 0, got {cost_bps_per_leg}")
    if rolling_window < 2:
        raise ValueError(f"rolling_window must be >= 2, got {rolling_window}")

    trading_days_list = list(trading_days)
    day_to_idx: dict[date, int] = {d: i for i, d in enumerate(trading_days_list)}
    # Cost per pair round-trip is computed PER PAIR inside the fold loop
    # under the (1+β) billing convention. The per-leg log cost is the
    # β-independent component used by CostBreakdown.cost_log_per_leg.
    cost_log_per_leg = cost_bps_per_leg / 10_000.0

    all_trades: list[Trade] = []
    portfolio_rows: list[dict] = []
    selected_per_fold: dict[int, list[Pair]] = {}
    per_pair_daily: dict[str, list[dict]] = {}

    funnel = {
        "n_folds": len(folds),
        "n_pair_selections": 0,
        "n_pair_fold_units": 0,  # (pair, fold) units where prices loaded OK
        "n_trades": 0,
        "n_force_closed": 0,
    }

    for fold in folds:
        pairs_in_fold = list(
            select_pairs_fn(fold.prior_start, fold.prior_end, fold.train_start, fold.train_end)
        )
        selected_per_fold[fold.fold_id] = pairs_in_fold
        funnel["n_pair_selections"] += len(pairs_in_fold)
        logger.info(
            "Fold {}: train {}→{}, test {}→{}; {} tradeable pair(s)",
            fold.fold_id,
            fold.train_start,
            fold.train_end,
            fold.test_start,
            fold.test_end,
            len(pairs_in_fold),
        )

        if not pairs_in_fold:
            continue

        # Determine warmup start = the (rolling_window - 1)th trading day BEFORE test_start.
        # We need at least `rolling_window` observations up to and including test_start
        # to have a finite rolling z at test_start.
        test_start_idx = day_to_idx[fold.test_start]
        test_end_idx = day_to_idx[fold.test_end]
        warmup_start_idx = max(0, test_start_idx - rolling_window + 1)
        warmup_start = trading_days_list[warmup_start_idx]
        test_dates_ref = trading_days_list[test_start_idx : test_end_idx + 1]
        n_test = len(test_dates_ref)

        # Per-pair stacks for portfolio aggregation
        gross_stack: list[np.ndarray] = []
        net_stack: list[np.ndarray] = []
        active_stack: list[np.ndarray] = []

        for pair in pairs_in_fold:
            dates_full, p_y, p_x = get_prices_fn(
                pair.y_sym, pair.x_sym, warmup_start, fold.test_end
            )
            if len(dates_full) == 0:
                logger.warning("Fold {} pair {}: no aligned prices", fold.fold_id, pair.key)
                continue
            funnel["n_pair_fold_units"] += 1

            spread_full = compute_spread(p_y, p_x, beta=pair.beta, intercept=pair.alpha)
            z_full = rolling_zscore(spread_full, window=rolling_window)

            # Slice to test
            try:
                test_first_in_pair = next(
                    i for i, d in enumerate(dates_full) if d >= fold.test_start
                )
            except StopIteration:
                continue
            dates_pair_test = dates_full[test_first_in_pair:]
            spread_pair_test = spread_full[test_first_in_pair:]
            z_pair_test = z_full[test_first_in_pair:]

            max_holding = max(
                2,
                min(
                    max_holding_cap_days,
                    int(math.ceil(pair.half_life * max_holding_half_life_multiplier)),
                ),
            )
            sig = generate_signals(
                z_pair_test,
                entry=entry_z,
                exit=exit_z,
                stop=stop_z,
                max_holding=max_holding,
            )

            # β-aware billing: (1+β) × per-leg log cost.
            cost_log_per_round_trip = (1.0 + float(pair.beta)) * cost_log_per_leg
            trades, gross_daily, net_daily = _identify_trades_and_returns(
                fold_id=fold.fold_id,
                pair_key=pair.key,
                dates_test=dates_pair_test,
                spread_test=spread_pair_test,
                z_test=z_pair_test,
                position=sig.position,
                exit_reason=sig.exit_reason,
                days_in_trade=sig.days_in_trade,
                cost_log_per_round_trip=cost_log_per_round_trip,
                pair_beta=float(pair.beta),
            )
            all_trades.extend(trades)
            funnel["n_trades"] += len(trades)
            funnel["n_force_closed"] += sum(1 for t in trades if t.exit_reason == "fold_boundary")

            # Reindex pair series onto the canonical test trading-day axis (length n_test);
            # any missing pair date contributes 0 (untraded that day).
            pair_idx = {d: i for i, d in enumerate(dates_pair_test)}
            gross_aligned = np.zeros(n_test, dtype=float)
            net_aligned = np.zeros(n_test, dtype=float)
            pos_aligned = np.zeros(n_test, dtype=np.int8)
            for j, d in enumerate(test_dates_ref):
                i = pair_idx.get(d)
                if i is not None:
                    gross_aligned[j] = gross_daily[i]
                    net_aligned[j] = net_daily[i]
                    pos_aligned[j] = sig.position[i]
            gross_stack.append(gross_aligned)
            net_stack.append(net_aligned)
            active_stack.append(pos_aligned != 0)

            # Accumulate per-pair daily series across folds
            per_pair_daily.setdefault(pair.key, []).extend(
                [
                    {
                        "date": d,
                        "fold_id": fold.fold_id,
                        "gross_log_ret": float(gross_aligned[j]),
                        "net_log_ret": float(net_aligned[j]),
                        "position": int(pos_aligned[j]),
                    }
                    for j, d in enumerate(test_dates_ref)
                ]
            )

        if not gross_stack:
            continue
        n_pairs = len(gross_stack)
        gross_arr = np.stack(gross_stack, axis=0)
        net_arr = np.stack(net_stack, axis=0)
        active_arr = np.stack(active_stack, axis=0)
        portfolio_gross = gross_arr.mean(axis=0)
        portfolio_net = net_arr.mean(axis=0)
        n_active_per_day = active_arr.sum(axis=0)

        for j, d in enumerate(test_dates_ref):
            portfolio_rows.append(
                {
                    "date": d,
                    "fold_id": fold.fold_id,
                    "gross_log_ret": float(portfolio_gross[j]),
                    "net_log_ret": float(portfolio_net[j]),
                    "n_selected_pairs": n_pairs,
                    "n_active_pairs": int(n_active_per_day[j]),
                }
            )

    portfolio_df = pl.DataFrame(portfolio_rows).sort("date") if portfolio_rows else pl.DataFrame()
    per_pair_df = {k: pl.DataFrame(v).sort("date") for k, v in per_pair_daily.items() if v}

    return WalkForwardResult(
        folds=folds,
        selected_pairs_per_fold=selected_per_fold,
        trades=all_trades,
        portfolio_daily=portfolio_df,
        per_pair_daily=per_pair_df,
        funnel=funnel,
    )
