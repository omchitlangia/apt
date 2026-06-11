"""Intraday two-regime backtest engine for one (pair, fold, regime).

Outputs minute-level gross/net log returns, the round-trip trade list, and
per-bar metadata. Daily aggregation is left to the orchestrator script.

Causality
---------
Same convention as Phase 2A's daily engine:

    ret[t] = position[t-1] * (spread[t] - spread[t-1])

i.e. the position state at the close of bar t-1 (set from z[t-1]) realizes
the close-to-close move from t-1 to t. For intraday, this means a signal
fired at bar t's close is acted on at the EARLIEST at bar t+1 — which is
why we never use ``position[t]`` to multiply ``spread[t] - spread[t-1]``.

Session boundaries
------------------
At the first bar of session s, ``position[t-1]`` belongs to the prior
session's last bar (or 0 in Regime A's force-close case). The
``return[t]`` at the session-open bar is therefore:

* Regime A — exactly 0 (we forced flat at end of the prior session, so
  ``position[t-1] = 0`` by construction).
* Regime B — the overnight gap return, ``prev_pos * (spread_open -
  spread_prev_close)``. This is realized as a single one-bar P&L event.

NaN spread bars (no-trade minutes for either leg) carry forward state
inside ``generate_signals`` and contribute 0 to the return at that bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apt.intraday.signals import IntradaySignalSeries


@dataclass(frozen=True)
class IntradayTrade:
    """One round-trip on one pair within one regime + fold.

    ``cost_log`` is the **billed** round-trip cost (``(1 + pair_beta) ×
    cost_log_per_leg`` under the v2-cost-beta schema). ``pair_beta``
    records the β actually used to bill, which lets a future re-stamp
    pass derive net P&L without re-running.
    """

    fold_id: int
    pair_key: str
    regime: str  # 'A' or 'B'
    direction: int  # +1 / -1
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_z: float
    exit_z: float
    bars_held: int
    sessions_held: int
    gross_log_pnl: float
    cost_log: float
    net_log_pnl: float
    exit_reason: str  # mean_revert | stop | time | session_close | fold_boundary
    pair_beta: float = float("nan")


@dataclass(frozen=True)
class IntradayPairFoldResult:
    """All artifacts from running one (pair, fold, regime, cost) combination."""

    timestamps: pd.DatetimeIndex
    session_id: np.ndarray
    spread: np.ndarray
    z: np.ndarray
    position: np.ndarray
    gross_log_ret: np.ndarray
    net_log_ret: np.ndarray
    trades: list[IntradayTrade]


def run_pair_fold(
    *,
    fold_id: int,
    pair_key: str,
    timestamps: pd.DatetimeIndex,
    session_id: np.ndarray,
    spread: np.ndarray,
    z: np.ndarray,
    signals: IntradaySignalSeries,
    cost_log_per_round_trip: float,
    finalize_fold_boundary: bool = True,
    pair_beta: float = float("nan"),
) -> IntradayPairFoldResult:
    """Run minute-level PnL accounting for one pair-fold under one regime/cost.

    Parameters
    ----------
    fold_id, pair_key
        Identifiers carried through into each emitted ``IntradayTrade``.
    timestamps
        Length-N tz-aware index over the test window.
    session_id
        Length-N int session rank. Used to zero out the overnight-gap
        return for Regime A and to count ``sessions_held``.
    spread, z
        Length-N spread (log) and z arrays. Non-tradeable minutes carry
        NaN; runs of NaN in ``z`` keep state via ``generate_signals``.
    signals
        Output of :func:`apt.intraday.signals.generate_signals_two_regime`
        (already regime-aware: Regime A is session-reset + force-close,
        Regime B is continuous).
    cost_log_per_round_trip
        Total cost (in log-return units) deducted on the EXIT bar of each
        round-trip. For a pair, this is ``2 × cost_bps_per_leg / 10_000``.
    finalize_fold_boundary
        If True (default), any position still open at the LAST bar of the
        test window is force-closed and emitted with
        ``exit_reason='fold_boundary'``. Used by Regime B.

    Returns
    -------
    IntradayPairFoldResult
    """
    n = int(timestamps.size)
    pos = np.asarray(signals.position, dtype=np.int8).copy()
    er: list = list(signals.exit_reason)
    sids = np.asarray(session_id)
    s = np.asarray(spread, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    regime = signals.regime

    # ------------------------------------------------------------------
    # 1. Minute-level gross returns: prev-position × spread diff
    # ------------------------------------------------------------------
    gross = np.zeros(n, dtype=float)
    if n > 1:
        ds = np.diff(s)
        # Treat NaN spread moves as 0 return — no signal, no fill
        ds = np.where(np.isfinite(ds), ds, 0.0)
        gross[1:] = pos[:-1].astype(float) * ds

    # Regime A — at session-open bars, prev-position belongs to the prior
    # session; force its contribution to 0 (we force-closed at the prior
    # session's end, so position[t-1] should already be 0, but the
    # session_id boundary is the authoritative cutpoint).
    if regime == "A":
        session_open_idx = np.concatenate(([0], np.flatnonzero(np.diff(sids)) + 1))
        gross[session_open_idx] = 0.0

    net = gross.copy()

    # ------------------------------------------------------------------
    # 2. Walk the position to extract round-trip trades + deduct costs
    # ------------------------------------------------------------------
    trades: list[IntradayTrade] = []
    in_trade_entry: int | None = None
    in_trade_dir = 0

    for i in range(n):
        prev_pos = 0 if i == 0 else int(pos[i - 1])
        cur_pos = int(pos[i])
        if prev_pos == 0 and cur_pos != 0:
            in_trade_entry = i
            in_trade_dir = cur_pos
        reason = er[i]
        if reason is not None and in_trade_entry is not None:
            entry_idx = in_trade_entry
            exit_idx = i
            direction = in_trade_dir
            # Gross is exit_spread - entry_spread, signed.
            gross_pnl = float(direction) * (s[exit_idx] - s[entry_idx])
            cost = cost_log_per_round_trip
            net_pnl = gross_pnl - cost
            net[exit_idx] -= cost
            bars = exit_idx - entry_idx
            sessions = int(sids[exit_idx] - sids[entry_idx])
            trades.append(
                IntradayTrade(
                    fold_id=fold_id,
                    pair_key=pair_key,
                    regime=regime,
                    direction=int(direction),
                    entry_ts=timestamps[entry_idx],
                    exit_ts=timestamps[exit_idx],
                    entry_z=float(z_arr[entry_idx])
                    if np.isfinite(z_arr[entry_idx])
                    else float("nan"),
                    exit_z=float(z_arr[exit_idx]) if np.isfinite(z_arr[exit_idx]) else float("nan"),
                    bars_held=int(bars),
                    sessions_held=sessions,
                    gross_log_pnl=float(gross_pnl),
                    cost_log=float(cost),
                    net_log_pnl=float(net_pnl),
                    exit_reason=str(reason),
                    pair_beta=float(pair_beta),
                )
            )
            in_trade_entry = None
            in_trade_dir = 0

    # ------------------------------------------------------------------
    # 3. Optional force-close at test-window boundary (Regime B mostly)
    # ------------------------------------------------------------------
    if finalize_fold_boundary and in_trade_entry is not None and n > 0:
        entry_idx = in_trade_entry
        exit_idx = n - 1
        direction = in_trade_dir
        # Find the last finite spread to mark-to-market against
        finite_idx_arr = np.flatnonzero(np.isfinite(s))
        if finite_idx_arr.size:
            exit_idx = int(finite_idx_arr[-1])
        gross_pnl = float(direction) * (s[exit_idx] - s[entry_idx])
        cost = cost_log_per_round_trip
        net_pnl = gross_pnl - cost
        net[exit_idx] -= cost
        bars = exit_idx - entry_idx
        sessions = int(sids[exit_idx] - sids[entry_idx])
        trades.append(
            IntradayTrade(
                fold_id=fold_id,
                pair_key=pair_key,
                regime=regime,
                direction=int(direction),
                entry_ts=timestamps[entry_idx],
                exit_ts=timestamps[exit_idx],
                entry_z=float(z_arr[entry_idx]) if np.isfinite(z_arr[entry_idx]) else float("nan"),
                exit_z=float(z_arr[exit_idx]) if np.isfinite(z_arr[exit_idx]) else float("nan"),
                bars_held=int(bars),
                sessions_held=sessions,
                gross_log_pnl=float(gross_pnl),
                cost_log=float(cost),
                net_log_pnl=float(net_pnl),
                exit_reason="fold_boundary",
                pair_beta=float(pair_beta),
            )
        )

    return IntradayPairFoldResult(
        timestamps=timestamps,
        session_id=sids,
        spread=s,
        z=z_arr,
        position=pos,
        gross_log_ret=gross,
        net_log_ret=net,
        trades=trades,
    )


def aggregate_to_daily(result: IntradayPairFoldResult) -> pd.DataFrame:
    """Per-session aggregated PnL (sum of minute returns within each session)."""
    if result.timestamps.size == 0:
        return pd.DataFrame(columns=["date", "gross_log_ret", "net_log_ret"])
    df = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(result.timestamps).date,
            "gross_log_ret": result.gross_log_ret,
            "net_log_ret": result.net_log_ret,
        }
    )
    return df.groupby("date", as_index=False)[["gross_log_ret", "net_log_ret"]].sum()


__all__ = [
    "IntradayTrade",
    "IntradayPairFoldResult",
    "run_pair_fold",
    "aggregate_to_daily",
]
