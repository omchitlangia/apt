"""Session-aware signal generation wrapping ``apt.signals.spread.generate_signals``.

Two regimes are exposed via the same engine, by varying the session-boundary
handling:

* **Regime A — intraday-only / deployable**. The state machine is run
  PER SESSION. At the start of each session position is FLAT and the
  rolling window's warm-up restarts within the session. At the last
  tradeable bar of the session, any open position is force-closed
  (``exit_reason='session_close'``) — modelling the squareoff that
  removes the multi-day-short caveat for Indian cash equity.

* **Regime B — multi-day carry / upper-bound**. The state machine is run
  on the FULL concatenated z. Positions persist across the overnight gap
  (which is realized as a one-bar return at the 09:15 bar of each new
  session). The fold-boundary force-close is handled by the backtest
  engine, exactly like Phase 2A.

Both regimes share the same identical z input — A vs B is ONLY about
session-boundary handling — so the difference is interpretable as the
overnight-reversion component of the PnL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apt.intraday.calendar import session_segments
from apt.signals.spread import SignalSeries, generate_signals


@dataclass(frozen=True)
class IntradaySignalSeries:
    """Container compatible with the daily ``SignalSeries`` interface plus a regime tag."""

    position: np.ndarray  # int8 in {-1, 0, +1}, length N
    days_in_trade: np.ndarray  # int32, length N (counts BARS in trade)
    exit_reason: list  # length N, optional[str]
    regime: str  # 'A' (intraday) or 'B' (carry)


def _generate_one_session(
    z_slice: np.ndarray,
    tradeable_slice: np.ndarray,
    *,
    entry: float,
    exit: float,
    stop: float,
    max_holding: int,
    force_close_at_last_tradeable: bool,
) -> SignalSeries:
    """Run the state machine on one session; optionally force-close at session end."""
    sig = generate_signals(
        z_slice,
        entry=entry,
        exit=exit,
        stop=stop,
        max_holding=max_holding,
    )
    if not force_close_at_last_tradeable:
        return sig

    n = z_slice.size
    if n == 0:
        return sig
    pos = sig.position.copy()
    er = list(sig.exit_reason)
    held = sig.days_in_trade.copy()
    # Find the LAST tradeable bar within this session — that's where the
    # market-on-close fill lands. If no tradeable bar exists, there's
    # nothing to close.
    last_idx_arr = np.flatnonzero(tradeable_slice)
    if last_idx_arr.size == 0:
        return SignalSeries(position=pos, days_in_trade=held, exit_reason=er)
    last = int(last_idx_arr[-1])
    # Position carried INTO bar `last` (i.e., pos[last-1] if it would have been open)
    # The convention: pos[t] is the state AFTER processing bar t.
    # We want to force pos[last] = 0 (closed at last bar). If the state machine
    # already flagged an exit at `last`, leave it alone (already closed).
    pos_before = int(pos[last - 1]) if last > 0 else 0
    if pos[last] != 0:
        # Position is open AT the close of `last`. Force a session-close exit there.
        pos[last] = 0
        held[last] = 0
        # Override exit_reason only if no natural exit already fired at this bar
        if er[last] is None:
            er[last] = "session_close"
    elif pos_before != 0 and er[last] is None:
        # Edge case: state machine flat at `last` but pos_before != 0 and no
        # exit_reason recorded. Shouldn't happen with the daily generate_signals
        # (flat-back transitions always set exit_reason), but be defensive.
        er[last] = "session_close"
    return SignalSeries(position=pos, days_in_trade=held, exit_reason=er)


def generate_signals_two_regime(
    z: np.ndarray,
    session_id: np.ndarray,
    tradeable: np.ndarray,
    *,
    regime: str,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 3.5,
    max_holding: int = 60,
) -> IntradaySignalSeries:
    """Generate intraday signals for one regime (``'A'`` or ``'B'``).

    Parameters
    ----------
    z
        Length-N z-score array. NaN where unwarmed or non-tradeable.
    session_id
        Length-N dense-rank session index (0..S-1).
    tradeable
        Length-N bool — True when both legs traded that minute. Used to
        locate the session's force-close bar in Regime A.
    regime
        ``'A'`` (intraday-only) or ``'B'`` (multi-day carry).
    entry, exit, stop, max_holding
        Same as :func:`apt.signals.spread.generate_signals`. ``max_holding``
        is in BARS (minutes), not days.
    """
    if regime not in ("A", "B"):
        raise ValueError(f"regime must be 'A' or 'B', got {regime!r}")
    z = np.asarray(z, dtype=float)
    sids = np.asarray(session_id)
    tr = np.asarray(tradeable, dtype=bool)
    if z.shape != sids.shape or z.shape != tr.shape:
        raise ValueError("shape mismatch among z / session_id / tradeable")

    pos = np.zeros(z.size, dtype=np.int8)
    held = np.zeros(z.size, dtype=np.int32)
    er: list = [None] * z.size

    if regime == "A":
        for a, b in session_segments(sids):
            sig = _generate_one_session(
                z[a:b],
                tr[a:b],
                entry=entry,
                exit=exit,
                stop=stop,
                max_holding=max_holding,
                force_close_at_last_tradeable=True,
            )
            pos[a:b] = sig.position
            held[a:b] = sig.days_in_trade
            er[a:b] = list(sig.exit_reason)
    else:  # Regime B — single continuous state machine
        sig = generate_signals(z, entry=entry, exit=exit, stop=stop, max_holding=max_holding)
        pos[:] = sig.position
        held[:] = sig.days_in_trade
        er[:] = list(sig.exit_reason)

    return IntradaySignalSeries(position=pos, days_in_trade=held, exit_reason=er, regime=regime)


def _generate_signals_ou_one_session(
    z_ou: np.ndarray,
    tradeable: np.ndarray,
    *,
    a_entry_z: float,
    stop_mode: str,
    stop_k_sigma: float,
    max_holding: int,
    force_close_at_last_tradeable: bool,
) -> SignalSeries:
    """OU state machine on one session slice (or one continuous span)."""
    n = z_ou.size
    pos = np.zeros(n, dtype=np.int8)
    held = np.zeros(n, dtype=np.int32)
    er: list = [None] * n

    use_hard_stop = stop_mode == "hard"
    cur_pos = 0
    cur_held = 0

    for t in range(n):
        z = float(z_ou[t]) if np.isfinite(z_ou[t]) else float("nan")
        if not np.isfinite(z):
            # NaN bar: carry state, do not increment held, no exit
            pos[t] = cur_pos
            held[t] = cur_held
            continue

        if cur_pos == 0:
            # Flat -> consider entry on tradeable bars
            if z >= a_entry_z:
                cur_pos = -1  # short spread (spread overpriced)
                cur_held = 0
            elif z <= -a_entry_z:
                cur_pos = +1  # long spread
                cur_held = 0
            pos[t] = cur_pos
            held[t] = cur_held
            if cur_pos != 0:
                cur_held = 1  # next bar will be the first bar held
        else:
            cur_held += 1
            # Check exits in priority order
            if cur_pos == -1:
                if use_hard_stop and z >= stop_k_sigma:
                    er[t] = "z_stop"
                    cur_pos = 0
                    cur_held = 0
                elif z <= 0.0:
                    er[t] = "mean_revert"
                    cur_pos = 0
                    cur_held = 0
                elif cur_held - 1 >= max_holding:
                    er[t] = "time_stop"
                    cur_pos = 0
                    cur_held = 0
            else:  # cur_pos == +1
                if use_hard_stop and z <= -stop_k_sigma:
                    er[t] = "z_stop"
                    cur_pos = 0
                    cur_held = 0
                elif z >= 0.0:
                    er[t] = "mean_revert"
                    cur_pos = 0
                    cur_held = 0
                elif cur_held - 1 >= max_holding:
                    er[t] = "time_stop"
                    cur_pos = 0
                    cur_held = 0
            pos[t] = cur_pos
            held[t] = cur_held if cur_pos != 0 else 0

    if force_close_at_last_tradeable and cur_pos != 0:
        last_tradeable = np.flatnonzero(tradeable)
        if last_tradeable.size:
            last = int(last_tradeable[-1])
            pos[last:] = 0
            if er[last] is None:
                er[last] = "session_close"
            held[last:] = 0

    return SignalSeries(position=pos, days_in_trade=held, exit_reason=er)


def generate_signals_ou(
    z_ou: np.ndarray,
    session_id: np.ndarray,
    tradeable: np.ndarray,
    *,
    regime: str,
    a_entry_z: float,
    stop_mode: str = "none",
    stop_k_sigma: float = 4.0,
    max_holding: int,
) -> IntradaySignalSeries:
    """OU-engine drop-in replacement for :func:`generate_signals_two_regime`.

    State machine evaluated on each completed bar's ``z_ou[t] = (X[t] -
    mu_OU) / sigma_eq`` (TRAIN-frozen ``mu_OU``, ``sigma_eq``):

    * flat -> short when ``z_ou >= +a_entry_z``
    * flat -> long  when ``z_ou <= -a_entry_z``
    * short -> flat when ``z_ou <= 0`` (``mean_revert``)
                    OR ``z_ou >= +stop_k_sigma`` (``z_stop``,
                       only if ``stop_mode == "hard"``)
                    OR ``held >= max_holding`` (``time_stop``)
    * long  -> flat: mirror of short.

    Regime A: per-session, force-close at the last tradeable bar of the
    session with ``exit_reason='session_close'`` (CSV emits this as
    ``eod_squareoff``). Regime B: continuous; the orchestrator handles
    fold-boundary close.
    Re-entry: state-machine fires immediately on the next bar that
    triggers an entry condition once flat — no cool-off.

    Parameters
    ----------
    z_ou
        Length-N TRAIN-frozen Z-OU array. NaN where unwarmed /
        non-tradeable; state carries through NaN runs.
    session_id
        Length-N dense-rank session index (0..S-1).
    tradeable
        Length-N bool — used to locate the session-close bar in Regime A.
    regime
        ``'A'`` (intraday-only, force-close per session) or ``'B'``
        (multi-day carry).
    a_entry_z
        Bertram optimal entry threshold in Z-OU units, > 0.
    stop_mode
        ``'none'`` (default) or ``'hard'``. Hard mode triggers a z-stop
        at ``|z_ou| >= stop_k_sigma``.
    stop_k_sigma
        K. Only consulted when ``stop_mode == 'hard'``.
    max_holding
        Maximum bars in trade before forced exit with
        ``exit_reason='time_stop'``.
    """
    if regime not in ("A", "B"):
        raise ValueError(f"regime must be 'A' or 'B', got {regime!r}")
    if stop_mode not in ("none", "hard"):
        raise ValueError(f"stop_mode must be 'none' or 'hard', got {stop_mode!r}")
    if not np.isfinite(a_entry_z) or a_entry_z <= 0:
        raise ValueError(f"a_entry_z must be positive finite, got {a_entry_z}")
    z = np.asarray(z_ou, dtype=float)
    sids = np.asarray(session_id)
    tr = np.asarray(tradeable, dtype=bool)
    if z.shape != sids.shape or z.shape != tr.shape:
        raise ValueError("shape mismatch among z_ou / session_id / tradeable")

    pos = np.zeros(z.size, dtype=np.int8)
    held = np.zeros(z.size, dtype=np.int32)
    er: list = [None] * z.size

    if regime == "A":
        for a, b in session_segments(sids):
            sig = _generate_signals_ou_one_session(
                z[a:b],
                tr[a:b],
                a_entry_z=a_entry_z,
                stop_mode=stop_mode,
                stop_k_sigma=stop_k_sigma,
                max_holding=max_holding,
                force_close_at_last_tradeable=True,
            )
            pos[a:b] = sig.position
            held[a:b] = sig.days_in_trade
            er[a:b] = list(sig.exit_reason)
    else:  # Regime B
        sig = _generate_signals_ou_one_session(
            z,
            tr,
            a_entry_z=a_entry_z,
            stop_mode=stop_mode,
            stop_k_sigma=stop_k_sigma,
            max_holding=max_holding,
            force_close_at_last_tradeable=False,
        )
        pos[:] = sig.position
        held[:] = sig.days_in_trade
        er[:] = list(sig.exit_reason)

    return IntradaySignalSeries(position=pos, days_in_trade=held, exit_reason=er, regime=regime)


__all__ = ["IntradaySignalSeries", "generate_signals_two_regime", "generate_signals_ou"]
