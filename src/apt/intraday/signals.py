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


__all__ = ["IntradaySignalSeries", "generate_signals_two_regime"]
