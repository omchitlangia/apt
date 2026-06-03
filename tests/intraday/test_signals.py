"""Tests for the two-regime intraday signal wrapper."""

import numpy as np

from apt.intraday.signals import generate_signals_two_regime


def _two_sessions_z(n_per: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic z that triggers an entry mid-session 1 and persists into session 2."""
    z = np.zeros(2 * n_per, dtype=float)
    # warm-up bars (NaN) at the start of each session
    z[:5] = np.nan
    z[n_per : n_per + 5] = np.nan
    # Strong negative z in the middle of session 1, persists into session 2
    z[10:n_per] = -3.0  # always |z| > entry (2.0), never reverts
    z[n_per + 5 :] = -3.0  # still > entry in session 2
    sids = np.concatenate([np.zeros(n_per, dtype=np.int32), np.ones(n_per, dtype=np.int32)])
    tr = np.ones(2 * n_per, dtype=bool)
    return z, sids, tr


def test_regime_a_resets_at_session_start_and_force_closes_at_session_end() -> None:
    z, sids, tr = _two_sessions_z(n_per=30)
    sigA = generate_signals_two_regime(
        z, sids, tr, regime="A", entry=2.0, exit=0.5, stop=3.5, max_holding=60
    )
    # Position at the very end of session 1 must be flat (force-closed).
    # Session 1 ends at index 29 (0-indexed bar 29).
    assert sigA.position[29] == 0
    # And exit_reason at 29 should be 'session_close'.
    assert sigA.exit_reason[29] == "session_close"
    # Session 2 must NOT carry a position from session 1 (its bar 0 is NaN,
    # state-flat). After warm-up (NaN ends at index 34), a fresh entry fires.
    assert sigA.position[30] == 0  # first bar of session 2: flat
    # By bar 35 onward the signal triggers a fresh long.
    assert (sigA.position[35:] == 1).any()


def test_regime_b_carries_position_across_overnight_gap() -> None:
    z, sids, tr = _two_sessions_z(n_per=30)
    sigB = generate_signals_two_regime(
        z, sids, tr, regime="B", entry=2.0, exit=0.5, stop=3.5, max_holding=999
    )
    # Position must persist through the session boundary at index 29 -> 30.
    assert sigB.position[29] == 1
    # The session-start bar of session 2 is NaN — state carries forward.
    assert sigB.position[30] == 1
    # No 'session_close' exits in Regime B
    assert not any(r == "session_close" for r in sigB.exit_reason)


def test_regime_a_no_session_close_when_already_flat() -> None:
    """If position is already flat at session end, no force-close emitted."""
    n_per = 20
    z = np.zeros(2 * n_per)
    z[:] = 0.0  # never triggers entry
    sids = np.concatenate([np.zeros(n_per, dtype=np.int32), np.ones(n_per, dtype=np.int32)])
    tr = np.ones(2 * n_per, dtype=bool)
    sig = generate_signals_two_regime(z, sids, tr, regime="A", entry=2.0, exit=0.5, stop=3.5)
    assert all(r is None for r in sig.exit_reason)
    assert (sig.position == 0).all()
