"""Tests for the OU signal generator (apt.intraday.signals.generate_signals_ou)."""

from __future__ import annotations

import numpy as np
import pytest

from apt.intraday.signals import generate_signals_ou


def test_simple_short_cycle_mean_revert() -> None:
    """Spread spikes up over +a, decays to 0 -> short entry, mean-revert exit."""
    n = 30
    z = np.zeros(n)
    z[5:15] = 2.5  # above +2.0 entry
    z[15:] = 0.0  # crosses 0 -> mean-revert exit
    sids = np.zeros(n, dtype=np.int32)
    tr = np.ones(n, dtype=bool)
    sig = generate_signals_ou(z, sids, tr, regime="B", a_entry_z=2.0, max_holding=100)
    # Entry should land at index 5 (first bar where z >= 2.0)
    assert sig.position[5] == -1
    # Position remains short until z crosses 0
    assert sig.position[14] == -1
    # First z<=0 bar is index 15 (z[15]=0.0 triggers mean_revert)
    assert sig.position[15] == 0
    assert sig.exit_reason[15] == "mean_revert"


def test_long_cycle_mean_revert() -> None:
    """Spread drops to -a, recovers to 0 -> long entry, mean-revert exit."""
    n = 30
    z = np.zeros(n)
    z[5:15] = -2.5  # below -2.0 entry
    z[15:] = 0.0
    sig = generate_signals_ou(
        z,
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=bool),
        regime="B",
        a_entry_z=2.0,
        max_holding=100,
    )
    assert sig.position[5] == +1
    assert sig.position[15] == 0
    assert sig.exit_reason[15] == "mean_revert"


def test_hard_stop_fires() -> None:
    """In stop_mode='hard', |z| >= K triggers z_stop."""
    n = 30
    z = np.zeros(n)
    z[5] = 2.5  # entry short
    z[6:10] = 3.0  # still in trade
    z[10] = 4.5  # stop trigger (|z| >= K=4.0)
    z[11:] = 0.0
    sig = generate_signals_ou(
        z,
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=bool),
        regime="B",
        a_entry_z=2.0,
        stop_mode="hard",
        stop_k_sigma=4.0,
        max_holding=100,
    )
    assert sig.position[5] == -1
    assert sig.exit_reason[10] == "z_stop"
    assert sig.position[10] == 0


def test_no_hard_stop_when_mode_none() -> None:
    """In stop_mode='none' (default), large |z| does NOT trigger any stop."""
    n = 30
    z = np.zeros(n)
    z[5] = 2.5
    z[6:20] = 5.0  # huge excursion but no stop
    z[20] = 0.0  # mean-revert
    sig = generate_signals_ou(
        z,
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=bool),
        regime="B",
        a_entry_z=2.0,
        stop_mode="none",
        max_holding=100,
    )
    assert sig.position[5] == -1
    # No exit between 5 and 19
    for t in range(5, 20):
        assert sig.position[t] == -1
        assert sig.exit_reason[t] is None
    assert sig.position[20] == 0
    assert sig.exit_reason[20] == "mean_revert"


def test_time_stop_fires() -> None:
    """Holding for max_holding+1 bars without exit triggers time_stop."""
    n = 30
    z = np.full(n, 2.5)  # entry + persistent
    z[0] = 0.0  # so entry happens at index 1
    sig = generate_signals_ou(
        z,
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=bool),
        regime="B",
        a_entry_z=2.0,
        max_holding=5,
    )
    # Entry at index 1; time_stop triggers when held >= max_holding=5,
    # i.e. at index 1 + 5 = 6
    assert sig.position[1] == -1
    assert sig.exit_reason[6] == "time_stop"
    assert sig.position[6] == 0


def test_regime_a_force_close_at_session_end() -> None:
    n = 30
    z = np.zeros(n)
    z[5:] = 2.5  # entry at index 5, holds through session
    sids = np.concatenate([np.zeros(20, dtype=np.int32), np.ones(10, dtype=np.int32)])
    tr = np.ones(n, dtype=bool)
    sig = generate_signals_ou(z, sids, tr, regime="A", a_entry_z=2.0, max_holding=100)
    # Session 0 last bar = index 19. Position must be 0 from index 19 onwards.
    assert sig.position[19] == 0
    assert sig.exit_reason[19] == "session_close"


def test_nan_z_carries_state() -> None:
    """NaN z bars must not exit; state carries through NaN runs."""
    n = 30
    z = np.zeros(n)
    z[5] = 2.5
    z[6:15] = np.nan  # carry state
    z[15] = 0.0  # mean-revert
    sig = generate_signals_ou(
        z,
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=bool),
        regime="B",
        a_entry_z=2.0,
        max_holding=100,
    )
    # Entry at 5; through NaN run position remains -1
    assert sig.position[5] == -1
    for t in range(6, 15):
        assert sig.position[t] == -1
    assert sig.position[15] == 0
    assert sig.exit_reason[15] == "mean_revert"


def test_reentry_after_recross() -> None:
    """Once flat, the next entry-trigger fires immediately (no cool-off)."""
    n = 50
    z = np.zeros(n)
    z[5:10] = 2.5  # short entry at 5, holds through 9
    z[10:15] = 0.5  # mean_revert exit at 10 (first z <= 0... but 0.5 > 0)
    # Actually for short pos: exit triggers when z <= 0. So z[10:15]=0.5 keeps
    # us in the trade. Use a downward crossing instead.
    z[10:15] = -0.1  # short pos sees z <= 0 -> mean_revert at 10
    z[15:20] = 0.0  # flat
    z[20:25] = -2.5  # long entry at 20
    z[25:30] = 0.1  # long pos sees z >= 0 -> mean_revert at 25
    sig = generate_signals_ou(
        z,
        np.zeros(n, dtype=np.int32),
        np.ones(n, dtype=bool),
        regime="B",
        a_entry_z=2.0,
        max_holding=100,
    )
    assert sig.position[5] == -1
    assert sig.exit_reason[10] == "mean_revert"
    assert sig.position[20] == +1
    assert sig.exit_reason[25] == "mean_revert"


def test_invalid_regime_raises() -> None:
    with pytest.raises(ValueError, match="regime"):
        generate_signals_ou(
            np.zeros(5),
            np.zeros(5, dtype=np.int32),
            np.ones(5, dtype=bool),
            regime="C",
            a_entry_z=2.0,
            max_holding=10,
        )


def test_invalid_a_entry_raises() -> None:
    with pytest.raises(ValueError, match="a_entry_z"):
        generate_signals_ou(
            np.zeros(5),
            np.zeros(5, dtype=np.int32),
            np.ones(5, dtype=bool),
            regime="A",
            a_entry_z=0.0,
            max_holding=10,
        )
