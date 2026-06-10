"""Within-session minute-bar aggregation to coarser bar frequencies.

A coarser bar at ``freq_minutes`` covers the half-open interval
``[bar_start, bar_start + freq_minutes)`` with **left-labelled**
timestamps. No bar spans the overnight gap — resampling groups by
``(session_id, bar // freq_minutes)`` so a bar at 15:25 with freq=10
does NOT include the 09:15 print of the next session.

This module is the bar-frequency primitive consumed by the OU pipeline
(``scripts/15_phase3_ou.py``). The 1-minute pass-through is a no-op
identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apt.intraday.loader import AlignedMinutePair


@dataclass(frozen=True)
class ResampledPair:
    """Coarser-bar projection of an :class:`AlignedMinutePair`.

    All arrays are aligned to the resampled-bar grid (length M <= N).
    ``tradeable`` is True iff any constituent 1-minute bar in the
    coarser-bar window was tradeable; ``close_*`` is the last finite
    close in the window (NaN if all sub-bars non-tradeable).
    """

    timestamps: pd.DatetimeIndex
    session_id: np.ndarray
    bar_in_session: np.ndarray
    close_y: np.ndarray
    close_x: np.ndarray
    tradeable: np.ndarray
    freq_minutes: int

    @property
    def n_bars(self) -> int:
        return int(self.close_y.size)

    @property
    def n_sessions(self) -> int:
        return int(self.session_id.max() + 1) if self.session_id.size else 0


def resample_within_session(
    aligned: AlignedMinutePair,
    *,
    freq_minutes: int,
) -> ResampledPair:
    """Aggregate ``aligned`` to ``freq_minutes`` bars, within session only.

    - Left-labelled: a bar at session bar-in-session ``k`` covers
      ``[k * freq_minutes, (k+1) * freq_minutes)`` minutes from session
      open.
    - No bar spans sessions.
    - ``close_y / close_x`` = last finite close in the window (NaN if all
      sub-bars non-tradeable).
    - ``tradeable`` = any-tradeable in the window. A coarser bar is
      considered tradeable iff there was at least one 1-minute trade
      with finite closes on BOTH legs within the window.
    - At ``freq_minutes == 1`` returns a pass-through identity.
    """
    if freq_minutes < 1:
        raise ValueError(f"freq_minutes must be >= 1, got {freq_minutes}")
    if freq_minutes == 1:
        return ResampledPair(
            timestamps=aligned.timestamps,
            session_id=aligned.session_id.astype(np.int32, copy=False),
            bar_in_session=aligned.bar_in_session.astype(np.int32, copy=False),
            close_y=aligned.close_y.astype(float, copy=False),
            close_x=aligned.close_x.astype(float, copy=False),
            tradeable=aligned.tradeable.astype(bool, copy=False),
            freq_minutes=1,
        )

    n = aligned.n_bars
    if n == 0:
        return ResampledPair(
            timestamps=aligned.timestamps,
            session_id=np.empty(0, dtype=np.int32),
            bar_in_session=np.empty(0, dtype=np.int32),
            close_y=np.empty(0, dtype=float),
            close_x=np.empty(0, dtype=float),
            tradeable=np.empty(0, dtype=bool),
            freq_minutes=freq_minutes,
        )

    sids = np.asarray(aligned.session_id, dtype=np.int64)
    bin_idx = np.asarray(aligned.bar_in_session, dtype=np.int64)
    coarse_bin = bin_idx // freq_minutes  # within-session coarser-bar index
    # Composite int key for (session, coarse_bin). Assumes coarse_bin < 65536.
    key = sids * (1 << 16) + coarse_bin

    # Sort by key to ensure groupby boundaries align with the original order
    # (it already does since session_id is monotonic and bin_in_session is
    # monotonic within session, so key is monotonic non-decreasing).
    # Forward-fill closes within group (vectorised), then take .last() per
    # group. This avoids the slow .apply(lambda) path.
    df = pd.DataFrame(
        {
            "key": key,
            "session_id": sids,
            "coarse_bin": coarse_bin,
            "ts": aligned.timestamps,
            "close_y": aligned.close_y,
            "close_x": aligned.close_x,
            "tradeable": aligned.tradeable.astype(bool),
        }
    )
    df["close_y_ff"] = df.groupby("key", sort=False)["close_y"].ffill()
    df["close_x_ff"] = df.groupby("key", sort=False)["close_x"].ffill()

    grouped = df.groupby("key", sort=True, as_index=False).agg(
        ts=("ts", "first"),
        session_id=("session_id", "first"),
        coarse_bin=("coarse_bin", "first"),
        any_tradeable=("tradeable", "any"),
        close_y=("close_y_ff", "last"),
        close_x=("close_x_ff", "last"),
    )
    # Tradeable = any-tradeable AND both legs have finite close
    tradeable_out = (
        grouped["any_tradeable"].to_numpy()
        & np.isfinite(grouped["close_y"].to_numpy())
        & np.isfinite(grouped["close_x"].to_numpy())
    )
    return ResampledPair(
        timestamps=pd.DatetimeIndex(grouped["ts"].values),
        session_id=grouped["session_id"].to_numpy(dtype=np.int32),
        bar_in_session=grouped["coarse_bin"].to_numpy(dtype=np.int32),
        close_y=grouped["close_y"].to_numpy(dtype=float),
        close_x=grouped["close_x"].to_numpy(dtype=float),
        tradeable=tradeable_out.astype(bool),
        freq_minutes=int(freq_minutes),
    )


__all__ = ["ResampledPair", "resample_within_session"]
