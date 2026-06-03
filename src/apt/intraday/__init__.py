"""Phase 3 — intraday validation of the APT pipeline on 1-minute bars.

Reuses the asset-agnostic spread/signal layer from `apt.signals.spread` and
the daily walk-forward selection from `apt.backtest.walkforward`. Adds only
what is intraday-specific:

* session calendar (NSE continuous session 09:15-15:30) and session segmentation,
* an Epps-safe pair loader that marks no-trade minutes non-tradeable rather
  than forward-filling stale prices,
* a sessionized rolling z-score that never blends across the overnight gap,
* a time-of-day volatility profile + TOD-adjusted z,
* a two-regime walk-forward engine: Regime A force-closes at session close
  (intraday-only / deployable), Regime B carries positions across sessions
  (upper-bound / unshortable caveat).

The daily (alpha, beta) is FROZEN per fold by the daily Phase 2A selection;
nothing here re-fits cointegration on minute data.
"""

from apt.intraday.calendar import (
    NSE_BARS_PER_SESSION,
    NSE_SESSION_END,
    NSE_SESSION_START,
    bar_of_session,
    build_session_grid,
    session_segments,
)
from apt.intraday.loader import AlignedMinutePair, load_minute_pair
from apt.intraday.signals import IntradaySignalSeries, generate_signals_two_regime
from apt.intraday.zscore import (
    fit_tod_vol_profile,
    intraday_rolling_zscore,
    sessionized_rolling_zscore,
    tod_adjusted_zscore,
)

__all__ = [
    "NSE_BARS_PER_SESSION",
    "NSE_SESSION_END",
    "NSE_SESSION_START",
    "bar_of_session",
    "build_session_grid",
    "session_segments",
    "AlignedMinutePair",
    "load_minute_pair",
    "IntradaySignalSeries",
    "generate_signals_two_regime",
    "fit_tod_vol_profile",
    "intraday_rolling_zscore",
    "sessionized_rolling_zscore",
    "tod_adjusted_zscore",
]
