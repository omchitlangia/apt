"""Intraday cost model for the Phase-3 two-regime backtest.

Component breakdown (all in basis points of notional, per LEG, per ROUND-TRIP):

* **STT** — Securities Transaction Tax on intraday equity is 0.025 % of
  the sell-side notional (no buy-side STT for intraday). Per leg, the
  round-trip has exactly one sell, so STT per leg per round-trip = 2.5 bps.
  [TODO: confirm current rate — circa 2025 budget changes may have altered
  this; the value here matches the rate in effect through 2021, which
  matches the minute-panel span.]

* **Brokerage** — flat-fee or per-trade-bps, configurable. Default 0.5 bps
  per leg per round-trip (≈ Rs 20/order on a Rs 4 lakh notional, common
  Indian discount-broker pricing).

* **Exchange + regulatory** — NSE transaction charge (~0.00345 %), SEBI
  turnover fee, stamp duty (intraday, small), 18 % GST on the above.
  Lumped together ≈ 1.5 bps per leg per round-trip (1.5 bps on EACH of
  buy + sell totals ~3 bps; the value used here is the per-LEG per-round-
  trip aggregate).

* **Bid-ask half-spread** — assumed, not measured. SWEPT over
  ``{0.5, 1.5, 2.5, 4.0}`` bps half-spread, i.e. total quoted spread
  ``{1, 3, 5, 8}`` bps. A round-trip on one leg crosses the spread once
  at entry and once at exit ⇒ the spread cost per leg per round-trip
  equals the FULL quoted spread (= 2 × half-spread).

Total per leg per round-trip = STT + brokerage + exchange + 2 × half-spread.
A pair trade has TWO legs, so the cost log deducted on each exit is
``2 × cost_bps_per_leg / 10_000`` — same convention as the Phase 2A engine
(`run_walkforward(cost_bps_per_leg=...)`), which keeps a single knob.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fixed components in bps (per leg, per round-trip).
STT_BPS_PER_LEG_RT: float = 2.5
BROKERAGE_BPS_PER_LEG_RT: float = 0.5
EXCHANGE_REG_BPS_PER_LEG_RT: float = 1.5
FIXED_PER_LEG_RT: float = (
    STT_BPS_PER_LEG_RT + BROKERAGE_BPS_PER_LEG_RT + EXCHANGE_REG_BPS_PER_LEG_RT
)

# Spread-sweep grid (total quoted spread in bps; half-spread = value/2).
SPREAD_SWEEP_BPS: tuple[int, ...] = (1, 3, 5, 8)


@dataclass(frozen=True)
class CostBreakdown:
    """Per-leg-per-round-trip cost components for one spread level."""

    total_spread_bps: int
    stt_bps: float = STT_BPS_PER_LEG_RT
    brokerage_bps: float = BROKERAGE_BPS_PER_LEG_RT
    exchange_reg_bps: float = EXCHANGE_REG_BPS_PER_LEG_RT

    @property
    def half_spread_bps(self) -> float:
        return self.total_spread_bps / 2.0

    @property
    def spread_cost_bps(self) -> float:
        """Full quoted spread is paid once per leg per round-trip (entry+exit)."""
        return float(self.total_spread_bps)

    @property
    def cost_bps_per_leg(self) -> float:
        """Total per-leg per-round-trip cost in bps. Feeds the engine knob."""
        return FIXED_PER_LEG_RT + self.spread_cost_bps

    @property
    def cost_per_pair_round_trip_bps(self) -> float:
        return 2.0 * self.cost_bps_per_leg

    @property
    def cost_log_per_pair_round_trip(self) -> float:
        return self.cost_per_pair_round_trip_bps / 10_000.0


def cost_breakdowns(spreads: tuple[int, ...] = SPREAD_SWEEP_BPS) -> list[CostBreakdown]:
    return [CostBreakdown(total_spread_bps=s) for s in spreads]


__all__ = [
    "CostBreakdown",
    "cost_breakdowns",
    "FIXED_PER_LEG_RT",
    "STT_BPS_PER_LEG_RT",
    "BROKERAGE_BPS_PER_LEG_RT",
    "EXCHANGE_REG_BPS_PER_LEG_RT",
    "SPREAD_SWEEP_BPS",
]
