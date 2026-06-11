"""Intraday cost model — β-aware (1+β) per-pair billing.

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

**β-aware billing.** The Phase-2 spread is ``Δ(log Y − β · log X)``, so a
pair round-trip pays cost on **(1 + β) units of notional**, not 2: the Y
leg is sized to 1 unit and the X leg to β units. The single central
billing entry point is :meth:`CostBreakdown.billed_cost_log_per_pair_round_trip`
which takes ``beta`` and returns the actual round-trip cost in log units.
β=1 reproduces the legacy 2× value exactly — that is the continuity pin
asserted in ``tests/intraday/test_backtest_cost_pin.py``.

Net exposure (informational, surfaced for the Kalman unit): the
``(1+β)`` billing assumes a hedged Y/X position. A long-spread pair
trade carries net long-market exposure ``(1 − β) × notional`` —
non-zero whenever β ≠ 1. Pair-trades are NOT market-neutral under this
convention; the (1−β) corollary should be reflected in any future
risk-budget machinery (Kalman unit).
"""

from __future__ import annotations

import math
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

# Trade-CSV schema version. Bump when the cost-billing convention or
# trade-row column set changes. v2 is the (1+β) convention.
TRADE_CSV_SCHEMA_VERSION: str = "v2-cost-beta-2026.06.11"


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
        """β-independent: total per-leg per-round-trip cost in bps."""
        return FIXED_PER_LEG_RT + self.spread_cost_bps

    @property
    def cost_log_per_leg(self) -> float:
        """β-independent: per-leg per-round-trip cost in log units."""
        return self.cost_bps_per_leg / 10_000.0

    def billed_cost_log_per_pair_round_trip(self, beta: float) -> float:
        """**β-aware billing** — the cost deducted on each pair round-trip.

        ``(1 + β) × cost_log_per_leg``. Derived from the spread arithmetic
        ``Δ(log Y − β · log X)``: Y leg sized to 1 unit, X leg to β units.

        - β = 0   : single-leg outright trade (1× per_leg)
        - β = 1   : reproduces the legacy 2× equal-notional value exactly
        - β > 1   : X leg is the larger; total cost > 2× per_leg
        - β < 1   : X leg is the smaller; total cost < 2× per_leg

        Raises ``ValueError`` for non-finite or negative β.
        """
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"beta must be non-negative finite, got {beta!r}")
        return (1.0 + beta) * self.cost_log_per_leg

    def billed_cost_bps_per_pair_round_trip(self, beta: float) -> float:
        """β-aware billing in bps. Mirror of the log-units method."""
        return self.billed_cost_log_per_pair_round_trip(beta) * 10_000.0


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
    "TRADE_CSV_SCHEMA_VERSION",
]
