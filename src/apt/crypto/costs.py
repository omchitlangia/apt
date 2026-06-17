"""Crypto cost model (Phase 4 Section 6c) — taker fee + spread; funding [TODO].

No funding-rate series exists in the data inventory (§6a), so the cost is
**taker fee + half-spread** with **funding deferred** ([TODO data] A8). The
taker fee is a LABELED DEFAULT (→ ASSUMPTIONS A7). Billing follows the same
(1+β) per-pair convention as NSE (``apt.intraday.costs``): a pair round-trip
pays on (1+β) units of notional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Labeled default (→ ASSUMPTIONS A7). Perp taker ≈ 0.05% one-way.
DEFAULT_TAKER_FEE_BPS_PER_SIDE: float = 5.0

# Spread sweep (total quoted spread, bps), mirroring the NSE grid.
CRYPTO_SPREAD_SWEEP_BPS: tuple[int, ...] = (1, 3, 5, 8)


@dataclass(frozen=True)
class CryptoCostBreakdown:
    """Per-leg per-round-trip crypto cost: taker (entry+exit) + full spread."""

    total_spread_bps: int
    taker_fee_bps_per_side: float = DEFAULT_TAKER_FEE_BPS_PER_SIDE
    funding_bps_per_pair_rt: float = 0.0  # [TODO data]: no funding series

    @property
    def taker_bps_per_leg_rt(self) -> float:
        """Taker crossed at entry AND exit = 2 × per-side fee."""
        return 2.0 * self.taker_fee_bps_per_side

    @property
    def cost_bps_per_leg(self) -> float:
        """β-independent per-leg per-round-trip cost in bps (taker + spread)."""
        return self.taker_bps_per_leg_rt + float(self.total_spread_bps)

    @property
    def cost_log_per_leg(self) -> float:
        return self.cost_bps_per_leg / 10_000.0

    def billed_cost_log_per_pair_round_trip(self, beta: float) -> float:
        """(1+β) × per-leg + funding (funding = 0 until a series exists)."""
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"beta must be non-negative finite, got {beta!r}")
        return (1.0 + beta) * self.cost_log_per_leg + self.funding_bps_per_pair_rt / 10_000.0


__all__ = [
    "CRYPTO_SPREAD_SWEEP_BPS",
    "DEFAULT_TAKER_FEE_BPS_PER_SIDE",
    "CryptoCostBreakdown",
]
