"""Walk-forward backtest engine — leakage-free, cost-realistic, asset-agnostic.

Phase 2A: :mod:`apt.backtest.walkforward` (fixed equal-notional).
Phase 2B: :mod:`apt.backtest.risk_managed` (sizing/risk ablation ladder)
          + :mod:`apt.backtest.sizing` + :mod:`apt.backtest.kill_switch`.
"""

from apt.backtest.risk_managed import (
    KillEvent,
    PrematureCut,
    RiskConfig,
    RiskManagedResult,
    RMTrade,
    run_walkforward_risk_managed,
)
from apt.backtest.walkforward import (
    Fold,
    Pair,
    Trade,
    WalkForwardResult,
    build_folds,
    compute_metrics,
    run_walkforward,
)

__all__ = [
    "Fold",
    "KillEvent",
    "Pair",
    "PrematureCut",
    "RMTrade",
    "RiskConfig",
    "RiskManagedResult",
    "Trade",
    "WalkForwardResult",
    "build_folds",
    "compute_metrics",
    "run_walkforward",
    "run_walkforward_risk_managed",
]
