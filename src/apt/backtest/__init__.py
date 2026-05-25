"""Walk-forward backtest engine — leakage-free, cost-realistic, asset-agnostic.

Public API lives in :mod:`apt.backtest.walkforward`.
"""

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
    "Pair",
    "Trade",
    "WalkForwardResult",
    "build_folds",
    "compute_metrics",
    "run_walkforward",
]
