"""Phase 4 OMNIBUS shared helpers (pure, small).

Holds only cross-section conveniences that several Phase-4 drivers reuse:

* :data:`CANONICAL_EXIT_MAP` / :func:`canonical_exit_reason` — remap the
  legacy/non-standard exit-reason strings that the persisted Phase-3 trade
  CSVs carry (``fold_boundary``, ``time``, ``stop``, ``session_close``) onto
  the FIXED five-category vocabulary in ``docs/reporting_standard.md`` §7
  (``mean_revert``, ``z_stop``, ``time_stop``, ``eod_squareoff``,
  ``fold_close``). Unknown strings fold to ``__OTHER__``.
* :func:`savefig_with_csv` — write a matplotlib figure to ``<path>.png`` and
  its backing data to ``<path>.csv`` (the reporting-standard companion-CSV
  rule), returning both paths.

No I/O-at-import, no config. Matplotlib is imported lazily inside the helper
so importing this module in a headless/test context is cheap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Fixed five-category vocabulary (reporting_standard.md §7).
FIVE_CATEGORY_EXIT_REASONS: tuple[str, ...] = (
    "mean_revert",
    "z_stop",
    "time_stop",
    "eod_squareoff",
    "fold_close",
)

# Legacy/persisted string -> canonical. The Phase-3 engines emitted
# ``fold_boundary``/``time``/``stop``/``session_close`` before the fixed
# vocabulary was frozen; the persisted CSVs still carry them.
CANONICAL_EXIT_MAP: dict[str, str] = {
    "mean_revert": "mean_revert",
    "z_stop": "z_stop",
    "stop": "z_stop",
    "hard_stop": "z_stop",
    "time_stop": "time_stop",
    "time": "time_stop",
    "eod_squareoff": "eod_squareoff",
    "session_close": "eod_squareoff",
    "eod": "eod_squareoff",
    "fold_close": "fold_close",
    "fold_boundary": "fold_close",
}


def canonical_exit_reason(raw: str) -> str:
    """Map a persisted exit-reason string onto the fixed five categories.

    Unknown values fold to ``__OTHER__`` (matching ``fig_h`` behaviour).
    """
    return CANONICAL_EXIT_MAP.get(str(raw).strip(), "__OTHER__")


def savefig_with_csv(fig, out_png: Path | str, data: pd.DataFrame) -> tuple[Path, Path]:
    """Save ``fig`` to ``out_png`` and ``data`` to the same-basename ``.csv``.

    Enforces the reporting-standard rule that every figure ships the exact
    numbers behind it. Returns ``(png_path, csv_path)``.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_png.with_suffix(".csv")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    data.to_csv(csv_path, index=False)
    return out_png, csv_path


__all__ = [
    "FIVE_CATEGORY_EXIT_REASONS",
    "CANONICAL_EXIT_MAP",
    "canonical_exit_reason",
    "savefig_with_csv",
]
