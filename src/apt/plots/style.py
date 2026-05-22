"""Central matplotlib style for APT figures.

Call :func:`apply_style` once at the top of any script that produces plots.
Project convention: each plot function in ``apt.plots`` takes a ``Path`` and
writes a PNG. No interactive ``plt.show()``.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

APT_PALETTE: list[str] = [
    "#2E86AB",  # blue
    "#A23B72",  # magenta
    "#F18F01",  # orange
    "#C73E1D",  # red
    "#3B8E5C",  # green
    "#6B5B95",  # purple
    "#7E8083",  # grey
]


def apply_style() -> None:
    """Apply the APT-wide matplotlib style (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
            "axes.prop_cycle": plt.cycler(color=APT_PALETTE),
        }
    )
