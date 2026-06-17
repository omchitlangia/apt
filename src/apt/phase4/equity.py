"""Equity-curve + drawdown plotting helpers (Phase 4 final).

Every Phase-4-final equity figure shows, per engine/panel: a **gross** NAV
line, a **net** NAV line, and an underwater **drawdown** panel. These helpers
enforce that contract and emit the backing data so each PNG ships its CSV.

Pure-ish: matplotlib imported lazily; no config, no global state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NavDrawdown:
    dates: np.ndarray
    gross_nav: np.ndarray
    net_nav: np.ndarray
    gross_drawdown: np.ndarray  # underwater, <= 0
    net_drawdown: np.ndarray


def compute_nav_drawdown(dates, gross_log_ret, net_log_ret) -> NavDrawdown:
    """NAV (start=1) and underwater drawdown from per-period log returns."""
    g = np.asarray(gross_log_ret, dtype=float)
    n = np.asarray(net_log_ret, dtype=float)
    g = np.nan_to_num(g)
    n = np.nan_to_num(n)
    gnav = np.exp(np.cumsum(g))
    nnav = np.exp(np.cumsum(n))
    gdd = gnav / np.maximum.accumulate(gnav) - 1.0
    ndd = nnav / np.maximum.accumulate(nnav) - 1.0
    return NavDrawdown(np.asarray(dates), gnav, nnav, gdd, ndd)


def nav_dataframe(panels: dict[str, NavDrawdown]) -> pd.DataFrame:
    """Long-format backing CSV for a multi-panel equity figure."""
    frames = []
    for label, nd in panels.items():
        frames.append(
            pd.DataFrame(
                {
                    "panel": label,
                    "date": nd.dates,
                    "gross_nav": nd.gross_nav,
                    "net_nav": nd.net_nav,
                    "gross_drawdown": nd.gross_drawdown,
                    "net_drawdown": nd.net_drawdown,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def plot_equity_panels(panels: dict[str, NavDrawdown], out_png, *, suptitle: str):
    """One column per panel; row 0 = NAV (gross+net), row 1 = underwater.

    Returns ``(fig, backing_dataframe)``. Caller saves via
    :func:`apt.phase4.savefig_with_csv`.
    """
    import matplotlib.pyplot as plt

    from apt.plots.style import apply_style

    apply_style()
    n = len(panels)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(5.2 * n, 6),
        sharex="col",
        gridspec_kw={"height_ratios": [2.4, 1]},
        squeeze=False,
    )
    for j, (label, nd) in enumerate(panels.items()):
        ax_nav, ax_dd = axes[0][j], axes[1][j]
        ax_nav.plot(nd.dates, nd.gross_nav, color="C0", lw=1.1, label="gross")
        ax_nav.plot(nd.dates, nd.net_nav, color="C3", lw=1.1, label="net")
        ax_nav.axhline(1.0, color="0.6", lw=0.6, ls=":")
        ax_nav.set_title(label, fontsize=10)
        ax_nav.set_ylabel("NAV (start=1)")
        ax_nav.legend(fontsize=8, loc="upper left")
        ax_dd.fill_between(nd.dates, nd.net_drawdown * 100, 0, color="C3", alpha=0.35, step="pre")
        ax_dd.plot(nd.dates, nd.gross_drawdown * 100, color="C0", lw=0.8)
        ax_dd.set_ylabel("drawdown %")
        ax_dd.set_ylim(min(nd.net_drawdown.min(), nd.gross_drawdown.min()) * 100 * 1.1, 1)
    fig.suptitle(suptitle, fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, nav_dataframe(panels)


__all__ = ["NavDrawdown", "compute_nav_drawdown", "nav_dataframe", "plot_equity_panels"]
