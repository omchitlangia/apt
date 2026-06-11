"""Standard figure functions for APT research reports.

Each function:

* takes pandas DataFrame inputs (typed at the docstring level — see the
  canonical schema notes per function) plus an ``out_dir`` ``Path``,
* writes ``<name>.png`` and ``<name>.csv`` (the figure data as a flat
  table) into ``out_dir``,
* returns ``(png_path, csv_path)``.

Titles always include full cell identity when applicable
(``engine / freq / regime / cost / stop``). Library: ``matplotlib`` only.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apt.plots.style import APT_PALETTE, apply_style

# Exit-reason vocabulary — FIXED. Future units MUST use these five strings
# and only these. See docs/reporting_standard.md.
EXIT_REASON_VOCAB: tuple[str, ...] = (
    "mean_revert",
    "z_stop",
    "time_stop",
    "eod_squareoff",
    "fold_close",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(out_dir: Path, name: str, fig: plt.Figure, data: pd.DataFrame) -> tuple[Path, Path]:
    """Save fig + data with matching basename; return (png, csv)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    csv = out_dir / f"{name}.csv"
    fig.savefig(png)
    plt.close(fig)
    data.to_csv(csv, index=False)
    return png, csv


def _cell_title(
    *, engine: str, freq_min: int, regime: str, spread_bps: float, stop_mode: str
) -> str:
    return (
        f"engine={engine} | freq={freq_min}-min | regime={regime}"
        f" | cost={spread_bps} bps | stop={stop_mode}"
    )


# ---------------------------------------------------------------------------
# (a) per-pair-fold equity curves — gross and net on the SAME axes
# ---------------------------------------------------------------------------


def fig_a_per_pair_fold_equity(
    pair_sessions: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    engine: str,
    freq_min: int,
    regime: str,
    spread_bps: float,
    stop_mode: str = "none",
) -> tuple[Path, Path]:
    """One panel per (fold_id, pair); both gross and net cumulative log P&L overlaid.

    ``pair_sessions`` schema: columns ``date, fold_id, pair, gross_log_ret,
    net_log_ret`` (one row per session). Function does NOT re-derive net
    from costs — it consumes whatever ``net_log_ret`` was emitted upstream.
    """
    apply_style()
    df = pair_sessions.sort_values(["fold_id", "pair", "date"]).copy()
    groups = list(df.groupby(["fold_id", "pair"], sort=True))
    n = len(groups)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no pair-fold data", ha="center", va="center")
        return _write(out_dir, name, fig, df)

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.6 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    rows = []
    for ax, ((fid, pair), g) in zip(axes, groups, strict=False):
        d = pd.to_datetime(g["date"].astype(str))
        g_cum = g["gross_log_ret"].fillna(0).cumsum().to_numpy()
        n_cum = g["net_log_ret"].fillna(0).cumsum().to_numpy()
        ax.plot(d, np.expm1(g_cum) * 100, color=APT_PALETTE[0], lw=1.1, label="gross")
        ax.plot(d, np.expm1(n_cum) * 100, color=APT_PALETTE[3], lw=1.1, label="net")
        ax.axhline(0, color="#888", lw=0.6, linestyle="--")
        ax.set_title(f"fold {int(fid)} · {pair}", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
        rows.append(
            {
                "fold_id": int(fid),
                "pair": pair,
                "n_sessions": len(g),
                "gross_total_pct": float(np.expm1(g_cum[-1]) * 100) if len(g_cum) else 0.0,
                "net_total_pct": float(np.expm1(n_cum[-1]) * 100) if len(n_cum) else 0.0,
            }
        )
    for ax in axes[n:]:
        ax.axis("off")

    title = "Per-pair-fold equity (gross vs net) — " + _cell_title(
        engine=engine, freq_min=freq_min, regime=regime, spread_bps=spread_bps, stop_mode=stop_mode
    )
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    return _write(out_dir, name, fig, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# (b) portfolio NAV per cell — gross vs net
# ---------------------------------------------------------------------------


def fig_b_portfolio_nav(
    pair_sessions: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    engine: str,
    freq_min: int,
    regime: str,
    spread_bps: float,
    stop_mode: str = "none",
) -> tuple[Path, Path]:
    """Equal-weighted portfolio NAV (gross vs net) for one cell.

    ``pair_sessions`` schema: ``date, fold_id, pair, gross_log_ret,
    net_log_ret``. Aggregation: mean across pairs per session date —
    matches the script 13 / 15 / 15b portfolio convention.
    """
    apply_style()
    df = pair_sessions.sort_values("date").copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.text(0.5, 0.5, "no portfolio data", ha="center", va="center")
        return _write(out_dir, name, fig, df)

    port = (
        df.groupby("date", as_index=False)
        .agg(
            gross_log_ret=("gross_log_ret", "mean"),
            net_log_ret=("net_log_ret", "mean"),
            n_pairs_active=("pair", "nunique"),
        )
        .sort_values("date")
    )
    port["gross_nav"] = np.expm1(port["gross_log_ret"].cumsum()) * 100
    port["net_nav"] = np.expm1(port["net_log_ret"].cumsum()) * 100
    d = pd.to_datetime(port["date"].astype(str))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d, port["gross_nav"], color=APT_PALETTE[0], lw=1.5, label="gross NAV")
    ax.plot(d, port["net_nav"], color=APT_PALETTE[3], lw=1.5, label="net NAV")
    ax.axhline(0, color="#888", lw=0.6, linestyle="--")
    ax.set_ylabel("cumulative P&L (%)")
    ax.legend(loc="best", fontsize=9)
    ax.set_title(
        "Portfolio NAV — "
        + _cell_title(
            engine=engine,
            freq_min=freq_min,
            regime=regime,
            spread_bps=spread_bps,
            stop_mode=stop_mode,
        )
    )
    fig.tight_layout()
    return _write(out_dir, name, fig, port)


# ---------------------------------------------------------------------------
# (c) spread + Z with entry/exit/stop markers — best and worst pair-fold per cell
# ---------------------------------------------------------------------------


def fig_c_spread_z_markers(
    spread_z: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    engine: str,
    freq_min: int,
    regime: str,
    spread_bps: float,
    stop_mode: str = "none",
    pair_fold_label: str = "",
) -> tuple[Path, Path]:
    """Two-panel chart (spread + Z) with entry/exit/stop markers from ``trades``.

    ``spread_z`` schema: ``ts, spread, z``.
    ``trades`` schema: ``entry_ts, exit_ts, entry_z, exit_z, direction, exit_reason``.

    For "best/worst pair-fold per cell": caller selects the slice and the
    label; this function just renders.
    """
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    if spread_z.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _write(out_dir, name, fig, spread_z)

    sz = spread_z.sort_values("ts").copy()
    ts = pd.to_datetime(sz["ts"])
    axes[0].plot(ts, sz["spread"], color=APT_PALETTE[0], lw=0.8)
    axes[0].set_ylabel("spread (log)")
    axes[1].plot(ts, sz["z"], color=APT_PALETTE[5], lw=0.8)
    axes[1].axhline(0, color="#888", lw=0.5, linestyle="--")
    axes[1].set_ylabel("Z")

    # Markers
    color_map = {
        "mean_revert": APT_PALETTE[4],
        "z_stop": APT_PALETTE[3],
        "time_stop": APT_PALETTE[2],
        "eod_squareoff": APT_PALETTE[6],
        "fold_close": APT_PALETTE[1],
    }
    for _, tr in trades.iterrows():
        et = pd.to_datetime(tr["entry_ts"])
        xt = pd.to_datetime(tr["exit_ts"])
        ez = float(tr["entry_z"]) if pd.notna(tr["entry_z"]) else np.nan
        xz = float(tr["exit_z"]) if pd.notna(tr["exit_z"]) else np.nan
        axes[1].scatter([et], [ez], s=18, color="black", marker="o", zorder=5)
        axes[1].scatter(
            [xt],
            [xz],
            s=22,
            color=color_map.get(str(tr.get("exit_reason", "")), "k"),
            marker="x",
            zorder=5,
        )

    title = f"Spread + Z trajectory with trade markers ({pair_fold_label}) — " + _cell_title(
        engine=engine,
        freq_min=freq_min,
        regime=regime,
        spread_bps=spread_bps,
        stop_mode=stop_mode,
    )
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()

    data = trades.assign(pair_fold_label=pair_fold_label)
    return _write(out_dir, name, fig, data)


# ---------------------------------------------------------------------------
# (d) cost ladder — net Sharpe AND net total return vs cost, engines overlaid
# ---------------------------------------------------------------------------


def fig_d_cost_ladder(
    metrics: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    freq_min: int,
    regime: str,
) -> tuple[Path, Path]:
    """Two-panel: net_sharpe and net_total_pct vs spread_bps, lines per engine.

    ``metrics`` schema: ``engine, freq_min, regime, spread_bps, stop_mode,
    net_sharpe, net_total_pct``. Function filters to (freq_min, regime,
    stop_mode='none') and overlays one line per engine.
    """
    apply_style()
    df = metrics[
        (metrics.freq_min == freq_min) & (metrics.regime == regime) & (metrics.stop_mode == "none")
    ].sort_values(["engine", "spread_bps"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _write(out_dir, name, fig, df)

    colors = {"ou": APT_PALETTE[0], "rolling_z": APT_PALETTE[3]}
    for engine, sub in df.groupby("engine"):
        c = colors.get(engine, APT_PALETTE[6])
        axes[0].plot(sub.spread_bps, sub.net_sharpe, marker="o", color=c, label=engine, lw=1.4)
        axes[1].plot(sub.spread_bps, sub.net_total_pct, marker="o", color=c, label=engine, lw=1.4)

    for ax in axes:
        ax.axhline(0, color="#888", lw=0.5, linestyle="--")
        ax.set_xlabel("cost (bps)")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("net Sharpe")
    axes[1].set_ylabel("net total return (%)")
    fig.suptitle(
        f"Cost ladder — freq={freq_min}-min · regime={regime} · stop=none (engines overlaid)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    return _write(out_dir, name, fig, df)


# ---------------------------------------------------------------------------
# (e) a* vs cost per frequency — Z units AND bps
# ---------------------------------------------------------------------------


def fig_e_a_star_vs_cost(
    a_star: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Two-panel: a* in Z units (left), a* in bps of spread (right), one line per (pair, freq).

    ``a_star`` schema: ``pair, freq_min, spread_bps, a_star_z, a_star_bps``.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    if a_star.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _write(out_dir, name, fig, a_star)

    df = a_star.sort_values(["pair", "freq_min", "spread_bps"]).copy()
    for i, (key, sub) in enumerate(df.groupby(["pair", "freq_min"])):
        pair, freq = key
        c = APT_PALETTE[i % len(APT_PALETTE)]
        axes[0].plot(
            sub.spread_bps, sub.a_star_z, marker="o", color=c, lw=1.2, label=f"{pair}·{freq}m"
        )
        axes[1].plot(
            sub.spread_bps, sub.a_star_bps, marker="o", color=c, lw=1.2, label=f"{pair}·{freq}m"
        )
    axes[0].set_ylabel("a* (Z units)")
    axes[1].set_ylabel("a* (bps of log-spread)")
    for ax in axes:
        ax.set_xlabel("cost (bps)")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Bertram entry threshold a* vs cost level", fontsize=10, y=1.02)
    fig.tight_layout()
    return _write(out_dir, name, fig, df)


# ---------------------------------------------------------------------------
# (f) half-life distribution per frequency with band boundaries
# ---------------------------------------------------------------------------


def fig_f_half_life_distribution(
    hl: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    band_a: tuple[float, float] = (30.0, 120.0),
    band_b: tuple[float, float] = (120.0, 1875.0),
) -> tuple[Path, Path]:
    """One panel per ``freq_min``; histogram of HL in minutes with band edges drawn.

    ``hl`` schema: ``freq_min, half_life_min``. NaN/Inf HLs are excluded
    (and counted in the companion CSV).
    """
    apply_style()
    df = hl.dropna(subset=["half_life_min"]).copy()
    df = df[np.isfinite(df.half_life_min)]
    freqs = sorted(df.freq_min.unique().tolist())

    if not freqs:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.text(0.5, 0.5, "no HL data", ha="center", va="center")
        return _write(out_dir, name, fig, df)

    n = len(freqs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    summary_rows: list[dict] = []
    for ax, f in zip(axes, freqs, strict=False):
        sub = df[df.freq_min == f]
        vals = sub.half_life_min.to_numpy()
        if len(vals) > 0:
            ax.hist(np.log10(np.maximum(vals, 1)), bins=20, color=APT_PALETTE[0], alpha=0.85)
        for edge, lbl in [(band_a[0], "A_lo"), (band_a[1], "A_hi/B_lo"), (band_b[1], "B_hi")]:
            ax.axvline(np.log10(edge), color=APT_PALETTE[3], linestyle="--", lw=1.0)
            ax.text(
                np.log10(edge),
                ax.get_ylim()[1] * 0.9,
                lbl,
                rotation=90,
                fontsize=7,
                color=APT_PALETTE[3],
            )
        ax.set_title(f"freq = {f} min  (n={len(vals)})", fontsize=9)
        ax.set_xlabel("log10(HL, minutes)")
        ax.set_ylabel("count")
        for lo, hi, lbl in [(band_a[0], band_a[1], "A"), (band_b[0], band_b[1], "B")]:
            n_in = int(((vals >= lo) & (vals <= hi)).sum())
            summary_rows.append(
                {"freq_min": int(f), "band": lbl, "lo": lo, "hi": hi, "n_in_band": n_in}
            )
    fig.suptitle(
        "Half-life distribution per frequency (Regime A / B bands drawn)", fontsize=10, y=1.02
    )
    fig.tight_layout()
    return _write(out_dir, name, fig, pd.DataFrame(summary_rows))


# ---------------------------------------------------------------------------
# (g) drift chart — Z-OU mean per pair-fold, ±0.5 flag lines
# ---------------------------------------------------------------------------


def fig_g_drift_chart(
    drift: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    flag_lo: float = -0.5,
    flag_hi: float = 0.5,
) -> tuple[Path, Path]:
    """Horizontal bar chart of test-slice Z-OU mean per (fold_id, pair, freq_min).

    ``drift`` schema: ``fold_id, pair, freq_min, drift_mean_sigma_eq,
    [regime, traded]``. Bars colored by regime if column present; markers
    for traded vs not.
    """
    apply_style()
    df = drift.copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no drift data", ha="center", va="center")
        return _write(out_dir, name, fig, df)

    df = df.sort_values(["freq_min", "regime", "drift_mean_sigma_eq"], ascending=[True, True, True])
    df["label"] = df.apply(lambda r: f"f{int(r.freq_min)}·fold{int(r.fold_id)}·{r['pair']}", axis=1)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.18 * len(df))))
    color_map = {"A": APT_PALETTE[2], "B": APT_PALETTE[0]}
    colors = [color_map.get(str(r), APT_PALETTE[6]) for r in df.get("regime", ["?"] * len(df))]
    y = np.arange(len(df))
    ax.barh(y, df.drift_mean_sigma_eq, color=colors, alpha=0.85)
    ax.axvline(0, color="#444", lw=0.7)
    ax.axvline(flag_lo, color=APT_PALETTE[3], linestyle="--", lw=1.0)
    ax.axvline(flag_hi, color=APT_PALETTE[3], linestyle="--", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df.label.tolist(), fontsize=7)
    ax.set_xlabel("test-slice mean Z-OU (σ_eq units)")
    ax.set_title("Frozen-μ drift per pair-fold — ±0.5σ_eq flag lines drawn")
    fig.tight_layout()
    return _write(out_dir, name, fig, df.drop(columns=["label"]))


# ---------------------------------------------------------------------------
# (h) exit-reason stacked bars per cell (five-category vocabulary)
# ---------------------------------------------------------------------------


def fig_h_exit_reason_stacked(
    trades: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    group_by: tuple[str, ...] = ("engine", "freq_min", "regime"),
) -> tuple[Path, Path]:
    """Stacked bar of exit-reason counts per cell, using the FIXED 5-category
    vocabulary at :data:`EXIT_REASON_VOCAB`.

    ``trades`` must contain ``exit_reason`` plus the columns in ``group_by``.
    Unknown exit_reason values are folded into a ``__OTHER__`` row and a
    warning is logged via the DataFrame (caller can inspect the companion CSV).
    """
    apply_style()
    if trades.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.text(0.5, 0.5, "no trades", ha="center", va="center")
        return _write(out_dir, name, fig, trades)

    df = trades.copy()
    df["exit_reason"] = df["exit_reason"].astype(str)
    known = df["exit_reason"].isin(EXIT_REASON_VOCAB)
    df.loc[~known, "exit_reason"] = "__OTHER__"
    pivot = df.groupby(list(group_by) + ["exit_reason"]).size().unstack(fill_value=0)
    cats = list(EXIT_REASON_VOCAB) + (["__OTHER__"] if "__OTHER__" in pivot.columns else [])
    for c in cats:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[cats]

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(pivot)), 4))
    bottom = np.zeros(len(pivot), dtype=float)
    colors = APT_PALETTE
    labels = [" · ".join(str(x) for x in idx if x is not None) for idx in pivot.index]
    for i, c in enumerate(cats):
        ax.bar(labels, pivot[c].values, bottom=bottom, color=colors[i % len(colors)], label=c)
        bottom = bottom + pivot[c].values
    ax.set_ylabel("trade count")
    ax.set_title("Exit-reason composition per cell (5-cat vocabulary)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    return _write(out_dir, name, fig, pivot.reset_index())


# ---------------------------------------------------------------------------
# (i) trade-count comparison across engines and frequencies
# ---------------------------------------------------------------------------


def fig_i_trade_counts(
    metrics: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    regime: str | None = None,
    stop_mode: str | None = "none",
    spread_bps: float | None = 3,
) -> tuple[Path, Path]:
    """Grouped bar chart: n_trades by (engine, freq_min) at a fixed cell.

    ``metrics`` schema: ``engine, freq_min, regime, spread_bps, stop_mode, n_trades``.
    """
    apply_style()
    df = metrics.copy()
    if regime is not None:
        df = df[df.regime == regime]
    if stop_mode is not None:
        df = df[df.stop_mode == stop_mode]
    if spread_bps is not None:
        df = df[df.spread_bps == spread_bps]
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _write(out_dir, name, fig, df)

    engines = sorted(df.engine.unique().tolist())
    freqs = sorted(df.freq_min.unique().tolist())
    width = 0.8 / max(len(engines), 1)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(freqs))
    for i, e in enumerate(engines):
        sub = df[df.engine == e].set_index("freq_min").reindex(freqs)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            sub.n_trades.fillna(0).values,
            width=width,
            label=e,
            color=APT_PALETTE[i % len(APT_PALETTE)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} min" for f in freqs])
    ax.set_ylabel("n_trades")
    cell_lbl = f"regime={regime} · cost={spread_bps} bps · stop={stop_mode}"
    ax.set_title(f"Trade counts by engine × frequency ({cell_lbl})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _write(out_dir, name, fig, df)


# ---------------------------------------------------------------------------
# (j) signed (1+β)/2 histogram — all pairs vs traded survivors
# ---------------------------------------------------------------------------


def fig_j_beta_histogram(
    betas: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
    traded_mask_col: str = "traded",
) -> tuple[Path, Path]:
    """Histogram of (1+β)/2 for all pairs, overlaid with the traded-survivor subset.

    ``betas`` schema: ``pair, one_plus_beta_over_2, <traded_mask_col>``
    (boolean). Function does NOT fold over folds — if the same pair appears
    in multiple folds, dedup is the caller's job (pass the deduped table or
    one row per (fold, pair) — the histogram interprets each row equally).
    """
    apply_style()
    if betas.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return _write(out_dir, name, fig, betas)
    df = betas.copy()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(
        df["one_plus_beta_over_2"],
        bins=20,
        color=APT_PALETTE[0],
        alpha=0.6,
        label=f"all ({len(df)})",
    )
    if traded_mask_col in df.columns:
        sub = df[df[traded_mask_col].astype(bool)]
        if len(sub):
            ax.hist(
                sub["one_plus_beta_over_2"],
                bins=20,
                color=APT_PALETTE[3],
                alpha=0.8,
                label=f"traded ({len(sub)})",
            )
    ax.axvline(1.0, color="#444", lw=1.0, linestyle="--", label="β = 1 (equal-notional)")
    ax.set_xlabel("(1 + β) / 2")
    ax.set_ylabel("count")
    ax.set_title("Signed (1+β)/2 distribution — all pairs vs traded survivors")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    summary = pd.DataFrame(
        {
            "set": ["all"] + ([traded_mask_col] if traded_mask_col in df.columns else []),
            "n": [len(df)]
            + (
                [int(df[traded_mask_col].astype(bool).sum())]
                if traded_mask_col in df.columns
                else []
            ),
            "mean": [float(df["one_plus_beta_over_2"].mean())]
            + (
                [float(df.loc[df[traded_mask_col].astype(bool), "one_plus_beta_over_2"].mean())]
                if traded_mask_col in df.columns
                else []
            ),
            "median": [float(df["one_plus_beta_over_2"].median())]
            + (
                [float(df.loc[df[traded_mask_col].astype(bool), "one_plus_beta_over_2"].median())]
                if traded_mask_col in df.columns
                else []
            ),
        }
    )
    return _write(out_dir, name, fig, summary)


# ---------------------------------------------------------------------------
# (k) exclusion funnel
# ---------------------------------------------------------------------------


def fig_k_exclusion_funnel(
    funnel: pd.DataFrame,
    *,
    out_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Horizontal funnel bars per ``regime × freq_min`` (or just per stage if a
    single group).

    ``funnel`` schema: ``stage, n``, optionally also ``regime, freq_min``.
    Stage labels appear in the order given; rows are NOT re-sorted.
    """
    apply_style()
    if funnel.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.text(0.5, 0.5, "no funnel data", ha="center", va="center")
        return _write(out_dir, name, fig, funnel)

    df = funnel.copy()
    has_groups = "regime" in df.columns and "freq_min" in df.columns
    groups = list(df.groupby(["freq_min", "regime"], sort=True)) if has_groups else [(None, df)]

    n = len(groups)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, g) in zip(axes, groups, strict=False):
        stages = g["stage"].tolist()
        counts = g["n"].tolist()
        y = np.arange(len(stages))[::-1]
        ax.barh(y, counts, color=APT_PALETTE[0])
        for yi, ni in zip(y, counts, strict=False):
            ax.text(ni, yi, f" {int(ni)}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(stages, fontsize=8)
        ax.set_xlabel("n pair-folds")
        if has_groups:
            f, r = key
            ax.set_title(f"freq={int(f)} min · regime={r}", fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Exclusion funnel — pair-folds attempted → traded", fontsize=10, y=1.02)
    fig.tight_layout()
    return _write(out_dir, name, fig, df)


# ---------------------------------------------------------------------------
# Convenience: register all functions for callers that want to iterate
# ---------------------------------------------------------------------------

FIGURE_LETTERS: dict[str, str] = {
    "a": "per_pair_fold_equity",
    "b": "portfolio_nav",
    "c": "spread_z_markers",
    "d": "cost_ladder",
    "e": "a_star_vs_cost",
    "f": "half_life_distribution",
    "g": "drift_chart",
    "h": "exit_reason_stacked",
    "i": "trade_counts",
    "j": "beta_histogram",
    "k": "exclusion_funnel",
}


def all_figure_names() -> Iterable[str]:
    return list(FIGURE_LETTERS.values())


__all__ = [
    "EXIT_REASON_VOCAB",
    "FIGURE_LETTERS",
    "all_figure_names",
    "fig_a_per_pair_fold_equity",
    "fig_b_portfolio_nav",
    "fig_c_spread_z_markers",
    "fig_d_cost_ladder",
    "fig_e_a_star_vs_cost",
    "fig_f_half_life_distribution",
    "fig_g_drift_chart",
    "fig_h_exit_reason_stacked",
    "fig_i_trade_counts",
    "fig_j_beta_histogram",
    "fig_k_exclusion_funnel",
]
