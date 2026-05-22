"""Fold-aware cointegration screening with robustness selection.

The defining requirements (Phase 1 Day 6):

* The screen is WINDOWED. Phase 2's walk-forward loop calls
  :func:`cointegrate_pairs` per fold with that fold's *training* window only.
  The OLS hedge ratio (β) is fit on the training window and carried forward
  causally — never re-fit on out-of-sample data.
* Selection is **robustness**, never a p-value ranking. ADF gates only the
  unit-root rejection; the real filters are half-life-in-band, Hurst<0.5,
  and cross-window stability.
* Multiple-comparisons inflation is made visible: the funnel reports both
  raw-p<0.05 count and the BH FDR-survivor count for the SAME candidate set.
* Share-class duplicates (DVRs) are excluded from the universe before
  candidate generation — they trivially cointegrate. Holdco/subsidiary pairs
  (BAJAJFINSV/BAJFINANCE etc.) are TAGGED in the output, not removed.

Public API:
    * :func:`engle_granger`              — OLS + ADF for one direction.
    * :func:`engle_granger_best_direction` — both directions, return the more
      stable.
    * :func:`half_life_ar1`              — OU half-life from AR(1) on the spread.
    * :func:`hurst_exponent`             — variance-of-differences estimator.
    * :func:`benjamini_hochberg`         — BH FDR survivor mask.
    * :func:`is_share_class_duplicate`   — DVR / non-INE-ISIN detector.
    * :func:`cointegrate_pairs`          — the windowed entry point.

Tradeable universe = every row in the returned frame where ``fdr_pass &
half_life_in_band & hurst_pass & stable_prior_window``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl
import statsmodels.api as sm
from loguru import logger
from statsmodels.tsa.stattools import adfuller

from apt.signals.correlation import screen_pairs

# ---------------------------------------------------------------------------
# Static knowledge: share-class duplicates and holdco/subsidiary structural pairs
# ---------------------------------------------------------------------------

# Hard exclusion list — DVR / preferential / non-equity share classes whose
# price is, by construction, an affine transform of the parent equity.
SHARE_CLASS_DUPLICATES: frozenset[str] = frozenset(
    {
        "TATAMTRDVR",  # Tata Motors DVR (parent: TATAMOTORS)
    }
)

# Tag (not exclude): obvious holdco/subsidiary pairs whose cointegration is
# driven by a controlling-stake economic identity rather than a tradeable
# market dislocation. Reported separately so we can see the universe with
# and without them.
STRUCTURAL_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"BAJAJFINSV", "BAJFINANCE"}),
        frozenset({"BAJAJHLDNG", "BAJAJ-AUTO"}),
        frozenset({"BAJAJHLDNG", "BAJAJFINSV"}),
        frozenset({"BAJAJHLDNG", "BAJFINANCE"}),
        frozenset({"M&M", "M&MFIN"}),
        frozenset({"HDFC", "HDFCBANK"}),
        frozenset({"HDFC", "HDFCLIFE"}),
        frozenset({"HDFC", "HDFCAMC"}),
        frozenset({"JSL", "JSLHISAR"}),
    }
)


def is_share_class_duplicate(
    symbol: str,
    isin: str | None = None,
    *,
    blacklist: Iterable[str] = SHARE_CLASS_DUPLICATES,
) -> bool:
    """True iff ``symbol`` is a DVR / non-equity share class.

    Detection rules (any one is sufficient):
      1. Symbol appears in the hardcoded blacklist (catches future additions).
      2. Symbol ends with ``DVR``.
      3. ISIN does NOT start with ``INE`` (Indian common-equity prefix).
         ``IN9...`` etc. mark differential / preference / debenture lines.
    """
    sym = symbol.upper()
    if sym in {s.upper() for s in blacklist}:
        return True
    if sym.endswith("DVR"):
        return True
    return isin is not None and not isin.upper().startswith("INE")


# ---------------------------------------------------------------------------
# Step 1 — Engle-Granger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EGResult:
    """One direction of an Engle-Granger test.

    ``y_sym``, ``x_sym`` are leg symbols; the regression is
    ``y = alpha + beta * x + residual``. ADF is the augmented-Dickey-Fuller
    p-value on the residual under the null of unit root.
    """

    y_sym: str
    x_sym: str
    alpha: float
    beta: float
    residual: np.ndarray
    adf_pvalue: float
    adf_stat: float
    n_obs: int


def engle_granger(y: np.ndarray, x: np.ndarray, *, y_sym: str = "y", x_sym: str = "x") -> EGResult:
    """OLS ``y = α + β·x + ε`` then ADF on ε.

    Inputs are *log prices* on a common date axis (caller aligns).
    """
    if y.shape != x.shape:
        raise ValueError(f"shape mismatch: y={y.shape} x={x.shape}")
    if y.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got y.ndim={y.ndim}")
    if y.size < 20:
        raise ValueError(f"need >= 20 obs for ADF, got {y.size}")

    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit()
    alpha = float(fit.params[0])
    beta = float(fit.params[1])
    residual = np.asarray(fit.resid, dtype=float)
    # autolag='AIC' chooses lag by AIC; standard for EG residual ADF.
    adf_stat, adf_p, *_ = adfuller(residual, autolag="AIC")
    return EGResult(
        y_sym=y_sym,
        x_sym=x_sym,
        alpha=alpha,
        beta=beta,
        residual=residual,
        adf_pvalue=float(adf_p),
        adf_stat=float(adf_stat),
        n_obs=int(y.size),
    )


def engle_granger_best_direction(
    p_a: np.ndarray, p_b: np.ndarray, *, sym_a: str, sym_b: str
) -> EGResult:
    """Run EG in both directions; return whichever rejects the unit root harder.

    "More stable" = smaller ADF p-value (equivalently, more-negative ADF stat).
    Phase 2 carries the returned ``beta`` forward as the hedge ratio.
    """
    eg_ab = engle_granger(p_a, p_b, y_sym=sym_a, x_sym=sym_b)
    eg_ba = engle_granger(p_b, p_a, y_sym=sym_b, x_sym=sym_a)
    return eg_ab if eg_ab.adf_pvalue <= eg_ba.adf_pvalue else eg_ba


# ---------------------------------------------------------------------------
# Step 2 — Half-life of mean reversion from AR(1) on the spread
# ---------------------------------------------------------------------------


def half_life_ar1(spread: np.ndarray) -> float:
    """OU half-life via OLS AR(1): ``s_t = c + φ·s_{t-1} + ε_t``.

    Returns ``-ln(2)/ln(φ)`` for ``0 < φ < 1`` (mean-reverting); ``inf`` for
    ``φ ≥ 1`` (random walk or explosive); ``nan`` for ``φ ≤ 0`` (anti-persistent
    flip — not mean-reverting in the OU sense).

    Equivalent regression: ``Δs_t = c + (φ-1)·s_{t-1} + ε_t`` — same φ. We
    use the level form for readability.
    """
    s = np.asarray(spread, dtype=float)
    if s.size < 3:
        return float("nan")
    y = s[1:]
    x = sm.add_constant(s[:-1])
    fit = sm.OLS(y, x).fit()
    phi = float(fit.params[1])
    if phi <= 0:
        return float("nan")
    if phi >= 1:
        return float("inf")
    return float(-np.log(2.0) / np.log(phi))


# ---------------------------------------------------------------------------
# Step 3 — Hurst exponent (variance-of-differences)
# ---------------------------------------------------------------------------


def hurst_exponent(series: np.ndarray, *, max_lag: int = 100) -> float:
    """Hurst H from the slope of ``log σ(τ)`` vs ``log τ``.

    For a series ``x`` with stationary increments,
    ``σ(τ) = std(x[t+τ] − x[t]) ∝ τ^H``. Slope of the log–log fit over a
    range of lags ``τ ∈ [2, max_lag]`` recovers H.

    Interpretation:
      * H ≈ 0.5  — random walk (no memory)
      * H < 0.5  — mean-reverting / anti-persistent
      * H > 0.5  — trending / persistent

    Returns ``nan`` if the series is too short or degenerate.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 20:
        return float("nan")
    max_lag = max(2, min(max_lag, n // 2))
    lags = np.arange(2, max_lag + 1)
    stdevs = np.empty(lags.size, dtype=float)
    for i, tau in enumerate(lags):
        diffs = x[tau:] - x[:-tau]
        stdevs[i] = float(np.std(diffs))
    # Drop any zero-stdev lags (degenerate)
    keep = stdevs > 0
    if keep.sum() < 4:
        return float("nan")
    log_lags = np.log(lags[keep])
    log_std = np.log(stdevs[keep])
    slope, _intercept = np.polyfit(log_lags, log_std, 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Step 4 — Benjamini-Hochberg FDR control
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH step-up: return a boolean mask of survivors.

    Sort p-values ascending; let k* = largest k such that
    ``p_{(k)} <= (k/n) * alpha``; reject all hypotheses with ``p <= p_{(k*)}``.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    k_star = int(np.max(np.where(below)[0]))
    cutoff = ranked[k_star]
    return p <= cutoff


# ---------------------------------------------------------------------------
# Step 5 — windowed entry point: cointegrate_pairs
# ---------------------------------------------------------------------------


_PAIR_SCHEMA: dict[str, pl.DataType] = {
    "sym1": pl.Utf8,
    "sym2": pl.Utf8,
    "sector": pl.Utf8,
    "corr": pl.Float64,
    "direction": pl.Utf8,
    "y_sym": pl.Utf8,
    "x_sym": pl.Utf8,
    "alpha": pl.Float64,
    "beta": pl.Float64,
    "adf_stat": pl.Float64,
    "adf_pvalue": pl.Float64,
    "raw_pass": pl.Boolean,
    "fdr_pass": pl.Boolean,
    "half_life": pl.Float64,
    "half_life_in_band": pl.Boolean,
    "hurst": pl.Float64,
    "hurst_pass": pl.Boolean,
    "stable_prior_window": pl.Boolean,
    "is_structural_pair": pl.Boolean,
    "n_obs": pl.Int64,
}


@dataclass(frozen=True)
class CointegrationFunnel:
    """Pair counts at each filter — the multiple-comparisons-inflation log."""

    window_start: date
    window_end: date
    prior_window_start: date | None
    prior_window_end: date | None
    eligible_symbols: int
    excluded_share_class: int
    candidates_after_corr: int
    candidates_after_share_class: int
    sufficient_overlap: int
    raw_p_below_alpha: int
    fdr_survivors: int
    half_life_in_band: int
    hurst_below_max: int
    cross_window_stable: int | None
    tradeable: int
    structural_in_tradeable: int


@dataclass(frozen=True)
class CointegrationResult:
    """Container for the per-pair frame and the diagnostic funnel."""

    pairs: pl.DataFrame
    funnel: CointegrationFunnel
    notes: list[str] = field(default_factory=list)


def _pair_logclose_panel(
    daily: pl.DataFrame, symbols: Iterable[str], start: date, end: date
) -> pl.DataFrame:
    """Wide (date × symbol) log-close panel over ``[start, end]``.

    No drop_nulls — per-pair alignment is done at the pair level (a hole in
    one symbol shouldn't penalise pairs that don't touch it).
    """
    syms = list(symbols)
    if not syms:
        return pl.DataFrame()
    win = (
        daily.filter(
            pl.col("symbol").is_in(syms) & (pl.col("date") >= start) & (pl.col("date") <= end)
        )
        .select(["symbol", "date", "close"])
        .with_columns(pl.col("close").log().alias("log_close"))
        .sort(["date"])
    )
    if win.is_empty():
        return pl.DataFrame()
    return win.pivot(index="date", on="symbol", values="log_close").sort("date")


def _pair_aligned(panel: pl.DataFrame, sym1: str, sym2: str) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned log-close arrays for (sym1, sym2) from the wide panel."""
    if sym1 not in panel.columns or sym2 not in panel.columns:
        return np.empty(0), np.empty(0)
    two = panel.select(["date", sym1, sym2]).drop_nulls()
    if two.height == 0:
        return np.empty(0), np.empty(0)
    return two[sym1].to_numpy(), two[sym2].to_numpy()


def _test_pair_in_window(
    daily: pl.DataFrame,
    sym1: str,
    sym2: str,
    start: date,
    end: date,
    *,
    min_overlap_days: int,
) -> EGResult | None:
    """Run EG (best direction) on the pair in ``[start, end]``.

    Returns ``None`` if the pair's aligned overlap is shorter than
    ``min_overlap_days``.
    """
    panel = _pair_logclose_panel(daily, [sym1, sym2], start, end)
    p1, p2 = _pair_aligned(panel, sym1, sym2)
    if p1.size < min_overlap_days:
        return None
    return engle_granger_best_direction(p1, p2, sym_a=sym1, sym_b=sym2)


def cointegrate_pairs(
    daily: pl.DataFrame,
    sectors: pl.DataFrame,
    *,
    start: date,
    end: date,
    prior_start: date | None = None,
    prior_end: date | None = None,
    corr_threshold: float = 0.50,
    fdr_alpha: float = 0.05,
    raw_alpha: float = 0.05,
    half_life_min_days: int = 5,
    half_life_max_days: int = 60,
    hurst_max: float = 0.5,
    hurst_max_lag: int = 100,
    min_overlap_days: int = 756,
    max_internal_gap_days: int = 10,
    share_class_blacklist: Iterable[str] = SHARE_CLASS_DUPLICATES,
    structural_pairs: Iterable[Iterable[str]] = STRUCTURAL_PAIRS,
) -> CointegrationResult:
    """Windowed, FDR-corrected, robustness-selected cointegration screen.

    Pipeline (each stage's count goes into the returned funnel):

      1. **Universe filter** — drop share-class duplicates (DVRs).
      2. **Correlation pre-filter** — call :func:`screen_pairs` at
         ``corr_threshold`` on the window. Both legs must have gap-free
         history covering ``[start, end]`` (that constraint is enforced by
         the gap guard inside :func:`screen_pairs`).
      3. **Sufficient overlap** — drop pairs whose pair-wise aligned date
         intersection is shorter than ``min_overlap_days``.
      4. **Engle-Granger (both directions)** — keep the direction with the
         smaller ADF p-value. β is the hedge ratio (training-window only).
      5. **FDR (BH) across the candidate set** — record both raw-p<α and
         FDR-survivor flags for the same set so the multiple-comparison
         inflation is visible.
      6. **Robustness filters** — half-life ∈ [min, max] trading days;
         Hurst on the residual < ``hurst_max``.
      7. **Cross-window stability** — when ``(prior_start, prior_end)`` is
         supplied, the pair must also pass raw ADF p<α in the prior adjacent
         window. This is the strongest anti-overfit filter; pairs flagged
         ``stable_prior_window=False`` are typically window-specific noise.

    The returned ``pairs`` frame is the full candidate diagnostic record (one
    row per (post-share-class-filter, sufficient-overlap) candidate) — caller
    filters down to the tradeable subset via the boolean flag columns.
    """
    if start >= end:
        raise ValueError(f"start {start} must be < end {end}")
    if (prior_start is None) ^ (prior_end is None):
        raise ValueError("prior_start and prior_end must be both set or both None")
    if prior_start is not None and prior_end is not None:
        if prior_start >= prior_end:
            raise ValueError(f"prior_start {prior_start} must be < prior_end {prior_end}")
        if prior_end >= start:
            raise ValueError(
                f"prior_end {prior_end} must be < start {start} (prior must precede train)"
            )

    blacklist = {s.upper() for s in share_class_blacklist}
    structural_set: set[frozenset[str]] = {
        frozenset(p) for p in structural_pairs if len(set(p)) == 2
    }

    # ---- Stage 1: universe filter (DVR / share-class exclusion) ----
    sym_isin: dict[str, str] = (
        dict(zip(sectors["symbol"], sectors["isin"], strict=True))
        if "isin" in sectors.columns
        else {}
    )
    keep_symbols: set[str] = set()
    excluded_share_class: set[str] = set()
    for s in sectors["symbol"].to_list():
        if is_share_class_duplicate(s, sym_isin.get(s), blacklist=blacklist):
            excluded_share_class.add(s)
        else:
            keep_symbols.add(s)

    sectors_filtered = sectors.filter(pl.col("symbol").is_in(list(keep_symbols)))
    daily_filtered = daily.filter(pl.col("symbol").is_in(list(keep_symbols)))

    n_eligible = daily_filtered["symbol"].n_unique()
    logger.info(
        "Stage 1 universe: {} symbols after dropping {} share-class duplicates",
        n_eligible,
        len(excluded_share_class),
    )

    # ---- Stage 2: correlation pre-filter ----
    candidates = screen_pairs(
        daily_filtered,
        sectors_filtered,
        start=start,
        end=end,
        corr_threshold=corr_threshold,
        max_internal_gap_days=max_internal_gap_days,
    )
    # screen_pairs already operates on the share-class-filtered universe.
    # We additionally drop any pair touching an excluded symbol (defence-in-depth).
    if not candidates.is_empty():
        candidates = candidates.filter(
            ~pl.col("sym1").is_in(list(excluded_share_class))
            & ~pl.col("sym2").is_in(list(excluded_share_class))
        )
    n_after_corr = candidates.height
    logger.info(
        "Stage 2 correlation pre-filter: {} same-sector pairs at corr > {:.2f}",
        n_after_corr,
        corr_threshold,
    )

    if candidates.is_empty():
        return CointegrationResult(
            pairs=pl.DataFrame(schema=_PAIR_SCHEMA),
            funnel=CointegrationFunnel(
                window_start=start,
                window_end=end,
                prior_window_start=prior_start,
                prior_window_end=prior_end,
                eligible_symbols=n_eligible,
                excluded_share_class=len(excluded_share_class),
                candidates_after_corr=0,
                candidates_after_share_class=0,
                sufficient_overlap=0,
                raw_p_below_alpha=0,
                fdr_survivors=0,
                half_life_in_band=0,
                hurst_below_max=0,
                cross_window_stable=None if prior_start is None else 0,
                tradeable=0,
                structural_in_tradeable=0,
            ),
            notes=["no correlation-screen candidates in this window"],
        )

    # Build one wide panel for the train window; pair alignment slices per pair.
    candidate_syms = sorted(set(candidates["sym1"].to_list()) | set(candidates["sym2"].to_list()))
    panel_train = _pair_logclose_panel(daily_filtered, candidate_syms, start, end)

    if prior_start is not None and prior_end is not None:
        panel_prior = _pair_logclose_panel(daily_filtered, candidate_syms, prior_start, prior_end)
    else:
        panel_prior = None

    # ---- Stages 3–4: per-pair EG with sufficient-overlap check ----
    rows: list[dict] = []
    for r in candidates.iter_rows(named=True):
        sym1, sym2 = r["sym1"], r["sym2"]
        p1, p2 = _pair_aligned(panel_train, sym1, sym2)
        if p1.size < min_overlap_days:
            continue
        eg = engle_granger_best_direction(p1, p2, sym_a=sym1, sym_b=sym2)
        hl = half_life_ar1(eg.residual)
        hu = hurst_exponent(eg.residual, max_lag=hurst_max_lag)
        rows.append(
            {
                "sym1": sym1,
                "sym2": sym2,
                "sector": r["sector"],
                "corr": float(r["corr"]),
                "direction": f"{eg.y_sym}~{eg.x_sym}",
                "y_sym": eg.y_sym,
                "x_sym": eg.x_sym,
                "alpha": eg.alpha,
                "beta": eg.beta,
                "adf_stat": eg.adf_stat,
                "adf_pvalue": eg.adf_pvalue,
                "half_life": hl,
                "hurst": hu,
                "n_obs": eg.n_obs,
                "is_structural_pair": frozenset({sym1, sym2}) in structural_set,
            }
        )

    n_overlap = len(rows)
    logger.info(
        "Stage 3 sufficient overlap (>= {} aligned days): {} pairs",
        min_overlap_days,
        n_overlap,
    )

    if n_overlap == 0:
        return CointegrationResult(
            pairs=pl.DataFrame(schema=_PAIR_SCHEMA),
            funnel=CointegrationFunnel(
                window_start=start,
                window_end=end,
                prior_window_start=prior_start,
                prior_window_end=prior_end,
                eligible_symbols=n_eligible,
                excluded_share_class=len(excluded_share_class),
                candidates_after_corr=n_after_corr,
                candidates_after_share_class=n_after_corr,
                sufficient_overlap=0,
                raw_p_below_alpha=0,
                fdr_survivors=0,
                half_life_in_band=0,
                hurst_below_max=0,
                cross_window_stable=None if prior_start is None else 0,
                tradeable=0,
                structural_in_tradeable=0,
            ),
            notes=["no candidates had sufficient overlap in this window"],
        )

    pvalues = np.array([r["adf_pvalue"] for r in rows], dtype=float)
    raw_mask = pvalues < raw_alpha
    fdr_mask = benjamini_hochberg(pvalues, alpha=fdr_alpha)

    for i, r in enumerate(rows):
        r["raw_pass"] = bool(raw_mask[i])
        r["fdr_pass"] = bool(fdr_mask[i])
        r["half_life_in_band"] = (
            r["fdr_pass"]
            and np.isfinite(r["half_life"])
            and half_life_min_days <= r["half_life"] <= half_life_max_days
        )
        r["hurst_pass"] = (
            r["half_life_in_band"] and np.isfinite(r["hurst"]) and r["hurst"] < hurst_max
        )

    n_raw = int(raw_mask.sum())
    n_fdr = int(fdr_mask.sum())
    n_hl = sum(r["half_life_in_band"] for r in rows)
    n_hu = sum(r["hurst_pass"] for r in rows)

    logger.info(
        "Stage 4 ADF: raw p<{:.2f} = {} | BH FDR survivors @ α={:.2f} = {} (inflation gap: {})",
        raw_alpha,
        n_raw,
        fdr_alpha,
        n_fdr,
        n_raw - n_fdr,
    )
    logger.info(
        "Stage 5 robustness: half-life ∈ [{},{}] = {} | Hurst < {:.2f} = {}",
        half_life_min_days,
        half_life_max_days,
        n_hl,
        hurst_max,
        n_hu,
    )

    # ---- Stage 6: cross-window stability ----
    n_stable: int | None
    if panel_prior is not None and prior_start is not None and prior_end is not None:
        n_stable = 0
        for r in rows:
            if not r["hurst_pass"]:
                r["stable_prior_window"] = False
                continue
            p1_prev, p2_prev = _pair_aligned(panel_prior, r["sym1"], r["sym2"])
            if p1_prev.size < min_overlap_days:
                r["stable_prior_window"] = False
                continue
            eg_prev = engle_granger_best_direction(
                p1_prev, p2_prev, sym_a=r["sym1"], sym_b=r["sym2"]
            )
            stable = eg_prev.adf_pvalue < raw_alpha
            r["stable_prior_window"] = bool(stable)
            if stable:
                n_stable += 1
        logger.info(
            "Stage 6 cross-window stability ([{}, {}]): {} pairs ALSO pass raw p<{:.2f}",
            prior_start,
            prior_end,
            n_stable,
            raw_alpha,
        )
    else:
        n_stable = None
        for r in rows:
            r["stable_prior_window"] = None  # unknown when no prior window

    n_tradeable = sum(
        bool(
            r["hurst_pass"]
            and (r["stable_prior_window"] is True or r["stable_prior_window"] is None)
        )
        for r in rows
    )
    n_structural_in_tradeable = sum(
        bool(
            r["hurst_pass"]
            and (r["stable_prior_window"] is True or r["stable_prior_window"] is None)
            and r["is_structural_pair"]
        )
        for r in rows
    )

    pairs_df = pl.DataFrame(rows, schema=_PAIR_SCHEMA).sort(
        ["fdr_pass", "hurst_pass", "adf_pvalue"], descending=[True, True, False]
    )

    funnel = CointegrationFunnel(
        window_start=start,
        window_end=end,
        prior_window_start=prior_start,
        prior_window_end=prior_end,
        eligible_symbols=n_eligible,
        excluded_share_class=len(excluded_share_class),
        candidates_after_corr=n_after_corr,
        candidates_after_share_class=n_after_corr,
        sufficient_overlap=n_overlap,
        raw_p_below_alpha=n_raw,
        fdr_survivors=n_fdr,
        half_life_in_band=n_hl,
        hurst_below_max=n_hu,
        cross_window_stable=n_stable,
        tradeable=n_tradeable,
        structural_in_tradeable=n_structural_in_tradeable,
    )

    return CointegrationResult(pairs=pairs_df, funnel=funnel)
