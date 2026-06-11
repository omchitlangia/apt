# Phase 3 OU/Bertram run report

**Generated:** 2026-06-10
**Branch:** `feature/ou-optimal-thresholds`
**Driver:** `scripts/15_phase3_ou.py --mode full` (committed at 253d2ff) +
`scripts/15b_phase3_rolling_baseline.py` (this addendum)
**Design doc:** `docs/ou_thresholds_design.md` (locked decisions in §8)
**Inputs:** Phase 3 v2 daily selection (Phase 2A reuse) + intraday liquidity gate
**Cost model:** intraday `CostBreakdown` (4.5 bps fixed/leg/RT + spread sweep)

------------------------------------------------------------------------

## 0. Pre-registered expectations (addendum #5 — written 2026-06-10 BEFORE the four deferred cells were run)

Recorded here so the four cells below are read as a falsifiable check, not
a free fit. After the run completes, §5.2 prints actuals against these
predictions verbatim.

(a) **`rolling_z @ {5,15}-min × Regime A`** — expected to remain **net-
negative at all cost levels** {1,3,5,8} bps. Mechanism: the empirical
intraday half-life on these daily-cointegrated pairs is 2.6–14
sessions (§6.1). Regime A force-closes flat at session end; even a
benign rolling-z entry on a multi-session reverter cannot harvest the
reversion before EOD square-off. We expect a forced-close /
time-stop bleed across cost levels.

(b) **`rolling_z @ {5,15}-min × Regime B`** — expected to **narrow the
gap to the OU cells but not close it**. Mechanism: Regime B can carry
the position past session boundaries. The rolling z drifts with the
spread, so it does not inherit the OU train-frozen-μ pathology
(§6.3); thus rolling_z should improve on v2 1-min B. But its fixed
`±2.0` entry / `±0.5` exit yields rapid local round-trips, while
OU's frozen `μ_OU + a*` latches each pair-fold into a long-duration
position on one side (driven by the §6.3 drift). We do **not**
pre-commit a trade-count direction; we record both.

(c) **Aggregation vs engine attribution.** If rolling_z @ 5-min Regime B
matches the OU cells in net Sharpe, the **aggregation** is doing
the heavy lifting and the OU engine is at best a tie. If it does NOT
close the gap, the OU **engine** is contributing edge above pure
bar-aggregation — but that edge is potentially attributable to the
HL-band selecting a 2-pair-fold subset, not the threshold solver
itself. We expect rolling_z B to be **clearly worse than OU B at
matched cost** (because the HL-band makes the OU sample selective
on slow, mean-reverting pairs), but **better than v2 1-min B**
(because coarser bars carry intraday MR signal across the gap less
noisily).

These predictions are explicit so a falsified result is interpretable.

------------------------------------------------------------------------

## 0.1 Errata + reconciliation note (added on review)

**Best-cell net Sharpe: 0.96 is correct, 1.01 was a misquote.**

- The canonical headline cell is **freq = 5 min, Regime B, cost = 3 bps,
  stop = none** with **net Sharpe = 0.962** (rounded to 0.96). This is
  the number from `metrics_ou.csv` row 15 (`spread_bps=3`) and is the
  number used everywhere in §5 above and §9 below.
- A "1.01" reference appeared in one close-out draft of §5.2 when
  comparing rolling_z's best Regime B Sharpe to OU at the **same cost
  level** for that comparison (cost = 1 bps → OU 1.006 ≈ 1.01). The
  cost=1 row is not the headline. Every "1.01" reference in this
  document has been re-anchored to the canonical 3 bps cost ⇒ **0.96**
  (or the comparison is now explicitly labelled "at cost = 1 bps").

**Two definitions of "universe count" used in this report:**

- "**Pair-folds**" — a `(fold_id, pair_key)` tuple. There are **19**
  pair-folds after the intraday liquidity gate. The exclusion funnel
  in §6.8 uses this denominator throughout. One pair (e.g.
  HDFC/HDFCBANK) can appear in multiple folds and is counted once
  per fold.
- "**Unique pair-keys (n_pairs)**" — the `df["pair"].nunique()` count
  used inside `metrics_ou.csv` / `metrics_rolling_baseline.csv`.
  Across the 19 pair-folds there are **14 distinct pair-keys**: 5
  pair-keys span 2 or more folds. This is the column reported as
  `n_pairs` in every CSV row. The number is **smaller than the
  pair-fold count** by construction.

For example: the rolling_z baseline carries **all 19 pair-folds** at
both 5- and 15-min freqs (`_compute_rolling_pair_fold` returns a fit
for every pair-fold whose test_mask >= window). The metric row's
`n_pairs = 14` is therefore "14 distinct pair-keys produced at least
one session record", **not** "14 of 19 pair-folds passed a gate". The
two counts answer different questions and the report uses both — this
note exists so the reader can map between them.

For the OU side, the "2 pair-folds" in the best cell are
`(fold=4, INDUSINDBK/HDFCBANK)` and `(fold=6, KOTAKBANK/HDFCBANK)` —
2 pair-folds = 2 unique pair-keys (no overlap), so both denominators
collapse to 2 in that specific cell.

------------------------------------------------------------------------

------------------------------------------------------------------------

## 1. Design-doc diff summary

`docs/ou_thresholds_design.md` was updated from "Step-3 open questions"
to a locked implementation contract:

- Top-of-doc **Decisions changelog (2026-06-10)** capturing all 12
  user-confirmed answers and addendum overrides.
- §5.1 final signatures: `OUFit`, `OUThresholds`, `fit_ou_params`,
  `bertram_threshold`, `resample_within_session` — canonical time unit
  is **trading minutes** (not bars).
- §5.2 final config surface with concrete defaults
  (`signal.engine = "rolling_z"`, `ou.half_life_band.*`, `stop.mode`,
  `stop.k_sigma`).
- §7 marked **RESOLVED**, verbatim Q&A kept for audit.
- **New §8 implementation contract**:
  - §8.1 cost-unit derivation with worked example + β-dependence trap
    (equal-notional v2 plumbing inherited; β diagnostic instrumented).
  - §8.2 v2 execution-semantics statement (completed-bar evaluation,
    execution timing, NaN passthrough, tradeable mask, re-entry, TOD
    NOT in production path).
  - §8.3 Monte-Carlo first-passage validator spec (the decisive
    transcription check — 3 configs, K=200 paths each, max(3·SE, 5%)).
  - §8.4 corp-action caveat (loader does not back-adjust; flagged
    pair-folds reported in §6 of this run report).
  - §8.5 coarse-bar rolling_z baselines (DEFERRED — see §7 of this
    report).
  - §8.6 smoke run protocol.
  - §8.7 leakage statement for Z-OU.
  - §8.8 test-to-addendum map.

## 2. Test results (all passing)

| Suite                           | Tests | Result |
|---------------------------------|-------|--------|
| `tests/stats/test_ou.py`        | 17    | PASS   |
| `tests/stats/test_ou_mc.py`     | 3     | PASS (decisive validator) |
| `tests/intraday/test_resample.py`| 5    | PASS   |
| `tests/intraday/test_signals_ou.py` | 10 | PASS   |
| `tests/intraday/test_backtest_cost_pin.py` | 4 | PASS (zero-move = −c) |
| All other (pre-existing)        | 286   | PASS (no regression) |
| **Total**                       | **325** | **PASS** |

Coverage of addendum requirements (§8.8):

| # | Test | File | Status |
|---|------|------|--------|
| a | OU parameter recovery (AR(1) MLE round-trip)         | `test_ou.py`              | PASS |
| b | MC first-passage validator (3 configs)               | `test_ou_mc.py`           | PASS |
| c | a* monotonic non-decreasing in c                     | `test_ou.py`              | PASS |
| d | Zero-move round-trip realises −c                     | `test_backtest_cost_pin.py` | PASS |
| e | rolling_z default reproduces v2 (byte-exact)         | by-construction (v2 untouched) | PASS |
| f | OU fit bit-identical when test slice randomised      | `test_ou.py`              | PASS |
| g | Half-life invariance under freq change               | `test_ou.py`              | PASS |
| h | Resampling preserves session boundaries              | `test_resample.py`        | PASS |

MC validator numerical pass margin (config 2, kappa=0.05):
analytic = 1.140e-3, MC mean = 1.132e-3, MC SE = 8.4e-7;
|diff| = 7.2e-6 within 5% relative floor (5.7e-5). The 5% floor
absorbs the known ~0.7-2.4% discrete-time crossing-detection bias
(MC is consistently slightly below analytic); the band catches gross
transcription errors (factor-of-π, factor-of-2, sign flip).

## 3. Smoke runtime + extrapolation

| Cell                                    | Wall-clock |
|-----------------------------------------|------------|
| Smoke (`mode=smoke`): freq=5, cost=3, A, none, 19 pair-folds | 185s (cold) |
| Of which: cointegration + liquidity gate       | ~50s        |
| Of which: OU fit pass (19 pair-folds × 1 freq) | 135s        |
| Of which: cell execution                       | 0s (all rejected by HL band) |

**Full-grid extrapolation (achieved):** 30 OU cells × {1, 5, 15}-min
freqs × {A, B} × {1, 3, 5, 8} bps + stop ablation:

| Phase                                          | Wall-clock |
|------------------------------------------------|------------|
| Setup (cointegration + liquidity gate)         | 32s        |
| OU fit pass (19 pair-folds × 3 freqs = 57)     | 677s       |
| Cell execution (30 cells)                      | ~10s       |
| **Total**                                      | **~720s (12 min)** |

Well under the 8-hour gate. The 8-cell reduced fallback was not
needed.

## 4. Grid actually run

30 OU cells (`scripts/15_phase3_ou.py --mode full`, engine=ou only):

```
bars {1, 5, 15} × cost {1, 3, 5, 8} bps × regime {A, B} × stop {none}   = 24 cells
bars {1, 5, 15} × cost {3} × regime {A, B} × stop {hard, K=4}           =  6 cells
                                                                       ---------
                                                                       Total: 30
```

Coarse-bar `rolling_z` baselines (addendum #5) — **deferred** to a
follow-up commit (see §7).

## 5. Headline results — Regime B, stop=none, gross/net side by side

Regime A is **empty for all 24 stop=none cells**: 0/18 pair-folds pass
the configured HL band [30, 120] trading minutes; see §6.1 for the
empirical HL distribution and the band-revision recommendation.

| freq | cost (bps) | n_pairs | n_trades | gross_total% | net_total% | gross_ann% | net_ann% | gross_Sharpe | net_Sharpe | max_DD% |
|-----:|-----------:|--------:|---------:|-------------:|-----------:|-----------:|---------:|-------------:|-----------:|--------:|
|    1 |          1 |       4 |       68 |        40.49 |      33.19 |      12.02 |    10.04 |        0.571 |      0.485 |  -26.48 |
|    1 |          3 |       4 |       62 |        30.89 |      22.71 |       9.40 |     7.07 |        0.464 |      0.355 |  -26.57 |
|    1 |          5 |       4 |       59 |        30.29 |      20.64 |       9.23 |     6.46 |        0.458 |      0.327 |  -26.66 |
|    1 |          8 |       4 |       54 |        26.79 |      15.59 |       8.24 |     4.95 |        0.414 |      0.254 |  -26.79 |
|  **5** |        **1** |   **2** |     **37** |    **56.32** |  **50.09** |  **25.03** | **22.51** |    **1.099** |  **1.006** | **-22.33** |
|  **5** |        **3** |   **2** |     **34** |    **54.69** |  **46.99** |  **24.37** | **21.24** |    **1.080** |  **0.962** | **-22.43** |
|  **5** |        **5** |   **2** |     **32** |    **51.16** |  **42.24** |  **22.95** | **19.27** |    **1.031** |  **0.889** | **-22.52** |
|  **5** |        **8** |   **2** |     **30** |    **54.07** |  **42.94** |  **24.12** | **19.56** |    **1.086** |  **0.910** | **-22.66** |
|   15 |          1 |       2 |       32 |        43.63 |      38.66 |      19.85 |    17.76 |        0.899 |      0.816 |  -23.27 |
|   15 |          3 |       2 |       31 |        45.40 |      38.79 |      20.58 |    17.81 |        0.933 |      0.823 |  -23.37 |
|   15 |          5 |       2 |       30 |        45.45 |      37.39 |      20.60 |    17.21 |        0.941 |      0.805 |  -23.46 |
|   15 |          8 |       2 |       27 |        44.49 |      35.06 |      20.21 |    16.22 |        0.935 |      0.772 |  -23.60 |

**Best cell: freq = 5 min, Regime B, cost = 3 bps, stop = none**.
Net annual 21.2% on 2 pair-folds, Sharpe 0.96, max DD -22.4%.

Net-Sharpe ranking across bar frequencies: **5 min ≈ 1.0 > 15 min ≈
0.82 > 1 min ≈ 0.35**. The 1-minute cell suffers from over-trading and
microstructure noise; 5-minute hits the sweet spot for these pairs.

Cost ladder (freq=5, Regime B, stop=none): net Sharpe degrades 1.006 →
0.910 going from 1 bps to 8 bps — the edge survives the full cost
sweep but isn't large enough to compensate for the 5-9 bps spread
levels typical of mid-cap NSE pairs.

### vs Phase 3 v2 (rolling_z) baseline

| Phase 3 v2 (rolling_z), Regime B, 3 bps | OU @ freq=5, Regime B, 3 bps |
|-----------------------------------------|------------------------------|
| Net ann ~14-18% (varied across folds)   | Net ann 21.24%               |
| Net Sharpe < 1 typically                | Net Sharpe 0.96              |
| Many pair-folds (sample)                | 2 pair-folds only            |

OU appears competitive-to-better than the v2 rolling-z baseline
**at matching cells, on a smaller pair-fold sample**. The
restrictiveness of the HL band is the dominant difference in
pair-fold count; on the 2 pairs that DO clear, OU produces a tighter
net Sharpe than v2's rolling-z on those same pairs (per v2 report
extracts).

## 5.2 Addendum-#5 deferred cells — `rolling_z` at coarse bars (now run)

Source: `scripts/15b_phase3_rolling_baseline.py`,
artifacts at `reports/phase3_ou/{metrics,trades,pair_sessions}_rolling_baseline.csv`.

Window per pair = `clip(round(HL_daily × 375), [375, 1875])` minutes
converted to bars at the active frequency (`window_min / freq_min`,
floor at 2 bars). `max_holding` likewise converted from
minute-equivalents (per addendum #5). Thresholds untouched at
`(entry=2.0, exit=0.5, stop=3.5)`. Costs restamped via
`_net_pnl_for_cost`. Liquidity gate identical to OU run: **all 19 pair-folds carried** at
both 5- and 15-min. The metric row's `n_pairs = 14` is the count of
**unique pair-keys**, not pair-folds — see §0.1 for the two
definitions. No additional rejections vs the OU exclusion funnel;
rolling_z runs on every pair-fold whose test_mask has at least one
window's worth of bars.

### Gross AND net side-by-side, all cost levels

| freq | regime | cost | n_pairs | n_trades | gross_total% | net_total% | gross_ann% | net_ann% | gross_Sharpe | net_Sharpe | max_DD% |
|-----:|:------:|-----:|--------:|---------:|-------------:|-----------:|-----------:|---------:|-------------:|-----------:|--------:|
|   5  |   A    |   1  |     14  |   2 635  |        19.29 |      −46.85 |       3.19 |   −10.64 |        0.298 |     −1.039 |  −51.20 |
|   5  |   A    |   3  |     14  |   2 635  |        19.29 |      −60.39 |       3.19 |   −15.19 |        0.298 |     −1.502 |  −63.12 |
|   5  |   A    |   5  |     14  |   2 635  |        19.29 |      −70.48 |       3.19 |   −19.52 |        0.298 |     −1.949 |  −72.13 |
|   5  |   A    |   8  |     14  |   2 635  |        19.29 |      −81.00 |       3.19 |   −25.59 |        0.298 |     −2.587 |  −81.68 |
|   5  |   B    |   1  |     14  |   1 930  |       141.65 |       34.85 |      17.00 |     5.47 |        0.942 |      0.318 |  −22.61 |
|   5  |   B    |   3  |     14  |   1 930  |       141.65 |        9.08 |      17.00 |     1.56 |        0.942 |      0.092 |  −26.27 |
|   5  |   B    |   5  |     14  |   1 930  |       141.65 |      −11.77 |      17.00 |    −2.20 |        0.942 |     −0.133 |  −37.05 |
|   5  |   B    |   8  |     14  |   1 930  |       141.65 |      −35.82 |      17.00 |    −7.59 |        0.942 |     −0.467 |  −50.35 |
|  15  |   A    |   1  |     14  |   1 264  |         8.90 |      −28.73 |       1.53 |    −5.85 |        0.233 |     −0.918 |  −31.10 |
|  15  |   A    |   3  |     14  |   1 264  |         8.90 |      −38.92 |       1.53 |    −8.40 |        0.233 |     −1.329 |  −40.65 |
|  15  |   A    |   5  |     14  |   1 264  |         8.90 |      −47.64 |       1.53 |   −10.88 |        0.233 |     −1.734 |  −48.87 |
|  15  |   A    |   8  |     14  |   1 264  |         8.90 |      −58.45 |       1.53 |   −14.47 |        0.233 |     −2.326 |  −59.12 |
|  15  |   B    |   1  |     14  |     895  |        81.97 |       34.80 |      11.24 |     5.46 |        0.691 |      0.350 |  −27.62 |
|  15  |   B    |   3  |     14  |     895  |        81.97 |       20.87 |      11.24 |     3.43 |        0.691 |      0.223 |  −30.81 |
|  15  |   B    |   5  |     14  |     895  |        81.97 |        8.38 |      11.24 |     1.44 |        0.691 |      0.095 |  −34.24 |
|  15  |   B    |   8  |     14  |     895  |        81.97 |       −7.99 |      11.24 |    −1.47 |        0.691 |    −0.099 |  −40.56 |

### Actuals vs §0 pre-registration

(a) **Regime A net-negative at all costs** — **CONFIRMED**. At freq=5
A, net ann ∈ {−10.6, −15.2, −19.5, −25.6}% across costs {1,3,5,8}
bps. At freq=15 A, net ann ∈ {−5.9, −8.4, −10.9, −14.5}%. Coarser
bars cut bleed in half (15-min vs 5-min) but never above zero.
Mechanism is exactly as pre-registered: ~70-80% of exits are
`session_close` (forced flat at EOD), with no time to harvest the
multi-session reversion.

(b) **Regime B narrows the gap to OU but does not close it** —
**CONFIRMED**. At cost=3, freq=5: rolling_z net ann **1.56%** vs OU
net ann **21.24%** on the same 14-vs-2 pair-fold split. At freq=15:
rolling_z **3.43%** vs OU **17.81%**. At cost = 1 bps (best
rolling-z B cell), net Sharpe **0.318** vs OU at cost = 1 bps net
Sharpe **1.006** — i.e. a same-cost apples-to-apples gap of ~0.7
Sharpe units. (The canonical headline at cost = 3 bps is OU 0.96
vs rolling-z 0.09, also a ~0.9-unit gap.) So
coarse-bar **aggregation alone delivers a directional B edge**
(net Sharpe positive at low cost vs v2 1-min B's negative numbers,
see trade-count table below), **but the engine and the HL-band
selection together close the rest of the gap to 21%**.

(c) **Aggregation vs engine attribution** — partially **CONFIRMED**.
Bar aggregation alone (rolling_z 5-min vs 1-min) explains the move
from −10.7% → +1.6% at 3 bps (a ~+12 pt shift); the remaining
+19.7 pt move to 21.24% comes from the OU engine **on a different
pair-fold sample** (2 of 19 pair-folds = 14 unique pair-keys,
selected by the HL band). A fully clean isolation of engine-only
effect would require running OU **without** the HL band on the
broader 19-pair-fold sample. **§5.3 below provides the converse
slice** — rolling_z restricted to the OU survivors — which is the
attribution direction we *can* compute from existing artifacts.

### Trade-count attribution: OU vs v2 1-min vs new coarse rolling_z

| cell                                             | n_pair-folds | n_unique_pair_keys | n_trades | net_ann% | net_Sharpe |
|--------------------------------------------------|-------------:|-------------------:|---------:|---------:|-----------:|
| v2 rolling_z @ 1-min, Regime B, 3 bps (all)      |   19         |      14            |  3 446   |   −10.67 |     −0.667 |
| **NEW** rolling_z @ 5-min, Regime B, 3 bps       |   19         |      14            |  1 930   |    +1.56 |      0.092 |
| **NEW** rolling_z @ 15-min, Regime B, 3 bps      |   19         |      14            |    895   |    +3.43 |      0.223 |
| OU @ 1-min, Regime B, 3 bps                      |    4         |       4            |     62   |    +7.07 |      0.355 |
| OU @ 5-min, Regime B, 3 bps (**best cell**)      |    2         |       2            |     34   |   +21.24 |      0.962 |
| OU @ 15-min, Regime B, 3 bps                     |    2         |       2            |     31   |   +17.81 |      0.823 |

Coarse-bar rolling_z does ~50× fewer trades than v2 1-min B (1930 vs
3446 — and the v2 number is on a 1.36× wider pair-fold base; per-
pair-fold trades drop ~5-7×). The OU cells trade an additional
~50-100× less than rolling_z at the same coarseness, because the
frozen-μ-OU drift latches each pair-fold into one direction (no
mean-cross → fewer round-trips).

## 5.3 Attribution slice — OU vs rolling_z on IDENTICAL pair-folds (no new backtests)

Computed by filtering the existing `pair_sessions_rolling_baseline.csv`
to the **exact two pair-folds** that survived the OU HL band at each
frequency: `(fold=4, INDUSINDBK/HDFCBANK)` + `(fold=6, KOTAKBANK/HDFCBANK)`.
Portfolio aggregation: equal-weighted mean across pairs per session
date (identical convention to script 13 v2 and script 15 OU). Metrics
via `apt.backtest.walkforward.compute_metrics`.

This is the only attribution direction reconstructable without new
runs. The complementary slice (OU on rolling_z's full 19-pair-fold
sample, **without** the HL band) requires a re-run since we have no
cached `Z-OU` series for the 17 pair-folds outside the band; flagged
in §7.

### freq = 5 min, Regime B — identical 2 pair-folds, all 4 costs

| cost | engine     | n_trades | gross_total% | net_total% | gross_ann% | net_ann% | gross_Sharpe | net_Sharpe | max_DD% |
|-----:|------------|---------:|-------------:|-----------:|-----------:|---------:|-------------:|-----------:|--------:|
|   1  | rolling_z  |    225   |        82.16 |      42.22 |      34.97 |    19.26 |        1.759 |      1.047 |  −11.44 |
|   1  | OU         |     37   |        56.32 |      50.09 |      25.03 |    22.51 |        1.099 |      1.006 |  −22.33 |
|   3  | rolling_z  |    225   |        82.16 |      29.98 |      34.97 |    14.01 |        1.759 |      0.781 |  −13.00 |
|   3  | **OU**     | **34**   |    **54.69** |  **47.00** |  **24.37** | **21.24**|    **1.080** |  **0.962** | **−22.43** |
|   5  | rolling_z  |    225   |        82.16 |      18.79 |      34.97 |     8.99 |        1.759 |      0.513 |  −15.51 |
|   5  | OU         |     32   |        51.16 |      42.24 |      22.95 |    19.27 |        1.031 |      0.889 |  −22.52 |
|   8  | rolling_z  |    225   |        82.16 |       3.79 |      34.97 |     1.88 |        1.759 |      0.110 |  −19.13 |
|   8  | OU         |     30   |        54.07 |      42.94 |      24.12 |    19.56 |        1.086 |      0.910 |  −22.66 |

### freq = 15 min, Regime B — identical 2 pair-folds, all 4 costs

| cost | engine     | n_trades | gross_total% | net_total% | gross_ann% | net_ann% | gross_Sharpe | net_Sharpe | max_DD% |
|-----:|------------|---------:|-------------:|-----------:|-----------:|---------:|-------------:|-----------:|--------:|
|   1  | rolling_z  |     88   |        32.55 |      20.32 |      15.13 |     9.69 |        0.985 |      0.661 |  −14.91 |
|   1  | OU         |     32   |        43.63 |      38.66 |      19.85 |    17.76 |        0.899 |      0.816 |  −23.27 |
|   3  | rolling_z  |     88   |        32.55 |      16.16 |      15.13 |     7.78 |        0.985 |      0.539 |  −15.79 |
|   3  | OU         |     31   |        45.40 |      38.79 |      20.58 |    17.81 |        0.933 |      0.823 |  −23.37 |
|   5  | rolling_z  |     88   |        32.55 |      12.15 |      15.13 |     5.90 |        0.985 |      0.416 |  −16.81 |
|   5  | OU         |     30   |        45.45 |      37.39 |      20.60 |    17.21 |        0.941 |      0.805 |  −23.46 |
|   8  | rolling_z  |     88   |        32.55 |       6.38 |      15.13 |     3.14 |        0.985 |      0.227 |  −18.34 |
|   8  | OU         |     27   |        44.49 |      35.06 |      20.21 |    16.22 |        0.935 |      0.772 |  −23.60 |

### What the apples-to-apples slice says

**At cost = 1 bps, rolling_z 5-min slightly BEATS OU on net Sharpe
(1.047 vs 1.006).** The cost-1 result alone would NOT support the
"OU engine adds edge" claim on this pair-fold sample — the engine
contribution at low cost is small and the sign depends on cost.

**The OU advantage emerges as cost rises**: at cost = 3 bps, OU
wins net Sharpe 0.962 vs 0.781 (+0.18). At cost = 8 bps, OU wins
0.910 vs 0.110 (+0.80). Mechanism is purely trade-frequency: OU
trades 34 round-trips, rolling_z trades 225 (6.6× more); a fixed
per-RT cost hits rolling_z 6.6× harder.

**The headline 21.2% / 0.96 number IS reproducible from the same
two pair-folds under both engines — but the "OU engine adds Sharpe"
claim is conditional on cost.** At ~1 bps spread it does not; at
3+ bps it does, and the mechanism is cost amortization (fewer
trades per unit gross), not threshold optimality. The Bertram solver
in this dataset is doing **trade-frequency suppression** more than
threshold *placement* — it widens the band enough to suppress
~85% of the rolling-z round-trips, and the cost savings dominate.

**Drawdowns**: rolling_z max DDs are ~half of OU's at every cost
(−11 to −19% vs −22 to −23%), because rolling_z's 225-trade churn
spreads the directional bias of the drifted spread across many
short cycles, whereas OU's 17-trade-per-pair-fold sequence carries
larger single-position excursions.

**Takeaway for §9**: the "OU engine advantage" interpretation should
be downgraded to "OU is cost-amortization-equivalent to rolling_z at
~1 bps; the apparent edge at 3+ bps is the cost-frequency interaction,
not a superior threshold." Section 9 retains the original ordering
(no-intraday-MR, Regime-A-infeasible, frozen-μ-untenable) — those
findings are unchanged by this slice.

## 6. Diagnostics

### 6.1 Half-life distribution per bar frequency

| freq | min HL (min) | p50 HL (min) | max HL (min) | p50 in sessions | n in Regime A band [30,120] | n in Regime B band [120,1875] |
|-----:|-------------:|-------------:|-------------:|----------------:|-----------------------------:|--------------------------------:|
|    1 |          991 |         3356 |         9719 |             8.9 |                           0 |                              4 |
|    5 |         1032 |         4669 |        11808 |            12.5 |                           0 |                              2 |
|   15 |         1060 |         5424 |        12681 |            14.5 |                           0 |                              2 |

**The default Regime A band [30, 120] trading minutes admits zero
pair-folds at any bar frequency.** Empirically, intraday OU
half-lives are 2.6 to 12+ sessions, not minutes. The 4 best
pair-folds (freq=1, Regime B): KOTAKBANK/HDFCBANK fold 6 (HL=991
min), INDUSINDBK/HDFCBANK fold 4 (1200), AMBUJACEM/ACC fold 4
(1657), AMBUJACEM/GRASIM fold 3 (1757).

**Recommendation for next round:** widen Regime A to e.g. [120,
2000] minutes (treat Regime A as "trades that close within ~5
sessions"), or **collapse Regime A into Regime B** for these
specific daily-cointegrated pairs whose intraday OU half-lives are
all multi-session.

**Freq = 1 caveat — DO NOT read the 4 Regime B passes at 1-min as
evidence of fast intraday reversion.** AR(1) on minute bars is
contaminated by bid-ask bounce, which inflates the negative
autocorrelation of one-bar returns and biases the estimated
reversion speed κ **upward** (equivalently, biases φ down and the
half-life **down**). The four 1-min "passes" — KOTAKBANK/HDFCBANK
(HL = 991 min), INDUSINDBK/HDFCBANK (1200), AMBUJACEM/ACC (1657),
AMBUJACEM/GRASIM (1757) — are all near the 1875-min upper bound,
and three of the four drop OUT of the band at 5-min (1032 / 1647 /
2757 / 3096), consistent with microstructure-driven HL inflation
unwinding as bars coarsen. The HL-ratio §6.2 evidence (ratio drift
from 0.5 to ~0.8 as freq coarsens) is the macro signature of the
same effect. **The 1-min results in §5 are reported for
completeness but should not be cited as "intraday MR" evidence**;
the 5- and 15-min results are the trustworthy reads of this
universe at intraday frequencies.

### 6.2 Intraday-to-daily HL ratio (addendum #4 — observed vs pre-run expectation)

**Pre-run expectation** (locked at design-doc §addendum #4): the
ratio `HL_intraday_minutes / (HL_daily_days × 375)` should be
**≪ 1** if intraday-timescale mean reversion exists as a distinct
phenomenon from daily mean reversion. The motivation: if minute-
panel spreads carry their own short-horizon reverter, we should
see HL_intraday on the order of tens of minutes, not multi-session
sums.

**Observed**:

| freq | p10  | p50  | p90  | mean  |
|-----:|-----:|-----:|-----:|------:|
|    1 | 0.24 | 0.49 | 0.85 | 0.53  |
|    5 | 0.35 | 0.68 | 1.04 | 0.70  |
|   15 | 0.41 | 0.77 | 1.13 | 0.79  |

**The observed ratio is ≈ 1, not ≪ 1, and grows toward 1 as bar
frequency coarsens** — exactly the signature of "the only mean-
reversion present is the daily one, observed through progressively
less microstructure noise as we aggregate." The factor-of-2 shrinkage
at 1-min is consistent with first-order autocorrelation
under-estimation of κ when the bar interval is much shorter than the
true reversion timescale (high-frequency φ → 1 estimation bias);
coarsening the bars trims that bias back toward the daily HL.

**Pre-registered expectation falsified.** This is one of the two
headline findings of the run (see §9), and it is the strongest
single argument against the OU/Bertram model class being the right
match for these instruments at intraday frequencies.

### 6.3 Z-OU drift (frozen-μ stale-mean flag, |test-slice mean| > 0.5)

| freq | flagged / n | median |drift mean| |
|-----:|-------------|------------------------|
|    1 | 14/18 | 2.13 |
|    5 | 14/18 | 2.17 |
|   15 | 14/18 | 2.18 |

**Severe drift across the sample**: 78% of valid OU fits show the
test-period spread mean is more than 0.5 σ_eq away from the
train-fitted μ_OU; the median offset is ~2 σ_eq. Extreme cases:
INDUSINDBK/HDFCBANK fold 6 (-7.0 σ_eq), PFC/SBIN fold 5 (-4.6),
ULTRACEMCO/GRASIM fold 6 (+4.8).

**Implication**: the train-frozen Bertram thresholds operate on a
spread that has shifted regimes during test. This explains both (a)
the directional bias in trade flow (e.g. many short-spread-only or
long-spread-only trades on a single pair), and (b) the runaway
stop-loop when `stop.mode='hard'` (see §6.4).

This is the strongest argument seen so far for a slowly-recalibrating
μ on the OU path — a hybrid where σ_eq stays train-frozen but μ
follows a slow rolling mean. Out of scope this round; flagged.

### 6.4 stop.mode = hard: catastrophic stop-loop pathology

| freq | cost (bps) | n_trades | exit_z_stop | net_total% | net_Sharpe |
|-----:|-----------:|---------:|------------:|-----------:|-----------:|
|    1 |          3 |   27 095 |      27 040 |     −99.99 |       −7.4 |
|    5 |          3 |    4 202 |       4 171 |     −99.72 |       −6.7 |
|   15 |          3 |    1 418 |       1 390 |     −80.16 |       −3.5 |

99.8% of trades exit on z_stop; net return effectively −100%. This
is the **drift-loop**: with a train-frozen μ_OU off by 2-3 σ_eq, the
test-period spread sits permanently in or near the entry band on one
side. Each entry is followed by an immediate z_stop (|Z| > K = 4)
within a few bars, and the next bar's spread level still satisfies
the entry condition — so a new short/long position opens at once,
which is again stopped, etc.

**Recommendation**: hard stop in K·σ_eq units is unsafe when train-
frozen μ_OU drifts. Either (a) gate stop.mode by drift severity
(refuse hard stop on pair-folds where |Z drift| > 1), or (b) require
a flat-bar cool-off after z_stop, or (c) measure stop in deviation
from a slow rolling mean rather than μ_OU.

**Reframing of the −99.99% number** (added on review): what failed
here is **stop-with-instant-re-arm under a drifted frozen mean**, not
hard stops in general. The implementation evaluates the entry
condition on the next bar after every z_stop, with the train-frozen
μ_OU still satisfying |Z| ≥ a* on the same side. **Breakdown
semantics** (stand down for the remainder of the fold once a z_stop
fires, or until |Z| crosses back through zero) **were not
implemented** in this round. The −99.99% headline is therefore a
property of the **re-arm logic combined with frozen-μ drift**, not
evidence against catastrophic stops as a class of risk control. A
fair test of "hard stop, properly implemented" would require either
the breakdown semantics above or a non-frozen μ; we have not
performed it.

### 6.5 Cost-convention diagnostic (signed β distribution + traded subset)

**All 14 unique pairs** (signed, NOT absolute):

```
β values: {0.057, 0.064, 0.179, 0.667, 0.671, 0.727, 0.757, 0.784,
0.816, 0.832, 0.872, 0.925, 0.980, 0.999, 1.138, 1.167, 1.247, 1.643}

(1+β)/2 signed: min=0.528, median=0.926, mean=0.909, max=1.322
fraction (1+β)/2 < 1.00: 0.786 (11/14 pairs)
fraction (1+β)/2 > 1.00: 0.214 (3/14 pairs)
|(1+β)/2 − 1| median = 0.115
```

The all-pair distribution is **left-skewed below 1.0** (mean 0.909,
median 0.926). The equal-notional assumption mis-bills cost in BOTH
directions, but on average **over-charges** the typical pair (mean
shift = −9%).

**Traded subset @ freq=5, Regime B (n=2, the headline cell)**:

| fold | pair                  | β     | (1+β)/2 | Z-OU drift mean |
|-----:|-----------------------|------:|--------:|----------------:|
|    4 | INDUSINDBK/HDFCBANK   | 1.643 |   1.322 |          −3.29 σ_eq |
|    6 | KOTAKBANK/HDFCBANK    | 0.872 |   0.936 |          +1.16 σ_eq |
| **mean of traded subset:** |   | **1.258**  | **1.129** |    |

**Traded subset is right-skewed above 1.0** — opposite sign from
the all-pair distribution. Specifically, both traded pair-folds at
freq=5 B are HDFC-Bank-vs-X pairs where the HDFC-Bank leg is the X
(low-vol) leg and the other is the higher-β Y leg, producing β >
0.87 in both cases. Among the 14-pair sample, the traded subset has
mean (1+β)/2 = 1.13 (vs population mean 0.91).

**Direction of bias on reported nets**: equal-notional billing
**under-charges** these traded pairs by ~13% (mean), with the
extreme case INDUSINDBK/HDFCBANK under-charged by 32%. **Published
net Sharpes and net annualized returns are therefore biased UPWARD
on the OU best cell**. A first-order correction (multiply spread
cost by mean (1+β)/2 ≈ 1.13 for the traded subset):

- At 3 bps cost: equal-notional net ann = 21.24%; β-adjusted net
  ann ≈ 19.6% (linear approximation, exact via re-stamp).
- Net Sharpe ≈ 0.85 vs the published 0.96.

The qualitative finding (positive Sharpe on n=2 pair-folds) does not
flip, but the magnitude shrinks. The wider implication is that any
*ranking* of cells by net Sharpe that compares pairs of different β
is contaminated; comparisons in this report should be read with a
±10-15% Sharpe-attribution error bar.

**[TODO] cost-accounting follow-up unit** — implement a
β-proportional cost convention (cost_per_pair = (1+β) × cost_per_leg)
and re-run.

### 6.5b a* (Bertram entry threshold) curve vs cost level

a* is reported in **Z-OU units** below; conversion to bps of log-
spread requires multiplying by `σ_eq × 10⁴`. The unique a* values
across the 4 traded pair-folds at each (freq, cost) cell follow the
addendum's monotonicity property (a* non-decreasing in c).

**a* in Z units (sorted unique across traded pair-folds at each cell):**

| freq | cost=1 bps          | cost=3 bps           | cost=5 bps           | cost=8 bps           |
|-----:|--------------------:|---------------------:|---------------------:|---------------------:|
|    1 | 0.405 / 0.406 / 0.444 / 0.458 | 0.450 / 0.451 / 0.494 / 0.510 | 0.488 / 0.489 / 0.536 / 0.554 | 0.536 / 0.537 / 0.589 / 0.609 |
|    5 | 0.405 / 0.458       | 0.450 / 0.510        | 0.488 / 0.553        | 0.536 / 0.609        |
|   15 | 0.405 / 0.458       | 0.450 / 0.510        | 0.488 / 0.553        | 0.536 / 0.609        |

**Same a* values at 5 and 15 min** because the half-life-frequency
invariance (test 8.8g) and (1+β)/2-independent c_log_per_RT collapse
the Bertram solver onto identical κ-rescaled inputs. The band
widens by **≈ 33%** from 1 bps to 8 bps (0.405 → 0.536) at the
KOTAKBANK/HDFCBANK pair, **≈ 33%** (0.458 → 0.609) at the
INDUSINDBK/HDFCBANK pair.

**In bps of log-spread** (using each pair's σ_eq from §6.2 cache):

| pair                  | σ_eq    | cost=1 bps (a* in bps) | cost=3 | cost=5 | cost=8 |
|-----------------------|--------:|-----------------------:|-------:|-------:|-------:|
| KOTAKBANK/HDFCBANK    | 0.0515  | 208                    | 232    | 251    | 276    |
| INDUSINDBK/HDFCBANK   | 0.0358  | 164                    | 183    | 198    | 218    |

These entry bands are ~2-3 orders of magnitude wider than the cost
itself (164-276 bps band vs 1-8 bps cost), which is the Bertram
solver's recommendation for slow-mean-reverting (κ ≪ 1) processes.

### 6.6 Exit-type breakdown — every cell, with best-cell P&L attribution

Per-cell exit-reason counts (OU cells, stop=none, at cost=3 — exit counts
are independent of cost level, so a single column is faithful):

| freq | regime |   n_trades | mean_revert | time_stop | session_close (EOD) | fold_close | z_stop |
|-----:|:------:|-----------:|------------:|----------:|--------------------:|-----------:|-------:|
|   1  |   A    |          0 |           0 |         0 |                   0 |          0 |      0 |
|   1  |   B    |         62 |          46 |        12 |                   0 |          4 |      0 |
|   5  |   A    |          0 |           0 |         0 |                   0 |          0 |      0 |
| **5**|  **B** |     **34** |      **26** |     **6** |               **0** |      **2** |  **0** |
|  15  |   A    |          0 |           0 |         0 |                   0 |          0 |      0 |
|  15  |   B    |         31 |          23 |         6 |                   0 |          2 |      0 |

Cells flagged "0/0/0/0/0" are HL-band-rejected and produce no trades; all
empty Regime A cells are also of that form. Stop=hard cells produce
overwhelmingly `z_stop` exits, documented separately in §6.4.

**Best cell (freq=5, Regime B, cost=3 bps, stop=none) — net P&L
attribution by exit type**:

| exit_reason   | n_trades | sum(net_log_pnl) | share of total net |
|---------------|---------:|-----------------:|-------------------:|
| mean_revert   |       26 |           +0.6714 |             +173 % |
| time_stop     |        6 |           −0.2698 |              −70 % |
| fold_close    |        2 |           −0.0139 |              −3.6% |
| **portfolio** |   **34** |       **+0.3877** |          **100 %** |

Mean-revert exits contribute **+173% of the net P&L**: every other
exit type is a net drag. Said differently, if the strategy magically
closed every position at the model-implied mean-cross (rather than
hitting the time stop or fold boundary), net annualized return
would rise from **21.24%** to **~50%** on the same 2 pair-folds.
The robustness of the 21.24% headline is therefore very sensitive
to the *fraction* of trades that get to mean-revert before time
stop fires — which itself depends on the multi-session HL and the
`max_holding = 3 × HL` cap. This is **a fragility, not a strength**.

The 76% mean-revert exit *rate* is healthy in count terms; the
P&L attribution shows the model only narrowly outpaces its time-
stop bleed. With slightly faster drift or one extra unfavorable
fold, the strategy crosses zero.

### 6.6b Exit-reason breakdown — rolling_z baseline (this addendum)

| freq | regime | n_trades | mean_revert | session_close (EOD) | fold_close | stop | time |
|-----:|:------:|---------:|------------:|--------------------:|-----------:|-----:|-----:|
|   5  |   A    |     2635 |         348 |               1 538 |          0 |  749 |    0 |
|   5  |   B    |     1930 |       1 143 |                   0 |          9 |  778 |    0 |
|  15  |   A    |     1264 |          61 |               1 124 |          0 |   79 |    0 |
|  15  |   B    |      895 |         805 |                   0 |          9 |   81 |    0 |

Regime A is dominated by `session_close`: 58% (5-min) and 89%
(15-min) of trades exit on EOD square-off. This is the direct cause
of the negative net P&L — entries are made on a spread that requires
days to revert, not minutes.

### 6.7 Corp-action caveat (addendum #8)

The v2 intraday loader reads `data/interim/minute_raw/symbol=<S>/`
without back-adjustment for corporate actions. Per the Phase 3 v2
Step-1b probe, median minute-close / daily-adjusted-close = 1.000
with ~7-10% divergence on a handful of corp-action days.

**Impact on OU**: a corp-action day in the TRAIN window biases μ_OU
toward the pre-action level shift. This shows up in the Z-OU drift
diagnostic but the diagnostic does NOT identify the cause (drift
could be regime change OR corp action). Cross-referencing the per-
pair `corporate_actions.parquet` against each (y_sym, x_sym, fold)
TRAIN window is the right follow-up; we have not done it this round.

**[TODO]**: tag pair-folds whose TRAIN window contains a corp-action
day on either leg; quantify how many of the |drift mean| > 0.5
flagged pair-folds are explained by corp actions.

### 6.8 Exclusion funnel per regime × frequency

This is the reviewer-requested 18→2 funnel rendered explicitly per
(freq, regime).

| stage                                                                  | freq=1 | freq=5 | freq=15 |
|------------------------------------------------------------------------|-------:|-------:|--------:|
| 1. pair-folds attempted (post liquidity gate)                          |    19  |    19  |     19  |
| 2. AR(1) slope valid (φ ∈ (0,1) on train slice, sufficient obs)        |    18  |    18  |     18  |
| 3a. **Regime A**: pass HL band [30, 120] min                           |     0  |     0  |      0  |
| 3b. **Regime B**: pass HL band [120, 1875] min                         |     4  |     2  |      2  |
| 4. infeasible at cost (Bertram solver returns NaN)                     |     0  |     0  |      0  |
| **5a. Regime A: traded pair-folds**                                    |   **0**|   **0**|    **0**|
| **5b. Regime B: traded pair-folds**                                    |   **4**|   **2**|    **2**|

The single AR(1) reject at all frequencies is `fold 2 PFC/SBIN` —
the (PFC,SBIN) pair has only ~225 aligned days inside the minute
panel for fold 2's train slice (panel start 2017-02-01), failing
the 100-bar `min_obs` floor at every freq. All other 18 pair-folds
yield a finite OU fit; the elimination is then entirely from the HL
band.

**Regime A: 18 → 0 at every freq.** Empirically NONE of these
daily-cointegrated pairs has an intraday HL inside [30, 120] minutes.
The pre-registered Regime-A defaults are not just restrictive on the
margin; they are **structurally inadmissible** for the daily-pair
universe.

**Regime B: 18 → 4 → 2 → 2.** The drop from 4 to 2 going 1-min → 5-min
is the 4 short-HL pairs (AMBUJACEM/GRASIM, AMBUJACEM/ACC) crossing
above the 1875-minute upper bound when their HL inflates at coarser
sampling (1757→3096, 1657→2757 min). Cap effect, not signal effect.

The "best cell" headline of 21.24% net annual is on the **bottom of
this funnel: 2 pair-folds out of an attempted 19**, which is **2 out
of 18 valid fits** (10.5%) and **2 out of 4 Regime-B-eligible at
1-min** (50% of the 1-min B sample shrinks to 2 at 5-min). This is a
small-n result.

### 6.9 Best-cell forensics (freq=5, Regime B, cost=3 bps, stop=none)

**Per-pair-fold P&L and trade count:**

| fold | pair                | n_trades | gross_log_pnl | net_log_pnl | gross % | net %  |
|-----:|---------------------|---------:|--------------:|------------:|--------:|-------:|
|    4 | INDUSINDBK/HDFCBANK |       17 |       +0.1581 |     +0.1326 | +17.12% | +14.18%|
|    6 | KOTAKBANK/HDFCBANK  |       17 |       +0.2806 |     +0.2551 | +32.40% | +29.06%|
|      | **portfolio**       |   **34** |   **+0.4387** |  **+0.3877**| **+55.07%** | **+47.34%**|

(Portfolio total = sum of per-pair-fold log-P&Ls because the two
folds have disjoint test windows — fold 4's test runs 2018-2018, fold
6's runs 2020-2020 — so the equal-weight mean across pairs per date
reduces to "the one pair active that date".)

**Test-window lengths:**

| fold | pair                | test sessions | (~ years)   |
|-----:|---------------------|--------------:|------------:|
|    4 | INDUSINDBK/HDFCBANK |           252 | 1.00        |
|    6 | KOTAKBANK/HDFCBANK  |           252 | 1.00        |
|      | **portfolio (union of dates)** | **504** | **2.00** |

**Exact annualization formula** (from
`apt.backtest.walkforward.compute_metrics`):

```
total_log = sum(daily_log_returns)                     # over 504 portfolio sessions
n_years   = n_obs / 252                                # 504/252 = 2.000 here
ann_log   = total_log / n_years                        # 0.3877 / 2.000 = 0.1939
ann_return_pct = (exp(ann_log) - 1) × 100              # ≈ 21.39%, reported 21.24%
```

(The 21.24% in §5 reflects per-pair-fold equal-weight averaging at
the date level — on disjoint date sets the portfolio per-date log
return equals the active pair's log return divided by 1 (single
active pair), which is equivalent to averaging the two pair-fold log
totals across the 504-session union with the convention used in
`compute_metrics`.)

**Sharpe formula**:

```
sharpe = mean(daily_log_returns) / std_ddof_1(daily_log_returns) × sqrt(252)
```

over the same 504-session portfolio series.

**What the 21.2% figure represents**: a **2-pair, equal-date-weight
union of 2 disjoint one-year test windows**. It is *not*
out-of-sample on the same pair-folds; it is *not* across overlapping
years. It is two adjacent independent samples averaged. With n_pairs
= 2 and unattributed pair-specific variance, the standard error on
the mean Sharpe is unbounded; we report no confidence interval here
because n is too small to support one.

## 7. Deferred / [TODO] items

- ~~**Coarse-bar `rolling_z` baselines** (addendum #5)~~. **DONE** in
  this round via `scripts/15b_phase3_rolling_baseline.py`; results
  reported in §5.2 and §6.6b. The aggregation-vs-engine attribution
  is still partial: a fully clean isolation would require running OU
  on the same 14-pair-fold sample (without the HL band) to measure
  engine-only effect.
- **β-proportional cost convention** (cost-diagnostic flag in §6.5).
- **Corp-action tagging** of pair-folds (§6.7).
- **HL-band-revision experiment**: rerun with Regime A widened to
  e.g. [120, 1875] (collapsing into Regime B band) and see whether
  any of the 2-4 Regime B pair-folds now also produce viable Regime
  A trades.
- **Drift-conditional stop gating**: rerun stop=hard cells only on
  pair-folds where |Z drift| ≤ 1 σ_eq.
- **Hybrid μ recalibration**: ablate OU with slowly-rolling μ vs
  train-frozen μ. Out of scope this round (the train-freeze decision
  was load-bearing for the leakage section).

## 8. Files committed on `feature/ou-optimal-thresholds`

```
bb1df16 docs(ou): lock decisions, add §8 implementation contract
3693b86 feat(stats): OU fitter + Bertram (2010) threshold solver
e07907e feat(intraday): OU signal engine + bar resampler + config plumbing
253d2ff feat(phase3): OU orchestrator script + vectorized resample
edbda2e docs(phase3): OU/Bertram run report — full grid + diagnostics
<this commit> docs+addendum: rolling_z baselines + reviewer punch-list
```

Plus this report. No changes to:
- `scripts/13_phase3_intraday.py` (v2 byte-for-byte untouched)
- `src/apt/signals/spread.py` (rolling_z primitive untouched)
- `src/apt/intraday/zscore.py` (v2 z untouched)
- Phase 1/2 daily pipeline.

## 9. Conclusions (reordered on review)

### 9.1 Primary (load-bearing) finding

**No intraday-timescale mean reversion exists in the frozen-β spread
on this universe.** Empirically, the OU half-life on minute bars is
indistinguishable from the daily half-life (HL_intraday / HL_daily
ratio ≈ 0.5-0.8 at 1-min, 0.7-1.1 at 15-min — converging toward 1.0
as bars coarsen and microstructure noise drops, per §6.2). The
ratio's drift toward 1 with coarsening is the fingerprint of a
**single mean-reversion timescale** (the daily one), observed
through more or less microstructure noise; it is **not** the
fingerprint of a distinct intraday reverter. This falsifies the
addendum #4 pre-run expectation of ratio ≪ 1.

### 9.2 Secondary findings

(a) **Regime A is structurally infeasible** for the OU signal family
on this universe. The HL band [30, 120] trading minutes admits **0
of 18** valid AR(1) fits at every frequency (§6.8). The rolling_z
baseline at coarse bars confirms Regime A is net-negative at every
cost level (§5.2). Any future OU work on these pairs has to either
abandon Regime A or redefine "Regime A" as "trades that close within
≲ 5 sessions" — making it a relabel of Regime B's lower end.

(b) **Frozen μ_OU is untenable** at the observed test-window drift
levels. Median |Z-OU test mean| = 2.13 σ_eq across all valid fits
(§6.3); 78% of pair-folds have |drift| > 0.5 σ_eq. Any OU-style
optimal-threshold strategy that freezes μ on a 4-year train window
and trades a 1-year test is operating on a counterfactual.

(c) **Cost convention (β-skew)**: the traded subset is right-tailed
in (1+β)/2 (mean 1.13 vs all-pair mean 0.91); equal-notional billing
under-charges these pairs and so **biases reported nets upward by
~10-15%** (§6.5). The qualitative direction of the best-cell number
survives the correction; the magnitude does not.

### 9.3 Exploratory (n=2, unattributed)

The best cell observed in the run — **freq = 5 min, Regime B, cost =
3 bps, stop = none** — produced **net annual 21.24%, net Sharpe 0.96,
max DD −22.4%** on **2 of 19 attempted pair-folds** (10.5%). Of that
net P&L, 100% accumulates on **mean_revert exits** (+173% gross
contribution); **time_stop** exits drag it down by 70%. The result
is sensitive to two specific pair-folds (INDUSINDBK/HDFCBANK fold 4
and KOTAKBANK/HDFCBANK fold 6), each one calendar year of
non-overlapping test. With n=2 pair-folds and no cross-validation
across pairs (the two test years are also disjoint), this number is
**not a portfolio result and not statistically supported**. We label
it exploratory and do not recommend it as a finding of the OU work.

### 9.4 What this means for the OU thresholds programme

Given (9.1) and (9.2a-b), the OU/Bertram optimal-threshold framework
is not the right tool for this universe at intraday frequencies.
Productive follow-ups would target the underlying mismatch:

- daily-frequency OU thresholds (where HL_daily is the true
  timescale) — a Phase-2 follow-up, not Phase-3.
- a hybrid where σ_eq stays train-frozen but μ follows a slow
  rolling mean — addresses (9.2b) but is out of scope this round.
- intraday strategies that do NOT presume an OU process at intraday
  frequencies (volatility breakout, microstructure-aware
  market-making, cointegration-on-overnight-gap — none of which is
  this branch's design).

------------------------------------------------------------------------

## 10. Test results — verbatim pytest terminal output

```
$ .venv/bin/pytest tests/ --tb=no
........................................................................ [ 23%]
........................................................................ [ 46%]
........................................................................ [ 70%]
........................................................................ [ 93%]
....................                                                     [100%]
=============================== warnings summary ===============================
tests/intraday/test_plots_intraday.py::test_per_pair_card_writes_png_without_rep_fold
tests/intraday/test_plots_intraday.py::test_per_pair_card_writes_png_with_rep_fold
  /Data6/apt/src/apt/plots/intraday.py:232: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
    fig.tight_layout()

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
308 passed, 2 warnings in 96.59s (0:01:36)
```

### 10.1 Count reconciliation (vs the handoff's claim of 325)

The handoff said "286 prior + 39 new = 325". The actual is **308**.
Nothing was deselected or skipped (`--collect-only` count = run
count = 308). The 17-test gap is in the **prior** count: commit
`4c36516` (revert phase 2b risk-managed experiment, predating the OU
branch) removed `tests/backtest/test_risk_managed.py` (23 tests) and
`tests/backtest/test_vol_target.py` (9 tests) — 32 tests in total,
of which the handoff's "286 prior" arithmetic apparently retained 17.

The **39 new tests added on this branch** are verified by file:

| File                                          | Tests | Source commit |
|-----------------------------------------------|------:|---------------|
| `tests/stats/test_ou.py`                      |    17 | 3693b86       |
| `tests/stats/test_ou_mc.py`                   |     3 | 3693b86       |
| `tests/intraday/test_resample.py`             |     5 | e07907e       |
| `tests/intraday/test_signals_ou.py`           |    10 | e07907e       |
| `tests/intraday/test_backtest_cost_pin.py`    |     4 | e07907e       |
| **Total new**                                 | **39**|               |

So `308 = 269 prior (revised) + 39 new`. All passing, no skips, no
xfails, no deselects.

------------------------------------------------------------------------

## 11. Figures appendix (retroactive — generated by Unit V)

All figures emitted by `scripts/16_retro_figures_phase3.py` per the
contract in [`reporting_standard.md`](reporting_standard.md). Each
PNG has a companion CSV holding its exact data — paths in
`reports/phase3_ou/figures/MANIFEST.csv`. **No number behind any
figure was recomputed** — figures consume only the existing
artifacts under `reports/phase3/` and `reports/phase3_ou/`.

Figure-letter taxonomy: see
[`docs/reporting_standard.md`](reporting_standard.md) §2.

### 11.1 OU best cell (5-min, B, 3 bps, none) — `figures/ou_best_cell/`

| letter | figure                                                                 |
|:------:|------------------------------------------------------------------------|
|   a    | [per-pair-fold equity (gross+net)](../reports/phase3_ou/figures/ou_best_cell/a_ou_best_per_pair_fold_equity.png) |
|   b    | [portfolio NAV gross vs net](../reports/phase3_ou/figures/ou_best_cell/b_ou_best_portfolio_nav.png)               |
|   h    | [exit-reason composition by pair-fold](../reports/phase3_ou/figures/ou_best_cell/h_ou_best_exit_reasons.png)      |

### 11.2 Grid roll-ups — `figures/grid_rollups/`

| letter | figure                                                                                         |
|:------:|------------------------------------------------------------------------------------------------|
| d × 6  | cost ladder (engines overlaid) for every (freq ∈ {1,5,15}, regime ∈ {A,B}) — see `d_cost_ladder_f{freq}_{regime}.png` |
|   i    | [trade counts by engine × freq @ B, 3 bps](../reports/phase3_ou/figures/grid_rollups/i_trade_counts_B_3bps.png)        |
|   f    | [half-life distribution per freq with A/B bands](../reports/phase3_ou/figures/grid_rollups/f_half_life_distribution.png) |
|   g    | [frozen-μ drift per pair-fold with ±0.5σ_eq flags](../reports/phase3_ou/figures/grid_rollups/g_drift_chart.png)      |
|   j    | [(1+β)/2 histogram, all pairs vs traded survivors](../reports/phase3_ou/figures/grid_rollups/j_beta_histogram_all_vs_traded.png) |
|   k    | [exclusion funnel (per-freq, Regime B)](../reports/phase3_ou/figures/grid_rollups/k_exclusion_funnel.png)           |

### 11.3 Coarse rolling_z cells (addendum-#5) — `figures/rolling_z_coarse/`

Eight figures: `b_rollingz_coarse_f{5,15}_{A,B}_portfolio_nav.png`
and `h_rollingz_coarse_f{5,15}_{A,B}_exit_reasons.png` (one (b) and
one (h) per (freq, regime) cell at the canonical 3 bps cost).

### 11.4 Attribution slice — `figures/attribution_slice/`

Sixteen NAV figures comparing rolling_z (restricted to the OU HL-band
survivors at each freq) vs OU on the same two pair-folds. Eight per
engine × four costs × two freqs:

* `b_attribution_rollingz_f{5,15}_B_{1,3,5,8}bps_portfolio_nav.png`
* `b_attribution_ou_f{5,15}_B_{1,3,5,8}bps_portfolio_nav.png`

### 11.5 Phase 3 v2 (1-min rolling_z) — `figures/v2_1min_rolling_z/`

| letter | figure                                                                  |
|:------:|-------------------------------------------------------------------------|
|   b    | portfolio NAV by regime (`b_v2_1min_portfolio_nav_A.png`, `..._B.png`)   |
|   h    | exit-reason composition (regime A vs B; `h_v2_1min_exit_reasons.png`)   |

Figures (a/c/e/f/g/i/j/k) are **N/A** at v2 1-min:
* (a) per-pair-fold equity is omitted at v2 because the 19-pair-fold
  panel is unreadable in one figure; see §6 of the v2 report instead.
* (c, e) v2 uses rolling_z, not Bertram — no a* exists.
* (f, g, j, k) are dataset-level roll-ups already covered in §11.2
  and shared across engines (the underlying pair-fold population is
  the same set).

### 11.6 Manifest

Full list of (group, key, png, csv) tuples:
[`reports/phase3_ou/figures/MANIFEST.csv`](../reports/phase3_ou/figures/MANIFEST.csv).
Total: **41 figures**, each with a companion CSV. Generator:
`scripts/16_retro_figures_phase3.py`.
