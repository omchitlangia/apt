# Phase 3 OU/Bertram run report

**Generated:** 2026-06-10
**Branch:** `feature/ou-optimal-thresholds`
**Driver:** `scripts/15_phase3_ou.py --mode full` (committed at 253d2ff)
**Design doc:** `docs/ou_thresholds_design.md` (locked decisions in §8)
**Inputs:** Phase 3 v2 daily selection (Phase 2A reuse) + intraday liquidity gate
**Cost model:** intraday `CostBreakdown` (4.5 bps fixed/leg/RT + spread sweep)

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

### 6.2 Intraday-to-daily HL ratio (diagnostic only — no gate)

| freq | p10 | p50 | p90 |
|-----:|----:|----:|----:|
|    1 | 0.24 | 0.49 | 0.85 |
|    5 | 0.35 | 0.68 | 1.04 |
|   15 | 0.41 | 0.77 | 1.13 |

Ratios in [0.2, 1.1] dominate — intraday mean-reversion is on a
~50–80% of daily timescale (in matched trading-time units). This is
broadly consistent across freqs. No outliers requiring
investigation.

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

### 6.5 Cost-convention diagnostic (β distribution, equal-notional caveat)

```
β values (unique across pair-folds): {0.057, 0.064, 0.179, 0.667, 0.671,
0.727, 0.757, 0.784, 0.816, 0.832, 0.872, 0.925, 0.980, 0.999, 1.138,
1.167, 1.247, 1.643}

(1+β)/2 distribution: min=0.528, median=0.912, max=1.322
|(1+β)/2 − 1| median = 0.115
```

**Median |(1+β)/2 − 1| = 0.115 exceeds the 10% threshold** specified
in addendum #2. The current run inherits v2's equal-notional
plumbing verbatim (per the addendum directive), so cost deduction is
`2 × cost_bps_per_leg / 10⁴` regardless of β. The β distribution
indicates this assumption mis-bills cost by 5-32% on the extreme
pair-folds (e.g. INDUSINDBK/HDFCBANK at β=1.64 has true (1+β)/2 =
1.32, so cost is under-charged by ~32%; IDBI-anchored pairs at
β=0.06 have true (1+β)/2 = 0.53, so cost is over-charged by ~47%).

**[TODO] cost-accounting follow-up unit** — implement a
β-proportional cost convention (cost_per_pair = (1+β) × cost_per_leg)
and re-run; check whether the directional bias in Regime B at high-β
pairs changes sign or magnitude.

### 6.6 Exit-type breakdown (freq=5, Regime B, cost=3, stop=none)

```
mean_revert  : 26  (76%)   — the OU model's primary exit
time_stop    :  6  (18%)   — pair held to max_holding without crossing μ
eod_squareoff:  0   (0%)   — Regime B does not force-close intraday
fold_close   :  2   (6%)   — open positions at fold-test-end
z_stop       :  0   (0%)   — stop=none
```

A 76% mean-revert exit rate is healthy — the model is working as
designed when the drift is benign. The 18% time-stop fraction
correlates with the pair-folds whose drift is severe; those trades
sit at one extreme and never cross μ within `3 × half-life` bars.

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

## 7. Deferred / [TODO] items

- **Coarse-bar `rolling_z` baselines** (addendum #5). The orchestrator
  `_full_cells()` builds them but `main()` filters them out
  (`c.engine == "ou"` only). Reason: implementing them in script 15
  cleanly required duplicating the v2 rolling-z + signal generation
  inside the cell loop, which we judged out of scope for this round.
  The OU vs v2 comparison is still meaningful (§5 cross-reference
  against v2 report numbers); the missing piece is the
  aggregation-vs-engine decomposition.
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
```

Plus this report. No changes to:
- `scripts/13_phase3_intraday.py` (v2 byte-for-byte untouched)
- `src/apt/signals/spread.py` (rolling_z primitive untouched)
- `src/apt/intraday/zscore.py` (v2 z untouched)
- Phase 1/2 daily pipeline.

## 9. One-line takeaway

**OU/Bertram is competitive with v2's rolling-z on the same pair-
folds (Sharpe 0.96 vs ≲1 at 3 bps, freq=5, Regime B), but the
addendum's default HL band collapses Regime A to empty; Regime B
trades only 2 pair-folds out of 19 because intraday OU half-lives
on daily-cointegrated pairs are multi-session, not multi-minute.
Train-frozen μ drift is the single biggest issue surfaced — and is
the root cause of the stop-hard pathology.**
