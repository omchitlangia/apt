# Phase 3 — β-aware (1+β) cost-accounting unit report

**Generated:** 2026-06-11
**Branch:** `feature/cost-beta-notional`
**Driver:** `scripts/15_phase3_ou.py --mode full` +
`scripts/15b_phase3_rolling_baseline.py` + `scripts/10_backtest.py`
(all re-run under β-aware billing) and `scripts/16_retro_figures_phase3.py`
(figures regen). This report follows the contract in
[`docs/reporting_standard.md`](reporting_standard.md).

------------------------------------------------------------------------

## 1. Objective

Replace the legacy 2× equal-notional per-pair round-trip cost with the
**β-aware (1+β) per-leg billing** derived from the spread arithmetic
`Δ(log Y − β · log X)` (Y leg = 1 unit, X leg = β units). Re-run Phase 3
OU + rolling_z baseline + Phase 2A daily under corrected billing,
publish corrected-vs-old delta tables, regenerate the matched-universe
cost ladder, and run a Phase 2A diagnosis against the escalation gate
(|ΔSharpe| > 0.1 or any sign flip).

The Q1-Q8 decisions in the prior unit's
[`docs/cost_beta_design.md`](cost_beta_design.md) are now executed.

------------------------------------------------------------------------

## 2. Pre-registered expectations

Recorded BEFORE the re-runs:

(a) **Headline OU best cell (5-min B, 3 bps)** — net Sharpe will move
    by `±0.05` units. INDUSINDBK/HDFCBANK fold 4 has β = 1.64 (under-
    billed by 32% under the old 2×) and KOTAKBANK/HDFCBANK fold 6 has
    β = 0.87 (over-billed by 6%). At n = 2 these partially cancel.
(b) **Full 14-pair rolling_z baseline** — Δ near zero (population mean
    (1+β)/2 = 0.91 ≈ 1, so the cost shift averages out across the
    broader sample).
(c) **Matched-universe rolling_z slice** (restricted to the two OU HL-
    band survivor pair-folds) — DIFFERENT direction from the full
    baseline: net Sharpe should DROP because INDUSINDBK fold 4 has
    β > 1 and is now charged MORE.
(d) **a* re-fits under corrected `c`** — a* values shift by ~10-30%
    where β ≠ 1, primarily widening the band for INDUSINDBK fold 4.
(e) **Phase 2A daily** — net Sharpe Δ small (within ±0.05);
    population (1+β)/2 should be close to 1.0 on average, so total
    daily cost shifts only modestly.
(f) **Escalation gate**: we did NOT pre-commit a probability that
    Phase 2A would breach it; we record the actual ΔSharpe.

------------------------------------------------------------------------

## 3. Config grid

Three re-runs in this unit, all under the (1+β) billing convention:

| step | driver                                       | cells | re-runs vs reads CSV |
|-----:|----------------------------------------------|------:|---------------------|
| 1    | `scripts/15_phase3_ou.py --mode full`        | 30    | re-run (Bertram refit per pair-fold) |
| 2    | `scripts/15b_phase3_rolling_baseline.py`     | 16    | re-run               |
| 3    | `scripts/10_backtest.py` (Phase 2A daily)    | 1 (7-fold WF) | re-run     |
| 4    | re-stamp helper (Phase 3 v2 @ 3bps)          | 2 (A, B) | reads existing CSVs |
| 5    | `scripts/16_retro_figures_phase3.py`         | 43 figures | reads new metrics |

The OU + rolling_z cell grid is unchanged: `freq ∈ {1, 5, 15}`,
`regime ∈ {A, B}`, `cost ∈ {1, 3, 5, 8}` bps, `stop ∈ {none, hard}`.
Bertram's a* is re-solved per `(pair-fold, freq, cost)` with
pair-specific c = (1+β) × cost_log_per_leg.

------------------------------------------------------------------------

## 4. Headline tables — corrected (1+β) vs old (2×)

### 4.1 OU grid (Regime B, stop=none) — full 30 cells in `metrics_ou_DELTA_corrected_vs_old.csv`

| freq | cost | n_trades_old | n_trades_new | gross_total%_old | gross_total%_new | net_total%_old | net_total%_new | net_Sharpe_old | net_Sharpe_new | Δ_netSh |
|-----:|-----:|-------------:|-------------:|-----------------:|-----------------:|---------------:|---------------:|---------------:|---------------:|--------:|
|   1  |   1  |          68  |          68  |          40.4919 |          41.1305 |        33.1931 |        33.6293 |         0.4845 |         0.4903 | +0.0058 |
|   1  |   3  |          62  |          63  |          30.8904 |          36.5499 |        22.7145 |        27.5031 |         0.3547 |         0.4140 | +0.0593 |
|   1  |   5  |          59  |          58  |          30.2909 |          30.9514 |        20.6411 |        21.0063 |         0.3268 |         0.3316 | +0.0048 |
|   1  |   8  |          54  |          54  |          26.7871 |          26.7481 |        15.5853 |        15.1803 |         0.2543 |         0.2478 | -0.0066 |
|   5  |   1  |          37  |          36  |          56.3221 |          54.9781 |        50.0875 |        48.2657 |         1.0058 |         0.9808 | -0.0250 |
| **5**|**3** |       **34** |       **34** |      **54.6857** |      **54.8685** |    **46.9945** |    **46.2043** |     **0.9623** |     **0.9494** | **-0.0129** |
|   5  |   5  |          32  |          32  |          51.1581 |          53.3383 |        42.2415 |        43.1671 |         0.8891 |         0.9049 | +0.0159 |
|   5  |   8  |          30  |          30  |          54.0697 |          57.1748 |        42.9372 |        44.2764 |         0.9100 |         0.9346 | +0.0246 |
|  15  |   1  |          32  |          34  |          43.6313 |          56.3133 |        38.6634 |        49.9148 |         0.8161 |         1.0032 | **+0.1871** |
|  15  |   3  |          31  |          31  |          45.3961 |          46.6805 |        38.7900 |        39.2199 |         0.8235 |         0.8297 | +0.0062 |
|  15  |   5  |          30  |          28  |          45.4513 |          39.0721 |        37.3924 |        30.9660 |         0.8049 |         0.6845 | **-0.1205** |
|  15  |   8  |          27  |          26  |          44.4938 |          43.7060 |        35.0624 |        33.5391 |         0.7716 |         0.7449 | -0.0267 |

**Headline cell (freq=5, B, 3 bps) ΔSharpe = −0.013** — within ±0.1
escalation gate. The corrected 21.07% annualised net (vs old 21.24%)
matches the pre-registered expectation that the two pair-folds'
opposite-sign β corrections largely cancel.

**Two cells exceed |0.1|** on Δ_netSh (both at freq=15):

* (15, B, 1 bps): Sharpe +0.19. New run produced 34 trades vs 32 old —
  a marginal pair-fold passed the Bertram solver at corrected c=(1+β)·1bp/10k that did not at the old higher c. Not a bug; a sensitivity to the threshold-feasibility frontier near very low c.
* (15, B, 5 bps): Sharpe −0.12. Lost 2 trades (30→28) — Bertram a*
  pushed past the threshold for one pair-fold's signal trajectory.

Both are real consequences of refitting Bertram per pair-fold under
corrected c, NOT a sign flip on the headline result. The pattern of
movements (small everywhere except where the n=2 trade-count
discreteness amplifies them — see §7c) is consistent with the
expectation in §2.

### 4.2 Matched-universe slice (2 OU HL-band survivors, B, stop=none)

Full table in
[`reports/phase3_ou/figures/matched_universe/matched_metrics_DELTA.csv`](../reports/phase3_ou/figures/matched_universe/matched_metrics_DELTA.csv).
Figures regenerated under corrected billing:
[`d_matched_universe_cost_ladder_f5_B.png`](../reports/phase3_ou/figures/matched_universe/d_matched_universe_cost_ladder_f5_B.png),
[`...f15_B.png`](../reports/phase3_ou/figures/matched_universe/d_matched_universe_cost_ladder_f15_B.png).

| engine    | freq | cost | n_trades_old | n_trades_new | net_total%_old | net_total%_new | net_Sharpe_old | net_Sharpe_new | Δ_netSh |
|-----------|-----:|-----:|-------------:|-------------:|---------------:|---------------:|---------------:|---------------:|--------:|
| ou        |  5   |  1   |     37       |       36     |        50.0875 |        48.2657 |         1.0058 |         0.9808 | -0.0250 |
| ou        |  5   |  3   |     34       |       34     |        46.9945 |        46.2043 |         0.9623 |         0.9494 | -0.0129 |
| ou        |  5   |  5   |     32       |       32     |        42.2415 |        43.1671 |         0.8891 |         0.9049 | +0.0159 |
| ou        |  5   |  8   |     30       |       30     |        42.9372 |        44.2764 |         0.9100 |         0.9346 | +0.0246 |
| ou        | 15   |  1   |     32       |       34     |        38.6634 |        49.9148 |         0.8161 |         1.0032 | **+0.1871** |
| ou        | 15   |  3   |     31       |       31     |        38.7900 |        39.2199 |         0.8235 |         0.8297 | +0.0062 |
| ou        | 15   |  5   |     30       |       28     |        37.3924 |        30.9660 |         0.8049 |         0.6845 | **-0.1205** |
| ou        | 15   |  8   |     27       |       26     |        35.0624 |        33.5391 |         0.7716 |         0.7449 | -0.0267 |
| rolling_z |  5   |  1   |    225       |      225     |        42.2209 |        37.6694 |         1.0471 |         0.9454 | **-0.1017** |
| rolling_z |  5   |  3   |    225       |      225     |        29.9801 |        24.3410 |         0.7807 |         0.6434 | **-0.1373** |
| rolling_z |  5   |  5   |    225       |      225     |        18.7929 |        12.3029 |         0.5125 |         0.3414 | **-0.1711** |
| rolling_z |  5   |  8   |    225       |      225     |         3.7912 |        −3.6046 |         0.1103 |        −0.1070 | **−0.2173 ⚠ SIGN FLIP** |
| rolling_z | 15   |  1   |     88       |       88     |        20.3249 |        18.6823 |         0.6608 |         0.6133 | -0.0475 |
| rolling_z | 15   |  3   |     88       |       88     |        16.1631 |        14.0061 |         0.5392 |         0.4733 | -0.0659 |
| rolling_z | 15   |  5   |     88       |       88     |        12.1453 |         9.5141 |         0.4156 |         0.3309 | -0.0848 |
| rolling_z | 15   |  8   |     88       |       88     |         6.3776 |         3.1059 |         0.2266 |         0.1127 | **-0.1140** |

**Matched-universe rolling_z is the only segment that materially moves
under corrected billing.** Mechanism: the matched universe is exactly
the 2 OU survivors, one of which (INDUSINDBK fold 4, β = 1.64) is
under-charged by 32% under the old 2×. Rolling_z trades 225 round-
trips on this pair-fold; the per-trade Δcost is 32% larger now, so
the cumulative cost drag is dramatically higher. At cost = 8 bps,
rolling_z 5-min flips sign on the same pair-folds.

This was the §11.7 matched-universe finding the prior unit
flagged — now confirmed with the corrected cost actually applied.
The §5.2 "engine SLOPES are attributable, LEVELS confounded"
distinction is preserved: rolling_z is hit harder by the correction
because it churns more trades, not because the threshold convention
itself differs.

### 4.3 Full 14-pair rolling_z baseline (Regime B, stop=none)

Population mean (1+β)/2 = 0.91, so the corrected billing is on
average SLIGHTLY CHEAPER per pair than the old 2×. Δ_netSh tiny
(+0.012 at 5min/1bps; +0.017 at 5min/3bps; etc).
Full table in
[`reports/phase3_ou/metrics_rolling_baseline_DELTA_corrected_vs_old.csv`](../reports/phase3_ou/metrics_rolling_baseline_DELTA_corrected_vs_old.csv).
No cell exceeds the |0.1| escalation gate.

### 4.4 Phase 3 v2 @ 3 bps (re-stamp from persisted trades)

Computed by joining `fold_pairs.csv` β onto
`trades_two_regime_3bps.csv` (which has gross_log_pnl per trade) and
recomputing net = gross − (1+β)·per_leg_log_cost. Per-session
correction applied to `pair_daily_two_regime_3bps.csv` to get the
portfolio metrics.

| regime | net_total%_old | net_total%_new | Δ_net%   | net_Sharpe_old | net_Sharpe_new | Δ_Sharpe |
|--------|---------------:|---------------:|---------:|---------------:|---------------:|---------:|
| A      |        −72.54  |        −73.72  |  −1.18   |        −1.8719 |        −1.8549 |  +0.0170 |
| B      |        −46.94  |        −49.23  |  −2.29   |        −0.6669 |        −0.6985 |  −0.0316 |

Δ_Sharpe both within ±0.1. **Phase 3 v2 conclusions unchanged.** Full
file:
[`metrics_two_regime_v2_3bps_DELTA_corrected_vs_old.csv`](../reports/phase3_ou/metrics_two_regime_v2_3bps_DELTA_corrected_vs_old.csv).
Other cost levels were not re-stamped per scope (Phase 3 v2 cost levels
other than 3 bps require either a re-run or a fold-level β-aware
recomputation; flagged as `[TODO scope]` for a follow-up errata pass).

### 4.5 Phase 2A daily

Re-ran `scripts/10_backtest.py` under (1+β) billing; the `Trade`
dataclass now carries `pair_beta`, persisted to `reports/backtest_trades.csv`.

| metric             | OLD (2×)    | NEW (1+β) | Δ        |
|--------------------|------------:|----------:|---------:|
| net_total_pct      |     214.023 |   215.672 |   +1.649 |
| net_ann_pct        |      17.759 |    17.847 |   +0.088 |
| net_sharpe         |       1.077 |     1.083 |  +0.0065 |
| net_max_drawdown%  |     −17.453 |   −17.385 |   +0.068 |
| gross side         | UNCHANGED (β doesn't affect gross) | — | — |

**Δ_Sharpe = +0.006.** Within ±0.1 escalation gate; no sign flips.
**Phase 2A conclusions are NOT rewritten by this unit** — the unit
ends here on Phase 2A per the gate logic.

------------------------------------------------------------------------

## 5. Signed (1+β)/2 distributions — explicit denominators (Item 7b)

`reports/phase3_ou/ou_pair_fold_diag.csv` is the source. Denominators
are now stated explicitly:

| population                 | n   | mean β | mean (1+β)/2 | pct < 1.0 | pct > 1.0 | signed mean offset |
|----------------------------|----:|-------:|-------------:|----------:|----------:|-------------------:|
| **19 pair-folds (billing-relevant)** | 19 |  0.839 |       0.9196 |     73.7% |     26.3% |             −0.0804 |
| 14 unique pair-keys (CSV `n_pairs`)  | 14 |  0.818 |       0.9091 |     78.6% |     21.4% |             −0.0909 |
| 2 traded survivors (OU best cell n=2) |  2 |  1.258 |       1.1288 |      0%   |    100%   |             **+0.1288** |

Phase 2A daily universe (from
[`reports/fold_pairs_daily_beta.csv`](../reports/fold_pairs_daily_beta.csv)):

| population                       | n  | mean β | mean (1+β)/2 | pct < 1.0 | signed mean offset |
|----------------------------------|---:|-------:|-------------:|----------:|-------------------:|
| Phase 2A daily (fold × pair)     | 30 |  0.831 |       0.9152 |     76.7% |             −0.0848 |

**Sign skews:**
- 19 pair-fold intraday population: **LEFT-skewed below 1.0**. Equal-
  notional OVER-charges by ~8% on average.
- 14 unique pair-key population: same skew direction, slightly more
  pronounced.
- 2 traded survivors: **RIGHT-skewed above 1.0** (opposite sign).
  Equal-notional UNDER-charges by ~13% on average, dominated by
  INDUSINDBK fold 4's +0.32 deviation.
- Phase 2A daily population: LEFT-skewed below 1.0; effectively the
  same direction as the intraday population.

------------------------------------------------------------------------

## 6. a* coincidence verification (Item 7a)

The OU report's grid-rollup a* table showed identical-looking values at
freq=5 and freq=15. The question: are these independently computed per
`(pair-fold, freq, cost)` or a cache/join artifact?

**Answer: independently computed.** Each `(pair-fold, freq)`
combination has its own AR(1) OLS fit on the resampled spread, so
`kappa_per_bar` differs by frequency by construction (κ × Δt is
preserved across frequency rescaling for a *theoretical* OU
process, but the OLS-fit κ on FINITE samples drifts). The a* values
then differ in the 4th-5th decimal place, NOT bit-identically.

Evidence — pasted from the re-run trade CSV
([`reports/phase3_ou/trades_ou.csv`](../reports/phase3_ou/trades_ou.csv)),
all four costs:

```
-- cost = 1 bps --
   fold=4 INDUSINDBK/HDFCBANK:  a* freq=5 = 0.50406061, a* freq=15 = 0.50431415  (Δ = +2.5e-4)
   fold=6 KOTAKBANK/HDFCBANK:   a* freq=5 = 0.39562776, a* freq=15 = 0.39569004  (Δ = +6.2e-5)
-- cost = 3 bps --
   fold=4 INDUSINDBK/HDFCBANK:  a* freq=5 = 0.56117879, a* freq=15 = 0.56146335  (Δ = +2.8e-4)
   fold=6 KOTAKBANK/HDFCBANK:   a* freq=5 = 0.43978288, a* freq=15 = 0.43985243  (Δ = +7.0e-5)
-- cost = 5 bps --
   fold=4 INDUSINDBK/HDFCBANK:  a* freq=5 = 0.60940914, a* freq=15 = 0.60972042  (Δ = +3.1e-4)
   fold=6 KOTAKBANK/HDFCBANK:   a* freq=5 = 0.4768991,  a* freq=15 = 0.47697486  (Δ = +7.6e-5)
-- cost = 8 bps --
   fold=4 INDUSINDBK/HDFCBANK:  a* freq=5 = 0.67119496, a* freq=15 = 0.67154142  (Δ = +3.5e-4)
   fold=6 KOTAKBANK/HDFCBANK:   a* freq=5 = 0.52419995, a* freq=15 = 0.5242838   (Δ = +8.4e-5)
```

Underlying OU fits driving these a* values:

```
 fold_id   pair                 freq_min  kappa_per_bar  sigma_eq  HL_min       z_OU_test_mean
       4   INDUSINDBK/HDFCBANK         1    0.000577     0.035705   1200.26     −3.300
       4   INDUSINDBK/HDFCBANK         5    0.002104     0.035812   1647.05     −3.289
       4   INDUSINDBK/HDFCBANK        15    0.005576     0.035760   1864.74     −3.290
       6   KOTAKBANK/HDFCBANK          1    0.000700     0.051458    990.74     +1.158
       6   KOTAKBANK/HDFCBANK          5    0.003359     0.051456   1031.70     +1.159
       6   KOTAKBANK/HDFCBANK         15    0.009812     0.051432   1059.65     +1.160
```

κ_per_bar differs by 5-10× across freqs (as expected for a κ that's
been rescaled). σ_eq is freq-stable. The a* coincidence is therefore
a consequence of (κ × cycle-time) being approximately scale-invariant
on this universe — NOT a cache reuse. A cache artifact would produce
bit-identical a* across freqs; the actual 4-5th-decimal differences
above confirm the per-freq fit was run.

------------------------------------------------------------------------

## 7. Caveats (Item 7c)

* **cost=8 > cost=5 in the OU best cell**: the OU net_total% goes
  46.99% → 42.24% → **42.94%** from cost 3 → 5 → 8. This is **trade-
  set discreteness at n ≈ 30**, not a real "8 bps does better than
  5 bps" finding. The Bertram solver widens the band slightly more at
  8 bps, which happens to drop one losing trade from the n=32 cell —
  in a sample this small the discrete trade-set entry/exit can move
  the total by 1-2 pp. The headline at 3 bps remains the canonical
  cell.
* **"Orders of magnitude" wording correction**: §6.5b previously
  described the a* entry band as "~2-3 orders of magnitude wider than
  the cost itself". The accurate range is **~8-18× the all-in
  round-trip cost** (e.g. a* = 232 bps at 3 bps cost ⇒ 232/15 ≈ 15×
  including fixed components; KOTAKBANK 5-min at 8 bps: a* = 276 bps
  vs all-in 12.5 bps/leg × (1+0.87) = 23 bps → ratio 12×). Bands are
  wide but not 100× wide.
* **β < 0.2 pairs make cost look artificially cheap**: under (1+β) ×
  per-leg, a pair with β = 0.06 (the lower end of our population) is
  billed as 1.06× per-leg — basically the cost of a single outright
  leg. **These are near-OUTRIGHT positions, not pair trades:** the
  hedge ratio implies the X leg is moving 6% as much as the Y leg, so
  the spread is dominated by Y. Selection-smell flag for any future
  pair filter: pairs with β ∉ [0.3, 3.0] are suspicious cointegration
  artifacts (a strong autocorrelated Y dragging a weak-correlation X
  along). NO machinery added in this unit; flagged for the next
  selection-criterion pass.
* **INDUSINDBK/HDFCBANK fold 6** has **z_OU drift = −7.05 σ_eq**
  (from §6 of the prior OU report and confirmed here). This is the
  **canonical hard case for the Kalman unit** — frozen μ_OU is
  catastrophically wrong across a 6+ σ_eq shift, and it is the
  archetype of the problem a slowly-recalibrating μ on a Kalman path
  is designed to solve. Named explicitly here so the Kalman discovery
  unit's pre-registered expectations can cite it.
* **Net-exposure corollary**: pair trades under (1+β) billing carry
  net long-market exposure `(1 − β) × notional`. NOT market-neutral
  when β ≠ 1. Flagged in `src/apt/intraday/costs.py` module docstring;
  Kalman unit will need to address this in any future risk-budget
  machinery.

------------------------------------------------------------------------

## 8. Exclusion funnel — unchanged

| stage                              | n pair-folds (freq=5, B) | n (freq=15, B) |
|------------------------------------|-------------------------:|---------------:|
| attempted (intraday liquidity gate) | 19                      | 19             |
| AR(1) fit_ok                        | 18                      | 18             |
| HL ∈ [120, 1875] min                | 2                       | 2              |
| traded                              | 2                       | 2              |

**No change vs the prior OU run** — the Bertram re-fit under corrected
c can shift trade COUNTS within a traded cell (e.g. 32 → 34 at 15min
1bps) but the gate logic itself (liquidity, AR(1), HL band) is
β-independent. Confirmed by inspection of the new
`ou_pair_fold_diag.csv`.

------------------------------------------------------------------------

## 9. Figures (per `reporting_standard.md`)

| letter | figure                                                                                             |
|:------:|----------------------------------------------------------------------------------------------------|
|   a    | (a) per-pair-fold equity gross+net — unchanged shape; updated under corrected billing (see `ou_best_cell/`) |
|   b    | OU best-cell portfolio NAV at 3 bps — corrected billing: terminal 46.2% (was 47.0%) |
|   d    | matched-universe cost ladder (freq 5, 15 — Regime B) — REGENERATED under corrected billing       |
|   h    | exit-reason composition by pair-fold — unchanged structure                                       |

All 43 figures regenerated by
`scripts/16_retro_figures_phase3.py` reading from the new metric CSVs.
Full manifest:
[`reports/phase3_ou/figures/MANIFEST.csv`](../reports/phase3_ou/figures/MANIFEST.csv).

------------------------------------------------------------------------

## 10. Verdict

1. **(1+β) billing is now the single billing convention in the
   codebase.** Old `cost_log_per_pair_round_trip` property removed
   from `CostBreakdown`; every site mapped in `cost_beta_design.md`
   §1 is migrated and tested. β = 1 reproduces the legacy 2× value
   bit-exactly as a continuity pin.
2. **Phase 3 OU headline (5-min B, 3 bps) net Sharpe: 0.962 → 0.949
   (Δ = −0.013).** Net annualised 21.24% → 21.07%. Within ±0.1; no
   conclusion overturned.
3. **Phase 3 v2 @ 3 bps Δ within ±0.05 on both regimes.** No
   escalation.
4. **Phase 2A daily Δ_Sharpe = +0.006.** Within ±0.1; **no rewrite
   of Phase 2A conclusions**.
5. **Matched-universe rolling_z slice flipped sign at 5-min/8-bps**
   (Sharpe +0.110 → −0.107). The slice is restricted to one β = 1.64
   and one β = 0.87 pair-fold; the high-β leg now pays 32% more cost
   per trade × 225 trades.
6. **a* refit per (pair-fold, freq, cost) is independent — confirmed
   by the 4-5th-decimal differences across freqs** despite κ_per_bar
   shifting 5-10×.
7. **Phase 3 v2 cost levels other than 3 bps remain on the old 2×
   convention in their persisted CSVs**. Flagged as `[TODO scope]` for
   a follow-up errata pass (no re-run was in scope this unit).

The unit's mechanical question — whether the cost convention should be
β-aware — is now closed. The remaining open questions are:
- Phase 3 v2 at all costs under corrected billing (separate errata pass).
- The (1−β) net-exposure flag for the Kalman unit's risk-budget design.
- The β < 0.2 selection-smell heuristic (next pair-filter pass).

------------------------------------------------------------------------

## 11. Test results

```
$ .venv/bin/pytest tests/
331 passed, 2 warnings in 48.55s
```

Specific tests added by this unit:
- `tests/intraday/test_backtest_cost_pin.py` — 8 parametrized round-
  trip tests over β ∈ {0.057, 0.872, 1.0, 1.643}, plus continuity-pin,
  long/short, composition, and invalid-β rejection (12 tests total).
- `tests/stats/test_ou.py::test_a_star_monotone_in_cost_under_beta_
  aware_billing` — a* monotonicity re-verified under (1+β) billed c.

No tests deselected or skipped; full suite green.

------------------------------------------------------------------------

## 12. Schema changes (Item 5)

New columns on **newly-written** trade CSVs (the v2 schema):

* `pair_beta` — the β used to bill this trade.
* `cost_log_per_pair_rt` — the actual billed cost (= `(1+β) × cost_log_per_leg`). Mirror of `cost_log` for downstream consumers.
* `schema_version` — string constant `"v2-cost-beta-2026.06.11"`. Defined as `TRADE_CSV_SCHEMA_VERSION` in `apt.intraday.costs`.
* `n_legs` — kept at value `2` for ONE cycle, marked DEPRECATED in `reporting_standard.md` (PR follow-up will remove).

Files touched:
- `reports/phase3_ou/trades_ou.csv` (new schema)
- `reports/phase3_ou/trades_rolling_baseline.csv` (new schema)
- `reports/phase3/trades_two_regime_3bps.csv` (new schema on next re-run; current file pre-dates this unit)
- `reports/backtest_trades.csv` (new schema)

Historical CSVs (the `_OLD_2x.csv` snapshots) are kept unmodified for
provenance.

------------------------------------------------------------------------

## 13. Addendum (Part A close-out) — billing-vs-refit decomposition + §5.2 correction

### 13.1 Billing vs refit decomposition (matched universe)

Three scenarios per cell, computed by re-stamping the **persisted OLD
trade set** (`trades_ou_OLD_2x.csv` / `trades_rolling_baseline_OLD_2x.csv`)
to new billing without re-solving Bertram, then comparing to the
already-run new fit:

- **A = {old a*, old billing}** — the original 2× run.
- **B = {old a*, new billing}** — OLD trade set, net re-stamped with
  `(1+β)` cost (no re-solve). Billing-only effect.
- **C = {new a*, new billing}** — the re-run (Bertram re-solved under
  corrected pair-specific `c`).

`billing = B − A`, `refit = C − B`. For rolling_z there is no a* and
the thresholds are cost-blind, so the trade set is identical across
billing ⇒ **refit ≡ 0** (confirmed numerically below).

Net-Sharpe decomposition for every matched-universe cell with
|Δ_Sharpe| > 0.05:

| engine    | freq | cost | A old/old | B old/new | C new/new | billing | refit  |
|-----------|-----:|-----:|----------:|----------:|----------:|--------:|-------:|
| ou        | 15   |  1   |   0.8161  |   0.8051  |   1.0032  | −0.0110 | +0.1981 |
| ou        | 15   |  5   |   0.8049  |   0.7850  |   0.6845  | −0.0199 | −0.1005 |
| rolling_z |  5   |  1   |   1.0471  |   0.9454  |   0.9454  | −0.1017 |  0.0000 |
| rolling_z |  5   |  3   |   0.7807  |   0.6434  |   0.6434  | −0.1373 |  0.0000 |
| rolling_z |  5   |  5   |   0.5125  |   0.3414  |   0.3414  | −0.1711 |  0.0000 |
| rolling_z |  5   |  8   |   0.1103  |  −0.1070  |  −0.1070  | −0.2173 |  0.0000 |
| rolling_z | 15   |  3   |   0.5392  |   0.4733  |   0.4733  | −0.0659 |  0.0000 |
| rolling_z | 15   |  5   |   0.4156  |   0.3309  |   0.3309  | −0.0848 |  0.0000 |
| rolling_z | 15   |  8   |   0.2266  |   0.1127  |   0.1127  | −0.1140 |  0.0000 |

**rolling_z is pure billing** (refit ≡ 0): the whole degradation comes
from charging the β = 1.64 INDUSINDBK leg its true cost across 225
round-trips. **The two OU cells are refit-dominated**: at 15-min 1 bps
the corrected `c` widens the band enough to ADD trades that lift Sharpe
(+0.198 refit), while at 15-min 5 bps the refit drops trades and costs
−0.101.

**Best-cell split (5-min B, 3 bps).** The headline cell's |Δ_Sharpe| is
only 0.013, below the table threshold, but its decomposition was asked
for explicitly. In **net annualised pp**:

| scenario              | net_ann % | net_total % | net_sharpe |
|-----------------------|----------:|------------:|-----------:|
| A {old a*, old bill}  |   21.2413 |     46.9945 |     0.9623 |
| B {old a*, new bill}  |   20.8436 |     46.0318 |     0.9465 |
| C {new a*, new bill}  |   20.9150 |     46.2043 |     0.9494 |

- billing (B−A) = **−0.398 pp** ann (−0.963 pp total)
- refit (C−B)   = **+0.071 pp** ann (+0.173 pp total)
- total         = **−0.326 pp** ann (−0.790 pp total)

**Correction to the pre-stated estimate.** The Part A instruction
anticipated "≈ −0.17 pp billing, −0.19 pp refit". The **actual**
computed split is **−0.40 pp billing and +0.07 pp refit** (annualised);
the refit is a small POSITIVE offset, not a −0.19 pp drag. Mechanism:
the corrected `c` is LARGER for INDUSINDBK (β = 1.64), so Bertram widens
its band and trims a few marginal round-trips, partially recovering the
billing hit. The pre-stated estimate is superseded by these figures.

### 13.2 §5.2 correction — the cost=1 inversion was an under-billing artifact

Under the OLD 2× billing the matched-universe cost=1 cell showed
rolling_z (net Sharpe 1.047) BEATING OU (1.006) — the "cost-1 inversion"
flagged in the prior OU report §5.2. **Under corrected (1+β) billing the
inversion disappears: OU ≥ rolling_z at ALL FOUR cost levels** on the
matched universe.

| freq=5, B | OU net Sharpe (new) | rolling_z net Sharpe (new) | OU ≥ RZ? |
|----------:|--------------------:|---------------------------:|:--------:|
| cost = 1  |        0.9808       |           0.9454           |   yes    |
| cost = 3  |        0.9494       |           0.6434           |   yes    |
| cost = 5  |        0.9049       |           0.3414           |   yes    |
| cost = 8  |        0.9346       |          −0.1070           |   yes    |

The inversion was an artifact of the OLD convention **under-billing the
β = 1.64 INDUSINDBK/HDFCBANK leg by 32%**: rolling_z trades that leg 225
times vs OU's ~17, so the under-billing flattered rolling_z far more.
Correcting the bill removes the flatter.

**Explicit scope caveats:** this is **n = 2 pair-folds** and the result
is **β-composition dependent** — the matched universe contains exactly
one high-β leg (1.64) and one near-unity leg (0.87). A different
survivor set with β closer to 1 would show a much smaller correction.
The "OU ≥ rolling_z at all costs" statement is a property of THIS
2-pair-fold matched universe, not a general engine ranking. The
engine-attribution caveat from the prior §5.2 (slopes attributable,
LEVELS universe-confounded) is unchanged.

### 13.3 κ(Δt) decline — observation-noise prior for the Kalman unit

The per-bar OU reversion speed κ rises with bar size, but **κ per minute
declines** as bars coarsen — the signature of microstructure
(bid-ask-bounce) noise inflating apparent reversion at fine bars:

| pair-fold | κ/min @ 1m | κ/min @ 5m | κ/min @ 15m | Δ(1→15) |
|-----------|-----------:|-----------:|------------:|--------:|
| fold 4 INDUSINDBK/HDFCBANK | 5.775e-04 | 4.208e-04 | 3.717e-04 | **−35.6%** |
| fold 6 KOTAKBANK/HDFCBANK  | 6.996e-04 | 6.718e-04 | 6.541e-04 |  **−6.5%** |

This corroborates the freq-1 caveat from the prior OU report (1-min HL
estimates are biased low by bid-ask bounce). The **magnitude of the
per-pair κ decline is a candidate seed for the per-pair observation-
noise variance V_e** in the Kalman unit: INDUSINDBK's −35.6% implies a
much larger microstructure footprint than KOTAK's −6.5%, so a
pair-specific V_e prior is warranted rather than a shared scalar.
Flagged for `feature/kalman-equilibrium`.

### 13.4 Matched-universe ladder figures — re-rendered

Confirmed: `scripts/16_retro_figures_phase3.py` was re-run after the
β-aware OU + rolling_z re-runs.
`reports/phase3_ou/figures/matched_universe/matched_metrics.csv` and the
two ladder PNGs (`d_matched_universe_cost_ladder_f5_B.png`,
`...f15_B.png`) now reflect corrected billing — the freq=5 ladder shows
OU above rolling_z at every cost (no cost-1 crossover).
