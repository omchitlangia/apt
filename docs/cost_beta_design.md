# Cost-accounting unit — β-proportional notional convention

**Branch:** `feature/cost-beta-notional` (off post-merge `main` at 134f57d)
**Status:** Discovery + design only. **No implementation, no behavior
changes** in this commit.
**Trigger:** Phase 3 OU report §6.5 — equal-notional billing
mis-charges round-trip cost by up to ±32% on the traded pair-folds;
biases the published net Sharpes upward at the OU best cell.

This document maps the current convention's footprint, defines the
corrected convention, surfaces re-stamping options, and ends with a
numbered question list that must be answered before any implementation
starts.

------------------------------------------------------------------------

## 1. Where the 2× equal-notional convention lives — exact paths & line ranges

### 1.1 Authoritative definition

- **`src/apt/intraday/costs.py:73-79`** —
  `CostBreakdown.cost_per_pair_round_trip_bps` returns
  `2.0 * cost_bps_per_leg`; `cost_log_per_pair_round_trip` divides by
  10⁴. **Hardcoded `2.0` multiplier.** Docstring at
  `src/apt/intraday/costs.py:28-31` states the convention explicitly:
  > A pair trade has TWO legs, so the cost log deducted on each exit
  > is `2 × cost_bps_per_leg / 10_000` — same convention as the Phase
  > 2A engine (`run_walkforward(cost_bps_per_leg=...)`), which keeps
  > a single knob.

### 1.2 Deduction sites — where `cost_log` is subtracted from net P&L

- **`src/apt/intraday/backtest.py:166-168`** — signal-driven exit:
  `cost = cost_log_per_round_trip; net_pnl = gross_pnl - cost;
  net[exit_idx] -= cost`. No β awareness.
- **`src/apt/intraday/backtest.py:206-208`** — fold-boundary force-close
  (Regime B): same arithmetic, same β-ignorance.
- **`src/apt/backtest/walkforward.py:244-246`** — Phase 2A daily
  signal-driven exit: same arithmetic. Cost computed once for the
  whole walk-forward at **`src/apt/backtest/walkforward.py:403`**:
  `cost_log_per_round_trip = 2.0 * cost_bps_per_leg / 10_000.0`.
- **`src/apt/backtest/walkforward.py:274-276`** — Phase 2A daily
  fold-boundary force-close: same arithmetic.

### 1.3 Cost re-stamp helpers (used by spread-sweep)

- **`scripts/13_phase3_intraday.py:380-419`** — `_net_pnl_for_cost`:
  re-derives net per-bar and per-trade by deducting `new_cost_log`
  on exit bars. Takes `new_cost_log` directly as a scalar — **no β
  applied**; receives whatever the caller computes.
- **`scripts/13_phase3_intraday.py:575-577`, `:904`, `:955`, `:1033`** —
  callers passing `new_cost_log = CostBreakdown(...).cost_log_per_pair_round_trip`
  (which is the 2× equal-notional scalar). Every spread-sweep cell
  reaches `_net_pnl_for_cost` with the same `2× per-leg` figure.
- **`scripts/15b_phase3_rolling_baseline.py:178`, `:216`, `:255`** —
  same pattern in the new rolling-z baseline runner.
- **`scripts/15_phase3_ou.py:255-257`** — OU orchestrator computes
  `cb.cost_log_per_pair_round_trip` and uses it as `cost_log_per_round_trip`
  to seed `run_pair_fold`; same 2× convention propagates.

### 1.4 Trade-CSV columns that encode the convention

- **`scripts/13_phase3_intraday.py:1029, 1061-1062`** —
  `n_legs = 2` and `cost_bps_excl_spread_per_leg_rt = FIXED_PER_LEG_RT`
  written into `trades_two_regime_3bps.csv` per row. **These two
  columns assert the 2× convention as data**, so any reader that
  reconstructs net from these columns inherits it.
- **`tests/intraday/test_backtest_cost_pin.py`** — pins the zero-move
  round-trip to realize exactly `−c` where `c =
  cost_log_per_pair_round_trip`. The test ENFORCES the 2× convention
  — it will fail under a β-corrected cost without an update. Lines
  noting the equal-notional convention: `tests/intraday/test_backtest_cost_pin.py:7`.

### 1.5 Phase 2A daily analogue — same defect

- **`src/apt/config.py:141`** — `cost_bps_per_leg: float = 25.0`
  (the daily default). With the 2× convention this is **50 bps per
  pair round-trip** which the project documents as conservative.
- **`src/apt/backtest/walkforward.py:31-34`** — docstring locks the
  convention in for the daily engine:
  > Cost model: `cost_bps_per_leg` is the round-trip cost of ONE leg
  > (brokerage + STT + slippage). Two legs per pair trade, so the
  > total log-cost deducted per round trip is
  > `2 × cost_bps_per_leg / 10000`.
- **`src/apt/backtest/walkforward.py:403`** — the `2.0 * ... / 10_000`
  multiplication.
- **Phase 2A daily reports** (`reports/backtest_trades.csv`,
  `reports/phase2_per_pair.csv`, `reports/backtest_portfolio_*.csv`)
  all carry the **same** equal-notional inflation. **The 25 bps per-
  leg convention is the same defect, just at 8× larger magnitude
  per trade.**

### 1.6 Summary of the audit

| site                                                | type            | β-aware? |
|-----------------------------------------------------|-----------------|---------:|
| `src/apt/intraday/costs.py:73-79`                   | definition      |       no |
| `src/apt/intraday/backtest.py:166-168`              | deduction site  |       no |
| `src/apt/intraday/backtest.py:206-208`              | deduction site  |       no |
| `src/apt/backtest/walkforward.py:244-246, 274-276`  | deduction site  |       no |
| `src/apt/backtest/walkforward.py:403`               | conversion site |       no |
| `scripts/13_phase3_intraday.py:380-419`             | re-stamp helper |       no |
| `scripts/13_phase3_intraday.py:1029-1062`           | trade-CSV cols  |       no |
| `scripts/15_phase3_ou.py:255-257`                   | call site       |       no |
| `scripts/15b_phase3_rolling_baseline.py:178+`       | call site       |       no |
| `tests/intraday/test_backtest_cost_pin.py`          | enforcement test |  no (will fail) |

------------------------------------------------------------------------

## 2. Notional convention the P&L implies

### 2.1 How the spread is constructed

Source: **`src/apt/signals/spread.py:67`** —
`compute_spread(p1, p2, beta, intercept)` returns
`log(p1) - beta * log(p2) - intercept`.

Orientation per pair is decided by **`src/apt/signals/cointegration.py:154-164`**:
`engle_granger_best_direction` runs both `(y=A, x=B)` and `(y=B, x=A)`
and returns whichever has the smaller ADF p-value (more-negative ADF
stat). The returned `EGResult` carries `y_sym`, `x_sym`, `alpha`, `beta`,
which are persisted to `Pair(y_sym, x_sym, alpha, beta, half_life, ...)`
at **`src/apt/backtest/walkforward.py:64-74`**.

`Pair.beta` is therefore **always > 0** for the orientation chosen by
EG (cointegration coefficients are signed but the chosen direction
gives a positive β by convention; if both directions yielded negative β,
the pair would not have cleared cointegration robustness).

### 2.2 Implied notional

The spread move
```
Δ(log Y) − β · Δ(log X)
```
is the **log-return** of a long-Y / short-X hedge sized so:

- **Y leg** = 1 unit of log-notional (i.e. 1 INR of Y).
- **X leg** = β units of log-notional (i.e. β INR of X).

(In log-space, a 1 bp move in Y contributes 1 bp to the spread; a
1 bp move in X contributes β bp. The hedge ratio is β-of-X-per-unit-Y.)

A pair round-trip therefore touches:

- 1 × `cost_bps_per_leg` on the Y leg (entry+exit, one leg, one RT)
- β × `cost_bps_per_leg` on the X leg

**Correct cost per pair round-trip = `(1 + β) × cost_bps_per_leg`.**

The current implementation uses `2 × cost_bps_per_leg`. The ratio is
```
true / current = (1 + β) / 2
```
which is exactly the **`(1+β)/2`** scalar already reported per pair in
`reports/phase3_ou/ou_pair_fold_diag.csv` (column `one_plus_beta_over_2`).

| β regime    | true/current cost ratio | direction of bias on net P&L |
|-------------|------------------------:|------------------------------|
| β = 1.00    | 1.00                    | exact — no bias              |
| β > 1       | > 1.00                  | UNDER-charged ⇒ net biased **upward**   |
| β < 1       | < 1.00                  | OVER-charged ⇒ net biased **downward**  |

### 2.3 Worked example — INDUSINDBK/HDFCBANK fold 4 at 3 bps

From `reports/phase3_ou/ou_pair_fold_diag.csv`:
- β = 1.6434
- (1+β)/2 = 1.3217

At cost = 3 bps:
- `cost_bps_per_leg` = `FIXED_PER_LEG_RT + spread_bps` = 4.5 + 3 = **7.5 bps/leg/RT**

**Current 2× equal-notional:**
- `cost_per_pair_RT_bps` = 2 × 7.5 = **15.0 bps**
- `cost_log` = 0.0015 per round-trip

**Corrected (1 + β) ×:**
- `cost_per_pair_RT_bps` = (1 + 1.6434) × 7.5 = 2.6434 × 7.5 = **19.83 bps**
- `cost_log` = 0.001983 per round-trip
- Δ per trade = +0.000483 (cost **under-billed** by 32%)

INDUSINDBK/HDFCBANK fold 4 ran 17 trades at the headline 3-bps cell.
- Gross log P&L (from §6.9): +0.1581
- Cost (current 2× conv.): 17 × 0.0015 = 0.0255 ⇒ net log P&L +0.1326 (matches §6.9)
- Cost (corrected (1+β)×): 17 × 0.001983 = 0.0337 ⇒ net log P&L **+0.1244**
- Pair-fold net %, current: +14.18% → corrected: **+13.25%** (≈ −1 pt)

KOTAKBANK/HDFCBANK fold 6 (β = 0.872, (1+β)/2 = 0.936) at the same cell:
- (1 + 0.872) × 7.5 = 14.04 bps; cost_log = 0.001404 per RT
- Δ per trade = −0.000096 (cost **over-billed** by 6.4%; corrected
  cost is slightly *lower*)
- 17 trades × 0.001404 = 0.0239 vs 0.0255 current
- Pair-fold net %, current: +29.06% → corrected: **+29.27%** (+0.2 pt)

**Headline portfolio re-stamp** (best cell, n=2):
- Current: net log P&L = 0.3877 ⇒ net total 47.34% ⇒ net ann ≈ **21.24%**
- Corrected: 0.1244 + 0.2568 = **0.3812** ⇒ net total 46.39% ⇒
  net ann ≈ **20.85%**

So at the headline cell the **billing-only** β-correction moves the net
annual by **−0.40 pt** (21.24% → ~20.85% on the sum-of-logs basis here;
20.84% on the engine's portfolio-mean basis — see [ERRATUM] below). The
two pair-folds have OFFSETTING signs (β = 1.64 over-pays, β = 0.87
under-pays) but do NOT net to neutral: INDUSINDBK's +32% under-billing
on the larger-P&L pair-fold dominates.

> **[ERRATUM 2026-06-12 — corrects this subsection]**
> This worked example originally reported the billing-only corrected
> annual as **"21.07%" (Δ −0.17 pt)**. That figure was an **arithmetic
> error in the annualization step**, NOT a per-leg/per-pair halving
> error. Diagnosis:
> - The per-trade billing deltas above are **correct and unchanged**:
>   `Δ/trade = ((1+β) − 2) × cost_log_per_leg` = **+0.000483**
>   (INDUSINDBK, β=1.64) and **−0.000096** (KOTAK, β=0.87). These match
>   the re-stamp in `phase3_cost_beta_report.md` §13.1 bit-for-bit, so
>   billing is applied per-pair `(1+β)×per_leg`, not double-halved.
> - The error was downstream: the corrected sum-of-logs (0.3812) was
>   transcribed to **21.07%** annual. Applying this subsection's OWN
>   total→annual ratio (21.24/47.34 ≈ 0.449) to its OWN corrected total
>   change (−0.95 pt) yields **≈ −0.43 pt → ~20.8%**, not 21.07%.
> - The authoritative billing-only figure on the engine's
>   **portfolio-mean** aggregation (mean across active pairs per session,
>   the `metrics_ou.csv` basis) is **net ann 20.8436%** (Δ −0.3977 pt).
>   See `phase3_cost_beta_report.md` §13.1 for the full
>   billing-vs-refit decomposition and the unrounded legs.
> - The §6.5 "−1.6 pt linear-mean" estimate used the subset-mean ratio
>   (1+β)/2 ≈ 1.13 applied UNIFORMLY to both pairs, which over-charges
>   KOTAK (true ratio 0.936); it is superseded by the per-pair re-stamp.

**The headline cell moves materially under the correction; other cells
(e.g. v2 Phase 3 1-min B on the full 19-pair-fold sample, daily Phase
2A) move per `phase3_cost_beta_report.md` §4.**

------------------------------------------------------------------------

## 3. Re-stamp feasibility from existing trade CSVs

Each round-trip's `gross_log_pnl` is **β-invariant** (it depends only
on `direction × Δspread`, and `Δspread = Δlog(Y) − β·Δlog(X)` carries
the β internally). To re-derive `net_log_pnl` under the corrected
convention, all we need is the per-trade β joined onto the trade row.

### 3.1 Intraday OU + rolling_z baseline — RE-STAMPABLE

- `reports/phase3_ou/trades_ou.csv` — has `fold_id, pair, ...,
  gross_log_pnl, cost_log, net_log_pnl, exit_reason`. **No β column**,
  but `reports/phase3_ou/ou_pair_fold_diag.csv` carries
  `pair_beta, one_plus_beta_over_2` per `(fold_id, pair, freq_min)`.
  **Joinable on (fold_id, pair) [and freq_min for OU's freq-keyed
  rows; rolling_z baseline's β is freq-invariant].**
- `reports/phase3_ou/trades_rolling_baseline.csv` — same. Joinable
  via `ou_pair_fold_diag.csv` or `reports/phase3/fold_pairs.csv`.

### 3.2 Intraday v2 (Phase 3 rolling_z) — RE-STAMPABLE

- `reports/phase3/trades_two_regime_3bps.csv` — has `fold_id, pair,
  ..., gross_log_pnl, net_log_pnl_at_3bps`. **β not in trade CSV.**
- `reports/phase3/fold_pairs.csv` carries `fold_id, pair, ..., alpha,
  beta, half_life_days`. **Joinable on (fold_id, pair)** — direct.
- Note: only 3-bps trades are persisted; the v2 metrics CSVs at
  other spread levels are aggregates. Re-stamp would either:
  - re-derive 3-bps net under corrected conv. for the trade CSV
    only (cheap), and
  - flag that the other-cost rows in `metrics_two_regime.csv`
    require re-running script 13 to be re-stamped exactly
    (because per-trade `gross_log_pnl` from the cache is in memory
    only at run time, not persisted across spread levels except
    in the 3-bps trade CSV).

### 3.3 Phase 2A daily — RE-STAMP NEEDS A NEW EMIT

- `reports/backtest_trades.csv` — has `fold_id, pair, ..., gross_log_pnl,
  cost_log, net_log_pnl`. **No β column.**
- No analogous `fold_pairs.csv` for the daily walk-forward (Phase 2A
  doesn't emit one; the `Pair` objects live in memory during the
  walk). `reports/phase2_per_pair.csv` has per-pair aggregates but
  no β.
- `data/pairs/cointegrated_pairs.parquet` carries β but is **not
  fold-specific** — daily uses fold-train-window-re-fitted β, which
  differs from the global cointegration β.
- **[TODO]**: emit a daily `fold_pairs.csv` (or attach β to
  `backtest_trades.csv` at run time). This is a one-line addition to
  `scripts/10_backtest.py` / `walkforward.run_walkforward`'s emit
  paths. Scope question for the design unit.

### 3.4 Summary

| dataset                                       | retroactive re-stamp from existing CSVs? |
|-----------------------------------------------|------------------------------------------|
| Phase 3 OU (3-bps trades)                     | yes (join ou_pair_fold_diag.csv)         |
| Phase 3 OU (other costs)                      | yes (gross is in trades_ou.csv per cost) |
| Phase 3 rolling_z baseline (all costs)        | yes (trades_rolling_baseline.csv has gross per cost) |
| Phase 3 v2 (3-bps trades)                     | yes (join fold_pairs.csv)                |
| Phase 3 v2 (other costs)                      | no — re-run script 13 *or* emit fold_pairs + per-cost trades |
| Phase 2A daily (any cost)                     | no — needs one-shot β emit + re-derivation |

------------------------------------------------------------------------

## 4. Open questions (numbered) — must be answered before implementation

1. **Retroactive re-stamp vs forward-only correction.** Two paths:
   - (a) Re-stamp every persisted result table (Phase 3 v2, Phase 3
     OU, Phase 3 rolling_z, Phase 2A daily) under the corrected
     convention; publish a delta-table in the OU report § as
     errata. Keeps the historical methodological line consistent but
     touches a lot of artifacts at once.
   - (b) Land the corrected convention forward; mark every pre-fix
     result table with a "[uncorrected — equal-notional, see
     cost_beta_design.md]" caveat; do NOT re-stamp the historical
     CSVs.

   Which path? If (a), is the scope every dataset in §3 or only the
   Phase 3 OU best-cell narrative?

2. **Does corrected `c` re-enter the Bertram solver?** The Bertram
   threshold `a*` is the root of an objective whose only data input
   is `c = cost_log_per_pair_round_trip`. Under the corrected
   convention `c` becomes pair-specific: `c_pair = (1 + β_pair) ×
   cost_log_per_leg`. So **each pair-fold gets its own Bertram
   threshold** instead of sharing the equal-notional `c`.

   - (a) Re-solve `a*` per pair-fold under corrected `c`, re-run the
     OU cells. Tests `test_ou_mc.py` validator still works pair-wise.
     A* changes by up to ~10-15% on extreme-β pair-folds; trade
     counts change.
   - (b) Keep the existing equal-notional `a*` (compute the corrected
     net at the SAME a* the run produced); re-stamp only the net.
     The threshold is then slightly suboptimal under the corrected
     cost. Acceptable approximation? Or do we want the
     joint correction?

   This unit is, mechanically, big enough that we should pick one
   and commit. **(b) is much cheaper; (a) is methodologically
   correct.** Pick.

3. **Phase 2A daily 25 bps convention — in or out of scope?** The
   defect applies identically to daily (`run_walkforward(cost_bps_per_leg=25)`
   → 50 bps per pair RT, equal-notional). The daily β distribution
   is different from intraday's (we have not tabulated it). Two
   sub-questions:
   - (a) Is the daily β distribution skewed enough to matter? If the
     daily distribution is roughly symmetric around 1.0 (population
     mean ≈ 1.0), the net effect on portfolio metrics is small.
   - (b) If yes, is the daily fix in this unit or a separate one?
     The intraday and daily code share `walkforward.py:403`'s
     `2.0 * ... / 10_000.0` line; fixing one without the other
     creates an inconsistency.

4. **Trade-CSV schema versioning.** The 2× convention is encoded in
   trade-CSV columns at `scripts/13_phase3_intraday.py:1061-1062`:
   `cost_bps_excl_spread_per_leg_rt` and `n_legs = 2`. Under the
   corrected convention we need either:
   - (a) Add `pair_beta` and `cost_per_pair_factor` columns (= (1+β)),
     deprecate `n_legs`. New CSVs version-bump (e.g.
     `trades_two_regime_3bps_v2.csv`).
   - (b) Replace `n_legs` with a per-row `notional_factor` = (1+β),
     overwrite the file in place.
   - (c) Keep `n_legs=2` and add `pair_beta` separately; downstream
     readers compute `notional_factor` themselves.

   This affects every downstream consumer that reads
   `trades_two_regime_3bps.csv` (the OU report, the per-pair plots,
   the v2 walk-forward report). Pick a schema and pin it before
   touching the emit sites.

5. **Test contract.** `tests/intraday/test_backtest_cost_pin.py`
   currently asserts that a zero-move round-trip realizes net log =
   `−cost_log_per_pair_round_trip` (the equal-notional scalar). Under
   the corrected convention this assertion changes shape: a zero-move
   round-trip on a pair with β realizes `−(1+β)/2 × current_c`. Test
   update direction:
   - (a) Parametrize the test over a synthetic β grid; assert the
     β-proportional relationship.
   - (b) Keep the current pin at β = 1 (synthetic) and add a NEW
     parameterized test for β ≠ 1.

   Plus: does the MC validator at `tests/stats/test_ou_mc.py` need
   any update? (It currently passes `cost_log_per_round_trip` as a
   scalar; if Bertram is re-solved per pair-fold, the validator
   should also be parameterized over β. If we picked Q2(b) — net-only
   re-stamp — the MC validator does not change.)

6. **Direction of the "1 unit / β units" convention.** The implied
   notional is per the spread arithmetic in
   `src/apt/signals/spread.py:67`. **Confirm**: is the convention
   really "Y leg = 1, X leg = β" (as derived above) — or is the
   project's intended sizing "equal dollar on each leg, hedge ratio
   maintained by re-balancing daily"? The latter (dollar-neutral
   re-balanced) carries the equal-notional cost as a feature, not a
   bug. The current code's silence on this is the source of the
   ambiguity; the design unit must declare the intended hedge
   convention before changing arithmetic.

7. **What population matters for the Phase 3 OU re-stamp?** The OU
   best cell (n=2) is dominated by INDUSINDBK/HDFCBANK at β = 1.64.
   The β-correction on that single pair-fold:
   - shrinks its 17-trade net contribution from +0.1326 to +0.1244;
   - the portfolio net moves from 47.34% to 46.42% (−0.9 pt of
     total return; ~−0.2 pt of annualized).

   Is this enough impact to gate the OU report's "exploratory n=2"
   claim, or is the n=2 caveat already absorbing this magnitude?
   (If absorbed, then the OU re-stamp is a numeric refinement, not
   a finding flip.)

8. **Bertram σ_eq invariance under cost convention.** `σ_eq` is fit
   from the TRAIN spread series (purely OU-process parameters, no
   cost). It does NOT change under the cost convention. Confirm — and
   note in §8.7 of `docs/ou_thresholds_design.md` as part of the
   leakage statement.

------------------------------------------------------------------------

## 5. Stop conditions

This document ends here per the unit's hard rules:
> **No implementation, no behavior changes** in this commit. Surface
> the questions, then stop.

**Next deliverable** (only after questions 1-6 are answered): a
follow-up commit on the same branch that lands the code+test changes,
re-stamps the chosen artifacts, and updates `docs/phase3_ou_report.md`
with the corrected numbers as an errata appendix.

------------------------------------------------------------------------

**End of design doc.**
