# Phase 4 OMNIBUS — single unattended run

**Branch:** `feature/phase4-omnibus` (off `feature/kalman-equilibrium` @ 19c9f47).
**Inherits:** the μ-only per-session Kalman engine (Unit K) AND the (1+β)
per-pair cost-billing convention. Reporting per
[`docs/reporting_standard.md`](reporting_standard.md). **Nothing is merged.**

Every number in this report traces to a persisted CSV under
`reports/phase4/<subunit>/` or `plots/phase4/<subunit>/` (PNGs gitignored;
companion CSVs ship beside each figure). Drivers: `scripts/phase4/s{1..6}_*.py`.
Regenerate any subunit by re-running its driver against the persisted
Phase-3 artifacts.

---

## 0. ASSUMPTIONS register (labeled defaults + [TODO]s)

Every labeled default and every deferred computation, with the value used,
why, and what would change the dependent output. Mirrored into the
fix-later punch list (§9), ranked by headline impact.

| # | assumption / [TODO] | value used | section | why | what would change it / fix |
|---|---------------------|------------|---------|-----|----------------------------|
| A1 | DSR/PBO sample length | n=2 pair-folds, ~504 sessions/cell | §2 | only 2 survivors exist on the matched NSE universe | broader universe (Johansen §5, crypto §6) → longer, cross-sectional return matrix |
| A2 | Honest trial count N | see §2a enumeration | §2 | reconstructed from design docs + grids | a pre-registered trial ledger maintained going forward |
| A3 | KOTAK fold-6 μ_t path not persisted | [TODO data] | §1c(d) | Unit-K overlay export set only included INDUSINDBK | add KOTAK to overlay set in `scripts/17` and re-emit (no model change) |
| A4 | fold-6 INDUSINDBK σ_eq_resid | [TODO data] | §1c(d) | not a traded survivor → no trade to recover σ_eq from | persist residual-fit σ_eq in `selection_table.csv` |
| A5 | β-escalation hyperparameters | one global config, train-only | §3 | matches Unit-K discipline (no per-fold tuning) | documented in `docs/beta_escalation_design.md` |
| A6 | coint-stability gate | window=60d, step=5, ADF p>0.10, 3 consec → blacklist | §4 | no prior calibration; chosen as a standard rolling-ADF setup | gates 19/19 at default (and 15/19 even at p>0.50/120d) → threshold needs a broad-universe base rate to calibrate; NSE binding [TODO]-pending-breadth |
| A7 | crypto taker fee | **5 bps/side** (10 bps/leg RT) | §6c | perp taker ≈ 0.05%; venue-dependent, not in data | confirm against the actual venue fee schedule; result is robust (gross already negative) |
| A8 | crypto funding-rate series | **ABSENT → funding = 0** [TODO data] | §6c | no funding file in `/home/om/data_combined` | pull a funding series; would add carry cost to multi-day Regime-B holds |
| A9 | crypto risk-management | `stop_z = 3.5` only, NOT tuned | §6d | long-deferred crypto risk framework; single hard z-stop placeholder | a dedicated risk-framework unit on crypto |
| A_LIQ | crypto liquidity gate | median daily quote-vol ≥ **$10M** | §6b | standard "tradeable major" cut | keeps 22/30 symbols; at $50M only 7 survive (too thin for breadth) |
| A10 | crypto engine = rolling-z only | OU/Bertram + Kalman = [TODO scope] | §6e | scaffold stage; reused the asset-agnostic engine first | port the adaptive engines → the fair crypto retest |
| A11 | crypto Regime A (intraday) | [TODO] — daily bars only | §6c | no intraday resample built yet | resample 1m→{1h,4h}; enables Regime A + a bar-freq sweep |

(Rows A5–A11 are populated as §§3–6 run; see each section.)

---

## 1. DSR / PBO gate — pinned at top (NSE + crypto)

> **This block is the gate and is pinned here regardless of run order.**
> NSE numbers from §2 (`reports/phase4/dsr_pbo/`). Crypto from §6e.

**Headline (NSE matched universe, N = 46 honest trials, V = 0.00182):**

| universe | selected candidate | ann net Sharpe | per-period SR | DSR | DSR p-value | clears N=46 bar? |
|----------|--------------------|---------------:|--------------:|----:|------------:|:----------------:|
| NSE matched | **kalman** best (f5/c1) | 2.145 | 0.135 | **0.808** | 0.192 | **yes** (only one) |
| NSE matched | frozen-OU best (f15/c1) | 1.003 | 0.063 | 0.234 | 0.766 | no |
| NSE matched | rolling_z best (f5/c1) | 0.945 | 0.063 | 0.230 | 0.770 | no |

**PBO (CSCV, 24 NSE cells, S=16, 12 870 splits) = 0.104.**
Source: [`2b_dsr.csv`](../reports/phase4/dsr_pbo/2b_dsr.csv),
[`2c_pbo.csv`](../reports/phase4/dsr_pbo/2c_pbo.csv).

Reading: under N=46 trials the *expected* best-by-luck per-period Sharpe is
SR₀ = 0.096 (annualized ≈ 1.52). Only the **kalman** best (2.145 ann) clears
it (DSR 0.808); frozen-OU (1.003) and rolling_z (0.945) sit **below** the
luck bar (DSR ≈ 0.23). PBO = 0.104 says the in-sample-best cell lands below
the OOS median only ~10% of the time — **low overfitting probability**. But
the kalman p-value is 0.192 — it does **not** reach conventional
significance, and DSR(kalman) decays to 0.647 at N=200
([`2b_dsr_sensitivity_to_N.csv`](../reports/phase4/dsr_pbo/2b_dsr_sensitivity_to_N.csv)).

| universe | candidate | per-period SR | DSR | DSR p-value | PBO |
|----------|-----------|--------------:|----:|------------:|----:|
| **crypto** (rolling_z, Regime B, 81 pairs × 1440 d) | portfolio @ 3 bps | −0.008 | **0.367** | 0.633 | **0.439** |

The crypto DSR/PBO **are** more trustworthy than the NSE ones — they run on 81
pair-return columns over 1440 daily sessions (real cross-sectional width, no
2-fold concatenation). What they show is a clean **negative**: the naive
rolling-z crypto pairs strategy has **no edge** (negative Sharpe, DSR 0.37
below the 0.5 line, PBO 0.44). See §6.

**MANDATORY caveat (applies to every DSR/PBO number in this report):** with
n = 2 NSE pair-folds the per-period series are short and are the
**concatenation of two disjoint folds** (≈2017 and ≈2019) — they violate the
iid assumption behind DSR and mix two regimes inside every CSCV block.
**Treat the NSE DSR/PBO as INDICATIVE, not definitive.** What would make them
trustworthy: universe breadth (many independent pair-folds), which §5
(Johansen) and §6 (crypto) begin to supply.

---

## 2. DSR / PBO — full results

Driver: `scripts/phase4/s2_dsr_pbo.py` → `reports/phase4/dsr_pbo/`. Modules
`apt.stats.dsr` (Bailey & López de Prado 2014) and `apt.stats.pbo` (CSCV,
Bailey et al. 2017), both unit-tested. Per-period return matrix: 504 daily
sessions × 24 none-stop candidate cells (8 per engine), matched universe.

### 2a — honest trial count

Source: [`2a_trial_ledger.csv`](../reports/phase4/dsr_pbo/2a_trial_ledger.csv).

| source | count |
|--------|------:|
| OU grid (freq{1,5,15}×regime{A,B}×cost{1,3,5,8}×stop{none}=24 + cost3×hard=6) | 30 |
| rolling_z coarse grid | 4 |
| kalman cells (freq{5,15}×cost{1,3,5,8}) | 8 |
| kalman re-anchor H-grid {∞,20,10,5} (train-only selection knob) | 4 |
| **TOTAL explicit N** | **46** |

Implicit selections (HL bands {A,B,C}; best-of-cell reporting) compound N
further; the headline uses N = 46 and §2b's sensitivity sweep covers
N ∈ {1, 30, 46, 100, 200}.

### 2b — Deflated Sharpe Ratio

Source: [`2b_dsr.csv`](../reports/phase4/dsr_pbo/2b_dsr.csv). V = 0.00182
(per-period cross-trial Sharpe variance over 28 non-degenerate evaluated
cells; 3 catastrophic hard-stop cells with annualized Sharpe < −2 dropped).

| engine | cell | per-period SR | skew | kurtosis | SR₀ | PSR(0) | **DSR** | p-value |
|--------|------|--------------:|-----:|---------:|----:|-------:|--------:|--------:|
| kalman | f5/c1 | 0.135 | −0.035 | 6.60 | 0.096 | 0.999 | **0.808** | 0.192 |
| frozen-OU | f15/c1 | 0.063 | −0.204 | 5.73 | 0.096 | 0.920 | 0.234 | 0.766 |
| rolling_z | f5/c1 | 0.063 | +0.511 | 22.24 | 0.096 | 0.922 | 0.230 | 0.770 |

Sensitivity ([`2b_dsr_sensitivity_to_N.csv`](../reports/phase4/dsr_pbo/2b_dsr_sensitivity_to_N.csv)):
DSR(kalman) = 0.999 (N=1) → 0.848 (N=30) → 0.808 (N=46) → 0.725 (N=100) →
0.647 (N=200). It stays above 0.5 across the whole range but never reaches
p < 0.05.

### 2c — PBO via CSCV

Source: [`2c_pbo.csv`](../reports/phase4/dsr_pbo/2c_pbo.csv); figure
[`pbo_logit_distribution.png`](../plots/phase4/dsr_pbo/pbo_logit_distribution.png).
**PBO = 0.104** over 12 870 symmetric splits (S=16) of the 24-cell matrix.
The logit distribution is centred well above 0 (median λ = 1.66): the
in-sample-best cell is below the OOS median only ~10% of the time — low
overfitting probability for the grid as a whole.

### 2d — honesty caveat (MANDATORY)

n = 2 pair-folds ⇒ the per-period series is the concatenation of two disjoint
folds (≈2017 and ≈2019). This **strains every assumption**: DSR treats
returns as iid (they are serially dependent and regime-mixed); CSCV blocks
straddle the fold join. The DSR/PBO numbers are therefore **INDICATIVE, not
definitive**. They would become trustworthy with **universe breadth** — many
independent pair-folds so the return matrix has genuine cross-sectional width
and the CSCV blocks are within-regime. §5 (Johansen) and §6 (crypto) are the
breadth path; §6e re-runs this exact machinery on crypto.

### 2 — verdict

The gate **separates the engines cleanly**: only the adaptive (kalman) arm
clears the N=46 multiple-testing bar (DSR 0.808), while frozen-OU and
rolling_z fall below the luck threshold. PBO is low (0.104). **But** kalman's
p-value is 0.192 — not significant — and the whole result rests on a
2-fold concatenation. **Indicative support for the adaptive arm; not a
deployable, significance-cleared edge.**

---

## Section 1 — Unit K verification (persisted artifacts only, no re-runs)

H stayed at 20; nothing was re-tuned. Driver:
`scripts/phase4/s1_unit_k_verification.py` →
`reports/phase4/verification/*.csv` + `plots/phase4/verification/*`.

### 1a — reconciliation: 20.886 vs 20.914967

Source: [`1a_reconciliation.csv`](../reports/phase4/verification/1a_reconciliation.csv).

- **20.914967 is canonical.** It is the `net_ann_pct` of the frozen-OU cell
  `(ou, 5-min, Regime B, 3 bps, stop=none)` — the **C-leg** (new a*, new
  (1+β) billing) — and matches the persisted
  `reports/phase3_ou/metrics_ou.csv` row to 6 dp. It is also cited verbatim
  in the **committed** `docs/phase3_cost_beta_report.md` §13.1, where the
  best-cell decomposition is pinned: **A** {old a*, old bill} = 21.241294,
  **B** {old a*, new bill} = 20.843599, **C** {new a*, new bill} =
  20.914967 (billing −0.397695 + refit +0.071368 = total −0.326327 = C−A).
- **Provenance correction.** `metrics_ou.csv` is **gitignored**
  (`reports/` is ignored; regenerated by `scripts/15_phase3_ou.py`). It does
  **not** exist as a tracked blob at 65b15f6 — so the instruction's phrasing
  "in metrics_ou.csv on main at 65b15f6" is imprecise: the committed anchor
  for 20.914967 is `docs/phase3_cost_beta_report.md`, not a committed CSV.
- **20.886 is untraceable.** It appears in **zero** working-tree files and
  **zero** git blobs across all history (`git log --all -S"20.886"` and
  `-S"20.88"` both empty). It equals none of A/B/C; it lies between the
  billing-only B-leg (20.843599) and the official C-leg (20.914967). Most
  likely an intermediate chat paste-back during Unit C, superseded by the
  finalized A/B/C decomposition. **It is not a number to carry forward.**
- **"Fix all occurrences":** there are **no in-repo occurrences** of 20.886
  to fix. This reconciliation is the errata note; it is recorded here and in
  `1a_reconciliation.csv`.

### 1b — full 8-cell table (kalman_mu vs frozen-OU vs rolling_z, matched)

Source: [`1b_eight_cell_table.csv`](../reports/phase4/verification/1b_eight_cell_table.csv).
All three engines on the **same matched 2 pair-folds** `{(fold 4,
INDUSINDBK/HDFCBANK), (fold 6, KOTAKBANK/HDFCBANK)}`, recomputed from the
persisted per-session returns (cross-checks `matched_metrics.csv` to
max |Δnet%| = 0.0000). The original Unit-K paste omitted cells 5/5, 15/1,
15/5; all 8 are present here.

| freq | cost | k_ntr | ou_ntr | rz_ntr | k_gross% | k_net% | ou_net% | rz_net% | k_net_Sh | ou_net_Sh | rz_net_Sh | k_net_ann% | k_net_DD% |
|-----:|-----:|------:|-------:|-------:|---------:|-------:|--------:|--------:|---------:|----------:|----------:|-----------:|----------:|
| 5 | 1 | 69 | 36 | 225 | 161.35 | 139.94 | 48.27 | 37.67 | 2.145 | 0.981 | 0.945 | 54.90 | −21.02 |
| 5 | 3 | 60 | 34 | 225 | 142.31 | 119.16 | 46.20 | 24.34 | 1.959 | 0.949 | 0.643 | 48.04 | −21.08 |
| 5 | 5 | 53 | 32 | 225 | 127.01 | 103.13 | 43.17 | 12.30 | 1.781 | 0.905 | 0.341 | 42.53 | −20.68 |
| 5 | 8 | 51 | 30 | 225 | 147.68 | 115.20 | 44.28 | −3.60 | 1.937 | 0.935 | −0.107 | 46.70 | −20.37 |
| 15 | 1 | 66 | 34 | 88 | 161.79 | 141.29 | 49.91 | 18.68 | 2.145 | 1.003 | 0.613 | 55.34 | −21.48 |
| 15 | 3 | 57 | 31 | 88 | 149.06 | 126.21 | 39.22 | 14.01 | 2.013 | 0.830 | 0.473 | 50.40 | −21.65 |
| 15 | 5 | 52 | 28 | 88 | 138.68 | 113.96 | 30.97 | 9.51 | 1.901 | 0.684 | 0.331 | 46.27 | −20.71 |
| 15 | 8 | 41 | 26 | 88 | 101.94 | 80.13 | 33.54 | 3.11 | 1.524 | 0.745 | 0.113 | 34.21 | −20.56 |

(gross totals + gross Sharpe for all three engines and net_ann/net_DD for
OU and rolling_z are in the CSV; abbreviated here for width.)

### 1c — mechanical P&L decomposition of the kalman best cell (5-min, 3 bps)

From persisted trades/sessions only; no re-runs.

**(a) exit-reason mix + share of net** —
[`1c_a_exit_reason_mix.csv`](../reports/phase4/verification/1c_a_exit_reason_mix.csv).
Persisted labels were remapped onto the fixed five-category vocabulary
(`fold_boundary`→`fold_close`; see §1g note).

| engine | mean_revert | time_stop | fold_close | mean_revert share-of-net |
|--------|------------:|----------:|-----------:|-------------------------:|
| kalman | 57 (95.0%) | 2 (3.3%) | 1 (1.7%) | **+125.2%** |
| frozen-OU | 26 (76.5%) | 6 (17.6%) | 2 (5.9%) | +175.4% |

For kalman the 3 non-mean-revert exits are net-negative (time_stop −25.2% of
net, fold_close −0.08%), so mean_revert supplies >100% of net — reproducing
the Unit-K report's "57/60, 125%".

**(b) per-trade net P&L** —
[`1c_b_per_trade_dist.csv`](../reports/phase4/verification/1c_b_per_trade_dist.csv).

| engine | mean | median | p5 | p95 | worst | win-rate |
|--------|-----:|-------:|---:|----:|------:|---------:|
| kalman | 131.2 bps | 167.9 | −168.0 | 392.9 | −1476.3 | 0.900 |
| frozen-OU | 112.4 bps | 220.1 | −604.6 | 449.4 | −1619.6 | 0.824 |

The adaptive arm's edge is **downside compression**, not bigger winners: its
p5 (−168 bps) is dramatically tighter than frozen's (−605 bps) and its
median is *lower*. Removing the slow drift truncates the left tail.

**(c) overlap with frozen losing time_stop/fold_close windows** —
[`1c_c_overlap.csv`](../reports/phase4/verification/1c_c_overlap.csv).
Fraction of kalman gross accrued in windows where the frozen arm held a
position that exited time_stop/fold_close **at a loss**, same pair-fold:

| pair-fold | n frozen losers | fraction of kalman gross in-window |
|-----------|----------------:|-----------------------------------:|
| fold 4 INDUSINDBK | 3 | **−60.8%** (kalman *also lost* there) |
| fold 6 KOTAK | 3 | +31.5% |
| **aggregate** | 6 | **+0.23%** |

The kalman edge is **not** concentrated in the windows where frozen got
stuck: in fold 4 kalman lost money during those windows, in fold 6 it gained
moderately, and they nearly cancel (0.23% of total gross). This **refutes**
the "the money comes from rescuing frozen's stuck trades" story.

**(d) μ_t tracking (profit-truncation-by-tracking)** —
[`1c_d_mu_tracking.csv`](../reports/phase4/verification/1c_d_mu_tracking.csv).
INDUSINDBK fold 4 (the one traded survivor with a persisted μ overlay;
σ_eq_resid = 0.028347 recovered from the entry-Z identity):

- total session-to-session |Δμ| = **11.45 σ_eq units** over the test window
  — the anchor moves a lot.
- **22 of 28 trades (78.6%)** had the exit anchor μ_t move *toward the entry
  spread* during the holding period — direct evidence of the
  profit-truncation / win-rate-inflation mechanism the Unit-K report flagged.

KOTAK fold 6 μ-path and the fold-6 INDUSINDBK σ_eq normalization are
**[TODO data]** (A3, A4): the Unit-K run persisted overlays only for
INDUSINDBK. The fix is mechanical (add KOTAK to the overlay export set in
`scripts/17`); no model change.

### 1d — the "4/4 admissible" denominator

Source: [`1d_admissible_denominator.csv`](../reports/phase4/verification/1d_admissible_denominator.csv).
The denominator is **4 = 2 pair-folds × 2 frequencies** (the static-HL gate
gives the same 2 survivors at each of freq 5 and 15; each is scored once per
H). So "4/4 admissible at H=20" means both survivors at both frequencies
passed the absorption guard. It is **not** 4 independent pair-folds.

### 1e — reviewer pre-registered predictions (scored)

Source: [`1e_predictions.csv`](../reports/phase4/verification/1e_predictions.csv).

| prediction | number | verdict |
|------------|-------:|---------|
| ≥80% mean_revert exits (best cell) | 95.0% | **MET** |
| avg capture 200–260 bps/trade | 147.9 bps gross (all trades) | **NOT MET** |
| majority of kalman gross in frozen losing time_stop windows | 0.23% | **NOT MET** |
| max_DD better than frozen −22% | −21.08% vs −22.54% | **MET** |

Note on capture: 147.9 bps is the all-trade avg gross; mean_revert-only avg
gross is 189.5 bps and winners-only 209.6 bps (both in
`trades_kalman.csv`) — even the most generous slice barely reaches the lower
bound, so the headline (all-trade) verdict is NOT MET. **Two of four
reviewer predictions are NOT MET** — the capture is smaller and the overlap
story is wrong, even though the headline net-Sharpe improvement is real.

### 1f — caveat: what the adaptive arm actually trades, and its failure mode

The adaptive arm trades **oscillation around a moving tracking level**, not
reversion to a fixed equilibrium. Its enumerated failure modes:

1. **Trending spread** — if the spread drifts persistently, μ_t chases it;
   the residual Z oscillates around ~0 by construction, so the engine keeps
   trading a level that is itself moving. P&L then depends on the *noise*
   around the trend, not on any true reversion. fold-6 INDUSINDBK (the −7σ
   case, §1c reference) is exactly this: μ-only tracking cannot reach ±1σ at
   any admissible speed (Unit-K §7.1).
2. **Shrinking oscillations** — if the oscillation amplitude collapses (the
   residual variance shrinks), the absorption guard is what *should* catch
   it on TRAIN, but on TEST a regime change is unhedged: bands set on a wider
   train residual become too wide, trades stop, or worse, fire late.
3. **No stop in the default config.** stop_mode = "none" for all 8 cells.
   There is no z-stop or hard stop; a trade that does not mean-revert exits
   only on time_stop or fold_close. On a trending spread this is the
   dangerous combination — the −1476 bps worst trade (1c(b)) is a time_stop.
   A stop is the obvious risk-management addition (deferred to §6's
   risk-management framework on crypto).

### 1g — note: exit-reason vocabulary gap in the persisted artifacts

The persisted Phase-3 trade CSVs use **non-standard** exit labels
(`fold_boundary`, `time`, `stop`, `session_close`) rather than the fixed
five-category vocabulary in `reporting_standard.md` §7
(`fold_close`, `time_stop`, `z_stop`, `eod_squareoff`). Phase 4 remaps them
centrally (`apt.phase4.canonical_exit_reason`); the mapping is
`fold_boundary→fold_close`, `time→time_stop`, `stop→z_stop`,
`session_close→eod_squareoff`. This is a pre-existing compliance gap, flagged
here; the fix is to bump the emitting engines to the fixed vocabulary.

### 1 — verdict

The net-Sharpe ≈ 2× improvement over frozen-OU on the matched universe
**reproduces** (1b). But the verification **weakens** two of the Unit-K
narrative claims: the edge is downside-tail compression rather than larger
captures (1c(b)), and it is **not** sourced from rescuing frozen's stuck
trades (1c(c), 0.23%). The win rate is mechanism-inflated (78.6% of trades
see the anchor move toward entry, 1c(d)). Two of four reviewer predictions
are NOT MET (1e). Net: **the improvement is real but its mechanism is
partly mis-stated, and it rests on n = 2 — exactly what §2 stress-tests.**

---

## Section 3 — β-escalation (β+μ joint Kalman)

Design + decision log (written **before** code):
[`docs/beta_escalation_design.md`](beta_escalation_design.md). Engine
`apt.stats.kalman_beta` (per-session causal joint (β, c) filter, 6 unit tests
incl. exact frozen-control equivalence). Driver
`scripts/phase4/s3_beta_escalation.py` → `reports/phase4/beta_escalation/`.
Run on the 2 traded survivors + the fold-6 INDUSINDBK −1.79σ diagnostic
(loaded from minute data; Pair/Fold from `fold_pairs.csv`, no re-selection).

### 3a — state/observation + train-only selection

State `θ_s = (β_s, c_s)`, `c_s = α + μ_s`; β observation is the **returns
regression** `β̂_s = cov_s(Δx,Δy)/var_s(Δx)` (collapse-prone by design),
level `ℓ̂_s = mean_s(y − β_s x)`; diagonal steady-state gains, `H_c = 20`
inherited from Unit K, `H_β` selected on train from `{∞,40,20,10}` with the
absorption guard **extended to β** (residual-HL guard + β-stability guard
β_s ∈ [0.25,4]×β_0). Source:
[`selection_beta_summary.csv`](../reports/phase4/beta_escalation/selection_beta_summary.csv).

| H_β | n_admissible / 4 | mean train criterion |
|----:|:----------------:|---------------------:|
| ∞ | **4** | **0.000227** (CHOSEN) |
| 40 | 0 | — (β-stability guard fails) |
| 20 | 0 | — |
| 10 | 0 | — |

**The selection chooses H_β = ∞ — i.e. it refuses to track β.** Every finite
H_β is inadmissible because β collapses out of the stability band on the
train window (§3b). The mean train criterion at H_β=∞ (0.000227) is identical
to the Unit-K μ-only value — as it must be (§3c equivalence).

### 3b — β-collapse diagnostic (the finding)

Source: [`beta_collapse.csv`](../reports/phase4/beta_escalation/beta_collapse.csv);
figure [`beta_paths_f5.png`](../plots/phase4/beta_escalation/beta_paths_f5.png).
At **every finite H_β**, on **every** pair-fold, β collapses toward 0 (5-min):

| pair-fold | β₀ | β_min (H_β=40 / 20 / 10) | min ratio @ H_β=10 |
|-----------|---:|--------------------------|-------------------:|
| fold 4 INDUSINDBK | 1.643 | 0.197 / 0.139 / 0.089 | **0.054** |
| fold 6 KOTAK | 0.872 | 0.134 / 0.039 / **−0.158** | −0.181 |
| fold 6 INDUSINDBK | 1.138 | 0.209 / 0.189 / 0.146 | 0.128 |

`beta_toward_zero = True` for all. This is **exactly the research report's
weak-identification warning**: on 5-min bars the legs' intraday co-movement
`cov(Δx,Δy)` is weak relative to the spread's own increments, so the
returns-β estimate `β̂_s → ~0` (KOTAK even flips negative), and the filter
drags β to the floor — the hedge evaporates. **We did not damp it; it is the
result.**

A nuance: the combined collapse flag (which also requires test-vs-train
residual-variance *instability*) does **not** fire, because β collapses on
**both** train and test symmetrically (var ratio ≈ 1). That is the stronger
statement — the collapse is **structural** (intraday β is unidentifiable on
these pairs), not a test-time regime break. (Punch list: refine the flag to
key on `beta_toward_zero` vs an absolute-stability reference, not a
train/test ratio.)

### 3c — β+μ vs μ-only vs frozen; the fold-6 success metric

Source: [`compare_3engine.csv`](../reports/phase4/beta_escalation/compare_3engine.csv),
[`drift_3way.csv`](../reports/phase4/beta_escalation/drift_3way.csv).

At the **selected** config (H_β=∞), β+μ **reproduces μ-only bit-for-bit**
(max |Δ net_total%| = 0.0 vs `metrics_kalman.csv`) — the frozen-control
equivalence holds at the P&L level, not just the residual. So on the matched
universe β-escalation delivers **no gross or net change** (e.g. 5min/1bps:
β+μ 139.94% net = μ-only 139.94%, both vs frozen-OU 48.27%).

**Success metric — fold-6 INDUSINDBK drift inside ±1σ at an ADMISSIBLE
config: NOT MET.**

| H_β | drift frozen | drift μ-only | drift β+μ | β min ratio | inside ±1σ | admissible |
|----:|-------------:|-------------:|----------:|------------:|:----------:|:----------:|
| ∞ | −6.99 | −1.79 | −1.79 | 1.00 | no | **no** |
| 40 | −6.99 | −1.79 | −0.73 | 0.18 | yes | no |
| 20 | −6.99 | −1.79 | −0.65 | 0.17 | yes | no |
| 10 | −6.99 | −1.79 | −0.55 | 0.13 | yes | no |

The β+μ drift *does* fall inside ±1σ at finite H_β — **but only by collapsing
β to ~0.13–0.18× its anchor**, which (i) fails the stability guard
(inadmissible) and (ii) inflates σ_eq, so the "improvement" is largely a
units artifact of an evaporated hedge, not a genuine reduction in mispricing.
**No admissible β+μ config neutralizes fold-6 that μ-only could not.**

### 3d — tests + verdict

Tests (`tests/stats/test_kalman_beta.py`, 6): synthetic β recovery, strict
causality / truncation invariance, **frozen-control equivalence**
(H_β=∞ reproduces `run_local_level_mu` exactly), β-collapse on vanishing
co-movement, β-stability guard, degenerate inputs — all pass.

**Verdict (β-escalation): FAILED GATE — and the failure is the finding.**
The NSE intraday hedge ratio cannot be tracked at 5/15-min frequency without
collapsing (weak identification, confirmed on all 3 pair-folds). Train
selection therefore freezes β (H_β=∞), at which point β+μ ≡ μ-only. The
canonical −7σ fold-6 case is **not** admissibly neutralized by a tracking-β
filter on this data. The escalation path forward is NOT "track β faster" — it
is either (a) estimate β at a *lower* frequency (daily co-movement is
stronger) or (b) a universe where the legs genuinely co-move intraday — which
is one more reason to push to crypto (§6).

## Section 4 — rolling cointegration-stability gate (NSE)

Engine `apt.stats.coint_stability` (rolling ADF on the spread level = the
frozen-(α,β) EG residual; 4 unit tests: stationary→no-gate, random-walk→gate,
regime-break→gate-partway, degenerate). Driver
`scripts/phase4/s4_coint_stability.py` → `reports/phase4/coint_stability/`.
Applied to **all 19 `fold_pairs` daily pair-folds** (broader than the 2
intraday survivors) using the daily spread from `daily_clean.parquet`.

### 4a — the gate

Labeled defaults (→ ASSUMPTIONS A6): rolling **window = 60** trading days,
**step = 5**, blacklist when ADF **p-value > 0.10** for **3 consecutive**
windows. The spread of a frozen-(α,β) pair *is* the EG residual, so a rolling
ADF on it tracks whether the cointegrating relation still holds.

### 4b — how many gate out, and the impact

Source: [`gate_summary.csv`](../reports/phase4/coint_stability/gate_summary.csv),
[`threshold_sensitivity.csv`](../reports/phase4/coint_stability/threshold_sensitivity.csv),
[`gate_impact.csv`](../reports/phase4/coint_stability/gate_impact.csv); figure
[`adf_pvalue_paths.png`](../plots/phase4/coint_stability/adf_pvalue_paths.png).

**At the default the gate fires on 19/19 pair-folds** — including both traded
survivors — with mean rolling ADF p-values of 0.30–0.62 (max ≈ 1.0). This is
**not** the anticipated near-null result, and the sensitivity sweep shows it
is robust, not a threshold accident:

| window | p>0.05 | p>0.10 | p>0.25 | p>0.50 |
|-------:|-------:|-------:|-------:|-------:|
| 60 | 19/19 | 19/19 | 19/19 | 19/19 |
| 120 | 19/19 | 19/19 | 18/19 | **15/19** |

Even at the very lenient p>0.50 / 120-day setting, 15/19 still gate. **The
daily spreads of the EG-selected NSE pairs are not robustly stationary on
rolling windows** — consistent with the broader Phase-3 finding that this
universe is marginal. Two forces are entangled and **cannot be separated on
n = 2**: (i) genuine marginal cointegration, and (ii) low ADF power on short
(60–120 obs) windows.

Gate impact on the kalman 5/3 survivors (`gate_impact.csv`): blacklisting
after the gate date cuts net total **119.2% → 51.8%** but *improves* net
Sharpe **1.96 → 2.93** and net maxDD **−21.1% → −6.8%** (170 of 504 sessions
survive). This is **threshold-driven, not a validated edge** — it merely
removes the back half of each fold.

**Verdict: [TODO]-pending-breadth (A6).** The machinery is correct and tested,
but the gate cannot be calibrated on 2 traded pair-folds without overfitting
the threshold, and short-window ADF is too low-powered to discriminate true
breakdowns from noise on this universe. The gate needs a **broad** universe
(many pair-folds → a base rate to set a relative threshold, and longer series
for ADF power) — i.e. crypto (§6) — before its binary blacklist can be
trusted. The NSE binding result is therefore marked `[TODO]`.

## Section 5 — Johansen pair selection (NSE)

> **CONFOUND — READ FIRST.** Johansen selects a **different universe** than
> EG+BH-FDR. Its 41 pairs are **NOT comparable** to the EG-FDR 13 pairs, and
> neither is comparable to the Phase-3 intraday-traded **2** survivors (which
> additionally passed HL-band, liquidity and intraday gates). **No Johansen
> pair is merged into any matched table.** This section reports the *selection*
> universe difference only — there is no backtest here.

Engine `apt.stats.johansen` (order-independent trace test; 4 unit tests
incl. order-independence). Driver `scripts/phase4/s5_johansen.py` runs Johansen
**and** a fresh EG+BH-FDR pass on the **same 234** correlation-screened
candidate pairs over a common 1008-day window, so the comparison is
apples-to-apples. Source:
[`comparison_summary.csv`](../reports/phase4/johansen/comparison_summary.csv),
[`johansen_selection.csv`](../reports/phase4/johansen/johansen_selection.csv);
figure [`eg_vs_johansen.png`](../plots/phase4/johansen/eg_vs_johansen.png).

### 5a/5b — selection comparison

| quantity | value |
|----------|------:|
| candidates tested | 234 |
| EG raw p<0.05 | 50 |
| **EG + BH-FDR** | **13** |
| **Johansen 95%** | **41** |
| both (Johansen ∩ EG-FDR) | 13 |
| **Johansen-only (NEW)** | **28** |
| EG-FDR-only (lost) | 0 |
| Jaccard(EG-FDR, Johansen) | 0.32 |
| fresh-EG vs persisted FDR agreement | **0.991** |

Two clean readings:

1. **The setup is validated.** My fresh EG+BH-FDR reproduces the *persisted*
   `cointegrated_pairs.parquet` FDR selection at **99.1%** agreement — so the
   apples-to-apples comparison is sound.
2. **Johansen ⊋ EG-FDR here, but the gap is mostly the FDR correction, not
   the test.** Johansen-95% (41) selects a strict superset of EG-FDR (13) —
   28 new pairs, 0 lost. But Johansen's 41 sits right next to **EG raw
   p<0.05 (50)** and far above **EG+BH-FDR (13)**: the expansion is
   overwhelmingly because the Johansen path carries **no multiple-testing
   correction**, not because order-independence finds structurally different
   pairs. Most of the 28 "new" pairs have EG single-test p ≈ 0.005–0.014
   (individually significant) that BH-FDR rejected. Example new pairs:
   HDFC/HDFCBANK, BAJFINANCE/HDFCBANK, CIPLA/LUPIN, HINDALCO/NMDC,
   LUPIN/SUNPHARMA (`johansen_selection.csv`).

### 5 — verdict

Johansen is a **correct, order-independent** selection path (tested), and on
this universe it would roughly **triple** the selected pair count vs EG+BH-FDR
(41 vs 13). **But the honest attribution is multiple-testing, not the
estimator:** apply an FDR-equivalent control to Johansen and the universes
would largely re-converge. The 28 new pairs are unvalidated candidates — not
FDR-corrected, not HL/liquidity/intraday-gated — so they are reported as a
**separate, non-comparable** universe and explicitly **not** carried into any
performance table. A genuine Johansen-vs-EG *performance* comparison would
require re-running the entire downstream pipeline on the Johansen universe
with matched corrections — out of scope this run (`[TODO scope]`).

## Section 6 — Crypto port (SCAFFOLD + FIRST RESULTS — not claimed validated)

Modules `apt.crypto` (Binance-kline loader + minimal cleaning, 2 unit tests)
and `apt.crypto.costs` (taker+spread, (1+β) billing). Driver
`scripts/phase4/s6_crypto.py` → `reports/phase4/crypto/`. Reuses the
**asset-agnostic** walk-forward engine (`apt.backtest.run_walkforward`) via
crypto callbacks — its leakage-free per-fold selection is preserved.

### 6a — data inventory (verified readable)

Source: [`inventory.csv`](../reports/phase4/crypto/inventory.csv),
[`symbol_coverage.csv`](../reports/phase4/crypto/symbol_coverage.csv).

| field | value |
|-------|-------|
| location | `/home/om/data_combined` (read access **OK**) |
| symbols | **30** USDT pairs (BTC, ETH, BNB, SOL, XRP, … SHIB) |
| bar frequency | **1-minute** Binance klines |
| columns | 12-col headerless: open_time, OHLCV, close_time, quote_volume, n_trades, taker_buy_base/quote, ignore |
| timestamp / tz | epoch in **UTC**; **mixed ms/µs** — Binance switched ms→µs in 2025 (the loader normalizes by magnitude) |
| date range | 2017-08-17 → 2026-04-30 (per-symbol coverage varies) |
| **funding-rate series** | **ABSENT** → cost model omits funding ([TODO data] A8) |

### 6b — pipeline + cointegration

Minimal crypto cleaning only (dedup open_time, drop non-positive OHLC, drop
zero-volume bars); the **NSE seven-rule cascade does not transfer** (no
corporate actions, 24/7) — the rest is `[TODO]` in `apt.crypto.loader`.
Liquidity gate (labeled default A_LIQ: median daily quote-volume ≥ $10M)
keeps **22/30** symbols. Common window 2018-05 → 2026-04 (2919 days).

Cointegration (`cointegration.csv`): of **231** pairs, **16 EG+BH-FDR** and
**85 Johansen-95%** are cointegrated. (The §5 confound recurs: Johansen
without FDR is far more permissive.) **Breadth achieved** — unlike NSE's 2.

### 6c — cost model

Taker fee = **5 bps/side** labeled default (A7) → 10 bps/leg round-trip;
plus the spread sweep {1,3,5,8} bps; (1+β) billing. **Funding deferred**
(A8) — no series exists. Both Regime A and B were intended (6c); at daily
frequency only **Regime B** (multi-day carry) is meaningful — **Regime A
(intraday squareoff) needs intraday bars → [TODO]**.

### 6d — risk management

Labeled-default `stop_z = 3.5` (A9), **flagged not tuned** — the long-deferred
crypto risk framework is represented by this single hard z-stop only; a full
framework is `[TODO scope]`.

### 6e — walk-forward + DSR/PBO (the first results)

Source: [`metrics.csv`](../reports/phase4/crypto/metrics.csv),
[`dsr_pbo.csv`](../reports/phase4/crypto/dsr_pbo.csv); figure
[`crypto_cost_ladder.png`](../plots/phase4/crypto/crypto_cost_ladder.png).
Engine = **rolling-z** (the OU/Bertram and μ-only/β+μ Kalman crypto engines
are `[TODO scope]`). 12 folds, 112 pair-fold selections, **529 trades**.

| cost | gross_total% | net_total% | gross_Sharpe | net_Sharpe | net_ann% | net_maxDD% |
|-----:|-------------:|-----------:|-------------:|-----------:|---------:|-----------:|
| 1 | −63.9 | −66.3 | −0.58 | −0.62 | −17.4 | −73.3 |
| 3 | −63.9 | −66.8 | −0.58 | −0.63 | −17.5 | −73.5 |
| 5 | −63.9 | −67.2 | −0.58 | −0.64 | −17.7 | −73.8 |
| 8 | −63.9 | −67.8 | −0.58 | −0.65 | −18.0 | −74.1 |

**Decisively negative — and gross is already negative**, so it is a **signal**
problem, not a cost problem: naive rolling-z mean-reversion has no edge on
trending crypto majors (−73% maxDD). DSR/PBO over **81 pair columns × 1440
sessions**: DSR = **0.367** (p 0.633), PBO = **0.439**.

**Does breadth make the DSR/PBO meaningful? YES** — this is the key §6e answer.
Unlike the NSE n=2 (a 2-fold concatenation that strains every assumption), the
crypto matrix has genuine cross-sectional width (81 independent pair series),
so the CSCV blocks are well-populated and the DSR iid approximation is far less
abused. And what the now-trustworthy numbers say is unambiguous: **no edge**
(DSR below 0.5, PBO near 0.5, negative Sharpe).

### 6f / 6 — verdict (SCAFFOLD)

The crypto port is **scaffold + first results, explicitly not validated.** The
honest first finding: **the naive rolling-z engine does not work on crypto**
(negative gross). But this is **not** a fair test of the adaptive-equilibrium
thesis — the OU/Bertram and Kalman engines (which on NSE doubled rolling-z) are
`[TODO scope]` here, as are Regime A (intraday), the bar-frequency sweep,
funding cost (A8), and the full risk framework (A9). What §6 **does** establish:
(i) the data is real, inventoried, and loads; (ii) the pipeline ports
end-to-end with breadth (231 pairs, 16 EG-FDR); (iii) **crypto breadth makes
DSR/PBO trustworthy** in a way NSE n=2 never could — so it is the right venue
to retest the adaptive engines and the §4 stability gate. That retest is the
next unit.

---

## §8. Figure inventory

*(assembled at §7; per-subunit figure CSVs ship beside each PNG under
`plots/phase4/<subunit>/`)*

- `plots/phase4/verification/h_exit_reason_stacked.png` — exit-reason mix,
  kalman vs frozen (best cell).
- `plots/phase4/verification/per_trade_pnl_dist.png` — per-trade net P&L
  histograms.
- `plots/phase4/verification/mu_tracking_indusindbk_fold4.png` — spread +
  μ_frozen + μ_kalman with trade markers.
- `plots/phase4/verification/d_cost_ladder_3engine.png` — net Sharpe vs cost,
  3 engines × {f5, f15}.

## §9. Fix-later punch list

*(assembled at §7; ranked by headline impact — see ASSUMPTIONS register)*

1. **(A1/A2) n=2 → DSR/PBO indicative only.** Highest impact: the headline
   Sharpe ≈ 2.0 is an n=2 statistic. Fix: breadth (§5, §6).
2. **(A3) persist KOTAK μ-overlay** — completes 1c(d) μ-tracking. Low effort,
   no model change.
3. **(1g) exit-reason vocabulary** — bump engines to the fixed five
   categories.

## §10. Per-section verdict

- **§1 Unit K:** improvement reproduces; mechanism partly mis-stated; 2/4
  reviewer predictions NOT MET; rests on n=2.
- **§§2–6:** [TODO compute] — see each section.
