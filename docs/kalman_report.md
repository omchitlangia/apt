# Unit K — adaptive-equilibrium run report

**Generated:** 2026-06-12
**Branch:** `feature/kalman-equilibrium`
**Driver:** `scripts/17_kalman_equilibrium.py --mode full` +
`scripts/18_kalman_figures.py`
**Design + rulings:** [`docs/kalman_design.md`](kalman_design.md)
(Decision Log Q1-Q8). Reporting per
[`docs/reporting_standard.md`](reporting_standard.md).

------------------------------------------------------------------------

## 1. Objective

Test whether a **per-session causal local-level filter** on the
equilibrium level `μ_t` (β frozen) neutralizes the frozen-μ drift that
crippled the OU engine (median 2.1 σ_eq, extreme −7.05 σ_eq) **without
destroying the OU engine's cost-amortization profile** on the matched
universe. μ-only state; the filter, its train-only selection, and the
absorption guard are the whole unit.

## 2. Quadrant framing (centre × thresholds)

The attribution is a 2×2 of **equilibrium centre** × **threshold engine**:

|                       | rolling_z thresholds         | OU-Bertram thresholds              |
|-----------------------|------------------------------|------------------------------------|
| **frozen centre**     | rolling_z baseline (on main) | frozen-OU (on main)                |
| **adaptive μ_t centre** | *incoherent — SKIPPED*     | **kalman_mu (this unit, 8 cells)** |

The **`adaptive × rolling_z`** quadrant is **skipped** (Decision Log
§6): rolling_z already centres on its OWN trailing rolling mean
(`intraday_rolling_zscore`), so there is no separate "centre" knob to
replace — the rolling mean IS the adaptive centre. A μ_t overlay on top
of a rolling-mean centre is double-centring, incoherent. The would-be
16-cell Kalman half of the 2×2 is therefore **8 real cells**
(`kalman_mu × OU-Bertram`) + 8 skipped, compared against the two
already-computed frozen quadrants from main.

## 3. Pre-registered expectations (verbatim, from the Unit-K instruction — recorded BEFORE the grid was run)

> (a) median |test-slice Z mean| < 0.5 under the selected half-life (gate);
> (b) the adaptive+Bertram arm recovers gross toward rolling_z levels while
> n_trades stays within ~1.5x of frozen-OU; (c) INDUSINDBK fold 6 drift
> shrinks from −7.05σ to inside ±1σ if the instability is level-type — if it
> does not, record it as evidence for the β-escalation trigger.

Scored in §11.

## 4. Config grid

- `kalman_mu × OU-Bertram`, `freq ∈ {5, 15}`, `cost ∈ {1, 3, 5, 8}` bps,
  Regime B, stop=none → **8 cells**.
- Static-HL gate (Decision Log Q6) ⇒ **identical 2 pair-folds** as the
  OU run: `(fold 4, INDUSINDBK/HDFCBANK)` + `(fold 6, KOTAKBANK/HDFCBANK)`.
- Selected global re-anchor half-life **H = 20 sessions** (§5).
- `(1+β)` billing (main's convention); a* re-solved per cost on the
  TRAIN-residual OU fit.

## 5. TRAIN-only half-life selection

Grid `{∞, 20, 10, 5}` sessions. Criterion = analytic Bertram
net-return-per-unit-time on the TRAIN-residual OU fit (β-aware 3 bps
reference), subject to the **absorption guard** (residual HL within
`[0.5×, 1.5×]` of the frozen-μ HL). Full per-`(H, freq, pair-fold)`
table: [`selection_table.csv`](../reports/phase3_kalman/selection_table.csv).

| H (sessions) | n_admissible / n_total | mean train ret/unit-time | verdict |
|-------------:|:----------------------:|-------------------------:|---------|
| ∞ (frozen)   | 4 / 4                  | 0.000169                 | admissible (baseline) |
| **20**       | **4 / 4**              | **0.000227**             | **CHOSEN** (best admissible) |
| 10           | 0 / 4                  | — (guard failed)         | inadmissible |
| 5            | 0 / 4                  | — (guard failed)         | inadmissible |

**The guard binds.** At H = 10 and H = 5 every survivor's residual HL
collapses below `0.5×` the frozen HL (ratios 0.31-0.45) — the filter
starts absorbing the OU oscillation itself, "inventing" fast reversion.
The guard rejects them. H = 20 sits just inside the floor (ratios
0.517-0.626) and beats frozen on the train criterion (0.000227 vs
0.000169), so it is selected. **This is selection working as designed:
the fastest admissible re-anchoring, no faster.**

Figure: [`selection.png`](../reports/phase3_kalman/figures/selection.png).

## 6. Headline results — gross AND net (kalman_mu, Regime B, H=20)

Per [`metrics_kalman.csv`](../reports/phase3_kalman/metrics_kalman.csv):

| freq | cost | n_trades | gross_total% | net_total% | gross_Sharpe | net_Sharpe | net_maxDD% |
|-----:|-----:|---------:|-------------:|-----------:|-------------:|-----------:|-----------:|
|  5   |  1   |    69    |       161.35 |     139.94 |        2.320 |      2.145 |     −21.02 |
|  5   |  3   |    60    |       142.31 |     119.16 |        2.169 |      1.959 |     −21.08 |
|  5   |  5   |    53    |       127.01 |     103.13 |        2.016 |      1.781 |     −20.68 |
|  5   |  8   |    51    |       147.68 |     115.20 |        2.227 |      1.937 |     −20.37 |
| 15   |  1   |    66    |       161.79 |     141.29 |        2.309 |      2.145 |     −21.48 |
| 15   |  3   |    57    |       149.06 |     126.21 |        2.208 |      2.013 |     −21.65 |
| 15   |  5   |    52    |       138.68 |     113.96 |        2.125 |      1.901 |     −20.71 |
| 15   |  8   |    41    |       101.94 |      80.13 |        1.772 |      1.524 |     −20.56 |

### 6.1 vs frozen-OU vs rolling_z (matched universe, net side)

| freq | cost | kalman net% / Sh | frozen-OU net% / Sh | rolling_z net% / Sh |
|-----:|-----:|-----------------:|--------------------:|--------------------:|
|  5   |  1   | 139.94 / 2.145   | 48.27 / 0.981       | 37.67 / 0.945       |
|  5   |  3   | 119.16 / 1.959   | 46.20 / 0.949       | 24.34 / 0.643       |
|  5   |  5   | 103.13 / 1.781   | 43.17 / 0.905       | 12.30 / 0.341       |
|  5   |  8   | 115.20 / 1.937   | 44.28 / 0.935       | −3.60 / −0.107      |
| 15   |  1   | 141.29 / 2.145   | 49.91 / 1.003       | 18.68 / 0.613       |
| 15   |  3   | 126.21 / 2.013   | 39.22 / 0.830       | 14.01 / 0.473       |
| 15   |  5   | 113.96 / 1.901   | 30.97 / 0.684       | 9.51 / 0.331        |
| 15   |  8   | 80.13 / 1.524    | 33.54 / 0.745       | 3.11 / 0.113        |

**The adaptive centre roughly DOUBLES net Sharpe over frozen-OU and
2-3× the net total**, and the cost-amortization profile is preserved
(the ladder stays flat — kalman net Sharpe degrades only ~0.2 over the
1→8 bps sweep at 5-min, vs rolling_z's collapse to negative). Figures:
[`cost_ladder_3engine_f5.png`](../reports/phase3_kalman/figures/cost_ladder_3engine_f5.png),
[`...f15.png`](../reports/phase3_kalman/figures/cost_ladder_3engine_f15.png),
NAV [`b_kalman_best_nav.png`](../reports/phase3_kalman/figures/b_kalman_best_nav.png).

**This result is real but n = 2 and exploratory — see §10 caveats.** The
magnitude (net Sharpe ~2.0, gross > rolling_z) is large enough to
demand scepticism; the mechanism and causality checks are in §7-§8.

## 7. Drift before/after (the gate diagnostic)

Per [`drift_before_after.csv`](../reports/phase3_kalman/figures/drift_before_after.csv).
Figure: [`drift_before_after.png`](../reports/phase3_kalman/figures/drift_before_after.png).

| pair-fold | freq | frozen Z-mean (σ_eq) | adaptive Z-mean (σ_eq) |
|-----------|-----:|---------------------:|-----------------------:|
| fold 4 INDUSINDBK/HDFCBANK | 5  | −3.289 | −0.918 |
| fold 6 KOTAKBANK/HDFCBANK  | 5  | +1.159 | +0.324 |
| fold 4 INDUSINDBK/HDFCBANK | 15 | −3.290 | −0.921 |
| fold 6 KOTAKBANK/HDFCBANK  | 15 | +1.160 | +0.325 |

**median |frozen| = 2.225 σ_eq → median |adaptive| = 0.621 σ_eq.** A
**72% reduction**, but the median sits just ABOVE the 0.5 gate because
INDUSINDBK fold 4 (β=1.64) still carries −0.92 σ_eq after the admissible
H=20. KOTAK is comfortably inside ±0.5.

### 7.1 fold-6 INDUSINDBK — the canonical −7σ hard case (β-escalation trigger)

`(fold 6, INDUSINDBK/HDFCBANK)` is the −7.05σ case from the OU report.
It is **NOT a traded survivor** — its frozen HL is ~5070 min, outside
the B band [120, 1875] — so it is rendered diagnostically only. The
**money figure**
[`mu_overlay_fold6.png`](../reports/phase3_kalman/figures/mu_overlay_fold6.png)
shows the frozen μ_OU ~7σ above a spread that collapses from −0.06 to
−0.5 over the test window, with the adaptive μ_t (H=20) tracking the
descent but lagging:

| H | drift_kalman (σ_eq) | guard | admissible? |
|--:|--------------------:|------:|:-----------:|
| 20 (selected) | **−1.79** | ratio 0.34 | **NO** (already < 0.5 on this pair-fold) |
| 10 | −1.12 | 0.25 | no |
| 5  | −0.71 | 0.17 | no |

**The −7σ drift does NOT reach inside ±1σ at any admissible re-anchoring
speed** (it only reaches −0.71σ at H=5, which the guard forbids). This
is **evidence for the β-escalation trigger** (pre-reg (c) fallback): a
μ-only filter cannot neutralize this pair-fold's instability without
collapsing the residual into noise — the instability is **not purely
level-type**, consistent with a co-drifting β. Named here as the
**canonical hard case for a future β-tracking unit.**

## 8. Mechanism + win-rate scrutiny

The money comes from removing the slow drift so the residual oscillates
cleanly around 0, which Bertram then harvests:

- Best cell (5-min, 3 bps): **57/60 exits are `mean_revert`**, and
  mean_revert exits supply **125% of net P&L** (the 3 non-mean-revert
  exits — 2 time_stop + 1 fold_boundary — are net-negative).
- Win rate (net > 0): **0.90** (vs frozen-OU 0.82).
- Per pair-fold net log P&L roughly doubles: INDUSINDBK fold 4
  0.134 → 0.247 (1.8×), KOTAK fold 6 0.248 → 0.540 (2.2×).

**Win-rate-inflation caveat:** the per-session re-anchoring moves the
exit target (Z=0 ⇔ X=μ_t) *toward* the price each session, which makes
the mean-revert exit easier and inflates the win rate vs a static
target. This is a **legitimate, causal property** of the filter (μ_{s+1}
is set at session s close, before session s+1 trades — unit-tested), not
a look-ahead. But it means the win rate is NOT comparable to a
fixed-target strategy's, and the result should be read with that
mechanism in mind.

**Causality is unit-tested:** μ applied during session s depends only on
sessions < s (`test_mu_path_is_strictly_causal`,
`test_truncation_invariance`); hyperparameters and init are train-only
(`test_init_and_hyperparameter_provenance`); H=∞ reproduces the OU cells
exactly (`test_frozen_control_equivalence_ou_translation_invariance`).

## 9. Exclusion accounting — unchanged

The static-HL gate (Decision Log Q6) is β-independent and identical to
the OU run: **19 pair-folds → AR(1) valid → HL ∈ [120,1875] min B band →
2 traded survivors** at each of freq 5 and 15. The Kalman layer changes
the *centre* and *thresholds*, never the selection. Confirmed: the 2
survivors here are the same `(fold 4, INDUSINDBK)` + `(fold 6, KOTAK)`
as the OU/matched-universe runs.

## 10. Caveats

- **n = 2, exploratory, time-disjoint.** The matched universe is 2
  pair-folds in different folds/years (fold 4 ≈ 2017, fold 6 ≈ 2019); the
  portfolio is essentially the concatenation of two single-pair curves,
  so the Sharpe ≈ 2.0 is an n=2 statistic, not a fleet result. Do not
  read it as a deployable edge.
- **Gross exceeds rolling_z** (161% vs 82%). Plausible — a drift-cleaned
  residual is more tradeable and Bertram's cost-aware band is tighter —
  but the magnitude is large; it should be re-tested on a broader
  universe before any weight is put on it.
- **Win-rate inflation** from the moving exit target (§8) — the 0.90 win
  rate is mechanism-dependent.
- **The gate (median |Z| < 0.5) is NOT met** (0.62), held above the line
  by INDUSINDBK fold 4; the absorption guard prevents the faster H that
  would close it.
- **β-escalation flagged, not implemented** — fold-6 INDUSINDBK's −7σ is
  the named hard case for a future β-tracking unit; out of scope here.
- **TOD ignored** (Decision Log Q1) — the OU path ignores it too, so the
  comparison is clean, but a TOD-aware `V_e` is the natural extension.

## 11. Pre-registration scoring

| pre-reg | outcome |
|---------|---------|
| (a) median \|test Z mean\| < 0.5 under selected H | **PARTIAL / not met.** 2.225 → 0.621 σ_eq (72% cut) but 0.62 > 0.5; INDUSINDBK fold 4 (−0.92) holds the median above the line. |
| (b) gross recovers toward rolling_z; n_trades within ~1.5× frozen | **EXCEEDED on gross, OVERSHOT on trades.** Gross recovered ABOVE rolling_z; n_trades ran 1.76-1.84× frozen (just over the ~1.5× expectation). |
| (c) INDUSINDBK fold 6 −7.05σ → inside ±1σ if level-type | **DID NOT — β-escalation recorded.** Only reaches −1.79σ at the admissible H=20 (−0.71σ at the guard-forbidden H=5). Logged as evidence the instability is not purely level-type. |

**Verdict.** The adaptive centre is a large, causal, cost-amortization-
preserving improvement on the matched universe (net Sharpe ~2× frozen-OU)
— but the success gate is **narrowly missed** (median 0.62 vs 0.5) and
the canonical −7σ case is **not** neutralized at any admissible speed,
escalating it to a β-tracking follow-up. The result is **promising but
n=2 exploratory**; the next step is a broader universe and the
β-escalation unit, not deployment.

## 12. Test results

```
$ .venv/bin/pytest tests/
341 passed, 2 warnings in 123.86s
```

New this unit: `tests/stats/test_kalman.py` (10 tests) — gain
arithmetic, random-walk μ recovery, EWMA-ramp-lag pin, residual OU
recovery, frozen constant-centre, strict causality + truncation
invariance, init/hyperparameter provenance (leakage analogue), absorption
guard on multi-session pure-OU, frozen-control equivalence.

## 13. Figure inventory

All in `reports/phase3_kalman/figures/`:
- `mu_overlay_fold4.png`, `mu_overlay_fold6.png` — the money figures
  (spread + frozen μ + adaptive μ_t), folds 4 (traded) and 6 (−7σ
  diagnostic).
- `drift_before_after.png` — drift per pair-fold, frozen vs adaptive.
- `selection.png` — train-only half-life selection (admissible vs
  guard-failed).
- `cost_ladder_3engine_f5.png`, `...f15.png` — kalman vs frozen-OU vs
  rolling_z net Sharpe + net total per cost.
- `b_kalman_best_nav.png` — portfolio NAV gross vs net (5-min, 3 bps).
- `h_kalman_exit_reasons.png` — exit-reason composition per cell.

MLflow: not wired (Decision Log Q7); the figures + companion CSVs are
the artefact catalogue (`reporting_standard.md` §8).
