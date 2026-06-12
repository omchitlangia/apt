# Unit K — Kalman / adaptive equilibrium

**Branch:** `feature/kalman-equilibrium` (off post-merge `main` at 65b15f6)
**Status:** Discovery + design note (§0-§8) → **IMPLEMENTATION** under
the locked rulings in the Decision Log below.
**Scope OUT (explicit):** pair selection, Johansen, regime/HMM,
meta-labeling, cost accounting (now fixed — `(1+β)` billing on main),
RL. This unit is ONLY about the adaptive equilibrium (μ; β stays frozen)
on an already-selected pair-fold.

------------------------------------------------------------------------

## Decision Log (2026-06-12 — rulings Q1-Q8 are FINAL)

The §7 questions are resolved as follows. These supersede the "default
argued" recommendations in §7 wherever they differ.

| Q | ruling |
|---|--------|
| **Q1 state vector** | **μ only.** β frozen as-is (the daily EG fit on `Pair`). No β tracking in this unit. TOD: **ignore** (option (C), mirror the OU path). |
| **Q2 update frequency** | **Per-session causal local-level update of μ_t.** The filter updates μ at **session close** and the updated μ is **applied from the next session open** (no intra-session look-ahead). Re-anchor half-life expressed **in SESSIONS**, one **global** value across all pair-folds. |
| **Q3 hyperparameters** | Re-anchor half-life selected on **TRAIN only** by grid `{∞, 20, 10, 5}` sessions. Selection criterion = **net return per unit time of the Bertram rule on train residuals**, SUBJECT to an **absorption guard**: residual half-life on train must lie in `[0.5×, 1.5×]` of the frozen-μ OU half-life; violating configs are **inadmissible**. One global value → test. Selecting on test is forbidden. (κ(Δt)-seeded `V_e` deferred — not used; the discount/half-life form carries adaptivity.) |
| **Q4 signal composition** | **Band on the adaptive-center Z**, `Z_k = (X − μ_t)/σ_eq_resid`, where `(κ, σ_eq_resid)` are OU params fit on **TRAIN residuals** and `a*` is **re-solved per cost level** under main's β-aware `(1+β)` billing. Entry/exit/time-stop/EOD semantics identical to the OU unit. This is option (b) — preserves the per-cost-level refit principle. |
| **Q5 attribution** | 16 new cells: `kalman_mu × OU-Bertram × freq{5,15} × cost{1,3,5,8} × Regime B`. Compared against frozen-OU and rolling_z matched-universe cells from main. The rolling_z-with-kalman-center arm is included ONLY if it is a trivial config toggle (see §6 ruling below). |
| **Q6 HL-band gate** | **Static train OU fit** (option (i)). Selection set identical to the OU run ⇒ identical pair-folds. Selection remains train-only. |
| **Q7 MLflow** | **Stay on `MANIFEST.csv`** for this unit; MLflow deferred. |
| **Q8 contradictions** | None found (confirmed §7.8). |

**Config surface (locked):** a new `signal.center ∈ {frozen, kalman_mu}`,
**default `frozen`** (the OU unit's behaviour is the default; `kalman_mu`
is opt-in). `signal.center == frozen` MUST reproduce the OU-unit cells
exactly (frozen-control equivalence, tested).

**rolling_z-with-kalman-center arm (Q6/§6 ruling):** rolling_z already
centers on its OWN trailing rolling mean (`intraday_rolling_zscore`), so
a second adaptive center is **incoherent** — there is no single "center"
knob to toggle; the rolling mean IS the adaptive center. Therefore the
8-cell rolling_z-with-kalman-center arm is **SKIPPED**. One paragraph in
the report states this. The 16 new cells are the `kalman_mu × OU-Bertram`
quadrant only.

------------------------------------------------------------------------

## 0. Motivation (restated from the OU run report)

The OU engine freezes the equilibrium level `μ_OU` (and `σ_eq`) on the
TRAIN window and applies them to the whole TEST window:

```
scripts/15_phase3_ou.py:174
    z_ou_test = (spread_test - fit.mu) / fit.sigma_eq
```

This produces severe **frozen-μ drift** on the test slice:

- **median |test-slice Z-OU mean| = 2.1 σ_eq** across valid OU fits.
- **14 / 18 pair-folds flagged** (|drift| > 0.5 σ_eq).
- **extreme −7.05 σ_eq**: `(fold 6, INDUSINDBK/HDFCBANK)` — the
  **canonical hard case** for this unit. The same pair at
  `(fold 4, INDUSINDBK/HDFCBANK)` (drift −3.29 σ_eq) is half of the
  OU headline best cell.

A 2.1 σ_eq median offset means the train-frozen Bertram thresholds
operate on a spread that has shifted equilibrium regime during test —
the source of the directional trade-flow bias and the hard-stop
drift-loop pathology (OU report §6.3-6.4).

**Success gate.** A Kalman/adaptive-equilibrium layer succeeds iff:

> median |test-slice standardized-residual mean| **< 0.5** across
> pair-folds, **WITHOUT** destroying the OU engine's cost-amortization
> profile on the matched universe (the flat OU cost-ladder slope from
> the cost-beta report §13.2 — OU ≥ rolling_z at all four costs).

The "standardized residual" is the Kalman analogue of Z-OU: the
innovation `e_t` deflated by its predicted standard deviation
`√Q_t` (defined in §4). The drift diagnostic becomes the test-slice
mean of `e_t/√Q_t` per pair-fold.

**Kill condition.** If NO admissible configuration (within the
hyperparameter envelope chosen on train, §3) brings the median drift
below 0.5, the equilibrium instability is **structural** — the pairs
are not co-integrated out-of-sample on the minute panel — and
**rolling-cointegration monitoring becomes the primary deliverable**
instead of an adaptive filter (i.e. detect and stand down, do not try
to track).

------------------------------------------------------------------------

## 1. Where β / intercept are fit, frozen, and consumed

### 1.1 Daily fit (the source of the frozen β, α)

- **OLS fit** — `src/apt/signals/cointegration.py:135-138`:
  ```
  X = sm.add_constant(x)
  fit = sm.OLS(y, X).fit()
  alpha = float(fit.params[0])     # intercept
  beta  = float(fit.params[1])     # hedge ratio
  ```
  on **log prices** (`engle_granger`, docstring line 124-126).
- **Direction pick** — `src/apt/signals/cointegration.py:154-164`
  (`engle_granger_best_direction`): runs both orientations, keeps the
  smaller-ADF-p-value one; the returned `(alpha, beta)` is the frozen
  pair fit.
- **Per-pair emission** — `cointegrate_pairs`
  (`src/apt/signals/cointegration.py:376`+; the EG call at line 527)
  writes one `(alpha, beta, half_life, …)` row per surviving pair.

### 1.2 Frozen into the `Pair` dataclass

- **`Pair`** — `src/apt/backtest/walkforward.py:67-80`,
  `@dataclass(frozen=True)` with fields `y_sym, x_sym, alpha, beta,
  half_life, sector, is_structural`. **Immutable by construction** —
  α, β cannot mutate after selection. This is the object a Kalman layer
  would have to either (a) leave alone and override downstream, or
  (b) replace with a time-varying state.
- **Construction (intraday)** —
  `scripts/13_phase3_intraday.py:166-174` from the cointegration table.
- **Construction (daily WF)** — `scripts/10_backtest.py:120-126`.

### 1.3 Every place the frozen values are consumed

| consumer | path:line | expression |
|----------|-----------|------------|
| rolling_z intraday spread | `scripts/13_phase3_intraday.py:267` | `spread_full = np.log(py) - pair.beta*np.log(px) - pair.alpha` |
| OU intraday spread | `scripts/15_phase3_ou.py:161` | `spread_full = np.log(py) - pair.beta*np.log(px) - pair.alpha` |
| OU train fit | `scripts/15_phase3_ou.py:168` | `fit = fit_ou_params(spread_full[train_mask], …)` → frozen `μ`, `σ_eq` |
| **OU frozen-μ application** | **`scripts/15_phase3_ou.py:174`** | **`z_ou_test = (spread_test - fit.mu) / fit.sigma_eq`** ← drift source |
| daily WF spread | `src/apt/backtest/walkforward.py:471` | `spread_full = compute_spread(p_y, p_x, beta=pair.beta, intercept=pair.alpha)` |
| `compute_spread` | `src/apt/signals/spread.py:67` | `return np.log(a) - beta*np.log(b) - intercept` |
| coarse rolling_z spread | `scripts/15b_phase3_rolling_baseline.py:133` | `spread_full = np.log(py) - pair.beta*np.log(px) - pair.alpha` |

**The single line a Kalman layer must displace is
`scripts/15_phase3_ou.py:174`** — replace the static `(spread − μ)/σ_eq`
with a causal filter that tracks `μ_t` (and optionally `β_t`) through
test. Everything upstream (the β·logX − α spread) and downstream
(Bertram, `run_pair_fold`) is reachable without touching the immutable
`Pair`. `[TODO design]` decide whether `β_t` tracking replaces the
spread construction at lines 161 / 267 / 471 too (state-vector
question, §7.1).

------------------------------------------------------------------------

## 2. The TOD vol profile and where a Kalman layer composes with it

### 2.1 Exact transform

- **`fit_tod_vol_profile`** — `src/apt/intraday/zscore.py:176-237`.
  Train-only. For each minute-of-session `m ∈ [0, 375)`:
  ```
  innovation = spread_train - chronological_rolling_mean(spread_train, window)
  sigma_tod[m] = std( innovation[ bar_in_session == m ] )   # ddof=1
  ```
  then a NaN-aware boxcar smoother of half-width `smooth_radius`
  (default 2 ⇒ 5-minute boxcar). Returns a length-375 σ profile.
- **`tod_adjusted_zscore`** — `src/apt/intraday/zscore.py:240-290`:
  ```
  z_tod[t] = (spread[t] - sessionized_rollmean[t]) / sigma_tod[ m(t) ]
  ```
  with the sessionized rolling std as the divisor fallback where the
  train profile is NaN.

### 2.2 Where it sits in the chain — and the crucial fact

- In the **rolling_z path** (`scripts/13_phase3_intraday.py`): the TOD
  profile is fit on train (line 292) and `z_tod` is computed on test
  (line 321), **but the production signal uses `z_flat`** (the flat
  sessionized z) at lines 319 / 341. `z_tod` is **diagnostic only** —
  the v2 report and `docs/ou_thresholds_design.md` §8.2 record "TOD NOT
  in production path".
- In the **OU path** (`scripts/15_phase3_ou.py`): **TOD is not used at
  all.** Z-OU is `(spread − μ_OU)/σ_eq` (line 174); `σ_eq` is the OU
  equilibrium std from the train fit, NOT the TOD profile.

**Composition options for a Kalman layer** (this is a real design fork,
not a detail):

- **(A) Kalman downstream of a TOD-normalized spread.** Run the filter
  on the TOD-deflated innovation. Keeps the intraday-vol shape out of
  the state-noise estimate but couples the filter to a second train-only
  artifact (the TOD profile) whose own warm-up/leakage envelope
  (`tests/intraday/test_zscore.py:69`) must be respected.
- **(B) Absorb TOD into the Kalman observation noise `V_e`.** Make the
  observation-noise variance time-of-day dependent: `V_e(m(t)) ∝
  sigma_tod[m(t)]²`. The filter then natively down-weights noisy
  minutes. Cleaner single-model story; requires the `V_e` selection
  (§3) to be a *profile*, not a scalar.
- **(C) Ignore TOD entirely** (mirror the current OU path, which does).
  Simplest; defensible because the OU best cell already ignores TOD and
  the drift problem is multi-session, not intraday-vol-shaped.

`[TODO design]` pick one; the question is surfaced in §7.1/§7.4. The
default recommendation (argued in §7.1) is **(C) for the first cut**
(match the OU path so the attribution is clean), with **(B)** flagged
as the natural extension once the κ(Δt)-derived `V_e` prior (§3) is in.

------------------------------------------------------------------------

## 3. Leakage discipline for a causal filter

### 3.1 Existing leakage tests (the contract to mirror)

| test | path:line | asserts |
|------|-----------|---------|
| OU fit sees train only | `tests/stats/test_ou.py:297` (`test_fit_does_not_see_post_train_data`) | appending post-train garbage cannot change the fit |
| selector sees (prior,train) only | `tests/backtest/test_walkforward.py:404` (`test_run_walkforward_selection_never_sees_test_dates`) | captured selector window ends strictly before `test_start` |
| rolling z is causal | `tests/signals/test_spread.py:89` | truncating the future never changes a past z |
| sessionized z is causal | `tests/intraday/test_zscore.py:44` | truncating the future never changes a past value |
| TOD profile finite-in-train | `tests/intraday/test_zscore.py:69` | profile NaN strictly outside the trained range (no future leakage) |

### 3.2 The filter-appropriate analogue (why causal test-time updates are NOT leakage)

A Kalman filter **updates its state at every test bar** using that
bar's observation. This is **not leakage** because:

- The update at bar `t` uses only `y_{≤t}` (the spread up to and
  including the current bar) — never `y_{>t}`. It is the same causality
  the rolling-z tests already enforce (`tests/signals/test_spread.py:89`,
  `tests/intraday/test_zscore.py:44`), extended to a recursive state.
- The **hyperparameters** (state-noise ratio δ, observation-noise
  `V_e`, initial state covariance `P_0`, and the burn-in length) are
  the only quantities selected with knowledge of the data
  distribution; they MUST be chosen on **train only** and then
  **frozen** through test. The filter then runs forward causally with
  frozen hyperparameters.

**What the train/test audit must assert** (the new leakage tests to
write in the implementation unit — specified here, not implemented):

1. **Causal-recursion test** (analogue of `test_sessionized_z_is_causal`):
   running the filter on `y[:k]` yields the same state trajectory
   `x̂_{0..k-1}` as running it on the full `y` and truncating — for
   every `k`. `[TODO test]`
2. **Hyperparameter-freeze test** (analogue of
   `test_fit_does_not_see_post_train_data`): δ, `V_e`, `P_0`, burn-in
   are functions of the TRAIN slice only; substituting random bytes for
   the test slice cannot change them. `[TODO test]`
3. **Burn-in containment**: the filter's burn-in window lies entirely
   within train (or within a documented warm-up prefix of test that
   produces NO trades, mirroring the rolling-z warm-up at
   `walkforward.py:26-29`). `[TODO test]`

------------------------------------------------------------------------

## 4. Filter form (notation only — no implementation)

The minimal local-level model on the spread `y_t = log P^Y_t − β logP^X_t − α`
(β, α frozen; state = the equilibrium level μ_t):

```
state:        μ_t = μ_{t-1} + w_t,        w_t ~ N(0, W)         (random-walk equilibrium)
observation:  y_t = μ_t   + e_t,          e_t ~ N(0, V_e)        (spread observed around equilibrium)
predict:      μ̂_{t|t-1} = μ̂_{t-1};   P_{t|t-1} = P_{t-1} + W
innovation:   e_t  = y_t − μ̂_{t|t-1};  Q_t = P_{t|t-1} + V_e
update:       K_t  = P_{t|t-1} / Q_t;  μ̂_t = μ̂_{t|t-1} + K_t e_t;  P_t = (1−K_t) P_{t|t-1}
```

`δ` (the state-noise ratio, West & Harrison discount form) parameterizes
`W = V_e · δ/(1−δ)` so a single δ ∈ (0,1) controls adaptivity. The
**standardized residual** that replaces Z-OU for the drift diagnostic
and the success gate is:

```
z_kalman_t = e_t / sqrt(Q_t)
```

The β-tracking extension promotes the state to `[β_t, μ_t]` with
observation matrix `[−logP^X_t, 1]` (a time-varying-parameter
regression). `[TODO design]` whether β tracking is in scope is §7.1.

This section is **notation to anchor the questions** — no signatures are
proposed; the implementation unit will define exact signatures under the
"exact signatures only" rule once §7 is answered.

------------------------------------------------------------------------

## 5. Attribution design (the 2×2)

`{frozen-β, Kalman} × {rolling_z, OU-style}` on **identical pair-folds**,
corrected `(1+β)` billing, `freq ∈ {5, 15}`, `cost ∈ {1, 3, 5, 8}`,
`stop = none`, **Regime B** (Regime A is structurally infeasible per the
OU report — out of the comparison grid; included only if a reviewer
overrides).

**Cell count.** 2 equilibrium-models × 2 signal-engines × 2 freqs × 4
costs × 1 regime = **32 cells**. The `{frozen-β, OU-style}` and
`{frozen-β, rolling_z}` quadrants are **already produced** (the OU run
and the rolling_z baseline) — only the two **Kalman** quadrants (16
cells) are new compute.

**Runtime estimate from Unit OU actuals.** The 30-cell OU full run was
**301.6 s wall**, dominated by the per-`(pair-fold, freq)` minute-panel
load + resample + OU fit cache (the per-cell loop logged ~0.1 s/cell).
A Kalman pass is `O(n_bars)` per pair-fold — cheap relative to the
panel load. So the two new Kalman quadrants reuse the same cache build
(~300 s) plus a negligible per-cell filter pass ⇒ **≈ 5-6 min wall for
the 16 new cells**, `[TODO compute]` pending the measured cost of one
causal filter pass over a full test slice (expected < 50 ms/pair-fold).

------------------------------------------------------------------------

## 6. HL-band gate under Kalman dynamics

The HL-band selection gate (`scripts/15_phase3_ou.py:268`,
`hl_min ≤ fit.half_life_minutes ≤ hl_max`) currently uses the **static
OU fit's** half-life. Two options:

- **(i) Keep the static OU HL** for selection (train-only), and let the
  Kalman filter only adapt μ at signal time. Selection set is then
  identical to the current OU run ⇒ the 2×2 attribution is on the SAME
  pair-folds by construction. **Recommended** (preserves the matched
  universe).
- **(ii) Re-derive HL from Kalman-implied dynamics** (e.g. from the
  steady-state Kalman gain ⇒ implied AR(1) coefficient). Changes the
  selection set, breaking the identical-pair-fold comparison. Only if
  the static HL is shown to mis-select.

**Either way the selection must remain train-only.** Question §7.6.

------------------------------------------------------------------------

## 7. Design questions (numbered)

1. **State vector.** `μ only` | `β + μ` | `β + μ with TOD handled
   upstream`. **Default argued: `μ only`** for the first cut. Rationale:
   (a) the drift diagnostic is a *level* offset (median 2.1 σ_eq), not a
   slope error — β instability would show as heteroscedastic
   innovations, not a mean offset; (b) `μ only` is a scalar
   local-level filter (one δ, one `V_e`), keeping the train-only
   hyperparameter surface tiny; (c) it displaces exactly one line
   (`scripts/15_phase3_ou.py:174`) and leaves the immutable `Pair` and
   the spread construction untouched, so the 2×2 attribution is clean.
   Promote to `β + μ` only if `μ only` fails the success gate on the
   high-drift pair-folds (INDUSINDBK fold 6). TOD: default **(C) ignore**
   for the first cut (match the OU path), with **(B) absorb-into-`V_e`**
   as the flagged extension.

2. **Update frequency.** Daily-refit-β applied intraday (cheap,
   fold-structured, re-estimates the equilibrium once per session
   boundary) **vs** full intraday Kalman on minute / 5-min bars. The
   drift is **multi-session** (HL ≈ daily HL, 990-5800 min), so a
   minute-bar filter spends most of its bars in the measurement-noise
   regime with little signal. Cost/benefit: a **per-session μ re-anchor**
   (re-fit μ on a trailing window at each session open, frozen
   intra-session) may capture most of the drift correction at a fraction
   of the state-noise risk. Surface both; the minute-bar filter is the
   more general but noisier option. `[TODO design]` which is the primary
   variant for the 2×2.

3. **Hyperparameters.** δ (state-noise ratio) and `V_e` selected on
   **train only** — grid search vs marginal-likelihood maximization of
   the filter's one-step-ahead predictive density on the train slice.
   Whether the per-pair **κ(Δt) decline** (cost-beta report §13.3:
   INDUSINDBK −35.6%, KOTAK −6.5% from 1m→15m) **seeds `V_e`** — a
   larger κ-decline implies a larger microstructure (observation-noise)
   footprint, so `V_e` should be pair-specific, not a shared scalar.
   `[TODO design]` grid vs likelihood; `[TODO design]` exact κ→`V_e` map.

4. **Signal composition — the crux.** Bertram assumes a **stationary OU
   level**; Kalman innovations `e_t` are ~white, so **Bertram-on-
   innovations is incoherent** (there is no mean-reversion left in a
   white series to harvest). Options:
   - **(a) OU/Bertram re-fit on the Kalman-detrended spread level**
     `y_t − μ̂_t` — but a well-tracking filter drives this to white too,
     so the OU fit degenerates. Coherent only if the filter is
     deliberately *slow* (small δ) so a residual stationary component
     remains.
   - **(b) Band rule on `e_t/√Q_t`** with a cost-aware width — enter
     when `|e_t/√Q_t| > a*`, exit at 0. The width `a*` chosen either by
     a **Bertram-analogue** (treat `e_t/√Q_t` as a unit-variance OU with
     the filter's steady-state autocorrelation) or by **empirical
     train-only optimization** of net-of-cost P&L. **This preserves the
     per-cost-level refit principle**: `a*` is re-solved per `(pair-fold,
     freq, cost)` under the corrected `(1+β)` `c`, exactly as the OU
     engine does now.
   - **(c) Hybrid**: slow filter for μ + Bertram on the slow residual.
   **Recommended: (b)** — it is the only option that keeps the
   cost-aware per-cell `a*` refit (the thing the success gate requires
   we not destroy) while being coherent with white innovations.
   `[TODO design]` confirm the `a*`-on-innovations Bertram-analogue vs
   empirical width.

5. **The 2×2 attribution.** Confirmed **32 cells** (2 equilibrium × 2
   engine × 2 freq × 4 cost × Regime B); **16 already exist** (frozen-β
   quadrants), **16 new** (Kalman quadrants). Runtime ≈ 5-6 min wall for
   the new cells (§5), `[TODO compute]` pending one measured filter pass.

6. **HL-band under Kalman.** Re-derive from Kalman-implied dynamics
   **(ii)** vs stay on the static OU fit **(i)**. **Recommended (i)** —
   keep the static train-only HL so the 2×2 runs on identical pair-folds.
   Selection stays train-only in either case (§6).

7. **MLflow.** Cost of wiring it in this unit (a new dependency,
   approval pending) vs staying on `MANIFEST.csv` (the interim artefact
   catalogue per `reporting_standard.md` §8). **Recommendation: stay on
   `MANIFEST.csv` for this unit**; wiring MLflow is ~0.5-1 day (add
   `mlflow` dep, a tracker-context wrapper in `apt.reporting`, and
   `log_artifact` calls in `scripts/16`), independent of the Kalman
   science and better done as its own infra unit. **Decision is the
   reviewer's** (the standard explicitly defers tracker selection to
   this unit's question list).

8. **Repo contradictions.** Scanned `src/`, `scripts/`, `config/`,
   `tests/` for existing `kalman` / `adaptive` / `rolling-cointegration`
   / `state-space` implementations — **none found** (the only `kalman`
   string is the forward-reference flag in `src/apt/intraday/costs.py`
   added by the cost-beta unit). `μ_OU` is confirmed frozen at
   `scripts/15_phase3_ou.py:174` with no existing rolling/recalibration
   path. **No contradiction with these instructions.** Nothing to STOP
   on.

------------------------------------------------------------------------

## 8. Stop conditions

This document ends here per the unit's hard rules: **no implementation,
no behaviour changes; exact signatures only when proposed (none are —
deferred to the build unit); `[TODO]` over invention; stop after the
numbered questions.**

**Next deliverable** (only after §7.1, §7.2, §7.4 are answered): a
filter module under `src/apt/stats/` (or `src/apt/intraday/`) with
exact signatures, train-only hyperparameter selection, the three
leakage tests specified in §3.2, and the 16 new Kalman cells of the
2×2 attribution — all under corrected `(1+β)` billing.
