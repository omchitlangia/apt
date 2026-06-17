# β-escalation design — joint (β, μ) per-session causal filter (Unit K+β)

**Phase 4 Section 3.** Extends the Unit-K μ-only filter
([`docs/kalman_design.md`](kalman_design.md)) by letting the **hedge ratio β
track** alongside the equilibrium level. Target: the INDUSINDBK pair-folds
that the μ-only filter could not neutralize (fold-4 residual drift −0.92 σ_eq;
fold-6 INDUSINDBK −1.79 σ_eq, both at the admissible H=20). Implemented in
`apt.stats.kalman_beta`; driver `scripts/phase4/s3_beta_escalation.py`.

This decision log is written **before** the code (per the Phase-4
instruction).

---

## 1. State / observation equations

Let `y_t = log P^Y_t`, `x_t = log P^X_t` at intraday bar `t`. Sessions `s`
index trading days; `μ`-style updates happen once per session and are carried
**causally** (the parameters applied while trading session `s` were fixed at
the close of session `s−1`; they depend only on sessions `< s`). This is the
exact causal structure of Unit K, lifted to a 2-vector state.

**State** (per session `s`): `θ_s = (β_s, c_s)` where `c_s = α + μ_s` folds
the frozen intercept `α` and the tracking level `μ_s`. Random-walk dynamics
with a diagonal, steady-state (constant-gain) update — the West & Harrison
discount form, one re-anchor half-life per dimension:

```
β_{s+1} = (1 − K_β) β_s + K_β · β̂_s        (β̂_s identified; else carry β_s)
c_{s+1} = (1 − K_c) c_s + K_c · ℓ̂_s
K_x     = 1 − 2^(−1/H_x)                    (H_x in sessions; H=∞ ⇒ K=0)
```

**Session observations** (computed from session-`s` bars, used to update the
state applied in session `s+1`):

* **β observation — returns/increment regression** (the collapse-prone
  estimator, by design — see §3):

  ```
  β̂_s = cov_s(Δx, Δy) / var_s(Δx)
  ```

  over consecutive tradeable intraday log-return increments within session
  `s`. Flagged **unidentified** (state carried, no β update) when the session
  has `< min_session_increments` usable pairs or `var_s(Δx) < min_var_x`.

* **Level observation — concentrated on the current β** (so the level is the
  mean residual at the carried hedge ratio):

  ```
  ℓ̂_s = mean_s( y − β_s · x )
  ```

**Trading residual** at each bar in session `s` (causal — uses the carried
`θ_s`):

```
r_t = y_t − β_s · x_t − c_s
Z_t = r_t / σ_eq_resid
```

`σ_eq_resid` and the Bertram entry threshold `a*` are fit on the **train**
residual (the joint filter run over train), exactly as Unit K, with `a*`
re-solved per cost under the (1+β) billing — billed with the **current**
`β_s` at each round-trip's entry (the cost convention follows the live β).

**Init:** `β_0 = β_frozen` (train daily-EG fit), `c_0 = α + μ_init`
(`μ_init` = train OU `μ`). These reproduce the Unit-K residual at `s = 0`.

## 2. Frozen-control equivalence (the key continuity pin)

Set `H_β = ∞ ⇒ K_β = 0 ⇒ β_s ≡ β_frozen`. Then
`ℓ̂_s = mean_s(y − β_frozen x) = α + mean_s(spread)` (since
`spread = y − β x − α`), so `c_s = α + μ_s` where `μ_s` is **exactly** the
Unit-K per-session level tracker, and
`r_t = y − β_frozen x − c_s = spread − μ_s` = the Unit-K residual,
bit-for-bit. ⇒ **`run_joint_beta_mu(H_β=∞, H_c=H)` reproduces
`run_local_level_mu(H)`** (unit-tested, `test_frozen_control_equivalence`).

## 3. β-collapse diagnostic (the weak-identification warning)

The β observation is a **returns regression** precisely because that is the
estimator that collapses: `β̂_s = cov_s(Δx,Δy)/var_s(Δx)`. When the two legs
**stop co-moving** (intraday `cov(Δx,Δy) → 0`) while the spread keeps moving,
`β̂_s → 0`. The filter then drags `β_s` toward 0, the "hedge" evaporates, and
the residual becomes the (unhedged) `y` with inflated increments. We do **not**
damp this — if it happens it is a FINDING.

Per pair-fold we report the `β_s` path and a **collapse flag**, raised when
both hold over the test window:

* `min_s(β_s) / β_0 < BETA_COLLAPSE_RATIO` (β drifts toward 0), and
* residual-variance instability: `var(test residual) / var(train residual)`
  outside `[1/RESID_VAR_TOL, RESID_VAR_TOL]` (the hedge stopped working).

Labeled defaults (→ ASSUMPTIONS A5): `BETA_COLLAPSE_RATIO = 0.5`,
`RESID_VAR_TOL = 3.0`, `min_session_increments = 10`, `min_var_x = 1e-10`.

## 4. Train-only hyperparameter selection (one global config)

`H_c` is **inherited** from Unit K (= 20 sessions; the μ-only selection is not
re-litigated). `H_β` is selected on **train only** from the grid
`{∞, 40, 20, 10}` sessions by the same criterion as Unit K — analytic Bertram
net-return-per-unit-time on the train residual OU fit at the β-aware 3 bps
reference cost — subject to the **absorption guard extended to β**:

1. residual HL within `[0.5×, 1.5×]` of the frozen-μ HL (Unit-K guard), and
2. **β-stability guard**: `β_s` stays within `[0.25×, 4×] β_0` over the train
   window (no train-time collapse — a collapsing β is inadmissible as a
   *selected* config, though it is still reported as a §3 finding on test).

One global `(H_β, H_c=20)` → test. Selection table persisted to
`reports/phase4/beta_escalation/selection_beta.csv`.

## 5. Decision log

| Q | ruling |
|---|--------|
| B1 state | `(β_s, c_s)`, per-session causal, diagonal constant-gain (West-Harrison discount). NOT a per-bar DLM — keeps exact Unit-K continuity and the per-session leakage tests. |
| B2 β estimator | returns regression `cov(Δx,Δy)/var(Δx)` — the collapse-prone estimator, on purpose (§3). |
| B3 level estimator | concentrated `mean(y − β_s x)` — gives exact frozen equivalence. |
| B4 H_c | inherited = 20 (Unit-K selection, not re-tuned). |
| B5 H_β grid | `{∞,40,20,10}` train-only, Bertram criterion + extended guard. |
| B6 billing | (1+β_s) per round-trip, β following the live filtered β at entry. |
| B7 collapse | reported, never damped. Inadmissible for *selection*, but run + reported on test. |
| B8 universe | the 2 traded survivors (fold-4 INDUSINDBK, fold-6 KOTAK) + the fold-6 INDUSINDBK −1.79σ diagnostic. No re-selection. |
