# OU optimal entry/exit thresholds — design proposal

**Status:** decisions locked 2026-06-10; implementation in progress on
branch `feature/ou-optimal-thresholds` (off `phase/3-intraday`).
The earlier "discovery only" stance (rounds 1-2) is superseded by §8
below. The open-questions block (§7) is RESOLVED — answers and overrides
are encoded as the authoritative defaults in §5 and §8.

------------------------------------------------------------------------

## Decisions changelog (2026-06-10)

User-confirmed decisions and overrides (verbatim where load-bearing,
paraphrased elsewhere). See §8 for the worked statements; this is the
index.

- **Cost model.** Canonical = intraday `CostBreakdown` (4.5 bps fixed +
  bid-ask spread sweep `{1, 3, 5, 8}`). Daily `cost_bps_per_leg: 25.0`
  in `config/default.yaml:98` is **delivery-regime** and must NOT enter
  the OU objective. Equal-notional deduction inherited verbatim from
  v2: `cost_log_per_pair_round_trip = 2 × cost_bps_per_leg / 10_000`
  (n_legs = 2). Do **NOT** correct to `(1+β) ×` — diagnostic table
  records per-pair `β` and `(1+β)/2` distribution; if median `|(1+β)/2
  − 1| > 0.10` a `[TODO]` flags cost-accounting as follow-up. The
  zero-price-move round-trip test pins deduction plumbing, not the
  notional convention.

- **Re-fit per cost level.** OU params fit ONCE per pair-fold ×
  bar-frequency (cost-independent). Bertram 1-D solve repeats per
  cost level. Evaluation is **diagonal**: policy fitted at `c_i` is
  evaluated at `c_i`. Cross-evaluation (fit at 3, evaluate elsewhere)
  is behind an optional flag, **default off, not run this round**.

- **Cost units.** Derived in the spread's own log units from
  `CostBreakdown`. Mandatory zero-move test pins it. See §8.1 for the
  worked example.

- **Signal engine.** `signal.engine ∈ {"rolling_z", "ou"}`, default
  `rolling_z`. OU path uses `Z-OU = (X − μ_OU)/σ_eq` with both
  TRAIN-frozen. Entry: short spread at `Z ≥ +a*`, long at `Z ≤ −a*`.
  Exit: mean-touch (`Z` crosses 0). Re-entry on band recross.
  Preserves v2 completed-bar / execution-timing / entry-confirmation
  semantics exactly (§8.2).

- **Stop.** `stop.mode ∈ {"none", "hard"}`, default `none`. `hard` →
  exit at `|Z| ≥ K`, default `K = stop.k_sigma = 4.0` against Z-OU.
  Existing `3 × half-life` time-stop and Regime A EOD square-off kept
  as-is (orthogonal overlays, not optimized this round).

- **Bar frequency.** Sweep `{1, 5, 15}` minutes. Resample within
  session only — no bar spans the overnight gap. OU fit and trading
  occur at the same frequency per cell. Left-labeled windows;
  completed-bar causality preserved at the new frequency.

- **Half-life band filter.** Config in **trading minutes** (canonical
  unit), converted to bars internally per active frequency. Defaults:
  Regime A `[30, 120]`, Regime B `[120, 1875]` minutes (375 trading
  minutes per session, 9:15–15:30 IST). Output full HL distribution
  across pair-folds.

- **HL ratio diagnostic.** Report `intraday_OU_HL_minutes /
  (daily_inherited_HL_days × 375)` per pair-fold. **No flag band, no
  hard drop** — the daily gate `[5, 60]` days makes ratios ≪ 1 the
  expected case; distribution informs band-setting at review.

- **Pair-fold exclusions.** AR(1) slope ≥ 0 on train slice → OU fit
  invalid → exclude pair-fold, counted. Report no-reversion fits,
  HL-band rejects, other rejects per cell.

- **Z-OU drift diagnostic.** Test-slice mean of `Z-OU` per pair-fold;
  flag `|mean| > 0.5` as stale-mean warning.

- **Coarse-bar rolling_z baselines.** Per addendum: add rolling_z runs
  at `{5, 15}` min × `{A, B}` to isolate the aggregation effect from
  the engine effect. Window/max-holding inherited via minute-equivalent
  conversion; thresholds unchanged.

- **Default-config regression.** `signal.engine == "rolling_z"` with
  all other knobs at v2 defaults must reproduce the v2 outputs on a
  fixed fold — exact byte equality if deterministic, else tight
  numerical tolerance. We expect exact (tradelist + per-bar gross/net)
  per the immutable-spec test.

- **Dependencies.** None added. AR(1) via `statsmodels.OLS`; `erfi`,
  bounded scalar opt via `scipy.special.erfi`, `scipy.optimize.
  minimize_scalar` (transitive dep via statsmodels).

- **References.** Bertram (2010) primary; Leung & Li (2015) cross-check;
  Zeng & Lee (2014) two-sided form. **No transcription from memory or
  from summaries.** Monte-Carlo first-passage test (§8.3) is the
  decisive validator.

- **Corp-action caveat.** v2 intraday loader does NOT adjust for
  corporate actions; raw minute closes can diverge from daily-adjusted
  by 7–10% on action days. We do not build adjustment machinery this
  round; affected pair-folds listed in the report as a data caveat
  (§8.4).

**Goal:** replace the ad-hoc z-score bands (enter ±2.0σ, take-profit
±0.5σ, stop ±3.5σ) in the intraday signal layer with Ornstein-Uhlenbeck
(OU) **transaction-cost-aware** entry/exit thresholds, plus a tighter
**half-life-band filter** at the intraday selection step. Out of scope:
Johansen, regime/HMM, Kalman-β, RL, Phase 1/2 daily pipeline.

This document is structured as:

1. Existing-code map (exact paths, line ranges, signatures).
2. OU fitting plan (TRAIN slice only, with a verifiable primary reference
   and unit-test plan).
3. Threshold derivation plan and integration points.
4. Half-life-band filter.
5. Proposed function signatures, config surface, gross/net reporting.
6. Leakage section.
7. Open questions for the user (Step 3 of the brief — answer these
   before any implementation).

------------------------------------------------------------------------

## 1. Existing-code map

All paths relative to `/Data6/apt`. Line ranges are inclusive.

### 1.1 Intraday spread construction and frozen daily β

**Daily β is fit by Engle-Granger (best of two regression directions) on
each fold's TRAIN window** and carried forward, never re-fit on the
test slice:

- **`src/apt/signals/cointegration.py:123-151`** —
  `engle_granger(y, x, *, y_sym='y', x_sym='x') -> EGResult`. Runs
  `statsmodels.api.OLS(y, sm.add_constant(x))` on log prices and ADF on
  the residual. The dataclass `EGResult` (lines 104-120) carries
  `alpha`, `beta`, `residual`, `adf_pvalue`, `adf_stat`, `n_obs`.

- **`src/apt/signals/cointegration.py:154-164`** —
  `engle_granger_best_direction(p_a, p_b, *, sym_a, sym_b) -> EGResult`.
  Tries both directions, returns whichever has the smaller ADF p-value;
  the returned `beta` is what gets frozen for the fold.

- **`src/apt/signals/cointegration.py:376-689`** —
  `cointegrate_pairs(daily, sectors, *, start, end, prior_start=None,
  prior_end=None, ...)`. The windowed entry point. Stage layout
  documented in the docstring at lines 396-422. Returns
  `CointegrationResult(pairs: pl.DataFrame, funnel:
  CointegrationFunnel, notes: list[str])`. The output `pairs` frame
  schema is `_PAIR_SCHEMA` at lines 266-287 and includes `y_sym`,
  `x_sym`, `alpha`, `beta`, `half_life`, `half_life_in_band`, `hurst`,
  `hurst_pass`, `fdr_pass`, `stable_prior_window`,
  `is_structural_pair`.

- **`src/apt/backtest/walkforward.py:64-78`** — `Pair` dataclass holds
  `y_sym`, `x_sym`, `alpha`, `beta`, `half_life`, `sector`,
  `is_structural`. This is the **frozen-per-fold pair specification**
  that the intraday engine consumes.

- **`src/apt/signals/spread.py:41-67`** — `compute_spread(p1, p2, *,
  beta, intercept=0.0) -> np.ndarray`. The actual application of the
  daily-fit β to a (potentially intraday) price series:
  ```python
  return np.log(a) - beta * np.log(b) - intercept
  ```
  No re-fit. **This is the universal spread definition used at both
  daily and minute frequency.**

- **`scripts/13_phase3_intraday.py:263-267`** (inside
  `_compute_pair_fold`) — the intraday application:
  ```python
  with np.errstate(invalid="ignore", divide="ignore"):
      py = np.where(aligned_full.tradeable, aligned_full.close_y, np.nan)
      px = np.where(aligned_full.tradeable, aligned_full.close_x, np.nan)
      spread_full = np.log(py) - pair.beta * np.log(px) - pair.alpha
  ```
  Note: the script inlines the spread formula rather than calling
  `compute_spread`, because the loader returns NaN for non-tradeable
  bars and `compute_spread` raises on non-positive prices. Semantically
  identical to `compute_spread` on the tradeable subset.

- **`src/apt/intraday/loader.py:46-72`** — `AlignedMinutePair` is the
  loader's output: `timestamps` (tz-aware IST), `session_id` (dense
  rank), `bar_in_session`, `close_y`, `close_x`, `volume_y`,
  `volume_x`, `tradeable`. Non-tradeable bars carry NaN closes — they
  are **never forward-filled** (Epps-safe).

- **`src/apt/intraday/loader.py:115-196`** — `load_minute_pair(y_sym,
  x_sym, start, end, *, root=...) -> AlignedMinutePair`. Reads the
  hive-partitioned parquet at `data/interim/minute_raw/symbol=<S>/
  year=<Y>/data.parquet`, drops pre-open / post-close prints
  (continuous session 09:15-15:29), unions both legs onto a common
  minute grid, marks non-trade minutes non-tradeable.

### 1.2 z-score computation: window, EWMA?, completed-bar vs intrabar

The production intraday signal uses a **multi-session chronological
rolling z** (no EWMA), causal, completed-bar (no intrabar look-ahead):

- **`src/apt/signals/spread.py:75-111`** — `rolling_zscore(spread, *,
  window=60, min_periods=None, ddof=1) -> np.ndarray`. The
  asset-agnostic primitive. Uses polars
  `Series.rolling_{mean,std}(window_size=window,
  min_samples=min_periods)`. Sample std (`ddof=1`). The docstring
  explicitly guarantees causality: index `t` depends only on
  `spread[max(0, t-window+1) : t+1]`.

- **`src/apt/intraday/zscore.py:51-98`** —
  `intraday_rolling_zscore(spread, session_id, *, window,
  min_periods=None, session_warmup_bars=0) -> np.ndarray`. The
  production intraday flavor: chronological multi-session window
  (rolling stats are over LEVELS, so a multi-session window does NOT
  "blend" the overnight return into the mean); first
  `session_warmup_bars` of each session are forced to NaN to suppress
  open-auction noise.

- **TOD-adjusted z** (`src/apt/intraday/zscore.py:240-293` —
  `tod_adjusted_zscore(...)`, with profile fitting at lines 176-237
  `fit_tod_vol_profile(...)`) was tested and **strictly worse than the
  flat z in v2 results**: sensitivity diag for HDFC/HDFCBANK fold 7
  showed `tod` produces 1055 trades vs `flat` 236, net Sharpe -2.22 vs
  -1.47. The production path uses the flat z only.

- **No EWMA** anywhere in the production path. No intrabar updates —
  every input quantity is bar-completed `close` data.

- **Window length policy** (per-pair, set at
  `scripts/13_phase3_intraday.py:269-276`):
  ```python
  window = int(np.clip(round(pair.half_life * NSE_BARS_PER_SESSION),
                       MIN_ROLLING_WINDOW_MIN,
                       MAX_ROLLING_WINDOW_MIN))
  ```
  i.e. `half_life_days * 375` clamped to `[1 session, 5 sessions]`.
  `MIN_ROLLING_WINDOW_MIN = 375`, `MAX_ROLLING_WINDOW_MIN = 5 * 375 =
  1875` at lines 98-99. `SESSION_WARMUP_BARS = 15` at line 101.

- **Production call site for the intraday z**
  (`scripts/13_phase3_intraday.py:310-318`):
  ```python
  z_full = intraday_rolling_zscore(
      spread_full,
      sids_full,
      window=window,
      session_warmup_bars=SESSION_WARMUP_BARS,
  )
  z_flat = z_full[test_mask]
  ```
  Critical observation: `z_full` is computed over **TRAIN+TEST
  concatenated** then sliced to test. This is intentional warm-up
  (rolling-window levels at test start are seeded from train data) and
  is leakage-free because rolling-z at `t` depends only on
  `spread[≤t]`. **The OU fit, in contrast, must use TRAIN data only
  — this is the key leakage-prevention point for the new work**.

### 1.3 Entry / take-profit / stop logic (±2 / ±0.5 / ±3.5)

The thresholds live in the asset-agnostic state machine:

- **`src/apt/signals/spread.py:141-239`** —
  `generate_signals(z, *, entry=2.0, exit=0.5, stop=3.5,
  max_holding=60) -> SignalSeries`. State machine:
  - flat → long when `z < -entry`
  - flat → short when `z > +entry`
  - long → flat: `z >= -exit` ⇒ `'mean_revert'`; `z <= -stop` ⇒
    `'stop'`; `days_in_trade >= max_holding` ⇒ `'time'`
  - short → flat: mirror.

  NaN-z bars carry state forward, `days_in_trade` does NOT increment
  through NaN. **All three thresholds parameterise the same z-score
  series — there is no separate notion of "TP" vs "exit".**

- **`src/apt/intraday/signals.py:98-158`** —
  `generate_signals_two_regime(z, session_id, tradeable, *, regime,
  entry=2.0, exit=0.5, stop=3.5, max_holding=60) ->
  IntradaySignalSeries`. Calls `generate_signals` per session (Regime
  A) or once continuously (Regime B). Adds a force-close at the last
  tradeable bar of each session in Regime A
  (`exit_reason='session_close'`).

- **Default values come from config**
  (`config/default.yaml:80-87`): `entry_z: 2.0`, `exit_z: 0.5`,
  `stop_z: 3.5`, plus `max_holding_cap_days: 60` and
  `max_holding_half_life_multiplier: 3.0`. **There is no separate
  intraday override section — daily and intraday share the same
  thresholds.**

- **Production call site** for the intraday entry/exit thresholds is
  `scripts/13_phase3_intraday.py:344-348`:
  ```python
  sig = generate_signals_two_regime(
      ...,
      entry=settings.signal.entry_z,
      exit=settings.signal.exit_z,
      stop=settings.signal.stop_z,
      max_holding=max_holding_bars,
  )
  ```
  `max_holding_bars` is clamped per-pair at lines 322-335 to
  `min(cap_days × 375, ceil(half_life × 375 × 3.0))`.

### 1.4 Engle-Granger + BH-FDR + half-life

EG and BH-FDR live in `src/apt/signals/cointegration.py`:

- **Engle-Granger**: `engle_granger` (123-151),
  `engle_granger_best_direction` (154-164) — covered above.

- **BH-FDR**: `benjamini_hochberg(pvalues, alpha=0.05) ->
  np.ndarray` (lines 240-258). Step-up procedure: sorts p-values,
  finds the largest k with `p_(k) <= (k/n)·alpha`, rejects all `p <=
  p_(k*)`.

- **Half-life IS already computed in the daily Phase 1 pipeline.**
  `half_life_ar1(spread) -> float` at **`src/apt/signals/cointegration.py:172-193`**:
  ```python
  y = s[1:]
  x = sm.add_constant(s[:-1])
  fit = sm.OLS(y, x).fit()
  phi = float(fit.params[1])
  ...
  return float(-np.log(2.0) / np.log(phi))
  ```
  This is the discrete AR(1) coefficient → half-life mapping. **It is
  fit on the TRAIN-window residual** (i.e. on the OLS residual returned
  by `engle_granger`, which is itself the train-window EG residual).
  The result lives in the `pairs` frame as the `half_life` column and
  is gated by `half_life_in_band` (lines 588-592), which checks
  `half_life_min_days <= hl <= half_life_max_days` with the band
  configured in `config/default.yaml:63-64` as `[5, 60]` days.

- **Hurst exponent** is at lines 201-232, similarly fit on the
  residual; gated by `hurst_pass` (lines 593-595) with `hurst_max=0.5`.

The half-life and Hurst on the **daily** residual are what the intraday
layer inherits via `Pair.half_life`. There is currently **no intraday
half-life estimation** anywhere.

### 1.5 Walk-forward train/test split exposure

- **`src/apt/backtest/walkforward.py:81-89`** — `Fold(fold_id,
  prior_start, prior_end, train_start, train_end, test_start,
  test_end)`. The atomic unit. `prior_start < prior_end < train_start
  <= train_end < test_start <= test_end`, all contiguous trading days.

- **`src/apt/backtest/walkforward.py:133-190`** —
  `build_folds(trading_days, *, prior_days, train_days, test_days,
  step_days=None) -> list[Fold]`. Default `step_days = test_days` (no
  overlap). Phase 2A uses `prior_days=1008, train_days=1008,
  test_days=252, step_days=252` (~4 train years, ~1 test year,
  annualised step).

- **The intraday script `scripts/13_phase3_intraday.py:116-132`**
  builds folds, then keeps only those whose test window overlaps the
  minute panel (2015-02-02..2021-06-23) → 6 in-scope folds (IDs 2-7).
  Daily selection is per-fold at lines 135-175
  (`_select_pairs_for_fold`); the train window is `fold.train_start ..
  fold.train_end`, and **`cointegrate_pairs` is called with `start=
  fold.train_start, end=fold.train_end`** — so β and α are TRAIN-ONLY.

- **The intraday per-pair worker `_compute_pair_fold` at
  `scripts/13_phase3_intraday.py:240-372`** takes a fold and loads
  TRAIN+TEST minute data (lines 256, 280-281), then **slices to a
  `train_mask` and `test_mask`**:
  ```python
  train_mask = (ts_full.date >= train_start) & (ts_full.date <= train_end)
  test_mask  = (ts_full.date >= test_start)  & (ts_full.date <= test_end)
  ```
  The TOD profile is fit on `spread_full[train_mask]` only (line 292);
  trades are emitted only on the test slice (lines 301-308). **This is
  the integration point for fitting OU on train and freezing for
  test.** A new helper analogous to `fit_tod_vol_profile` — call it
  `fit_ou_params` — would take the same train mask and produce frozen
  OU parameters.

- **Critical detail**: the rolling-z `z_full` (lines 312-317) is
  computed over the **full TRAIN+TEST series**, then sliced. This is
  causal (rolling at `t` only sees past), so the test slice has a
  warmed-up z at its first bar. **OU parameter estimation must NOT do
  this** — OU MLE/OLS uses all observations in the fit window, so it
  has to see TRAIN data only.

### 1.6 Transaction cost — how it enters P&L and in what units

- **`src/apt/intraday/costs.py`** is the canonical cost model
  (full file, 95 lines). The unit is bps per LEG per ROUND-TRIP.
  Components:
  - `STT_BPS_PER_LEG_RT = 2.5` (sell-side 0.025% per leg, [TODO at
    line 8: confirm 2025 budget rate]).
  - `BROKERAGE_BPS_PER_LEG_RT = 0.5`.
  - `EXCHANGE_REG_BPS_PER_LEG_RT = 1.5`.
  - `FIXED_PER_LEG_RT = 4.5` (sum).
  - `SPREAD_SWEEP_BPS = (1, 3, 5, 8)` (total quoted spread; half-spread
    = value/2). **A round-trip on one leg crosses the spread once at
    entry and once at exit, so the per-leg per-round-trip spread cost
    equals the FULL quoted spread.**

- **`CostBreakdown` dataclass** (lines 50-79) computes:
  - `cost_bps_per_leg = FIXED_PER_LEG_RT + total_spread_bps`
  - `cost_per_pair_round_trip_bps = 2 × cost_bps_per_leg`
  - `cost_log_per_pair_round_trip = cost_per_pair_round_trip_bps /
    10_000` (LOG-RETURN UNITS — this is what gets subtracted from the
    log spread move on the exit bar).

- **The "25 bps/leg" config value** at `config/default.yaml:98`
  (`cost_bps_per_leg: 25.0`) is the **Phase 2A daily** number, used by
  `apt.backtest.walkforward.run_walkforward`. The Phase 3 intraday
  path does **NOT** use this — it uses the `CostBreakdown` model
  directly with the {1,3,5,8} bps sweep, and the fixed per-leg
  per-round-trip is 4.5 bps not 25 bps. The 25-bps figure was
  brokerage + STT + slippage lumped at daily frequency where
  slippage dominates and the half-day-hold cost amortizes; intraday
  unbundles it.

- **How cost enters P&L** — at `src/apt/intraday/backtest.py:166-168`,
  per round-trip on the exit bar:
  ```python
  cost = cost_log_per_round_trip
  net_pnl = gross_pnl - cost
  net[exit_idx] -= cost
  ```
  So **`gross_log_pnl = direction × (spread[exit] - spread[entry])`**
  (line 165), and the cost is a single scalar deducted on the exit bar
  only. Re-stamping the cost for a different spread sweep is done
  cheaply by `_net_pnl_for_cost` at
  `scripts/13_phase3_intraday.py:380-422`, which subtracts the new
  `cost_log` per trade.

### 1.7 Schema of trade CSVs and minute CSVs

**Trade CSVs** at `reports/phase3/trades_all_pairs_{A,B}.csv` (committed,
gitignored locally) — 20 columns:

```
fold_id (int), regime ('A'|'B'), pair (str like 'PFC/SBIN'),
sector (str), is_structural (bool), is_hdfcbank_anchored (bool),
side ('long_spread'|'short_spread'),
entry_ts (ISO 8601 with +05:30 tz), exit_ts (same),
z_entry (float), z_exit (float),
bars_held (int), sessions_held (int),
gross_log_pnl (float), net_log_pnl_at_3bps (float),
gross_pct (float, =expm1(gross_log)*100),
net_pct_at_3bps (float),
cost_bps_excl_spread_per_leg_rt (= 4.5),
n_legs (= 2),
exit_reason ('mean_revert' | 'z_stop' | 'time_stop' |
             'eod_squareoff' | 'fold_close')
```

Note the exit_reason taxonomy in the CSV is renamed from the engine's
internal names (`stop` → `z_stop`, `time` → `time_stop`,
`session_close` → `eod_squareoff`, `fold_boundary` → `fold_close`).

**Raw minute CSVs** at `/Data6/db/minute/<SYMBOL>_minute-data.csv` (492
symbols, ~33-35 MB each, ~590k rows per symbol):

```
date,open,high,low,close,volume
2015-02-02 09:15:00+05:30,1554.9,1556.7,1544.8,1549.6,1441
```

- `date`: ISO 8601 with `+05:30` IST tz suffix; the column name is
  `date` even though the value is a full datetime.
- `open, high, low, close`: floats (raw rupee prices, **NOT
  adjusted** — see Step 1b adjustment cross-check in the Phase 3 v2
  report: median minute-close / daily-adjusted-close = 1.000 with
  ~7-10% max divergence on a handful of corporate-action days).
- `volume`: int, shares traded in the minute.
- Bar-stamp: **OPEN-labeled** (the bar at 09:15 covers `[09:15,
  09:16)`).

The repository's converted parquet form at
`data/interim/minute_raw/symbol=<S>/year=<Y>/data.parquet` adds
`timestamp` (tz-aware `Asia/Kolkata`), `date` (date32), and a
`partial_day_flag` bool. Schema source: Phase 3 Step 0 probe.

**β / spread units relative to price**: β is the OLS slope from
log-price regression on the TRAIN window; the spread is
`log(p_y) − β·log(p_x) − α` and is therefore dimensionless (in log-units
of price). A spread change of `Δs` corresponds to a log return of `Δs`
on a unit-notional long-y / short-(β units of x) portfolio. Costs are
in **bps of notional** and convert to log-return units by dividing by
10,000.

------------------------------------------------------------------------

## 2. OU fitting — primary reference and unit-test plan

### 2.1 The two-step plan

**Discrete observed minute bars → continuous-time OU process**.

The Ornstein-Uhlenbeck SDE has many equivalent parameterisations and
different sources name the reversion-speed parameter differently
(some call it α, others μ, others θ, others κ). The closed-forms for
Bertram's optimal thresholds depend on this convention. **I will
verify both the OU MLE estimator and the Bertram closed-forms against
a single primary reference rather than transcribing from memory.**

**Primary reference (proposed):**

- Bertram, W. K. (2010). "Analytic solutions for optimal statistical
  arbitrage trading." *Physica A: Statistical Mechanics and its
  Applications*, 389(11), 2234-2243.

  This is the original paper, peer-reviewed, with the closed-form
  optimal entry/exit thresholds for an OU mean-reverting spread under
  the **expected-return-per-unit-time** objective net of round-trip
  cost. **The formulas and parameterisation in this paper are the
  authoritative source — I will not rely on summaries.**

- Backup / cross-check reference: Leung & Li (2015), "Optimal Mean
  Reversion Trading," World Scientific. Provides a textbook treatment
  of OU and a more general optimal-stopping derivation that should
  reproduce Bertram as a special case.

I will not transcribe specific formulas into this design doc — that's
where transcription errors live. Implementation will quote the paper
formulas verbatim into a module docstring with equation numbers, and
the unit-test plan below will cross-check those formulas against three
independent sanity checks before any backtest result is generated.

### 2.2 OU MLE on the TRAIN slice

The OU SDE on the (log-)spread is:

```
dX_t = κ (μ − X_t) dt + σ dW_t          (1)
```

where `κ > 0` is the mean-reversion speed, `μ` the long-run mean,
`σ` the diffusion coefficient. **Half-life is `ln(2)/κ`**, equivalently
the time at which an OU innovation decays to half its initial
displacement.

For discrete minute bars sampled at `Δt = 1/60 hour = 1/(60·6.25)
session = 1/375 session-day`, the exact discretisation is:

```
X_{t+Δt} = μ (1 − e^{−κΔt}) + e^{−κΔt} X_t + ε_t          (2)
ε_t ~ N(0, σ² (1 − e^{−2κΔt}) / (2κ))
```

This is **exactly an AR(1) on `X_t`** with coefficient `φ = e^{−κΔt}`
and intercept `μ(1 − φ)`. The existing `half_life_ar1` at
`src/apt/signals/cointegration.py:172-193` already computes this
quantity at daily frequency on the EG residual. The intraday port:

1. Fit AR(1) on the TRAIN-window log spread (chronological, the
   sample is the train-mask slice of `spread_full` in `_compute_pair_fold`):
   ```
   X_{t+1} = c + φ · X_t + ε_t,    ε_t ~ N(0, σ_ε²)
   ```
2. Map AR(1) parameters → OU:
   - `κ = −ln(φ) / Δt`  (requires `0 < φ < 1`)
   - `μ = c / (1 − φ)`
   - `σ² = σ_ε² · 2κ / (1 − φ²) = σ_ε² · 2κ / (1 − e^{−2κΔt})`
   - `half_life = ln(2) / κ` (in the same time units as Δt)
3. Equilibrium variance: `σ_eq² = σ² / (2κ)`. This is the
   **stationary variance of the OU process** and is what defines the
   "natural" z-score scale: `(X − μ) / σ_eq` is a unit-variance
   stationary z under the OU model.

**Units convention to be FROZEN in code**:

- `Δt = 1` bar = 1 minute. The OU κ comes out in **per-bar units**;
  half-life in bars.
- Half-life in days = half_life_bars / 375.
- When deriving Bertram's thresholds the time unit and the cost scale
  must agree — to be checked in the unit tests below.

### 2.3 Unit-test plan for OU fitting

This is the leverage point — if these tests don't pass we don't trust
the downstream thresholds.

**Test 1 — round-trip on synthetic data.** Simulate an OU process with
known `(κ, μ, σ)` via the exact transition (eq. 2). Fit AR(1) → OU
map. Assert recovered params are within tolerance:
- κ recovered within ±5% with `N = 50,000` samples.
- μ recovered within ±0.01σ_eq with `N = 50,000` samples.
- σ recovered within ±2% with `N = 50,000` samples.

**Test 2 — half-life invariance.** Same simulation; assert
`half_life_bars × Δt = half_life_days` matches the closed-form
`ln(2)/κ` regardless of which time unit the AR(1) was fit on (sanity
check that κ rescales correctly when Δt changes).

**Test 3 — degenerate cases.**
- φ ≤ 0 → return `nan` (anti-persistent flip, matching existing
  `half_life_ar1`).
- φ ≥ 1 → return `inf` for half-life, `nan` for κ, signal "not OU
  mean-reverting" so the pair is dropped (matches existing semantics).
- Constant series → `nan` everything.

**Test 4 — agreement with `half_life_ar1` on daily residuals.** Run
the new OU fitter on the existing Phase 1 EG residuals (the
`half_life` column already in the cointegration funnel) and assert the
half-life agrees with the existing `half_life_ar1` to within numerical
tolerance — proves we did not introduce a regression in the daily
pipeline (which we are NOT supposed to touch). This is also a backstop
against transcription errors in (2).

**Test 5 — Bertram threshold sanity (closed-form cross-check).**

Before trusting Bertram's optimal-threshold formulas numerically, the
implementation must pass three independent sanity checks (paper
formulas to be transcribed verbatim with equation numbers; values
below are the **qualitative** properties any correct implementation
must satisfy, not numerical claims):

- **(a)** Zero-cost limit: as `c → 0`, the optimal entry threshold
  `a* → 0` (with continuous costless rebalancing the OU strategy
  trades on every reversion). The implementation must reproduce this
  behavior numerically; it does NOT need to match Bertram-paper
  numerical examples (which I will not transcribe).
- **(b)** Large-cost limit: as `c → ∞`, expected per-unit-time return
  drops to 0 from below; the optimal `a*` grows monotonically; eventually
  trading stops (an "infeasible" verdict).
- **(c)** κ-monotonicity: at fixed cost and σ, holding `σ_eq²` constant,
  faster mean reversion (larger κ) implies higher expected return per
  unit time at the same `a*` — and lower optimal `a*`.

These three are coordinate-free properties of the OU optimisation
problem and any correct implementation must satisfy them. Failure of
any of them means a transcription error in the paper formula.

**Test 6 — known-value pinning, deferred.** Once Test 5 (a/b/c)
passes, pin a single set of numerical thresholds at a chosen `(κ, μ,
σ_eq, c)` to lock in the implementation. The pinned values come from
the implementation's own first-pass output AFTER (a/b/c) pass, not
from outside — this guards against silent drift in later refactors.
**This pin is generated by the test the first time it runs, not
transcribed by me.** [TODO: choose the (κ, σ_eq, c) reference point
once the implementation is in place.]

### 2.4 OU long-run mean vs rolling z-mean — which to use in the z?

The Phase 3 production z is `(spread − rollmean) / rollstd` with a
window of `clip(half_life × 375, [375, 1875])` (i.e. 1-5 sessions).
The rolling mean **drifts with the spread** — this is the design
choice that makes the strategy adapt to slow regime change but it also
means the "mean" the strategy reverts to is itself noisy.

The OU model gives a **train-fit, constant `μ` and `σ_eq`**. Three
options for the integration:

- **Option Z-rolling (the existing default)**: keep `(spread −
  rollmean) / rollstd` as the z definition, but use Bertram's
  thresholds (expressed in z-units after conversion: `a_z = a / σ_eq`)
  in place of the literal ±2 / ±0.5 / ±3.5.
- **Option Z-OU (proposed)**: use `(spread − μ_OU) / σ_eq` as the z,
  where `μ_OU` and `σ_eq` are TRAIN-frozen. This is the model-consistent
  z; Bertram's thresholds are exactly the z-bands. **No noisy
  rolling-window estimator anywhere in the signal layer.**
- **Option Z-hybrid**: hybrid `(spread − rollmean) / σ_eq` — robust to
  slow regime drift via rollmean, but the σ_eq scale is OU-frozen
  (less noisy denominator than rolling-std).

I will implement Option Z-OU first because it is the model-consistent
choice and removes one source of overfitting (the rolling-window
length). Options Z-rolling and Z-hybrid will live behind a config flag
for ablation. **The chosen default is open question Q-Z below.**

------------------------------------------------------------------------

## 3. Threshold derivation and integration points

### 3.1 Bertram (2010) thresholds — what they replace

Bertram's framework solves for an entry threshold `a` and exit
threshold `b` (with `b = μ` in the simplest version, "exit at the
mean") that maximises the **expected return per unit time** of an OU
trading strategy NET of round-trip cost `c`. The threshold pair
`(a, m)` (entry at `a` above/below mean, exit at the mean) is solved
in closed form from `(κ, σ_eq, c)`.

**Closed-form formulas: to be transcribed verbatim from the Bertram
2010 paper into a module docstring with the equation numbers and
notation kept identical to the paper.** I will NOT write the formulas
here, to avoid transcription errors propagating into requirements.
**The implementation will quote the paper and be verified by the unit
tests in §2.3.**

What changes in the signal layer:

```
current:   enter when |z| > 2.0, exit when |z| < 0.5, stop when |z| > 3.5
proposed:  enter when |X − μ| > a*,                   (a* from Bertram)
           exit  when X crosses μ,                    (the OU long-run mean)
           stop:  see §3.3 below — separate decision.
```

### 3.2 Where OU output plugs in — proposed integration

The least-invasive integration point is **`_compute_pair_fold` in
`scripts/13_phase3_intraday.py:240-372`**. Concretely:

1. Right after the train_mask / test_mask are built (around line 308),
   add a call to a new `fit_ou_params(spread_full[train_mask],
   minute_bars_per_unit) -> OUFit` (signature below).
2. From `OUFit` derive the optimal threshold `a*` and (optionally)
   the exit threshold via a new
   `bertram_thresholds(ou_fit, cost_log_per_round_trip) ->
   OUThresholds`.
3. Pass these into a new `generate_signals_ou(spread, session_id,
   tradeable, *, ou_fit, ou_thresholds, regime, ...)` that replaces
   the current `generate_signals_two_regime` call when `signal.engine
   == "ou"` is set in config.

The existing `generate_signals_two_regime` is **kept and left as the
default** for now (open Q-default below).

### 3.3 The ±3.5σ stop under a "exit at the mean" model

In Bertram's framework the strategy holds until the spread crosses the
mean, with no explicit "diverging-spread stop." Three options:

- **Option Stop-drop**: remove the stop entirely. The OU half-life cap
  in §4 plus the existing `max_holding_bars` cap (already present at
  lines 322-335) bound the maximum loss horizon. Bertram's expected
  return is computed under no-stop semantics, so this is the only
  option that is theoretically consistent with the formula.
- **Option Stop-overlay**: keep `±stop_z` as a hard catastrophic stop
  (in z-units against `σ_eq`, not against rolling-std), but treat it
  as **out-of-model** insurance. Bertram's expected-return claim is
  weakened.
- **Option Stop-ablate**: implement both, run side-by-side in the
  walk-forward, report Sharpe / drawdown for each. The data decides.

I propose Option **Stop-ablate** — implement once with a config flag,
run both, report. Open Q-stop below.

### 3.4 Reporting gross AND net side by side

The existing `_net_pnl_for_cost` (`scripts/13_phase3_intraday.py:380-
422`) already produces gross / net at any spread level on demand from
the cached gross-only series. The OU work will reuse this verbatim —
the spread-sweep table at `reports/phase3/metrics_two_regime_v2.csv`
already reports `gross_total_pct, net_total_pct, gross_ann_pct,
net_ann_pct, gross_sharpe, net_sharpe, net_max_drawdown_pct` per
(regime × spread_bps × variant). The new OU run will produce the same
schema in a new file `reports/phase3/metrics_ou_thresholds.csv` so
the comparison vs the existing ad-hoc-threshold result is one
join-on-(regime, spread_bps, variant).

------------------------------------------------------------------------

## 4. Half-life-band filter at the intraday selection step

The current selection runs the daily Phase 2A funnel and inherits
`pair.half_life` (daily AR(1) on the daily EG residual). Two issues:

- **The daily half-life can be in the daily band [5, 60] days while the
  intraday half-life is wildly different** (much shorter when the
  pair has microstructure noise; much longer when daily mean reversion
  doesn't manifest intraday). The Phase 3 v2 ONGC/OIL fold-7 result
  (the only Regime-A pair-fold to clear costs) suggests at least some
  pairs DO mean-revert intraday on a tractable timescale.

- **No filter exists today on the intraday half-life.** Pairs with
  intraday half-lives of a few bars (noise) or > 5 sessions (does not
  fit in the holding horizon) are currently traded the same way.

Proposed filter, applied AFTER the existing fill-rate gate (so we
still measure intraday half-life on liquid pairs only):

```
ou_half_life_min_bars  (config, default [TODO])
ou_half_life_max_bars  (config, default [TODO])
```

A pair-fold is kept iff `ou_half_life_min_bars <= hl_bars <=
ou_half_life_max_bars`, **where hl_bars is the half-life from the
TRAIN-window OU fit, not the daily inherited value.** Defaults to
remain open (Q-band).

The intraday half-life is also a sanity check on the spread itself: a
half-life > train-window-length implies the train window doesn't
exhibit mean reversion at all, and the OU MLE is fitting noise — drop
the pair-fold.

------------------------------------------------------------------------

## 5. Final function signatures and config surface (locked 2026-06-10)

Decisions in the Changelog above are reflected here. Bars and minutes
are connected by the active bar frequency: `bars_per_minute = 1 / freq_min`
where `freq_min ∈ {1, 5, 15}`.

### 5.1 New module: `src/apt/stats/ou.py`

The OU fitter is asset-agnostic and pure (no config, no I/O); it lives
under `apt.stats` next to the existing cointegration / spread primitives.
The signal-layer wrapper that turns an `OUFit` + `OUThresholds` into an
`IntradaySignalSeries` lives in `apt.intraday.signals` (next to
`generate_signals_two_regime`).

```python
@dataclass(frozen=True)
class OUFit:
    """Per-pair OU parameters fit on the TRAIN window only.

    All time-bearing fields use the FIT FREQUENCY's "bar" as the
    discrete-time unit. Convert with `half_life_minutes =
    half_life_bars * freq_minutes`.

    Attributes
    ----------
    kappa : float           # mean-reversion speed (per bar)
    mu : float              # long-run mean (log-spread units)
    sigma : float           # diffusion coefficient (per sqrt(bar))
    sigma_eq : float        # equilibrium std = sigma / sqrt(2 kappa)
    half_life_bars : float
    half_life_minutes : float
    phi : float             # AR(1) coefficient = exp(-kappa)
    sigma_eps : float       # AR(1) innovation std
    n_obs : int             # train sample size after dropping NaN
    freq_minutes : int      # active bar frequency in minutes
    fit_ok : bool           # False if phi <= 0 (no mean reversion) or
                            # phi >= 1 (unit root) or n_obs too small
    reason : str            # human-readable rejection reason, '' if ok
    """

@dataclass(frozen=True)
class OUThresholds:
    """Bertram-derived optimal entry/exit thresholds in Z-OU units.

    Attributes
    ----------
    a_entry_z : float       # entry threshold IN Z-OU UNITS; enter
                            # short-spread at Z >= +a_entry_z, long at
                            # Z <= -a_entry_z. Exit at Z = 0.
    cost_log : float        # round-trip cost the threshold was
                            # optimized against (log-return units)
    expected_return_per_unit_time : float  # the Bertram objective at
                                           # (a_entry_z, exit-at-mean)
    note : str              # impl version + paper-equation pointer
    fit_ok : bool           # False if optimum is infeasible
                            # (objective non-positive at any a > 0)
    """

def fit_ou_params(
    spread: np.ndarray,
    *,
    freq_minutes: int = 1,
    min_obs: int = 60,
) -> OUFit:
    """Fit OU parameters by AR(1) MLE on a TRAIN-window spread.

    Equivalent regression on the log spread X_t (NaN dropped):
        X_{t+1} = c + phi * X_t + eps_t,   eps_t ~ N(0, sigma_eps^2)

    Mapping (dt = 1 bar at the active fit frequency):
        phi   = exp(-kappa)                 (so kappa = -ln(phi))
        mu    = c / (1 - phi)
        sigma = sigma_eps * sqrt(2 kappa / (1 - phi^2))
        half_life_bars = ln(2) / kappa
        sigma_eq = sigma / sqrt(2 kappa)
        half_life_minutes = half_life_bars * freq_minutes

    Returns OUFit with `fit_ok=False` and a `reason` if:
      * fewer than `min_obs` finite observations,
      * phi <= 0 (no mean reversion — anti-persistent),
      * phi >= 1 (random-walk regime).
    Orchestrator excludes any (fit_ok=False) pair-fold and counts the
    rejection in exclusion accounting.
    """

def bertram_threshold(
    fit: OUFit,
    *,
    cost_log_per_round_trip: float,
) -> OUThresholds:
    """Closed-form optimal entry threshold a* in Z-OU units.

    The Bertram (2010) objective is expected log-return per unit time
    net of a fixed round-trip cost c, under the trading rule:
        enter at X = mu +/- a (in spread units),
        exit at  X = mu (mean-touch),
        re-enter on band recross.

    Formulas: see module docstring with verbatim equation numbers from
    Bertram (2010), eq (2.17)-(2.22), kept in paper notation. Verified
    by the unit tests in tests/stats/test_ou.py and the Monte-Carlo
    first-passage validator in tests/stats/test_ou_mc.py (§8.3).

    Returns a* expressed in Z-OU units (a/sigma_eq), so the signal
    layer can use a single canonical band irrespective of pair scale.
    `fit_ok=False` if the optimum solve fails or the objective is
    non-positive at every a > 0 (infeasible at this cost).
    """

def resample_within_session(
    spread: np.ndarray,
    timestamps: pd.DatetimeIndex,
    session_id: np.ndarray,
    tradeable: np.ndarray,
    *,
    freq_minutes: int,
) -> ResampledSpread:
    """Aggregate the 1-minute panel to coarser within-session bars.

    - Left-labeled, completed-bar (the bar at 09:15 covers [09:15, 09:15+f))
    - No bar spans the overnight gap (resample groups by (session, bin))
    - close = last finite close in bin; tradeable = any(tradeable_bin)
    - session_id preserved
    - returns aligned arrays + a new (timestamps, session_id, tradeable, close)

    At freq_minutes=1 this is a no-op pass-through.
    """
```

### 5.1b New helper in `apt.intraday.signals`

```python
def generate_signals_ou(
    z_ou: np.ndarray,                  # (X - mu_OU) / sigma_eq, NaN-safe
    session_id: np.ndarray,
    tradeable: np.ndarray,
    *,
    regime: str,                       # 'A' | 'B'
    a_entry_z: float,                  # Bertram a* in Z-OU units
    stop_mode: str = "none",           # 'none' | 'hard'
    stop_k_sigma: float = 4.0,         # K, only if stop_mode='hard'
    max_holding: int,                  # bars; 3 x intraday HL clamped
) -> IntradaySignalSeries:
    """OU-engine drop-in replacement for generate_signals_two_regime.

    State machine at each completed bar (sees only z_ou[t]):
      flat -> short when  z_ou[t] >= +a_entry_z       (sells the spread)
      flat -> long  when  z_ou[t] <= -a_entry_z       (buys the spread)
      short -> flat when  z_ou[t] <= 0    -> 'mean_revert'
                    or    z_ou[t] >= +K   -> 'z_stop' (only if hard)
                    or    held >= max_h   -> 'time_stop'
      long  -> flat: mirror.

    Regime A: per-session, force-close last tradeable bar with
    'eod_squareoff'. Regime B: continuous, fold-boundary close
    handled by run_pair_fold ('fold_close').
    Re-entry: state-machine flips immediately at the next bar that
    triggers an entry condition once flat — no cool-off.

    Preserves v2 completed-bar / execution-timing / NaN-passthrough
    semantics exactly (see §8.2).
    """
```

### 5.2 Config surface — `config/default.yaml` additions (final)

```yaml
signal:
  # ... existing ad-hoc-band knobs (entry_z, exit_z, stop_z) unchanged ...
  # ... existing max_holding_cap_days, max_holding_half_life_multiplier unchanged ...
  engine: "rolling_z"               # "rolling_z" (default) | "ou"
  ou:
    freq_minutes: 1                 # 1 | 5 | 15 (one cell per choice)
    min_obs: 60                     # AR(1) fit floor
    # Half-life band in TRADING MINUTES, per regime.
    # 375 trading minutes per session, 9:15-15:30 IST.
    half_life_band:
      A_min_minutes: 30
      A_max_minutes: 120
      B_min_minutes: 120
      B_max_minutes: 1875           # = 5 sessions
    # Z-OU drift diagnostic flag threshold (|test-slice mean| > this)
    drift_flag_abs_mean: 0.5
  stop:
    mode: "none"                    # "none" | "hard"
    k_sigma: 4.0                    # |Z-OU| >= K triggers exit if mode='hard'
```

### 5.3 Orchestrator integration in `scripts/13_phase3_intraday.py`

Minimal, optional behind `settings.signal.engine == "ou"`:

```python
# inside _compute_pair_fold, right after train_mask / test_mask are built:
if settings.signal.engine == "ou":
    train_spread = spread_full[train_mask]
    train_spread_finite = train_spread[np.isfinite(train_spread)]
    ou_fit = fit_ou_params(train_spread_finite, bars_per_unit=1)

    # Half-life band gate (drops the pair-fold)
    if (not np.isfinite(ou_fit.half_life_bars)
        or ou_fit.half_life_bars < settings.signal.ou.half_life_min_bars
        or ou_fit.half_life_bars > settings.signal.ou.half_life_max_bars):
        return None

    # Bertram thresholds at the cost the spread sweep optimises against
    cb_anchor = CostBreakdown(total_spread_bps=3)   # see Q-anchor
    thresholds = bertram_thresholds(
        ou_fit,
        cost_log_per_round_trip=cb_anchor.cost_log_per_pair_round_trip,
    )

    for regime in ("A", "B"):
        sig = generate_signals_ou(
            spread=s_test, session_id=sids_test, tradeable=tradeable_test,
            ou_fit=ou_fit, ou_thresholds=thresholds, regime=regime,
            max_holding=max_holding_bars,
            catastrophic_stop_in_sigma_eq=settings.signal.ou.catastrophic_stop_in_sigma_eq,
        )
        res = run_pair_fold(...)
        out[regime] = res
```

`run_pair_fold` and `_net_pnl_for_cost` are unchanged. The
spread-sweep loop at lines 562-707 is unchanged. The result CSVs
gain one new file (`metrics_ou_thresholds.csv`) but the existing
`metrics_two_regime_v2.csv` is untouched.

### 5.4 Gross / net reporting

No change to the reporting structure — the existing
`_emit_phase3_v2_outputs` produces gross and net per (regime ×
spread_bps × variant) and is re-used. The new OU run emits a parallel
table for direct comparison; the user reads them side by side.

------------------------------------------------------------------------

## 6. Leakage section

**OU parameters and Bertram thresholds are fit on TRAIN and FROZEN for
TEST.** The audit trail:

- `fit_ou_params(spread)` is called with `spread_full[train_mask]`
  only — the call site is statically right after the train_mask is
  defined and before the test_mask is used in any signal-relevant
  expression. Test: a unit test asserts `fit_ou_params` is called with
  exactly `train_mask`-sliced data in `_compute_pair_fold` (pytest
  `caplog` on a debug log emitted from the script).

- `bertram_thresholds(ou_fit, cost_log_per_round_trip)` is pure (no
  side effects, no globals); the inputs are the train-fit `OUFit` and
  a cost scalar known at compile time. Test: a unit test asserts the
  function is deterministic and depends only on its arguments
  (property-based: equal inputs ⇒ equal outputs).

- `generate_signals_ou(...)` receives `OUFit` and `OUThresholds` as
  frozen dataclasses (immutable) and never modifies them. Test: an
  immutability test on the frozen dataclasses (already enforced by
  `@dataclass(frozen=True)`).

- The state machine itself is causal: at index `t` it sees `spread[t]`
  but compares to the frozen `μ`, `a_entry`, `b_exit` — all known
  before the test slice begins. **No rolling estimator updates inside
  the state machine**, so there is no possibility of test-data leaking
  back into the thresholds.

- **Mirroring the existing test `test_no_lookahead_truncation_invariance`
  in `tests/intraday/test_backtest.py`**: a new test will assert that
  truncating the future cannot change a past bar's signal under
  the OU engine — same invariance, same fixture pattern.

- **The half-life band gate** uses the train-window half-life only —
  the gate runs before the test slice is touched.

[TODO if any of the above turns out to need a different architecture
during implementation — flag immediately and revise this section.]

------------------------------------------------------------------------

## 7. Open questions (Step 3) — RESOLVED (see Decisions changelog + §8)

The questions below were the Step-3 surface, answered by the user on
2026-06-10. Kept verbatim for audit trail. The authoritative defaults
live in §5.2 and the supporting derivations in §8.



### Q-cost — the cost-per-round-trip to feed Bertram's optimisation

The OU/Bertram objective is **expected return per unit time net of a
fixed round-trip cost `c`**. The repo has two cost figures and they
disagree:

- `config/default.yaml:98` says `cost_bps_per_leg: 25.0` (Phase 2A
  daily, used by `apt.backtest.walkforward`). At 2 legs ⇒ 50 bps
  total per round-trip.
- `src/apt/intraday/costs.py` builds it from components: fixed = 4.5
  bps/leg/RT (STT 2.5 + brokerage 0.5 + exchange/reg 1.5), spread =
  sweep `{1, 3, 5, 8}` bps total. At 4.5 + 3 = 7.5 bps/leg/RT × 2
  legs ⇒ 15 bps total per round-trip at the 3-bps anchor.

  **Q-cost.1:** Confirm the OU optimisation should be anchored to a
  single point in the spread sweep (proposed 3 bps total spread ⇒ 15
  bps total round-trip in log units, `c_log = 0.0015`), with the
  remaining sweep levels used only for **reporting** (i.e. fit
  thresholds once at 3 bps, then re-sweep cost-deduction at exit for
  {1,3,5,8} bps as Phase 3 v2 does)?

  **Q-cost.2:** Or should Bertram thresholds be re-fit at EVERY
  spread-sweep level, producing one signal series per level? (More
  expensive — each level's thresholds → its own trade list — but
  honest if the user cares about the optimal at each cost level.)

  **Q-cost.3:** The repo's 25 bps/leg daily figure includes "slippage"
  that the intraday model breaks out as bid-ask spread. **For the OU
  intraday work, the canonical cost is the intraday `CostBreakdown`
  model (4.5 fixed + spread sweep), not the 25 bps daily figure** —
  confirm or override.

### Q-stop — what to do with the existing `±3.5σ` stop

Three options spelled out in §3.3:

- **Stop-drop**: remove the stop entirely. Theoretically consistent
  with Bertram. Recommended starting point.
- **Stop-overlay**: keep a hard catastrophic stop at K × σ_eq for
  some K. **Pick a K.** [TODO: K = ?]
- **Stop-ablate**: implement both behind a config flag, run side by
  side, compare. Highest information yield; longest wall-clock.

  **Q-stop:** Stop-drop, Stop-overlay (with K), or Stop-ablate (both)?

### Q-bar-freq — bar frequency to fit and trade at

The Phase 3 v2 pipeline operates at 1-minute resolution. Three
options:

- **Fix at 1-minute** (matches existing pipeline; nothing else
  changes).
- **Sweep {1, 5, 15} minute aggregation** at the spread level
  (re-aggregate `close` to `t/5/15`-minute bars, recompute spread, fit
  OU at each frequency). More compute; potentially reveals an optimal
  bar size where the OU model fits cleanest.
- **Tick / volume / dollar bars** would require a tick feed.
  **`/Data6/db/minute/` and the converted parquet are minute-only.
  No tick data is in the repo or this host AFAICT.** [TODO if I am
  wrong about tick availability — please confirm.]

  **Q-bar-freq:** Fix at 1-minute, or sweep {1, 5, 15}, or fix at a
  different value? (If tick data exists somewhere I'm not seeing —
  point me at the path.)

### Q-default — should OU become the default, or stay opt-in via config flag?

§5.2 proposes `signal.engine: "rolling_z" | "ou"` defaulting to
`rolling_z` (existing behavior). Alternative: flip the default to
`ou` once the first walk-forward pass is done and reviewed.

  **Q-default:** ship OU as opt-in (recommended for round 1) or
  default once tested?

### Q-Z — z-score basis under OU

§2.4 lists three options for the z definition the OU thresholds
operate on:

- **Z-OU** (proposed default): `(spread − μ_OU) / σ_eq` with both
  TRAIN-frozen. Model-consistent.
- **Z-rolling**: keep the existing rolling z definition, scale
  Bertram thresholds back to z-units via σ_eq.
- **Z-hybrid**: `(spread − rollmean) / σ_eq`. Drifty mean, frozen
  scale.

  **Q-Z:** Z-OU, Z-rolling, or Z-hybrid as the production default?

### Q-band — half-life-band filter values

§4 introduces an intraday-specific half-life band filter, in BARS.
The Phase 3 v2 sample shows daily-fit half-lives of 11.8-30 days
(`fold_pairs.csv`); intraday half-lives will likely differ by orders
of magnitude. I do not have an a-priori range.

  **Q-band.1:** Should the default be permissive (e.g. `[30, 5*375] =
  [30 min, 5 sessions]`) and let the data filter naturally, or
  restrictive (e.g. `[1*375, 3*375] = [1 session, 3 sessions]`) to
  match the existing rolling-window clamp?

  **Q-band.2:** Should pairs whose **intraday OU half-life disagrees
  with their daily-inherited half-life by more than a factor F** be
  flagged or dropped? (E.g. if daily HL says 20 days but intraday HL
  says 0.5 sessions, the pair has totally different mean-reversion
  dynamics intraday — possibly correct, possibly noise.)

### Q-deps — dependency policy

`statsmodels` is already a dependency and supplies all we need for
AR(1) MLE. Bertram closed-forms are elementary (use `scipy.special`
for the relevant special functions; `scipy` is also already a
transitive dependency via `statsmodels`).

I will implement from scratch with **no new dependency**, unless you
prefer a vetted library port (e.g. `arch` for OU, or pulling Quant
Finance Stack Exchange canonical implementations). **Recommended:
implement from scratch; the formulas are short and the unit tests
in §2.3 are the defense.**

  **Q-deps:** Implement from scratch (recommended) or add a library?
  If a library — which?

### Q-units — time unit convention to freeze

OU half-life is in the same unit as `1/κ`. We can express in bars
(natural, fits the engine which works on bars) or in sessions / days
(easier for cross-checks against the existing daily `half_life_ar1`
output).

  **Q-units:** Keep `half_life_bars` as the canonical (internal) unit
  and report `half_life_minutes / half_life_sessions` for humans, or
  default canonical to minutes?

### Q-reference — primary reference for the closed-form

Proposed Bertram (2010), backup Leung-Li (2015). If the user has a
preferred reference or a known-good implementation to validate
against, point me at it; otherwise I will use Bertram as primary and
the unit tests in §2.3 as the cross-check.

  **Q-reference:** OK with Bertram (2010) as primary, Leung-Li (2015)
  as backup cross-check? Or substitute?

------------------------------------------------------------------------

**End of §7 (resolved).**

------------------------------------------------------------------------

## 8. Implementation contract (2026-06-10)

This section locks the contract beyond §5 signatures.

### 8.1 Cost-unit derivation with worked numeric example

The Phase 3 v2 deduction at `src/apt/intraday/backtest.py:165-168` is
exactly:

```
gross_log_pnl = direction * (spread[exit] - spread[entry])
net_log_pnl   = gross_log_pnl - cost_log_per_pair_round_trip
```

`spread` is the log-spread `log(p_y) − β·log(p_x) − α`, dimensionless
(log-return units of a unit-notional long-y / short-(β units of x)
portfolio). `cost_log_per_pair_round_trip` is computed in
`src/apt/intraday/costs.py:CostBreakdown` as:

```
cost_bps_per_leg          = FIXED_PER_LEG_RT + total_spread_bps
                          = 4.5 + total_spread_bps          (bps / leg / round-trip)
cost_bps_per_pair_rt      = 2 * cost_bps_per_leg            (n_legs = 2, equal-notional)
cost_log_per_pair_rt      = cost_bps_per_pair_rt / 10_000   (log-return units)
```

**Worked example at the 3-bps anchor:**

```
total_spread_bps          = 3
cost_bps_per_leg          = 4.5 + 3 = 7.5
cost_bps_per_pair_rt      = 15
cost_log_per_pair_rt      = 15 / 10_000 = 0.0015
```

Sweep:

| total_spread_bps | cost_bps_per_leg | cost_bps_per_pair_rt | cost_log    |
|------------------|------------------|----------------------|-------------|
| 1                | 5.5              | 11                   | 0.00110     |
| 3                | 7.5              | 15                   | 0.00150     |
| 5                | 9.5              | 19                   | 0.00190     |
| 8                | 12.5             | 25                   | 0.00250     |

**β-dependence trap (addendum #2 decision).** The current v2
accounting is equal-notional in both legs (n_legs = 2, no β scaling).
A theoretically tighter convention would scale the second-leg cost by
β, giving `(1 + β)` legs of cost rather than 2 — making `c` pair-and-
fold dependent. We inherit v2's convention verbatim for both `rolling_z`
and `ou` arms to keep the comparison clean, and instrument the
distribution of `β` (and `(1+β)/2`) in the diagnostic CSV. If median
`|(1+β)/2 − 1| > 0.10` across pair-folds, a `[TODO]` in the report
flags cost-accounting as a follow-up work item — we do not change
plumbing this round.

**Zero-move round-trip test (Stage-2 mandatory).** A simulated trade
where `spread[exit] == spread[entry]` must realize `net_log_pnl ==
−cost_log_per_pair_round_trip` (within float64 epsilon). The test
constructs a synthetic 4-bar series with the entry signal forced on
bar 1 and an exit forced on bar 3 with no spread change, runs
`run_pair_fold`, and asserts the cached `gross_log_ret[exit_idx] -
net_log_ret[exit_idx] == cost_log`. This pins the deduction plumbing,
not the notional convention.

### 8.2 v2 execution semantics statement

The OU engine MUST preserve these v2 behaviours exactly:

- **Completed-bar evaluation.** Signals are evaluated on the close of
  bar `t` and the resulting position is held from the close of bar
  `t` to the close of bar `t+1`. No intrabar evaluation.

- **Execution timing.** The state transition fires at bar `t`'s close;
  the implied P&L flows on bar `t+1`'s close (`gross_log_pnl[t+1] =
  position[t] · (spread[t+1] − spread[t])`). This is exactly how
  `run_pair_fold` in `apt.intraday.backtest` already accounts.

- **NaN-pass-through.** NaN inputs (unwarmed, non-tradeable, gap) carry
  state forward without incrementing `days_in_trade` — matches
  `apt.signals.spread.generate_signals` lines 188-196.

- **Tradeable mask.** Non-tradeable bars block entry. Regime A
  force-close lands on the LAST tradeable bar of the session (not the
  scheduled 15:29). Regime B never force-closes on session boundaries
  but DOES close on fold boundary via `run_pair_fold`.

- **Re-entry policy.** Once flat, the next bar that satisfies an entry
  condition fires immediately — no cool-off.

- **TOD-normalisation NOT in path.** The production v2 z is the FLAT
  `intraday_rolling_zscore`. `fit_tod_vol_profile` is computed but
  used only by the diagnostic `tod_adjusted_zscore` branch
  (`scripts/13_phase3_intraday.py:319-320`), not consumed by trading.
  Consequence (addendum #1): the OU fit lives in the SAME train-frozen
  raw-log-spread coordinates the production z uses. No TOD transform
  in the OU chain; no inconsistency to resolve.

### 8.3 Monte-Carlo first-passage validator (decisive test)

`tests/stats/test_ou_mc.py` runs three (κ, μ, σ, c) configurations.
For each:

1. Solve `bertram_threshold(fit, cost_log_per_round_trip=c)` → `a*`.
2. Compute the analytic objective `E[return per unit time] − c · cycle_rate`
   at `a*` per Bertram (2010) eq (2.17–2.22).
3. Simulate `K = 200` independent OU paths of length `T = 4 ·
   half_life_bars · 1000` bars using the exact discretisation
   (eq (2)). Seed each path with `seed = base + k`.
4. Run the LITERAL trading rule on each path: enter short at
   `Z = +a*`, long at `Z = −a*`, exit at `Z = 0`, deduct cost on
   exit. Compute realized cumulative log P&L per path; divide by
   total elapsed bars; take mean across paths.
5. Require `|MC_mean - analytic| < 3 · MC_se` where
   `MC_se = std_across_paths / sqrt(K)`. Three configs must all pass.

Why this is decisive: it catches BOTH transcription errors in the
closed form AND cycle-definition mismatches (e.g. confusing
"half-cycle" with "full-cycle" objective scalings) regardless of which
paper's notation the impl ends up using. Failure on any of the three
configs blocks merge.

### 8.4 Corporate-action caveat (addendum #8)

The v2 intraday loader reads `data/interim/minute_raw/symbol=<S>/`
without back-adjustment. The Phase 3 v2 report's Step 1b probe showed
median minute-close / daily-adjusted-close = 1.000 with ~7–10%
divergence on a handful of corporate-action days. The v2 backtest
absorbs this as discontinuities at the un-adjusted bar — gross P&L on
the action day is wrong; subsequent days are wrong by a constant
log-shift in the spread until the daily β / α are re-fit at the next
fold boundary.

The OU work this round does NOT build new adjustment machinery. If any
pair-fold's TRAIN window straddles a corp-action day for either leg,
the OU fit picks up an artificial level shift that biases `μ_OU`. We
list affected pair-folds in the report as a data caveat (cross-
reference daily `corporate_actions.parquet` against pair y/x and the
fold's TRAIN window). If we cannot identify affected pair-folds
cheaply, the caveat is recorded as `[TODO]` in the report.

### 8.5 Coarse-bar rolling_z baseline (addendum #5)

Per the addendum, add `rolling_z` runs at bars `{5, 15}` × regimes
`{A, B}` to isolate the **aggregation** effect from the **engine**
effect.

- Window per pair: existing `clip(half_life_days × 375, [375, 1875])`
  but in **minute-equivalent units**. Convert to bars at the active
  frequency: `window_bars = max(round(window_min / freq_min), 1)`.
- `max_holding`: convert from minute-equivalent to bars likewise.
- Thresholds unchanged at `(2.0, 0.5, 3.5)`.
- Cost levels: re-stamped via existing `_net_pnl_for_cost`.

Total backtest cells: 24 (OU full grid) + 6 (OU stop-ablation at
3-bps) + 4 (rolling_z coarse-bar) = **34 cells**. Wall-clock
projected from the smoke run (§8.6). If projected > 8 hours, fall
back to the reduced grid: bars `{5, 15}` × cost `{3, 8}` × regimes
`{A, B}` × stop `{none}` = 8 cells, and say so in the report.

### 8.6 Smoke run protocol

Before launching the full grid, run **one cell**: bars = 5,
cost = 3 bps, Regime A, stop = none. Validate:

- The OU dispatch fires (`signal.engine == "ou"`).
- The trade list is non-empty for at least one pair-fold.
- Output schema matches `metrics_two_regime_v2.csv` shape with
  added columns: `engine`, `freq_min`, `stop_mode`, `n_excluded_no_reversion`,
  `n_excluded_hl_band`, `n_excluded_other`, `hl_minutes_p50`,
  `hl_minutes_p10_p90`, `hl_ratio_p50`, `z_drift_mean_abs_p50`,
  `z_drift_flagged_count`.
- Per-pair-fold diagnostics emitted to
  `reports/phase3/ou_pair_fold_diag.csv`.

Then time the cell end-to-end and extrapolate to the full grid.

### 8.7 Updated leakage statement for Z-OU

- `μ_OU`, `σ_eq`, `a*` are TRAIN-only and bit-identical when test
  rows are replaced by random noise (test §8.8).
- `Z-OU[t] = (X[t] − μ_OU) / σ_eq` uses only `X[t]` and TRAIN-frozen
  scalars — strictly causal, no rolling window inside the state
  machine.
- Bar resampling at coarser frequencies is left-labeled and
  within-session; no future bar contributes to the current
  resampled close.
- Default-config regression (rolling_z) reproduces v2 byte-for-byte
  on a fixed fold (exact, not numerical — the dispatch is pure
  plumbing; existing arithmetic unchanged).

### 8.8 Test plan summary (mapped to addendum requirements)

| # | Test | File | Addendum item |
|---|------|------|---------------|
| a | OU parameter recovery (AR(1) MLE round-trip) | `tests/stats/test_ou.py` | (5a) |
| b | MC first-passage validator (3 configs) | `tests/stats/test_ou_mc.py` | (5b) decisive |
| c | a* monotonic non-decreasing in c | `tests/stats/test_ou.py` | (5c) |
| d | Zero-move round-trip realizes −c | `tests/intraday/test_backtest_cost_pin.py` | (5d) |
| e | rolling_z default reproduces v2 (byte-exact) | `tests/intraday/test_engine_dispatch.py` | (5e) |
| f | OU fit bit-identical when test slice randomised | `tests/stats/test_ou_leakage.py` | (5f) |
| g | Half-life invariance under freq change | `tests/stats/test_ou.py` | sanity |
| h | Resampling preserves session boundaries | `tests/intraday/test_resample.py` | (7) |

------------------------------------------------------------------------

**End of design doc.** Implementation begins now. Branch:
`feature/ou-optimal-thresholds`. Default config: `signal.engine ==
"rolling_z"` so v2 paths are byte-for-byte untouched.
