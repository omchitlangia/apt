# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.11 via uv)
uv sync --extra dev          # or: make install

# Run all tests
.venv/bin/pytest tests/ -q   # or: make test

# Run a single test file or test
.venv/bin/pytest tests/data/test_ingest.py -v
.venv/bin/pytest tests/data/test_ingest.py::test_zero_volume_flagged -v

# Run tests excluding slow ones (those hitting real xlsx or yfinance)
.venv/bin/pytest -m "not slow" -q

# Coverage gate (must pass before merging stats/signals work)
.venv/bin/pytest --cov=src/apt/stats --cov=src/apt/signals --cov-fail-under=80

# Lint and format (always run before committing)
make lint
make format

# Run the full 10-script pipeline in order
make pipeline

# Run a single pipeline script
.venv/bin/python scripts/01_ingest_daily.py
```

## Architecture

### Data flow

```
/Data6/db/*.csv  (read-only)
    │
    ▼  scripts/01_ingest_daily.py  →  apt.data.ingest
data/interim/daily_raw.parquet        (long-format, ~1.6M rows, 492 symbols, has `year` column)

/Data6/db/Merge_21May2021.xlsx
    │
    ▼  scripts/03_parse_sectors.py  →  apt.data.sectors
data/interim/sectors.parquet          (symbol, company_name, industry, isin, bse_industry)

    ▼  scripts/04_corporate_actions.py  →  apt.data.corporate_actions
data/interim/corporate_actions.parquet
data/processed/daily_adjusted.parquet  (back-adjusted OHLC)

    ▼  scripts/05_clean_and_align.py
data/processed/daily_clean.parquet     (universe-filtered, log returns added)

    ▼  scripts/07_screen_correlation.py  →  apt.stats.correlation
data/pairs/correlated_pairs.parquet

    ▼  scripts/08_test_cointegration.py  →  apt.stats.cointegration
data/pairs/cointegrated_pairs.parquet
reports/cointegrated_pairs_ranked.csv

    ▼  scripts/09-10  →  apt.stats.spread, apt.plots
data/pairs/top20_spreads.parquet
plots/phase1/pairs/*.png
```

### Package layout (`src/apt/`)

- **`config.py`** — pydantic-settings singleton loaded from `config/default.yaml`. Import `from apt.config import settings` everywhere. All relative paths in config are resolved relative to the project root at load time.
- **`utils/paths.py`** — convenience functions `interim()`, `processed()`, `pairs()`, `plots()`, `reports()` that prepend the configured directories. Call `ensure_dirs()` once at script entry.
- **`utils/logging.py`** — call `setup_logging(log_file=...)` once at the top of each script entry point. Never configure loguru elsewhere.
- **`utils/parallel.py`** — `parallel_map(func, items, prefer="threads"|"processes")` wraps joblib + tqdm. Respects `settings.parallel.n_jobs`.
- **`data/ingest.py`** — pure functions, no global state. `discover_daily_csvs()` excludes `sbin.csv` and `NSE_Instruments_23June2021.csv`. `parse_daily_csv()` returns `None` on error (never raises).
- **`data/sectors.py`** — parses `ind_nifty500list` sheet; optionally enriches with BSE sub-industry via ISIN join.
- **`stats/`** — pure, I/O-free functions. Each cointegration result is a typed dataclass/NamedTuple. No side effects.
- **`signals/`** — threshold logic only. Pure functions, fully unit-tested.
- **`plots/`** — all plot functions take a `Path` and write a PNG. Style configured centrally in `apt.plots.style`.

### Script conventions

- Scripts are numbered `01`–`10` and live in `scripts/`. Each is idempotent (re-running overwrites outputs).
- Every script calls `setup_logging()` and `ensure_dirs()` before doing anything.
- Scripts print a `=== NN_script_name complete ===` summary to stdout; detailed logging goes to `logs/`.
- Pre-commit hook blocks direct commits to `main` — use `--no-verify` only for initial bootstrap commits.

### Raw data facts

- Raw CSVs: `/Data6/db/` — **treat as read-only at all times**.
- Daily CSV format: `date,open,high,low,close,volume` where date is `YYYY-MM-DD HH:MM:SS+05:30`.
- `sbin.csv` (lowercase) is a duplicate of `SBIN.csv` and is excluded from ingest.
- `NSE_Instruments_23June2021.csv` is an instrument master, not price data — excluded.
- Sector xlsx: `/Data6/db/Merge_21May2021.xlsx`, sheet `ind_nifty500list`. Columns: Company Name, Industry, Symbol, Series, ISIN Code.
- Minute data lives in `/Data6/db/minute/` with filenames like `SYMBOL_minute-data.csv`.

### Configuration overrides

Any `config/default.yaml` value can be overridden via environment variable with prefix `APT_` and double-underscore nesting:

```bash
APT_SCREENING__CORRELATION_THRESHOLD=0.90 .venv/bin/python scripts/07_screen_correlation.py
APT_PARALLEL__N_JOBS=4 make pipeline
```

### Testing patterns

- Fixtures live in `tests/conftest.py` (synthetic data — no real files in fast tests).
- Tests that read `/Data6/db/` are marked `@pytest.mark.slow`.
- Loguru doesn't propagate to Python's `logging`, so use a loguru sink (not `caplog`) to capture log output in tests:
  ```python
  messages = []
  sid = logger.add(messages.append, format="{message}", level="WARNING")
  try:
      call_function_under_test()
  finally:
      logger.remove(sid)
  ```
