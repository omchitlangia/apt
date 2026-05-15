# Algorithmic Pairs Trading (APT)

Statistical arbitrage on the NSE Nifty-500 universe using classical cointegration methods and, in later phases, reinforcement learning.

## Project overview

The pipeline ingests raw daily and minute OHLCV data for ~490 NSE symbols, applies corporate-action adjustments, screens for cointegrated pairs using Engle-Granger, computes spread and z-score signals, and (in Phase 2) trains an RL agent to trade the spread.

## File layout

```
apt/
├── config/             # YAML configuration (default.yaml)
├── data/
│   ├── interim/        # raw-as-parquet outputs
│   ├── processed/      # adjusted, cleaned data
│   └── pairs/          # pair-level spreads and signals
├── plots/              # all generated PNGs
│   └── phase1/
│       ├── universe/   # universe characterisation plots
│       └── pairs/      # per-pair diagnostic plots
├── reports/            # human-readable CSV summaries
├── scripts/            # numbered pipeline scripts 01–10
├── src/apt/            # package source
│   ├── config.py       # pydantic-settings config loader
│   ├── data/           # ingest, corporate actions
│   ├── stats/          # cointegration, spread, half-life, Hurst
│   ├── signals/        # threshold / signal generation
│   ├── plots/          # plot functions
│   └── utils/          # logging, paths, parallel
├── tests/              # pytest suite
├── pyproject.toml
├── Makefile
└── uv.lock
```

## Prerequisites

- Python 3.11 (managed via `uv`)
- [uv](https://docs.astral.sh/uv/) ≥ 0.4

## Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
make install
```

## Running the pipeline

Run all 10 scripts in order:

```bash
make pipeline
```

Or run individual steps:

```bash
.venv/bin/python scripts/01_ingest_daily.py
.venv/bin/python scripts/03_parse_sectors.py
# ... etc
```

## Running tests

```bash
# Fast unit tests only
make test

# With coverage (must hit 80 % on stats + signals)
.venv/bin/pytest --cov=src/apt/stats --cov=src/apt/signals --cov-fail-under=80

# Exclude slow integration tests
.venv/bin/pytest -m "not slow"
```

## Linting and formatting

```bash
make lint
make format
```

## Configuration

Edit `config/default.yaml` to change universe filters, correlation thresholds, cointegration settings, and rolling windows. All parameters can also be overridden via environment variables prefixed with `APT_`, e.g.:

```bash
APT_SCREENING__CORRELATION_THRESHOLD=0.90 make pipeline
```

## MLflow UI

MLflow tracking is used from Phase 2 onwards. To view the UI from a remote server via SSH port forwarding:

```bash
# On your local machine:
ssh -L 5000:localhost:5000 <user>@<server>

# On the server:
.venv/bin/mlflow ui --host 0.0.0.0 --port 5000
```

Then open http://localhost:5000 in your browser.
