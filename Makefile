.PHONY: install test lint format clean pipeline

PYTHON := .venv/bin/python
UV     := uv

install:
	$(UV) sync --extra dev

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

format:
	$(PYTHON) -m ruff format src/ tests/ scripts/
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache htmlcov .coverage

pipeline:
	@echo "=== 01 Ingest daily ==="
	$(PYTHON) scripts/01_ingest_daily.py
	@echo "=== 02 Ingest minute ==="
	$(PYTHON) scripts/02_ingest_minute.py &
	@echo "=== 03 Parse sectors ==="
	$(PYTHON) scripts/03_parse_sectors.py
	@echo "=== 04 Corporate actions ==="
	$(PYTHON) scripts/04_corporate_actions.py
	@echo "=== 05 Clean and align ==="
	$(PYTHON) scripts/05_clean_and_align.py
	@echo "=== 06 Universe plots ==="
	$(PYTHON) scripts/06_universe_plots.py
	@echo "=== 07 Screen correlation ==="
	$(PYTHON) scripts/07_screen_correlation.py
	@echo "=== 08 Test cointegration ==="
	$(PYTHON) scripts/08_test_cointegration.py
	@echo "=== 09 Compute spreads ==="
	$(PYTHON) scripts/09_compute_spreads.py
	@echo "=== 10 Pair plots ==="
	$(PYTHON) scripts/10_pair_plots.py
	@echo "=== Pipeline complete ==="
