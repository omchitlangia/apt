"""Smoke test: import package and confirm paths resolve correctly."""

from pathlib import Path

from apt.config import settings
from apt.utils.paths import ensure_dirs, interim, pairs, plots, processed, reports


def test_settings_loads():
    """Config loads without error and has expected attributes."""
    assert hasattr(settings, "paths")
    assert hasattr(settings, "universe")
    assert hasattr(settings, "screening")
    assert hasattr(settings, "cointegration")
    assert hasattr(settings, "spread")
    assert hasattr(settings, "parallel")


def test_paths_are_absolute():
    """All configured paths resolve to absolute paths."""
    for attr in (
        "raw_data_dir",
        "interim_dir",
        "processed_dir",
        "pairs_dir",
        "plots_dir",
        "reports_dir",
        "logs_dir",
    ):
        p: Path = getattr(settings.paths, attr)
        assert p.is_absolute(), f"{attr} is not absolute: {p}"


def test_raw_data_dir_exists():
    """Raw data directory must already exist (read-only source)."""
    assert settings.paths.raw_data_dir.exists(), (
        f"raw_data_dir not found: {settings.paths.raw_data_dir}"
    )


def test_ensure_dirs_creates_directories(tmp_path):
    """ensure_dirs creates all output directories."""
    from apt.config import (
        CointegrationConfig,
        ParallelConfig,
        PathsConfig,
        ScreeningConfig,
        Settings,
        SpreadConfig,
        UniverseConfig,
    )

    cfg = Settings(
        paths=PathsConfig(
            raw_data_dir=tmp_path / "db",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
            pairs_dir=tmp_path / "pairs",
            plots_dir=tmp_path / "plots",
            reports_dir=tmp_path / "reports",
            logs_dir=tmp_path / "logs",
        ),
        universe=UniverseConfig(),
        screening=ScreeningConfig(),
        cointegration=CointegrationConfig(),
        spread=SpreadConfig(),
        parallel=ParallelConfig(),
    )
    ensure_dirs(cfg)
    assert (tmp_path / "interim").is_dir()
    assert (tmp_path / "processed").is_dir()
    assert (tmp_path / "pairs").is_dir()
    assert (tmp_path / "plots" / "phase1" / "universe").is_dir()
    assert (tmp_path / "plots" / "phase1" / "pairs").is_dir()
    assert (tmp_path / "reports").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_helper_functions_return_paths():
    """Path helper functions return Path objects under the configured directories."""
    p = interim("test.parquet")
    assert isinstance(p, Path)
    assert "interim" in str(p)

    p = processed("test.parquet")
    assert "processed" in str(p)

    p = pairs("test.parquet")
    assert "pairs" in str(p)

    p = reports("test.csv")
    assert "reports" in str(p)

    p = plots("phase1/universe", "test.png")
    assert "plots" in str(p)


def test_universe_defaults():
    assert settings.universe.min_history_days == 756


def test_screening_defaults():
    assert settings.screening.correlation_threshold == 0.50
    assert settings.screening.n_corr_days == 504
    assert settings.screening.min_overlap_days == 504


def test_cointegration_defaults():
    assert settings.cointegration.max_pvalue == 0.05
    assert settings.cointegration.fdr_alpha == 0.05
    assert settings.cointegration.n_train_days == 1008
    assert settings.cointegration.min_train_days == 756
    assert settings.cointegration.half_life_min_days == 5
    assert settings.cointegration.half_life_max_days == 60
    assert settings.cointegration.hurst_max == 0.5
    assert settings.cointegration.hurst_max_lag == 100


def test_spread_defaults():
    assert settings.spread.rolling_window == 252
