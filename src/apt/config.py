"""Central configuration loader backed by config/default.yaml with env-var overrides."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_YAML = _PROJECT_ROOT / "config" / "default.yaml"


class PathsConfig(BaseSettings):
    raw_data_dir: Path = Path("/Data6/db")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    pairs_dir: Path = Path("data/pairs")
    plots_dir: Path = Path("plots")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")

    @field_validator("*", mode="before")
    @classmethod
    def resolve(cls, v: Any) -> Path:
        p = Path(v)
        if not p.is_absolute():
            return _PROJECT_ROOT / p
        return p

    model_config = SettingsConfigDict(env_prefix="APT_PATHS_")


class UniverseConfig(BaseSettings):
    min_history_days: int = 756

    model_config = SettingsConfigDict(env_prefix="APT_UNIVERSE_")


class LiquidityConfig(BaseSettings):
    min_adv_inr: float = 10_000_000.0
    rolling_window: int = 60
    rolling_min_periods: int = 20

    model_config = SettingsConfigDict(env_prefix="APT_LIQUIDITY_")


class CleaningConfig(BaseSettings):
    phantom_jump_threshold: float = 0.65
    validation_jump_threshold: float = 0.40
    validation_start: date = date(2011, 1, 1)

    model_config = SettingsConfigDict(env_prefix="APT_CLEANING_")


class ScreeningConfig(BaseSettings):
    correlation_threshold: float = 0.85
    n_corr_days: int = 504
    min_overlap_days: int = 504

    model_config = SettingsConfigDict(env_prefix="APT_SCREENING_")


class CointegrationConfig(BaseSettings):
    significance_level: float = 0.05
    max_pvalue: float = 0.05

    model_config = SettingsConfigDict(env_prefix="APT_COINTEGRATION_")


class SpreadConfig(BaseSettings):
    rolling_window: int = 252

    model_config = SettingsConfigDict(env_prefix="APT_SPREAD_")


class ParallelConfig(BaseSettings):
    n_jobs: int = -1

    model_config = SettingsConfigDict(env_prefix="APT_PARALLEL_")


class Settings(BaseSettings):
    """Top-level settings; sub-sections populated from YAML then overrideable by env vars."""

    paths: PathsConfig = PathsConfig()
    universe: UniverseConfig = UniverseConfig()
    liquidity: LiquidityConfig = LiquidityConfig()
    cleaning: CleaningConfig = CleaningConfig()
    screening: ScreeningConfig = ScreeningConfig()
    cointegration: CointegrationConfig = CointegrationConfig()
    spread: SpreadConfig = SpreadConfig()
    parallel: ParallelConfig = ParallelConfig()

    model_config = SettingsConfigDict(env_prefix="APT_")

    @classmethod
    def from_yaml(cls, yaml_path: Path = _DEFAULT_YAML) -> Settings:
        raw = _load_yaml(yaml_path)
        return cls(
            paths=PathsConfig(**raw.get("paths", {})),
            universe=UniverseConfig(**raw.get("universe", {})),
            liquidity=LiquidityConfig(**raw.get("liquidity", {})),
            cleaning=CleaningConfig(**raw.get("cleaning", {})),
            screening=ScreeningConfig(**raw.get("screening", {})),
            cointegration=CointegrationConfig(**raw.get("cointegration", {})),
            spread=SpreadConfig(**raw.get("spread", {})),
            parallel=ParallelConfig(**raw.get("parallel", {})),
        )


# Module-level singleton — import this everywhere
settings = Settings.from_yaml()
