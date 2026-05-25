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
    # Rule 7 — contiguity filter (added after the ADV filter, before the
    # 756-day floor). Gaps > this many calendar days between consecutive
    # rows split a symbol into segments; only the longest is kept.
    contiguity_max_gap_days: int = 10
    # Tiebreaker preference: segments overlapping >= this date win over
    # segments that don't, regardless of length.
    contiguity_prefer_overlap_after: date = date(2015, 1, 1)

    model_config = SettingsConfigDict(env_prefix="APT_CLEANING_")


class ScreeningConfig(BaseSettings):
    correlation_threshold: float = 0.50
    n_corr_days: int = 504
    min_overlap_days: int = 504

    model_config = SettingsConfigDict(env_prefix="APT_SCREENING_")


class CointegrationConfig(BaseSettings):
    significance_level: float = 0.05
    max_pvalue: float = 0.05
    fdr_alpha: float = 0.05
    n_train_days: int = 1008
    min_train_days: int = 756
    half_life_min_days: int = 5
    half_life_max_days: int = 60
    hurst_max: float = 0.5
    hurst_max_lag: int = 100

    model_config = SettingsConfigDict(env_prefix="APT_COINTEGRATION_")


class SpreadConfig(BaseSettings):
    rolling_window: int = 60

    model_config = SettingsConfigDict(env_prefix="APT_SPREAD_")


class SignalConfig(BaseSettings):
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding_cap_days: int = 60
    max_holding_half_life_multiplier: float = 3.0

    model_config = SettingsConfigDict(env_prefix="APT_SIGNAL_")


class BacktestConfig(BaseSettings):
    test_days_per_fold: int = 252
    step_days: int = 252
    cost_bps_per_leg: float = 25.0

    model_config = SettingsConfigDict(env_prefix="APT_BACKTEST_")


class RiskManagedConfig(BaseSettings):
    risk_frac: float = 0.01
    per_pair_cap: float = 0.12
    cluster_cap: float = 0.05
    total_cap: float = 0.10
    gross_cap: float = 2.0
    kill_K: int = 4
    kill_cap: float = 0.04
    kill_check_interval_days: int = 21
    kill_relationship_window_days: int = 252
    kill_relationship_adf_alpha: float = 0.05
    kill_relationship_halflife_max_days: int = 60
    kill_relationship_vol_ratio_max: float = 2.0

    model_config = SettingsConfigDict(env_prefix="APT_RISK_MANAGED_")


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
    signal: SignalConfig = SignalConfig()
    backtest: BacktestConfig = BacktestConfig()
    risk_managed: RiskManagedConfig = RiskManagedConfig()
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
            signal=SignalConfig(**raw.get("signal", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            risk_managed=RiskManagedConfig(**raw.get("risk_managed", {})),
            parallel=ParallelConfig(**raw.get("parallel", {})),
        )


# Module-level singleton — import this everywhere
settings = Settings.from_yaml()
