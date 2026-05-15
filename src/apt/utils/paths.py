"""Path resolution and directory creation from config."""

from __future__ import annotations

from pathlib import Path

from apt.config import Settings
from apt.config import settings as _default_settings


def ensure_dirs(cfg: Settings | None = None) -> None:
    """Create all output directories if they don't exist."""
    s = cfg or _default_settings
    for attr in (
        "interim_dir",
        "processed_dir",
        "pairs_dir",
        "plots_dir",
        "reports_dir",
        "logs_dir",
    ):
        path: Path = getattr(s.paths, attr)
        path.mkdir(parents=True, exist_ok=True)

    # Phase sub-directories
    (s.paths.plots_dir / "phase1" / "universe").mkdir(parents=True, exist_ok=True)
    (s.paths.plots_dir / "phase1" / "pairs").mkdir(parents=True, exist_ok=True)


def interim(filename: str, cfg: Settings | None = None) -> Path:
    s = cfg or _default_settings
    return s.paths.interim_dir / filename


def processed(filename: str, cfg: Settings | None = None) -> Path:
    s = cfg or _default_settings
    return s.paths.processed_dir / filename


def pairs(filename: str, cfg: Settings | None = None) -> Path:
    s = cfg or _default_settings
    return s.paths.pairs_dir / filename


def plots(subdir: str, filename: str, cfg: Settings | None = None) -> Path:
    s = cfg or _default_settings
    return s.paths.plots_dir / subdir / filename


def reports(filename: str, cfg: Settings | None = None) -> Path:
    s = cfg or _default_settings
    return s.paths.reports_dir / filename
