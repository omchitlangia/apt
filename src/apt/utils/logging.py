"""Loguru-based logging setup. Call setup_logging() once at script entry."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_file: Path | None = None, level: str = "INFO") -> None:
    """Configure loguru: stderr + optional rotating file sink."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            rotation="50 MB",
            retention=5,
            compression="gz",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
        )
