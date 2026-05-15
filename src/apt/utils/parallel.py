"""Joblib parallel wrapper with tqdm progress bar."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from joblib import Parallel, delayed
from tqdm import tqdm

from apt.config import settings

T = TypeVar("T")


def parallel_map(
    func: Callable[..., T],
    items: Iterable[Any],
    *,
    n_jobs: int | None = None,
    desc: str = "",
    prefer: str = "processes",
    **joblib_kwargs: Any,
) -> list[T]:
    """Run func over items in parallel with a tqdm progress bar.

    Args:
        func: callable to apply to each item.
        items: iterable of arguments. Each element is passed as the first positional arg.
        n_jobs: override config n_jobs.
        desc: tqdm description string.
        prefer: passed to joblib (threads or processes).
    """
    jobs = n_jobs if n_jobs is not None else settings.parallel.n_jobs
    item_list = list(items)
    results = Parallel(n_jobs=jobs, prefer=prefer, **joblib_kwargs)(
        delayed(func)(item) for item in tqdm(item_list, desc=desc, ncols=90)
    )
    return results  # type: ignore[return-value]
