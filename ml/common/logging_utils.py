"""
Logging utilities for best-effort, non-blocking patterns.

These helpers keep logging off hot paths and avoid raising from logging failures.
"""
from __future__ import annotations

from typing import Literal


_LogLevel = Literal["debug", "info", "warning", "error", "critical"]


def log_best_effort(
    logger: object,
    level: _LogLevel,
    msg: str,
    *args: object,
    **kwargs: object,
) -> None:
    """
    Log a message inside an exception handler without affecting control flow.

    - Avoid allocations in tight loops; prefer this in cold paths or at disabled levels.
    - Always safe: swallows secondary logging exceptions and returns None.
    - Supports both stdlib and structlog-interop loggers.
    """
    try:
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(msg, *args, **kwargs)
    except Exception:
        return None
