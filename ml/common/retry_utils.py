"""
Typed retry/backoff utilities for cold-path operations (API calls, I/O).

These helpers are allocation-light and configurable. Do not use in hot paths.

"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def _compute_delay(
    attempt: int,
    *,
    initial_delay: float,
    multiplier: float,
    max_delay: float,
    jitter: float,
) -> float:
    base: float = float(initial_delay) * (float(multiplier) ** float(max(0, attempt)))
    base = float(min(float(max_delay), max(0.0, base)))
    if jitter > 0.0:
        # Apply +/- jitter fraction
        jitter_frac = random.uniform(-jitter, jitter)
        base = max(0.0, base * (1.0 + jitter_frac))
    return float(base)


def retry_with_backoff(
    call: Callable[[], T],
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 0.0,
    sleep_fn: Callable[[float], None] | None = None,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_exception: Callable[[int, BaseException], None] | None = None,
) -> T:
    """
    Execute `call` with retry and exponential backoff.

    Parameters
    ----------
    call : Callable[[], T]
        Zero-arg callable to execute.
    max_attempts : int
        Total attempts including the first call.
    initial_delay : float
        Base delay (seconds) before the first retry.
    multiplier : float
        Exponential multiplier between retries.
    max_delay : float
        Maximum delay cap (seconds).
    jitter : float
        Fractional jitter (0..1) applied to delay (+/-).
    sleep_fn : Callable[[float], None] | None
        Optional sleep function; defaults to time.sleep.
    retry_on : tuple[type[BaseException], ...]
        Exception types to retry on.
    on_exception : Callable[[int, BaseException], None] | None
        Optional callback invoked on exceptions with (attempt_index, exception).

    Returns
    -------
    T
        The result of `call()` on success.

    Raises
    ------
    BaseException
        Re-raises the last exception if all attempts fail.

    """
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            return call()
        except retry_on as exc:  # pragma: no cover - exercised via callers
            if on_exception is not None:
                try:
                    on_exception(attempt, exc)
                except Exception:
                    # Observability must not affect control flow
                    pass
            if attempt >= attempts - 1:
                raise
            delay = _compute_delay(
                attempt,
                initial_delay=initial_delay,
                multiplier=multiplier,
                max_delay=max_delay,
                jitter=jitter,
            )
            (sleep_fn or time.sleep)(delay)
    # Unreachable
    raise RuntimeError("retry_with_backoff: exhausted attempts without result")


__all__ = ["retry_with_backoff"]
