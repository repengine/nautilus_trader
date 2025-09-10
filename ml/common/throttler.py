"""
Non-blocking token-bucket throttler for event publishing.

Provides per-key (topic) rate limiting to avoid flooding external buses. The
token bucket grants up to ``burst`` tokens and refills at ``rate_per_sec``.
``should_publish(key, now_ns)`` returns True if a token is available.

"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Final


_NS_PER_SEC: Final[float] = 1_000_000_000.0


@dataclass
class _Bucket:
    tokens: float
    last_refill_ns: int


class Throttler:
    """
    Simple per-key token-bucket throttler.
    """

    def __init__(self, *, rate_per_sec: float, burst: int) -> None:
        if rate_per_sec <= 0.0:
            raise ValueError("rate_per_sec must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")
        self._rate = float(rate_per_sec)
        self._burst = int(burst)
        self._buckets: MutableMapping[str, _Bucket] = {}

    def should_publish(self, key: str, now_ns: int) -> bool:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=float(self._burst), last_refill_ns=now_ns)
            self._buckets[key] = bucket

        # Refill tokens based on elapsed time
        elapsed_ns = max(0, now_ns - bucket.last_refill_ns)
        if elapsed_ns > 0:
            refill = (elapsed_ns / _NS_PER_SEC) * self._rate
            bucket.tokens = min(float(self._burst), bucket.tokens + refill)
            bucket.last_refill_ns = now_ns

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False


__all__: Final[list[str]] = ["Throttler"]
