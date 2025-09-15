from __future__ import annotations

import pytest

try:  # optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from typing import Final

from ml.common.throttler import Throttler


NS_PER_SEC: Final[int] = 1_000_000_000


def test_throttler_burst_limit_single_timestamp() -> None:
    t = Throttler(rate_per_sec=10.0, burst=3)
    now = 0
    # First 3 calls allowed (burst tokens), then denied
    assert t.should_publish("k", now) is True
    assert t.should_publish("k", now) is True
    assert t.should_publish("k", now) is True
    assert t.should_publish("k", now) is False


@given(rate=st.floats(min_value=1.0, max_value=1000.0))
def test_throttler_refill_at_rate_allows_steady_stream(rate: float) -> None:
    # With burst=1 and exact spacing 1/r seconds, all calls should be allowed
    t = Throttler(rate_per_sec=rate, burst=1)
    now = 0
    # Use a ceiling-like step to ensure at least one full token refills
    dt_ns = int(NS_PER_SEC / rate) + 1
    for _ in range(10):
        assert t.should_publish("k", now) is True
        now += dt_ns


@given(
    rate=st.floats(min_value=0.1, max_value=10_000.0),
    burst=st.integers(min_value=1, max_value=20),
    attempts=st.integers(min_value=1, max_value=100),
)
def test_throttler_burst_caps_single_tick(rate: float, burst: int, attempts: int) -> None:
    """
    At a single timestamp, allowed publishes cannot exceed burst tokens.
    """
    t = Throttler(rate_per_sec=rate, burst=burst)
    now = 123456789
    allowed = sum(1 for _ in range(attempts) if t.should_publish("k", now))
    assert allowed <= burst


@given(
    rate=st.floats(min_value=0.5, max_value=5_000.0),
    burst=st.integers(min_value=1, max_value=10),
    steps=st.integers(min_value=1, max_value=200),
)
def test_throttler_tokens_accumulate_over_time(rate: float, burst: int, steps: int) -> None:
    """
    Over total time T, tokens available are <= burst + floor(T * rate).

    Ensure allowed count does not exceed this upper bound.

    """
    t = Throttler(rate_per_sec=rate, burst=burst)
    now = 0
    # Choose dt to vary coverage; ensure some accumulation
    dt_ns = int(NS_PER_SEC / max(rate, 1e-9))
    allowed = 0
    for i in range(steps):
        if t.should_publish("key", now):
            allowed += 1
        now += dt_ns
    total_time_sec = (steps * dt_ns) / NS_PER_SEC
    theoretical_cap = burst + int(total_time_sec * rate) + 1  # +1 slack for integer rounding
    assert allowed <= theoretical_cap
