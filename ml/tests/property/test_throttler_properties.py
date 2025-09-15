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
