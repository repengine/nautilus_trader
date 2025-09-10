from __future__ import annotations

import time

from ml.common.throttler import Throttler


def test_throttler_basic_bucket() -> None:
    t = Throttler(rate_per_sec=10.0, burst=2)
    now = time.time_ns()
    # Burst allows two immediate publishes
    assert t.should_publish("k", now) is True
    assert t.should_publish("k", now) is True
    # Third should be throttled until refill
    assert t.should_publish("k", now) is False
    # Advance time by 0.2s -> 2 tokens
    later = now + int(0.2 * 1_000_000_000)
    assert t.should_publish("k", later) is True
    assert t.should_publish("k", later) is True
    # Burst exhausted again
    assert t.should_publish("k", later) is False
