from __future__ import annotations

from datetime import UTC, datetime

from ml.observability.migrations import _month_bounds


def test_month_bounds_returns_start_end_month() -> None:
    dt = datetime(2025, 3, 15, tzinfo=UTC)
    start, end = _month_bounds(dt)
    assert start == datetime(2025, 3, 1, tzinfo=UTC)
    assert end == datetime(2025, 4, 1, tzinfo=UTC)
