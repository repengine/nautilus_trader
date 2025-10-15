from __future__ import annotations

from datetime import datetime, timezone, timedelta, UTC

import pytest

from ml._imports import HAS_NAUTILUS_CORE
from ml._imports import NAUTILUS_CORE_IMPORT_ERROR

if not HAS_NAUTILUS_CORE:  # pragma: no cover - depends on native extensions
    pytest.skip(
        f"Nautilus Trader core extensions unavailable: {NAUTILUS_CORE_IMPORT_ERROR}",
        allow_module_level=True,
    )

from ml.deployment.scheduling_utils import (
    DailyTime,
    compute_next_utc_run,
    parse_daily_spec,
)


def test_parse_hhmm() -> None:
    dt = parse_daily_spec("17:05")
    assert isinstance(dt, DailyTime)
    assert dt.hour == 17 and dt.minute == 5


def test_parse_cron_like() -> None:
    dt = parse_daily_spec("0 6 * * *")
    assert dt.hour == 6 and dt.minute == 0


@pytest.mark.parametrize(
    "now_str, spec, expected_hour, expected_minute, delta_days",
    [
        ("2024-01-01T12:00:00Z", "13:30", 13, 30, 0),  # later same day
        ("2024-01-01T23:59:00Z", "00:00", 0, 0, 1),    # next day rollover
        ("2024-01-01T16:59:59Z", "0 17 * * *", 17, 0, 0),
        ("2024-01-01T17:00:01Z", "0 17 * * *", 17, 0, 1),
    ],
)
def test_compute_next_utc_run(
    now_str: str,
    spec: str,
    expected_hour: int,
    expected_minute: int,
    delta_days: int,
) -> None:
    now = datetime.fromisoformat(now_str.replace("Z", "+00:00"))
    daily = parse_daily_spec(spec)
    nxt = compute_next_utc_run(now, daily)
    assert nxt.tzinfo == UTC
    assert nxt.hour == expected_hour and nxt.minute == expected_minute
    # Date progression check
    expected_date = (now.astimezone(UTC) + timedelta(days=delta_days)).date()
    assert nxt.date() == expected_date or (delta_days == 1 and nxt.date() == expected_date)


def test_invalid_spec_raises() -> None:
    with pytest.raises(ValueError):
        parse_daily_spec("bad spec")
