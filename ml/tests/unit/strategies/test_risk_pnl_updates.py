from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ml.strategies.risk import RiskManager

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def test_update_daily_pnl_resets_using_event_timestamp() -> None:
    rm = RiskManager()
    future_reset = datetime.now() + timedelta(hours=1)
    rm._daily_reset_time = future_reset
    rm._daily_pnl = 10.0
    rm._trades_today = 2
    rm._losses_today = 1

    event_time = future_reset + timedelta(minutes=1)
    ts_event = int(event_time.timestamp() * 1_000_000_000)

    rm.update_daily_pnl(-5.0, ts_event=ts_event)

    assert rm._daily_pnl == -5.0
    assert rm._trades_today == 1
    assert rm._losses_today == 1
