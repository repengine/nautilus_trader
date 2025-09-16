#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from ml.common.message_topics import build_topic_for_stage
from ml.config.bus import MessageBusConfig
from ml.config.events import Source, Stage
from ml.stores.strategy_store import StrategyStore


class CapturePublisher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> None:  # noqa: D401 - simple protocol
        self.messages.append((topic, payload))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.database,
    pytest.mark.usefixtures("clean_postgres_db_module"),
]


def test_strategy_store_publishes_batch_event(postgres_connection: str) -> None:
    pub = CapturePublisher()
    store = StrategyStore(
        connection_string=postgres_connection,
        batch_size=2,
        flush_interval_seconds=0.01,
        enable_publishing=True,
        publisher=pub,
        publish_mode="batch",
    )

    t0 = 1_700_000_000_000_000_000
    store.write_signal(
        strategy_id="S",
        instrument_id="EUR/USD",
        signal_type="BUY",
        strength=0.9,
        model_predictions={"m": 0.9},
        risk_metrics={},
        execution_params={},
        ts_event=t0,
        is_live=False,
    )
    store.write_signal(
        strategy_id="S",
        instrument_id="EUR/USD",
        signal_type="SELL",
        strength=0.1,
        model_predictions={"m": 0.1},
        risk_metrics={},
        execution_params={},
        ts_event=t0 + 1,
        is_live=False,
    )

    # batch_size=2 triggers flush; verify one batch publish occurred
    assert len(pub.messages) >= 1
    topic, payload = pub.messages[0]

    expected_topic = build_topic_for_stage(Stage.SIGNAL_EMITTED, "EUR/USD")
    assert topic == expected_topic
    assert payload["dataset_id"] == "signals"
    assert payload["count"] == 2
    assert payload["source"] == "strategy"
    assert int(payload["ts_min"]) == t0
    assert int(payload["ts_max"]) == t0 + 1
