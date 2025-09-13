from __future__ import annotations

import time
from typing import Any

import pytest

from ml.common.message_bus import MessagePublisherProtocol
from ml.stores.base import StrategySignal
from ml.stores.strategy_store import StrategyStore


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


@pytest.mark.parametrize("mode,expected_extra", [
    ("batch", 1),
    ("row", 2),
    ("both", 3),
])
def test_strategy_store_publishing_modes(mode: str, expected_extra: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid DB init
    monkeypatch.setattr(StrategyStore, "_init_engine_and_tables", lambda self: None)
    cap = CapturePublisher()
    store = StrategyStore(connection_string=None, enable_publishing=True, publisher=cap, publish_mode=mode)

    # Monkeypatch the mixin upsert to only publish
    from ml.stores._batch_utils import publish_batch_and_rows

    def _stub_execute_write(values: list[dict[str, Any]]) -> None:  # type: ignore[no-redef]
        publish_batch_and_rows(
            enable_publishing=bool(getattr(store, "_enable_publishing", False)),
            publisher=getattr(store, "publisher", None),
            publish_mode=getattr(store, "_publish_mode", "batch"),
            topic_scheme=getattr(store, "_topic_scheme", "domain_op"),
            topic_prefix=getattr(store, "_topic_prefix", "events.ml"),
            stage=__import__("ml.config.events", fromlist=["Stage"]).Stage.SIGNAL_EMITTED,  # avoid import at module level
            dataset_id="signals",
            instrument_key="instrument_id",
            ts_field="ts_event",
            rows=values,
            run_id_batch="strategy_store_write",
            run_id_row="strategy_store_row",
            source="strategy",
            logger=__import__("logging").getLogger(__name__),
        )

    monkeypatch.setattr(store, "_execute_write", _stub_execute_write)

    now = time.time_ns()
    signals = [
        StrategySignal(
            strategy_id="S",
            instrument_id="SPY",
            signal_type="BUY",
            strength=0.5,
            model_predictions={},
            risk_metrics={},
            execution_params={},
            _ts_event=now,
            _ts_init=now,
        ),
        StrategySignal(
            strategy_id="S",
            instrument_id="SPY",
            signal_type="SELL",
            strength=0.4,
            model_predictions={},
            risk_metrics={},
            execution_params={},
            _ts_event=now + 1,
            _ts_init=now + 1,
        ),
    ]

    store.write_batch(signals)

    assert len(cap.calls) == expected_extra
    for _, payload in cap.calls:
        assert payload.get("dataset_id") == "signals"
        assert "stage" in payload
