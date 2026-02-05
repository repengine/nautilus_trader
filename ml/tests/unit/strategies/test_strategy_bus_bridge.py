from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any

from ml.actors.base import MLSignal
from ml.common.bus_bridge import DomainEventBridge
from ml.common.message_bus import MessagePublisherProtocol
from ml.config.base import MLStrategyConfig
from ml.strategies.ml_strategy import MLTradingStrategy
from nautilus_trader.model.identifiers import InstrumentId


@contextmanager
def env(vars: dict[str, str]) -> None:
    old = {k: os.environ.get(k) for k in vars}
    try:
        os.environ.update(vars)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.calls.append((topic, payload))
        return True


def test_strategy_bus_bridge_enqueues_publish(monkeypatch) -> None:
    publisher = CapturePublisher()

    def _fake_factory(_cfg: Any) -> MessagePublisherProtocol:
        return publisher

    with env(
        {
            "ML_BUS_ENABLE": "1",
            "ML_BUS_FROM_STRATEGY": "1",
            "ML_BUS_FROM_STORE": "0",
        },
    ):
        monkeypatch.setattr("ml.common.message_bus.publisher_from_config", _fake_factory)

        cfg = MLStrategyConfig(
            strategy_id="S-ASYNC",
            ml_signal_source="SRC",
            instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
            position_size_pct=0.02,
            min_confidence=0.5,
            max_positions=1,
            execute_trades=False,
            use_strategy_store=False,
        )
        strat = MLTradingStrategy(cfg)

        assert isinstance(strat._bus_bridge, DomainEventBridge)
        assert strat._bus_publisher is strat._bus_bridge

        sig = MLSignal(
            instrument_id=cfg.instrument_id,
            model_id="M",
            prediction=0.9,
            confidence=0.8,
            metadata={"decision_metadata": {"version": "v1"}},
            ts_event=123,
            ts_init=123,
        )
        strat._publish_decision_event(
            signal=sig,
            decision_type="BUY",
            risk_metrics={"r": 1.0},
            execution_params={"e": 2},
            model_predictions={"M": 0.9},
        )

        time.sleep(0.05)

        bridge = getattr(strat, "_bus_bridge", None)
        if bridge is not None:
            bridge.stop(drain=True, timeout=1.0)

    assert publisher.calls, "Expected async bridge to publish at least one event"
