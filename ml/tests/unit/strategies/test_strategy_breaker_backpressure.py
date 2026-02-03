from __future__ import annotations

from typing import Any, cast

from ml.actors.base import MLSignal
from ml.common.in_memory_bus import InMemoryPublisher
from ml.config.base import CircuitBreakerConfig, MLStrategyConfig
from ml.config.events import EventStatus, Stage
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.stores.protocols import StrategyStoreProtocol
from nautilus_trader.model.identifiers import InstrumentId


class _FailingStore:
    def __init__(self) -> None:
        self.calls = 0

    def write_signal(
        self,
        *,
        strategy_id: str,
        instrument_id: str,
        signal_type: str,
        strength: float,
        model_predictions: dict[str, float],
        risk_metrics: dict[str, float],
        execution_params: dict[str, Any],
        ts_event: int,
        decision_metadata: dict[str, Any],
        is_live: bool = False,
        run_id: str | None = None,
    ) -> None:
        self.calls += 1
        raise RuntimeError("store down")

    def flush(self) -> None:  # pragma: no cover - not used
        return None

    def write_batch(self, data: list[Any]) -> None:  # pragma: no cover - cold path
        raise RuntimeError("store down")

    def read_signals(
        self,
        strategy_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> list[Any]:  # pragma: no cover - cold path
        return []

    def get_strategy_performance(
        self,
        strategy_id: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, Any]:  # pragma: no cover - cold path
        return {}

    def get_signal_distribution(
        self,
        strategy_id: str | None = None,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> dict[str, int]:  # pragma: no cover - cold path
        return {}


class _StubStrategy(MLTradingStrategy):
    def __init__(self, cfg: MLStrategyConfig) -> None:
        super().__init__(cfg)
        # Inject failing store and an in-memory publisher to observe fallback
        self.stub_store = _FailingStore()
        self.strategy_store = cast(StrategyStoreProtocol, self.stub_store)
        self._bus_publisher = InMemoryPublisher()
        self._events: list[tuple[str, dict[str, Any]]] = []
        self._bus_publisher.subscribe("ml.strategies.created.#", self._capture)
        self._bus_publisher.subscribe("events.ml.SIGNAL_EMITTED.#", self._capture)

    def _capture(self, topic: str, payload: dict[str, Any]) -> None:
        self._events.append((topic, payload))


def test_store_breaker_opens_and_emits_partial_event() -> None:
    cfg = MLStrategyConfig(
        strategy_id="S-TEST",
        ml_signal_source="SRC",
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        position_size_pct=0.02,
        min_confidence=0.0,
        max_positions=1,
        execute_trades=False,
        use_strategy_store=True,
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1, success_threshold=2),
    )
    strat = _StubStrategy(cfg)
    sig = MLSignal(
        instrument_id=cfg.instrument_id,
        model_id="M",
        prediction=0.9,
        confidence=0.8,
        metadata={"decision_metadata": {"version": "v1"}},
        ts_event=111,
        ts_init=111,
    )

    # First two attempts fail and advance breaker towards open
    for _ in range(2):
        strat._persist_strategy_decision(
            signal=sig,
            decision_type="BUY",
            position_size=None,
            risk_metrics=None,
            execution_params=None,
        )

    # Third call should see breaker open and publish PARTIAL without invoking store
    before_calls = strat.stub_store.calls
    strat._persist_strategy_decision(
        signal=sig,
        decision_type="BUY",
        position_size=None,
        risk_metrics=None,
        execution_params=None,
    )
    after_calls = strat.stub_store.calls
    assert after_calls == before_calls, "Expected store write to be skipped when breaker open"

    # Validate a PARTIAL event was emitted
    assert strat._events, "Expected a fallback PARTIAL event"
    _, payload = strat._events[-1]
    assert payload["stage"] == Stage.SIGNAL_EMITTED.value
    assert payload["status"] == EventStatus.PARTIAL.value
