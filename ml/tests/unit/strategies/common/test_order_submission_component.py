"""
Unit tests for OrderSubmissionComponent.

This module tests the order submission component extracted from BaseMLStrategy,
verifying market order creation, smart order execution, stop loss placement,
circuit breaker handling, and metrics recording.

Test Categories:
- Market order creation and submission
- Smart order creation with executor
- Stop loss order placement
- Circuit breaker backpressure handling
- Metrics recording
- Error handling and edge cases

Test Cases Satisfied:
1. test_place_market_order_basic
2. test_place_market_order_circuit_breaker_suppression
3. test_place_market_order_metrics_recorded
4. test_place_market_order_reduce_only
5. test_submit_smart_order_uses_executor
6. test_submit_smart_order_fallback_to_market
7. test_submit_smart_order_circuit_breaker_degradation
8. test_submit_smart_order_builds_market_state
9. test_submit_smart_order_performance_recording
10. test_place_stop_loss_basic
11. test_place_stop_loss_correct_trigger_type
12. test_place_stop_loss_metrics_recorded
13. test_submit_smart_order_unwraps_order_result
14. test_submit_smart_order_exception_fallback
15. test_place_market_order_returns_client_order_id

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from nautilus_trader.model.enums import OrderSide, TriggerType
from nautilus_trader.model.identifiers import (
    ClientOrderId,
    InstrumentId,
    StrategyId,
    Symbol,
    TraderId,
    Venue,
)
from nautilus_trader.model.objects import Price, Quantity


# ---------------------------------------------------------------------------
# Test Fixtures and Stubs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MockPrice:
    """Mock price object with as_double method."""

    _value: float

    def as_double(self) -> float:
        """Return price as float."""
        return self._value


@dataclass(slots=True)
class MockQuoteTick:
    """Mock quote tick with bid/ask prices."""

    bid_price: MockPrice = field(default_factory=lambda: MockPrice(99.0))
    ask_price: MockPrice = field(default_factory=lambda: MockPrice(101.0))
    ts_event: int = 0


@dataclass(slots=True)
class MockInstrument:
    """Mock instrument with venue."""

    id: InstrumentId
    venue: Venue


class MockClientOrderId:
    """Mock client order ID."""

    _counter: int = 0

    def __init__(self, value: str | None = None) -> None:
        """Initialize mock client order ID."""
        if value is None:
            MockClientOrderId._counter += 1
            self._value = f"O-{MockClientOrderId._counter:06d}"
        else:
            self._value = value

    @property
    def value(self) -> str:
        """Return the value."""
        return self._value

    def __str__(self) -> str:
        """Return string representation."""
        return self._value


class MockCache:
    """Mock cache for testing."""

    def __init__(
        self,
        *,
        instrument: MockInstrument | None = None,
        quote_tick: MockQuoteTick | None = None,
    ) -> None:
        """Initialize mock cache."""
        self._instrument = instrument
        self._quote_tick = quote_tick
        self._order_id_counter = 0

    def instrument(self, instrument_id: Any) -> MockInstrument | None:
        """Return mock instrument."""
        return self._instrument

    def quote_tick(self, instrument_id: Any) -> MockQuoteTick | None:
        """Return mock quote tick."""
        return self._quote_tick

    def client_order_id(self) -> ClientOrderId:
        """Generate a new client order ID."""
        self._order_id_counter += 1
        return ClientOrderId(f"O-{self._order_id_counter:06d}")


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(
        self,
        *,
        can_execute_result: bool = True,
        should_raise: bool = False,
    ) -> None:
        """Initialize mock circuit breaker."""
        self._can_execute_result = can_execute_result
        self._should_raise = should_raise
        self.can_execute_calls: int = 0
        self.record_success_calls: int = 0
        self.record_failure_calls: int = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        self.can_execute_calls += 1
        if self._should_raise:
            raise RuntimeError("Circuit breaker error")
        return self._can_execute_result

    def record_success(self) -> None:
        """Record a successful execution."""
        self.record_success_calls += 1

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.record_failure_calls += 1


class MockOrderExecutor:
    """Mock order executor for testing."""

    def __init__(
        self,
        *,
        return_value: Any = None,
        should_return_none: bool = False,
        should_raise: bool = False,
    ) -> None:
        """Initialize mock order executor."""
        self._return_value = return_value
        self._should_return_none = should_return_none
        self._should_raise = should_raise
        self.create_order_calls: list[dict[str, Any]] = []

    def create_order(
        self,
        side: Any,
        quantity: Any,
        signal: Any,
        market_state: dict[str, float],
        instrument: Any,
        *,
        trader_id: Any = None,
        strategy_id: Any = None,
        client_order_id: Any = None,
        init_id: Any = None,
        ts_init: int | None = None,
    ) -> Any:
        """Create an order based on signal and market conditions."""
        self.create_order_calls.append({
            "side": side,
            "quantity": quantity,
            "signal": signal,
            "market_state": market_state,
            "instrument": instrument,
            "trader_id": trader_id,
            "strategy_id": strategy_id,
            "client_order_id": client_order_id,
            "init_id": init_id,
            "ts_init": ts_init,
        })

        if self._should_raise:
            raise RuntimeError("Order executor error")
        if self._should_return_none:
            return None
        return self._return_value


class MockOrderResult:
    """Mock order result with unwrap method."""

    def __init__(self, order: Any) -> None:
        """Initialize mock order result."""
        self._order = order
        self.client_order_id = order.client_order_id if order else None

    def unwrap(self) -> Any:
        """Unwrap the order."""
        return self._order


class MockOrder:
    """Mock order for testing."""

    def __init__(
        self,
        client_order_id: ClientOrderId | None = None,
        order_side: OrderSide = OrderSide.BUY,
    ) -> None:
        """Initialize mock order."""
        self.client_order_id = client_order_id or ClientOrderId("O-TEST-001")
        self.order_side = order_side


class MockPerformanceTracker:
    """Mock performance tracker for testing."""

    def __init__(
        self,
        *,
        should_raise: bool = False,
    ) -> None:
        """Initialize mock performance tracker."""
        self._should_raise = should_raise
        self.record_order_calls: list[tuple[Any, Any]] = []

    def record_order(self, order: Any, signal: Any) -> None:
        """Record an order placement."""
        self.record_order_calls.append((order, signal))
        if self._should_raise:
            raise RuntimeError("Performance tracker error")


class MockMetric:
    """Mock Prometheus metric for testing."""

    def __init__(self) -> None:
        """Initialize mock metric."""
        self.label_calls: list[dict[str, str]] = []
        self.inc_calls: int = 0

    def labels(self, **kwargs: str) -> MockMetric:
        """Add labels to metric."""
        self.label_calls.append(kwargs)
        return self

    def inc(self) -> None:
        """Increment the counter."""
        self.inc_calls += 1


class MockClock:
    """Mock clock for testing."""

    def __init__(self, ts: int = 1_000_000_000) -> None:
        """Initialize mock clock."""
        self._ts = ts

    def timestamp_ns(self) -> int:
        """Return timestamp in nanoseconds."""
        return self._ts


class MockMLSignal:
    """Mock ML signal for testing."""

    def __init__(
        self,
        *,
        instrument_id: InstrumentId | None = None,
        model_id: str = "test_model",
        prediction: float = 0.7,
        confidence: float = 0.8,
        ts_event: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize mock signal."""
        self.instrument_id = instrument_id or InstrumentId(Symbol("EURUSD"), Venue("SIM"))
        self.model_id = model_id
        self.prediction = prediction
        self.confidence = confidence
        self.ts_event = ts_event
        self.metadata = metadata or {}


class MockLogger:
    """Mock logger for testing."""

    def __init__(self) -> None:
        """Initialize mock logger."""
        self.debug_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.info_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.warning_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.error_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, *args: Any, **kwargs: Any) -> None:
        """Record debug call."""
        self.debug_calls.append((args, kwargs))

    def info(self, *args: Any, **kwargs: Any) -> None:
        """Record info call."""
        self.info_calls.append((args, kwargs))

    def warning(self, *args: Any, **kwargs: Any) -> None:
        """Record warning call."""
        self.warning_calls.append((args, kwargs))

    def error(self, *args: Any, **kwargs: Any) -> None:
        """Record error call."""
        self.error_calls.append((args, kwargs))


@pytest.fixture
def instrument_id() -> InstrumentId:
    """Create test instrument ID."""
    return InstrumentId(Symbol("EURUSD"), Venue("SIM"))


@pytest.fixture
def mock_instrument(instrument_id: InstrumentId) -> MockInstrument:
    """Create mock instrument."""
    return MockInstrument(
        id=instrument_id,
        venue=instrument_id.venue,
    )


@pytest.fixture
def mock_quote_tick() -> MockQuoteTick:
    """Create mock quote tick with bid=99, ask=101."""
    return MockQuoteTick(
        bid_price=MockPrice(99.0),
        ask_price=MockPrice(101.0),
    )


@pytest.fixture
def mock_cache(
    mock_instrument: MockInstrument,
    mock_quote_tick: MockQuoteTick,
) -> MockCache:
    """Create mock cache with all components."""
    return MockCache(
        instrument=mock_instrument,
        quote_tick=mock_quote_tick,
    )


@pytest.fixture
def mock_signal(instrument_id: InstrumentId) -> MockMLSignal:
    """Create mock ML signal."""
    return MockMLSignal(
        instrument_id=instrument_id,
        model_id="test_model",
        prediction=0.7,
        confidence=0.8,
    )


@pytest.fixture
def mock_logger() -> MockLogger:
    """Create mock logger."""
    return MockLogger()


@pytest.fixture
def mock_clock() -> MockClock:
    """Create mock clock."""
    return MockClock()


@pytest.fixture
def mock_metric() -> MockMetric:
    """Create mock metric."""
    return MockMetric()


@pytest.fixture
def mock_trader_id() -> TraderId:
    """Create mock trader ID."""
    return TraderId("TESTER-001")


@pytest.fixture
def mock_strategy_id() -> StrategyId:
    """Create mock strategy ID."""
    return StrategyId("test_strategy-001")


@pytest.fixture
def order_submission_component(
    instrument_id: InstrumentId,
    mock_cache: MockCache,
    mock_logger: MockLogger,
    mock_clock: MockClock,
    mock_trader_id: TraderId,
    mock_strategy_id: StrategyId,
) -> OrderSubmissionComponent:
    """Create order submission component with standard configuration."""
    from ml.strategies.common.order_submission import OrderSubmissionComponent

    submitted_orders: list[Any] = []

    def submit_callback(order: Any) -> None:
        submitted_orders.append(order)

    component = OrderSubmissionComponent(
        strategy_id=mock_strategy_id,
        cache=mock_cache,
        submit_order_callback=submit_callback,
        log=mock_logger,
        instrument_id=instrument_id,
        trader_id=mock_trader_id,
        clock=mock_clock,
    )
    # Attach submitted_orders for test verification
    component._submitted_orders = submitted_orders  # type: ignore
    return component


# ---------------------------------------------------------------------------
# Test: Market Order Submission
# ---------------------------------------------------------------------------


class TestPlaceMarketOrder:
    """Tests for market order placement."""

    def test_place_market_order_basic(
        self,
        order_submission_component: OrderSubmissionComponent,
        instrument_id: InstrumentId,
    ) -> None:
        """Test case 1: Verify basic market order creation."""
        order_id = order_submission_component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
        )

        # Should return a ClientOrderId
        assert order_id is not None
        assert isinstance(order_id, ClientOrderId)

        # Should have submitted an order
        assert len(order_submission_component._submitted_orders) == 1  # type: ignore

        # Verify order properties
        submitted_order = order_submission_component._submitted_orders[0]  # type: ignore
        assert submitted_order.side == OrderSide.BUY
        assert float(submitted_order.quantity.as_double()) == 10.0

    def test_place_market_order_circuit_breaker_suppression(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 2: Verify order suppressed when circuit breaker open."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        circuit_breaker = MockCircuitBreaker(can_execute_result=False)
        submitted_orders: list[Any] = []

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            circuit_breaker=circuit_breaker,
            cache=mock_cache,
            submit_order_callback=lambda o: submitted_orders.append(o),
            log=mock_logger,
            clock=mock_clock,
        )

        initial_dry_run = component.dry_run_trades

        order_id = component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
        )

        # Should return a client order ID (from cache) but not submit
        assert order_id is not None
        assert len(submitted_orders) == 0
        assert component.dry_run_trades == initial_dry_run + 1

        # Check info log was called
        assert len(mock_logger.info_calls) > 0

    def test_place_market_order_metrics_recorded(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_logger: MockLogger,
        mock_clock: MockClock,
        mock_trader_id: TraderId,
        mock_strategy_id: StrategyId,
    ) -> None:
        """Test case 3: Verify order metrics recorded."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        mock_metric = MockMetric()

        component = OrderSubmissionComponent(
            strategy_id=mock_strategy_id,
            cache=mock_cache,
            submit_order_callback=lambda o: None,
            log=mock_logger,
            clock=mock_clock,
            orders_submitted_metric=mock_metric,
            trader_id=mock_trader_id,
        )

        component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
        )

        # Metric should have been incremented
        assert mock_metric.inc_calls == 1
        assert len(mock_metric.label_calls) == 1
        assert mock_metric.label_calls[0]["order_side"] == "BUY"

    def test_place_market_order_reduce_only(
        self,
        order_submission_component: OrderSubmissionComponent,
        instrument_id: InstrumentId,
    ) -> None:
        """Test case 4: Verify reduce_only flag passed correctly."""
        order_submission_component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.SELL,
            quantity=Quantity.from_str("5.0"),
            reduce_only=True,
        )

        # Verify reduce_only on submitted order
        submitted_order = order_submission_component._submitted_orders[0]  # type: ignore
        assert submitted_order.is_reduce_only is True

    def test_place_market_order_returns_client_order_id(
        self,
        order_submission_component: OrderSubmissionComponent,
        instrument_id: InstrumentId,
    ) -> None:
        """Test case 15: Verify client order ID returned."""
        order_id = order_submission_component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
        )

        # Should return a ClientOrderId instance
        assert order_id is not None
        assert isinstance(order_id, ClientOrderId)
        assert order_id.value.startswith("O-")


# ---------------------------------------------------------------------------
# Test: Smart Order Submission
# ---------------------------------------------------------------------------


class TestSubmitSmartOrder:
    """Tests for smart order submission."""

    def test_submit_smart_order_uses_executor(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 5: Verify smart executor is used when available."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        # Create a mock order to return
        mock_order = MockOrder(client_order_id=ClientOrderId("O-SMART-001"))
        executor = MockOrderExecutor(return_value=mock_order)
        submitted_orders: list[Any] = []

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            order_executor=executor,
            cache=mock_cache,
            submit_order_callback=lambda o: submitted_orders.append(o),
            log=mock_logger,
            clock=mock_clock,
            trader_id="TESTER-001",
        )

        result = component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=mock_instrument,
        )

        # Executor should have been called
        assert len(executor.create_order_calls) == 1
        assert executor.create_order_calls[0]["side"] == OrderSide.BUY

    def test_submit_smart_order_fallback_to_market(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 6: Verify fallback to market order when executor returns None."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        executor = MockOrderExecutor(should_return_none=True)
        submitted_orders: list[Any] = []

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            order_executor=executor,
            cache=mock_cache,
            submit_order_callback=lambda o: submitted_orders.append(o),
            log=mock_logger,
            clock=mock_clock,
            trader_id="TESTER-001",
        )

        result = component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=mock_instrument,
        )

        # Should have fallen back to market order
        assert result is not None
        assert len(submitted_orders) == 1

    def test_submit_smart_order_circuit_breaker_degradation(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 7: Verify circuit breaker causes dry run."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        circuit_breaker = MockCircuitBreaker(can_execute_result=False)
        submitted_orders: list[Any] = []

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            circuit_breaker=circuit_breaker,
            cache=mock_cache,
            submit_order_callback=lambda o: submitted_orders.append(o),
            log=mock_logger,
            clock=mock_clock,
        )

        initial_dry_run = component.dry_run_trades

        result = component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=mock_instrument,
        )

        # Should return None and increment dry_run_trades
        assert result is None
        assert len(submitted_orders) == 0
        assert component.dry_run_trades == initial_dry_run + 1

    def test_submit_smart_order_builds_market_state(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 8: Verify market state snapshot built correctly."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        mock_order = MockOrder(client_order_id=ClientOrderId("O-SMART-001"))
        executor = MockOrderExecutor(return_value=mock_order)

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            order_executor=executor,
            cache=mock_cache,
            submit_order_callback=lambda o: None,
            log=mock_logger,
            clock=mock_clock,
            trader_id="TESTER-001",
        )

        component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=mock_instrument,
        )

        # Check market state was built correctly
        market_state = executor.create_order_calls[0]["market_state"]
        assert "bid" in market_state
        assert "ask" in market_state
        assert "spread_bps" in market_state
        assert market_state["bid"] == 99.0
        assert market_state["ask"] == 101.0

    def test_submit_smart_order_performance_recording(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 9: Verify performance tracker records order."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        mock_order = MockOrder(client_order_id=ClientOrderId("O-SMART-001"))
        executor = MockOrderExecutor(return_value=mock_order)
        performance_tracker = MockPerformanceTracker()

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            order_executor=executor,
            performance_tracker=performance_tracker,
            cache=mock_cache,
            submit_order_callback=lambda o: None,
            log=mock_logger,
            clock=mock_clock,
            trader_id="TESTER-001",
        )

        component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=mock_instrument,
        )

        # Performance tracker should have been called
        assert len(performance_tracker.record_order_calls) == 1

    def test_submit_smart_order_unwraps_order_result(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 13: Verify OrderResult wrapper is unwrapped."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        # Create an order wrapped in OrderResult
        inner_order = MockOrder(client_order_id=ClientOrderId("O-INNER-001"))
        wrapped_order = MockOrderResult(inner_order)
        executor = MockOrderExecutor(return_value=wrapped_order)
        submitted_orders: list[Any] = []

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            order_executor=executor,
            cache=mock_cache,
            submit_order_callback=lambda o: submitted_orders.append(o),
            log=mock_logger,
            clock=mock_clock,
            trader_id="TESTER-001",
        )

        component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=mock_instrument,
        )

        # Should have unwrapped and submitted the inner order
        assert len(submitted_orders) == 1
        assert submitted_orders[0] is inner_order

    def test_submit_smart_order_exception_fallback(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 14: Verify fallback on executor exception."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        executor = MockOrderExecutor(should_raise=True)
        submitted_orders: list[Any] = []

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            order_executor=executor,
            cache=mock_cache,
            submit_order_callback=lambda o: submitted_orders.append(o),
            log=mock_logger,
            clock=mock_clock,
            trader_id="TESTER-001",
        )

        result = component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=mock_instrument,
        )

        # Should have logged error and fallen back to market order
        assert len(mock_logger.error_calls) > 0
        assert result is not None
        assert len(submitted_orders) == 1


# ---------------------------------------------------------------------------
# Test: Stop Loss Order Placement
# ---------------------------------------------------------------------------


class TestPlaceStopLoss:
    """Tests for stop loss order placement."""

    def test_place_stop_loss_basic(
        self,
        order_submission_component: OrderSubmissionComponent,
        instrument_id: InstrumentId,
    ) -> None:
        """Test case 10: Verify stop loss order creation."""
        order_id = order_submission_component.place_stop_loss(
            instrument_id=instrument_id,
            side=OrderSide.SELL,
            quantity=Quantity.from_str("10.0"),
            trigger_price=Price.from_str("95.0"),
        )

        # Should return a ClientOrderId
        assert order_id is not None
        assert isinstance(order_id, ClientOrderId)

        # Should have submitted an order
        assert len(order_submission_component._submitted_orders) == 1  # type: ignore

        # Verify order properties
        submitted_order = order_submission_component._submitted_orders[0]  # type: ignore
        assert submitted_order.side == OrderSide.SELL
        assert submitted_order.is_reduce_only is True
        assert float(submitted_order.trigger_price.as_double()) == 95.0

    def test_place_stop_loss_correct_trigger_type(
        self,
        order_submission_component: OrderSubmissionComponent,
        instrument_id: InstrumentId,
    ) -> None:
        """Test case 11: Verify stop loss uses correct trigger type."""
        order_submission_component.place_stop_loss(
            instrument_id=instrument_id,
            side=OrderSide.SELL,
            quantity=Quantity.from_str("10.0"),
            trigger_price=Price.from_str("95.0"),
        )

        # Verify trigger type
        submitted_order = order_submission_component._submitted_orders[0]  # type: ignore
        assert submitted_order.trigger_type == TriggerType.DEFAULT

    def test_place_stop_loss_metrics_recorded(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Test case 12: Verify stop loss metrics recorded."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        mock_metric = MockMetric()

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            cache=mock_cache,
            submit_order_callback=lambda o: None,
            log=mock_logger,
            clock=mock_clock,
            orders_submitted_metric=mock_metric,
            trader_id="TESTER-001",
        )

        component.place_stop_loss(
            instrument_id=instrument_id,
            side=OrderSide.SELL,
            quantity=Quantity.from_str("10.0"),
            trigger_price=Price.from_str("95.0"),
        )

        # Metric should have been incremented
        assert mock_metric.inc_calls == 1
        assert len(mock_metric.label_calls) == 1
        assert mock_metric.label_calls[0]["order_side"] == "SELL"


# ---------------------------------------------------------------------------
# Test: Component Configuration
# ---------------------------------------------------------------------------


class TestComponentConfiguration:
    """Tests for component configuration and properties."""

    def test_properties_accessible(
        self,
        order_submission_component: OrderSubmissionComponent,
    ) -> None:
        """Verify all properties are accessible."""
        assert str(order_submission_component.strategy_id) == "test_strategy-001"
        assert order_submission_component.order_executor is None
        assert order_submission_component.circuit_breaker is None
        assert order_submission_component.dry_run_trades == 0
        assert order_submission_component.trades_executed == 0
        assert order_submission_component.pending_orders == 0

    def test_update_config(
        self,
        order_submission_component: OrderSubmissionComponent,
    ) -> None:
        """Verify configuration can be updated."""
        new_instrument_id = InstrumentId(Symbol("GBPUSD"), Venue("SIM"))
        new_clock = MockClock(ts=2_000_000_000)

        order_submission_component.update_config(
            instrument_id=new_instrument_id,
            clock=new_clock,
        )

        assert order_submission_component._instrument_id == new_instrument_id
        assert order_submission_component._clock == new_clock


# ---------------------------------------------------------------------------
# Test: Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_place_market_order_no_cache(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when cache not available."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            cache=None,
            log=mock_logger,
        )

        result = component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
        )

        assert result is None
        assert len(mock_logger.error_calls) > 0

    def test_place_stop_loss_no_cache(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when cache not available for stop loss."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            cache=None,
            log=mock_logger,
        )

        result = component.place_stop_loss(
            instrument_id=instrument_id,
            side=OrderSide.SELL,
            quantity=Quantity.from_str("10.0"),
            trigger_price=Price.from_str("95.0"),
        )

        assert result is None
        assert len(mock_logger.error_calls) > 0

    def test_submit_smart_order_null_instrument(
        self,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when instrument is None."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            log=mock_logger,
        )

        result = component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=None,
        )

        assert result is None

    def test_circuit_breaker_exception_handling(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Verify circuit breaker exceptions are handled gracefully."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        circuit_breaker = MockCircuitBreaker(should_raise=True)
        submitted_orders: list[Any] = []

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            circuit_breaker=circuit_breaker,
            cache=mock_cache,
            submit_order_callback=lambda o: submitted_orders.append(o),
            log=mock_logger,
            clock=mock_clock,
            trader_id="TESTER-001",
        )

        # Should not raise, should proceed with order
        result = component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
        )

        assert result is not None
        assert len(submitted_orders) == 1
        assert len(mock_logger.debug_calls) > 0

    def test_performance_tracker_exception_handling(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
        mock_clock: MockClock,
    ) -> None:
        """Verify performance tracker exceptions don't break order submission."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        mock_order = MockOrder(client_order_id=ClientOrderId("O-SMART-001"))
        executor = MockOrderExecutor(return_value=mock_order)
        performance_tracker = MockPerformanceTracker(should_raise=True)
        submitted_orders: list[Any] = []

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            order_executor=executor,
            performance_tracker=performance_tracker,
            cache=mock_cache,
            submit_order_callback=lambda o: submitted_orders.append(o),
            log=mock_logger,
            clock=mock_clock,
            trader_id="TESTER-001",
        )

        # Should not raise, order should still be submitted
        result = component.submit_smart_order(
            signal=mock_signal,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
            instrument=mock_instrument,
        )

        assert len(submitted_orders) == 1
        assert len(mock_logger.debug_calls) > 0

    def test_trades_executed_counter_increments(
        self,
        order_submission_component: OrderSubmissionComponent,
        instrument_id: InstrumentId,
    ) -> None:
        """Verify trades_executed counter increments on order submission."""
        initial_count = order_submission_component.trades_executed

        order_submission_component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
        )

        assert order_submission_component.trades_executed == initial_count + 1

    def test_pending_orders_counter_increments(
        self,
        order_submission_component: OrderSubmissionComponent,
        instrument_id: InstrumentId,
    ) -> None:
        """Verify pending_orders counter increments on order submission."""
        initial_count = order_submission_component.pending_orders

        order_submission_component.place_market_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("10.0"),
        )

        assert order_submission_component.pending_orders == initial_count + 1


# ---------------------------------------------------------------------------
# Test: Market State Building
# ---------------------------------------------------------------------------


class TestMarketStateBuilding:
    """Tests for market state snapshot building."""

    def test_build_market_state_with_quote(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_logger: MockLogger,
    ) -> None:
        """Verify market state built correctly from quote tick."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            cache=mock_cache,
            log=mock_logger,
        )

        market_state, is_stale = component._build_market_state(instrument_id)

        assert market_state["bid"] == 99.0
        assert market_state["ask"] == 101.0
        # spread_bps = ((101 - 99) / 100) * 10000 = 200
        assert abs(market_state["spread_bps"] - 200.0) < 0.1
        assert is_stale is False

    def test_build_market_state_no_quote(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify market state defaults when no quote available."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        cache = MockCache(quote_tick=None)

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            cache=cache,
            log=mock_logger,
        )

        market_state, is_stale = component._build_market_state(instrument_id)

        assert market_state["bid"] == 0.0
        assert market_state["ask"] == 0.0
        assert market_state["spread_bps"] == 0.0
        assert is_stale is False

    def test_build_market_state_no_cache(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify market state defaults when no cache available."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            cache=None,
            log=mock_logger,
        )

        market_state, is_stale = component._build_market_state(instrument_id)

        assert market_state["bid"] == 0.0
        assert market_state["ask"] == 0.0
        assert market_state["spread_bps"] == 0.0
        assert is_stale is False

    def test_build_market_state_when_quote_stale_returns_stale_flag(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify stale quotes are detected and flagged."""
        from ml.strategies.common.order_submission import OrderSubmissionComponent

        cache = MockCache(
            quote_tick=MockQuoteTick(
                bid_price=MockPrice(99.0),
                ask_price=MockPrice(101.0),
                ts_event=1_000_000_000,
            ),
        )

        component = OrderSubmissionComponent(
            strategy_id="test-strategy-001",
            cache=cache,
            log=mock_logger,
            max_quote_age_ms=1,
        )

        market_state, is_stale = component._build_market_state(
            instrument_id,
            reference_ts=2_500_000_000,
        )

        assert market_state["bid"] == 0.0
        assert market_state["ask"] == 0.0
        assert market_state["spread_bps"] == 0.0
        assert is_stale is True
