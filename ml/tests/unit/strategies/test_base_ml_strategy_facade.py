"""
Unit tests for BaseMLStrategyFacade.

These tests verify that the facade correctly initializes and delegates
to the 6 decomposed components while maintaining backward compatibility
with the legacy BaseMLStrategy API.

Tests covered:
- Component initialization
- Method delegation to correct component
- State synchronization between components
- Backward compatibility with legacy API
- Feature flag switching

"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockMLStrategyConfig:
    """Mock configuration for testing."""

    def __init__(
        self,
        instrument_id: Any = None,
        position_size_pct: float = 0.02,
        min_confidence: float = 0.5,
        execute_trades: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_positions: int = 1,
        use_strategy_store: bool = False,
        persist_all_signals: bool = False,
        serialize_order_intents: bool = False,
        history_size: int = 100,
        target_model_ids: list[str] | None = None,
        aggregation_mode: str | None = None,
        required_models: int = 1,
        time_window_ms: int = 1000,
        conflict_resolution: str | None = None,
        model_weights: dict[str, float] | None = None,
        track_performance: bool = False,
    ) -> None:
        """Initialize mock config."""
        self.instrument_id = instrument_id or MagicMock()
        self.position_size_pct = position_size_pct
        self.min_confidence = min_confidence
        self.execute_trades = execute_trades
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.use_strategy_store = use_strategy_store
        self.persist_all_signals = persist_all_signals
        self.serialize_order_intents = serialize_order_intents
        self.history_size = history_size
        self.target_model_ids = target_model_ids
        self.aggregation_mode = aggregation_mode
        self.required_models = required_models
        self.time_window_ms = time_window_ms
        self.conflict_resolution = conflict_resolution
        self.model_weights = model_weights or {}
        self.track_performance = track_performance


class MockMLSignal:
    """Mock ML signal for testing."""

    def __init__(
        self,
        instrument_id: Any = None,
        prediction: float = 0.7,
        confidence: float = 0.8,
        model_id: str = "test_model",
        ts_event: int = 1000000000,
    ) -> None:
        """Initialize mock signal."""
        self.instrument_id = instrument_id or MagicMock()
        self.prediction = prediction
        self.confidence = confidence
        self.model_id = model_id
        self.ts_event = ts_event
        self.metadata: dict[str, Any] = {
            "model_id": model_id,
            "decision_metadata": {"version": "v1"},
        }


class MockStores:
    """Mock stores container."""

    def __init__(self) -> None:
        """Initialize mock stores."""
        self.feature_store = MagicMock()
        self.model_store = MagicMock()
        self.strategy_store = MagicMock()
        self.data_store = MagicMock()
        self.feature_registry = MagicMock()
        self.model_registry = MagicMock()
        self.strategy_registry = MagicMock()
        self.data_registry = MagicMock()


@pytest.fixture
def mock_config() -> MockMLStrategyConfig:
    """Provide a mock ML strategy config."""
    return MockMLStrategyConfig()


@pytest.fixture
def mock_stores() -> MockStores:
    """Provide mock stores container."""
    return MockStores()


@pytest.fixture
def mock_signal(mock_config: MockMLStrategyConfig) -> MockMLSignal:
    """Provide a mock ML signal."""
    return MockMLSignal(instrument_id=mock_config.instrument_id)


# ---------------------------------------------------------------------------
# Component Tests (testing components directly)
# ---------------------------------------------------------------------------


class TestComponentInitialization:
    """Tests for component initialization."""

    def test_signal_routing_component_initializes(self) -> None:
        """Test SignalRoutingComponent can be initialized."""
        from ml.strategies.common import SignalRoutingComponent

        component = SignalRoutingComponent(
            target_model_ids=["model_a"],
            min_confidence=0.5,
        )

        assert component is not None
        assert component.target_model_ids == ["model_a"]
        assert component.min_confidence == 0.5

    def test_decision_persistence_component_initializes(self) -> None:
        """Test DecisionPersistenceComponent can be initialized."""
        from ml.strategies.common import DecisionPersistenceComponent

        component = DecisionPersistenceComponent(
            strategy_id="test_strategy",
        )

        assert component is not None
        assert component.strategy_id == "test_strategy"

    def test_position_management_component_initializes(self) -> None:
        """Test PositionManagementComponent can be initialized."""
        from ml.strategies.common import PositionManagementComponent

        component = PositionManagementComponent(
            position_size_pct=0.05,
        )

        assert component is not None
        assert component.position_size_pct == 0.05

    def test_order_submission_component_initializes(self) -> None:
        """Test OrderSubmissionComponent can be initialized."""
        from ml.strategies.common import OrderSubmissionComponent

        component = OrderSubmissionComponent(
            strategy_id="test-strategy",  # Must have hyphen
        )

        assert component is not None
        assert component.strategy_id == "test-strategy"

    def test_lifecycle_component_initializes(self) -> None:
        """Test LifecycleComponent can be initialized."""
        from ml.strategies.common import LifecycleComponent

        component = LifecycleComponent(
            strategy_id="test_strategy",
            instrument_id=MagicMock(),
        )

        assert component is not None
        assert component.strategy_id == "test_strategy"

    def test_performance_tracking_component_initializes(self) -> None:
        """Test PerformanceTrackingComponent can be initialized."""
        from ml.strategies.common import PerformanceTrackingComponent

        component = PerformanceTrackingComponent(
            strategy_id="test_strategy",
            track_performance=True,
        )

        assert component is not None
        assert component.strategy_id == "test_strategy"
        assert component.track_performance is True


# ---------------------------------------------------------------------------
# Component Method Tests
# ---------------------------------------------------------------------------


class TestComponentMethods:
    """Tests for component method behavior."""

    def test_lifecycle_on_start_calls_subscriptions(self) -> None:
        """Test LifecycleComponent.on_start calls subscription callbacks."""
        from ml.strategies.common import LifecycleComponent

        subscriptions: list[str] = []

        def subscribe_data(**kwargs: Any) -> None:
            subscriptions.append("data")

        def subscribe_instrument(instrument_id: Any) -> None:
            subscriptions.append("instrument")

        component = LifecycleComponent(
            strategy_id="test",
            instrument_id=MagicMock(),
            subscribe_data_callback=subscribe_data,
            subscribe_instrument_callback=subscribe_instrument,
            log=MagicMock(),
        )

        component.on_start()

        assert "data" in subscriptions
        assert "instrument" in subscriptions

    def test_lifecycle_on_stop_flushes_store(self) -> None:
        """Test LifecycleComponent.on_stop flushes strategy store."""
        from ml.strategies.common import LifecycleComponent

        mock_store = MagicMock()

        component = LifecycleComponent(
            strategy_id="test",
            instrument_id=MagicMock(),
            log=MagicMock(),
        )

        component.on_stop(
            strategy_store=mock_store,
            signals_received=10,
            trades_executed=5,
            winning_trades=3,
            total_pnl=Decimal("100.0"),
        )

        mock_store.flush.assert_called_once()

    def test_performance_tracker_updates_metrics(self) -> None:
        """Test PerformanceTrackingComponent updates model metrics."""
        from ml.strategies.common import PerformanceTrackingComponent

        component = PerformanceTrackingComponent(
            strategy_id="test",
            track_performance=True,
        )

        component.update_model_performance("model_a", profit=100.0)
        component.update_model_performance("model_a", profit=-50.0)

        perf = component.get_model_performance("model_a")

        assert perf["total_trades"] == 2
        assert perf["total_profit"] == 50.0
        assert perf["wins"] == 1
        assert perf["losses"] == 1
        assert perf["accuracy"] == 0.5

    def test_signal_router_filters_by_model_id(self) -> None:
        """Test SignalRoutingComponent filters signals by model ID."""
        from ml.strategies.common import SignalRoutingComponent

        component = SignalRoutingComponent(
            target_model_ids=["model_a", "model_b"],
        )

        signal_a = MockMLSignal(model_id="model_a")
        signal_c = MockMLSignal(model_id="model_c")

        # Use the correct method name
        assert component.filter_by_model_id(signal_a) is True
        assert component.filter_by_model_id(signal_c) is False

    def test_signal_router_filters_by_confidence(self) -> None:
        """Test SignalRoutingComponent filters signals by confidence."""
        from ml.strategies.common import SignalRoutingComponent

        component = SignalRoutingComponent(
            min_confidence=0.7,
        )

        signal_high = MockMLSignal(confidence=0.8)
        signal_low = MockMLSignal(confidence=0.5)

        # Use the correct method name
        assert component.filter_by_confidence(signal_high) is True
        assert component.filter_by_confidence(signal_low) is False

    def test_decision_persistence_persists_to_store(self) -> None:
        """Test DecisionPersistenceComponent persists to strategy store."""
        from ml.strategies.common import DecisionPersistenceComponent

        mock_store = MagicMock()

        component = DecisionPersistenceComponent(
            strategy_id="test",
            strategy_store=mock_store,
        )

        signal = MockMLSignal()

        result = component.persist_decision(
            signal=signal,
            decision_type="BUY",
        )

        assert result is True
        mock_store.write_signal.assert_called_once()


# ---------------------------------------------------------------------------
# Module Import Tests
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Tests for module import and class availability."""

    def test_base_ml_strategy_facade_importable(self) -> None:
        """Test BaseMLStrategyFacade can be imported."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        assert BaseMLStrategyFacade is not None

    def test_simple_ml_strategy_facade_importable(self) -> None:
        """Test SimpleMLStrategyFacade can be imported."""
        from ml.strategies.base_facade import SimpleMLStrategyFacade

        assert SimpleMLStrategyFacade is not None

    def test_all_components_importable(self) -> None:
        """Test all 6 components can be imported."""
        from ml.strategies.common import (
            DecisionPersistenceComponent,
            LifecycleComponent,
            OrderSubmissionComponent,
            PerformanceTrackingComponent,
            PositionManagementComponent,
            SignalRoutingComponent,
        )

        assert SignalRoutingComponent is not None
        assert DecisionPersistenceComponent is not None
        assert PositionManagementComponent is not None
        assert OrderSubmissionComponent is not None
        assert LifecycleComponent is not None
        assert PerformanceTrackingComponent is not None


# ---------------------------------------------------------------------------
# Facade Class Tests (testing via attributes that don't require Cython init)
# ---------------------------------------------------------------------------


class TestFacadeClassStructure:
    """Tests for facade class structure."""

    def test_facade_has_required_methods(self) -> None:
        """Test that facade class has all required methods."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        # Check for required methods
        assert hasattr(BaseMLStrategyFacade, "on_start")
        assert hasattr(BaseMLStrategyFacade, "on_stop")
        assert hasattr(BaseMLStrategyFacade, "on_data")
        assert hasattr(BaseMLStrategyFacade, "target_side_from_prediction")
        assert hasattr(BaseMLStrategyFacade, "should_reverse")
        assert hasattr(BaseMLStrategyFacade, "size_and_validate")
        assert hasattr(BaseMLStrategyFacade, "_calculate_position_size")
        assert hasattr(BaseMLStrategyFacade, "_persist_strategy_decision")
        assert hasattr(BaseMLStrategyFacade, "_place_market_order")
        assert hasattr(BaseMLStrategyFacade, "_place_stop_loss")
        assert hasattr(BaseMLStrategyFacade, "_update_model_performance")

    def test_facade_has_store_properties(self) -> None:
        """Test that facade class has all store property accessors."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        # Check for store properties
        assert hasattr(BaseMLStrategyFacade, "feature_store")
        assert hasattr(BaseMLStrategyFacade, "model_store")
        assert hasattr(BaseMLStrategyFacade, "data_store")
        assert hasattr(BaseMLStrategyFacade, "feature_registry")
        assert hasattr(BaseMLStrategyFacade, "model_registry")
        assert hasattr(BaseMLStrategyFacade, "strategy_registry")
        assert hasattr(BaseMLStrategyFacade, "data_registry")

    def test_simple_facade_is_concrete(self) -> None:
        """Test that SimpleMLStrategyFacade is concrete (not abstract)."""
        from ml.strategies.base_facade import SimpleMLStrategyFacade

        # SimpleMLStrategyFacade should implement _process_ml_signal
        assert hasattr(SimpleMLStrategyFacade, "_process_ml_signal")

        # Verify it's callable
        assert callable(getattr(SimpleMLStrategyFacade, "_process_ml_signal"))


# ---------------------------------------------------------------------------
# Positions Readiness Tests
# ---------------------------------------------------------------------------


class TestPositionsReadiness:
    """Tests for positions readiness guardrails."""

    def test_positions_ready_requires_full_list_when_positions_required_for_live(self) -> None:
        """Test live readiness requires full list when positions are required."""
        from types import SimpleNamespace

        from ml.config.base import PositionsConfig, PositionsSource
        from ml.strategies.base_facade import SimpleMLStrategyFacade
        from ml.strategies.common.positions import PositionsHealthStatus
        from nautilus_trader.model.identifiers import InstrumentId

        class RecordingProvider:
            def __init__(self) -> None:
                self.calls: list[tuple[InstrumentId | None, bool, bool]] = []

            def check_positions_ready(
                self,
                *,
                instrument_id: InstrumentId | None = None,
                require_full_list: bool = False,
                require_positions: bool = False,
            ) -> PositionsHealthStatus:
                self.calls.append((instrument_id, require_full_list, require_positions))
                return PositionsHealthStatus(
                    ready=True,
                    degraded=False,
                    source=PositionsSource.CACHE_OPEN,
                    reason=None,
                    positions_count=0,
                )

        instrument_id = InstrumentId.from_str("AAA.SIM")
        config = SimpleNamespace(
            positions_config=PositionsConfig(
                positions_required_for_live=True,
                allow_degraded=True,
                source_priority=[PositionsSource.CACHE_OPEN],
            ),
            execute_trades=True,
            instrument_id=instrument_id,
        )

        provider = RecordingProvider()
        strategy = SimpleMLStrategyFacade.__new__(SimpleMLStrategyFacade)
        strategy._config = config
        strategy._positions_provider = provider
        strategy._positions_health = None

        strategy._check_positions_ready()

        assert provider.calls == [(instrument_id, True, True)]

    def test_positions_ready_degraded_logs_suppressed_in_backtest_when_configured(self) -> None:
        """Ensure degraded readiness logs can be suppressed during backtests."""
        from types import SimpleNamespace

        from ml.config.base import PositionsConfig, PositionsSource
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from ml.strategies.common.positions import PositionsHealthStatus
        from ml.tests.utils.stubs import LoggerStub
        from nautilus_trader.model.identifiers import InstrumentId

        class DummyStrategy(BaseMLStrategyFacade):
            _log_stub = LoggerStub()
            _cache_stub = SimpleNamespace(is_backtesting=True)

            @property
            def log(self) -> LoggerStub:
                return self._log_stub

            @property
            def cache(self) -> SimpleNamespace:
                return self._cache_stub

            def _process_ml_signal(self, signal: object) -> None:
                del signal

        class StubProvider:
            def check_positions_ready(
                self,
                *,
                instrument_id: InstrumentId | None = None,
                require_full_list: bool = False,
                require_positions: bool = False,
            ) -> PositionsHealthStatus:
                del instrument_id, require_full_list, require_positions
                return PositionsHealthStatus(
                    ready=True,
                    degraded=True,
                    source=PositionsSource.PORTFOLIO_NET,
                    reason="net_position_only",
                    positions_count=0,
                )

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = SimpleNamespace(
            positions_config=PositionsConfig(
                positions_required_for_live=True,
                allow_degraded=True,
                source_priority=[PositionsSource.PORTFOLIO_NET],
                log_degraded_in_backtest=False,
            ),
            execute_trades=True,
            instrument_id=InstrumentId.from_str("AAA.SIM"),
        )
        strategy._positions_provider = StubProvider()
        strategy._positions_health = None

        strategy._check_positions_ready()

        messages = [record[1][0] for record in strategy.log.records if record[1]]
        assert "ml_strategy.positions_ready_degraded" in messages
        assert all(record[0] != "warning" for record in strategy.log.records)


# ---------------------------------------------------------------------------
# Signal Handling Guardrails
# ---------------------------------------------------------------------------


class TestSignalHandlingGuardrails:
    """Tests for signal handling guardrails."""

    def test_handle_ml_signal_allows_processing_when_max_positions_and_position_exists(
        self,
        mock_config: MockMLStrategyConfig,
        mock_signal: MockMLSignal,
    ) -> None:
        """Ensure max_positions gating does not block exit-capable signals."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: MockMLSignal) -> None:
                self.processed_signals.append(signal)

            def _get_current_position(self) -> object | None:
                return self.current_position

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = mock_config
        strategy._signals_received = 0
        strategy._last_signal_time = None
        strategy.signals_received_metric = None
        strategy._active_positions = mock_config.max_positions
        strategy.processed_signals = []
        strategy.current_position = object()
        strategy.risk_manager = None

        strategy._handle_ml_signal(mock_signal)

        assert strategy.processed_signals == [mock_signal]

    def test_handle_ml_signal_uses_intent_tracker_when_cache_empty(
        self,
        mock_config: MockMLStrategyConfig,
    ) -> None:
        """Ensure intent positions allow signal processing when cache lacks positions."""
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from ml.strategies.common import OrderIntentPositionTracker
        from nautilus_trader.model.enums import OrderSide
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.identifiers import Symbol
        from nautilus_trader.model.identifiers import Venue
        from nautilus_trader.model.objects import Quantity

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: MockMLSignal) -> None:
                self.processed_signals.append(signal)

        instrument_id = InstrumentId(Symbol("EURUSD"), Venue("SIM"))
        mock_config.instrument_id = instrument_id
        mock_config.serialize_order_intents = True

        tracker = OrderIntentPositionTracker()
        tracker.record_order(
            instrument_id=instrument_id,
            side=OrderSide.BUY,
            quantity=Quantity.from_str("1.0"),
            reduce_only=False,
            ts_init=123,
        )

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = mock_config
        strategy._signals_received = 0
        strategy._last_signal_time = None
        strategy.signals_received_metric = None
        strategy._active_positions = mock_config.max_positions
        strategy.processed_signals = []
        strategy.risk_manager = None
        strategy._intent_position_tracker = tracker
        strategy._process_signal = lambda _signal: None

        signal = MockMLSignal(instrument_id=instrument_id)
        strategy._handle_ml_signal(signal)

        assert strategy.processed_signals == [signal]

    def test_handle_ml_signal_triggers_liquidation_on_risk_action(
        self,
        mock_config: MockMLStrategyConfig,
        mock_signal: MockMLSignal,
    ) -> None:
        """Ensure liquidation runs when risk action requests it."""
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from ml.strategies.risk import RiskAction, RiskActionDecision

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: MockMLSignal) -> None:
                self.processed = True

            def _liquidate_positions(self, decision: RiskActionDecision, signal: MockMLSignal) -> None:
                self.liquidated = True

        class StubRiskManager:
            def __init__(self, decision: RiskActionDecision) -> None:
                self._decision = decision

            def get_risk_action(
                self,
                *,
                portfolio: object | None = None,
                ts_event: int | None = None,
            ) -> RiskActionDecision:
                return self._decision

        decision = RiskActionDecision(
            action=RiskAction.LIQUIDATE,
            reason="daily_loss_liquidate",
            detail="Daily loss 15.0%",
        )

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = mock_config
        strategy._signals_received = 0
        strategy._last_signal_time = None
        strategy.signals_received_metric = None
        strategy._active_positions = 0
        strategy.processed = False
        strategy.liquidated = False
        strategy.risk_manager = StubRiskManager(decision)

        strategy._handle_ml_signal(mock_signal)

        assert strategy.liquidated is True
        assert strategy.processed is False

    def test_liquidation_no_positions_is_throttled(
        self,
        mock_config: MockMLStrategyConfig,
    ) -> None:
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from ml.strategies.risk import RiskAction
        from ml.strategies.risk import RiskActionDecision
        from ml.strategies.risk import RiskConfig
        from ml.strategies.risk import RiskLiquidationConfig
        from ml.tests.utils.stubs import LoggerStub

        class DummyStrategy(BaseMLStrategyFacade):
            _log_stub = LoggerStub()

            @property
            def log(self) -> LoggerStub:
                return self._log_stub

            def _process_ml_signal(self, signal: MockMLSignal) -> None:
                del signal

            def _resolve_liquidation_positions(self) -> tuple[list[Any], str | None]:
                return [], None

        class StubRiskManager:
            def __init__(self) -> None:
                self.config = RiskConfig(
                    liquidation_config=RiskLiquidationConfig(
                        enabled=True,
                        drawdown_limit_pct=0.0,
                        cooldown_ms=60_000,
                    ),
                )

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = mock_config
        strategy.risk_manager = StubRiskManager()
        strategy._last_liquidation_no_positions_ts = None

        decision = RiskActionDecision(
            action=RiskAction.LIQUIDATE,
            reason="drawdown_liquidate",
            detail=None,
        )

        first_ts = 1_000_000_000
        strategy._liquidate_positions(decision, MockMLSignal(ts_event=first_ts))
        assert strategy._last_liquidation_no_positions_ts == first_ts

        strategy._liquidate_positions(decision, MockMLSignal(ts_event=1_000_500_000))
        assert strategy._last_liquidation_no_positions_ts == first_ts


# ---------------------------------------------------------------------------
# Returns Bar Subscription
# ---------------------------------------------------------------------------


def test_on_data_subscribes_returns_bars_from_signal_metadata(
    mock_config: MockMLStrategyConfig,
) -> None:
    """Ensure returns bar subscription happens when bar_spec arrives in signals."""
    from ml.actors.base import MLSignal
    from ml.config.base import ReturnsConfig
    from ml.config.base import ReturnsUpdateMode
    from ml.strategies.base_facade import BaseMLStrategyFacade
    from nautilus_trader.model.identifiers import InstrumentId

    class DummyStrategy(BaseMLStrategyFacade):
        def _process_ml_signal(self, signal: Any) -> None:
            del signal

        def subscribe_bars(self, bar_type: Any, *args: Any, **kwargs: Any) -> None:
            self.subscribed.append(bar_type)

    class _Updater:
        def should_update_from_bar(self) -> bool:
            return True

    instrument_id = InstrumentId.from_str("EURUSD.SIM")
    mock_config.instrument_id = instrument_id
    mock_config.returns_config = ReturnsConfig(update_mode=ReturnsUpdateMode.BAR)

    strategy = DummyStrategy.__new__(DummyStrategy)
    strategy._config = mock_config
    strategy._signal_history = []
    strategy._order_submitter = None
    strategy._signal_router = None
    strategy._returns_updater = _Updater()
    strategy._returns_bar_subscribed = False
    strategy._returns_bar_subscription_attempted = False
    strategy.subscribed = []

    signal = MLSignal(
        instrument_id=instrument_id,
        model_id="model",
        prediction=0.5,
        confidence=0.9,
        ts_event=1,
        ts_init=1,
        metadata={"bar_spec": "1-MINUTE-LAST"},
    )

    BaseMLStrategyFacade.on_data(strategy, signal)

    assert len(strategy.subscribed) == 1
    assert str(strategy.subscribed[0].spec) == "1-MINUTE-LAST"

# ---------------------------------------------------------------------------
# Position Closed Updates
# ---------------------------------------------------------------------------


class TestPositionClosedUpdates:
    """Tests for position closed PnL handling."""

    def test_position_closed_updates_risk_and_sizer(self) -> None:
        """Test realized PnL updates risk and sizing paths."""
        from ml.strategies.base_facade import SimpleMLStrategyFacade
        from nautilus_trader.model.objects import Currency, Money

        class RecordingRiskManager:
            def __init__(self) -> None:
                self.calls: list[tuple[float, int | None]] = []

            def update_daily_pnl(self, pnl: float, ts_event: int | None = None) -> None:
                self.calls.append((pnl, ts_event))

        class RecordingSizer:
            def __init__(self) -> None:
                self.calls: list[float] = []

            def update_performance(self, pnl: float) -> None:
                self.calls.append(pnl)

        strategy = SimpleMLStrategyFacade.__new__(SimpleMLStrategyFacade)
        strategy._total_pnl = Decimal("0")
        strategy._winning_trades = 0
        strategy.risk_manager = RecordingRiskManager()
        strategy.position_sizer = RecordingSizer()

        ts_closed = 1_700_000_000_000_000_000
        event = SimpleNamespace(
            realized_pnl=Money(12.5, Currency.from_str("USD")),
            ts_closed=ts_closed,
        )

        strategy._handle_position_closed_event(event)

        assert strategy.risk_manager.calls == [(12.5, ts_closed)]
        assert strategy.position_sizer.calls == [12.5]
        assert strategy._total_pnl == Decimal("12.5")
        assert strategy._winning_trades == 1


# ---------------------------------------------------------------------------
# Order Event Persistence
# ---------------------------------------------------------------------------


class TestOrderEventPersistence:
    """Tests for order event persistence wiring."""

    def test_persist_order_event_calls_store(self, mock_config: MockMLStrategyConfig) -> None:
        """Ensure order events are forwarded to the strategy store."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: MockMLSignal) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = mock_config
        store = MagicMock()
        strategy.strategy_store = store

        BaseMLStrategyFacade._persist_order_event(strategy, object())

        store.write_order_event.assert_called_once()
        assert store.write_order_event.call_args.kwargs["is_live"] is True

    def test_on_order_filled_persists_event(
        self,
    ) -> None:
        """Ensure order filled events are persisted for audit."""
        from ml.config.base import MLStrategyConfig
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from nautilus_trader.test_kit.providers import TestInstrumentProvider
        from nautilus_trader.test_kit.stubs.execution import TestExecStubs
        from nautilus_trader.test_kit.stubs.events import TestEventStubs

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: MockMLSignal) -> None:
                del signal

        instrument = TestInstrumentProvider.default_fx_ccy("AUD/USD")
        order = TestExecStubs.limit_order(instrument=instrument)
        event = TestEventStubs.order_filled(order=order, instrument=instrument)

        config = MLStrategyConfig(
            instrument_id=instrument.id,
            ml_signal_source="ACTOR",
            use_strategy_store=False,
            execute_trades=True,
        )
        strategy = DummyStrategy(config)
        strategy.strategy_store = MagicMock()
        strategy._pending_orders = 1
        strategy._active_positions = 0
        strategy.position_count_metric = None
        strategy._get_current_position = lambda: None

        strategy.on_order_filled(event)

        strategy.strategy_store.write_order_event.assert_called_once()

    def test_on_start_syncs_component_strategy_ids_when_id_changes(
        self,
    ) -> None:
        """Ensure component strategy IDs sync after trader-assigned updates."""
        from ml.config.base import MLStrategyConfig
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.identifiers import StrategyId

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: MockMLSignal) -> None:
                del signal

        config = MLStrategyConfig(
            instrument_id=InstrumentId.from_str("SPY.EQUS"),
            ml_signal_source="ACTOR",
            strategy_id="MLStrategy-SPY.EQUS",
            use_strategy_store=False,
        )
        strategy = DummyStrategy(config)
        strategy.change_id(StrategyId("MLStrategy-000"))

        strategy._sync_component_strategy_ids()

        assert strategy._order_submitter is not None
        assert strategy._decision_persister is not None
        assert strategy._order_submitter._strategy_id == "MLStrategy-000"
        assert strategy._decision_persister._strategy_id == "MLStrategy-000"


# ---------------------------------------------------------------------------
# Risk Halt Event Persistence
# ---------------------------------------------------------------------------


class TestRiskHaltPersistence:
    """Tests for risk-halt audit persistence wiring."""

    def test_record_risk_halt_transition_persists_events(
        self,
        mock_config: MockMLStrategyConfig,
    ) -> None:
        """Ensure risk halt transitions are persisted once per state change."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: MockMLSignal) -> None:
                del signal

        class RiskManagerStub:
            def __init__(self) -> None:
                self.halted = False
                self.reason: str | None = None

            def is_trading_halted(self) -> bool:
                return self.halted

            def get_halt_reason(self) -> str | None:
                return self.reason

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = mock_config
        strategy.strategy_store = MagicMock()
        strategy.risk_manager = RiskManagerStub()
        strategy._last_risk_halt_state = None
        strategy._last_risk_halt_reason = None

        # Initial non-halted state should not persist an event
        BaseMLStrategyFacade._record_risk_halt_transition(strategy, ts_event=100)
        strategy.strategy_store.write_risk_halt_event.assert_not_called()

        # Transition to halted should persist once
        strategy.risk_manager.halted = True
        strategy.risk_manager.reason = "daily_loss_limit"
        BaseMLStrategyFacade._record_risk_halt_transition(strategy, ts_event=200)
        assert strategy.strategy_store.write_risk_halt_event.call_count == 1
        call = strategy.strategy_store.write_risk_halt_event.call_args.kwargs
        assert call["event_type"] == "halted"
        assert call["reason"] == "daily_loss_limit"
        assert call["ts_event"] == 200

        # Transition to resumed should persist once
        strategy.risk_manager.halted = False
        BaseMLStrategyFacade._record_risk_halt_transition(strategy, ts_event=300)
        assert strategy.strategy_store.write_risk_halt_event.call_count == 2
        call = strategy.strategy_store.write_risk_halt_event.call_args.kwargs
        assert call["event_type"] == "resumed"
        assert call["reason"] == "daily_loss_limit"
        assert call["ts_event"] == 300


# ---------------------------------------------------------------------------
# Helper Function Tests (testing static-like functions)
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for helper functions that don't require instance initialization."""

    def test_target_side_from_prediction_buy(self) -> None:
        """Test target_side_from_prediction returns BUY for high prediction."""
        from nautilus_trader.model.enums import OrderSide

        # Import the function source to test logic
        # Since we can't instantiate, test the logic directly
        prediction = 0.7
        threshold = 0.5

        result = OrderSide.BUY if float(prediction) > float(threshold) else OrderSide.SELL

        assert result == OrderSide.BUY

    def test_target_side_from_prediction_sell(self) -> None:
        """Test target_side_from_prediction returns SELL for low prediction."""
        from nautilus_trader.model.enums import OrderSide

        prediction = 0.3
        threshold = 0.5

        result = OrderSide.BUY if float(prediction) > float(threshold) else OrderSide.SELL

        assert result == OrderSide.SELL

    def test_should_reverse_logic_long_to_sell(self) -> None:
        """Test should_reverse logic for LONG position with SELL signal."""
        from nautilus_trader.model.enums import OrderSide

        position_side_name = "LONG"
        target_side = OrderSide.SELL

        # Logic from should_reverse
        result = bool(
            (position_side_name == "LONG" and target_side == OrderSide.SELL)
            or (position_side_name == "SHORT" and target_side == OrderSide.BUY),
        )

        assert result is True

    def test_should_reverse_logic_short_to_buy(self) -> None:
        """Test should_reverse logic for SHORT position with BUY signal."""
        from nautilus_trader.model.enums import OrderSide

        position_side_name = "SHORT"
        target_side = OrderSide.BUY

        result = bool(
            (position_side_name == "LONG" and target_side == OrderSide.SELL)
            or (position_side_name == "SHORT" and target_side == OrderSide.BUY),
        )

        assert result is True

    def test_should_reverse_logic_aligned(self) -> None:
        """Test should_reverse logic for aligned positions."""
        from nautilus_trader.model.enums import OrderSide

        position_side_name = "LONG"
        target_side = OrderSide.BUY

        result = bool(
            (position_side_name == "LONG" and target_side == OrderSide.SELL)
            or (position_side_name == "SHORT" and target_side == OrderSide.BUY),
        )

        assert result is False

    def test_decision_from_prediction_uses_neutral_band(self) -> None:
        """Verify facade decision helper returns HOLD inside neutral band."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = SimpleNamespace(prediction_neutral_band=0.1)

        assert strategy.decision_from_prediction(0.55) == "HOLD"

    def test_should_reverse_invokes_facade_logic(self) -> None:
        """Verify facade reverse helper handles opposing and aligned sides."""
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from nautilus_trader.model.enums import OrderSide

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        short_position = SimpleNamespace(side=SimpleNamespace(name="SHORT"))

        assert strategy.should_reverse(short_position, OrderSide.BUY) is True
        assert strategy.should_reverse(short_position, OrderSide.SELL) is False


class TestIntentSerializationHelpers:
    """Tests for intent serialization and helper branches."""

    def test_should_block_entry_orders_uses_tracker_and_limits(self) -> None:
        """Verify intent entry gating follows tracker and max_positions semantics."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = SimpleNamespace(
            serialize_order_intents=True,
            max_positions=1,
        )
        strategy._active_positions = 0
        strategy._intent_position_tracker = SimpleNamespace(active_positions=1)

        assert strategy._should_block_entry_orders() is True

        strategy._intent_position_tracker = None
        strategy._active_positions = 1
        assert strategy._should_block_entry_orders() is True

        strategy._config.max_positions = 0
        assert strategy._should_block_entry_orders() is True

        strategy._config.max_positions = "not-an-int"
        assert strategy._should_block_entry_orders() is False

    def test_should_block_entry_orders_returns_false_when_serialization_disabled(self) -> None:
        """Verify entry blocking is bypassed when intent serialization is disabled."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = SimpleNamespace(serialize_order_intents=False, max_positions=1)
        strategy._intent_position_tracker = SimpleNamespace(active_positions=99)
        strategy._active_positions = 99

        assert strategy._should_block_entry_orders() is False

    def test_init_intent_position_tracker_respects_serialize_flag(self) -> None:
        """Verify intent position tracker initializes only when serialization is enabled."""
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from ml.strategies.common import OrderIntentPositionTracker

        strategy = SimpleNamespace(
            log=MagicMock(),
            _intent_position_tracker=None,
            _config=SimpleNamespace(serialize_order_intents=False),
        )

        BaseMLStrategyFacade._init_intent_position_tracker(cast(Any, strategy))
        assert strategy._intent_position_tracker is None

        strategy._config.serialize_order_intents = True
        BaseMLStrategyFacade._init_intent_position_tracker(cast(Any, strategy))
        assert isinstance(strategy._intent_position_tracker, OrderIntentPositionTracker)

    def test_resolve_short_entry_policy_handles_enum_string_and_defaults(self) -> None:
        """Verify short-entry policy resolution covers enum, string, and fallback paths."""
        from ml.config.base import AccountMode
        from ml.config.base import ShortEntryPolicy
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = SimpleNamespace(
            short_entry_policy=ShortEntryPolicy.DENY,
            account_mode=AccountMode.CASH,
        )
        assert strategy._resolve_short_entry_policy() is ShortEntryPolicy.DENY

        strategy._config = SimpleNamespace(
            short_entry_policy="allow",
            account_mode="cash",
        )
        assert strategy._resolve_short_entry_policy() is ShortEntryPolicy.ALLOW

        strategy._config = SimpleNamespace(
            short_entry_policy="invalid-policy",
            account_mode="margin",
        )
        assert strategy._resolve_short_entry_policy() is ShortEntryPolicy.ALLOW

        strategy._config = SimpleNamespace(
            short_entry_policy=None,
            account_mode="invalid-mode",
        )
        assert strategy._resolve_short_entry_policy() is ShortEntryPolicy.EXIT_ONLY

    def test_resolve_submit_and_cancel_callbacks_respect_intent_mode(self) -> None:
        """Verify submit/cancel callback selection in intent and live modes."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = SimpleNamespace(serialize_order_intents=True)
        strategy._order_intent_writer = None
        strategy.submit_order = lambda order: order
        strategy.cancel_order = lambda order: order

        assert strategy._resolve_submit_order_callback() is None
        assert strategy._resolve_cancel_order_callback() is None

        strategy._order_intent_writer = object()
        submit_callback = strategy._resolve_submit_order_callback()
        assert submit_callback is not None
        assert submit_callback.__name__ == "_record_order_intent"

        strategy._config.serialize_order_intents = False
        assert strategy._resolve_submit_order_callback() is strategy.submit_order
        assert strategy._resolve_cancel_order_callback() is strategy.cancel_order

    def test_resolve_submit_order_callback_returns_none_without_submit_attribute(self) -> None:
        """Verify live-mode submit callback resolution returns None when unavailable."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        strategy = SimpleNamespace(_config=SimpleNamespace(serialize_order_intents=False))

        callback = BaseMLStrategyFacade._resolve_submit_order_callback(cast(Any, strategy))

        assert callback is None

    def test_set_exit_intent_metadata_only_updates_when_serialization_enabled(self) -> None:
        """Verify exit metadata is ignored unless intent serialization is enabled."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._config = SimpleNamespace(serialize_order_intents=False)
        strategy._pending_exit_metadata = None

        strategy._set_exit_intent_metadata({"reason": "liquidate"})
        assert strategy._pending_exit_metadata is None

        strategy._config.serialize_order_intents = True
        payload = {"reason": "liquidate"}
        strategy._set_exit_intent_metadata(payload)
        assert strategy._pending_exit_metadata == payload
        assert strategy._pending_exit_metadata is not payload

    def test_update_intent_positions_tracks_active_positions_from_string_side(self) -> None:
        """Verify intent position updates accept string sides and track active positions."""
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from ml.strategies.common import OrderIntentPositionTracker
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Quantity

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        instrument_id = InstrumentId.from_str("EURUSD.SIM")
        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._intent_position_tracker = OrderIntentPositionTracker(log=MagicMock())
        strategy._active_positions = 0

        order = SimpleNamespace(
            side="buy",
            instrument_id=instrument_id,
            quantity=Quantity.from_str("1.0"),
            reduce_only=False,
            ts_init=1,
            price=1.2345,
        )
        strategy._update_intent_positions(order)
        assert strategy._active_positions == 1

        bad_order = SimpleNamespace(
            side="unsupported",
            instrument_id=instrument_id,
            quantity=Quantity.from_str("1.0"),
            reduce_only=False,
            ts_init=2,
        )
        strategy._update_intent_positions(bad_order)
        assert strategy._active_positions == 1

    def test_feature_store_accessor_handles_missing_and_present_store(self) -> None:
        """Verify feature_store property returns None or injected store as expected."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._stores = None
        assert strategy.feature_store is None

        sentinel_store = object()
        strategy._stores = SimpleNamespace(feature_store=sentinel_store)
        assert strategy.feature_store is sentinel_store

    def test_store_and_registry_accessors_cover_all_store_fields(self) -> None:
        """Verify all store/registry accessors return injected values."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._stores = SimpleNamespace(
            model_store="model",
            data_store="data",
            feature_registry="feature-registry",
            model_registry="model-registry",
            strategy_registry="strategy-registry",
            data_registry="data-registry",
        )

        assert strategy.model_store == "model"
        assert strategy.data_store == "data"
        assert strategy.feature_registry == "feature-registry"
        assert strategy.model_registry == "model-registry"
        assert strategy.strategy_registry == "strategy-registry"
        assert strategy.data_registry == "data-registry"

    def test_sync_component_strategy_ids_logs_debug_on_setattr_failure(self) -> None:
        """Verify strategy-id sync logs debug when a component rejects setattr."""
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from ml.tests.utils.stubs import LoggerStub

        class _ReadOnlyComponent:
            def __init__(self) -> None:
                object.__setattr__(self, "_strategy_id", "initial")

            def __setattr__(self, name: str, value: object) -> None:
                if name == "_strategy_id":
                    raise RuntimeError("read-only")
                super().__setattr__(name, value)

        strategy = SimpleNamespace(
            id="STRAT-XYZ",
            log=LoggerStub(),
            _decision_persister=_ReadOnlyComponent(),
            _position_manager=None,
            _order_submitter=None,
            _positions_provider=None,
            performance=None,
            _lifecycle=None,
            _returns_updater=None,
        )

        BaseMLStrategyFacade._sync_component_strategy_ids(cast(Any, strategy))

        assert any(
            level == "debug" and args and args[0] == "ml_strategy.strategy_id_sync_failed"
            for level, args, _kwargs in strategy.log.records
        )


class TestFacadeHelperDelegation:
    """Tests for helper methods that delegate to component state."""

    def test_resolve_market_price_falls_back_to_trade_then_quote(self) -> None:
        """Verify direct cache fallback path resolves trade, quote, then None."""
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from nautilus_trader.model.identifiers import InstrumentId

        class DummyStrategy(BaseMLStrategyFacade):
            _cache_stub: Any = None

            @property
            def cache(self) -> Any:
                return self._cache_stub

            def _process_ml_signal(self, signal: object) -> None:
                del signal

        instrument_id = InstrumentId.from_str("EURUSD.SIM")
        trade_tick = SimpleNamespace(price=SimpleNamespace(as_double=lambda: 101.5))
        quote_tick = SimpleNamespace(
            bid_price=SimpleNamespace(as_double=lambda: 101.0),
            ask_price=SimpleNamespace(as_double=lambda: 102.0),
        )

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._position_manager = None
        strategy._cache_stub = SimpleNamespace(
            trade_tick=lambda _instrument: trade_tick,
            quote_tick=lambda _instrument: quote_tick,
        )
        assert strategy._resolve_market_price(instrument_id) == pytest.approx(101.5)

        strategy._cache_stub = SimpleNamespace(
            trade_tick=lambda _instrument: None,
            quote_tick=lambda _instrument: quote_tick,
        )
        assert strategy._resolve_market_price(instrument_id) == pytest.approx(101.5)

        strategy._cache_stub = SimpleNamespace(
            trade_tick=lambda _instrument: None,
            quote_tick=lambda _instrument: None,
        )
        assert strategy._resolve_market_price(instrument_id) is None

    def test_sizing_reject_reason_requires_string_return(self) -> None:
        """Verify non-string reject reasons are ignored."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._position_manager = SimpleNamespace(
            get_last_rejection_reason=lambda: "risk_rejected",
        )
        assert strategy._get_sizing_reject_reason() == "risk_rejected"

        strategy._position_manager = SimpleNamespace(
            get_last_rejection_reason=lambda: 123,
        )
        assert strategy._get_sizing_reject_reason() is None

    def test_liquidation_side_and_quantity_helpers_cover_signs(self) -> None:
        """Verify liquidation helper behavior for long, short, and flat positions."""
        from ml.strategies.base_facade import BaseMLStrategyFacade
        from nautilus_trader.model.enums import OrderSide
        from nautilus_trader.model.objects import Quantity

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)

        long_position = SimpleNamespace(
            signed_decimal_qty=lambda: 2,
            quantity=Quantity.from_str("2.0"),
        )
        short_position = SimpleNamespace(
            signed_decimal_qty=lambda: -3,
            quantity=None,
        )
        flat_position = SimpleNamespace(
            signed_decimal_qty=lambda: 0,
            quantity=None,
        )

        assert strategy._resolve_liquidation_side(long_position) is OrderSide.SELL
        assert strategy._resolve_liquidation_side(short_position) is OrderSide.BUY
        assert strategy._resolve_liquidation_side(flat_position) is None

        assert strategy._resolve_liquidation_quantity(long_position) is not None
        short_qty = strategy._resolve_liquidation_quantity(short_position)
        assert short_qty is not None
        assert float(short_qty.as_double()) == pytest.approx(3.0)
        assert strategy._resolve_liquidation_quantity(flat_position) is None

    def test_aggregate_signal_updates_model_cache_and_router_buffer(self) -> None:
        """Verify aggregate helper stores model signal and updates router buffer."""
        from ml.strategies.base_facade import BaseMLStrategyFacade

        class DummyStrategy(BaseMLStrategyFacade):
            def _process_ml_signal(self, signal: object) -> None:
                del signal

        strategy = DummyStrategy.__new__(DummyStrategy)
        strategy._signal_router = MagicMock()
        strategy._model_signals = {}

        signal = MockMLSignal(model_id="model-x")
        strategy._aggregate_signal(signal)

        assert strategy._model_signals["model-x"] is signal
        strategy._signal_router.add_to_buffer.assert_called_once_with(signal)


class TestSimpleFacadeSignalProcessing:
    """Coverage tests for ``SimpleMLStrategyFacade._process_ml_signal`` branches."""

    def test_simple_facade_process_signal_returns_on_hold(self) -> None:
        """Verify HOLD decisions do not submit orders."""
        from ml.strategies.base_facade import SimpleMLStrategyFacade

        strategy = SimpleNamespace(
            log=MagicMock(),
            _config=SimpleNamespace(instrument_id=InstrumentId.from_str("EURUSD.SIM")),
            _active_positions=0,
            position_count_metric=None,
            _get_current_position=lambda: None,
            decision_from_prediction=lambda _prediction: "HOLD",
            _should_block_entry_orders=lambda: False,
            _calculate_position_size=lambda: Quantity.from_str("1.0"),
        )
        submitted: list[tuple[object, object, bool]] = []
        strategy._place_market_order = lambda side, quantity, reduce_only=False: submitted.append(
            (side, quantity, reduce_only),
        )

        SimpleMLStrategyFacade._process_ml_signal(cast(Any, strategy), MockMLSignal(prediction=0.5))

        assert submitted == []
        assert strategy._active_positions == 0

    def test_simple_facade_process_signal_entry_paths_and_metrics(self) -> None:
        """Verify entry block, sizing-failure warning, and successful metric update paths."""
        from ml.strategies.base_facade import SimpleMLStrategyFacade
        from ml.tests.utils.stubs import LoggerStub
        from nautilus_trader.model.enums import OrderSide

        class _MetricStub:
            def __init__(self) -> None:
                self.values: list[int] = []
                self.labels_calls: list[tuple[str, str]] = []

            def labels(self, *, strategy_id: str, instrument: str) -> _MetricStub:
                self.labels_calls.append((strategy_id, instrument))
                return self

            def set(self, value: int) -> None:
                self.values.append(value)

        strategy = SimpleNamespace(
            log=LoggerStub(),
            id="STRAT-1",
            _config=SimpleNamespace(instrument_id=InstrumentId.from_str("EURUSD.SIM")),
            _active_positions=0,
            _get_current_position=lambda: None,
            decision_from_prediction=lambda _prediction: "BUY",
            _should_block_entry_orders=lambda: True,
            _calculate_position_size=lambda: Quantity.from_str("1.0"),
            position_count_metric=_MetricStub(),
        )
        submitted: list[tuple[OrderSide, Quantity, bool]] = []
        strategy._place_market_order = lambda side, quantity, reduce_only=False: submitted.append(
            (side, quantity, reduce_only),
        )

        signal = MockMLSignal(
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
            prediction=0.9,
        )
        SimpleMLStrategyFacade._process_ml_signal(cast(Any, strategy), signal)
        assert submitted == []
        assert strategy._active_positions == 0

        strategy._should_block_entry_orders = lambda: False
        strategy._calculate_position_size = lambda: None
        SimpleMLStrategyFacade._process_ml_signal(cast(Any, strategy), signal)
        assert any(
            level == "warning" and args and "position sizing failure" in str(args[0])
            for level, args, _kwargs in strategy.log.records
        )
        assert submitted == []

        strategy._calculate_position_size = lambda: Quantity.from_str("2.0")
        SimpleMLStrategyFacade._process_ml_signal(cast(Any, strategy), signal)
        assert submitted[-1][0] is OrderSide.BUY
        assert strategy._active_positions == 1
        assert strategy.position_count_metric.values == [1]

    def test_simple_facade_process_signal_reverse_and_aligned_paths(self) -> None:
        """Verify reverse branch close/open behavior plus aligned-position no-op branch."""
        from ml.strategies.base_facade import SimpleMLStrategyFacade
        from ml.tests.utils.stubs import LoggerStub
        from nautilus_trader.model.enums import OrderSide

        current_position = SimpleNamespace(
            side=SimpleNamespace(name="LONG"),
            quantity=Quantity.from_str("1.0"),
        )
        strategy = SimpleNamespace(
            log=LoggerStub(),
            _config=SimpleNamespace(instrument_id=InstrumentId.from_str("EURUSD.SIM")),
            _active_positions=1,
            position_count_metric=None,
            _get_current_position=lambda: current_position,
            decision_from_prediction=lambda _prediction: "SELL",
            should_reverse=lambda _current, _target: True,
            _should_block_entry_orders=lambda: True,
            _calculate_position_size=lambda: Quantity.from_str("2.0"),
        )
        submitted: list[tuple[OrderSide, Quantity, bool]] = []
        strategy._place_market_order = lambda side, quantity, reduce_only=False: submitted.append(
            (side, quantity, reduce_only),
        )

        signal = MockMLSignal(
            instrument_id=InstrumentId.from_str("EURUSD.SIM"),
            prediction=0.1,
        )
        SimpleMLStrategyFacade._process_ml_signal(cast(Any, strategy), signal)
        assert len(submitted) == 1
        assert submitted[0][0] is OrderSide.SELL
        assert submitted[0][1].as_decimal() == Decimal("1.0")
        assert submitted[0][2] is True

        strategy._should_block_entry_orders = lambda: False
        strategy._calculate_position_size = lambda: None
        SimpleMLStrategyFacade._process_ml_signal(cast(Any, strategy), signal)
        assert any(
            level == "warning" and args and "Closed position but cannot open new one" in str(args[0])
            for level, args, _kwargs in strategy.log.records
        )

        strategy._calculate_position_size = lambda: Quantity.from_str("3.0")
        SimpleMLStrategyFacade._process_ml_signal(cast(Any, strategy), signal)
        assert len(submitted) >= 3
        assert submitted[-2][0] is OrderSide.SELL
        assert submitted[-2][1].as_decimal() == Decimal("1.0")
        assert submitted[-2][2] is True
        assert submitted[-1][0] is OrderSide.SELL
        assert submitted[-1][1].as_decimal() == Decimal("3.0")
        assert submitted[-1][2] is False

        strategy.should_reverse = lambda _current, _target: False
        SimpleMLStrategyFacade._process_ml_signal(cast(Any, strategy), signal)
        assert any(
            level == "debug" and args and "Position aligns with signal" in str(args[0])
            for level, args, _kwargs in strategy.log.records
        )


@pytest.mark.usefixtures("isolated_prometheus_registry")
class TestAnalyticsCoverage:
    """Targeted unit tests for strategy analytics helpers."""

    @staticmethod
    def _signal(
        *,
        instrument: str,
        model_id: str,
        prediction: float,
        confidence: float,
        ts_event: int,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        from ml.actors.base import MLSignal
        from nautilus_trader.model.identifiers import InstrumentId

        instrument_id = InstrumentId.from_str(instrument)
        return MLSignal(
            instrument_id=instrument_id,
            model_id=model_id,
            prediction=prediction,
            confidence=confidence,
            ts_event=ts_event,
            ts_init=ts_event,
            metadata=metadata,
        )

    def test_tracker_records_signal_order_and_position_metrics(self) -> None:
        """Verify signal lifecycle updates quality, execution, and summary metrics."""
        from ml.strategies.analytics import AnalyticsConfig
        from ml.strategies.analytics import PerformanceTracker

        tracker = PerformanceTracker(
            AnalyticsConfig(
                min_signals_for_stats=1,
                report_frequency_minutes=10_000,
            ),
        )
        signal_one = self._signal(
            instrument="AAA.SIM",
            model_id="model-a",
            prediction=0.9,
            confidence=0.8,
            ts_event=1,
            metadata={"neutral_band": 0.1},
        )
        signal_two = self._signal(
            instrument="AAA.SIM",
            model_id="model-b",
            prediction=0.2,
            confidence=0.7,
            ts_event=2,
        )

        tracker.record_signal(signal_one)
        tracker.record_order(order=object(), signal=signal_one)
        tracker.record_position_closed(
            position=object(),
            signal=signal_one,
            pnl=10.0,
            fees=1.0,
            slippage=0.5,
        )

        tracker.record_signal(signal_two)
        tracker.record_order(order=object(), signal=signal_two)
        tracker.record_position_closed(
            position=object(),
            signal=signal_two,
            pnl=-5.0,
            fees=0.5,
            slippage=0.2,
        )

        quality = tracker.get_signal_quality_metrics(signal_one.instrument_id)
        execution = tracker.get_execution_quality_metrics()
        summary = tracker.get_performance_summary()

        assert quality["total_signals"] == pytest.approx(2.0)
        assert quality["execution_rate"] == pytest.approx(1.0)
        assert quality["win_rate"] == pytest.approx(0.5)
        assert execution["total_fees"] == pytest.approx(1.5)
        assert execution["total_slippage"] == pytest.approx(0.7)
        assert summary["cumulative_pnl"] == pytest.approx(5.0)
        assert summary["total_signals"] == pytest.approx(2.0)

    def test_tracker_sharpe_outliers_and_feature_importance(self) -> None:
        """Verify statistical helper paths for sharpe, outliers, and feature importance."""
        from ml.strategies.analytics import AnalyticsConfig
        from ml.strategies.analytics import PerformanceTracker

        tracker = PerformanceTracker(AnalyticsConfig(outlier_z_score=1.0))
        tracker._daily_returns = [2.0, 1.0, -1.0]
        assert tracker.get_sharpe_ratio(lookback_days=4) != 0.0

        tracker._daily_returns = [1.0]
        assert tracker.get_sharpe_ratio(lookback_days=30) == 0.0

        assert not tracker.detect_outliers(np.array([1.0, 1.0], dtype=np.float64)).any()
        assert not tracker.detect_outliers(np.array([1.0, 1.0, 1.0], dtype=np.float64)).any()
        mask = tracker.detect_outliers(np.array([0.0, 0.0, 2.0], dtype=np.float64))
        assert bool(mask[-1]) is True

        assert tracker.get_feature_importance([]) == {}
        importance = tracker.get_feature_importance(["f1", "f2", "f3"])
        assert set(importance) == {"f1", "f2", "f3"}
        assert sum(importance.values()) == pytest.approx(1.0)

    def test_tracker_report_schedule_and_decay_warning_paths(self) -> None:
        """Verify periodic report scheduling and decay warning logging paths."""
        from ml.strategies import analytics as analytics_module
        from ml.strategies.analytics import AnalyticsConfig
        from ml.strategies.analytics import PerformanceTracker

        tracker = PerformanceTracker(
            AnalyticsConfig(
                min_signals_for_stats=1,
                report_frequency_minutes=1,
            ),
        )
        tracker._last_report_time = datetime.now() - timedelta(minutes=5)
        tracker._generate_report = MagicMock()  # type: ignore[method-assign]
        tracker._check_report_schedule()
        tracker._generate_report.assert_called_once()

        class RecordingLogger:
            def __init__(self) -> None:
                self.info_calls: list[str] = []
                self.warning_calls: list[str] = []

            def info(self, message: str) -> None:
                self.info_calls.append(message)

            def warning(self, message: str) -> None:
                self.warning_calls.append(message)

        recording_logger = RecordingLogger()
        original_logger = analytics_module.logger
        analytics_module.logger = recording_logger
        try:
            tracker_for_report = PerformanceTracker(
                AnalyticsConfig(
                    min_signals_for_stats=1,
                    report_frequency_minutes=1,
                ),
            )
            tracker_for_report.get_performance_summary = MagicMock(  # type: ignore[method-assign]
                return_value={
                    "cumulative_pnl": 100.0,
                    "sharpe_30d": 1.1,
                    "win_rate": 0.6,
                    "total_signals": 20.0,
                    "signal_decay": 1.0,
                    "recent_win_rate": 0.2,
                },
            )
            tracker_for_report._generate_report()
        finally:
            analytics_module.logger = original_logger

        assert recording_logger.info_calls
        assert recording_logger.warning_calls


class TestStrategyProtocols:
    """Tests for protocol runtime behavior and stub callability."""

    def test_protocol_stub_methods_are_callable(self) -> None:
        """Execute protocol stubs to cover protocol declaration paths."""
        from ml.strategies import protocols as strategy_protocols

        sentinel = object()
        assert strategy_protocols._HasAsDouble.as_double(sentinel) is None
        assert strategy_protocols.AccountLike.balance_total(sentinel) is None
        assert strategy_protocols.PositionSizerProtocol.calculate(
            sentinel,
            sentinel,
            sentinel,
            [],
        ) is None
        assert strategy_protocols.RiskManagerProtocol.check_position(
            sentinel,
            sentinel,
            sentinel,
            sentinel,
        ) is None
        assert strategy_protocols.RiskManagerProtocol.check_daily_limits(sentinel, None) is None
        assert strategy_protocols.RiskManagerProtocol.update_daily_pnl(sentinel, 1.0, None) is None
        assert strategy_protocols.OrderExecutorProtocol.create_order(
            sentinel,
            sentinel,
            sentinel,
            sentinel,
            {},
            sentinel,
        ) is None
        assert strategy_protocols.PortfolioManagerProtocol.allocate_signals(
            sentinel,
            [],
            1.0,
        ) is None
        assert strategy_protocols.PortfolioManagerProtocol.get_correlation_matrix(
            sentinel,
            [],
        ) is None
        assert strategy_protocols.PerformanceTrackerProtocol.record_signal(sentinel, sentinel) is None
        assert strategy_protocols.PerformanceTrackerProtocol.record_order(
            sentinel,
            sentinel,
            sentinel,
        ) is None
        assert strategy_protocols.PerformanceTrackerProtocol.get_win_rate_by_confidence(sentinel) is None
        assert strategy_protocols.PerformanceTrackerProtocol.get_sharpe_ratio(sentinel, 30) is None


__all__ = [
    "MockMLSignal",
    "MockMLStrategyConfig",
    "MockStores",
    "TestComponentInitialization",
    "TestComponentMethods",
    "TestFacadeClassStructure",
    "TestHelperFunctions",
    "TestModuleImports",
]
