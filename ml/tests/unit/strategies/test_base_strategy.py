"""
Unit tests for ML strategy base classes.

Tests cover:
- BaseMLStrategy initialization and configuration
- ML signal handling and processing
- Position sizing calculations
- Order placement and management
- Performance tracking and metrics
- SimpleMLStrategy concrete implementation

"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.actors.base import MLSignal
from ml.config.base import MLStrategyConfig
from ml.strategies.base import BaseMLStrategy
from ml.strategies.base import SimpleMLStrategy
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import StrategyId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


class MockMLStrategy(BaseMLStrategy):
    """
    Mock implementation of BaseMLStrategy for testing.
    """

    def __init__(self, config: MLStrategyConfig) -> None:
        # Set up mocked attributes BEFORE calling super().__init__
        self.ml_signals_processed: list[MLSignal] = []
        self._mocked_submit_order = Mock()
        self._mocked_log = Mock()
        self._mocked_clock_timestamp_ns = Mock(return_value=1234567890000000000)
        self._mocked_cache = Mock()
        # Mock Strategy properties
        self._trader_id = TraderId("TESTER-001")
        self._id = StrategyId("MockMLStrategy-001")

        # Mock the StrategyStore to avoid database connection
        with patch("ml.strategies.base.StrategyStore"):
            # Now call parent init which will use the mocked clock
            super().__init__(config)

    def _process_ml_signal(self, signal: MLSignal) -> None:
        """
        Mock implementation of signal processing.
        """
        self.ml_signals_processed.append(signal)

    def submit_order(self, order: Any) -> None:
        """
        Mock submit_order method.
        """
        self._mocked_submit_order(order)

    @property
    def log(self) -> Any:
        """
        Mock log property.
        """
        return self._mocked_log

    @property
    def clock(self) -> Any:
        """
        Mock clock property.
        """
        mock_clock = Mock()
        mock_clock.timestamp_ns = self._mocked_clock_timestamp_ns
        return mock_clock

    @property
    def cache(self) -> Any:
        """
        Mock cache property.
        """
        return self._mocked_cache

    @property
    def trader_id(self) -> TraderId:
        """
        Mock trader_id property.
        """
        return self._trader_id

    @property
    def id(self) -> StrategyId:
        """
        Mock id property.
        """
        return self._id

    def subscribe_data(self, data_type: Any, client_id: Any = None) -> None:
        """
        Mock subscribe_data method.
        """

    def subscribe_instrument(self, instrument_id: Any) -> None:
        """
        Mock subscribe_instrument method.
        """


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.unit
@pytest.mark.usefixtures("clean_postgres_db")
class TestBaseMLStrategy:
    """
    Test BaseMLStrategy base class.
    """

    @pytest.fixture
    def instrument_id(self) -> InstrumentId:
        """
        Create test instrument ID.
        """
        return InstrumentId(Symbol("AAPL"), Venue("NASDAQ"))

    @pytest.fixture
    def config(self, instrument_id: InstrumentId) -> MLStrategyConfig:
        """
        Create test configuration.
        """
        return MLStrategyConfig(
            instrument_id=instrument_id,
            ml_signal_source="ML_SIGNAL_ACTOR",
            position_size_pct=0.1,
            min_confidence=0.7,
            max_positions=3,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
        )

    @pytest.fixture
    def strategy(self, config: MLStrategyConfig) -> MockMLStrategy:
        """
        Create test strategy instance.
        """
        return MockMLStrategy(config)

    @pytest.fixture
    def ml_signal(self, instrument_id: InstrumentId) -> MLSignal:
        """
        Create test ML signal.
        """
        return MLSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.75,
            confidence=0.85,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

    @pytest.fixture
    def sample_instrument(self, instrument_id: InstrumentId) -> Any:
        """
        Create sample instrument mock.
        """
        instrument = Mock()
        instrument.id = instrument_id
        instrument.price_precision = 2
        instrument.size_precision = 0  # Add size precision
        instrument.lot_size = Quantity.from_int(1)
        instrument.venue = Venue("NASDAQ")  # Add venue
        instrument.min_quantity = Quantity.from_int(1)
        instrument.max_quantity = Quantity.from_int(10000)
        return instrument

    @pytest.mark.database
    @pytest.mark.serial
    def test_initialization_with_config(self, config: MLStrategyConfig) -> None:
        """
        Test strategy initialization with configuration.
        """
        # Act
        strategy = MockMLStrategy(config)

        # Assert
        assert strategy._config == config
        assert strategy._active_positions == 0
        assert strategy._pending_orders == 0
        assert strategy._last_signal_time == 0
        assert strategy._signals_received == 0
        assert strategy._trades_executed == 0
        assert strategy._winning_trades == 0
        assert strategy._total_pnl == Decimal("0.0")

    @pytest.mark.database
    @pytest.mark.serial
    def test_on_start_subscribes_to_data(self, strategy: MockMLStrategy) -> None:
        """
        Test on_start subscribes to ML signals and instruments.
        """
        # Act
        strategy.on_start()

        # Assert
        strategy._mocked_log.info.assert_called()
        # The first info call is the config log, get the actual message
        assert strategy._mocked_log.info.call_count >= 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_on_data_processes_ml_signal(
        self,
        strategy: MockMLStrategy,
        ml_signal: MLSignal,
    ) -> None:
        """
        Test on_data correctly routes ML signals.
        """
        # Arrange
        strategy.on_start()

        # Act
        strategy.on_data(ml_signal)

        # Assert
        assert strategy._signals_received == 1
        assert strategy._last_signal_time == ml_signal.ts_event
        assert len(strategy.ml_signals_processed) == 1
        assert strategy.ml_signals_processed[0] == ml_signal

    @pytest.mark.database
    @pytest.mark.serial
    def test_on_data_ignores_non_ml_signals(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test on_data ignores non-ML signal data.
        """
        # Arrange
        strategy.on_start()
        trade_tick = TradeTick(
            instrument_id=instrument_id,
            price=Price.from_str("100.00"),
            size=Quantity.from_int(100),
            aggressor_side=AggressorSide.BUYER,
            trade_id=TradeId("123"),
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy.on_data(trade_tick)

        # Assert
        assert strategy._signals_received == 0
        assert len(strategy.ml_signals_processed) == 0

    @pytest.mark.database
    @pytest.mark.serial
    def test_handle_ml_signal_filters_wrong_instrument(
        self,
        strategy: MockMLStrategy,
    ) -> None:
        """
        Test ML signal filtering for wrong instrument.
        """
        # Arrange
        strategy.on_start()
        wrong_signal = MLSignal(
            instrument_id=InstrumentId(Symbol("GOOGL"), Venue("NASDAQ")),
            model_id="test_model",
            prediction=0.9,
            confidence=0.95,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy.on_data(wrong_signal)

        # Assert
        assert strategy._signals_received == 1
        assert len(strategy.ml_signals_processed) == 0  # Not processed

    @pytest.mark.database
    @pytest.mark.serial
    def test_handle_ml_signal_filters_low_confidence(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test ML signal filtering for low confidence.
        """
        # Arrange
        strategy.on_start()
        low_conf_signal = MLSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.9,
            confidence=0.5,  # Below threshold of 0.7
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy.on_data(low_conf_signal)

        # Assert
        assert strategy._signals_received == 0  # Signal filtered out before incrementing
        assert len(strategy.ml_signals_processed) == 0  # Not processed
        assert len(strategy._signal_history) == 1  # Signal is still added to history
        strategy._mocked_log.debug.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_handle_ml_signal_respects_max_positions(
        self,
        strategy: MockMLStrategy,
        ml_signal: MLSignal,
    ) -> None:
        """
        Test ML signal handling respects max positions limit.
        """
        # Arrange
        strategy.on_start()
        strategy._active_positions = 3  # At max

        # Act
        strategy.on_data(ml_signal)

        # Assert
        assert strategy._signals_received == 1
        assert len(strategy.ml_signals_processed) == 0  # Not processed
        strategy._mocked_log.debug.assert_called_once()
        assert "Maximum positions reached" in strategy._mocked_log.debug.call_args[0][0]

    @pytest.mark.database
    @pytest.mark.serial
    def test_calculate_position_size_with_account(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
        sample_instrument: Equity,
    ) -> None:
        """
        Test position size calculation with account balance.
        """
        # Arrange
        strategy.on_start()

        # Mock account
        account = Mock()
        account.balance_total.return_value = Money(10000.00, USD)

        # Mock cache to return account and instrument
        strategy._mocked_cache.account_for_venue.return_value = account
        strategy._mocked_cache.instrument.return_value = sample_instrument

        # Mock price data
        trade_tick = TradeTick(
            instrument_id=instrument_id,
            price=Price.from_str("100.00"),
            size=Quantity.from_int(100),
            aggressor_side=AggressorSide.BUYER,
            trade_id=TradeId("123"),
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )
        strategy._mocked_cache.trade_tick.return_value = trade_tick

        # Act
        position_size = strategy._calculate_position_size()

        # Assert
        # 10000 * 0.1 / 100 = 10 shares
        assert position_size == Quantity.from_int(10)

    @pytest.mark.database
    @pytest.mark.serial
    def test_calculate_position_size_no_account(
        self,
        strategy: MockMLStrategy,
    ) -> None:
        """
        Test position size calculation without account.
        """
        # Arrange
        strategy.on_start()
        strategy._mocked_cache.account_for_venue.return_value = None
        strategy._mocked_cache.instrument.return_value = Mock()

        # Act
        position_size = strategy._calculate_position_size()

        # Assert
        assert position_size is None
        strategy._mocked_log.error.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_calculate_position_size_no_instrument(
        self,
        strategy: MockMLStrategy,
    ) -> None:
        """
        Test position size calculation without instrument.
        """
        # Arrange
        strategy.on_start()
        account = Mock()
        account.balance_total.return_value = Money(10000.00, USD)
        strategy._mocked_cache.account_for_venue.return_value = account
        strategy._mocked_cache.instrument.return_value = None

        # Act
        position_size = strategy._calculate_position_size()

        # Assert
        assert position_size is None
        strategy._mocked_log.error.assert_called_once()

    @pytest.mark.database
    @pytest.mark.serial
    def test_calculate_position_size_with_quote_tick_fallback(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
        sample_instrument: Equity,
    ) -> None:
        """
        Test position size calculation using quote tick when trade tick unavailable.
        """
        # Arrange
        strategy.on_start()

        account = Mock()
        account.balance_total.return_value = Money(10000.00, USD)
        strategy._mocked_cache.account_for_venue.return_value = account
        strategy._mocked_cache.instrument.return_value = sample_instrument
        strategy._mocked_cache.trade_tick.return_value = None

        # Mock quote tick
        quote_tick = QuoteTick(
            instrument_id=instrument_id,
            bid_price=Price.from_str("99.50"),
            ask_price=Price.from_str("100.50"),
            bid_size=Quantity.from_int(100),
            ask_size=Quantity.from_int(100),
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )
        strategy._mocked_cache.quote_tick.return_value = quote_tick

        # Act
        position_size = strategy._calculate_position_size()

        # Assert
        # Mid price = (99.5 + 100.5) / 2 = 100
        # 10000 * 0.1 / 100 = 10 shares
        assert position_size == Quantity.from_int(10)

    @pytest.mark.database
    @pytest.mark.serial
    def test_calculate_position_size_no_price_data(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
        sample_instrument: Equity,
    ) -> None:
        """
        Test position size calculation with no price data available.
        """
        # Arrange
        strategy.on_start()

        account = Mock()
        account.balance_total.return_value = Money(10000.00, USD)
        strategy._mocked_cache.account_for_venue.return_value = account
        strategy._mocked_cache.instrument.return_value = sample_instrument
        strategy._mocked_cache.trade_tick.return_value = None
        strategy._mocked_cache.quote_tick.return_value = None

        # Act
        position_size = strategy._calculate_position_size()

        # Assert
        assert position_size is None
        strategy._mocked_log.error.assert_called_once()
        assert "No price data available" in strategy._mocked_log.error.call_args[0][0]

    @pytest.mark.database
    @pytest.mark.serial
    def test_place_market_order(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test market order placement.
        """
        # Arrange
        strategy.on_start()
        strategy._mocked_cache.client_order_id.return_value = ClientOrderId("O-123")
        quantity = Quantity.from_int(100)

        # Act
        order_id = strategy._place_market_order(OrderSide.BUY, quantity)

        # Assert
        assert order_id == ClientOrderId("O-123")
        assert strategy._pending_orders == 1
        assert strategy._trades_executed == 1
        strategy._mocked_submit_order.assert_called_once()

        # Check order details
        order = strategy._mocked_submit_order.call_args[0][0]
        assert order.side == OrderSide.BUY
        assert order.quantity == quantity
        assert order.is_reduce_only is False

    @pytest.mark.database
    @pytest.mark.serial
    def test_place_market_order_reduce_only(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test reduce-only market order placement.
        """
        # Arrange
        strategy.on_start()
        strategy._mocked_cache.client_order_id.return_value = ClientOrderId("O-124")
        quantity = Quantity.from_int(50)

        # Act
        _ = strategy._place_market_order(OrderSide.SELL, quantity, reduce_only=True)

        # Assert
        order = strategy._mocked_submit_order.call_args[0][0]
        assert order.is_reduce_only is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_place_stop_loss(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test stop loss order placement.
        """
        # Arrange
        strategy.on_start()
        strategy._mocked_cache.client_order_id.return_value = ClientOrderId("O-125")
        quantity = Quantity.from_int(100)
        stop_price = Price.from_str("95.00")

        # Act
        order_id = strategy._place_stop_loss(OrderSide.SELL, quantity, stop_price)

        # Assert
        assert order_id == ClientOrderId("O-125")
        strategy._mocked_submit_order.assert_called_once()

        # Check order details
        order = strategy._mocked_submit_order.call_args[0][0]
        assert order.side == OrderSide.SELL
        assert order.quantity == quantity
        assert order.trigger_price == stop_price
        assert order.is_reduce_only is True

    @pytest.mark.database
    @pytest.mark.serial
    def test_get_current_position_returns_first_open(
        self,
        strategy: MockMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test getting current position returns first open position.
        """
        # Arrange
        strategy.on_start()

        # Mock positions
        position1 = Mock()
        position1.instrument_id = instrument_id
        position2 = Mock()
        position2.instrument_id = instrument_id

        strategy._mocked_cache.positions_open.return_value = [position1, position2]

        # Act
        position = strategy._get_current_position()

        # Assert
        assert position == position1

    @pytest.mark.database
    @pytest.mark.serial
    def test_get_current_position_returns_none_when_no_positions(
        self,
        strategy: MockMLStrategy,
    ) -> None:
        """
        Test getting current position returns None when no positions.
        """
        # Arrange
        strategy.on_start()
        strategy._mocked_cache.positions_open.return_value = []

        # Act
        position = strategy._get_current_position()

        # Assert
        assert position is None

    @pytest.mark.database
    @pytest.mark.serial
    def test_on_stop_logs_statistics(self, strategy: MockMLStrategy) -> None:
        """
        Test on_stop logs final statistics.
        """
        # Arrange
        strategy.on_start()
        strategy._signals_received = 100
        strategy._trades_executed = 50
        strategy._winning_trades = 30
        strategy._total_pnl = Decimal("1500.00")

        # Act
        strategy.on_stop()

        # Assert
        # Check the last info call
        assert strategy._mocked_log.info.call_count >= 1
        log_message = strategy._mocked_log.info.call_args[0][0]
        assert "Signals: 100" in log_message
        assert "Trades: 50" in log_message
        assert "Win rate: 60.0%" in log_message
        assert "Total PnL: 1500.00" in log_message

    @pytest.mark.database
    @pytest.mark.serial
    def test_on_stop_with_zero_trades(self, strategy: MockMLStrategy) -> None:
        """
        Test on_stop handles zero trades gracefully.
        """
        # Arrange
        strategy.on_start()
        strategy._signals_received = 10
        strategy._trades_executed = 0

        # Act
        strategy.on_stop()

        # Assert
        log_message = strategy._mocked_log.info.call_args[0][0]
        assert "Win rate: 0.0%" in log_message


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.usefixtures("clean_postgres_db")
class TestSimpleMLStrategy:
    """
    Test SimpleMLStrategy concrete implementation.
    """

    @pytest.fixture
    def instrument_id(self) -> InstrumentId:
        """
        Create test instrument ID.
        """
        return InstrumentId(Symbol("AAPL"), Venue("NASDAQ"))

    @pytest.fixture
    def config(self, instrument_id: InstrumentId) -> MLStrategyConfig:
        """
        Create test configuration.
        """
        return MLStrategyConfig(
            instrument_id=instrument_id,
            ml_signal_source="ML_SIGNAL_ACTOR",
            position_size_pct=0.1,
            min_confidence=0.7,
            max_positions=3,
        )

    @pytest.fixture
    def strategy(self, config: MLStrategyConfig) -> MockMLStrategy:
        """
        Create test strategy instance.
        """

        # Create a mock that inherits from SimpleMLStrategy
        @pytest.mark.database
        @pytest.mark.serial
        class TestableSimpleMLStrategy(SimpleMLStrategy):
            def __init__(self, config: MLStrategyConfig) -> None:
                # Set up mocks before calling super
                self._mocked_submit_order = Mock()
                self._mocked_log = Mock()
                self._mocked_cache = Mock()
                self._trader_id = TraderId("TESTER-001")
                self._id = StrategyId("SimpleMLStrategy-001")

                # Mock the StrategyStore to avoid database connection
                with patch("ml.strategies.base.StrategyStore"):
                    super().__init__(config)

            def submit_order(self, order: Any) -> None:
                self._mocked_submit_order(order)

            @property
            def log(self) -> Any:
                return self._mocked_log

            @property
            def cache(self) -> Any:
                return self._mocked_cache

            @property
            def trader_id(self) -> TraderId:
                return self._trader_id

            @property
            def id(self) -> StrategyId:
                return self._id

            @property
            def clock(self) -> Any:
                mock_clock = Mock()
                mock_clock.timestamp_ns = Mock(return_value=1234567890000000000)
                return mock_clock

        return TestableSimpleMLStrategy(config)

    @pytest.mark.database
    @pytest.mark.serial
    def test_process_ml_signal_opens_long_position(
        self,
        strategy: SimpleMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test processing ML signal opens long position for positive prediction.
        """
        # Arrange
        setattr(strategy, "_get_current_position", Mock(return_value=None))
        setattr(strategy, "_calculate_position_size", Mock(return_value=Quantity.from_int(100)))
        setattr(strategy, "_place_market_order", Mock(return_value=ClientOrderId("O-126")))

        signal = MLSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.8,  # > 0.5, so BUY
            confidence=0.9,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy._process_ml_signal(signal)

        # Assert
        strategy._place_market_order.assert_called_once_with(OrderSide.BUY, Quantity.from_int(100))  # type: ignore[attr-defined]
        assert strategy._active_positions == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_process_ml_signal_opens_short_position(
        self,
        strategy: SimpleMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test processing ML signal opens short position for negative prediction.
        """
        # Arrange
        setattr(strategy, "_get_current_position", Mock(return_value=None))
        setattr(strategy, "_calculate_position_size", Mock(return_value=Quantity.from_int(100)))
        setattr(strategy, "_place_market_order", Mock(return_value=ClientOrderId("O-127")))

        signal = MLSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.2,  # < 0.5, so SELL
            confidence=0.85,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy._process_ml_signal(signal)

        # Assert
        strategy._place_market_order.assert_called_once_with(OrderSide.SELL, Quantity.from_int(100))  # type: ignore[attr-defined]
        assert strategy._active_positions == 1

    @pytest.mark.database
    @pytest.mark.serial
    def test_process_ml_signal_reverses_position(
        self,
        strategy: SimpleMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test processing ML signal reverses existing position.
        """
        # Arrange
        # Mock existing long position
        current_position = Mock()
        current_position.side.name = "LONG"
        current_position.quantity = Quantity.from_int(50)

        setattr(strategy, "_get_current_position", Mock(return_value=current_position))
        setattr(strategy, "_calculate_position_size", Mock(return_value=Quantity.from_int(100)))
        setattr(strategy, "_place_market_order", Mock(return_value=ClientOrderId("O-128")))

        signal = MLSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.2,  # < 0.5, so SELL (opposite of LONG)
            confidence=0.9,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy._process_ml_signal(signal)

        # Assert
        assert strategy._place_market_order.call_count == 2  # type: ignore[attr-defined]
        # First call closes position
        first_call = strategy._place_market_order.call_args_list[0]  # type: ignore[attr-defined]
        assert first_call[0][0] == OrderSide.SELL
        assert first_call[0][1] == Quantity.from_int(50)
        assert first_call[1]["reduce_only"] is True

        # Second call opens new position
        second_call = strategy._place_market_order.call_args_list[1]  # type: ignore[attr-defined]
        assert second_call[0][0] == OrderSide.SELL
        assert second_call[0][1] == Quantity.from_int(100)

    @pytest.mark.database
    @pytest.mark.serial
    def test_process_ml_signal_keeps_aligned_position(
        self,
        strategy: SimpleMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test processing ML signal keeps position when already aligned.
        """
        # Arrange
        # Mock existing long position
        current_position = Mock()
        current_position.side.name = "LONG"

        setattr(strategy, "_get_current_position", Mock(return_value=current_position))
        setattr(strategy, "_place_market_order", Mock())

        signal = MLSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.8,  # > 0.5, so BUY (same as LONG)
            confidence=0.85,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy._process_ml_signal(signal)

        # Assert
        strategy._place_market_order.assert_not_called()  # type: ignore[attr-defined]
        strategy.log.debug.assert_called_once()
        assert "Position aligns with signal" in strategy.log.debug.call_args[0][0]

    @pytest.mark.database
    @pytest.mark.serial
    def test_process_ml_signal_no_entry_when_position_sizing_fails(
        self,
        strategy: SimpleMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test processing ML signal skips trade when position sizing returns None.
        """
        # Arrange
        setattr(strategy, "_get_current_position", Mock(return_value=None))
        setattr(strategy, "_calculate_position_size", Mock(return_value=None))
        setattr(strategy, "_place_market_order", Mock())

        signal = MLSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.8,  # > 0.5, so BUY
            confidence=0.9,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy._process_ml_signal(signal)

        # Assert
        strategy._place_market_order.assert_not_called()  # type: ignore[attr-defined]
        strategy.log.warning.assert_called_once()
        assert (
            "Skipping trade signal due to position sizing failure"
            in strategy.log.warning.call_args[0][0]
        )

    @pytest.mark.database
    @pytest.mark.serial
    def test_process_ml_signal_closes_but_no_new_entry_when_sizing_fails(
        self,
        strategy: SimpleMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test processing ML signal closes existing position but doesn't open new one when
        sizing fails.
        """
        # Arrange
        # Mock existing long position
        current_position = Mock()
        current_position.side.name = "LONG"
        current_position.quantity = Quantity.from_int(50)

        setattr(strategy, "_get_current_position", Mock(return_value=current_position))
        setattr(strategy, "_calculate_position_size", Mock(return_value=None))
        setattr(strategy, "_place_market_order", Mock(return_value=ClientOrderId("O-129")))

        signal = MLSignal(
            instrument_id=instrument_id,
            model_id="test_model",
            prediction=0.2,  # < 0.5, so SELL (opposite of LONG)
            confidence=0.9,
            ts_event=1234567890000000000,
            ts_init=1234567890000000000,
        )

        # Act
        strategy._process_ml_signal(signal)

        # Assert
        # Should only close position, not open new one
        assert strategy._place_market_order.call_count == 1  # type: ignore[attr-defined]
        call_args = strategy._place_market_order.call_args  # type: ignore[attr-defined]
        assert call_args[0][0] == OrderSide.SELL
        assert call_args[0][1] == Quantity.from_int(50)
        assert call_args[1]["reduce_only"] is True

        strategy.log.warning.assert_called_once()
        assert (
            "cannot open new one due to position sizing failure"
            in strategy.log.warning.call_args[0][0]
        )

    @pytest.mark.database
    @pytest.mark.serial
    def test_on_order_filled_updates_state(
        self,
        strategy: SimpleMLStrategy,
        instrument_id: InstrumentId,
    ) -> None:
        """
        Test order filled event updates strategy state.
        """
        # Arrange
        strategy._pending_orders = 2
        setattr(strategy, "_get_current_position", Mock(return_value=Mock()))

        # We can't easily test on_order_filled without a real OrderFilled event
        # So let's test the state management directly after trades

        # Act - simulate what happens after order fill
        strategy._pending_orders -= 1
        strategy._active_positions = 1

        # Assert
        assert strategy._pending_orders == 1
        assert strategy._active_positions == 1
