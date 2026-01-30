"""
Unit tests for PositionManagementComponent.

This module tests the position management component extracted from BaseMLStrategy,
verifying position sizing, risk validation, portfolio allocation, and quantity
conversion functionality.

Test Categories:
- Basic position sizing from balance percentage
- Position sizing with sizer/risk/portfolio integration
- Market price resolution (trade tick, quote tick)
- Value to quantity conversion
- Portfolio allocation handling
- Error handling and edge cases

"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest

from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Quantity


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
class MockQuantity:
    """Mock quantity object with as_double method."""

    _value: float

    def as_double(self) -> float:
        """Return quantity as float."""
        return self._value


@dataclass(slots=True)
class MockBalance:
    """Mock balance object."""

    _value: float

    def as_double(self) -> float:
        """Return balance as float."""
        return self._value


@dataclass(slots=True)
class MockAccount:
    """Mock account with balance_total method."""

    balance_value: float = 10_000.0

    def balance_total(self) -> MockBalance:
        """Return mock balance."""
        return MockBalance(self.balance_value)


@dataclass(slots=True)
class MockTradeTick:
    """Mock trade tick with price."""

    price: MockPrice = field(default_factory=lambda: MockPrice(100.0))


@dataclass(slots=True)
class MockQuoteTick:
    """Mock quote tick with bid/ask prices."""

    bid_price: MockPrice = field(default_factory=lambda: MockPrice(99.0))
    ask_price: MockPrice = field(default_factory=lambda: MockPrice(101.0))


@dataclass(slots=True)
class MockInstrument:
    """Mock instrument with precision and min_quantity."""

    id: InstrumentId
    venue: Venue
    size_precision: int = 4
    min_quantity: MockQuantity = field(default_factory=lambda: MockQuantity(0.0001))

    def make_qty(self, value: float, round_down: bool = False) -> Quantity:
        """Return Quantity aligned with instrument precision."""
        precision = int(self.size_precision)
        scale = 10**precision
        if round_down:
            rounded_value = math.floor(float(value) * scale) / scale
        else:
            rounded_value = round(float(value), precision)
        return Quantity(rounded_value, precision)


class MockCache:
    """Mock cache for testing."""

    def __init__(
        self,
        *,
        instrument: MockInstrument | None = None,
        account: MockAccount | None = None,
        trade_tick: MockTradeTick | None = None,
        quote_tick: MockQuoteTick | None = None,
        positions: list[Any] | None = None,
    ) -> None:
        """Initialize mock cache."""
        self._instrument = instrument
        self._account = account
        self._trade_tick = trade_tick
        self._quote_tick = quote_tick
        self._positions = positions or []

    def instrument(self, instrument_id: Any) -> MockInstrument | None:
        """Return mock instrument."""
        return self._instrument

    def account_for_venue(self, venue: Any) -> MockAccount | None:
        """Return mock account."""
        return self._account

    def trade_tick(self, instrument_id: Any) -> MockTradeTick | None:
        """Return mock trade tick."""
        return self._trade_tick

    def quote_tick(self, instrument_id: Any) -> MockQuoteTick | None:
        """Return mock quote tick."""
        return self._quote_tick

    def positions_open(
        self,
        venue: Any = None,
        instrument_id: Any = None,
    ) -> list[Any]:
        """Return mock positions."""
        return self._positions


class MockPositionSizer:
    """Mock position sizer for testing."""

    def __init__(
        self,
        *,
        return_value: Quantity | None = None,
        should_raise: bool = False,
    ) -> None:
        """Initialize mock sizer."""
        self._return_value = return_value
        self._should_raise = should_raise
        self.calculate_calls: list[tuple[Any, Any, list[Any]]] = []

    def calculate(
        self,
        signal: Any,
        account: Any,
        current_positions: list[Any],
    ) -> Quantity | None:
        """Return mock calculation result."""
        self.calculate_calls.append((signal, account, current_positions))
        if self._should_raise:
            raise RuntimeError("Sizer error")
        return self._return_value


class MockRiskManager:
    """Mock risk manager for testing."""

    def __init__(
        self,
        *,
        return_value: Quantity | None = None,
        should_reject: bool = False,
        should_raise: bool = False,
    ) -> None:
        """Initialize mock risk manager."""
        self._return_value = return_value
        self._should_reject = should_reject
        self._should_raise = should_raise
        self.check_position_calls: list[tuple[Any, Any, Any]] = []

    def check_position(
        self,
        proposed_size: Any | None,
        instrument: Any,
        portfolio: Any,
    ) -> Quantity | None:
        """Return mock check result."""
        self.check_position_calls.append((proposed_size, instrument, portfolio))
        if self._should_raise:
            raise RuntimeError("Risk manager error")
        if self._should_reject:
            return None
        return self._return_value if self._return_value is not None else proposed_size


class MockPortfolioManager:
    """Mock portfolio manager for testing."""

    def __init__(
        self,
        *,
        allocations: dict[Any, float] | None = None,
        should_raise: bool = False,
    ) -> None:
        """Initialize mock portfolio manager."""
        self._allocations = allocations or {}
        self._should_raise = should_raise
        self.allocate_calls: list[tuple[list[Any], float]] = []

    def allocate_signals(
        self,
        signals: list[Any],
        available_capital: float,
    ) -> dict[Any, float]:
        """Return mock allocations."""
        self.allocate_calls.append((signals, available_capital))
        if self._should_raise:
            raise RuntimeError("Portfolio manager error")
        return self._allocations


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
        size_precision=4,
        min_quantity=MockQuantity(0.0001),
    )


@pytest.fixture
def mock_account() -> MockAccount:
    """Create mock account with 10,000 balance."""
    return MockAccount(balance_value=10_000.0)


@pytest.fixture
def mock_trade_tick() -> MockTradeTick:
    """Create mock trade tick at price 100."""
    return MockTradeTick(price=MockPrice(100.0))


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
    mock_account: MockAccount,
    mock_trade_tick: MockTradeTick,
    mock_quote_tick: MockQuoteTick,
) -> MockCache:
    """Create mock cache with all components."""
    return MockCache(
        instrument=mock_instrument,
        account=mock_account,
        trade_tick=mock_trade_tick,
        quote_tick=mock_quote_tick,
        positions=[],
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
def position_management_component(
    instrument_id: InstrumentId,
    mock_cache: MockCache,
    mock_logger: MockLogger,
) -> PositionManagementComponent:
    """Create position management component with standard configuration."""
    from ml.strategies.common.position_management import PositionManagementComponent

    return PositionManagementComponent(
        position_size_pct=0.05,
        cache=mock_cache,
        instrument_id=instrument_id,
        log=mock_logger,
        strategy_id="test_strategy",
    )


# ---------------------------------------------------------------------------
# Test: Basic Position Sizing
# ---------------------------------------------------------------------------


class TestCalculatePositionSizeBasic:
    """Tests for basic position sizing from balance percentage."""

    def test_calculate_position_size_basic(
        self,
        position_management_component: PositionManagementComponent,
    ) -> None:
        """Verify basic position sizing from balance percentage."""
        # Account balance: 10,000
        # Position size pct: 0.05 (5%)
        # Position value: 500
        # Current price: 100
        # Expected quantity: 500 / 100 = 5.0

        quantity = position_management_component.calculate_position_size()

        assert quantity is not None
        # Quantity should be approximately 5.0
        assert abs(float(quantity.as_double()) - 5.0) < 0.01

    def test_calculate_position_size_no_account(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when account not available."""
        from ml.strategies.common.position_management import PositionManagementComponent

        cache = MockCache(
            instrument=mock_instrument,
            account=None,  # No account
            trade_tick=MockTradeTick(),
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        quantity = component.calculate_position_size()

        assert quantity is None
        assert len(mock_logger.error_calls) > 0

    def test_calculate_position_size_no_instrument(
        self,
        instrument_id: InstrumentId,
        mock_account: MockAccount,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when instrument not found."""
        from ml.strategies.common.position_management import PositionManagementComponent

        cache = MockCache(
            instrument=None,  # No instrument
            account=mock_account,
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        quantity = component.calculate_position_size()

        assert quantity is None
        assert len(mock_logger.error_calls) > 0

    def test_calculate_position_size_respects_min_quantity(
        self,
        instrument_id: InstrumentId,
        mock_account: MockAccount,
        mock_logger: MockLogger,
    ) -> None:
        """Verify minimum quantity enforced."""
        from ml.strategies.common.position_management import PositionManagementComponent

        # Create instrument with high min_quantity
        instrument = MockInstrument(
            id=instrument_id,
            venue=instrument_id.venue,
            size_precision=4,
            min_quantity=MockQuantity(10.0),  # Min quantity of 10
        )

        cache = MockCache(
            instrument=instrument,
            account=mock_account,
            trade_tick=MockTradeTick(price=MockPrice(1000.0)),  # High price -> low quantity
        )

        component = PositionManagementComponent(
            position_size_pct=0.01,  # 1% of 10,000 = 100 / 1000 = 0.1 (less than min)
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        quantity = component.calculate_position_size()

        assert quantity is not None
        assert float(quantity.as_double()) >= 10.0  # Should be at least min_quantity

    def test_calculate_position_size_respects_precision(
        self,
        instrument_id: InstrumentId,
        mock_account: MockAccount,
        mock_logger: MockLogger,
    ) -> None:
        """Verify quantity rounded to instrument precision."""
        from ml.strategies.common.position_management import PositionManagementComponent

        # Create instrument with precision=2
        instrument = MockInstrument(
            id=instrument_id,
            venue=instrument_id.venue,
            size_precision=2,
            min_quantity=MockQuantity(0.01),
        )

        cache = MockCache(
            instrument=instrument,
            account=mock_account,
            trade_tick=MockTradeTick(price=MockPrice(33.33)),  # Results in non-round quantity
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,  # 500 / 33.33 = 15.0015...
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        quantity = component.calculate_position_size()

        assert quantity is not None
        # Check precision (2 decimal places)
        qty_str = str(quantity)
        if "." in qty_str:
            decimal_places = len(qty_str.split(".")[1])
            assert decimal_places <= 2

    def test_calculate_position_size_precision_zero_uses_instrument_precision(
        self,
        instrument_id: InstrumentId,
        mock_account: MockAccount,
        mock_logger: MockLogger,
    ) -> None:
        """Verify size_precision=0 yields precision-aligned quantity."""
        from ml.strategies.common.position_management import PositionManagementComponent

        instrument = MockInstrument(
            id=instrument_id,
            venue=instrument_id.venue,
            size_precision=0,
            min_quantity=MockQuantity(1.0),
        )

        cache = MockCache(
            instrument=instrument,
            account=mock_account,
            trade_tick=MockTradeTick(price=MockPrice(100.0)),
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        quantity = component.calculate_position_size()

        assert quantity is not None
        assert quantity.precision == instrument.size_precision


# ---------------------------------------------------------------------------
# Test: Size and Validate with Position Sizer
# ---------------------------------------------------------------------------


class TestSizeAndValidateWithSizer:
    """Tests for size_and_validate using position sizer."""

    def test_size_and_validate_uses_position_sizer(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify position sizer is invoked when available."""
        from ml.strategies.common.position_management import PositionManagementComponent

        sizer = MockPositionSizer(return_value=Quantity.from_str("10.0"))

        component = PositionManagementComponent(
            position_size_pct=0.05,
            position_sizer=sizer,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        # Sizer should have been called
        assert len(sizer.calculate_calls) == 1

    def test_size_and_validate_converts_value_sizer_to_quantity(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify value-based sizer outputs convert to quantity once."""
        from ml.strategies.common.position_management import PositionManagementComponent

        # Value of 500 at price 100 should yield quantity 5.0
        sizer = MockPositionSizer(return_value=Quantity.from_str("500.0"))

        component = PositionManagementComponent(
            position_size_pct=0.05,
            position_sizer=sizer,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is not None
        assert abs(float(result.as_double()) - 5.0) < 0.01

    def test_size_and_validate_fallback_to_basic_sizing(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify fallback when sizer returns None."""
        from ml.strategies.common.position_management import PositionManagementComponent

        sizer = MockPositionSizer(return_value=None)  # Sizer returns None

        component = PositionManagementComponent(
            position_size_pct=0.05,
            position_sizer=sizer,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        # Should fall back to basic sizing and return valid quantity
        assert result is not None

    def test_size_and_validate_sizer_exception_handling(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify fallback on sizer exception."""
        from ml.strategies.common.position_management import PositionManagementComponent

        sizer = MockPositionSizer(should_raise=True)  # Sizer raises exception

        component = PositionManagementComponent(
            position_size_pct=0.05,
            position_sizer=sizer,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        # Should handle exception and fall back to basic sizing
        assert result is not None
        # Debug log should have been called for the exception
        assert len(mock_logger.debug_calls) > 0


# ---------------------------------------------------------------------------
# Test: Risk Manager Integration
# ---------------------------------------------------------------------------


class TestSizeAndValidateWithRiskManager:
    """Tests for size_and_validate with risk manager integration."""

    def test_size_and_validate_risk_manager_rejection(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when risk manager rejects."""
        from ml.strategies.common.position_management import PositionManagementComponent

        risk_manager = MockRiskManager(should_reject=True)

        component = PositionManagementComponent(
            position_size_pct=0.05,
            risk_manager=risk_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is None
        assert len(risk_manager.check_position_calls) == 1

    def test_size_and_validate_risk_manager_approves(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify quantity returned when risk manager approves."""
        from ml.strategies.common.position_management import PositionManagementComponent

        risk_manager = MockRiskManager(return_value=Quantity.from_str("5.0"))

        component = PositionManagementComponent(
            position_size_pct=0.05,
            risk_manager=risk_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is not None

    def test_size_and_validate_passes_value_to_risk_manager(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify risk manager receives value (not quantity)."""
        from ml.strategies.common.position_management import PositionManagementComponent

        risk_manager = MockRiskManager(return_value=Quantity.from_str("500.0"))

        component = PositionManagementComponent(
            position_size_pct=0.05,
            risk_manager=risk_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is not None
        assert len(risk_manager.check_position_calls) == 1
        proposed_size = risk_manager.check_position_calls[0][0]
        assert float(proposed_size.as_double()) == pytest.approx(500.0)
        assert abs(float(result.as_double()) - 5.0) < 0.01


# ---------------------------------------------------------------------------
# Test: Portfolio Allocation
# ---------------------------------------------------------------------------


class TestPortfolioAllocation:
    """Tests for portfolio allocation integration."""

    def test_size_and_validate_portfolio_allocation_scaling(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify position scaled by portfolio allocation."""
        from ml.strategies.common.position_management import PositionManagementComponent

        # Portfolio manager allocates less than proposed
        portfolio_manager = MockPortfolioManager(
            allocations={instrument_id: 250.0},  # Half of expected 500
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,  # 500 value
            portfolio_manager=portfolio_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        # Portfolio manager should have been consulted
        assert len(portfolio_manager.allocate_calls) == 1

    def test_size_and_validate_caps_allocation_above_proposed_value(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify allocation cannot exceed proposed value from sizing."""
        from ml.strategies.common.position_management import PositionManagementComponent

        # Proposed value from sizing: 500.0
        # Portfolio manager attempts to allocate 800.0 (should cap to 500.0)
        sizer = MockPositionSizer(return_value=Quantity.from_str("500.0"))
        portfolio_manager = MockPortfolioManager(
            allocations={instrument_id: 800.0},
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            position_sizer=sizer,
            portfolio_manager=portfolio_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is not None
        # price=100 -> quantity should be 5.0 (500/100)
        assert abs(float(result.as_double()) - 5.0) < 0.01

    def test_size_and_validate_zero_allocation_returns_none(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when allocation is zero."""
        from ml.strategies.common.position_management import PositionManagementComponent

        # Portfolio manager returns zero allocation
        portfolio_manager = MockPortfolioManager(
            allocations={instrument_id: 0.0},
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            portfolio_manager=portfolio_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is None
        assert len(mock_logger.debug_calls) > 0

    def test_apply_portfolio_allocation_when_manager_none(
        self,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify proposed value returned when no manager."""
        from ml.strategies.common.position_management import PositionManagementComponent

        component = PositionManagementComponent(
            position_size_pct=0.05,
            portfolio_manager=None,
            log=mock_logger,
        )

        result = component.apply_portfolio_allocation(
            signal=mock_signal,
            proposed_value=500.0,
        )

        assert result == 500.0

    def test_apply_portfolio_allocation_exception_handling(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify proposed value returned on exception."""
        from ml.strategies.common.position_management import PositionManagementComponent

        portfolio_manager = MockPortfolioManager(should_raise=True)

        component = PositionManagementComponent(
            position_size_pct=0.05,
            portfolio_manager=portfolio_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.apply_portfolio_allocation(
            signal=mock_signal,
            proposed_value=500.0,
            account=MockAccount(),
        )

        # Should return proposed value on exception
        assert result == 500.0
        assert len(mock_logger.debug_calls) > 0


# ---------------------------------------------------------------------------
# Test: Market Price Resolution
# ---------------------------------------------------------------------------


class TestResolveMarketPrice:
    """Tests for market price resolution."""

    def test_resolve_market_price_from_trade_tick(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify price resolved from trade tick."""
        from ml.strategies.common.position_management import PositionManagementComponent

        cache = MockCache(
            trade_tick=MockTradeTick(price=MockPrice(100.5)),
            quote_tick=MockQuoteTick(),
        )

        component = PositionManagementComponent(
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        price = component.resolve_market_price(instrument_id)

        assert price == 100.5

    def test_resolve_market_price_from_quote_tick(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify price resolved from quote tick midpoint."""
        from ml.strategies.common.position_management import PositionManagementComponent

        cache = MockCache(
            trade_tick=None,  # No trade tick
            quote_tick=MockQuoteTick(
                bid_price=MockPrice(99.0),
                ask_price=MockPrice(101.0),
            ),
        )

        component = PositionManagementComponent(
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        price = component.resolve_market_price(instrument_id)

        # Midpoint of 99 and 101 is 100
        assert price == 100.0

    def test_resolve_market_price_none_when_no_data(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when no market data."""
        from ml.strategies.common.position_management import PositionManagementComponent

        cache = MockCache(
            trade_tick=None,
            quote_tick=None,
        )

        component = PositionManagementComponent(
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        price = component.resolve_market_price(instrument_id)

        assert price is None
        assert len(mock_logger.error_calls) > 0

    def test_resolve_market_price_no_cache(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when cache not available."""
        from ml.strategies.common.position_management import PositionManagementComponent

        component = PositionManagementComponent(
            cache=None,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        price = component.resolve_market_price(instrument_id)

        assert price is None


# ---------------------------------------------------------------------------
# Test: Value to Quantity Conversion
# ---------------------------------------------------------------------------


class TestValueToQuantityConversion:
    """Tests for value to quantity conversion."""

    def test_value_to_quantity_conversion(
        self,
        mock_instrument: MockInstrument,
        mock_logger: MockLogger,
    ) -> None:
        """Verify correct conversion from value to quantity."""
        from ml.strategies.common.position_management import PositionManagementComponent

        component = PositionManagementComponent(log=mock_logger)

        quantity = component.value_to_quantity(
            value=500.0,
            price=100.0,
            instrument=mock_instrument,
        )

        # 500 / 100 = 5.0
        assert abs(float(quantity.as_double()) - 5.0) < 0.0001

    def test_quantity_min_floor(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify quantity not less than min_quantity."""
        from ml.strategies.common.position_management import PositionManagementComponent

        # Create instrument with high min_quantity
        instrument = MockInstrument(
            id=instrument_id,
            venue=instrument_id.venue,
            size_precision=4,
            min_quantity=MockQuantity(1.0),  # Min of 1.0
        )

        component = PositionManagementComponent(log=mock_logger)

        # Very small value that would result in qty < min
        quantity = component.value_to_quantity(
            value=0.01,  # Would be 0.0001 at price 100
            price=100.0,
            instrument=instrument,
        )

        # Should be floored to min_quantity
        assert float(quantity.as_double()) >= 1.0

    def test_value_to_quantity_precision(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify quantity respects precision."""
        from ml.strategies.common.position_management import PositionManagementComponent

        # Create instrument with precision=2
        instrument = MockInstrument(
            id=instrument_id,
            venue=instrument_id.venue,
            size_precision=2,
            min_quantity=MockQuantity(0.01),
        )

        component = PositionManagementComponent(log=mock_logger)

        quantity = component.value_to_quantity(
            value=333.33,  # Results in 3.3333... at price 100
            price=100.0,
            instrument=instrument,
        )

        # Check precision (2 decimal places)
        qty_str = str(quantity)
        if "." in qty_str:
            decimal_places = len(qty_str.split(".")[1])
            assert decimal_places <= 2

    def test_value_to_quantity_precision_zero_uses_instrument_precision(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify size_precision=0 yields precision-aligned quantity."""
        from ml.strategies.common.position_management import PositionManagementComponent

        instrument = MockInstrument(
            id=instrument_id,
            venue=instrument_id.venue,
            size_precision=0,
            min_quantity=MockQuantity(1.0),
        )

        component = PositionManagementComponent(log=mock_logger)

        quantity = component.value_to_quantity(
            value=500.0,
            price=100.0,
            instrument=instrument,
        )

        assert quantity.precision == instrument.size_precision


# ---------------------------------------------------------------------------
# Test: Component Configuration
# ---------------------------------------------------------------------------


class TestComponentConfiguration:
    """Tests for component configuration and properties."""

    def test_properties_accessible(
        self,
        position_management_component: PositionManagementComponent,
        instrument_id: InstrumentId,
    ) -> None:
        """Verify all properties are accessible."""
        assert position_management_component.position_size_pct == 0.05
        assert position_management_component.instrument_id == instrument_id
        assert position_management_component.position_sizer is None
        assert position_management_component.risk_manager is None
        assert position_management_component.portfolio_manager is None

    def test_update_config(
        self,
        position_management_component: PositionManagementComponent,
    ) -> None:
        """Verify configuration can be updated."""
        new_instrument_id = InstrumentId(Symbol("GBPUSD"), Venue("SIM"))

        position_management_component.update_config(
            position_size_pct=0.10,
            instrument_id=new_instrument_id,
        )

        assert position_management_component.position_size_pct == 0.10
        assert position_management_component.instrument_id == new_instrument_id


# ---------------------------------------------------------------------------
# Test: Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_size_and_validate_no_cache(
        self,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when cache not available."""
        from ml.strategies.common.position_management import PositionManagementComponent

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=None,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is None
        assert len(mock_logger.error_calls) > 0

    def test_size_and_validate_no_instrument_id(
        self,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when instrument_id not configured."""
        from ml.strategies.common.position_management import PositionManagementComponent

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=mock_cache,
            instrument_id=None,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is None
        assert len(mock_logger.error_calls) > 0

    def test_size_and_validate_instrument_not_found(
        self,
        instrument_id: InstrumentId,
        mock_account: MockAccount,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when instrument not in cache."""
        from ml.strategies.common.position_management import PositionManagementComponent

        cache = MockCache(
            instrument=None,  # No instrument
            account=mock_account,
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is None

    def test_size_and_validate_no_account(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when no account available."""
        from ml.strategies.common.position_management import PositionManagementComponent

        cache = MockCache(
            instrument=mock_instrument,
            account=None,  # No account
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is None

    def test_size_and_validate_no_market_price(
        self,
        instrument_id: InstrumentId,
        mock_instrument: MockInstrument,
        mock_account: MockAccount,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when no market price available."""
        from ml.strategies.common.position_management import PositionManagementComponent

        cache = MockCache(
            instrument=mock_instrument,
            account=mock_account,
            trade_tick=None,
            quote_tick=None,
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is None

    def test_calculate_position_size_no_cache(
        self,
        instrument_id: InstrumentId,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when cache not available for basic sizing."""
        from ml.strategies.common.position_management import PositionManagementComponent

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=None,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.calculate_position_size()

        assert result is None

    def test_calculate_position_size_no_instrument_id(
        self,
        mock_cache: MockCache,
        mock_logger: MockLogger,
    ) -> None:
        """Verify None returned when instrument_id not configured."""
        from ml.strategies.common.position_management import PositionManagementComponent

        component = PositionManagementComponent(
            position_size_pct=0.05,
            cache=mock_cache,
            instrument_id=None,
            log=mock_logger,
        )

        result = component.calculate_position_size()

        assert result is None


# ---------------------------------------------------------------------------
# Test: Integration with Full Pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Tests for full position management pipeline."""

    def test_full_pipeline_with_all_components(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify full pipeline with sizer, risk manager, and portfolio manager."""
        from ml.strategies.common.position_management import PositionManagementComponent

        sizer = MockPositionSizer(return_value=Quantity.from_str("10.0"))
        risk_manager = MockRiskManager(return_value=Quantity.from_str("8.0"))
        portfolio_manager = MockPortfolioManager(
            allocations={instrument_id: 800.0},
        )

        component = PositionManagementComponent(
            position_size_pct=0.05,
            position_sizer=sizer,
            risk_manager=risk_manager,
            portfolio_manager=portfolio_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        # All components should have been called
        assert len(sizer.calculate_calls) == 1
        assert len(portfolio_manager.allocate_calls) == 1
        assert len(risk_manager.check_position_calls) == 1
        assert result is not None

    def test_pipeline_stops_at_risk_rejection(
        self,
        instrument_id: InstrumentId,
        mock_cache: MockCache,
        mock_signal: MockMLSignal,
        mock_logger: MockLogger,
    ) -> None:
        """Verify pipeline stops when risk manager rejects."""
        from ml.strategies.common.position_management import PositionManagementComponent

        sizer = MockPositionSizer(return_value=Quantity.from_str("10.0"))
        risk_manager = MockRiskManager(should_reject=True)

        component = PositionManagementComponent(
            position_size_pct=0.05,
            position_sizer=sizer,
            risk_manager=risk_manager,
            cache=mock_cache,
            instrument_id=instrument_id,
            log=mock_logger,
        )

        result = component.size_and_validate(mock_signal)

        assert result is None
        assert len(sizer.calculate_calls) == 1
        assert len(risk_manager.check_position_calls) == 1
