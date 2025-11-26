"""
Position management component for MLTradingStrategy decomposition.

This component extracts position sizing, risk validation, and portfolio allocation
logic from BaseMLStrategy following the Protocol-First Interface Design pattern.

Responsibility:
- Calculate position sizes from account balance
- Validate positions with risk manager
- Apply portfolio allocation constraints
- Convert values to instrument-aware quantities
- Resolve market prices for calculations

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.model.position import Position

    from ml.actors.base import MLSignal


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache access."""

    def account_for_venue(self, venue: Any) -> Any:
        """Get account for a venue."""
        ...

    def instrument(self, instrument_id: Any) -> Any:
        """Get instrument by ID."""
        ...

    def trade_tick(self, instrument_id: Any) -> Any:
        """Get latest trade tick for instrument."""
        ...

    def quote_tick(self, instrument_id: Any) -> Any:
        """Get latest quote tick for instrument."""
        ...

    def positions_open(
        self,
        venue: Any = None,
        instrument_id: Any = None,
    ) -> list[Any]:
        """Get open positions."""
        ...


@runtime_checkable
class PositionSizerProtocol(Protocol):
    """Protocol for position sizing."""

    def calculate(
        self,
        signal: Any,
        account: Any,
        current_positions: list[Any],
    ) -> Any | None:
        """Calculate position size based on signal and account state."""
        ...


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Protocol for risk management."""

    def check_position(
        self,
        proposed_size: Any | None,
        instrument: Any,
        portfolio: Any,
    ) -> Any | None:
        """Check and potentially adjust proposed position size based on risk limits."""
        ...


@runtime_checkable
class PortfolioManagerProtocol(Protocol):
    """Protocol for portfolio management."""

    def allocate_signals(
        self,
        signals: list[Any],
        available_capital: float,
    ) -> dict[Any, float]:
        """Allocate capital across multiple signals."""
        ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logging interface."""

    def debug(self, *args: object, **kwargs: object) -> None:
        """Log debug message."""
        ...

    def info(self, *args: object, **kwargs: object) -> None:
        """Log info message."""
        ...

    def warning(self, *args: object, **kwargs: object) -> None:
        """Log warning message."""
        ...

    def error(self, *args: object, **kwargs: object) -> None:
        """Log error message."""
        ...


class _NoOpLogger:
    """No-op logger for when no logger is provided."""

    def debug(self, *args: object, **kwargs: object) -> None:
        """No-op debug."""
        del args, kwargs

    def info(self, *args: object, **kwargs: object) -> None:
        """No-op info."""
        del args, kwargs

    def warning(self, *args: object, **kwargs: object) -> None:
        """No-op warning."""
        del args, kwargs

    def error(self, *args: object, **kwargs: object) -> None:
        """No-op error."""
        del args, kwargs


class PositionManagementComponent:
    """
    Manages position sizing, risk validation, and portfolio allocation.

    This component is extracted from BaseMLStrategy to provide focused,
    testable position management functionality following the facade pattern.

    Responsibilities:
    - Calculate position sizes from account balance
    - Validate positions with risk manager
    - Apply portfolio allocation constraints
    - Convert values to instrument-aware quantities
    - Resolve market prices for calculations

    Parameters
    ----------
    position_size_pct : float, default 0.02
        Position size as percentage of account balance.
    position_sizer : PositionSizerProtocol | None, optional
        Advanced position sizer (e.g., CompositeSizer).
    risk_manager : RiskManagerProtocol | None, optional
        Risk manager for position validation.
    portfolio_manager : PortfolioManagerProtocol | None, optional
        Portfolio manager for allocation constraints.
    cache : CacheProtocol | None, optional
        Cache for instrument/account/market data access.
    portfolio : Any, optional
        Portfolio instance for risk manager checks.
    instrument_id : Any, optional
        Target instrument ID for position sizing.
    account_id : Any, optional
        Account ID for balance lookup.
    log : LoggerProtocol | None, optional
        Logger instance for debug output.
    strategy_id : str, default ""
        Strategy identifier for logging context.

    Examples
    --------
    >>> component = PositionManagementComponent(
    ...     position_size_pct=0.05,
    ...     position_sizer=composite_sizer,
    ...     risk_manager=risk_manager,
    ...     cache=cache,
    ...     instrument_id=instrument_id,
    ... )
    >>> quantity = component.size_and_validate(signal)

    """

    def __init__(
        self,
        position_size_pct: float = 0.02,
        position_sizer: PositionSizerProtocol | None = None,
        risk_manager: RiskManagerProtocol | None = None,
        portfolio_manager: PortfolioManagerProtocol | None = None,
        cache: CacheProtocol | None = None,
        portfolio: Any = None,
        instrument_id: Any = None,
        account_id: Any = None,
        log: Any = None,
        strategy_id: str = "",
    ) -> None:
        """Initialize the position management component."""
        self._position_size_pct = position_size_pct
        self._position_sizer = position_sizer
        self._risk_manager = risk_manager
        self._portfolio_manager = portfolio_manager
        self._cache = cache
        self._portfolio = portfolio
        self._instrument_id = instrument_id
        self._account_id = account_id
        self._log = log if log is not None else _NoOpLogger()
        self._strategy_id = strategy_id

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    @property
    def position_size_pct(self) -> float:
        """Get the position size percentage."""
        return self._position_size_pct

    @property
    def position_sizer(self) -> PositionSizerProtocol | None:
        """Get the position sizer."""
        return self._position_sizer

    @property
    def risk_manager(self) -> RiskManagerProtocol | None:
        """Get the risk manager."""
        return self._risk_manager

    @property
    def portfolio_manager(self) -> PortfolioManagerProtocol | None:
        """Get the portfolio manager."""
        return self._portfolio_manager

    @property
    def instrument_id(self) -> Any:
        """Get the target instrument ID."""
        return self._instrument_id

    # -------------------------------------------------------------------------
    # Configuration Update Methods
    # -------------------------------------------------------------------------

    def update_config(
        self,
        *,
        position_size_pct: float | None = None,
        instrument_id: Any = None,
        cache: CacheProtocol | None = None,
        portfolio: Any = None,
    ) -> None:
        """
        Update component configuration.

        Parameters
        ----------
        position_size_pct : float | None, optional
            Updated position size percentage.
        instrument_id : Any, optional
            Updated target instrument ID.
        cache : CacheProtocol | None, optional
            Updated cache instance.
        portfolio : Any, optional
            Updated portfolio instance.

        """
        if position_size_pct is not None:
            self._position_size_pct = position_size_pct
        if instrument_id is not None:
            self._instrument_id = instrument_id
        if cache is not None:
            self._cache = cache
        if portfolio is not None:
            self._portfolio = portfolio

    # -------------------------------------------------------------------------
    # Market Price Resolution
    # -------------------------------------------------------------------------

    def resolve_market_price(self, instrument_id: Any = None) -> float | None:
        """
        Resolve current market price from trade tick or quote tick midpoint.

        This method attempts to get the current market price for position
        sizing calculations. It first checks for trade ticks, then falls
        back to quote tick midpoint if no trade ticks are available.

        Parameters
        ----------
        instrument_id : Any, optional
            The instrument ID to resolve price for. If None, uses the
            configured instrument_id.

        Returns
        -------
        float | None
            The current market price as float, or None if no data available.

        Examples
        --------
        >>> price = component.resolve_market_price()
        >>> if price is not None:
        ...     print(f"Current price: {price}")

        """
        if self._cache is None:
            return None

        target_id = instrument_id if instrument_id is not None else self._instrument_id
        if target_id is None:
            return None

        # Try trade tick first
        try:
            trade_tick = self._cache.trade_tick(target_id)
            if trade_tick is not None:
                return float(trade_tick.price.as_double())
        except (AttributeError, TypeError):
            pass

        # Fall back to quote tick midpoint
        try:
            quote_tick = self._cache.quote_tick(target_id)
            if quote_tick is not None:
                bid_price = float(quote_tick.bid_price.as_double())
                ask_price = float(quote_tick.ask_price.as_double())
                return (bid_price + ask_price) / 2.0
        except (AttributeError, TypeError):
            pass

        # Log error if no market data available
        instrument_label = getattr(target_id, "value", str(target_id))
        self._log.error(
            f"No market price available for instrument {instrument_label}",
        )
        return None

    # -------------------------------------------------------------------------
    # Basic Position Sizing
    # -------------------------------------------------------------------------

    def calculate_position_size(self) -> Quantity | None:
        """
        Calculate basic position size from account balance and percentage.

        This method implements the basic position sizing formula:
        quantity = (account_balance * position_size_pct) / current_price

        The quantity is rounded to instrument precision and floored to
        the instrument's minimum quantity.

        Returns
        -------
        Quantity | None
            The calculated position size, or None if calculation fails.

        Examples
        --------
        >>> quantity = component.calculate_position_size()
        >>> if quantity is not None:
        ...     print(f"Position size: {quantity}")

        """
        from nautilus_trader.model.objects import Quantity

        if self._cache is None:
            self._log.error(
                "Cannot calculate position size: Cache not available",
            )
            return None

        if self._instrument_id is None:
            self._log.error(
                "Cannot calculate position size: Instrument ID not configured",
            )
            return None

        # Get instrument
        instrument = self._cache.instrument(self._instrument_id)
        if instrument is None:
            self._log.error(
                f"Cannot calculate position size: Instrument "
                f"{self._instrument_id} not found. "
                "Ensure instrument is subscribed and available in cache.",
            )
            return None

        # Get account for venue
        account = self._cache.account_for_venue(instrument.venue)
        if account is None:
            self._log.error(
                f"Cannot calculate position size: No account found for venue "
                f"{instrument.venue}. Position sizing requires account information.",
            )
            return None

        # Get account balance
        try:
            account_balance = float(account.balance_total().as_double())
        except (AttributeError, TypeError) as exc:
            self._log.error(
                f"Cannot calculate position size: Failed to get account balance: {exc}",
            )
            return None

        # Calculate position value
        position_value = account_balance * self._position_size_pct

        # Get current price
        current_price = self.resolve_market_price(self._instrument_id)
        if current_price is None:
            return None

        # Avoid division by zero
        if current_price <= 0:
            self._log.error(
                f"Cannot calculate position size: Invalid market price {current_price}",
            )
            return None

        # Calculate raw quantity
        raw_quantity = position_value / current_price

        # Round to instrument precision
        precision = instrument.size_precision
        quantity_value = round(raw_quantity, precision)

        # Ensure minimum size
        min_quantity = float(instrument.min_quantity.as_double())
        quantity_value = max(quantity_value, min_quantity)

        return Quantity.from_str(str(quantity_value))

    # -------------------------------------------------------------------------
    # Value to Quantity Conversion
    # -------------------------------------------------------------------------

    def value_to_quantity(
        self,
        value: float,
        price: float,
        instrument: Any,
    ) -> Quantity:
        """
        Convert a value to instrument-aware Quantity.

        This method converts a monetary value to a quantity using the
        current price, then applies precision rounding and minimum
        quantity constraints from the instrument specification.

        Parameters
        ----------
        value : float
            The value to convert (e.g., position value in base currency).
        price : float
            The current price for conversion.
        instrument : Any
            The instrument with precision and min_quantity attributes.

        Returns
        -------
        Quantity
            The converted quantity with proper precision and minimum applied.

        Examples
        --------
        >>> quantity = component.value_to_quantity(500.0, 100.0, instrument)
        >>> # Returns Quantity of 5.0 (500 / 100)

        """
        from nautilus_trader.model.objects import Quantity

        # Avoid division by zero
        safe_price = max(price, 1e-12)
        raw_qty = value / safe_price

        # Apply precision rounding
        precision = instrument.size_precision
        qty_value = round(raw_qty, precision)

        # Apply minimum quantity floor
        min_quantity = float(instrument.min_quantity.as_double())
        qty_value = max(qty_value, min_quantity)

        return Quantity.from_str(str(qty_value))

    # -------------------------------------------------------------------------
    # Portfolio Allocation
    # -------------------------------------------------------------------------

    def apply_portfolio_allocation(
        self,
        signal: MLSignal,
        proposed_value: float,
        account: Any = None,
    ) -> float:
        """
        Apply portfolio allocation constraints to proposed value.

        If a portfolio manager is configured, this method uses it to
        determine the appropriate allocation for the signal. If no
        manager is configured or an error occurs, the proposed value
        is returned unchanged.

        Parameters
        ----------
        signal : MLSignal
            The ML signal being processed.
        proposed_value : float
            The proposed position value before allocation.
        account : Any, optional
            Account for capital lookup. If None, attempts to get from cache.

        Returns
        -------
        float
            The allocated value (may be less than proposed).

        Examples
        --------
        >>> allocated = component.apply_portfolio_allocation(signal, 500.0)
        >>> # May return 400.0 if portfolio manager limits allocation

        """
        if self._portfolio_manager is None:
            return proposed_value

        # Get account for capital calculation
        if account is None and self._cache is not None and self._instrument_id is not None:
            instrument = self._cache.instrument(self._instrument_id)
            if instrument is not None:
                account = self._cache.account_for_venue(instrument.venue)

        if account is None:
            return proposed_value

        # Get available capital
        try:
            available_capital = float(account.balance_total().as_double())
        except (AttributeError, TypeError) as exc:
            self._log.debug(
                "ml_strategy.account_balance_unavailable",
                strategy_id=self._strategy_id,
                instrument=str(signal.instrument_id),
                exc_info=True,
                error=str(exc),
            )
            return proposed_value

        # Get allocations from portfolio manager
        try:
            allocations = self._portfolio_manager.allocate_signals(
                [signal],
                available_capital,
            )
        except Exception as exc:
            self._log.debug(
                "ml_strategy.portfolio_allocation_failed",
                strategy_id=self._strategy_id,
                instrument=str(signal.instrument_id),
                exc_info=True,
                error=str(exc),
            )
            return proposed_value

        # Get allocation for this signal's instrument
        allocated_value = allocations.get(signal.instrument_id)
        if allocated_value is None:
            return proposed_value

        return max(float(allocated_value), 0.0)

    # -------------------------------------------------------------------------
    # Main Sizing Method with Risk Validation
    # -------------------------------------------------------------------------

    def size_and_validate(
        self,
        signal: MLSignal,
        current_position: Position | None = None,
    ) -> Quantity | None:
        """
        Comprehensive position sizing with risk validation and portfolio allocation.

        This method implements the full position sizing workflow:
        1. Use position sizer if available, else fall back to basic sizing
        2. Apply portfolio allocation constraints
        3. Validate with risk manager
        4. Convert value to instrument-aware quantity

        Parameters
        ----------
        signal : MLSignal
            The ML signal triggering the position.
        current_position : Position | None, optional
            The current open position, if any.

        Returns
        -------
        Quantity | None
            The validated position size, or None if the trade should not proceed.

        Examples
        --------
        >>> quantity = component.size_and_validate(signal)
        >>> if quantity is not None:
        ...     # Proceed with trade
        ...     pass

        """
        from nautilus_trader.model.objects import Quantity

        if self._cache is None:
            self._log.error("Cache not available for position sizing")
            return None

        if self._instrument_id is None:
            self._log.error("Instrument ID not configured for position sizing")
            return None

        # Resolve instrument and account
        instrument = self._cache.instrument(self._instrument_id)
        if instrument is None:
            self._log.error(
                f"Instrument {self._instrument_id} not found in cache",
            )
            return None

        account = self._cache.account_for_venue(instrument.venue)
        if account is None:
            self._log.error(f"No account for venue {instrument.venue}")
            return None

        # Resolve market price
        market_price = self.resolve_market_price(self._instrument_id)
        if market_price is None:
            return None

        # Gather current open positions
        positions: list[Position] = self._cache.positions_open(
            venue=None,
            instrument_id=self._instrument_id,
        )

        # Step 1: Calculate position size using position sizer or fallback
        proposed_value_qty: Quantity | None = None
        if self._position_sizer is not None:
            try:
                proposed_value_qty = self._position_sizer.calculate(
                    signal,
                    account,
                    positions,
                )
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.position_sizer_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )

        # Fallback to basic sizing if sizer returns None or not configured
        if proposed_value_qty is None:
            proposed_value_qty = self.calculate_position_size()

        if proposed_value_qty is None:
            return None

        # Step 2: Apply portfolio allocation
        proposed_value = float(proposed_value_qty.as_double()) * market_price
        allocated_value = self.apply_portfolio_allocation(
            signal=signal,
            proposed_value=proposed_value,
            account=account,
        )

        # Check for zero allocation
        if allocated_value <= 0.0:
            self._log.debug(
                "ml_strategy.portfolio_allocation_zero",
                strategy_id=self._strategy_id,
                instrument=str(signal.instrument_id),
            )
            return None

        # Scale quantity if allocation is less than proposed
        if proposed_value > 0.0 and allocated_value < proposed_value:
            scale = allocated_value / proposed_value
            scaled_qty = max(
                float(proposed_value_qty.as_double()) * scale,
                float(instrument.min_quantity.as_double()),
            )
            precision = instrument.size_precision
            proposed_value_qty = Quantity.from_str(str(round(scaled_qty, precision)))

        # Step 3: Risk manager validation
        approved_value_qty: Quantity | None = proposed_value_qty
        if self._risk_manager is not None:
            try:
                approved_value_qty = self._risk_manager.check_position(
                    proposed_size=proposed_value_qty,
                    instrument=instrument.id,
                    portfolio=self._portfolio,
                )
            except Exception as exc:
                self._log.debug(
                    "ml_strategy.risk_manager_failed",
                    strategy_id=self._strategy_id,
                    exc_info=True,
                    error=str(exc),
                )
                return None

        if approved_value_qty is None:
            return None

        # Step 4: Convert approved value to quantity using market price
        # Re-resolve price to ensure freshness
        current_price = self.resolve_market_price(self._instrument_id)
        if current_price is None:
            return None

        # Convert quantity value to proper quantity
        val = float(approved_value_qty.as_double())
        return self.value_to_quantity(
            value=val * current_price,  # Convert back to value for consistent conversion
            price=current_price,
            instrument=instrument,
        )


__all__ = [
    "CacheProtocol",
    "LoggerProtocol",
    "PortfolioManagerProtocol",
    "PositionManagementComponent",
    "PositionSizerProtocol",
    "RiskManagerProtocol",
]
