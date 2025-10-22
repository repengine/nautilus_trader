"""
Smart order execution for ML trading strategies.

This module provides intelligent order placement based on signal confidence,
market conditions, and fee optimization. Designed for small accounts with
focus on minimizing costs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import StrategyId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ml.actors.base import MLSignal
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.orders import Order


try:  # Pragmatic compatibility for downstream callers and tests
    from nautilus_trader.model.orders import LimitOrder as _NTLimitOrder

    if not hasattr(_NTLimitOrder, "post_only"):
        setattr(_NTLimitOrder, "post_only", property(lambda self: bool(self.is_post_only)))
except Exception as exc:  # pragma: no cover - optional enhancement if C-extension unavailable
    logging.getLogger(__name__).debug(
        "LimitOrder post_only augmentation skipped: %s",
        exc,
        exc_info=True,
    )

# ===== Metrics =====
orders_created_total = get_counter(
    "ml_orders_created_total",
    "Total orders created",
    labels=["order_type", "urgency"],
)

order_creation_latency_seconds = get_histogram(
    "ml_order_creation_latency_seconds",
    "Order creation latency",
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005],
    labels=["order_type"],
)

fee_savings_total = get_counter(
    "ml_fee_savings_total",
    "Cumulative fee savings from smart execution",
    labels=["method"],
)


@dataclass(slots=True)
class _OrderIds:
    trader_id: TraderId
    strategy_id: StrategyId
    client_order_id: ClientOrderId
    init_id: UUID4
    ts_init: int


@dataclass(slots=True)
class _OrderResult:
    """Wrapper exposing additional metadata while delegating attribute access."""

    native: Order
    post_only: bool | None = None

    def unwrap(self) -> Order:
        return self.native

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

# ===== Configuration =====
@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for order execution."""

    # Confidence thresholds for order type selection
    market_order_threshold: float = 0.9  # Use market above 90% confidence
    limit_order_threshold: float = 0.7   # Use limit between 70-90%
    min_confidence: float = 0.5          # Don't trade below 50%

    # Limit order settings
    limit_offset_bps: int = 5            # Basis points offset for limit orders
    aggressive_offset_bps: int = 2       # Aggressive limit (closer to market)
    passive_offset_bps: int = 10         # Passive limit (further from market)

    # Time management
    limit_order_ttl_seconds: int = 60      # Time to live for limit orders
    use_time_in_force_ioc: bool = True     # Use IOC for immediate or cancel attempts
    ttl_max_attempts: int = 3               # Max cancel–replace attempts for resting limits
    ttl_replace_cadence_seconds: int = 5    # Replace cadence when waiting (seconds)

    # Fee optimization
    prefer_maker_orders: bool = True      # Prefer maker fees when available
    max_spread_bps: int = 20              # Avoid limits when very wide spreads
    maker_fee_bps: float = 2.0            # Venue maker fee in bps
    taker_fee_bps: float = 4.0            # Venue taker fee in bps
    prefer_maker_spread_bps: int = 5      # Spread threshold to prefer maker even at high confidence


# ===== Order Executor =====
class OrderExecutor:
    """
    Smart order execution based on signal confidence and market conditions.

    Selects appropriate order types, manages timing, and optimizes for fees.
    """

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        """
        Initialize order executor.

        Parameters
        ----------
        config : ExecutionConfig, optional
            Execution configuration.

        """
        self.config = config or ExecutionConfig()

        # Track execution statistics
        self._total_orders: int = 0
        self._market_orders: int = 0
        self._limit_orders: int = 0
        self._fee_savings: float = 0.0
        self._last_ttl_plan: dict[str, float | int | str] | None = None

    def create_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        signal: MLSignal,
        market_state: dict[str, float],
        instrument: Instrument,
        *,
        trader_id: TraderId | None = None,
        strategy_id: StrategyId | None = None,
        client_order_id: ClientOrderId | None = None,
        init_id: UUID4 | None = None,
        ts_init: int | None = None,
    ) -> Order | None:
        """
        Create an order based on signal confidence and market conditions.

        Parameters
        ----------
        side : OrderSide
            Order side (BUY or SELL).
        quantity : Quantity
            Position size.
        signal : MLSignal
            The ML signal with confidence.
        market_state : dict[str, float]
            Current market state (bid, ask, spread, etc).
        instrument : Instrument
            The instrument to trade.

        Returns
        -------
        Order | None
            The order to submit, or None if conditions not met.

        """
        start_time = time.perf_counter()

        try:
            # Check minimum confidence
            if signal.confidence < self.config.min_confidence:
                logger.info(
                    f"Signal confidence {signal.confidence:.2f} below minimum "
                    f"{self.config.min_confidence:.2f}, not trading"
                )
                return None

            # Get market prices
            bid = market_state.get("bid", 0.0)
            ask = market_state.get("ask", 0.0)
            spread_bps = market_state.get("spread_bps", 0.0)

            if bid <= 0 or ask <= 0:
                logger.error("Invalid market prices, cannot create order")
                return None

            # Determine urgency level (with fee/spread calibration)
            urgency = self._determine_urgency(signal.confidence, spread_bps)
            if urgency == "high" and self._should_prefer_maker(spread_bps):
                # Downgrade one level to try a limit first when spread is tight
                urgency = "medium"

            # Select order type based on urgency
            order_ids = self._resolve_order_ids(
                trader_id=trader_id,
                strategy_id=strategy_id,
                client_order_id=client_order_id,
                init_id=init_id,
                ts_init=ts_init,
            )

            post_only_flag: bool | None = None

            if urgency == "high":
                order = self._create_market_order(
                    side=side,
                    quantity=quantity,
                    instrument=instrument,
                    order_ids=order_ids,
                )
                order_type = "market"

            elif urgency == "medium":
                order = self._create_aggressive_limit(
                    side=side,
                    quantity=quantity,
                    bid=bid,
                    ask=ask,
                    instrument=instrument,
                    order_ids=order_ids,
                )
                order_type = "limit_aggressive"
                post_only_flag = False

            else:  # low urgency
                order = self._create_passive_limit(
                    side=side,
                    quantity=quantity,
                    bid=bid,
                    ask=ask,
                    instrument=instrument,
                    order_ids=order_ids,
                )
                order_type = "limit_passive"
                post_only_flag = self.config.prefer_maker_orders

            # Track metrics
            if order:
                self._total_orders += 1
                orders_created_total.labels(order_type=order_type, urgency=urgency).inc()
                return _OrderResult(order, post_only=post_only_flag)

            return None

        finally:
            order_creation_latency_seconds.labels(order_type="smart").observe(
                time.perf_counter() - start_time
            )

    def _determine_urgency(self, confidence: float, spread_bps: float) -> str:
        """
        Determine order urgency based on confidence and spread.

        Parameters
        ----------
        confidence : float
            Signal confidence.
        spread_bps : float
            Current bid-ask spread in basis points.

        Returns
        -------
        str
            Urgency level: "high", "medium", or "low".

        """
        # High confidence or tight spread -> urgent execution
        if confidence >= self.config.market_order_threshold:
            return "high"

        # Wide spread -> be patient
        if spread_bps > self.config.max_spread_bps:
            return "low"

        # Medium confidence -> balanced approach
        if confidence >= self.config.limit_order_threshold:
            return "medium"

        return "low"

    def _should_prefer_maker(self, spread_bps: float) -> bool:
        """
        Decide if we should actively prefer maker orders given current spread.

        Returns True when maker preference is enabled and the spread is tight
        (<= prefer_maker_spread_bps), making queue placement attractive.
        """
        return bool(
            self.config.prefer_maker_orders and spread_bps <= float(self.config.prefer_maker_spread_bps)
        )

    def _resolve_order_ids(
        self,
        *,
        trader_id: TraderId | None,
        strategy_id: StrategyId | None,
        client_order_id: ClientOrderId | None,
        init_id: UUID4 | None,
        ts_init: int | None,
    ) -> _OrderIds:
        resolved_init = init_id or UUID4()
        resolved_ts = ts_init if ts_init is not None else int(time.time_ns())
        resolved_trader = trader_id or TraderId(str(UUID4()))
        resolved_strategy = strategy_id or StrategyId(str(UUID4()))
        resolved_client_id = client_order_id or ClientOrderId(str(UUID4()))
        return _OrderIds(
            trader_id=resolved_trader,
            strategy_id=resolved_strategy,
            client_order_id=resolved_client_id,
            init_id=resolved_init,
            ts_init=resolved_ts,
        )

    def _create_market_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        instrument: Instrument,
        *,
        order_ids: _OrderIds,
    ) -> Order:
        """
        Create a market order for immediate execution.

        Parameters
        ----------
        side : OrderSide
            Order side.
        quantity : Quantity
            Order quantity.
        instrument : Instrument
            Instrument to trade.

        Returns
        -------
        MarketOrder
            Market order for immediate execution.

        """
        from nautilus_trader.model.orders import MarketOrder

        self._market_orders += 1

        return MarketOrder(
            trader_id=order_ids.trader_id,
            strategy_id=order_ids.strategy_id,
            instrument_id=instrument.id,
            client_order_id=order_ids.client_order_id,
            order_side=side,
            quantity=quantity,
            init_id=order_ids.init_id,
            ts_init=order_ids.ts_init,
            time_in_force=TimeInForce.IOC if self.config.use_time_in_force_ioc else TimeInForce.GTC,
        )

    def _create_aggressive_limit(
        self,
        side: OrderSide,
        quantity: Quantity,
        bid: float,
        ask: float,
        instrument: Instrument,
        *,
        order_ids: _OrderIds,
    ) -> Order:
        """
        Create an aggressive limit order (close to market).

        Parameters
        ----------
        side : OrderSide
            Order side.
        quantity : Quantity
            Order quantity.
        bid : float
            Current bid price.
        ask : float
            Current ask price.
        instrument : Instrument
            Instrument to trade.

        Returns
        -------
        LimitOrder
            Aggressive limit order.

        """
        from nautilus_trader.model.orders import LimitOrder

        self._limit_orders += 1

        # Calculate aggressive price (closer to market)
        offset_bps = self.config.aggressive_offset_bps / 10000

        if side == OrderSide.BUY:
            # Buy slightly below ask
            limit_price = ask * (1 - offset_bps)
        else:
            # Sell slightly above bid
            limit_price = bid * (1 + offset_bps)

        # Round to instrument precision
        limit_price = self._round_price(limit_price, instrument)

        order = LimitOrder(
            trader_id=order_ids.trader_id,
            strategy_id=order_ids.strategy_id,
            instrument_id=instrument.id,
            client_order_id=order_ids.client_order_id,
            order_side=side,
            quantity=quantity,
            price=Price.from_str(str(limit_price)),
            init_id=order_ids.init_id,
            ts_init=order_ids.ts_init,
            time_in_force=TimeInForce.IOC if self.config.use_time_in_force_ioc else TimeInForce.GTC,
            post_only=False,  # Aggressive, not post-only
        )
        self._record_ttl_plan(
            order_type="limit_aggressive",
            will_rest=(not self.config.use_time_in_force_ioc),
        )
        return order

    def _create_passive_limit(
        self,
        side: OrderSide,
        quantity: Quantity,
        bid: float,
        ask: float,
        instrument: Instrument,
        *,
        order_ids: _OrderIds,
    ) -> Order:
        """
        Create a passive limit order (maker fee).

        Parameters
        ----------
        side : OrderSide
            Order side.
        quantity : Quantity
            Order quantity.
        bid : float
            Current bid price.
        ask : float
            Current ask price.
        instrument : Instrument
            Instrument to trade.

        Returns
        -------
        LimitOrder
            Passive limit order.

        """
        from nautilus_trader.model.orders import LimitOrder

        self._limit_orders += 1

        # Calculate passive price (join the queue)
        offset_bps = self.config.passive_offset_bps / 10000

        if side == OrderSide.BUY:
            # Buy at or below bid (maker)
            limit_price = bid * (1 - offset_bps)
        else:
            # Sell at or above ask (maker)
            limit_price = ask * (1 + offset_bps)

        # Round to instrument precision
        limit_price = self._round_price(limit_price, instrument)

        # Track potential fee savings
        if self.config.prefer_maker_orders:
            # Estimate fee savings (maker vs taker)
            # Typical: maker 0.02%, taker 0.04%
            fee_diff = max(0.0, (self.config.taker_fee_bps - self.config.maker_fee_bps)) / 10_000.0
            saved = float(quantity.as_double()) * limit_price * fee_diff
            self._fee_savings += saved
            fee_savings_total.labels(method="maker_order").inc(saved)

        order = LimitOrder(
            trader_id=order_ids.trader_id,
            strategy_id=order_ids.strategy_id,
            instrument_id=instrument.id,
            client_order_id=order_ids.client_order_id,
            order_side=side,
            quantity=quantity,
            price=Price.from_str(str(limit_price)),
            init_id=order_ids.init_id,
            ts_init=order_ids.ts_init,
            time_in_force=TimeInForce.GTC,  # Let it sit
            post_only=self.config.prefer_maker_orders,
        )
        self._record_ttl_plan(order_type="limit_passive", will_rest=True)
        return order

    def _round_price(self, price: float, instrument: Instrument) -> float:
        """
        Round price to instrument precision.

        Parameters
        ----------
        price : float
            Raw price.
        instrument : Instrument
            Instrument for precision.

        Returns
        -------
        float
            Rounded price.

        """
        precision: int = int(getattr(instrument, "price_precision", 0))
        return round(price, precision)

    def get_execution_stats(self) -> dict[str, float]:
        """
        Get execution statistics.

        Returns
        -------
        dict[str, float]
            Execution statistics.

        """
        return {
            "total_orders": float(self._total_orders),
            "market_orders": float(self._market_orders),
            "limit_orders": float(self._limit_orders),
            "market_order_pct": self._market_orders / max(self._total_orders, 1),
            "limit_order_pct": self._limit_orders / max(self._total_orders, 1),
            "estimated_fee_savings": self._fee_savings,
        }

    def _record_ttl_plan(self, *, order_type: str, will_rest: bool) -> None:
        """
        Record a simple TTL/cancel–replace plan for the last created limit order.

        This does not schedule anything; strategies or services can read the
        plan and implement scheduling off the hot path. For IOC orders
        (will_rest=False), the attempts default to 0.
        """
        attempts = int(self.config.ttl_max_attempts) if will_rest else 0
        self._last_ttl_plan = {
            "order_type": order_type,
            "ttl_seconds": float(self.config.limit_order_ttl_seconds),
            "attempts": attempts,
            "cadence_seconds": float(self.config.ttl_replace_cadence_seconds),
        }

    def get_last_ttl_plan(self) -> dict[str, float | int | str] | None:
        """Return the last recorded TTL plan (if any)."""
        return self._last_ttl_plan


# ===== Public API =====
__all__ = [
    "ExecutionConfig",
    "OrderExecutor",
]
