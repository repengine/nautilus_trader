"""
Facade for MLTradingStrategy decomposition.

This module provides BaseMLStrategyFacade, a facade that wires together
the 6 extracted strategy components while maintaining backward compatibility
with the legacy BaseMLStrategy API.

Components Wired:
- SignalRoutingComponent: Signal filtering, aggregation, and routing
- DecisionPersistenceComponent: Strategy decision persistence and event publishing
- PositionManagementComponent: Position sizing, risk validation, portfolio allocation
- OrderSubmissionComponent: Order creation, smart execution, stop loss placement
- LifecycleComponent: Strategy lifecycle management (startup, shutdown, subscriptions)
- PerformanceTrackingComponent: Model performance tracking and metrics recording

"""

from __future__ import annotations

import time
from abc import ABC
from abc import abstractmethod
from collections import deque
from collections.abc import Callable
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml.common import decision_from_probability
from ml.config.base import AccountMode
from ml.config.base import ReturnsUpdateMode
from ml.config.base import ShortEntryPolicy
from ml.strategies.common import DecisionPersistenceComponent
from ml.strategies.common import LifecycleComponent
from ml.strategies.common import OrderIntentPositionTracker
from ml.strategies.common import OrderSubmissionComponent
from ml.strategies.common import PerformanceTrackingComponent
from ml.strategies.common import PositionManagementComponent
from ml.strategies.common import SignalRoutingComponent
from ml.strategies.common.decision_persistence import _SafeLogger
from ml.strategies.common.order_submission import OrderIntentWriter
from ml.strategies.common.order_submission import resolve_order_intent_path


if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.identifiers import ClientOrderId
    from nautilus_trader.model.objects import Price
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.model.position import Position

    from ml.actors.base import MLSignal
    from ml.config.base import MLStrategyConfig
    from ml.stores.protocols import StrategyStoreProtocol
    from ml.strategies.common.positions import PositionsHealthStatus
    from ml.strategies.common.positions import PositionsProviderProtocol
    from ml.strategies.common.positions import PositionViewProtocol
    from ml.strategies.risk import RiskActionDecision
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.events import PositionClosed


# Import the runtime base class for inheritance
try:
    from nautilus_trader.trading.strategy import Strategy as _RuntimeStrategy
except Exception as exc:  # pragma: no cover - dependency guard
    raise ImportError("nautilus_trader is required for ML strategies") from exc

if TYPE_CHECKING:
    from nautilus_trader.trading.strategy import Strategy as StrategyBase
else:
    StrategyBase = cast(type[Any], _RuntimeStrategy)


class BaseMLStrategyFacade(StrategyBase, ABC):  # type: ignore[misc]
    """
    Facade that wires all 6 strategy components together.

    Maintains backward compatibility with BaseMLStrategy API while
    delegating to focused, testable components.

    Components:
    - SignalRoutingComponent: Signal filtering and aggregation
    - DecisionPersistenceComponent: Store writes and event publishing
    - PositionManagementComponent: Position sizing and risk validation
    - OrderSubmissionComponent: Order creation and submission
    - LifecycleComponent: Subscription and lifecycle management
    - PerformanceTrackingComponent: Model performance tracking

    Parameters
    ----------
    config : MLStrategyConfig
        The configuration for the ML strategy.
    stores : object | None, optional
        Container with stores and registries from init_ml_stores_and_registries.

    Examples
    --------
    >>> config = MLStrategyConfig(
    ...     instrument_id=InstrumentId.from_str("EURUSD.SIM"),
    ...     position_size_pct=0.05,
    ...     min_confidence=0.6,
    ... )
    >>> strategy = ConcreteMLStrategy(config)

    """

    def __init__(self, config: MLStrategyConfig, stores: object | None = None) -> None:
        """
        Initialize the ML strategy facade with dependency injection support.

        Parameters
        ----------
        config : MLStrategyConfig
            The configuration for the ML strategy.
        stores : ActorStoresRegistries, optional
            Container with all 4 stores and 4 registries.

        """
        super().__init__(config)
        self._config = config
        self._stores = stores

        # Trading state (same as legacy for backward compatibility)
        self._active_positions = 0
        self._pending_orders = 0
        self._last_signal_time = 0
        self._signals_received = 0
        self._trades_executed = 0
        self._winning_trades = 0
        self._total_pnl = Decimal("0.0")
        self._dry_run_trades = 0
        self._last_risk_halt_state: bool | None = None
        self._last_risk_halt_reason: str | None = None
        self._last_liquidation_no_positions_ts: int | None = None

        # Signal management (same as legacy)
        history_size = getattr(config, "history_size", 100)
        self._signal_history: deque[MLSignal] = deque(maxlen=history_size)
        self._signal_buffer: dict[str, MLSignal] = {}
        self._model_signals: dict[str, MLSignal] = {}
        self._model_performance: dict[str, dict[str, Any]] = {}

        # Expose legacy attributes for backward compatibility
        self.target_model_ids: list[str] | None = getattr(config, "target_model_ids", None)
        self.aggregation_mode: str | None = getattr(config, "aggregation_mode", None)
        self.required_models: int = getattr(config, "required_models", 1)
        self.time_window_ms: int = getattr(config, "time_window_ms", 1000)
        self.conflict_resolution: str | None = getattr(config, "conflict_resolution", None)
        self.model_weights: dict[str, float] = getattr(config, "model_weights", {})
        self.track_performance: bool = getattr(config, "track_performance", False)

        # Initialize strategy store
        self.strategy_store: StrategyStoreProtocol | None = None
        self._init_strategy_store()

        # Metrics placeholders (initialized in _init_metrics; Optional for fallback path)
        self.signals_received_metric: Any | None = None
        self.orders_submitted_metric: Any | None = None
        self.position_count_metric: Any | None = None
        self._strategy_decisions_persisted: Any | None = None
        self._strategy_store_write_latency: Any | None = None
        self._strategy_store_batch_size: Any | None = None
        self._signal_to_trade_latency: Any | None = None

        # Initialize metrics (same as legacy)
        self._init_metrics()

        # Initialize optional components (position sizer, risk manager, etc.)
        self.position_sizer: Any = None
        self.risk_manager: Any = None
        self.portfolio_manager: Any = None
        self.order_executor: Any = None
        self.performance: Any = None
        self._returns_updater: Any | None = None
        self._store_breaker: Any = None
        self._order_breaker: Any = None
        self._bus_publisher: Any = None
        self._bus_bridge: Any = None
        self._decision_publisher: Any = None
        self._init_optional_components()
        self._positions_provider: PositionsProviderProtocol | None = None
        self._positions_health: PositionsHealthStatus | None = None
        self._init_positions_provider()
        self._order_intent_writer: OrderIntentWriter | None = None
        self._order_intent_path: Path | None = None
        self._intent_position_tracker: OrderIntentPositionTracker | None = None
        self._pending_exit_metadata: dict[str, object] | None = None
        self._pending_execution_metadata: dict[str, object] | None = None
        self._init_intent_position_tracker()
        self._returns_bar_subscribed = False
        self._returns_bar_subscription_attempted = False
        self._init_order_intent_writer()

        # Initialize the 6 decomposed components
        self._signal_router: SignalRoutingComponent | None = None
        self._decision_persister: DecisionPersistenceComponent | None = None
        self._position_manager: PositionManagementComponent | None = None
        self._order_submitter: OrderSubmissionComponent | None = None
        self._lifecycle: LifecycleComponent | None = None
        self._performance_tracker: PerformanceTrackingComponent | None = None
        self._init_components()

    def _init_order_intent_writer(self) -> None:
        """
        Initialize order intent serialization when configured.
        """
        if not getattr(self._config, "serialize_order_intents", False):
            return

        try:
            resolved = resolve_order_intent_path(
                getattr(self._config, "order_intent_path", None),
            )
            if resolved is None:
                self.log.warning(
                    f"ml_strategy.order_intent_path_missing strategy_id={self.id}",
                )
                return
            self._order_intent_path = resolved
            self._order_intent_writer = OrderIntentWriter(resolved, log=self.log)
            self.log.info(
                f"ml_strategy.order_intent_serialization_enabled "
                f"strategy_id={self.id} path={resolved}",
            )
        except Exception as exc:
            self.log.debug(
                f"ml_strategy.order_intent_writer_init_failed "
                f"strategy_id={self.id} error={exc}",
                exc_info=True,
            )

    def _init_intent_position_tracker(self) -> None:
        """
        Initialize the in-memory tracker for order intent positions.
        """
        if not getattr(self._config, "serialize_order_intents", False):
            return
        self._intent_position_tracker = OrderIntentPositionTracker(log=self.log)

    def _should_block_entry_orders(self) -> bool:
        """
        Return True when entry orders should be blocked in intent mode.
        """
        if not getattr(self._config, "serialize_order_intents", False):
            return False
        max_positions = getattr(self._config, "max_positions", 1)
        try:
            max_positions_int = int(max_positions)
        except (TypeError, ValueError):
            return False
        if max_positions_int <= 0:
            return True
        tracker = self._intent_position_tracker
        if tracker is None:
            return self._active_positions >= max_positions_int
        return tracker.active_positions >= max_positions_int

    def _resolve_short_entry_policy(self) -> ShortEntryPolicy:
        """
        Resolve the short-entry policy from config defaults.
        """
        policy = getattr(self._config, "short_entry_policy", None)
        if isinstance(policy, ShortEntryPolicy):
            return policy
        if isinstance(policy, str):
            try:
                return ShortEntryPolicy(policy)
            except ValueError:
                pass
        account_mode = getattr(self._config, "account_mode", None)
        if isinstance(account_mode, str):
            try:
                account_mode = AccountMode(account_mode)
            except ValueError:
                account_mode = AccountMode.CASH
        if account_mode is AccountMode.MARGIN:
            return ShortEntryPolicy.ALLOW
        return ShortEntryPolicy.EXIT_ONLY

    def _update_intent_positions(self, order: Any) -> None:
        """
        Update intent positions from a serialized order.
        """
        if self._intent_position_tracker is None:
            return

        side = getattr(order, "side", getattr(order, "order_side", None))
        instrument_id = getattr(order, "instrument_id", None)
        quantity = getattr(order, "quantity", None)
        reduce_only = bool(getattr(order, "is_reduce_only", getattr(order, "reduce_only", False)))
        ts_init = getattr(order, "ts_init", None)

        if instrument_id is None or side is None or quantity is None:
            return

        try:
            from nautilus_trader.model.enums import OrderSide
        except Exception:
            return

        if not isinstance(side, OrderSide):
            name = getattr(side, "name", None)
            if isinstance(name, str):
                try:
                    side = OrderSide[name.upper()]
                except Exception:
                    return
            elif isinstance(side, str):
                try:
                    side = OrderSide[side.upper()]
                except Exception:
                    return
            else:
                return

        entry_price = None
        for attr in ("price", "limit_price", "trigger_price"):
            value = getattr(order, attr, None)
            if value is None:
                continue
            try:
                entry_price = float(value)
            except (TypeError, ValueError):
                entry_price = None
            if entry_price is not None and entry_price > 0.0:
                break

        self._intent_position_tracker.record_order(
            instrument_id=instrument_id,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            ts_init=ts_init,
            entry_price=entry_price,
        )
        self._active_positions = self._intent_position_tracker.active_positions

    def _resolve_submit_order_callback(self) -> Callable[[Any], None] | None:
        """
        Resolve the order submission callback based on configuration.

        When order intent serialization is enabled, returns the JSONL writer stub to
        avoid broker submission.

        """
        if getattr(self._config, "serialize_order_intents", False):
            if self._order_intent_writer is None:
                return None
            return self._record_order_intent
        if hasattr(self, "submit_order"):
            return cast(Callable[[Any], None], self.submit_order)
        return None

    def _resolve_cancel_order_callback(self) -> Callable[[Any], None] | None:
        """
        Resolve the order cancellation callback based on configuration.

        Returns None when order intent serialization is enabled.

        """
        if getattr(self._config, "serialize_order_intents", False):
            return None
        if hasattr(self, "cancel_order"):
            return cast(Callable[[Any], None], self.cancel_order)
        return None

    def _record_order_intent(self, order: Any) -> None:
        """
        Persist order intent to JSONL for manual inspection.
        """
        if self._order_intent_writer is None:
            return
        from ml.strategies.common.positions import build_positions_metadata

        is_live = True
        try:
            if hasattr(self, "cache") and getattr(self.cache, "is_backtesting", False):
                is_live = False
        except Exception as exc:
            self.log.debug(
                f"ml_strategy.order_intent_live_check_failed " f"strategy_id={self.id} error={exc}",
                exc_info=True,
            )
        quote_metadata = None
        if self._order_submitter is not None:
            quote_metadata = self._order_submitter.pop_last_quote_metadata()
        execution_metadata = self._pending_execution_metadata
        self._pending_execution_metadata = None
        if self._order_submitter is not None:
            submitter_metadata = self._order_submitter.pop_last_execution_metadata()
            if submitter_metadata is not None:
                execution_metadata = submitter_metadata
        exit_metadata = self._pending_exit_metadata
        self._pending_exit_metadata = None
        self._update_intent_positions(order)
        self._order_intent_writer.write(
            order,
            is_live=is_live,
            positions_metadata=build_positions_metadata(self._positions_health),
            quote_metadata=quote_metadata,
            exit_metadata=exit_metadata,
            execution_metadata=execution_metadata,
        )

    def _set_exit_intent_metadata(self, metadata: Mapping[str, object] | None) -> None:
        """
        Set pending exit metadata for order intent serialization.

        Parameters
        ----------
        metadata : Mapping[str, object] | None
            Exit metadata payload to attach to the next order intent.

        """
        if not getattr(self._config, "serialize_order_intents", False):
            return
        self._pending_exit_metadata = dict(metadata) if metadata is not None else None

    def _init_strategy_store(self) -> None:
        """
        Initialize strategy store via dependency injection or config.
        """
        # First try to get stores from injected container
        if self._stores is not None and hasattr(self._stores, "strategy_store"):
            self.strategy_store = self._stores.strategy_store
        # Fall back to centralized initialization via integration helper
        elif self._config.use_strategy_store:
            try:
                from ml.core.integration import init_ml_stores_and_registries as _init_stores

                stores = _init_stores(self._config)
                self.strategy_store = cast(
                    "StrategyStoreProtocol | None",
                    getattr(stores, "strategy_store", None),
                )
            except Exception as exc:
                self.log.warning(
                    "ml_strategy.strategy_store_unavailable",
                    strategy_id=str(self.id),
                    exc_info=True,
                    error=str(exc),
                )
                self.strategy_store = None

    def _init_metrics(self) -> None:
        """
        Initialize Prometheus metrics for monitoring.
        """
        from ml.config.names import LABEL_INSTRUMENT
        from ml.config.names import LABEL_ORDER_SIDE
        from ml.config.names import LABEL_SIGNAL_SOURCE
        from ml.config.names import LABEL_STRATEGY_ID
        from ml.config.names import METRIC_POSITION_COUNT
        from ml.config.names import METRIC_SIGNAL_TO_TRADE_LATENCY_SECONDS
        from ml.config.names import METRIC_SIGNALS_RECEIVED_TOTAL
        from ml.config.names import METRIC_STRATEGY_DECISIONS_PERSISTED_TOTAL
        from ml.config.names import METRIC_STRATEGY_STORE_BATCH_SIZE
        from ml.config.names import METRIC_STRATEGY_STORE_WRITE_LATENCY_SECONDS
        from ml.config.names import METRIC_TRADES_EXECUTED_TOTAL

        try:
            from ml.common.metrics_manager import MetricsManager

            mm = MetricsManager.default()

            self.signals_received_metric = mm.counter(
                METRIC_SIGNALS_RECEIVED_TOTAL,
                "Total number of ML signals received",
                [LABEL_STRATEGY_ID, LABEL_SIGNAL_SOURCE],
            )
            self.orders_submitted_metric = mm.counter(
                METRIC_TRADES_EXECUTED_TOTAL,
                "Total number of trades executed based on ML signals",
                [LABEL_STRATEGY_ID, LABEL_ORDER_SIDE],
            )
            self.position_count_metric = mm.gauge(
                METRIC_POSITION_COUNT,
                "Current number of open positions",
                [LABEL_STRATEGY_ID, LABEL_INSTRUMENT],
            )
            self._strategy_decisions_persisted = mm.counter(
                METRIC_STRATEGY_DECISIONS_PERSISTED_TOTAL,
                "Total number of strategy decisions persisted to store",
                [LABEL_STRATEGY_ID],
            )
            self._strategy_store_write_latency = mm.histogram(
                METRIC_STRATEGY_STORE_WRITE_LATENCY_SECONDS,
                "Latency of writing to strategy store",
                [LABEL_STRATEGY_ID],
            )
            self._strategy_store_batch_size = mm.gauge(
                METRIC_STRATEGY_STORE_BATCH_SIZE,
                "Current batch size in strategy store buffer",
                [LABEL_STRATEGY_ID],
            )
            self._signal_to_trade_latency = mm.histogram(
                METRIC_SIGNAL_TO_TRADE_LATENCY_SECONDS,
                "Latency from signal reception to trade execution",
                [LABEL_STRATEGY_ID],
            )
        except Exception as exc:
            self.log.debug(
                "ml_strategy.metrics_init_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )
            # Set all metrics to None if initialization fails
            self.signals_received_metric = None
            self.orders_submitted_metric = None
            self.position_count_metric = None
            self._strategy_decisions_persisted = None
            self._strategy_store_write_latency = None
            self._strategy_store_batch_size = None
            self._signal_to_trade_latency = None

    def _init_optional_components(self) -> None:
        """
        Initialize optional sub-components (position sizer, risk manager, etc.).
        """
        try:
            from ml.strategies.analytics import AnalyticsConfig
            from ml.strategies.analytics import PerformanceTracker
            from ml.strategies.execution import ExecutionConfig
            from ml.strategies.execution import OrderExecutor
            from ml.strategies.portfolio import PortfolioConfig
            from ml.strategies.portfolio import PortfolioManager
            from ml.strategies.risk import RiskConfig
            from ml.strategies.risk import RiskManager
            from ml.strategies.sizing import CompositeSizer
            from ml.strategies.sizing import SizingConfig

            sizing_cfg = getattr(self._config, "sizing_config", None)
            risk_cfg = getattr(self._config, "risk_config", None)
            exec_cfg = getattr(self._config, "execution_config", None)
            port_cfg = getattr(self._config, "portfolio_config", None)
            analytics_cfg = getattr(self._config, "analytics_config", None)

            self.position_sizer = CompositeSizer(
                cast("SizingConfig | None", sizing_cfg),
            )
            self.risk_manager = RiskManager(
                cast("RiskConfig | None", risk_cfg),
            )
            self.portfolio_manager = PortfolioManager(
                cast("PortfolioConfig | None", port_cfg),
            )
            self.order_executor = OrderExecutor(
                cast("ExecutionConfig | None", exec_cfg),
            )
            self.performance = PerformanceTracker(
                cast("AnalyticsConfig | None", analytics_cfg),
            )

            if (
                self.risk_manager is not None
                and hasattr(self.risk_manager, "set_correlation_provider")
                and self.portfolio_manager is not None
            ):
                try:
                    from ml.strategies.common.correlation import CorrelationProviderProtocol

                    if isinstance(self.portfolio_manager, CorrelationProviderProtocol):
                        self.risk_manager.set_correlation_provider(self.portfolio_manager)
                except Exception as exc:
                    self.log.debug(
                        "ml_strategy.correlation_provider_init_failed",
                        strategy_id=str(self.id),
                        exc_info=True,
                        error=str(exc),
                    )
        except Exception as exc:
            self.log.debug(
                "ml_strategy.optional_components_unavailable",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )

        try:
            from ml.config.base import ReturnsConfig
            from ml.strategies.common import ReturnsUpdater

            returns_cfg = getattr(self._config, "returns_config", None)
            if returns_cfg is None:
                max_age_ms = getattr(self._config, "max_quote_age_ms", None)
                returns_cfg = ReturnsConfig(
                    max_price_age_ms=max_age_ms,
                    update_mode=ReturnsUpdateMode.BOTH,
                )
            if returns_cfg is not None and (
                self.position_sizer is not None or self.portfolio_manager is not None
            ):
                self._returns_updater = ReturnsUpdater(
                    config=returns_cfg,
                    position_sizer=self.position_sizer,
                    portfolio_manager=self.portfolio_manager,
                    log=self.log,
                    strategy_id=str(self.id),
                )
        except Exception as exc:
            self._returns_updater = None
            self.log.debug(
                "ml_strategy.returns_updater_init_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )

        # Initialize circuit breakers
        try:
            from ml.actors.base import CircuitBreaker

            self._store_breaker = CircuitBreaker(
                getattr(self._config, "circuit_breaker_config", None),
                component_id="strategy_store_write",
            )
            self._order_breaker = CircuitBreaker(
                getattr(self._config, "circuit_breaker_config", None),
                component_id="order_submission",
            )
        except Exception as exc:
            self._store_breaker = None
            self._order_breaker = None
            self.log.debug(
                "ml_strategy.breaker_init_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )

        # Initialize message bus publisher (async bridge preferred)
        try:
            from ml.common.bus_bridge import DomainEventBridge
            from ml.common.bus_bridge import parse_topic_throttles_from_env
            from ml.common.message_bus import publisher_from_config as _pfc
            from ml.common.throttler import Throttler
            from ml.config.actor_bus import ActorBusConfig as _ABCfg
            from ml.config.bus import MessageBusConfig as _MBCfg

            bridge_cfg = _ABCfg.from_env()
            bus_cfg = _MBCfg.from_env()
            if bridge_cfg.from_strategy and bus_cfg.enabled:
                publisher = _pfc(bus_cfg)
                throttler = (
                    Throttler(
                        rate_per_sec=float(bridge_cfg.throttle_rate_per_sec),
                        burst=int(bridge_cfg.throttle_burst),
                    )
                    if bridge_cfg.throttle_enabled
                    else None
                )
                per_topic_throttles = parse_topic_throttles_from_env()
                self._bus_bridge = DomainEventBridge(
                    publisher,
                    max_queue=int(bridge_cfg.max_queue),
                    throttler=throttler,
                    per_topic_throttles=per_topic_throttles,
                    component_id="ml_strategy",
                )
                self._bus_bridge.start()
                self._bus_publisher = self._bus_bridge

                # Mutual exclusion: disable store-level publishing when strategy bridge active
                try:
                    if self.strategy_store is not None:
                        if hasattr(self.strategy_store, "publisher"):
                            setattr(self.strategy_store, "publisher", None)
                        if hasattr(self.strategy_store, "_enable_publishing"):
                            setattr(self.strategy_store, "_enable_publishing", False)
                except Exception as exc:
                    self.log.debug(
                        "ml_strategy.bus_store_mutual_exclusion_failed",
                        strategy_id=str(self.id),
                        exc_info=True,
                        error=str(exc),
                    )
            elif bridge_cfg.from_store:
                from ml.common.message_bus import NoopPublisher

                # Store-level publishing is responsible for bus output.
                self._bus_publisher = NoopPublisher()
            elif not bus_cfg.enabled:
                # Disabled bus (noop publisher)
                self._bus_publisher = _pfc(bus_cfg)
            else:
                from ml.common.message_bus import NoopPublisher

                # Bus enabled but no async bridge configured; avoid sync hot-path I/O.
                self._bus_publisher = NoopPublisher()
                _SafeLogger(self.log).debug(
                    "ml_strategy.bus_publish_disabled_no_async_bridge",
                    strategy_id=str(self.id),
                )
        except Exception as exc:
            self.log.debug(
                "ml_strategy.bus_publisher_init_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )

    def _init_positions_provider(self) -> None:
        """
        Initialize positions provider wiring for sizing and risk checks.
        """
        safe_log = _SafeLogger(self.log)
        try:
            from ml.strategies.common.positions_provider import NautilusPositionsProvider

            positions_cfg = getattr(self._config, "positions_config", None)
            self._positions_provider = NautilusPositionsProvider(
                cache=self.cache if hasattr(self, "cache") else None,
                portfolio=self.portfolio if hasattr(self, "portfolio") else None,
                config=positions_cfg,
                log=self.log,
                strategy_id=str(self.id),
            )
            if self.risk_manager is not None and hasattr(self.risk_manager, "set_positions_provider"):
                self.risk_manager.set_positions_provider(self._positions_provider)
            if self.risk_manager is not None and hasattr(self.risk_manager, "set_market_price_provider"):
                self.risk_manager.set_market_price_provider(
                    self.cache if hasattr(self, "cache") else None,
                )
        except Exception as exc:
            self._positions_provider = None
            safe_log.debug(
                "ml_strategy.positions_provider_init_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )

    def _check_positions_ready(self) -> None:
        """
        Evaluate positions readiness for live trading.
        """
        from ml.config.base import PositionsConfig
        from ml.strategies.common.positions import PositionsHealthStatus

        positions_cfg = getattr(self._config, "positions_config", None) or PositionsConfig()
        safe_log = _SafeLogger(self.log)
        is_backtesting = False
        try:
            if hasattr(self, "cache"):
                is_backtesting = bool(getattr(self.cache, "is_backtesting", False))
        except Exception as exc:
            safe_log.debug(
                "ml_strategy.positions_live_check_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )

        is_live = bool(getattr(self._config, "execute_trades", False)) and not is_backtesting
        require_positions = positions_cfg.positions_required_for_live and is_live
        require_full_list = require_positions or not positions_cfg.allow_degraded
        suppress_backtest_logs = is_backtesting and not positions_cfg.log_degraded_in_backtest

        if self._positions_provider is None:
            self._positions_health = PositionsHealthStatus(
                ready=not require_positions,
                degraded=True,
                source=positions_cfg.source_priority[0],
                reason="positions_unavailable",
                positions_count=0,
            )
            if require_positions:
                safe_log.error(
                    "ml_strategy.positions_provider_missing",
                    strategy_id=str(self.id),
                    reason=self._positions_health.reason,
                    source=self._positions_health.source.value,
                )
            elif suppress_backtest_logs:
                safe_log.debug(
                    "ml_strategy.positions_provider_missing",
                    strategy_id=str(self.id),
                    reason=self._positions_health.reason,
                    source=self._positions_health.source.value,
                )
            else:
                safe_log.warning(
                    "ml_strategy.positions_provider_missing",
                    strategy_id=str(self.id),
                    reason=self._positions_health.reason,
                    source=self._positions_health.source.value,
                )
            return

        try:
            self._positions_health = self._positions_provider.check_positions_ready(
                instrument_id=getattr(self._config, "instrument_id", None),
                require_full_list=require_full_list,
                require_positions=require_positions,
            )
        except Exception as exc:
            self._positions_health = PositionsHealthStatus(
                ready=not require_positions,
                degraded=True,
                source=positions_cfg.source_priority[0],
                reason="positions_check_failed",
                positions_count=0,
            )
            safe_log.debug(
                "ml_strategy.positions_ready_check_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )
            return

        if self._positions_health.degraded:
            if require_positions and not self._positions_health.ready:
                safe_log.error(
                    "ml_strategy.positions_ready_degraded",
                    strategy_id=str(self.id),
                    reason=self._positions_health.reason,
                    source=self._positions_health.source.value,
                    positions_count=self._positions_health.positions_count,
                )
            elif suppress_backtest_logs:
                safe_log.debug(
                    "ml_strategy.positions_ready_degraded",
                    strategy_id=str(self.id),
                    reason=self._positions_health.reason,
                    source=self._positions_health.source.value,
                    positions_count=self._positions_health.positions_count,
                )
            else:
                safe_log.warning(
                    "ml_strategy.positions_ready_degraded",
                    strategy_id=str(self.id),
                    reason=self._positions_health.reason,
                    source=self._positions_health.source.value,
                    positions_count=self._positions_health.positions_count,
                )

    def _init_components(self) -> None:
        """
        Initialize the 6 decomposed components.
        """
        strategy_id = str(self.id)

        # 1. Signal Routing Component
        self._signal_router = SignalRoutingComponent(
            target_model_ids=self.target_model_ids,
            aggregation_mode=self.aggregation_mode,
            required_models=self.required_models,
            time_window_ms=self.time_window_ms,
            conflict_resolution=self.conflict_resolution,
            model_weights=self.model_weights,
            min_confidence=self._config.min_confidence,
            history_size=getattr(self._config, "history_size", 100),
            instrument_id=self._config.instrument_id,
            log=self.log,
        )

        # 2. Decision Persistence Component
        self._decision_persister = DecisionPersistenceComponent(
            strategy_id=strategy_id,
            strategy_store=self.strategy_store,
            circuit_breaker=self._store_breaker,
            bus_publisher=self._bus_publisher,
            persist_all_signals=self._config.persist_all_signals,
            log=self.log,
            active_positions=self._active_positions,
            pending_orders=self._pending_orders,
            stop_loss_pct=self._config.stop_loss_pct,
            take_profit_pct=self._config.take_profit_pct,
            max_positions=self._config.max_positions,
            is_backtesting=(
                getattr(self.cache, "is_backtesting", False) if hasattr(self, "cache") else False
            ),
            model_signals=self._model_signals,
            run_id=getattr(self._config, "run_id", None),
        )

        # 3. Position Management Component
        portfolio_cfg = getattr(self._config, "portfolio_config", None)
        portfolio_batching_config = None
        if portfolio_cfg is not None:
            portfolio_batching_config = getattr(portfolio_cfg, "batching_config", None)
        if portfolio_batching_config is None:
            try:
                from ml.strategies.portfolio import PortfolioBatchingConfig

                portfolio_batching_config = PortfolioBatchingConfig()
            except Exception as exc:
                self.log.debug(
                    "ml_strategy.portfolio_batching_config_unavailable",
                    strategy_id=strategy_id,
                    exc_info=True,
                    error=str(exc),
                )
                portfolio_batching_config = None
        self._position_manager = PositionManagementComponent(
            position_size_pct=self._config.position_size_pct,
            position_sizer=self.position_sizer,
            risk_manager=self.risk_manager,
            portfolio_manager=self.portfolio_manager,
            cache=self.cache if hasattr(self, "cache") else None,
            portfolio=self.portfolio if hasattr(self, "portfolio") else None,
            instrument_id=self._config.instrument_id,
            portfolio_batching_config=portfolio_batching_config,
            positions_provider=self._positions_provider,
            log=self.log,
            strategy_id=strategy_id,
            allow_min_quantity_fallback=getattr(self._config, "serialize_order_intents", False),
        )

        # 4. Order Submission Component
        self._order_submitter = OrderSubmissionComponent(
            strategy_id=strategy_id,
            order_executor=self.order_executor,
            circuit_breaker=self._order_breaker,
            risk_halt_provider=self.risk_manager,
            performance_tracker=self.performance,
            cache=self.cache if hasattr(self, "cache") else None,
            submit_order_callback=self._resolve_submit_order_callback(),
            cancel_order_callback=self._resolve_cancel_order_callback(),
            log=self.log,
            instrument_id=self._config.instrument_id,
            trader_id=self.trader_id if hasattr(self, "trader_id") else None,
            clock=self.clock if hasattr(self, "clock") else None,
            max_quote_age_ms=self._config.max_quote_age_ms,
            orders_submitted_metric=self.orders_submitted_metric,
        )

        # 5. Lifecycle Component
        returns_bar_spec: str | None = None
        returns_cfg = getattr(self._config, "returns_config", None)
        if returns_cfg is not None:
            update_mode = getattr(returns_cfg, "update_mode", None)
            if update_mode in (ReturnsUpdateMode.BAR, ReturnsUpdateMode.BOTH):
                returns_bar_spec = getattr(returns_cfg, "bar_spec", None)
        self._lifecycle = LifecycleComponent(
            strategy_id=strategy_id,
            instrument_id=self._config.instrument_id,
            signal_client_id=getattr(self._config, "signal_client_id", None),
            signal_source=getattr(self._config, "ml_signal_source", None),
            target_model_ids=self.target_model_ids,
            aggregation_mode=self.aggregation_mode,
            position_size_pct=self._config.position_size_pct,
            min_confidence=self._config.min_confidence,
            execute_trades=self._config.execute_trades,
            subscribe_quote_ticks=self._config.subscribe_quote_ticks,
            quote_schema=self._config.quote_schema,
            returns_bar_spec=returns_bar_spec,
            subscribe_data_callback=(
                self.subscribe_data if hasattr(self, "subscribe_data") else None
            ),
            subscribe_instrument_callback=(
                self.subscribe_instrument if hasattr(self, "subscribe_instrument") else None
            ),
            subscribe_quote_ticks_callback=(
                self.subscribe_quote_ticks if hasattr(self, "subscribe_quote_ticks") else None
            ),
            subscribe_bars_callback=(
                self.subscribe_bars if hasattr(self, "subscribe_bars") else None
            ),
            log=self.log,
        )

        # 6. Performance Tracking Component
        self._performance_tracker = PerformanceTrackingComponent(
            strategy_id=strategy_id,
            track_performance=self.track_performance,
            log=self.log,
        )

    # -------------------------------------------------------------------------
    # Store Property Accessors (backward compatibility)
    # -------------------------------------------------------------------------

    @property
    def feature_store(self) -> object | None:
        """
        Access the feature store from the injected stores container.
        """
        if self._stores is not None and hasattr(self._stores, "feature_store"):
            return cast(object, self._stores.feature_store)
        return None

    @property
    def model_store(self) -> object | None:
        """
        Access the model store from the injected stores container.
        """
        if self._stores is not None and hasattr(self._stores, "model_store"):
            return cast(object, self._stores.model_store)
        return None

    @property
    def data_store(self) -> object | None:
        """
        Access the data store from the injected stores container.
        """
        if self._stores is not None and hasattr(self._stores, "data_store"):
            return cast(object, self._stores.data_store)
        return None

    @property
    def feature_registry(self) -> object | None:
        """
        Access the feature registry from the injected stores container.
        """
        if self._stores is not None and hasattr(self._stores, "feature_registry"):
            return cast(object, self._stores.feature_registry)
        return None

    @property
    def model_registry(self) -> object | None:
        """
        Access the model registry from the injected stores container.
        """
        if self._stores is not None and hasattr(self._stores, "model_registry"):
            return cast(object, self._stores.model_registry)
        return None

    @property
    def strategy_registry(self) -> object | None:
        """
        Access the strategy registry from the injected stores container.
        """
        if self._stores is not None and hasattr(self._stores, "strategy_registry"):
            return cast(object, self._stores.strategy_registry)
        return None

    @property
    def data_registry(self) -> object | None:
        """
        Access the data registry from the injected stores container.
        """
        if self._stores is not None and hasattr(self._stores, "data_registry"):
            return cast(object, self._stores.data_registry)
        return None

    # -------------------------------------------------------------------------
    # Decision Helper Methods (backward compatibility)
    # -------------------------------------------------------------------------

    def decision_from_prediction(self, prediction: float) -> str:
        """
        Map a prediction probability to BUY/SELL/HOLD using the neutral band.

        Parameters
        ----------
        prediction : float
            Model prediction probability in [0, 1].

        Returns
        -------
        str
            One of "BUY", "SELL", or "HOLD".

        """
        neutral_band = float(getattr(self._config, "prediction_neutral_band", 0.0))
        return decision_from_probability(
            prediction,
            neutral_band=neutral_band,
        )

    def target_side_from_prediction(self, prediction: float, threshold: float = 0.5) -> OrderSide:
        """
        Map a prediction score to an order side using a threshold.

        Parameters
        ----------
        prediction : float
            Model prediction probability in [0, 1].
        threshold : float
            Decision threshold for BUY/SELL split.

        Returns
        -------
        OrderSide
            BUY if prediction > threshold, else SELL.

        """
        from nautilus_trader.model.enums import OrderSide

        return OrderSide.BUY if float(prediction) > float(threshold) else OrderSide.SELL

    def should_reverse(self, current_position: Position, target_side: OrderSide) -> bool:
        """
        Check if an existing position should be reversed given a target side.

        Returns
        -------
        bool
            True if reversing is required.

        """
        from nautilus_trader.model.enums import OrderSide

        return bool(
            (current_position.side.name == "LONG" and target_side == OrderSide.SELL)
            or (current_position.side.name == "SHORT" and target_side == OrderSide.BUY),
        )

    # -------------------------------------------------------------------------
    # Lifecycle Methods (delegated to LifecycleComponent)
    # -------------------------------------------------------------------------

    def on_start(self) -> None:
        """
        Initialize the strategy and subscribe to ML signals.

        Delegates to LifecycleComponent.on_start().

        """
        self._sync_component_strategy_ids()
        self._check_positions_ready()
        if self._lifecycle is not None:
            self._lifecycle.on_start()

    def _sync_component_strategy_ids(self) -> None:
        """
        Ensure components track the current strategy ID.
        """
        strategy_id = str(self.id)
        components = (
            self._decision_persister,
            self._position_manager,
            self._order_submitter,
            self._positions_provider,
            self.performance,
            self._lifecycle,
            self._returns_updater,
        )
        for component in components:
            if component is None or not hasattr(component, "_strategy_id"):
                continue
            try:
                setattr(component, "_strategy_id", strategy_id)
            except Exception as exc:
                self.log.debug(
                    "ml_strategy.strategy_id_sync_failed",
                    strategy_id=str(self.id),
                    exc_info=True,
                    error=str(exc),
                )

    def on_stop(self) -> None:
        """
        Log final statistics when the strategy stops.

        Delegates to LifecycleComponent.on_stop().

        """
        if self._lifecycle is not None:
            self._lifecycle.on_stop(
                strategy_store=self.strategy_store,
                signals_received=self._signals_received,
                trades_executed=self._trades_executed,
                winning_trades=self._winning_trades,
                total_pnl=self._total_pnl,
                dry_run_trades=self._dry_run_trades,
            )
        bridge = getattr(self, "_bus_bridge", None)
        if bridge is not None:
            try:
                bridge.stop(drain=True, timeout=1.0)
            except Exception as exc:
                self.log.debug(
                    "ml_strategy.bus_bridge_stop_failed",
                    strategy_id=str(self.id),
                    exc_info=True,
                    error=str(exc),
                )

    # -------------------------------------------------------------------------
    # Position Events
    # -------------------------------------------------------------------------

    def on_position_closed(self, event: PositionClosed) -> None:
        """
        Handle position closed events for realized PnL tracking.
        """
        super().on_position_closed(event)
        self._handle_position_closed_event(event)

    # -------------------------------------------------------------------------
    # Data Handling (delegated to SignalRoutingComponent)
    # -------------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar data for returns updates.
        """
        super().on_bar(bar)
        updater = getattr(self, "_returns_updater", None)
        if updater is None:
            return
        should_update = getattr(updater, "should_update_from_bar", None)
        if callable(should_update) and not should_update():
            return
        cache = self.cache if hasattr(self, "cache") else None
        try:
            updater.update_from_bar(
                bar,
                cache=cache,
                reference_ts=getattr(bar, "ts_event", None),
            )
        except Exception as exc:
            self.log.debug(
                "ml_strategy.returns_bar_update_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )

    def on_data(self, data: Any) -> None:
        """
        Process incoming data, particularly ML signals.

        Delegates signal routing to SignalRoutingComponent.

        Parameters
        ----------
        data : Data
            The incoming data object.

        """
        from ml.actors.base import MLSignal

        if not isinstance(data, MLSignal):
            return

        # Add to local history for backward compatibility
        self._signal_history.append(data)
        self._maybe_subscribe_returns_bars(data)

        if self._order_submitter is not None:
            self._order_submitter.update_config(
                cache=self.cache if hasattr(self, "cache") else None,
                submit_order_callback=self._resolve_submit_order_callback(),
                cancel_order_callback=self._resolve_cancel_order_callback(),
                order_executor=self.order_executor if hasattr(self, "order_executor") else None,
                trader_id=self.trader_id if hasattr(self, "trader_id") else None,
                clock=self.clock if hasattr(self, "clock") else None,
            )
            self._order_submitter.process_pending_limit_orders()

        if self._signal_router is None:
            return

        # Route signal through component
        routed_signal = self._signal_router.route_signal(data)

        if routed_signal is None:
            return

        # Record analytics (cold-path safe)
        try:
            if self.performance is not None:
                self.performance.record_signal(routed_signal)
        except Exception as exc:
            self.log.debug(
                "ml_strategy.performance_record_failed",
                strategy_id=str(self.id),
                signal_model=str(getattr(routed_signal, "model_id", "")),
                exc_info=True,
                error=str(exc),
            )

        # Handle the signal
        self._handle_ml_signal(routed_signal)

    def _maybe_subscribe_returns_bars(self, signal: MLSignal) -> None:
        """
        Subscribe to return bars lazily when bar spec arrives with signals.
        """
        if getattr(self, "_returns_bar_subscribed", False):
            return
        if getattr(self, "_returns_bar_subscription_attempted", False):
            return
        updater = getattr(self, "_returns_updater", None)
        if updater is None:
            return
        should_update = getattr(updater, "should_update_from_bar", None)
        if callable(should_update) and not should_update():
            return

        bar_spec: str | None = None
        returns_cfg = getattr(self._config, "returns_config", None)
        if returns_cfg is not None:
            bar_spec = getattr(returns_cfg, "bar_spec", None)
        if not bar_spec:
            meta = getattr(signal, "metadata", None) or {}
            bar_spec = meta.get("bar_spec")
        if not bar_spec:
            return

        subscribe_bars = getattr(self, "subscribe_bars", None)
        if not callable(subscribe_bars):
            self._returns_bar_subscription_attempted = True
            self.log.warning(
                "ml_strategy.subscribe_bars_callback_not_configured "
                f"strategy_id={self.id}",
            )
            return

        try:
            from nautilus_trader.model.data import BarSpecification
            from nautilus_trader.model.data import BarType

            bar_type = BarType(
                self._config.instrument_id,
                BarSpecification.from_str(str(bar_spec)),
            )
            subscribe_bars(bar_type)
            self._returns_bar_subscribed = True
        except Exception as exc:
            self.log.debug(
                "ml_strategy.subscribe_return_bars_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )
        finally:
            self._returns_bar_subscription_attempted = True

    def on_order_event(self, event: Any) -> None:
        """
        Handle order events and persist them for audit.
        """
        super().on_order_event(event)
        self._persist_order_event(event)

    def on_order_filled(self, event: Any) -> None:
        """
        Handle order filled events for position tracking.
        """
        super().on_order_filled(event)
        self._persist_order_event(event)

        # Update pending orders count
        self._pending_orders = max(0, self._pending_orders - 1)

        # Update position count
        current_position = self._get_current_position()
        if current_position is None:
            self._active_positions = 0
        else:
            self._active_positions = 1

        # Record position count metric
        if self.position_count_metric:
            self.position_count_metric.labels(
                strategy_id=str(self.id),
                instrument=str(self._config.instrument_id),
            ).set(self._active_positions)

        self.log.info(
            f"Order filled: {event.order_side.name} {event.last_qty} @ {event.last_px}, "
            f"Active positions: {self._active_positions}",
        )

    def _persist_order_event(self, event: Any) -> None:
        """
        Persist order events to the strategy store when available.
        """
        safe_log = _SafeLogger(getattr(self, "log", None))
        store = getattr(self, "strategy_store", None)
        if store is None:
            return
        write_order_event = getattr(store, "write_order_event", None)
        if not callable(write_order_event):
            return

        is_live = True
        try:
            if hasattr(self, "cache") and getattr(self.cache, "is_backtesting", False):
                is_live = False
        except Exception as exc:
            safe_log.debug(
                "ml_strategy.order_event_live_check_failed",
                strategy_id=str(getattr(self, "id", "unknown")),
                exc_info=True,
                error=str(exc),
            )

        try:
            write_order_event(
                event,
                is_live=is_live,
                run_id=getattr(self._config, "run_id", None),
            )
        except Exception as exc:
            safe_log.debug(
                "ml_strategy.order_event_persist_failed",
                strategy_id=str(getattr(self, "id", "unknown")),
                exc_info=True,
                error=str(exc),
            )

    def _persist_risk_halt_event(
        self,
        *,
        event_type: str,
        reason: str,
        detail: str | None,
        ts_event: int,
    ) -> None:
        """
        Persist risk-halt transitions to the strategy store when available.
        """
        safe_log = _SafeLogger(getattr(self, "log", None))
        store = getattr(self, "strategy_store", None)
        if store is None:
            return
        writer = getattr(store, "write_risk_halt_event", None)
        if not callable(writer):
            return

        is_live = True
        try:
            if hasattr(self, "cache") and getattr(self.cache, "is_backtesting", False):
                is_live = False
        except Exception as exc:
            safe_log.debug(
                "ml_strategy.risk_halt_live_check_failed",
                strategy_id=str(getattr(self, "id", "unknown")),
                exc_info=True,
                error=str(exc),
            )

        try:
            writer(
                strategy_id=str(getattr(self, "id", "unknown")),
                instrument_id=str(getattr(self._config, "instrument_id", "")),
                event_type=str(event_type),
                reason=str(reason),
                detail=detail,
                ts_event=int(ts_event),
                is_live=is_live,
                run_id=getattr(self._config, "run_id", None),
            )
        except Exception as exc:
            safe_log.debug(
                "ml_strategy.risk_halt_persist_failed",
                strategy_id=str(getattr(self, "id", "unknown")),
                exc_info=True,
                error=str(exc),
            )

    def _record_risk_halt_transition(self, *, ts_event: int | None) -> None:
        """
        Record risk-halt state transitions for audit.
        """
        if self.risk_manager is None:
            return
        state_getter = getattr(self.risk_manager, "is_trading_halted", None)
        reason_getter = getattr(self.risk_manager, "get_halt_reason", None)
        if not callable(state_getter):
            return
        try:
            halted = bool(state_getter())
        except Exception as exc:
            self.log.debug(
                "ml_strategy.risk_halt_state_unavailable",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )
            return
        reason = None
        if callable(reason_getter):
            try:
                reason = reason_getter()
            except Exception as exc:
                self.log.debug(
                    "ml_strategy.risk_halt_reason_unavailable",
                    strategy_id=str(self.id),
                    exc_info=True,
                    error=str(exc),
                )
        resolved_reason = str(reason or "unknown")
        if ts_event is not None:
            resolved_ts = int(ts_event)
        elif hasattr(self, "clock"):
            try:
                resolved_ts = int(self.clock.timestamp_ns())
            except Exception:
                resolved_ts = time.time_ns()
        else:
            resolved_ts = time.time_ns()

        if self._last_risk_halt_state is None:
            self._last_risk_halt_state = halted
            if halted:
                self._last_risk_halt_reason = resolved_reason
                detail = None
                detail_getter = getattr(self.risk_manager, "get_halt_detail", None)
                if callable(detail_getter):
                    try:
                        detail = detail_getter()
                    except Exception:
                        detail = None
                self._persist_risk_halt_event(
                    event_type="halted",
                    reason=resolved_reason,
                    detail=detail,
                    ts_event=resolved_ts,
                )
            return

        if halted == self._last_risk_halt_state:
            return

        if halted:
            self._last_risk_halt_state = True
            self._last_risk_halt_reason = resolved_reason
            detail = None
            detail_getter = getattr(self.risk_manager, "get_halt_detail", None)
            if callable(detail_getter):
                try:
                    detail = detail_getter()
                except Exception:
                    detail = None
            self._persist_risk_halt_event(
                event_type="halted",
                reason=resolved_reason,
                detail=detail,
                ts_event=resolved_ts,
            )
            return

        resume_reason = self._last_risk_halt_reason or resolved_reason
        self._last_risk_halt_state = False
        self._last_risk_halt_reason = None
        self._persist_risk_halt_event(
            event_type="resumed",
            reason=resume_reason,
            detail=None,
            ts_event=resolved_ts,
        )

    def _handle_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and potentially execute trades.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process.

        """
        self._signals_received += 1
        self._last_signal_time = signal.ts_event

        # Record metrics for signal received
        if self.signals_received_metric:
            model_id = getattr(signal, "model_id", None) or signal.metadata.get(
                "model_id",
                "unknown",
            )
            self.signals_received_metric.labels(
                strategy_id=str(self.id),
                signal_source=model_id,
            ).inc()

        # Check if signal is for our instrument
        if signal.instrument_id != self._config.instrument_id:
            return

        self._record_risk_halt_transition(ts_event=signal.ts_event)
        if self._maybe_apply_risk_action(signal):
            return

        # Check position limits
        if self._active_positions >= self._config.max_positions:
            current_position = self._get_current_position()
            if current_position is None:
                self.log.debug("Maximum positions reached, ignoring signal")
                return

        # Let concrete strategy decide on the signal
        self._process_signal(signal)
        self._process_ml_signal(signal)

    def _maybe_apply_risk_action(self, signal: MLSignal) -> bool:
        """
        Apply staged risk actions before signal processing when needed.

        Returns
        -------
        bool
            True when a staged risk action was applied and signal processing should stop.

        """
        if self.risk_manager is None:
            return False
        action_getter = getattr(self.risk_manager, "get_risk_action", None)
        if not callable(action_getter):
            return False
        try:
            decision = action_getter(
                portfolio=self.portfolio if hasattr(self, "portfolio") else None,
                ts_event=signal.ts_event,
            )
        except Exception as exc:
            self.log.debug(
                "ml_strategy.risk_action_check_failed",
                strategy_id=str(self.id),
                exc_info=True,
                error=str(exc),
            )
            return False

        try:
            from ml.strategies.risk import RiskAction
            from ml.strategies.risk import RiskActionDecision
        except Exception:
            return False

        if (
            isinstance(decision, RiskActionDecision)
            and decision.action is RiskAction.LIQUIDATE
        ):
            self._liquidate_positions(decision, signal)
            return True
        return False

    def _liquidate_positions(self, decision: RiskActionDecision, signal: MLSignal) -> None:
        """
        Submit reduce-only liquidation orders for open positions.
        """
        from nautilus_trader.model.enums import OrderSide

        positions, source = self._resolve_liquidation_positions()
        if not positions:
            reason = decision.reason or "unknown"
            now_ns = int(getattr(signal, "ts_event", 0) or 0) or time.time_ns()
            cooldown_ns: int | None = None
            liquidation_cfg = None
            if self.risk_manager is not None:
                liquidation_cfg = getattr(
                    getattr(self.risk_manager, "config", None),
                    "liquidation_config",
                    None,
                )
            if liquidation_cfg is not None:
                cooldown_ms = getattr(liquidation_cfg, "cooldown_ms", None)
                if cooldown_ms is not None:
                    cooldown_ns = int(cooldown_ms) * 1_000_000

            last_ts = self._last_liquidation_no_positions_ts
            if last_ts is None:
                should_log = True
            elif cooldown_ns is None:
                should_log = False
            else:
                should_log = (now_ns - last_ts) >= cooldown_ns
            if should_log:
                self._last_liquidation_no_positions_ts = now_ns
                self.log.warning(
                    "ml_strategy.liquidation_no_positions "
                    f"strategy_id={self.id} reason={reason}",
                )
            else:
                self.log.debug(
                    "ml_strategy.liquidation_no_positions_suppressed",
                    strategy_id=str(self.id),
                    reason=reason,
                )
            return

        source_label = source or "unknown"
        risk_metrics = None
        if self.risk_manager is not None and hasattr(
            self.risk_manager,
            "get_risk_metrics",
        ):
            try:
                risk_metrics = self.risk_manager.get_risk_metrics()
            except Exception as exc:
                self.log.debug(
                    "ml_strategy.risk_metrics_unavailable",
                    strategy_id=str(self.id),
                    exc_info=True,
                    error=str(exc),
                )

        reason = decision.reason or "risk_liquidation"
        for position in positions:
            try:
                if not position.is_open:
                    continue
            except Exception:
                continue
            exit_side = self._resolve_liquidation_side(position)
            quantity = self._resolve_liquidation_quantity(position)
            if exit_side is None or quantity is None:
                continue
            exit_metadata = {
                "reason": reason,
                "detail": decision.detail,
                "action": "liquidate",
                "positions_source": source_label,
            }
            execution_params = {
                "action": "liquidate",
                "exit": exit_metadata,
                "risk_action": reason,
                "risk_action_detail": decision.detail,
                "positions_source": source_label,
                "current_side": (
                    "LONG"
                    if exit_side is OrderSide.SELL
                    else "SHORT"
                ),
            }
            self._set_exit_intent_metadata(exit_metadata)
            self._persist_strategy_decision(
                signal=signal,
                decision_type=exit_side.name,
                position_size=quantity,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
            )
            if self._config.execute_trades:
                self._place_market_order(exit_side, quantity, reduce_only=True)
            else:
                self._dry_run_trades += 1
                self.log.info(
                    "[DRY RUN] Would liquidate "
                    f"{position.instrument_id} due to {reason} "
                    "(execute_trades=False)",
                )

    def _resolve_liquidation_positions(self) -> tuple[list[PositionViewProtocol], str | None]:
        """
        Resolve positions for liquidation using provider fallbacks.
        """
        from ml.config.base import PositionsSource

        positions: list[PositionViewProtocol] = []
        source: str | None = None
        require_full_list = False

        if self.risk_manager is not None:
            liquidation_cfg = getattr(
                getattr(self.risk_manager, "config", None),
                "liquidation_config",
                None,
            )
            if liquidation_cfg is not None:
                require_full_list = bool(
                    getattr(liquidation_cfg, "require_full_positions", False),
                )

        if self._positions_provider is not None:
            try:
                snapshot = self._positions_provider.get_positions_snapshot(
                    instrument_id=None if require_full_list else self._config.instrument_id,
                    require_full_list=require_full_list,
                )
                positions = list(snapshot.positions)
                source = snapshot.source.value
            except Exception as exc:
                self.log.debug(
                    "ml_strategy.liquidation_positions_failed",
                    strategy_id=str(self.id),
                    exc_info=True,
                    error=str(exc),
                )

        if not positions:
            current_position = self._get_current_position()
            if current_position is not None:
                positions = [current_position]
                source = PositionsSource.CACHE_OPEN.value

        return positions, source

    def _resolve_liquidation_side(self, position: PositionViewProtocol) -> OrderSide | None:
        """
        Resolve the exit side for a liquidation order.
        """
        from nautilus_trader.model.enums import OrderSide

        try:
            signed_qty = position.signed_decimal_qty()
        except Exception:
            return None
        if signed_qty > 0:
            return OrderSide.SELL
        if signed_qty < 0:
            return OrderSide.BUY
        return None

    def _resolve_liquidation_quantity(self, position: PositionViewProtocol) -> Quantity | None:
        """
        Resolve the quantity for a liquidation order.
        """
        from nautilus_trader.model.objects import Quantity

        quantity = getattr(position, "quantity", None)
        if quantity is not None:
            return quantity
        try:
            signed_qty = position.signed_decimal_qty()
        except Exception:
            return None
        if signed_qty == 0:
            return None
        return Quantity.from_str(str(abs(signed_qty)))

    # -------------------------------------------------------------------------
    # Decision Persistence (delegated to DecisionPersistenceComponent)
    # -------------------------------------------------------------------------

    def _persist_strategy_decision(
        self,
        signal: MLSignal,
        decision_type: str,
        position_size: Quantity | None = None,
        risk_metrics: dict[str, float] | None = None,
        execution_params: dict[str, Any] | None = None,
        persist_hold: bool = False,
    ) -> None:
        """
        Persist strategy decision to StrategyStore.

        Delegates to DecisionPersistenceComponent.persist_decision().

        Parameters
        ----------
        signal : MLSignal
            The ML signal that triggered the decision.
        decision_type : str
            The decision type (BUY, SELL, or HOLD).
        position_size : Quantity, optional
            The position size for the trade.
        risk_metrics : dict[str, float], optional
            Risk metrics calculated for this decision.
        execution_params : dict[str, Any], optional
            Execution parameters for the trade.
        persist_hold : bool, optional
            Force persistence of HOLD decisions when configured.

        """
        if self._decision_persister is None:
            return

        from ml.strategies.common.positions import build_positions_metadata

        self._decision_persister.update_dependencies(
            strategy_store=self.strategy_store,
            bus_publisher=self._bus_publisher,
        )

        # Update component state
        self._decision_persister.update_state(
            active_positions=self._active_positions,
            pending_orders=self._pending_orders,
            is_backtesting=(
                getattr(self.cache, "is_backtesting", False) if hasattr(self, "cache") else False
            ),
            model_signals=self._model_signals,
            positions_metadata=build_positions_metadata(self._positions_health),
        )

        self._decision_persister.persist_decision(
            signal=signal,
            decision_type=decision_type,
            position_size=position_size,
            risk_metrics=risk_metrics,
            execution_params=execution_params,
            model_signals=self._model_signals,
            persist_hold=persist_hold,
        )

    def _get_decision_publisher(self) -> Any:
        """
        Lazily create and return the decision publisher.

        Delegates to DecisionPersistenceComponent.get_decision_publisher().

        """
        if self._decision_persister is not None:
            return self._decision_persister.get_decision_publisher()
        return None

    # -------------------------------------------------------------------------
    # Position Management (delegated to PositionManagementComponent)
    # -------------------------------------------------------------------------

    def _calculate_position_size(self) -> Quantity | None:
        """
        Calculate position size based on configuration and account balance.

        Delegates to PositionManagementComponent.calculate_position_size().

        Returns
        -------
        Quantity | None
            The calculated position size, or None if insufficient data.

        """
        if self._position_manager is None:
            return None

        # Ensure component has current cache reference
        if hasattr(self, "cache"):
            self._position_manager.update_config(
                cache=self.cache,
                portfolio=self.portfolio if hasattr(self, "portfolio") else None,
                positions_provider=self._positions_provider,
                allow_min_quantity_fallback=getattr(self._config, "serialize_order_intents", False),
            )

        return self._position_manager.calculate_position_size()

    def size_and_validate(self, signal: MLSignal) -> Quantity | None:
        """
        Determine a safe, risk-adjusted quantity for an order.

        Delegates to PositionManagementComponent.size_and_validate().

        Parameters
        ----------
        signal : MLSignal
            The triggering signal.

        Returns
        -------
        Quantity | None
            Final quantity to trade, or None if trade should not proceed.

        """
        if self._position_manager is None:
            return None

        # Ensure component has current cache reference
        if hasattr(self, "cache"):
            self._position_manager.update_config(
                cache=self.cache,
                portfolio=self.portfolio if hasattr(self, "portfolio") else None,
                positions_provider=self._positions_provider,
                allow_min_quantity_fallback=getattr(self._config, "serialize_order_intents", False),
            )

        return self._position_manager.size_and_validate(signal)

    def _get_sizing_reject_reason(self) -> str | None:
        """
        Return the last sizing rejection reason from the position manager.
        """
        if self._position_manager is None:
            return None
        getter = getattr(self._position_manager, "get_last_rejection_reason", None)
        if callable(getter):
            reason = getter()
            return reason if isinstance(reason, str) else None
        return None

    def _resolve_market_price(self, instrument_id: Any) -> float | None:
        """
        Resolve the latest tradable price for portfolio calculations.

        Delegates to PositionManagementComponent.resolve_market_price().

        """
        if self._position_manager is not None:
            return self._position_manager.resolve_market_price(instrument_id)

        # Fallback to direct cache access
        if not hasattr(self, "cache"):
            return None

        last_tick = self.cache.trade_tick(instrument_id)
        if last_tick is not None:
            return float(last_tick.price.as_double())

        quote_tick = self.cache.quote_tick(instrument_id)
        if quote_tick is not None:
            bid_price = float(quote_tick.bid_price.as_double())
            ask_price = float(quote_tick.ask_price.as_double())
            return (bid_price + ask_price) / 2.0

        return None

    def _apply_portfolio_allocation(
        self,
        *,
        signal: MLSignal,
        account: Any,
        proposed_value: float,
    ) -> float:
        """
        Apply portfolio manager allocation rules when configured.

        Delegates to PositionManagementComponent.apply_portfolio_allocation().

        """
        if self._position_manager is not None:
            return self._position_manager.apply_portfolio_allocation(
                signal=signal,
                proposed_value=proposed_value,
                account=account,
            )
        return proposed_value

    # -------------------------------------------------------------------------
    # Order Submission (delegated to OrderSubmissionComponent)
    # -------------------------------------------------------------------------

    def _place_market_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        reduce_only: bool = False,
    ) -> ClientOrderId:
        """
        Place a market order with optional stop loss and take profit.

        Delegates to OrderSubmissionComponent.place_market_order().

        Parameters
        ----------
        side : OrderSide
            The order side (BUY or SELL).
        quantity : Quantity
            The order quantity.
        reduce_only : bool, default False
            Whether this is a reduce-only order.

        Returns
        -------
        ClientOrderId
            The client order ID of the placed order.

        """
        if self._order_submitter is not None:
            # Update component config
            self._order_submitter.update_config(
                cache=self.cache if hasattr(self, "cache") else None,
                submit_order_callback=self._resolve_submit_order_callback(),
                cancel_order_callback=self._resolve_cancel_order_callback(),
                order_executor=self.order_executor if hasattr(self, "order_executor") else None,
                trader_id=self.trader_id if hasattr(self, "trader_id") else None,
                clock=self.clock if hasattr(self, "clock") else None,
            )

            order_id = self._order_submitter.place_market_order(
                instrument_id=self._config.instrument_id,
                side=side,
                quantity=quantity,
                reduce_only=reduce_only,
            )

            if order_id is not None:
                # Sync state from component
                self._dry_run_trades = self._order_submitter.dry_run_trades
                self._trades_executed += 1
                self._pending_orders += 1

            return order_id or self.cache.client_order_id()

        # Fallback: create order directly (shouldn't happen with proper init)
        return self.cache.client_order_id()

    def _submit_smart_order(
        self,
        side: OrderSide,
        quantity: Quantity,
        signal: MLSignal,
        reduce_only: bool = False,
    ) -> ClientOrderId | None:
        """
        Create and submit an order using the smart executor when available.

        Delegates to OrderSubmissionComponent.submit_smart_order().

        """
        if self._order_submitter is None:
            return None

        if getattr(self, "order_executor", None) is None:
            if getattr(self._config, "serialize_order_intents", False):
                self._pending_execution_metadata = {
                    "mode": "market",
                    "fallback_reason": "executor_unavailable",
                }
            return self._place_market_order(
                side=side,
                quantity=quantity,
                reduce_only=reduce_only,
            )

        # Update component config
        self._order_submitter.update_config(
            cache=self.cache if hasattr(self, "cache") else None,
            submit_order_callback=self._resolve_submit_order_callback(),
            cancel_order_callback=self._resolve_cancel_order_callback(),
            order_executor=self.order_executor if hasattr(self, "order_executor") else None,
            trader_id=self.trader_id if hasattr(self, "trader_id") else None,
            clock=self.clock if hasattr(self, "clock") else None,
        )

        instrument = (
            self.cache.instrument(self._config.instrument_id) if hasattr(self, "cache") else None
        )

        order_id = self._order_submitter.submit_smart_order(
            signal=signal,
            side=side,
            quantity=quantity,
            instrument=instrument,
            reduce_only=reduce_only,
        )

        # Sync state from component
        self._dry_run_trades = self._order_submitter.dry_run_trades

        return order_id

    def _place_stop_loss(
        self,
        side: OrderSide,
        quantity: Quantity,
        trigger_price: Price,
    ) -> ClientOrderId:
        """
        Place a stop loss order.

        Delegates to OrderSubmissionComponent.place_stop_loss().

        Parameters
        ----------
        side : OrderSide
            The order side (opposite of main position).
        quantity : Quantity
            The order quantity.
        trigger_price : Price
            The stop loss trigger price.

        Returns
        -------
        ClientOrderId
            The client order ID of the placed order.

        """
        if self._order_submitter is not None:
            # Update component config
            self._order_submitter.update_config(
                cache=self.cache if hasattr(self, "cache") else None,
                submit_order_callback=self._resolve_submit_order_callback(),
                order_executor=self.order_executor if hasattr(self, "order_executor") else None,
                trader_id=self.trader_id if hasattr(self, "trader_id") else None,
                clock=self.clock if hasattr(self, "clock") else None,
            )

            order_id = self._order_submitter.place_stop_loss(
                instrument_id=self._config.instrument_id,
                side=side,
                quantity=quantity,
                trigger_price=trigger_price,
            )

            if order_id is not None:
                return order_id

        return self.cache.client_order_id()

    # -------------------------------------------------------------------------
    # Performance Tracking (delegated to PerformanceTrackingComponent)
    # -------------------------------------------------------------------------

    def _update_model_performance(self, model_id: str, profit: float) -> None:
        """
        Update model performance metrics.

        Delegates to PerformanceTrackingComponent.update_model_performance().

        Parameters
        ----------
        model_id : str
            The model identifier.
        profit : float
            The profit from the trade.

        """
        if self._performance_tracker is not None:
            self._performance_tracker.update_model_performance(model_id, profit)

        # Update local state for backward compatibility
        if model_id not in self._model_performance:
            self._model_performance[model_id] = {
                "total_trades": 0,
                "total_profit": 0.0,
                "wins": 0,
                "losses": 0,
                "accuracy": 0.0,
            }

        self._model_performance[model_id]["total_trades"] += 1
        self._model_performance[model_id]["total_profit"] += profit

        if profit > 0:
            self._model_performance[model_id]["wins"] += 1
        else:
            self._model_performance[model_id]["losses"] += 1

        total = self._model_performance[model_id]["total_trades"]
        wins = self._model_performance[model_id]["wins"]
        self._model_performance[model_id]["accuracy"] = wins / total if total > 0 else 0.0

    def _record_metrics_usage(self) -> None:
        """
        Ensure metrics are recognized by validation tools.

        Delegates to PerformanceTrackingComponent if available.

        """
        if self._performance_tracker is not None:
            self._performance_tracker.record_metrics_usage(
                signals_received=self._signals_received,
                trades_executed=self._trades_executed,
                active_positions=self._active_positions,
            )

    def _handle_position_closed_event(self, event: PositionClosed) -> None:
        """
        Update risk and sizing state based on realized PnL.

        Parameters
        ----------
        event : PositionClosed
            The position closed event with realized PnL details.
        """
        try:
            realized = getattr(event, "realized_pnl", None)
            if realized is None:
                return
            pnl = float(realized.as_double())
        except Exception as pnl_exc:
            self.log.debug(
                "ml_strategy.position_pnl_calc_failed",
                exc_info=True,
                error=str(pnl_exc),
            )
            return

        ts_event: int | None = None
        try:
            ts_candidate = getattr(event, "ts_closed", None) or getattr(event, "ts_event", None)
            if ts_candidate is not None:
                ts_event = int(ts_candidate)
        except Exception as ts_exc:
            self.log.debug(
                "ml_strategy.position_ts_extract_failed",
                exc_info=True,
                error=str(ts_exc),
            )
            ts_event = None

        self._record_trade_outcome(pnl, ts_event)

    def _record_trade_outcome(self, pnl: float, ts_event: int | None = None) -> None:
        """
        Record realized PnL into risk and sizing trackers.

        Parameters
        ----------
        pnl : float
            Realized profit or loss for the closed position.
        ts_event : int | None, optional
            Event timestamp (nanoseconds) for daily reset alignment.
        """
        try:
            self._total_pnl += Decimal(str(pnl))
            if pnl > 0:
                self._winning_trades += 1
        except Exception as pnl_exc:
            self.log.debug(
                "ml_strategy.total_pnl_update_failed",
                exc_info=True,
                error=str(pnl_exc),
            )

        if self.risk_manager is not None:
            updated = False
            try:
                self.risk_manager.update_daily_pnl(pnl, ts_event=ts_event)
                updated = True
            except TypeError as type_exc:
                self.log.debug(
                    "ml_strategy.risk_daily_update_signature_mismatch",
                    exc_info=True,
                    error=str(type_exc),
                )
                try:
                    self.risk_manager.update_daily_pnl(pnl)
                    updated = True
                except Exception as risk_exc:
                    self.log.debug(
                        "ml_strategy.risk_daily_update_failed",
                        exc_info=True,
                        error=str(risk_exc),
                    )
            except Exception as risk_exc:
                self.log.debug(
                    "ml_strategy.risk_daily_update_failed",
                    exc_info=True,
                    error=str(risk_exc),
                )
            if updated:
                self._record_risk_halt_transition(ts_event=ts_event)

        if self.position_sizer is not None:
            try:
                updater = getattr(self.position_sizer, "update_performance", None)
                if callable(updater):
                    updater(pnl)
            except Exception as sizer_exc:
                self.log.debug(
                    "ml_strategy.sizer_performance_update_failed",
                    exc_info=True,
                    error=str(sizer_exc),
                )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_current_position(self) -> Position | None:
        """
        Get the current position for the configured instrument.

        Returns
        -------
        Position | None
            The current position, or None if no position exists.

        """
        if not hasattr(self, "cache") or self.cache is None:
            if self._intent_position_tracker is not None:
                return self._intent_position_tracker.get_position(self._config.instrument_id)
            return None

        positions = self.cache.positions_open(
            venue=None,
            instrument_id=self._config.instrument_id,
        )

        if positions:
            return positions[0]
        if self._intent_position_tracker is not None:
            return self._intent_position_tracker.get_position(self._config.instrument_id)
        return None

    # -------------------------------------------------------------------------
    # Signal Aggregation (delegated to SignalRoutingComponent)
    # -------------------------------------------------------------------------

    def _aggregate_signal(self, signal: MLSignal) -> None:
        """
        Aggregate signals from multiple models.

        This is handled automatically by SignalRoutingComponent.route_signal(),
        but this method is kept for backward compatibility with code that
        calls it directly.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to aggregate.

        """
        if self._signal_router is None:
            return

        # Add to buffer
        model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id")
        if model_id:
            self._model_signals[model_id] = signal
            self._signal_router.add_to_buffer(signal)

    # -------------------------------------------------------------------------
    # Stub Methods for Subclass Compatibility
    # -------------------------------------------------------------------------

    def _process_signal(self, signal: MLSignal) -> None:
        """
        Process individual signal (stub for subclass compatibility).

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process.

        """

    def _make_decision(self, decision: dict[str, Any]) -> None:
        """
        Make trading decision (stub for subclass compatibility).

        Parameters
        ----------
        decision : dict[str, Any]
            The decision data.

        """
        del decision

    def _execute_trade(self, trade: dict[str, Any]) -> None:
        """
        Execute trade based on signal (stub for subclass compatibility).

        Parameters
        ----------
        trade : dict[str, Any]
            The trade data.

        """

    def _publish_decision_event(
        self,
        signal: MLSignal,
        decision_type: str,
        risk_metrics: dict[str, float] | None,
        execution_params: dict[str, Any] | None,
        model_predictions: dict[str, float],
    ) -> None:
        """
        Publish a strategy decision event using the configured message bus.

        Delegates to DecisionPersistenceComponent.publish_decision_event().

        """
        if self._decision_persister is not None:
            self._decision_persister.publish_decision_event(
                signal=signal,
                decision_type=decision_type,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
                model_predictions=model_predictions,
                is_live=(
                    not getattr(self.cache, "is_backtesting", False)
                    if hasattr(self, "cache")
                    else True
                ),
            )

    # -------------------------------------------------------------------------
    # Abstract Method
    # -------------------------------------------------------------------------

    @abstractmethod
    def _process_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and execute trading logic.

        This method should be implemented by concrete strategies to define
        how ML signals are translated into trading actions.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process.

        """
        ...


class SimpleMLStrategyFacade(BaseMLStrategyFacade):
    """
    Simple ML strategy facade that trades based on binary ML signals.

    This is the facade equivalent of SimpleMLStrategy, demonstrating
    basic implementation that:
    - Goes long on bullish probabilities (prediction > 0.5)
    - Goes short on bearish probabilities (prediction < 0.5)
    - Holds when prediction falls within the configured neutral band
    - Implements basic position management

    """

    def _process_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and execute simple trading logic.

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process.

        """
        from nautilus_trader.model.enums import OrderSide

        current_position = self._get_current_position()

        decision = self.decision_from_prediction(signal.prediction)
        if decision == "HOLD":
            return
        target_side = OrderSide.BUY if decision == "BUY" else OrderSide.SELL

        if current_position is None:
            # No position, enter new one
            if self._should_block_entry_orders():
                self.log.info(
                    "Intent entry suppressed due to max_positions; reduce-only orders only",
                )
                return
            quantity = self._calculate_position_size()
            if quantity is None:
                self.log.warning(
                    f"Skipping trade signal due to position sizing failure for {signal.instrument_id}",
                )
                return
            self._place_market_order(target_side, quantity)
            self._active_positions += 1

            # Record position count metric
            if self.position_count_metric:
                self.position_count_metric.labels(
                    strategy_id=str(self.id),
                    instrument=str(self._config.instrument_id),
                ).set(self._active_positions)

        elif self.should_reverse(current_position, target_side):
            # Position exists but signal suggests opposite direction
            close_side = OrderSide.SELL if current_position.side.name == "LONG" else OrderSide.BUY
            self._place_market_order(
                close_side,
                current_position.quantity,
                reduce_only=True,
            )

            # Then open new position
            if self._should_block_entry_orders():
                self.log.info(
                    "Intent entry suppressed due to max_positions; reduce-only orders only",
                )
                return
            quantity = self._calculate_position_size()
            if quantity is None:
                self.log.warning(
                    f"Closed position but cannot open new one due to position sizing failure for {signal.instrument_id}",
                )
                return
            self._place_market_order(target_side, quantity)

        else:
            # Position aligns with signal, no action needed
            self.log.debug("Position aligns with signal, no action taken")

__all__ = [
    "BaseMLStrategyFacade",
    "SimpleMLStrategyFacade",
]
