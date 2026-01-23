"""
Risk management for ML trading strategies.

This module provides unified risk management including per-trade checks, portfolio-level
limits, and dynamic adjustments. Designed for capital preservation with <5ms hot path
checks.

"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram
from ml.config.base import CorrelationDataConfig
from ml.config.base import ExposurePriceConfig
from ml.config.base import ExposurePriceSource
from ml.config.base import PositionsSource
from ml.strategies.common.correlation import CorrelationProviderProtocol
from ml.strategies.common.correlation import CorrelationSnapshot
from ml.strategies.common.positions import PositionsProviderProtocol
from ml.strategies.common.positions import PositionsSnapshot


if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Quantity

    from ml.strategies.common.positions import PositionViewProtocol
    from nautilus_trader.portfolio import Portfolio


logger = logging.getLogger(__name__)

# ===== Metrics =====
risk_checks_total = get_counter(
    "ml_risk_checks_total",
    "Total risk checks performed",
    labels=["check_type", "result"],
)

risk_check_latency_seconds = get_histogram(
    "ml_risk_check_latency_seconds",
    "Risk check latency",
    buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005],
    labels=["check_type"],
)

daily_loss_gauge = get_gauge(
    "ml_daily_loss_pct",
    "Current daily loss percentage",
    labels=[],
)

exposure_gauge = get_gauge(
    "ml_total_exposure_pct",
    "Total portfolio exposure percentage",
    labels=[],
)

positions_exposure_degraded_total = get_counter(
    "ml_positions_exposure_degraded_total",
    "Positions exposure degraded",
    labels=["reason"],
)

correlation_degraded_total = get_counter(
    "ml_risk_correlation_degraded_total",
    "Correlation data degraded in risk checks",
    labels=["reason"],
)

risk_action_total = get_counter(
    "ml_risk_action_total",
    "Total staged risk actions triggered",
    labels=["action", "reason"],
)


@runtime_checkable
class MarketPriceProviderProtocol(Protocol):
    """
    Protocol for market data access used in exposure calculations.
    """

    def quote_tick(self, instrument_id: InstrumentId) -> Any:
        """
        Return the latest quote tick for an instrument.
        """
        ...

    def price(self, instrument_id: InstrumentId, price_type: Any) -> Any:
        """
        Return the latest cached price for an instrument.
        """
        ...


class RiskAction(str, Enum):
    """
    Risk action stage for strategy controls.
    """

    NORMAL = "normal"
    HALT = "halt"
    LIQUIDATE = "liquidate"


@dataclass(frozen=True, slots=True)
class RiskActionDecision:
    """
    Decision describing the current risk action.
    """

    action: RiskAction
    reason: str | None
    detail: str | None


@dataclass(frozen=True)
class RiskLiquidationConfig:
    """
    Configuration for staged liquidation behavior.

    Parameters
    ----------
    enabled : bool, default False
        Whether liquidation triggers are active.
    daily_loss_limit_pct : float | None, optional
        Liquidate when daily loss exceeds this percentage.
    drawdown_limit_pct : float | None, optional
        Liquidate when drawdown exceeds this percentage.
    unrealized_loss_limit_pct : float | None, optional
        Liquidate when unrealized loss exceeds this percentage.
    cooldown_ms : int | None, optional
        Minimum time between liquidation attempts in milliseconds.
    require_full_positions : bool, default True
        Whether liquidation requires full positions list availability.

    """

    enabled: bool = False
    daily_loss_limit_pct: float | None = None
    drawdown_limit_pct: float | None = None
    unrealized_loss_limit_pct: float | None = None
    cooldown_ms: int | None = None
    require_full_positions: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.daily_loss_limit_pct, "daily_loss_limit_pct"),
            (self.drawdown_limit_pct, "drawdown_limit_pct"),
            (self.unrealized_loss_limit_pct, "unrealized_loss_limit_pct"),
        ):
            if value is None:
                continue
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.cooldown_ms is not None and self.cooldown_ms < 0:
            raise ValueError("cooldown_ms must be non-negative")


# ===== Configuration =====
@dataclass(frozen=True)
class RiskConfig:
    """
    Configuration for risk management.
    """

    # Per-trade limits
    max_loss_per_trade_pct: float = 0.02  # 2% of account balance max loss per trade
    stop_loss_pct: float = 0.02  # 2% assumed stop distance on position value
    max_position_pct: float = 0.15  # 15% max per position

    # Portfolio limits
    daily_loss_limit_pct: float = 0.06  # 6% daily loss circuit breaker
    max_total_exposure: float = 1.0  # 100% max exposure (no leverage)
    max_correlated_positions: int = 2  # Max correlated positions

    # Drawdown controls
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    drawdown_reduction_factor: float = 0.5  # Reduce size by 50% in drawdown
    allow_reduce_only_when_halted: bool = True
    liquidation_config: RiskLiquidationConfig | None = None

    # Correlation threshold
    correlation_threshold: float = 0.7  # Consider correlated above 0.7
    correlation_data_config: CorrelationDataConfig = field(
        default_factory=CorrelationDataConfig,
    )

    # Exposure pricing
    exposure_price_config: ExposurePriceConfig = field(
        default_factory=ExposurePriceConfig,
    )

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_loss_per_trade_pct, "max_loss_per_trade_pct"),
            (self.stop_loss_pct, "stop_loss_pct"),
            (self.max_position_pct, "max_position_pct"),
            (self.daily_loss_limit_pct, "daily_loss_limit_pct"),
            (self.max_total_exposure, "max_total_exposure"),
            (self.max_drawdown_pct, "max_drawdown_pct"),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.liquidation_config is not None and not self.liquidation_config.enabled:
            if (
                self.liquidation_config.daily_loss_limit_pct is not None
                or self.liquidation_config.drawdown_limit_pct is not None
                or self.liquidation_config.unrealized_loss_limit_pct is not None
            ):
                raise ValueError("liquidation_config.enabled must be True to set thresholds")


# ===== Unified Risk Manager =====
class RiskManager:
    """
    Unified risk management with all checks in one place.

    Handles per-trade validation, portfolio limits, correlation checks, and dynamic
    adjustments based on performance.

    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        positions_provider: PositionsProviderProtocol | None = None,
        market_price_provider: MarketPriceProviderProtocol | None = None,
        correlation_provider: CorrelationProviderProtocol | None = None,
    ) -> None:
        """
        Initialize risk manager.

        Parameters
        ----------
        config : RiskConfig, optional
            Risk management configuration.
        positions_provider : PositionsProviderProtocol | None, optional
            Optional positions provider for normalized snapshots.
        correlation_provider : CorrelationProviderProtocol | None, optional
            Optional provider for correlation snapshots.

        """
        self.config = config or RiskConfig()
        self._positions_provider = positions_provider
        self._market_price_provider = market_price_provider
        self._correlation_provider = correlation_provider

        # Daily tracking (reset at midnight)
        self._daily_pnl: float = 0.0
        self._daily_reset_time: datetime = self._get_next_reset_time()
        self._trades_today: int = 0
        self._losses_today: int = 0

        # Drawdown tracking
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0

        # Position tracking
        self._position_values: dict[InstrumentId, float] = {}
        self._position_correlations: dict[
            tuple[InstrumentId, InstrumentId],
            CorrelationSnapshot,
        ] = {}

        # Circuit breaker state
        self._trading_halted: bool = False
        self._halt_reason: str | None = None
        self._halt_detail: str | None = None
        self._last_liquidation_ts_ns: int | None = None

    def set_positions_provider(
        self,
        provider: PositionsProviderProtocol | None,
    ) -> None:
        """
        Update the positions provider used for portfolio checks.

        Parameters
        ----------
        provider : PositionsProviderProtocol | None
            Positions provider instance to use.

        """
        self._positions_provider = provider

    def set_market_price_provider(
        self,
        provider: MarketPriceProviderProtocol | None,
    ) -> None:
        """
        Update the market price provider used for exposure calculations.

        Parameters
        ----------
        provider : MarketPriceProviderProtocol | None
            Market price provider instance to use.

        """
        self._market_price_provider = provider

    def set_correlation_provider(
        self,
        provider: CorrelationProviderProtocol | None,
    ) -> None:
        """
        Update the correlation provider used for correlation checks.

        Parameters
        ----------
        provider : CorrelationProviderProtocol | None
            Correlation provider instance to use.

        """
        self._correlation_provider = provider

    def is_trading_halted(self) -> bool:
        """
        Return True when trading is halted by risk controls.
        """
        return self._trading_halted

    def get_halt_reason(self) -> str | None:
        """
        Return a stable reason label for the current halt.
        """
        return self._halt_reason

    def get_halt_detail(self) -> str | None:
        """
        Return a detailed halt message when available.
        """
        return self._halt_detail

    def allow_reduce_only_when_halted(self) -> bool:
        """
        Return True when reduce-only orders may bypass a halt.
        """
        return bool(self.config.allow_reduce_only_when_halted)

    def get_risk_action(
        self,
        *,
        portfolio: Portfolio | None = None,
        ts_event: int | None = None,
    ) -> RiskActionDecision:
        """
        Return the staged risk action based on current metrics.
        """
        self._check_daily_reset(ts_event)
        liquidation_cfg = self.config.liquidation_config

        action = RiskAction.NORMAL
        reason: str | None = None
        detail: str | None = None

        if liquidation_cfg is not None and liquidation_cfg.enabled:
            now_ns = self._resolve_time_ns(ts_event)
            if self._is_liquidation_on_cooldown(now_ns, liquidation_cfg):
                return RiskActionDecision(
                    action=RiskAction.HALT,
                    reason="liquidation_cooldown",
                    detail=None,
                )

            daily_loss_pct = self._current_daily_loss_pct()
            drawdown_pct = self._current_drawdown_pct()
            unrealized_loss_pct = self._current_unrealized_loss_pct(
                portfolio,
                require_full_list=liquidation_cfg.require_full_positions,
            )

            if (
                liquidation_cfg.daily_loss_limit_pct is not None
                and daily_loss_pct >= liquidation_cfg.daily_loss_limit_pct
            ):
                action = RiskAction.LIQUIDATE
                reason = "daily_loss_liquidate"
                detail = f"Daily loss {daily_loss_pct:.1%}"
            elif (
                liquidation_cfg.drawdown_limit_pct is not None
                and drawdown_pct >= liquidation_cfg.drawdown_limit_pct
            ):
                action = RiskAction.LIQUIDATE
                reason = "drawdown_liquidate"
                detail = f"Drawdown {drawdown_pct:.1%}"
            elif (
                liquidation_cfg.unrealized_loss_limit_pct is not None
                and unrealized_loss_pct is not None
                and unrealized_loss_pct >= liquidation_cfg.unrealized_loss_limit_pct
            ):
                action = RiskAction.LIQUIDATE
                reason = "unrealized_loss_liquidate"
                detail = f"Unrealized loss {unrealized_loss_pct:.1%}"

            if action is RiskAction.LIQUIDATE:
                self._trading_halted = True
                self._halt_reason = reason
                self._halt_detail = detail
                self._last_liquidation_ts_ns = now_ns
                risk_action_total.labels(action=action.value, reason=reason or "unknown").inc()
                return RiskActionDecision(action=action, reason=reason, detail=detail)

        if self._trading_halted:
            reason = self._halt_reason or "unknown"
            detail = self._halt_detail
            risk_action_total.labels(action=RiskAction.HALT.value, reason=reason).inc()
            return RiskActionDecision(action=RiskAction.HALT, reason=reason, detail=detail)

        return RiskActionDecision(action=RiskAction.NORMAL, reason=None, detail=None)

    def check_position(
        self,
        proposed_size: Quantity | None,
        instrument: InstrumentId,
        portfolio: Portfolio,
    ) -> Quantity | None:
        """
        Check and potentially adjust proposed position size based on risk limits.

        This is the main hot-path method that performs all risk checks.

        Parameters
        ----------
        proposed_size : Quantity | None
            The proposed position size.
        instrument : InstrumentId
            The instrument to trade.
        portfolio : Portfolio
            Current portfolio state.

        Returns
        -------
        Quantity | None
            Approved position size (may be reduced), or None if rejected.

        """
        start_time = time.perf_counter()

        try:
            # Check daily reset
            self._check_daily_reset()

            # Circuit breaker check (fastest)
            if self._trading_halted:
                reason = self._halt_detail or self._halt_reason or "unknown"
                logger.warning(f"Trading halted: {reason}")
                risk_checks_total.labels(check_type="circuit_breaker", result="rejected").inc()
                return None

            if proposed_size is None:
                return None

            # Get position value
            account = portfolio.account(instrument.venue)
            if account is None:
                logger.error(f"No account for venue {instrument.venue}")
                return None

            balance = float(account.balance_total().as_double())
            position_value = float(proposed_size.as_double())  # This is value, not qty

            # Per-trade checks
            if not self._check_trade_limits(position_value, balance):
                risk_checks_total.labels(check_type="trade_limit", result="rejected").inc()
                return None

            # Portfolio exposure check
            if not self._check_portfolio_exposure(position_value, balance, portfolio):
                risk_checks_total.labels(check_type="exposure", result="rejected").inc()
                return None

            # Correlation check
            if not self._check_correlation_limits(instrument, portfolio):
                risk_checks_total.labels(check_type="correlation", result="rejected").inc()
                return None

            # Daily loss limit check
            if not self.check_daily_limits():
                risk_checks_total.labels(check_type="daily_limit", result="rejected").inc()
                return None

            # Drawdown adjustment
            adjusted_size = self._apply_drawdown_adjustment(proposed_size)

            risk_checks_total.labels(check_type="all", result="approved").inc()
            return adjusted_size

        finally:
            risk_check_latency_seconds.labels(check_type="full").observe(
                time.perf_counter() - start_time,
            )

    def _check_trade_limits(self, position_value: float, balance: float) -> bool:
        """
        Check per-trade risk limits.

        Parameters
        ----------
        position_value : float
            Proposed position value.
        balance : float
            Account balance.

        Returns
        -------
        bool
            True if trade passes limits.

        """
        if balance <= 0:
            return False

        position_pct = position_value / balance

        # Check max position size
        if position_pct > self.config.max_position_pct:
            logger.warning(
                f"Position size {position_pct:.1%} exceeds max {self.config.max_position_pct:.1%}",
            )
            return False

        # Check max loss per trade using assumed stop distance
        # Allowed loss is a fraction of account balance; potential loss is value * stop_loss_pct
        potential_loss = position_value * self.config.stop_loss_pct
        allowed_loss = balance * self.config.max_loss_per_trade_pct
        if potential_loss > allowed_loss:
            logger.warning(
                "Potential loss %.2f exceeds per-trade limit %.2f (value=%.2f, stop=%.2f%%, balance=%.2f, limit=%.2f%%)",
                potential_loss,
                allowed_loss,
                position_value,
                self.config.stop_loss_pct * 100.0,
                balance,
                self.config.max_loss_per_trade_pct * 100.0,
            )
            return False

        return True

    def _check_portfolio_exposure(
        self,
        new_position_value: float,
        balance: float,
        portfolio: Portfolio,
    ) -> bool:
        """
        Check portfolio-wide exposure limits.

        Parameters
        ----------
        new_position_value : float
            New position value to add.
        balance : float
            Account balance.
        portfolio : Portfolio
            Current portfolio.

        Returns
        -------
        bool
            True if portfolio exposure is acceptable.

        """
        # Calculate current exposure
        total_exposure = 0.0
        snapshot = self._iter_positions(
            portfolio,
            instrument=None,
            require_full_list=True,
        )
        if snapshot.source is PositionsSource.PORTFOLIO_NET:
            logger.debug("risk.positions_limited_net_position", extra={"instrument": None})
            return True
        for position in snapshot.positions:
            value = self._resolve_position_value(position)
            if value is None:
                continue
            total_exposure += value

        # Add new position
        total_exposure += new_position_value

        exposure_pct = total_exposure / balance if balance > 0 else 0

        # Update metric (handle zero-label gauge gracefully)
        try:
            exposure_gauge.labels().set(exposure_pct)
        except Exception as gauge_exc:
            try:
                exposure_gauge.set(exposure_pct)
            except Exception as fallback_exc:
                logger.debug(
                    "risk.exposure_metric_failed error=%s",
                    fallback_exc,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "risk.exposure_metric_labels_missing error=%s",
                    gauge_exc,
                    exc_info=True,
                )

        if exposure_pct > self.config.max_total_exposure:
            logger.warning(
                f"Total exposure {exposure_pct:.1%} would exceed max {self.config.max_total_exposure:.1%}",
            )
            return False

        return True

    def _check_correlation_limits(
        self,
        instrument: InstrumentId,
        portfolio: Portfolio,
    ) -> bool:
        """
        Check correlation-based position limits.

        Parameters
        ----------
        instrument : InstrumentId
            Instrument to trade.
        portfolio : Portfolio
            Current portfolio.

        Returns
        -------
        bool
            True if correlation limits not exceeded.

        """
        # Count correlated positions
        correlated_count = 0
        open_instruments: list[InstrumentId] = []
        snapshot = self._iter_positions(
            portfolio,
            instrument=instrument,
            require_full_list=True,
        )
        if snapshot.source is PositionsSource.PORTFOLIO_NET:
            logger.debug(
                "risk.positions_limited_net_position",
                extra={"instrument": str(instrument)},
            )
            return True
        for position in snapshot.positions:
            try:
                if position.is_open:
                    open_instruments.append(position.instrument_id)
            except Exception as exc:
                logger.debug(
                    "risk.position_correlation_failed",
                    exc_info=True,
                    extra={"error": str(exc), "instrument": str(instrument)},
                )

        for open_inst in open_instruments:
            correlation = self._get_correlation(instrument, open_inst)
            if correlation > self.config.correlation_threshold:
                correlated_count += 1

        if correlated_count >= self.config.max_correlated_positions:
            logger.warning(
                f"Already have {correlated_count} correlated positions, limit is {self.config.max_correlated_positions}",
            )
            return False

        return True

    def _iter_positions(
        self,
        portfolio: Portfolio,
        *,
        instrument: InstrumentId | None,
        require_full_list: bool = False,
    ) -> PositionsSnapshot:
        """
        Safely retrieve positions with API-compat fallbacks.

        Returns an empty snapshot when no compatible positions API is available.

        """
        if self._positions_provider is not None:
            try:
                return self._positions_provider.get_positions_snapshot(
                    instrument_id=instrument,
                    require_full_list=require_full_list,
                )
            except Exception as exc:
                logger.debug(
                    "risk.positions_provider_failed",
                    exc_info=True,
                    extra={
                        "error": str(exc),
                        "instrument": str(instrument) if instrument else None,
                    },
                )

        if hasattr(portfolio, "positions"):
            try:
                return PositionsSnapshot(
                    positions=list(portfolio.positions()),
                    source=PositionsSource.PORTFOLIO_POSITIONS,
                )
            except Exception as exc:
                logger.debug(
                    "risk.positions_method_failed",
                    exc_info=True,
                    extra={
                        "error": str(exc),
                        "instrument": str(instrument) if instrument else None,
                    },
                )

        if hasattr(portfolio, "positions_open"):
            try:
                return PositionsSnapshot(
                    positions=list(portfolio.positions_open()),
                    source=PositionsSource.PORTFOLIO_POSITIONS_OPEN,
                )
            except Exception as exc:
                logger.debug(
                    "risk.positions_open_method_failed",
                    exc_info=True,
                    extra={
                        "error": str(exc),
                        "instrument": str(instrument) if instrument else None,
                    },
                )

        logger.debug(
            "risk.positions_unavailable",
            extra={"instrument": str(instrument) if instrument else None},
        )
        return PositionsSnapshot(
            positions=[],
            source=PositionsSource.PORTFOLIO_POSITIONS_OPEN,
        )

    def _resolve_position_value(self, position: PositionViewProtocol) -> float | None:
        """
        Resolve a numeric position value for exposure calculations.

        Returns None when the position payload lacks numeric quantity data.
        """
        quantity = self._resolve_position_quantity(position)
        if quantity is None:
            return None
        price = self._resolve_position_price(position)
        multiplier = self._resolve_position_multiplier(position)
        if price is not None:
            return abs(quantity) * price * multiplier
        self._emit_exposure_degraded_metric("price_missing")
        return abs(quantity)

    def _resolve_position_quantity(self, position: PositionViewProtocol) -> float | None:
        """
        Resolve signed position quantity as a float.
        """
        try:
            quantity = getattr(position, "quantity", None)
            if quantity is not None and hasattr(quantity, "as_double"):
                return float(quantity.as_double())
        except Exception as exc:
            logger.debug(
                "risk.position_quantity_failed",
                exc_info=True,
                extra={"error": str(exc)},
            )

        try:
            signed_qty = self._to_float(getattr(position, "signed_qty", None))
            if signed_qty is not None:
                return signed_qty
        except Exception as exc:
            logger.debug(
                "risk.position_signed_qty_failed",
                exc_info=True,
                extra={"error": str(exc)},
            )

        try:
            return float(position.signed_decimal_qty())
        except Exception as exc:
            logger.debug(
                "risk.position_decimal_qty_failed",
                exc_info=True,
                extra={"error": str(exc)},
            )
        return None

    def _resolve_position_price(self, position: PositionViewProtocol) -> float | None:
        """
        Resolve a usable price for notional exposure calculations.
        """
        config = self.config.exposure_price_config
        for source in config.source_priority:
            if source is ExposurePriceSource.QUOTE_MID:
                price = self._resolve_quote_mid_price(position)
            elif source is ExposurePriceSource.POSITION_AVG:
                price = self._resolve_position_avg_price(position)
            elif source is ExposurePriceSource.CACHE_LAST:
                price = self._resolve_cache_last_price(position)
            else:
                price = None
            if price is not None and price > 0:
                return price
        return None

    def _resolve_position_avg_price(self, position: PositionViewProtocol) -> float | None:
        """
        Resolve an average or entry price from the position payload.
        """
        candidates = (
            "avg_px_open",
            "avg_px_close",
            "avg_price",
            "entry_price",
            "price",
            "last_px",
        )
        for attr in candidates:
            value = getattr(position, attr, None)
            price = self._to_float(value)
            if price is not None and price > 0:
                return price
        return None

    def _resolve_mark_price(self, position: PositionViewProtocol) -> float | None:
        """
        Resolve a mark price for unrealized loss checks.
        """
        price = self._resolve_quote_mid_price(position)
        if price is not None and price > 0:
            return price
        return self._resolve_cache_last_price(position)

    def _resolve_position_signed_quantity(self, position: PositionViewProtocol) -> float | None:
        """
        Resolve a signed quantity for unrealized PnL checks.
        """
        quantity = self._resolve_position_quantity(position)
        if quantity is None:
            return None
        side_name = getattr(getattr(position, "side", object()), "name", "")
        if side_name == "SHORT" and quantity > 0:
            return -quantity
        if side_name == "LONG" and quantity < 0:
            return abs(quantity)
        return quantity

    def _resolve_quote_mid_price(self, position: PositionViewProtocol) -> float | None:
        """
        Resolve the midpoint from the latest quote tick.
        """
        provider = self._market_price_provider
        if provider is None or not hasattr(provider, "quote_tick"):
            return None
        try:
            quote_tick = provider.quote_tick(position.instrument_id)
        except Exception as exc:
            logger.debug(
                "risk.quote_tick_lookup_failed",
                exc_info=True,
                extra={"error": str(exc), "instrument": str(position.instrument_id)},
            )
            return None
        if quote_tick is None:
            return None
        bid = self._to_float(getattr(quote_tick, "bid_price", None))
        ask = self._to_float(getattr(quote_tick, "ask_price", None))
        if bid is None or ask is None:
            return None
        if bid <= 0 or ask <= 0:
            return None
        return (bid + ask) / 2.0

    def _resolve_cache_last_price(self, position: PositionViewProtocol) -> float | None:
        """
        Resolve a cached last price for the position instrument.
        """
        provider = self._market_price_provider
        if provider is None or not hasattr(provider, "price"):
            return None
        try:
            from nautilus_trader.model.enums import PriceType

            price = provider.price(position.instrument_id, PriceType.LAST)
        except Exception as exc:
            logger.debug(
                "risk.cache_price_lookup_failed",
                exc_info=True,
                extra={"error": str(exc), "instrument": str(position.instrument_id)},
            )
            return None
        return self._to_float(price)

    def _resolve_position_multiplier(self, position: PositionViewProtocol) -> float:
        """
        Resolve the contract multiplier for a position.
        """
        multiplier = self._to_float(getattr(position, "multiplier", None))
        if multiplier is None or multiplier <= 0:
            return 1.0
        return multiplier

    def _resolve_account_balance(
        self,
        portfolio: Portfolio,
        positions: list[PositionViewProtocol],
    ) -> float | None:
        """
        Resolve a portfolio balance for liquidation metrics.
        """
        if positions:
            instrument_id = positions[0].instrument_id
            try:
                account = portfolio.account(instrument_id.venue)
            except Exception:
                account = None
            if account is not None:
                return float(account.balance_total().as_double())
        try:
            accounts = getattr(portfolio, "accounts", None)
            if callable(accounts):
                candidates = list(accounts())
                if candidates:
                    return float(candidates[0].balance_total().as_double())
        except Exception:
            return None
        return None

    def _emit_exposure_degraded_metric(self, reason: str) -> None:
        """
        Emit a degraded metric for exposure fallback paths.
        """
        try:
            positions_exposure_degraded_total.labels(reason=reason).inc()
        except Exception as exc:
            logger.debug(
                "risk.exposure_degraded_metric_failed",
                exc_info=True,
                extra={"error": str(exc), "reason": reason},
            )

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            if hasattr(value, "as_double"):
                return float(value.as_double())
            if hasattr(value, "as_decimal"):
                return float(value.as_decimal())
            return float(str(value))
        except Exception:
            return None

    def _get_correlation(
        self,
        inst1: InstrumentId,
        inst2: InstrumentId,
    ) -> float:
        """
        Get correlation between two instruments.

        Parameters
        ----------
        inst1 : InstrumentId
            First instrument.
        inst2 : InstrumentId
            Second instrument.

        Returns
        -------
        float
            Correlation coefficient.

        """
        snapshot = self._resolve_correlation_snapshot(inst1, inst2)
        if snapshot is None:
            return self.config.correlation_data_config.fallback_value
        return snapshot.value

    def _resolve_correlation_snapshot(
        self,
        inst1: InstrumentId,
        inst2: InstrumentId,
    ) -> CorrelationSnapshot | None:
        key = (inst1, inst2) if str(inst1) < str(inst2) else (inst2, inst1)
        cached = self._position_correlations.get(key)
        if cached is not None:
            if self._is_correlation_fresh(cached.ts_event):
                return cached
            self._emit_correlation_degraded_metric("stale_cache")
            self._position_correlations.pop(key, None)

        provider = self._correlation_provider
        if provider is None:
            self._emit_correlation_degraded_metric("provider_missing")
            return None
        try:
            snapshot = provider.get_correlation_snapshot(inst1, inst2)
        except Exception as exc:
            logger.debug(
                "risk.correlation_provider_failed",
                exc_info=True,
                extra={"error": str(exc), "inst1": str(inst1), "inst2": str(inst2)},
            )
            self._emit_correlation_degraded_metric("provider_error")
            return None
        if snapshot is None:
            self._emit_correlation_degraded_metric("correlation_missing")
            return None
        if not self._is_correlation_value_valid(snapshot.value):
            self._emit_correlation_degraded_metric("invalid_value")
            return None
        if not self._is_correlation_fresh(snapshot.ts_event):
            self._emit_correlation_degraded_metric("stale")
            return None
        self._position_correlations[key] = snapshot
        return snapshot

    def _is_correlation_fresh(self, ts_event: int | None) -> bool:
        max_age = int(self.config.correlation_data_config.max_age_seconds)
        if max_age <= 0:
            return True
        if ts_event is None:
            return False
        age_seconds = time.time() - (ts_event / 1_000_000_000)
        return age_seconds <= max_age

    @staticmethod
    def _is_correlation_value_valid(value: float) -> bool:
        if not math.isfinite(value):
            return False
        return -1.0 <= value <= 1.0

    def _emit_correlation_degraded_metric(self, reason: str) -> None:
        try:
            correlation_degraded_total.labels(reason=reason).inc()
        except Exception as exc:
            logger.debug(
                "risk.correlation_metric_failed",
                exc_info=True,
                extra={"error": str(exc), "reason": reason},
            )

    def _resolve_time_ns(self, ts_event: int | None) -> int:
        if ts_event is None:
            return time.time_ns()
        return int(ts_event)

    def _is_liquidation_on_cooldown(
        self,
        now_ns: int,
        config: RiskLiquidationConfig,
    ) -> bool:
        if config.cooldown_ms is None or self._last_liquidation_ts_ns is None:
            return False
        cooldown_ns = int(config.cooldown_ms) * 1_000_000
        if cooldown_ns <= 0:
            return False
        return now_ns - self._last_liquidation_ts_ns < cooldown_ns

    def _current_daily_loss_pct(self) -> float:
        if self._daily_pnl >= 0 or self._current_equity <= 0:
            return 0.0
        return abs(self._daily_pnl / self._current_equity)

    def _current_drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def _current_unrealized_loss_pct(
        self,
        portfolio: Portfolio | None,
        *,
        require_full_list: bool,
    ) -> float | None:
        if portfolio is None:
            return None
        snapshot = self._iter_positions(
            portfolio,
            instrument=None,
            require_full_list=require_full_list,
        )
        if snapshot.source is PositionsSource.PORTFOLIO_NET:
            return None
        balance = self._resolve_account_balance(portfolio, list(snapshot.positions))
        if balance is None or balance <= 0:
            return None
        unrealized_loss = 0.0
        for position in snapshot.positions:
            try:
                if not position.is_open:
                    continue
            except Exception:
                continue
            signed_qty = self._resolve_position_signed_quantity(position)
            if signed_qty is None or signed_qty == 0.0:
                continue
            entry_price = self._resolve_position_avg_price(position)
            mark_price = self._resolve_mark_price(position)
            if entry_price is None or mark_price is None:
                continue
            pnl = (mark_price - entry_price) * signed_qty * self._resolve_position_multiplier(position)
            if pnl < 0:
                unrealized_loss += abs(pnl)
        if unrealized_loss <= 0:
            return 0.0
        return unrealized_loss / balance

    def check_daily_limits(self, ts_event: int | None = None) -> bool:
        """
        Check if daily risk limits have been exceeded.

        Parameters
        ----------
        ts_event : int | None, optional
            Event timestamp (nanoseconds) used for daily reset alignment.

        Returns
        -------
        bool
            True if trading can continue, False if limits exceeded.

        """
        # Check daily reset
        self._check_daily_reset(ts_event)

        # Check daily loss limit
        if self._daily_pnl < 0:
            daily_loss_pct = self._current_daily_loss_pct()

            # Update metric (handle zero-label gauge gracefully)
            try:
                daily_loss_gauge.labels().set(daily_loss_pct)
            except Exception as gauge_exc:
                try:
                    daily_loss_gauge.set(daily_loss_pct)
                except Exception as fallback_exc:
                    logger.debug(
                        "risk.daily_loss_metric_failed error=%s",
                        fallback_exc,
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "risk.daily_loss_metric_labels_missing error=%s",
                        gauge_exc,
                        exc_info=True,
                    )

            if daily_loss_pct >= self.config.daily_loss_limit_pct:
                self._trading_halted = True
                detail = f"Daily loss limit reached: {daily_loss_pct:.1%}"
                self._halt_reason = "daily_loss_limit"
                self._halt_detail = detail
                logger.error(detail)
                return False

        drawdown_pct = self._current_drawdown_pct()
        if self.config.max_drawdown_pct > 0 and drawdown_pct >= self.config.max_drawdown_pct:
            self._trading_halted = True
            detail = f"Drawdown limit reached: {drawdown_pct:.1%}"
            self._halt_reason = "drawdown_limit"
            self._halt_detail = detail
            logger.error(detail)
            return False

        # Check consecutive losses
        if self._losses_today >= 5:
            logger.warning(f"High loss count today: {self._losses_today}")
            # Could implement additional logic here

        return True

    def _apply_drawdown_adjustment(self, size: Quantity) -> Quantity:
        """
        Apply drawdown-based position size adjustment.

        Parameters
        ----------
        size : Quantity
            Original position size.

        Returns
        -------
        Quantity
            Adjusted position size.

        """
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        if self._peak_equity <= 0:
            return size

        drawdown_pct = (self._peak_equity - self._current_equity) / self._peak_equity

        if drawdown_pct > 0.05:  # In drawdown > 5%
            # Reduce size proportionally
            reduction_factor = 1.0 - (drawdown_pct * self.config.drawdown_reduction_factor)
            reduction_factor = max(reduction_factor, 0.3)  # Minimum 30% of original

            from nautilus_trader.model.objects import Quantity

            adjusted_value = float(size.as_double()) * reduction_factor
            return Quantity.from_str(str(adjusted_value))

        return size

    def update_daily_pnl(self, pnl: float, ts_event: int | None = None) -> None:
        """
        Update daily P&L for risk tracking.

        Parameters
        ----------
        pnl : float
            P&L to add to daily total.
        ts_event : int | None, optional
            Event timestamp (nanoseconds) used for daily reset alignment.

        """
        self._check_daily_reset(ts_event)
        self._daily_pnl += pnl
        self._current_equity += pnl
        self._trades_today += 1

        if pnl < 0:
            self._losses_today += 1

        # Check if we should halt trading
        self.check_daily_limits(ts_event)

    def _check_daily_reset(self, ts_event: int | None = None) -> None:
        """
        Check if daily counters should be reset.
        """
        now = self._resolve_reset_reference(ts_event)
        if now >= self._daily_reset_time:
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._losses_today = 0
            self._trading_halted = False
            self._halt_reason = None
            self._halt_detail = None
            self._daily_reset_time = self._get_next_reset_time(now)
            logger.info("Daily risk counters reset")

    def _resolve_reset_reference(self, ts_event: int | None) -> datetime:
        """
        Resolve the timestamp used for daily resets.

        Parameters
        ----------
        ts_event : int | None
            Event timestamp (nanoseconds) when available.

        Returns
        -------
        datetime
            Timestamp used for daily reset comparisons.
        """
        if ts_event is None:
            return datetime.now()
        return datetime.fromtimestamp(ts_event / 1_000_000_000)

    def _get_next_reset_time(self, now: datetime | None = None) -> datetime:
        """
        Get next daily reset time (midnight).
        """
        reference = now or datetime.now()
        tomorrow = reference.date() + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time())

    def get_risk_metrics(self) -> dict[str, float]:
        """
        Get current risk metrics.

        Returns
        -------
        dict[str, float]
            Current risk metrics.

        """
        drawdown_pct = 0.0
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - self._current_equity) / self._peak_equity

        return {
            "daily_pnl": self._daily_pnl,
            "daily_trades": float(self._trades_today),
            "daily_losses": float(self._losses_today),
            "current_drawdown_pct": drawdown_pct,
            "peak_equity": self._peak_equity,
            "current_equity": self._current_equity,
            "trading_halted": float(self._trading_halted),
        }


# ===== Public API =====
__all__ = [
    "RiskAction",
    "RiskActionDecision",
    "RiskConfig",
    "RiskLiquidationConfig",
    "RiskManager",
]
