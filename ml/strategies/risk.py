"""
Risk management for ML trading strategies.

This module provides unified risk management including per-trade checks, portfolio-level
limits, and dynamic adjustments. Designed for capital preservation with <5ms hot path
checks.

"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.objects import Quantity
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

    # Correlation threshold
    correlation_threshold: float = 0.7  # Consider correlated above 0.7


# ===== Unified Risk Manager =====
class RiskManager:
    """
    Unified risk management with all checks in one place.

    Handles per-trade validation, portfolio limits, correlation checks, and dynamic
    adjustments based on performance.

    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        """
        Initialize risk manager.

        Parameters
        ----------
        config : RiskConfig, optional
            Risk management configuration.

        """
        self.config = config or RiskConfig()

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
        self._position_correlations: dict[tuple[InstrumentId, InstrumentId], float] = {}

        # Circuit breaker state
        self._trading_halted: bool = False
        self._halt_reason: str = ""

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
                logger.warning(f"Trading halted: {self._halt_reason}")
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
        for position in self._iter_positions(portfolio, instrument=None):
            try:
                if position.is_open:
                    total_exposure += abs(float(position.quantity.as_double()))
            except Exception as exc:
                logger.debug(
                    "risk.position_exposure_failed",
                    exc_info=True,
                    extra={"error": str(exc)},
                )

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
        for position in self._iter_positions(portfolio, instrument=instrument):
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
    ) -> list[Any]:
        """
        Safely retrieve positions from a portfolio with API-compat fallback.

        Returns an empty list when the portfolio does not expose a compatible positions
        API or when calls fail.

        """
        if hasattr(portfolio, "positions"):
            try:
                return list(portfolio.positions())
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
                return list(portfolio.positions_open())
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
        return []

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
        # Check cache
        key = (inst1, inst2) if str(inst1) < str(inst2) else (inst2, inst1)
        if key in self._position_correlations:
            return self._position_correlations[key]

        # For now, return a simple heuristic
        # In production, calculate from historical returns
        if inst1.symbol == inst2.symbol:
            correlation = 1.0
        elif inst1.venue == inst2.venue:
            correlation = 0.3  # Same venue, some correlation
        else:
            correlation = 0.1  # Different venues, low correlation

        self._position_correlations[key] = correlation
        return correlation

    def check_daily_limits(self) -> bool:
        """
        Check if daily risk limits have been exceeded.

        Returns
        -------
        bool
            True if trading can continue, False if limits exceeded.

        """
        # Check daily reset
        self._check_daily_reset()

        # Check daily loss limit
        if self._daily_pnl < 0:
            daily_loss_pct = (
                abs(self._daily_pnl / self._current_equity) if self._current_equity > 0 else 0
            )

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
                self._halt_reason = f"Daily loss limit reached: {daily_loss_pct:.1%}"
                logger.error(self._halt_reason)
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

    def update_daily_pnl(self, pnl: float) -> None:
        """
        Update daily P&L for risk tracking.

        Parameters
        ----------
        pnl : float
            P&L to add to daily total.

        """
        self._daily_pnl += pnl
        self._current_equity += pnl
        self._trades_today += 1

        if pnl < 0:
            self._losses_today += 1

        # Check if we should halt trading
        self.check_daily_limits()

    def _check_daily_reset(self) -> None:
        """
        Check if daily counters should be reset.
        """
        now = datetime.now()
        if now >= self._daily_reset_time:
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._losses_today = 0
            self._trading_halted = False
            self._halt_reason = ""
            self._daily_reset_time = self._get_next_reset_time()
            logger.info("Daily risk counters reset")

    def _get_next_reset_time(self) -> datetime:
        """
        Get next daily reset time (midnight).
        """
        now = datetime.now()
        tomorrow = now.date() + timedelta(days=1)
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
    "RiskConfig",
    "RiskManager",
]
