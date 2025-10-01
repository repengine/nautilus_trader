"""Trading control integration service."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, cast

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.core.db_engine import EngineManager
from ml.dashboard.services.base_service import BaseIntegrationService


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager


logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    """Trading execution modes supported by the dashboard."""

    STOPPED = "STOPPED"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(slots=True)
class TradingStateSnapshot:
    """Snapshot of the current trading controller state."""

    mode: TradingMode
    trading_enabled: bool
    last_transition: str | None


@dataclass(slots=True)
class TradingToggleRequest:
    """Request model for toggling live trading."""

    enable: bool
    safety_checks: Mapping[str, bool] | None = None


@dataclass(slots=True)
class TradingToggleResult:
    """Response when live trading is toggled."""

    success: bool
    live_trading_enabled: bool
    timestamp: str
    safety_checks_passed: bool
    mode: TradingMode = TradingMode.STOPPED
    controller_state: TradingStateSnapshot | None = None
    error: str | None = None


@dataclass(slots=True)
class EmergencyStopActions:
    """Summary of actions performed during an emergency stop."""

    orders_cancelled: int = 0
    positions_closed: int = 0
    actors_stopped: int = 0
    data_feeds_stopped: bool = True
    risk_manager_notified: bool = True


@dataclass(slots=True)
class EmergencyStopResult:
    """Feedback returned after an emergency stop."""

    success: bool
    timestamp: str
    actions_taken: EmergencyStopActions
    message: str
    error: str | None = None


@dataclass(slots=True)
class TradingHealthSnapshot:
    """Current health information for trading components."""

    healthy: bool
    trading_enabled: bool
    market_data: str
    risk_manager: str
    mode: TradingMode
    last_transition: str | None
    total_positions: int | None = None
    total_exposure: float | None = None
    unrealized_pnl: float | None = None


@dataclass(slots=True)
class TradingMetrics:
    """Aggregated trading metrics surfaced to the UI."""

    total_positions: int = 0
    total_exposure: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    strategies: tuple[StrategyExposure, ...] = ()


@dataclass(slots=True)
class StrategyExposure:
    """Per-strategy exposure view for the trading dashboard."""

    strategy_id: str
    positions: int
    exposure: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass(slots=True)
class TradingMetricsSnapshot:
    """Serializable trading metrics payload for the UI."""

    generated_at: str
    total_positions: int
    total_exposure: float
    unrealized_pnl: float
    realized_pnl: float
    strategies: tuple[StrategyExposure, ...]


class TradingControllerProtocol(Protocol):
    """Protocol describing trading controller capabilities."""

    def get_state(self) -> TradingStateSnapshot: ...

    def run_safety_checks(self) -> Mapping[str, bool]: ...

    def enable_trading(self, *, mode: TradingMode) -> None: ...

    def disable_trading(self) -> None: ...

    def emergency_stop(self) -> EmergencyStopActions: ...


class TradingIntegrationService(BaseIntegrationService):
    """Integration facade for trading controls."""

    def __init__(
        self,
        integration_manager: MLIntegrationManager | None,
    ) -> None:
        super().__init__(integration_manager)
        self._controller: TradingControllerProtocol | None = None
        self._trading_metrics = TradingMetrics()
        self._initialise_controller(integration_manager)

    def get_service_name(self) -> str:
        return "trading_integration"

    def set_integration_manager(self, integration_manager: MLIntegrationManager | None) -> None:
        super().set_integration_manager(integration_manager)
        self._initialise_controller(integration_manager)

    async def health_check(self) -> dict[str, Any]:
        engine = self._resolve_engine()
        if engine is not None:
            try:
                self._trading_metrics = await self._run_async(lambda: self._collect_metrics(engine))
            except Exception:  # pragma: no cover - defensive
                logger.debug("trading metrics aggregation failed", exc_info=True)

        if self._controller is None:
            snapshot = TradingHealthSnapshot(
                healthy=True,
                trading_enabled=False,
                market_data="unknown",
                risk_manager="unknown",
                mode=TradingMode.STOPPED,
                last_transition=None,
                total_positions=self._trading_metrics.total_positions or 0,
                total_exposure=self._trading_metrics.total_exposure,
                unrealized_pnl=self._trading_metrics.unrealized_pnl,
            )
            return asdict(snapshot)

        try:
            state = await self._run_async(self._controller.get_state)
        except Exception:  # pragma: no cover - defensive
            logger.debug("trading controller state retrieval failed", exc_info=True)
            snapshot = TradingHealthSnapshot(
                healthy=False,
                trading_enabled=False,
                market_data="unknown",
                risk_manager="unknown",
                mode=TradingMode.STOPPED,
                last_transition=None,
                total_positions=self._trading_metrics.total_positions,
                total_exposure=self._trading_metrics.total_exposure,
                unrealized_pnl=self._trading_metrics.unrealized_pnl,
            )
            return asdict(snapshot)

        snapshot = TradingHealthSnapshot(
            healthy=True,
            trading_enabled=state.trading_enabled,
            market_data="connected" if state.trading_enabled else "standby",
            risk_manager="active" if state.trading_enabled else "idle",
            mode=state.mode,
            last_transition=state.last_transition,
            total_positions=self._trading_metrics.total_positions,
            total_exposure=self._trading_metrics.total_exposure,
            unrealized_pnl=self._trading_metrics.unrealized_pnl,
        )
        return asdict(snapshot)

    async def get_trading_metrics(self) -> TradingMetricsSnapshot:
        """Return aggregated trading metrics for dashboard widgets."""
        engine = self._resolve_engine()
        metrics = self._trading_metrics
        if engine is not None:
            try:
                metrics = await self._run_async(lambda: self._collect_metrics(engine))
                self._trading_metrics = metrics
            except Exception:  # pragma: no cover - defensive
                logger.debug("trading metrics query failed", exc_info=True)

        snapshot = TradingMetricsSnapshot(
            generated_at=dt.datetime.now(tz=dt.UTC).isoformat(),
            total_positions=metrics.total_positions,
            total_exposure=metrics.total_exposure,
            unrealized_pnl=metrics.unrealized_pnl,
            realized_pnl=metrics.realized_pnl,
            strategies=metrics.strategies,
        )
        return snapshot

    async def toggle_live_trading(self, request: TradingToggleRequest) -> TradingToggleResult:
        """Toggle live trading with safety checks."""
        self._track_operation(operation="toggle_trading", status="started")

        safety_passed = not request.enable
        if request.enable:
            ui_checks = request.safety_checks or {}
            missing_acknowledgements = [name for name, passed in ui_checks.items() if not passed]
            if missing_acknowledgements:
                self._track_operation(operation="toggle_trading", status="failed_ui_checks")
                return TradingToggleResult(
                    success=False,
                    live_trading_enabled=False,
                    timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
                    safety_checks_passed=False,
                    error=f"User safety checks failed: {', '.join(missing_acknowledgements)}",
                )

        controller = self._controller
        if controller is None:
            if request.enable and not safety_passed:
                safety_passed = bool(request.safety_checks)
            result = TradingToggleResult(
                success=True,
                live_trading_enabled=request.enable,
                timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
                safety_checks_passed=safety_passed,
                mode=TradingMode.LIVE if request.enable else TradingMode.STOPPED,
            )
            self._track_operation(operation="toggle_trading", status="success")
            return result

        if request.enable:
            try:
                safety_results = await self._run_async(controller.run_safety_checks)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("safety checks failed", exc_info=True)
                self._track_operation(operation="toggle_trading", status="safety_error")
                return TradingToggleResult(
                    success=False,
                    live_trading_enabled=False,
                    timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
                    safety_checks_passed=False,
                    error=str(exc),
                )

            failed_controller_checks = [name for name, passed in safety_results.items() if not passed]
            if failed_controller_checks:
                self._track_operation(operation="toggle_trading", status="failed_checks")
                return TradingToggleResult(
                    success=False,
                    live_trading_enabled=False,
                    timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
                    safety_checks_passed=False,
                    error=f"Controller safety checks failed: {', '.join(failed_controller_checks)}",
                )
            safety_passed = True

            try:
                await self._run_async(lambda: controller.enable_trading(mode=TradingMode.LIVE))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("enable trading failed", exc_info=True)
                self._track_operation(operation="toggle_trading", status="enable_failed")
                return TradingToggleResult(
                    success=False,
                    live_trading_enabled=False,
                    timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
                    safety_checks_passed=True,
                    error=str(exc),
                )
        else:
            try:
                await self._run_async(controller.disable_trading)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("disable trading failed", exc_info=True)
                self._track_operation(operation="toggle_trading", status="disable_failed")
                return TradingToggleResult(
                    success=False,
                    live_trading_enabled=True,
                    timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
                    safety_checks_passed=True,
                    error=str(exc),
                )

        try:
            controller_state = await self._run_async(controller.get_state)
        except Exception:  # pragma: no cover - defensive
            controller_state = TradingStateSnapshot(
                mode=TradingMode.LIVE if request.enable else TradingMode.STOPPED,
                trading_enabled=request.enable,
                last_transition=None,
            )

        result = TradingToggleResult(
            success=True,
            live_trading_enabled=controller_state.trading_enabled,
            timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
            safety_checks_passed=safety_passed,
            mode=controller_state.mode,
            controller_state=controller_state,
        )
        self._track_operation(operation="toggle_trading", status="success")
        return result

    async def emergency_stop(self) -> EmergencyStopResult:
        """Execute the emergency stop procedure."""
        self._track_operation(operation="emergency_stop", status="triggered")

        controller = self._controller
        actions = EmergencyStopActions()
        if controller is not None:
            try:
                actions = await self._run_async(controller.emergency_stop)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("controller emergency stop failed", exc_info=True)
                result = EmergencyStopResult(
                    success=False,
                    timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
                    actions_taken=actions,
                    message="Emergency stop encountered an error",
                    error=str(exc),
                )
                self._track_operation(operation="emergency_stop", status="failed")
                return result

        result = EmergencyStopResult(
            success=True,
            timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
            actions_taken=actions,
            message="Emergency stop executed successfully",
        )
        self._track_operation(operation="emergency_stop", status="completed")
        return result

    def _initialise_controller(self, integration_manager: MLIntegrationManager | None) -> None:
        controller = None
        if integration_manager is not None:
            candidate = getattr(integration_manager, "trading_controller", None)
            if candidate and all(hasattr(candidate, attr) for attr in ("get_state", "enable_trading", "disable_trading")):
                controller = cast(TradingControllerProtocol, candidate)

        self._controller = controller

    def _resolve_engine(self) -> Engine | None:
        integration = self._integration
        if integration is None:
            return None
        connection = getattr(integration, "db_connection", None)
        if not connection:
            return None
        try:
            return EngineManager.get_engine(connection)
        except Exception:  # pragma: no cover - defensive
            return None

    def _collect_metrics(self, engine: Engine) -> TradingMetrics:
        metrics = TradingMetrics()
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS position_count,
                        COALESCE(SUM(exposure), 0) AS total_exposure,
                        COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
                        COALESCE(SUM(realized_pnl), 0) AS realized_pnl
                    FROM ml_positions
                    """
                ),
            ).one_or_none()
            strategy_rows = conn.execute(
                text(
                    """
                    SELECT
                        strategy_id,
                        COUNT(*) AS position_count,
                        COALESCE(SUM(exposure), 0) AS total_exposure,
                        COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
                        COALESCE(SUM(realized_pnl), 0) AS realized_pnl
                    FROM ml_positions
                    GROUP BY strategy_id
                    ORDER BY strategy_id
                    """
                ),
            ).fetchall()
        if row is None:
            return metrics

        mapping = row._mapping
        metrics.total_positions = int(mapping.get("position_count") or 0)
        metrics.total_exposure = float(mapping.get("total_exposure") or 0.0)
        metrics.unrealized_pnl = float(mapping.get("unrealized_pnl") or 0.0)
        metrics.realized_pnl = float(mapping.get("realized_pnl") or 0.0)
        breakdown: list[StrategyExposure] = []
        for strategy_row in strategy_rows:
            strategy_map = strategy_row._mapping
            strategy_id = str(strategy_map.get("strategy_id") or "unknown")
            breakdown.append(
                StrategyExposure(
                    strategy_id=strategy_id,
                    positions=int(strategy_map.get("position_count") or 0),
                    exposure=float(strategy_map.get("total_exposure") or 0.0),
                    unrealized_pnl=float(strategy_map.get("unrealized_pnl") or 0.0),
                    realized_pnl=float(strategy_map.get("realized_pnl") or 0.0),
                )
            )
        metrics.strategies = tuple(breakdown)
        return metrics


__all__ = [
    "EmergencyStopActions",
    "EmergencyStopResult",
    "StrategyExposure",
    "TradingHealthSnapshot",
    "TradingIntegrationService",
    "TradingMetricsSnapshot",
    "TradingMode",
    "TradingStateSnapshot",
    "TradingToggleRequest",
    "TradingToggleResult",
]
