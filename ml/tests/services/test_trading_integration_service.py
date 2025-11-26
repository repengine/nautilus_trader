"""Tests for the trading integration service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

import pytest

from ml.dashboard.services.trading_service import (
    EmergencyStopActions,
    TradingHealthSnapshot,
    TradingIntegrationService,
    TradingMode,
    TradingStateSnapshot,
    TradingToggleRequest,
    TradingMetricsSnapshot,
)


@dataclass(slots=True)
class DummyIntegrationManager:
    """Stub integration manager exposing a trading controller."""

    trading_controller: Any | None = None
    db_connection: str | None = None


class DummyTradingController:
    """Deterministic trading controller used in service tests."""

    def __init__(self) -> None:
        self.enabled = False
        self.mode = TradingMode.STOPPED
        self.last_transition: str | None = None
        self.safety_pass = True
        self.safety_checks = {
            "risk_limits": True,
            "account_balance": True,
            "model_health": True,
        }
        self.enable_calls = 0
        self.disable_calls = 0
        self.emergency_calls = 0

    def get_state(self) -> TradingStateSnapshot:
        return TradingStateSnapshot(
            mode=self.mode,
            trading_enabled=self.enabled,
            last_transition=self.last_transition,
        )

    def run_safety_checks(self) -> dict[str, bool]:
        if not self.safety_pass:
            checks = self.safety_checks.copy()
            checks["model_health"] = False
            return checks
        return dict(self.safety_checks)

    def enable_trading(self, *, mode: TradingMode) -> None:
        self.enable_calls += 1
        self.mode = mode
        self.enabled = True
        self.last_transition = "enabled"

    def disable_trading(self) -> None:
        self.disable_calls += 1
        self.mode = TradingMode.STOPPED
        self.enabled = False
        self.last_transition = "disabled"

    def emergency_stop(self) -> EmergencyStopActions:
        self.emergency_calls += 1
        self.disable_trading()
        return EmergencyStopActions(orders_cancelled=10, positions_closed=3, actors_stopped=2)


@pytest.mark.asyncio
async def test_toggle_trading_with_controller_success() -> None:
    controller = DummyTradingController()
    service = TradingIntegrationService(DummyIntegrationManager(trading_controller=controller))

    request = TradingToggleRequest(
        enable=True,
        safety_checks={"risk_limits": True, "account_balance": True, "model_health": True},
    )

    result = await service.toggle_live_trading(request)

    assert result.success is True
    assert result.mode == TradingMode.LIVE
    assert result.live_trading_enabled is True
    assert controller.enable_calls == 1

    disable_result = await service.toggle_live_trading(TradingToggleRequest(enable=False))
    assert disable_result.live_trading_enabled is False
    assert controller.disable_calls == 1


@pytest.mark.asyncio
async def test_toggle_trading_fails_when_safety_check_fails() -> None:
    controller = DummyTradingController()
    controller.safety_pass = False
    service = TradingIntegrationService(DummyIntegrationManager(trading_controller=controller))

    result = await service.toggle_live_trading(
        TradingToggleRequest(
            enable=True,
            safety_checks={"risk_limits": True, "account_balance": True, "model_health": True},
        )
    )

    assert result.success is False
    assert "Controller safety checks failed" in (result.error or "")
    assert controller.enable_calls == 0


@pytest.mark.asyncio
async def test_emergency_stop_uses_controller_actions() -> None:
    controller = DummyTradingController()
    service = TradingIntegrationService(DummyIntegrationManager(trading_controller=controller))

    result = await service.emergency_stop()

    assert result.success is True
    assert result.actions_taken.orders_cancelled == 10
    assert controller.emergency_calls == 1


@pytest.mark.asyncio
async def test_health_check_reflects_controller_state() -> None:
    controller = DummyTradingController()
    controller.enable_trading(mode=TradingMode.PAPER)
    service = TradingIntegrationService(DummyIntegrationManager(trading_controller=controller))

    snapshot = await service.health_check()
    health = TradingHealthSnapshot(**snapshot)

    assert health.trading_enabled is True
    assert health.mode == TradingMode.PAPER


@pytest.mark.asyncio
async def test_toggle_trading_without_controller_uses_fallback() -> None:
    service = TradingIntegrationService(DummyIntegrationManager(trading_controller=None))


@pytest.mark.asyncio
async def test_health_check_includes_portfolio_metrics(test_database: Any) -> None:
    controller = DummyTradingController()
    manager = DummyIntegrationManager(
        trading_controller=controller,
        db_connection=test_database.connection_string,
    )
    service = TradingIntegrationService(manager)

    with test_database.engine.begin() as conn:
        conn.execute(text("DELETE FROM ml_positions"))
        conn.execute(
            text(
                """
                INSERT INTO ml_positions (
                    strategy_id,
                    instrument_id,
                    quantity,
                    side,
                    entry_price,
                    current_price,
                    unrealized_pnl,
                    realized_pnl,
                    position_value,
                    exposure,
                    var_95,
                    entry_time,
                    last_update
                ) VALUES
                    ('strat-gamma', 'BTC/USD', 0.5, 'LONG', 20000.0, 21000.0, 500.0, 100.0, 10000.0, 5000.0, 50.0, 0, 0)
                """
            )
        )

    snapshot = await service.health_check()
    health = TradingHealthSnapshot(**snapshot)

    assert health.total_positions == 1
    assert health.total_exposure == pytest.approx(5000.0)
    assert health.unrealized_pnl == pytest.approx(500.0)

    metrics = await service.get_trading_metrics()
    assert isinstance(metrics, TradingMetricsSnapshot)
    assert metrics.total_positions == 1
    assert metrics.total_exposure == pytest.approx(5000.0)
    assert metrics.strategies and metrics.strategies[0].strategy_id == "strat-gamma"

    result = await service.toggle_live_trading(
        TradingToggleRequest(enable=True, safety_checks={"risk_limits": True})
    )

    assert result.success is True
    assert result.mode == TradingMode.LIVE
    assert result.live_trading_enabled is True
