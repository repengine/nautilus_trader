"""
Tests for staged risk actions in RiskManager.
"""

from __future__ import annotations

import pytest

from ml.strategies.risk import RiskAction
from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskLiquidationConfig
from ml.strategies.risk import RiskManager


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


def test_get_risk_action_liquidates_on_daily_loss_threshold() -> None:
    """Ensure daily loss liquidation triggers when configured."""
    config = RiskConfig(
        liquidation_config=RiskLiquidationConfig(
            enabled=True,
            daily_loss_limit_pct=0.1,
        ),
    )
    manager = RiskManager(config)
    manager._current_equity = 1000.0
    manager._peak_equity = 1000.0
    manager._daily_pnl = -200.0

    decision = manager.get_risk_action()

    assert decision.action is RiskAction.LIQUIDATE
    assert decision.reason == "daily_loss_liquidate"
    assert manager.is_trading_halted() is True


def test_get_risk_action_liquidates_on_drawdown_threshold() -> None:
    """Ensure drawdown liquidation triggers when configured."""
    config = RiskConfig(
        liquidation_config=RiskLiquidationConfig(
            enabled=True,
            drawdown_limit_pct=0.2,
        ),
    )
    manager = RiskManager(config)
    manager._peak_equity = 1000.0
    manager._current_equity = 700.0
    manager._daily_pnl = 0.0

    decision = manager.get_risk_action()

    assert decision.action is RiskAction.LIQUIDATE
    assert decision.reason == "drawdown_liquidate"
    assert manager.is_trading_halted() is True


def test_get_risk_action_halts_on_cooldown() -> None:
    """Ensure liquidation cooldown halts repeat actions."""
    config = RiskConfig(
        liquidation_config=RiskLiquidationConfig(
            enabled=True,
            daily_loss_limit_pct=0.1,
            cooldown_ms=60_000,
        ),
    )
    manager = RiskManager(config)
    manager._current_equity = 1000.0
    manager._peak_equity = 1000.0
    manager._daily_pnl = -200.0

    decision = manager.get_risk_action(ts_event=1_000_000_000)

    assert decision.action is RiskAction.LIQUIDATE

    cooldown_decision = manager.get_risk_action(ts_event=1_000_000_000 + 1_000_000)

    assert cooldown_decision.action is RiskAction.HALT
    assert cooldown_decision.reason == "liquidation_cooldown"
