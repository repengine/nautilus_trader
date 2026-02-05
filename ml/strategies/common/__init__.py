"""
Strategy components for MLTradingStrategy decomposition.

This package contains extracted components from BaseMLStrategy following
the facade pattern with feature flag control for backward compatibility.

Components:
- SignalRoutingComponent: Signal filtering, aggregation, and routing
- DecisionPersistenceComponent: Strategy decision persistence and event publishing
- PositionManagementComponent: Position sizing, risk validation, portfolio allocation
- OrderSubmissionComponent: Order creation, smart execution, stop loss placement
- LifecycleComponent: Strategy lifecycle management (startup, shutdown, subscriptions)
- PerformanceTrackingComponent: Model performance tracking and metrics recording
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ml.strategies.common.correlation import CorrelationProviderProtocol
from ml.strategies.common.correlation import CorrelationSnapshot
from ml.strategies.common.decision_persistence import CircuitBreakerProtocol
from ml.strategies.common.decision_persistence import DecisionPersistenceComponent
from ml.strategies.common.decision_persistence import LoggerProtocol
from ml.strategies.common.decision_persistence import StrategyStoreProtocol
from ml.strategies.common.intent_positions import IntentPosition
from ml.strategies.common.intent_positions import OrderIntentPositionTracker
from ml.strategies.common.lifecycle import LifecycleComponent
from ml.strategies.common.model_exit_policy import ExitDecision
from ml.strategies.common.model_exit_policy import ModelSignalProtocol
from ml.strategies.common.model_exit_policy import PositionSideProtocol
from ml.strategies.common.model_exit_policy import evaluate_model_exit
from ml.strategies.common.model_exit_policy import resolve_time_in_trade_ns
from ml.strategies.common.order_submission import OrderExecutorProtocol
from ml.strategies.common.order_submission import OrderSubmissionComponent
from ml.strategies.common.order_submission import PerformanceTrackerProtocol
from ml.strategies.common.performance_tracking import PerformanceTrackingComponent
from ml.strategies.common.position_management import CacheProtocol
from ml.strategies.common.position_management import PortfolioManagerProtocol
from ml.strategies.common.position_management import PositionManagementComponent
from ml.strategies.common.position_management import PositionSizerProtocol
from ml.strategies.common.position_management import RiskManagerProtocol
from ml.strategies.common.positions import PositionsHealthStatus
from ml.strategies.common.positions import PositionsMetadata
from ml.strategies.common.positions import PositionsProviderProtocol
from ml.strategies.common.positions import PositionsSnapshot
from ml.strategies.common.positions import PositionViewProtocol
from ml.strategies.common.positions import build_positions_metadata
from ml.strategies.common.positions_provider import NautilusPositionsProvider
from ml.strategies.common.signal_routing import SignalRoutingComponent


if TYPE_CHECKING:
    from ml.strategies.common.returns_updater import ReturnsUpdater
    from ml.strategies.common.returns_updater import ReturnUpdateResult


_LAZY_STRATEGY_EXPORTS: dict[str, tuple[str, str]] = {
    "ReturnUpdateResult": ("ml.strategies.common.returns_updater", "ReturnUpdateResult"),
    "ReturnsUpdater": ("ml.strategies.common.returns_updater", "ReturnsUpdater"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_STRATEGY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


__all__ = [
    "CacheProtocol",
    "CircuitBreakerProtocol",
    "CorrelationProviderProtocol",
    "CorrelationSnapshot",
    "DecisionPersistenceComponent",
    "ExitDecision",
    "IntentPosition",
    "LifecycleComponent",
    "LoggerProtocol",
    "ModelSignalProtocol",
    "NautilusPositionsProvider",
    "OrderExecutorProtocol",
    "OrderIntentPositionTracker",
    "OrderSubmissionComponent",
    "PerformanceTrackerProtocol",
    "PerformanceTrackingComponent",
    "PortfolioManagerProtocol",
    "PositionManagementComponent",
    "PositionSideProtocol",
    "PositionSizerProtocol",
    "PositionViewProtocol",
    "PositionsHealthStatus",
    "PositionsMetadata",
    "PositionsProviderProtocol",
    "PositionsSnapshot",
    "ReturnUpdateResult",
    "ReturnsUpdater",
    "RiskManagerProtocol",
    "SignalRoutingComponent",
    "StrategyStoreProtocol",
    "build_positions_metadata",
    "evaluate_model_exit",
    "resolve_time_in_trade_ns",
]
