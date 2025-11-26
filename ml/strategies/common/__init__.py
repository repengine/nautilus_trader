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

from ml.strategies.common.decision_persistence import CircuitBreakerProtocol
from ml.strategies.common.decision_persistence import DecisionPersistenceComponent
from ml.strategies.common.decision_persistence import LoggerProtocol
from ml.strategies.common.decision_persistence import StrategyStoreProtocol
from ml.strategies.common.lifecycle import LifecycleComponent
from ml.strategies.common.order_submission import OrderExecutorProtocol
from ml.strategies.common.order_submission import OrderSubmissionComponent
from ml.strategies.common.order_submission import PerformanceTrackerProtocol
from ml.strategies.common.performance_tracking import PerformanceTrackingComponent
from ml.strategies.common.position_management import CacheProtocol
from ml.strategies.common.position_management import PortfolioManagerProtocol
from ml.strategies.common.position_management import PositionManagementComponent
from ml.strategies.common.position_management import PositionSizerProtocol
from ml.strategies.common.position_management import RiskManagerProtocol
from ml.strategies.common.signal_routing import SignalRoutingComponent


__all__ = [
    "CacheProtocol",
    "CircuitBreakerProtocol",
    "DecisionPersistenceComponent",
    "LifecycleComponent",
    "LoggerProtocol",
    "OrderExecutorProtocol",
    "OrderSubmissionComponent",
    "PerformanceTrackerProtocol",
    "PerformanceTrackingComponent",
    "PortfolioManagerProtocol",
    "PositionManagementComponent",
    "PositionSizerProtocol",
    "RiskManagerProtocol",
    "SignalRoutingComponent",
    "StrategyStoreProtocol",
]
