"""
ML Signal Actor compatibility layer.

This module preserves the public API of ``ml.actors.signal`` while delegating
actor behavior to the component-based facade implementation.

"""

from __future__ import annotations

from enum import Enum
from typing import Any

from ml.actors.base import MLSignal
from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent
from ml.actors.common.signal_strategy import AdaptiveStrategy
from ml.actors.common.signal_strategy import EnsembleStrategy
from ml.actors.common.signal_strategy import ExtremesStrategy
from ml.actors.common.signal_strategy import MomentumStrategy
from ml.actors.common.signal_strategy import SignalGenerationStrategy
from ml.actors.common.signal_strategy import SignalPolicy
from ml.actors.common.signal_strategy import SignalPolicySwapper
from ml.actors.common.signal_strategy import SignalStrategy
from ml.actors.common.signal_strategy import StrategySwapper
from ml.actors.common.signal_strategy import ThresholdSignalStrategy
from ml.actors.common.signal_strategy import ThresholdStrategy
from ml.actors.signal_facade_impl import MLSignalActorFacade as MLSignalActor
from ml.common import metrics_bootstrap
from ml.common.metrics_manager import MetricsManager
from ml.config.actors import MLSignalActorConfig
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig
from ml.config.names import FEATURE_TIME_BUCKETS, LABEL_ACTOR_ID


AdaptiveSignal = MLSignal
PerformanceMonitor = PerformanceMonitoringComponent

_feature_time_by_feature_set_metric = metrics_bootstrap.get_histogram(
    "ml_feature_time_by_set_seconds",
    "Feature computation latency by feature_set_id",
    ["actor_id", "feature_set_id"],
    buckets=FEATURE_TIME_BUCKETS,
)

_metrics_manager = MetricsManager.default()
_feature_parity_checks_total = _metrics_manager.counter(
    "ml_feature_parity_checks_total",
    "Total parity smoke-checks executed",
    [LABEL_ACTOR_ID],
)
_feature_parity_drift = _metrics_manager.gauge(
    "ml_feature_parity_drift",
    "Max absolute feature difference in parity smoke-check",
    [LABEL_ACTOR_ID],
)


class OptimizationLevel(Enum):
    """
    Performance optimization level.
    """

    STANDARD = "standard"
    OPTIMIZED = "optimized"


class ModelSwapper:
    """
    Atomic model swapping for hot reload.
    """

    def __init__(self) -> None:
        """
        Initialize the ModelSwapper.
        """
        self._current_model: object | None = None
        self._current_metadata: dict[str, Any] | None = None
        self._next_model: object | None = None
        self._next_metadata: dict[str, Any] | None = None
        self._swap_pending = False
        self._load_error: Exception | None = None

    @property
    def current_model(self) -> object | None:
        """
        Get current model.
        """
        return self._current_model

    @property
    def current_metadata(self) -> dict[str, Any] | None:
        """
        Get current metadata.
        """
        return self._current_metadata

    @property
    def swap_pending(self) -> bool:
        """
        Check if swap is pending.
        """
        return self._swap_pending

    @property
    def load_error(self) -> Exception | None:
        """
        Get load error if any.
        """
        return self._load_error

    def set_current_model(self, model: object, metadata: dict[str, Any] | None = None) -> None:
        """
        Set current model.
        """
        self._current_model = model
        self._current_metadata = metadata or {}
        self._load_error = None

    def set_current(self, model: object, metadata: dict[str, Any] | None = None) -> None:
        """
        Set current model (backward compatibility).
        """
        self.set_current_model(model, metadata)

    def prepare_swap(self, model: object, metadata: dict[str, Any] | None = None) -> None:
        """
        Prepare model swap.
        """
        self._next_model = model
        self._next_metadata = metadata or {}
        self._swap_pending = True
        self._load_error = None

    def prepare_swap_with_error(self, error: Exception) -> None:
        """
        Set error when model loading fails.
        """
        self._load_error = error
        self._swap_pending = False

    def execute_swap(self) -> bool:
        """
        Execute model swap atomically.
        """
        if not self._swap_pending:
            return False

        old_model = self._current_model
        self._current_model = self._next_model
        self._current_metadata = self._next_metadata
        self._next_model = None
        self._next_metadata = None
        self._swap_pending = False
        del old_model
        return True


__all__ = [
    "AdaptiveSignal",
    "AdaptiveStrategy",
    "EnsembleStrategy",
    "ExtremesStrategy",
    "MLSignalActor",
    "MLSignalActorConfig",
    "ModelSwapper",
    "MomentumStrategy",
    "OptimizationConfig",
    "OptimizationLevel",
    "PerformanceMonitor",
    "SignalGenerationStrategy",
    "SignalPolicy",
    "SignalPolicySwapper",
    "SignalStrategy",
    "StrategyConfig",
    "StrategySwapper",
    "ThresholdSignalStrategy",
    "ThresholdStrategy",
]
