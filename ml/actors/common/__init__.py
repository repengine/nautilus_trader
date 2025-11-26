"""
ML Actor Components.

This module provides reusable components for ML actors, enabling cleaner separation of
concerns and easier testing.

Components follow the Protocol-First design pattern, allowing flexible composition and
dependency injection.

"""

from ml.actors.components.adaptive_threshold import AdaptiveThresholdComponent
from ml.actors.components.features import FeaturesComponent
from ml.actors.components.features import FeaturesProtocol
from ml.actors.components.model import ModelComponent
from ml.actors.components.model import ModelProtocol
from ml.actors.components.model_warmup import ModelWarmUpComponent
from ml.actors.components.performance_monitoring import PerformanceMonitoringComponent
from ml.actors.components.prediction_buffer import PredictionBufferComponent
from ml.actors.components.registry import RegistryComponent
from ml.actors.components.registry import RegistryProtocol
from ml.actors.components.signal_strategy import AdaptiveStrategy
from ml.actors.components.signal_strategy import EnsembleStrategy
from ml.actors.components.signal_strategy import ExtremesStrategy
from ml.actors.components.signal_strategy import MomentumStrategy
from ml.actors.components.signal_strategy import SignalGenerationStrategy
from ml.actors.components.signal_strategy import SignalPolicy
from ml.actors.components.signal_strategy import SignalPolicySwapper
from ml.actors.components.signal_strategy import SignalStrategy
from ml.actors.components.signal_strategy import SignalStrategyComponent
from ml.actors.components.signal_strategy import StrategySwapper
from ml.actors.components.signal_strategy import ThresholdSignalStrategy
from ml.actors.components.signal_strategy import ThresholdStrategy
from ml.actors.components.store_operations import StoreOperationsComponent
from ml.actors.components.store_operations import StoreOperationsProtocol


__all__ = [
    "AdaptiveStrategy",
    "AdaptiveThresholdComponent",
    "EnsembleStrategy",
    "ExtremesStrategy",
    "FeaturesComponent",
    "FeaturesProtocol",
    "ModelComponent",
    "ModelProtocol",
    "ModelWarmUpComponent",
    "MomentumStrategy",
    "PerformanceMonitoringComponent",
    "PredictionBufferComponent",
    "RegistryComponent",
    "RegistryProtocol",
    "SignalGenerationStrategy",
    "SignalPolicy",
    "SignalPolicySwapper",
    "SignalStrategy",
    "SignalStrategyComponent",
    "StoreOperationsComponent",
    "StoreOperationsProtocol",
    "StrategySwapper",
    "ThresholdSignalStrategy",
    "ThresholdStrategy",
]
