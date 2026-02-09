"""
ML Actor Components.

This module provides reusable components for ML actors, enabling cleaner separation of
concerns and easier testing.

Components follow the Protocol-First design pattern, allowing flexible composition and
dependency injection.

"""

from ml.actors.common.adaptive_threshold import AdaptiveThresholdComponent
from ml.actors.common.chronos_inference import ChronosInferenceAdapter
from ml.actors.common.chronos_inference import ChronosPredictorProtocol
from ml.actors.common.drift_monitoring import DriftMonitoringComponent
from ml.actors.common.drift_monitoring import DriftObservation
from ml.actors.common.drift_monitoring import DriftPolicyConfig
from ml.actors.common.drift_monitoring import DriftPolicyOutcome
from ml.actors.common.drift_monitoring import DriftThresholds
from ml.actors.common.drift_monitoring import evaluate_drift_policy_outcome
from ml.actors.common.drift_monitoring import resolve_drift_policy_config
from ml.actors.common.drift_monitoring import resolve_replay_safe_drift_action
from ml.actors.common.drift_monitoring import threshold_for_policy_action
from ml.actors.common.features import FeaturesComponent
from ml.actors.common.features import FeaturesProtocol
from ml.actors.common.model import ModelComponent
from ml.actors.common.model import ModelProtocol
from ml.actors.common.model_warmup import ModelWarmUpComponent
from ml.actors.common.performance_monitoring import PerformanceMonitoringComponent
from ml.actors.common.prediction_buffer import PredictionBufferComponent
from ml.actors.common.registry import RegistryComponent
from ml.actors.common.registry import RegistryProtocol
from ml.actors.common.signal_metadata import build_prediction_surface_metadata
from ml.actors.common.signal_metadata import build_signal_metadata
from ml.actors.common.signal_strategy import AdaptiveStrategy
from ml.actors.common.signal_strategy import EnsembleStrategy
from ml.actors.common.signal_strategy import ExtremesStrategy
from ml.actors.common.signal_strategy import MomentumStrategy
from ml.actors.common.signal_strategy import SignalGenerationStrategy
from ml.actors.common.signal_strategy import SignalPolicy
from ml.actors.common.signal_strategy import SignalPolicySwapper
from ml.actors.common.signal_strategy import SignalStrategy
from ml.actors.common.signal_strategy import SignalStrategyComponent
from ml.actors.common.signal_strategy import StrategySwapper
from ml.actors.common.signal_strategy import ThresholdSignalStrategy
from ml.actors.common.signal_strategy import ThresholdStrategy
from ml.actors.common.store_operations import StoreOperationsComponent
from ml.actors.common.store_operations import StoreOperationsProtocol


__all__ = [
    "AdaptiveStrategy",
    "AdaptiveThresholdComponent",
    "ChronosInferenceAdapter",
    "ChronosPredictorProtocol",
    "DriftMonitoringComponent",
    "DriftObservation",
    "DriftPolicyConfig",
    "DriftPolicyOutcome",
    "DriftThresholds",
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
    "build_prediction_surface_metadata",
    "build_signal_metadata",
    "evaluate_drift_policy_outcome",
    "resolve_drift_policy_config",
    "resolve_replay_safe_drift_action",
    "threshold_for_policy_action",
]
