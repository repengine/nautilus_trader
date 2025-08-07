# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Enhanced configuration classes for optimized ML Signal Actor.

This module provides configuration classes specifically designed for hot path
optimization with advanced signal generation strategies and performance tuning
parameters.

"""

from __future__ import annotations

from enum import Enum

# Import MLSignalActorConfig for inheritance
from typing import TYPE_CHECKING, Any

from nautilus_trader.common.config import PositiveInt


if TYPE_CHECKING:
    from ml.actors.signal import MLSignalActorConfig


class SignalStrategy(Enum):
    """
    Signal generation strategy enumeration.

    Defines different approaches for converting model predictions into trading signals
    with varying levels of sophistication and computational complexity.

    """

    THRESHOLD = "threshold"
    EXTREMES = "extremes"
    MOMENTUM = "momentum"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


class ThresholdStrategy(Enum):
    """
    Threshold calculation strategy for signal generation.

    Defines how prediction thresholds are calculated and applied for signal generation
    decisions.

    """

    FIXED = "fixed"
    PERCENTILE = "percentile"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    REGIME_AWARE = "regime_aware"


class ONNXOptimizationConfig:
    """
    ONNX runtime optimization configuration for minimal latency.

    This configuration is specifically tuned for single-threaded inference
    with maximum performance for real-time trading applications.

    Parameters
    ----------
    graph_optimization_level : str, default "ORT_ENABLE_ALL"
        Graph optimization level for ONNX runtime.
    execution_mode : str, default "ORT_SEQUENTIAL"
        Execution mode for ONNX runtime (sequential for predictable latency).
    intra_op_num_threads : int, default 1
        Number of threads for intra-op parallelism (1 for predictable latency).
    inter_op_num_threads : int, default 1
        Number of threads for inter-op parallelism (1 for predictable latency).
    enable_cpu_mem_arena : bool, default False
        Whether to enable CPU memory arena (disabled for lower latency).
    enable_mem_pattern : bool, default False
        Whether to enable memory pattern optimization.
    enable_mem_reuse : bool, default True
        Whether to enable memory reuse optimization.
    provider_options : dict[str, Any], optional
        Additional provider-specific options for CPU execution provider.

    """

    def __init__(
        self,
        graph_optimization_level: str = "ORT_ENABLE_ALL",
        execution_mode: str = "ORT_SEQUENTIAL",
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 1,
        enable_cpu_mem_arena: bool = False,
        enable_mem_pattern: bool = False,
        enable_mem_reuse: bool = True,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        self.graph_optimization_level = graph_optimization_level
        self.execution_mode = execution_mode
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads
        self.enable_cpu_mem_arena = enable_cpu_mem_arena
        self.enable_mem_pattern = enable_mem_pattern
        self.enable_mem_reuse = enable_mem_reuse
        self.provider_options = provider_options or {
            "arena_extend_strategy": "kSameAsRequested",
        }


class AdaptiveThresholdsConfig:
    """
    Configuration for adaptive threshold calculation.

    Parameters
    ----------
    base_threshold : float, default 0.7
        Base threshold value before adaptations.
    volatility_factor : float, default 2.0
        Factor to adjust threshold based on market volatility.
    trend_factor : float, default 1.5
        Factor to adjust threshold based on trend strength.
    volume_factor : float, default 1.2
        Factor to adjust threshold based on volume patterns.
    regime_multipliers : dict[str, float], optional
        Regime-specific threshold multipliers.
    min_threshold : float, default 0.1
        Minimum allowed threshold value.
    max_threshold : float, default 0.95
        Maximum allowed threshold value.
    update_frequency : int, default 10
        How often to update adaptive thresholds (in bars).

    """

    def __init__(
        self,
        base_threshold: float = 0.7,
        volatility_factor: float = 2.0,
        trend_factor: float = 1.5,
        volume_factor: float = 1.2,
        regime_multipliers: dict[str, float] | None = None,
        min_threshold: float = 0.1,
        max_threshold: float = 0.95,
        update_frequency: int = 10,
    ) -> None:
        self.base_threshold = base_threshold
        self.volatility_factor = volatility_factor
        self.trend_factor = trend_factor
        self.volume_factor = volume_factor
        self.regime_multipliers = regime_multipliers or {
            "volatile": 1.3,
            "trending": 0.9,
            "ranging": 1.1,
        }
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.update_frequency = update_frequency


class HotPathOptimizationConfig:
    """
    Configuration for hot path performance optimizations.

    Parameters
    ----------
    enable_zero_copy : bool, default True
        Whether to enable zero-copy operations where possible.
    pre_allocate_buffers : bool, default True
        Whether to pre-allocate all buffers at initialization.
    use_memoryviews : bool, default True
        Whether to use memoryviews for buffer access.
    feature_buffer_size : int, default 256
        Size of feature buffer (must match model input size).
    history_buffer_size : int, default 1000
        Size of prediction history buffers.
    reservoir_sample_size : int, default 1000
        Size of reservoir for percentile sampling.
    max_feature_latency_us : int, default 500
        Maximum allowed feature computation latency in microseconds.
    max_inference_latency_us : int, default 2000
        Maximum allowed inference latency in microseconds.
    circuit_breaker_threshold : float, default 0.1
        Failure rate threshold for circuit breaker activation.
    circuit_breaker_timeout_ms : int, default 5000
        Circuit breaker timeout in milliseconds.

    """

    def __init__(
        self,
        enable_zero_copy: bool = True,
        pre_allocate_buffers: bool = True,
        use_memoryviews: bool = True,
        feature_buffer_size: int = 256,
        history_buffer_size: int = 1000,
        reservoir_sample_size: int = 1000,
        max_feature_latency_us: int = 500,
        max_inference_latency_us: int = 2000,
        circuit_breaker_threshold: float = 0.1,
        circuit_breaker_timeout_ms: int = 5000,
    ) -> None:
        self.enable_zero_copy = enable_zero_copy
        self.pre_allocate_buffers = pre_allocate_buffers
        self.use_memoryviews = use_memoryviews
        self.feature_buffer_size = feature_buffer_size
        self.history_buffer_size = history_buffer_size
        self.reservoir_sample_size = reservoir_sample_size
        self.max_feature_latency_us = max_feature_latency_us
        self.max_inference_latency_us = max_inference_latency_us
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout_ms = circuit_breaker_timeout_ms


class OptimizedMLSignalActorConfig:
    """
    Configuration for optimized ML Signal Actor with advanced performance tuning.

    This configuration provides additional parameters specifically designed for
    high-performance signal generation with <2ms latency requirements.

    Parameters
    ----------
    base_config : MLSignalActorConfig
        Base configuration from MLSignalActorConfig.
    threshold_strategy : ThresholdStrategy, default ThresholdStrategy.REGIME_AWARE
        The threshold calculation strategy.
    enable_hot_reload : bool, default True
        Whether to enable model hot reloading.
    hot_reload_interval : PositiveInt, default 300
        Hot reload check interval in seconds.
    preserve_state_on_reload : bool, default True
        Whether to preserve state during hot reload.
    onnx_config : ONNXOptimizationConfig, optional
        ONNX runtime optimization configuration.
    adaptive_config : AdaptiveThresholdsConfig, optional
        Adaptive thresholds configuration.
    hotpath_config : HotPathOptimizationConfig, optional
        Hot path optimization configuration.
    enable_model_warm_up : bool, default True
        Whether to warm up the model at startup.
    warm_up_iterations : PositiveInt, default 100
        Number of warm-up inference iterations.

    """

    def __init__(
        self,
        base_config: MLSignalActorConfig | None = None,
        threshold_strategy: ThresholdStrategy = ThresholdStrategy.REGIME_AWARE,
        enable_hot_reload: bool = True,
        hot_reload_interval: PositiveInt = 300,
        preserve_state_on_reload: bool = True,
        onnx_config: ONNXOptimizationConfig | None = None,
        adaptive_config: AdaptiveThresholdsConfig | None = None,
        hotpath_config: HotPathOptimizationConfig | None = None,
        enable_model_warm_up: bool = True,
        warm_up_iterations: PositiveInt = 100,
        **kwargs: Any,  # Accept additional args for base_config if provided
    ) -> None:
        # Import here to avoid circular import
        from ml.actors.signal import MLSignalActorConfig
        from ml.actors.signal import SignalStrategy

        # Create base config if not provided
        if base_config is None:
            # Extract base config args from kwargs
            base_args = {}
            for key in [
                "actor_id",
                "model_path",
                "prediction_threshold",
                "max_feature_latency_ms",
                "health_check_interval_seconds",
                "log_predictions",
                "feature_config",
                "signal_strategy",
                "extremes_top_pct",
                "momentum_lookback",
                "ensemble_weights",
                "adaptive_window",
                "adaptive_volatility_factor",
                "min_signal_separation_bars",
                "feature_importance_threshold",
                "enable_regime_detection",
            ]:
                if key in kwargs:
                    base_args[key] = kwargs[key]

            # Set defaults for optimization
            if "signal_strategy" not in base_args:
                base_args["signal_strategy"] = SignalStrategy.ADAPTIVE

            self.base_config = MLSignalActorConfig(**base_args)
        else:
            self.base_config = base_config

        # Optimization-specific config
        self.threshold_strategy = threshold_strategy
        self.enable_hot_reload = enable_hot_reload
        self.hot_reload_interval = hot_reload_interval
        self.preserve_state_on_reload = preserve_state_on_reload
        self.enable_model_warm_up = enable_model_warm_up
        self.warm_up_iterations = warm_up_iterations
        self.onnx_config = onnx_config
        self.adaptive_config = adaptive_config
        self.hotpath_config = hotpath_config

    # Delegate base config properties
    @property
    def signal_strategy(self) -> SignalStrategy:
        # Import the correct SignalStrategy from ml.actors.signal
        # Convert between the two enums if needed
        ml_strategy = self.base_config.signal_strategy
        if hasattr(ml_strategy, "value"):
            return SignalStrategy(ml_strategy.value)
        return SignalStrategy.ADAPTIVE

    @property
    def adaptive_window(self) -> PositiveInt:
        return self.base_config.adaptive_window

    @property
    def min_signal_separation_bars(self) -> PositiveInt:
        return self.base_config.min_signal_separation_bars

    @property
    def enable_regime_detection(self) -> bool:
        return self.base_config.enable_regime_detection

    # Delegate all other base config attributes
    def __getattr__(self, name: str) -> Any:
        # Try to get attribute from base_config if not found in self
        return getattr(self.base_config, name)

    def get_onnx_config(self) -> ONNXOptimizationConfig:
        """
        Get ONNX configuration with defaults.
        """
        return self.onnx_config or ONNXOptimizationConfig()

    def get_adaptive_config(self) -> AdaptiveThresholdsConfig:
        """
        Get adaptive configuration with defaults.
        """
        return self.adaptive_config or AdaptiveThresholdsConfig()

    def get_hotpath_config(self) -> HotPathOptimizationConfig:
        """
        Get hot path configuration with defaults.
        """
        return self.hotpath_config or HotPathOptimizationConfig()

    def get_ensemble_weights(self) -> dict[str, float]:
        """
        Get ensemble weights with defaults.
        """
        if (
            hasattr(self.base_config, "ensemble_weights")
            and self.base_config.ensemble_weights is not None
        ):
            return self.base_config.ensemble_weights
        return {
            "threshold": 0.4,
            "extremes": 0.3,
            "momentum": 0.3,
        }
