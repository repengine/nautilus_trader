"""
Default config constructors.
"""

from __future__ import annotations

from ml.config.actors import MLSignalActorConfig
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig
from ml.config.runtime import OnnxRuntimeConfig


def default_signal_actor_config() -> MLSignalActorConfig:
    return MLSignalActorConfig(
        optimization_config=OptimizationConfig(),
        strategy_config=StrategyConfig(),
        onnx_runtime_config=OnnxRuntimeConfig(),
    )


def default_onnx_runtime_config() -> OnnxRuntimeConfig:
    return OnnxRuntimeConfig()


__all__ = ["default_onnx_runtime_config", "default_signal_actor_config"]
