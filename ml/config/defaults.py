"""
Default config constructors.
"""

from __future__ import annotations

from ml.config.actors import MLSignalActorConfig
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig
from ml.config.runtime import OnnxRuntimeConfig


def default_signal_actor_config() -> MLSignalActorConfig:
    """
    Construct a default MLSignalActorConfig with sane sub-configs.
    """
    return MLSignalActorConfig(  # type: ignore[call-arg]
        optimization_config=OptimizationConfig(),
        strategy_config=StrategyConfig(),
        onnx_runtime_config=OnnxRuntimeConfig(),
    )


def default_onnx_runtime_config() -> OnnxRuntimeConfig:
    """
    Return default ONNX Runtime settings for inference sessions.
    """
    return OnnxRuntimeConfig()


__all__ = ["default_onnx_runtime_config", "default_signal_actor_config"]
