"""
Runtime configuration and helpers (ONNX Runtime).
"""

from __future__ import annotations

from typing import Literal

import msgspec

from ml._imports import ort
from ml.config.constants import Providers
from nautilus_trader.common.config import NautilusConfig


GraphOptLevel = Literal["disable", "basic", "extended", "all"]
ExecutionMode = Literal["sequential", "parallel"]


class OnnxRuntimeConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    ONNX Runtime configuration used by inference loaders.
    """

    graph_optimization_level: GraphOptLevel = "all"
    execution_mode: ExecutionMode = "sequential"
    providers: list[str] = msgspec.field(default_factory=lambda: [Providers.CPU])
    intra_threads: int | None = None
    inter_threads: int | None = None


def to_session_options(cfg: OnnxRuntimeConfig) -> tuple[ort.SessionOptions, list[str]]:
    """
    Convert OnnxRuntimeConfig into ORT SessionOptions and provider list.
    """
    session_options = ort.SessionOptions()
    level_map = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    exec_map = {
        "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
        "parallel": ort.ExecutionMode.ORT_PARALLEL,
    }
    session_options.graph_optimization_level = level_map[cfg.graph_optimization_level]
    session_options.execution_mode = exec_map[cfg.execution_mode]
    if cfg.intra_threads is not None:
        session_options.intra_op_num_threads = int(cfg.intra_threads)
    if cfg.inter_threads is not None:
        session_options.inter_op_num_threads = int(cfg.inter_threads)
    providers = cfg.providers or [Providers.CPU]
    return session_options, providers


__all__ = ["ExecutionMode", "GraphOptLevel", "OnnxRuntimeConfig", "to_session_options"]
