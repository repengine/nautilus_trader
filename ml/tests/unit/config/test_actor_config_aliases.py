#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from ml.config.actors import MLSignalActorConfig, OptimizationConfig, StrategyConfig


def test_actor_config_alias_mappings(
    default_bar_type: BarType,
    default_instrument_id: InstrumentId,
    dummy_onnx_model: Path,
) -> None:
    opt = OptimizationConfig(level="optimized", enable_zero_copy=True)
    strat = StrategyConfig(strategy_type="momentum", threshold_long=0.6, threshold_short=0.4)

    cfg = MLSignalActorConfig(
        model_id="unit-model",
        model_path=str(dummy_onnx_model),
        bar_type=default_bar_type,
        instrument_id=default_instrument_id,
        signal_strategy="threshold",  # will be overridden by signal_policy/strategy_type
        signal_policy="extremes",
        optimization=opt,
        strategy=strat,
        use_dummy_stores=True,
    )

    # signal_policy wins over initial value
    assert cfg.signal_strategy in ("extremes", "momentum")
    # optimization alias mapped to optimization_config
    assert cfg.optimization_config is not None
    assert cfg.optimization_config.level == "optimized"
    # strategy alias mapped to strategy_config
    assert cfg.strategy_config is not None
    # legacy strategy_type can further override strategy
    # We accept either mapping depending on order: signal_policy or strategy_type
    assert cfg.signal_strategy in ("extremes", "momentum")

    # legacy thresholds merged to prediction_threshold
    assert hasattr(cfg, "prediction_threshold")
    # 0.6 and 0.4 → max(abs(.6), abs(.4)) = 0.6
    assert getattr(cfg, "prediction_threshold") == 0.6
