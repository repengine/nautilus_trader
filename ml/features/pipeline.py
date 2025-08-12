#!/usr/bin/env python3

"""
Declarative feature pipeline scaffolding.

Provides a minimal transform catalog, pipeline spec, and runner utilities for schema
computation, signature hashing, and capability gating. The execution of batch/online
math remains sourced from FeatureEngineer for now.

"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Protocol

from ml.registry.base import DataRequirements


class FeatureTransform(Protocol):
    """
    Protocol for feature transform plugins.
    """

    name: str

    def feature_names(self, params: Mapping[str, Any]) -> list[str]: ...

    def requires(self) -> DataRequirements:
        """
        Required data level for this transform (used for gating).
        """
        ...


class _ReturnsTransform:
    name = "returns"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        periods: Iterable[int] = params.get("periods", [1, 5, 10, 20])  # type: ignore[assignment]
        return [f"return_{int(p)}" for p in periods]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _MomentumTransform:
    name = "momentum"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        periods: Iterable[int] = params.get("periods", [5, 10, 20])  # type: ignore[assignment]
        return [f"momentum_{int(p)}" for p in periods]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _VolatilityTransform:
    name = "volatility"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        # Fixed names used in engineering.py
        return [
            "volatility_5",
            "volatility_20",
        ]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _VolumeRatioTransform:
    name = "volume_ratio"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        periods: Iterable[int] = params.get("periods", [5, 10, 20])  # type: ignore[assignment]
        return [f"volume_ratio_{int(p)}" for p in periods]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


class _CoreIndicatorsTransform:
    name = "core_indicators"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        # Mirrors engineering.py core indicator outputs
        return [
            "rsi",
            "rsi_overbought",
            "rsi_oversold",
            "bb_width",
            "bb_position",
            "atr_normalized",
            "ema_fast_dist",
            "ema_slow_dist",
            "ema_cross",
            "macd_line",
            "macd_signal",
            "macd_diff",
            "price_position_20",
            "hl_spread",
        ]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_ONLY


_CATALOG: dict[str, FeatureTransform] = {
    _ReturnsTransform().name: _ReturnsTransform(),
    _MomentumTransform().name: _MomentumTransform(),
    _VolatilityTransform().name: _VolatilityTransform(),
    _VolumeRatioTransform().name: _VolumeRatioTransform(),
    _CoreIndicatorsTransform().name: _CoreIndicatorsTransform(),
}


def register_transform(transform: FeatureTransform) -> None:
    _CATALOG[transform.name] = transform


# Optional/richer transforms for teacher/offline pipelines
class _KeltnerTransform:
    name = "keltner"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return ["keltner_width", "keltner_position"]

    def requires(self) -> DataRequirements:
        # Requires at least high/low/close + ATR/EMA; treat as L1_L2 to gate in student
        return DataRequirements.L1_L2


class _OBVTransform:
    name = "obv"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return ["obv_norm"]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_L2


class _MicrostructureTransform:
    name = "microstructure"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return [
            "spread_mean",
            "spread_std",
            "spread_relative",
            "size_imbalance_mean",
            "size_imbalance_std",
            "mid_return_std",
            "mid_return_autocorr",
        ]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_L2


class _TradeFlowTransform:
    name = "trade_flow"

    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return [
            "trade_flow_imbalance",
            "vwap",
            "trade_intensity",
            "avg_price_impact",
        ]

    def requires(self) -> DataRequirements:
        return DataRequirements.L1_L2


# Register optional transforms
register_transform(_KeltnerTransform())
register_transform(_OBVTransform())
register_transform(_MicrostructureTransform())
register_transform(_TradeFlowTransform())


@dataclass(slots=True)
class TransformSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineSpec:
    transforms: list[TransformSpec]


def _hash_pipeline(transforms: list[TransformSpec]) -> str:
    h = hashlib.sha256()
    for spec in transforms:
        h.update(spec.name.encode("utf-8"))
        h.update(b"\0")
        # Stable param hashing (sorted by key)
        for k in sorted(spec.params.keys()):
            v = spec.params[k]
            h.update(str(k).encode("utf-8"))
            h.update(b"=")
            h.update(str(v).encode("utf-8"))
            h.update(b";")
        h.update(b"\n")
    return h.hexdigest()


class PipelineRunner:
    """
    Compile-time utilities for feature pipelines.
    """

    def __init__(self, spec: PipelineSpec, allowable: DataRequirements) -> None:
        self._spec = spec
        self._allowable = allowable
        self._transforms: list[FeatureTransform] = []
        self._compile()

    def _compile(self) -> None:
        self._transforms.clear()
        for ts in self._spec.transforms:
            if ts.name not in _CATALOG:
                msg = f"Unknown transform: {ts.name}"
                raise ValueError(msg)
            t = _CATALOG[ts.name]
            # Gate by data requirements (simple lattice: L1_ONLY < L1_L2 < L1_L2_L3)
            req = t.requires()
            if self._allowable == DataRequirements.L1_ONLY and req != DataRequirements.L1_ONLY:
                msg = f"Transform {ts.name} requires {req.value}, not allowed for {self._allowable.value}"
                raise ValueError(msg)
            self._transforms.append(t)

    def compute_feature_names(self) -> list[str]:
        names: list[str] = []
        for ts, t in zip(self._spec.transforms, self._transforms):
            names.extend(t.feature_names(ts.params))
        return names

    def compute_signature(self) -> str:
        return _hash_pipeline(self._spec.transforms)
