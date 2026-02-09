"""
Streaming execution backend for canonical feature pipeline specs.

This module provides a lightweight executor that maps PipelineSpec transforms
onto the online FeatureCalculator path. OHLCV and microstructure/trade-flow
transforms are computed via FeatureCalculator, with optional macro/calendar/event
joins when preloaded providers are supplied via PipelineStreamContext.

L2 order-book features remain gated in streaming mode (no live backend) until
the Databento subscription is active again; see ``ml/features/l2_aggregate.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from ml._imports import pl as pl_runtime
from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.config import FeatureConfig
from ml.features.config import derive_ohlcv_feature_config
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.features.pipeline import transform_feature_names
from ml.registry.base import DataRequirements


if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

    from ml.data.providers.calendar import MarketCalendarProvider
    from ml.data.providers.events import EventScheduleProvider
    from ml.features.indicators import IndicatorManager


PL: Any = cast(Any, pl_runtime)


class MacroTransformProtocol(Protocol):
    """
    Minimal protocol for macro transforms used in streaming mode.
    """

    macro_series_ids: list[str]
    include_revisions: bool
    revision_mode: str
    include_composites: bool

    def compute_realtime(
        self,
        bar: Bar | None = None,
        ts_event: int | None = None,
    ) -> dict[str, float]: ...


_CALCULATOR_TRANSFORMS: set[str] = {
    "returns",
    "momentum",
    "volatility",
    "volume_ratio",
    "core_indicators",
    "microstructure",
    "trade_flow",
}

_UNSUPPORTED_REASONS: dict[str, str] = {
    "macro_deltas": "macro deltas require batch history (no streaming backend)",
    "macro_indicators": "requires macro indicators provider in streaming path",
    "static_covariates": "requires instrument metadata in streaming path",
    "instrument_metadata": "requires instrument metadata in streaming path",
    "ewma_beta": "requires multi-instrument streaming inputs",
    "zscore_spread": "requires multi-instrument streaming inputs",
    "keltner": "not implemented in online FeatureCalculator",
    "obv": "not implemented in online FeatureCalculator",
}


@dataclass(slots=True)
class PipelineStreamContext:
    """
    Context for streaming pipeline execution.

    Attributes:
        feature_config: Base FeatureConfig to seed derived OHLCV settings.
        indicator_manager: IndicatorManager tracking hot-path indicator state.
        scaler: Optional fitted scaler for feature normalization.
    """

    feature_config: FeatureConfig
    indicator_manager: IndicatorManager
    scaler: Any | None = None
    macro_transform: MacroTransformProtocol | None = None
    calendar_provider: MarketCalendarProvider | None = None
    calendar_exchange: str = "NYSE"
    event_provider: EventScheduleProvider | None = None
    event_instruments: list[str] | None = None


class PipelineStreamExecutor:
    """
    Execute canonical PipelineSpec transforms for online inference.

    OHLCV/microstructure/trade-flow transforms are supported via FeatureCalculator.
    Calendar/event/macro transforms are supported when preloaded providers or
    transforms are passed via PipelineStreamContext. Other transforms must be gated
    by DataRequirements until live inputs exist.

    L2 depth aggregates are intentionally excluded here and remain gated until a
    dedicated streaming order-book backend is wired and the Databento subscription
    is active again. This keeps the live path allocation-free and avoids silently
    diverging from batch L2 aggregations.
    """

    def __init__(
        self,
        spec: PipelineSpec,
        *,
        allowable: DataRequirements,
        context: PipelineStreamContext,
    ) -> None:
        """
        Initialize the streaming executor.

        Args:
            spec: PipelineSpec defining the transform ordering.
            allowable: DataRequirements gate for transforms.
            context: Streaming execution context (config + indicator state).

        Raises:
            ValueError: If unsupported transforms are present.
        """
        self._spec = spec
        self._allowable = allowable
        self._context = context
        self._macro_transform = context.macro_transform
        self._calendar_provider = context.calendar_provider
        self._event_provider = context.event_provider
        self._validate_supported_transforms([ts.name for ts in spec.transforms])

        self._runner = PipelineRunner(spec, allowable)
        self._feature_names = tuple(self._runner.compute_feature_names())
        self._feature_index = {name: idx for idx, name in enumerate(self._feature_names)}
        self._transform_features = self._build_transform_features(spec.transforms)
        self._config = derive_ohlcv_feature_config(
            context.feature_config,
            spec.transforms,
            allowable=allowable,
        )
        self._calculator = FeatureCalculator(self._config)
        self._calculator_names = self._transform_features_for(_CALCULATOR_TRANSFORMS)
        self._calculator_indices = self._indices_for(self._calculator_names)

        self._macro_names = self._transform_features_for({"macro", "macro_composites"})
        self._macro_indices = self._indices_for(self._macro_names)
        self._calendar_names = self._transform_features_for({"calendar"})
        self._calendar_indices = self._indices_for(self._calendar_names)
        self._event_names = self._transform_features_for({"event_schedule"})
        self._event_indices = self._indices_for(self._event_names)

        self._use_output_buffer = bool(
            self._macro_indices or self._calendar_indices or self._event_indices
        )
        self._output_buffer = (
            np.zeros(len(self._feature_names), dtype=np.float32)
            if self._use_output_buffer
            else None
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """
        Return canonical feature names in pipeline order.
        """
        return self._feature_names

    def execute(
        self,
        current_bar: dict[str, float],
        *,
        timestamp_ns: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Compute streaming features for the current bar.

        Args:
            current_bar: Dict with OHLCV values for the current bar.
            timestamp_ns: Optional nanosecond timestamp. Required for macro,
                calendar, and event schedule transforms.

        Returns:
            Feature array aligned to ``feature_names``.
        """
        features = self._calculator.calculate_features(
            current_bar,
            mode="online",
            indicator_manager=self._context.indicator_manager,
            scaler=self._context.scaler,
        )
        if not self._use_output_buffer:
            return features

        output = self._output_buffer
        assert output is not None
        output.fill(0.0)

        if self._calculator_indices:
            output[self._calculator_indices] = features

        require_timestamp = bool(self._macro_indices or self._calendar_indices or self._event_indices)
        resolved_timestamp_ns = (
            self._require_timestamp(timestamp_ns, include_macro=bool(self._macro_indices))
            if require_timestamp
            else None
        )

        if self._macro_indices and self._macro_transform is not None:
            assert resolved_timestamp_ns is not None
            macro_features = self._macro_transform.compute_realtime(ts_event=resolved_timestamp_ns)
            self._fill_from_mapping(output, self._macro_names, macro_features)

        if self._calendar_indices:
            assert resolved_timestamp_ns is not None
            calendar_features = self._calendar_feature_map(resolved_timestamp_ns)
            self._fill_from_mapping(output, self._calendar_names, calendar_features)

        if self._event_indices:
            assert resolved_timestamp_ns is not None
            event_features = self._event_feature_map(resolved_timestamp_ns)
            self._fill_from_mapping(output, self._event_names, event_features)

        return output

    def _validate_supported_transforms(self, transform_names: Sequence[str]) -> None:
        unsupported: list[str] = []

        if "macro_composites" in transform_names and "macro" not in transform_names:
            unsupported.append("macro_composites (requires macro transform)")

        for name in transform_names:
            if name in _CALCULATOR_TRANSFORMS:
                continue
            if name == "calendar":
                if self._calendar_provider is None:
                    unsupported.append("calendar (requires calendar_provider in context)")
                continue
            if name == "event_schedule":
                if self._event_provider is None:
                    unsupported.append("event_schedule (requires event_provider in context)")
                continue
            if name in {"macro", "macro_composites"}:
                if self._macro_transform is None:
                    unsupported.append(f"{name} (requires macro_transform in context)")
                    continue
                self._validate_macro_transform(name)
                continue

            reason = _UNSUPPORTED_REASONS.get(name, "no streaming backend")
            unsupported.append(f"{name} ({reason})")

        if unsupported:
            msg = (
                "Stream executor does not support transforms: "
                + ", ".join(sorted(unsupported))
                + ". Gate by DataRequirements or remove from spec until live backends exist."
            )
            raise ValueError(msg)

    def _validate_macro_transform(self, name: str) -> None:
        transform = self._macro_transform
        if transform is None:
            return
        macro_spec = next(
            (ts for ts in self._spec.transforms if ts.name == "macro"),
            None,
        )
        if macro_spec is None:
            return
        params = macro_spec.params
        expected_series = tuple(params.get("series_ids", []))
        if expected_series:
            actual_series = tuple(transform.macro_series_ids)
            if expected_series != actual_series:
                msg = (
                    "macro_transform series_ids mismatch: "
                    f"spec={expected_series} transform={actual_series}"
                )
                raise ValueError(msg)
        expected_revisions = bool(params.get("include_revisions", False))
        if expected_revisions != transform.include_revisions:
            msg = (
                "macro_transform include_revisions mismatch: "
                f"spec={expected_revisions} transform={transform.include_revisions}"
            )
            raise ValueError(msg)
        expected_mode = str(params.get("revision_mode", "core"))
        if expected_mode != str(transform.revision_mode):
            msg = (
                "macro_transform revision_mode mismatch: "
                f"spec={expected_mode} transform={transform.revision_mode}"
            )
            raise ValueError(msg)
        if name == "macro_composites" and not transform.include_composites:
            raise ValueError("macro_composites requires macro_transform.include_composites")

    def _build_transform_features(
        self,
        transforms: Sequence[TransformSpec],
    ) -> dict[str, tuple[str, ...]]:
        features: dict[str, tuple[str, ...]] = {}
        for ts in transforms:
            features[ts.name] = tuple(transform_feature_names(ts))
        return features

    def _transform_features_for(self, names: set[str]) -> tuple[str, ...]:
        ordered: list[str] = []
        for ts in self._spec.transforms:
            if ts.name in names:
                ordered.extend(self._transform_features.get(ts.name, ()))
        return tuple(ordered)

    def _indices_for(self, names: tuple[str, ...]) -> list[int]:
        return [self._feature_index[name] for name in names if name in self._feature_index]

    def _fill_from_mapping(
        self,
        output: npt.NDArray[np.float32],
        names: tuple[str, ...],
        values: dict[str, Any],
    ) -> None:
        for name in names:
            idx = self._feature_index.get(name)
            if idx is None:
                continue
            value = values.get(name)
            if value is None:
                continue
            try:
                output[idx] = float(value)
            except Exception:
                output[idx] = 0.0

    def _require_timestamp(
        self,
        timestamp_ns: int | None,
        *,
        include_macro: bool = False,
    ) -> int:
        if timestamp_ns is None:
            msg = (
                "timestamp_ns is required for macro/calendar/event transforms"
                if include_macro
                else "timestamp_ns is required for calendar/event transforms"
            )
            raise ValueError(msg)
        return self._coerce_timestamp_ns(timestamp_ns)

    @staticmethod
    def _coerce_timestamp_ns(value: float | datetime) -> int:
        if isinstance(value, datetime):
            return int(value.timestamp() * 1e9)
        return int(value)

    def _calendar_feature_map(self, timestamp_ns: int) -> dict[str, Any]:
        provider = self._calendar_provider
        if provider is None:
            return {}
        if PL is None:
            raise RuntimeError("Polars is required for calendar features")
        series = PL.Series([timestamp_ns])
        calendar_df = provider.compute_features(series, exchange=self._context.calendar_exchange)
        if calendar_df.is_empty():
            return {}
        return calendar_df.row(0, named=True)

    def _event_feature_map(self, timestamp_ns: int) -> dict[str, Any]:
        provider = self._event_provider
        if provider is None:
            return {}
        if PL is None:
            raise RuntimeError("Polars is required for event schedule features")
        series = PL.Series([timestamp_ns])
        instruments = self._context.event_instruments
        event_df = provider.compute_features(series, instruments=instruments or [])
        if event_df.is_empty():
            return {}
        return event_df.row(0, named=True)
