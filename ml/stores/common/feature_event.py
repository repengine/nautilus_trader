#!/usr/bin/env python3

"""
Feature event component for FeatureStore.

Extracted from FeatureStore (Phase 3.7.5). Provides event emission and
DataRegistry integration for feature computation operations.

ALL methods are COLD path (async operations acceptable, non-blocking).

This component extracts:
- _emit_historical_event() - Emit FEATURE_COMPUTED event for historical
- _record_observability_stage_boundary() - Observability data recording
- Event emission for realtime computation (integrated in compute_realtime)
- DataRegistry integration

"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ml._imports import HAS_PROMETHEUS
from ml.common.error_handlers import with_fallback
from ml.common.event_emitter import emit_dataset_event_and_watermark
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage


if TYPE_CHECKING:
    from ml.common.observability_utils import ObservabilityLike
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)


# =========================================================================
# No-op Metrics for when Prometheus is unavailable
# =========================================================================


class _NoOpMetric:
    """
    No-op metric for when Prometheus is unavailable.
    """

    def labels(self, **_: Any) -> _NoOpMetric:
        """
        No-op labels method.
        """
        return self

    def inc(self, *_: object, **__: object) -> None:
        """
        No-op inc method.
        """
        return None


# Declare metric variables once
feature_event_emission_counter: Any = _NoOpMetric()
feature_event_emission_errors: Any = _NoOpMetric()

try:
    from ml.common.metrics_bootstrap import get_counter

    feature_event_emission_counter = get_counter(
        "ml_feature_event_emissions_total",
        "Total number of feature events emitted by FeatureStore",
        labelnames=["event_type", "status"],
    )
    feature_event_emission_errors = get_counter(
        "ml_feature_event_emission_errors_total",
        "Total number of feature event emission errors",
        labelnames=["event_type", "error_type"],
    )
except Exception:
    logger.debug("Metrics bootstrap failed; using no-op counters", exc_info=True)


# =========================================================================
# Protocols
# =========================================================================


@runtime_checkable
class FeatureEventProtocol(Protocol):
    """
    Protocol for feature event emission operations.

    Defines the interface for emitting feature-related events and recording
    observability data. All methods are NON-BLOCKING.

    """

    def emit_historical_event(
        self,
        instrument_id: str,
        timestamps: npt.NDArray[np.int64],
        row_count: int,
    ) -> None:
        """
        Emit FEATURE_COMPUTED event for historical computation.

        Non-blocking operation - failures are logged but don't affect feature computation.

        Args:
            instrument_id: Instrument identifier
            timestamps: Array of timestamps for the computed features
            row_count: Number of rows computed

        """
        ...

    def emit_realtime_event(
        self,
        bar: Any,
        feature_set_id: str,
    ) -> None:
        """
        Emit FEATURE_COMPUTED event for realtime computation.

        Non-blocking operation - failures are logged but don't affect feature computation.

        Args:
            bar: Bar object with ts_event and instrument_id attributes
            feature_set_id: Feature set identifier for component metadata

        """
        ...

    def record_observability_stage_boundary(
        self,
        *,
        stage: str,
        instrument_id: str,
        ts_stage_start: int,
        ts_stage_end: int,
        row_count: int = 1,
    ) -> None:
        """
        Record observability data via centralized helper (cold path only).

        Args:
            stage: Pipeline stage name
            instrument_id: Instrument identifier
            ts_stage_start: Stage start timestamp (ns)
            ts_stage_end: Stage end timestamp (ns)
            row_count: Optional row count for labels

        """
        ...


# =========================================================================
# Configuration
# =========================================================================


@dataclass(frozen=True)
class FeatureEventConfig:
    """
    Configuration for feature event emission.

    Attributes:
        component_name: Component identifier for correlation and metrics
        dataset_id: Default dataset identifier for events
        enable_observability: Whether to enable observability recording

    Example:
        >>> config = FeatureEventConfig(
        ...     component_name="feature_store",
        ...     dataset_id="features",
        ...     enable_observability=True,
        ... )

    """

    component_name: str = "feature_store"
    dataset_id: str = "features"
    enable_observability: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.component_name:
            raise ValueError("component_name cannot be empty")
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")


# =========================================================================
# FeatureEventComponent
# =========================================================================


class FeatureEventComponent:
    """
    Event emission and DataRegistry integration for FeatureStore.

    Extracted from FeatureStore (Phase 3.7.5).
    All methods are COLD path (async operations acceptable, non-blocking).

    Provides:
    - Historical feature computation event emission
    - Realtime feature computation event emission
    - Observability stage boundary recording
    - DataRegistry integration

    CRITICAL: All event emission must be NON-BLOCKING. Wrap in try/except
    to ensure failures don't crash feature operations.

    Example
    -------
    >>> from ml.stores.common.feature_event import FeatureEventComponent
    >>> component = FeatureEventComponent(
    ...     config=FeatureEventConfig(),
    ...     get_registry=lambda: registry,
    ...     get_feature_set_id=lambda: "fs_abc123",
    ... )
    >>> component.emit_historical_event(
    ...     instrument_id="SPY.DATABENTO",
    ...     timestamps=np.array([1700000000000000000], dtype=np.int64),
    ...     row_count=100,
    ... )

    """

    def __init__(
        self,
        config: FeatureEventConfig | None = None,
        get_registry: Any | None = None,
        get_feature_set_id: Any | None = None,
        observability_service: ObservabilityLike | None = None,
    ) -> None:
        """
        Initialize feature event component.

        Args:
            config: Configuration for event emission
            get_registry: Callable that returns the DataRegistry or None
            get_feature_set_id: Callable that returns the feature set ID
            observability_service: Optional observability service for stage recording

        """
        self._config = config or FeatureEventConfig()
        self._get_registry = get_registry
        self._get_feature_set_id = get_feature_set_id
        self._observability_service = observability_service

    # =========================================================================
    # Public API - All COLD PATH (non-blocking)
    # =========================================================================

    @with_fallback(
        fallback_value=None,
        log_level="warning",
        operation_name="emit feature computation event",
    )
    def emit_historical_event(
        self,
        instrument_id: str,
        timestamps: npt.NDArray[np.int64],
        row_count: int,
    ) -> None:
        """
        Emit FEATURE_COMPUTED event for historical computation.

        EXTRACTED FROM: ml/stores/feature_store.py:954-1014
        COLD PATH: Event emission is async, non-blocking

        Non-blocking operation - failures are logged but don't affect feature computation.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier.
        timestamps : npt.NDArray[np.int64]
            Array of timestamps for the computed features.
        row_count : int
            Number of rows computed.

        Examples
        --------
        >>> component.emit_historical_event(
        ...     instrument_id="SPY.DATABENTO",
        ...     timestamps=np.array([1700000000000000000], dtype=np.int64),
        ...     row_count=100,
        ... )

        """
        registry = self._get_data_registry()
        if registry is None:
            logger.debug(
                "No registry available for historical event emission: instrument=%s",
                instrument_id,
            )
            return

        # Generate unique run ID for this computation
        run_id = f"feature_historical_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        # Use canonical dataset id and feature set id
        feature_set_id = self._resolve_feature_set_id()
        dataset_id = self._config.dataset_id

        # Get the time range from timestamps
        ts_min = int(timestamps[0]) if len(timestamps) > 0 else 0
        ts_max = int(timestamps[-1]) if len(timestamps) > 0 else 0

        try:
            # Emit via shared helper (event + watermark + metrics)
            emit_dataset_event_and_watermark(
                registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=Stage.FEATURE_COMPUTED,
                source=Source.HISTORICAL,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=row_count,
                status=EventStatus.SUCCESS,
                dataset_type="features",
                component=feature_set_id,
            )

            # Record success metric
            if HAS_PROMETHEUS:
                feature_event_emission_counter.labels(
                    event_type="emit_historical_event",
                    status="success",
                ).inc()

            logger.debug(
                "Emitted FEATURE_COMPUTED event for historical computation: "
                "dataset=%s, instrument=%s, count=%d, ts_range=[%d, %d]",
                dataset_id,
                instrument_id,
                row_count,
                ts_min,
                ts_max,
            )

        except Exception as exc:
            # Non-blocking: log and continue
            logger.warning(
                "Failed to emit historical feature event for %s: %s",
                instrument_id,
                exc,
                exc_info=True,
            )
            if HAS_PROMETHEUS:
                feature_event_emission_errors.labels(
                    event_type="emit_historical_event",
                    error_type="emission_error",
                ).inc()

    def emit_realtime_event(
        self,
        bar: Any,
        feature_set_id: str,
    ) -> None:
        """
        Emit FEATURE_COMPUTED event for realtime computation.

        EXTRACTED FROM: ml/stores/feature_store.py:683-729
        COLD PATH: Event emission is async, non-blocking

        Non-blocking operation - failures are logged but don't affect feature computation.

        Parameters
        ----------
        bar : Any
            Bar object with ts_event and instrument_id attributes.
        feature_set_id : str
            Feature set identifier for component metadata.

        Examples
        --------
        >>> component.emit_realtime_event(
        ...     bar=bar_object,
        ...     feature_set_id="fs_abc123",
        ... )

        """
        try:
            registry = self._get_data_registry()
            if registry is None:
                logger.debug(
                    "No registry available for realtime event emission",
                )
                return

            # Generate unique run ID for this computation
            run_id = f"feature_realtime_{uuid.uuid4().hex[:8]}_{int(time.time())}"

            # Extract instrument_id from bar
            instrument_id_str = self._extract_instrument_id(bar)
            dataset_id = self._config.dataset_id

            # Get ts_event from bar
            ts_event = int(getattr(bar, "ts_event", 0))

            # Emit via shared helper (event + watermark + metrics)
            emit_dataset_event_and_watermark(
                registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id_str,
                stage=Stage.FEATURE_COMPUTED,
                source=Source.LIVE,
                run_id=run_id,
                ts_min=ts_event,
                ts_max=ts_event,
                count=1,
                status=EventStatus.SUCCESS,
                dataset_type="features",
                component=feature_set_id,
            )

            # Record success metric
            if HAS_PROMETHEUS:
                feature_event_emission_counter.labels(
                    event_type="emit_realtime_event",
                    status="success",
                ).inc()

            logger.debug(
                "Emitted FEATURE_COMPUTED event for realtime computation: "
                "dataset=%s, instrument=%s, ts_event=%d",
                dataset_id,
                instrument_id_str,
                ts_event,
            )

        except Exception as exc:
            # Non-blocking: log but don't fail the feature computation
            logger.warning(
                "Failed to emit realtime feature event: %s",
                exc,
                exc_info=True,
            )
            if HAS_PROMETHEUS:
                feature_event_emission_errors.labels(
                    event_type="emit_realtime_event",
                    error_type="emission_error",
                ).inc()

    def record_observability_stage_boundary(
        self,
        *,
        stage: str,
        instrument_id: str,
        ts_stage_start: int,
        ts_stage_end: int,
        row_count: int = 1,
    ) -> None:
        """
        Record observability data via centralized helper (cold path only).

        EXTRACTED FROM: ml/stores/feature_store.py:1270-1293
        COLD PATH: Observability recording is non-blocking

        Parameters
        ----------
        stage : str
            Pipeline stage name.
        instrument_id : str
            Instrument identifier.
        ts_stage_start : int
            Stage start timestamp (ns).
        ts_stage_end : int
            Stage end timestamp (ns).
        row_count : int, default 1
            Optional row count for labels.

        Examples
        --------
        >>> component.record_observability_stage_boundary(
        ...     stage="feature_computation",
        ...     instrument_id="SPY.DATABENTO",
        ...     ts_stage_start=1700000000000000000,
        ...     ts_stage_end=1700000001000000000,
        ...     row_count=100,
        ... )

        """
        try:
            from ml.common.observability_utils import record_stage_boundary

            obs_service = self._observability_service
            record_stage_boundary(
                obs_service,
                component=self._config.component_name,
                instrument_id=instrument_id,
                stage=stage,
                ts_stage_start=ts_stage_start,
                ts_stage_end=ts_stage_end,
                row_count=row_count,
            )

            logger.debug(
                "Recorded observability stage boundary: stage=%s, instrument=%s, "
                "duration_ns=%d, row_count=%d",
                stage,
                instrument_id,
                ts_stage_end - ts_stage_start,
                row_count,
            )

        except Exception as exc:
            # Non-blocking: log but don't fail the operation
            logger.debug(
                "Failed to record observability stage boundary: %s",
                exc,
                exc_info=True,
            )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _get_data_registry(self) -> RegistryProtocol | None:
        """
        Get the DataRegistry via the provided callable.

        Returns
        -------
        RegistryProtocol | None
            The registry instance or None if not available.

        """
        if self._get_registry is None:
            return None

        try:
            result = self._get_registry()
            return result  # type: ignore[no-any-return]
        except Exception as exc:
            logger.debug(
                "Failed to get data registry: %s",
                exc,
                exc_info=True,
            )
            return None

    def _resolve_feature_set_id(self) -> str:
        """
        Resolve the feature set ID via the provided callable.

        Returns
        -------
        str
            The feature set ID or "unknown" if not available.

        """
        if self._get_feature_set_id is None:
            return "unknown"

        try:
            result = self._get_feature_set_id()
            if isinstance(result, str):
                return result
            return "unknown"
        except Exception as exc:
            logger.debug(
                "Failed to get feature set ID: %s",
                exc,
                exc_info=True,
            )
            return "unknown"

    def _extract_instrument_id(self, bar: Any) -> str:
        """
        Extract instrument ID from a bar object.

        Parameters
        ----------
        bar : Any
            Bar object that may have instrument_id directly or via bar_type.

        Returns
        -------
        str
            The instrument ID string or "unknown".

        """
        try:
            # Try bar_type.instrument_id first (Nautilus pattern)
            if hasattr(bar, "bar_type") and hasattr(bar.bar_type, "instrument_id"):
                return str(bar.bar_type.instrument_id)
            # Fallback to direct instrument_id
            if hasattr(bar, "instrument_id"):
                return str(bar.instrument_id)
            return "unknown"
        except Exception:
            return "unknown"

    # =========================================================================
    # Configuration and Setup
    # =========================================================================

    def set_observability_service(
        self,
        service: ObservabilityLike | None,
    ) -> None:
        """
        Set the observability service for stage boundary recording.

        Parameters
        ----------
        service : ObservabilityLike | None
            The observability service instance.

        """
        self._observability_service = service

    def set_registry_getter(
        self,
        getter: Any,
    ) -> None:
        """
        Set the callable that returns the DataRegistry.

        Parameters
        ----------
        getter : Callable
            Function that returns the registry or None.

        """
        self._get_registry = getter

    def set_feature_set_id_getter(
        self,
        getter: Any,
    ) -> None:
        """
        Set the callable that returns the feature set ID.

        Parameters
        ----------
        getter : Callable
            Function that returns the feature set ID string.

        """
        self._get_feature_set_id = getter


__all__ = [
    "FeatureEventComponent",
    "FeatureEventConfig",
    "FeatureEventProtocol",
]
