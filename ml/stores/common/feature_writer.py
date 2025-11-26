#!/usr/bin/env python3

"""
Feature writer component for FeatureStore.

Extracted from FeatureStore (Phase 3.7.1). Provides write operations for feature data
with circuit breaker integration, message bus publishing, and upsert semantics.

All write operations are COLD path (async operations acceptable), except for the
single-row write path which is designed to be fast but respects circuit breaker gating.

"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ml.common.message_topics import build_topic_for_stage
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.events import EventStatus
from ml.config.events import Stage


if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.engine import Engine

    from ml.stores.protocols import CircuitBreakerProtocol


logger = logging.getLogger(__name__)


# =========================================================================
# Protocols
# =========================================================================


@runtime_checkable
class MessagePublisherProtocol(Protocol):
    """
    Protocol for message bus publishers.

    Stores use this protocol to publish events to the message bus without
    importing actor or bus modules. Keeps coupling minimal.

    """

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """
        Publish a payload to the message bus.

        Args:
            topic: Topic string for routing
            payload: Dictionary payload to publish

        """
        ...


@runtime_checkable
class FeatureWriterProtocol(Protocol):
    """
    Protocol for feature writing operations.

    Defines the interface for writing computed features to storage with
    circuit breaker integration and message bus publishing.

    """

    def write_features(
        self,
        feature_set_id: str | None = None,
        instrument_id: str | None = None,
        features: Mapping[str, float] | None = None,
        ts_event: int | None = None,
        ts_init: int | None = None,
        data: Any | None = None,
        *,
        publish_bus: bool = True,
    ) -> None:
        """
        Write computed features to storage.

        Supports both explicit-args signature and a backwards-compatible
        form where callers pass a FeatureData or list[FeatureData].

        Args:
            feature_set_id: Feature set identifier (explicit mode)
            instrument_id: Instrument identifier (explicit mode)
            features: Feature name to value mapping (explicit mode)
            ts_event: Event timestamp in nanoseconds (explicit mode)
            ts_init: Initialization timestamp in nanoseconds (explicit mode)
            data: Backwards-compat: a FeatureData or list[FeatureData]
            publish_bus: When True and publishing enabled, publish to message bus

        """
        ...

    def write_batch(self, data: list[object]) -> None:
        """
        Write a batch of FeatureData rows.

        Args:
            data: List of FeatureData-like objects to write

        """
        ...

    def store_features(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible alias for write_features.

        """
        ...


# =========================================================================
# Configuration
# =========================================================================


@dataclass(frozen=True)
class FeatureWriterConfig:
    """
    Configuration for FeatureWriterComponent.

    Attributes
    ----------
    enable_publishing : bool
        Whether to enable message bus publishing (default: False)
    publish_mode : Literal["batch", "row", "both"]
        When to publish events: batch (after batch), row (per row), or both
    topic_scheme : str
        Topic scheme for build_topic_for_stage: "domain_op" or "stage_first"
    topic_prefix : str
        Topic prefix for stage_first scheme

    """

    enable_publishing: bool = False
    publish_mode: Literal["batch", "row", "both"] = "batch"
    topic_scheme: str = "domain_op"
    topic_prefix: str = "events.ml"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.publish_mode not in ("batch", "row", "both"):
            raise ValueError(
                f"Invalid publish_mode '{self.publish_mode}', "
                "must be one of: 'batch', 'row', 'both'"
            )


# =========================================================================
# Component Implementation
# =========================================================================


@dataclass
class FeatureWriterComponent:
    """
    Feature writing operations for FeatureStore.

    Extracted from FeatureStore (Phase 3.7.1).

    Provides:
    - write_features() - Write computed features (explicit args or batch)
    - write_batch() - Batch write API with buffer management
    - store_features() - Backward-compatible alias
    - _execute_write() - Single-row upsert with circuit breaker

    Example
    -------
    >>> from ml.stores.common.feature_writer import FeatureWriterComponent
    >>> writer = FeatureWriterComponent(
    ...     engine=engine,
    ...     table=feature_values_table,
    ...     get_feature_set_id=lambda: "fs_001",
    ... )
    >>> writer.write_features(
    ...     feature_set_id="fs_001",
    ...     instrument_id="SPY.DATABENTO",
    ...     features={"close_return": 0.01},
    ...     ts_event=1700000000000000000,
    ... )

    """

    engine: Engine
    table: Table
    get_feature_set_id: Callable[[], str]
    circuit_breaker: CircuitBreakerProtocol | None = None
    publisher: MessagePublisherProtocol | None = None
    config: FeatureWriterConfig = field(default_factory=FeatureWriterConfig)
    _write_buffer: list[Any] = field(default_factory=list)
    _observability_service: Any = field(default=None)

    def __post_init__(self) -> None:
        """Initialize internal state."""
        # Alias for backward compatibility (tests expect _buffer)
        self._buffer: list[Any] = self._write_buffer

    def write_features(
        self,
        feature_set_id: str | None = None,
        instrument_id: str | None = None,
        features: Mapping[str, float] | None = None,
        ts_event: int | None = None,
        ts_init: int | None = None,
        data: Any | None = None,
        *,
        publish_bus: bool = True,
    ) -> None:
        """
        Write computed features to storage.

        Supports both the explicit-args signature and a backwards-compatible
        form where callers pass a FeatureData or list[FeatureData]. This helps
        legacy tests which call `write_features([FeatureData])`.

        Args:
            feature_set_id: Feature set identifier (explicit mode)
            instrument_id: Instrument identifier (explicit mode)
            features: Feature name to value mapping (explicit mode)
            ts_event: Event timestamp in nanoseconds (explicit mode)
            ts_init: Initialization timestamp in nanoseconds (explicit mode)
            data: Backwards-compat: a FeatureData or list[FeatureData]
            publish_bus: When True and publishing enabled, publish a summary
                payload to the configured message bus. Set to False to suppress
                publishing when orchestrated by a higher-level facade.

        Raises
        ------
        TypeError
            When explicit mode is used but required arguments are missing,
            or when data type is unsupported.

        Example
        -------
        >>> # Explicit args mode
        >>> writer.write_features(
        ...     feature_set_id="fs_001",
        ...     instrument_id="SPY.DATABENTO",
        ...     features={"close_return": 0.01, "volume_ratio": 1.5},
        ...     ts_event=1700000000000000000,
        ... )
        >>> # Batch mode (backwards compatible)
        >>> writer.write_features([feature_data_1, feature_data_2])

        """
        # Backwards compatibility: support write_features([FeatureData]) / (batch)
        batch_data: list[Any] | None = None
        if data is None and feature_set_id is not None and isinstance(feature_set_id, list):
            # Called as write_features([FeatureData])
            batch_data = feature_set_id
            feature_set_id = None
        elif data is not None:
            if isinstance(data, list):
                batch_data = data
            elif hasattr(data, "feature_values") and hasattr(data, "feature_set_id"):
                batch_data = [data]
            else:
                msg = "Unsupported data type for write_features"
                raise TypeError(msg)

        if batch_data is not None:
            self._write_batch_internal(batch_data, publish_bus=publish_bus)
            return

        # Explicit-args mode
        if (
            feature_set_id is None
            or instrument_id is None
            or features is None
            or ts_event is None
        ):
            raise TypeError(
                "write_features requires explicit arguments or a FeatureData batch",
            )

        ts_init_val = int(ts_init) if ts_init is not None else int(ts_event)

        # Normalize features mapping defensively
        features_payload: dict[str, float] = {
            str(k): float(v) for k, v in dict(features or {}).items()
        }

        # Insert with ON CONFLICT for idempotency
        row = {
            "feature_set_id": feature_set_id,
            "instrument_id": instrument_id,
            "ts_event": int(ts_event),
            "ts_init": ts_init_val,
            "values": features_payload,
            "is_live": False,
            "source": "computed",
        }
        self._execute_write(row)

        # Optional publish single-row event
        if (
            self.config.enable_publishing
            and self.publisher is not None
            and instrument_id is not None
            and ts_event is not None
            and self.config.publish_mode in ("batch", "both")
            and publish_bus
        ):
            self._publish_single_row_event(instrument_id, int(ts_event))

    def _write_batch_internal(
        self,
        batch: list[Any],
        *,
        publish_bus: bool = True,
    ) -> None:
        """
        Internal method to process a batch of FeatureData items.

        Args:
            batch: List of FeatureData-like objects
            publish_bus: Whether to publish batch event to message bus

        """
        # Perform upserts per item
        for item in batch:
            fs_id = getattr(item, "feature_set_id", None)
            inst = getattr(item, "instrument_id", None)
            # Use safe accessor to avoid collisions with base class methods
            try:
                vals: dict[str, float] = item.feature_values
            except Exception:
                vals = {}
            tse = int(getattr(item, "ts_event", 0))
            tsi = int(getattr(item, "ts_init", tse))

            row = {
                "feature_set_id": fs_id,
                "instrument_id": inst,
                "ts_event": tse,
                "ts_init": tsi,
                "values": vals,
                "is_live": False,
                "source": "computed",
            }
            self._execute_write(row)

        # Optional publish per-batch summary
        if (
            self.config.enable_publishing
            and self.publisher is not None
            and batch
            and self.config.publish_mode in ("batch", "both")
            and publish_bus
        ):
            self._publish_batch_summary_event(batch)

    def _publish_batch_summary_event(self, batch: list[Any]) -> None:
        """
        Publish a batch summary event to the message bus.

        Args:
            batch: List of FeatureData-like objects that were written

        """
        try:
            stage = Stage.FEATURE_COMPUTED
            inst_any = getattr(batch[0], "instrument_id", "UNKNOWN")
            topic = build_topic_for_stage(
                stage,
                str(inst_any),
                scheme=self.config.topic_scheme,
                prefix=self.config.topic_prefix,
            )
            ts_min = min(int(getattr(b, "ts_event", 0)) for b in batch)
            ts_max = max(int(getattr(b, "ts_event", 0)) for b in batch)
            payload: dict[str, Any] = {
                "dataset_id": "features",
                "instrument_id": str(inst_any),
                "stage": stage.value,
                "source": "computed",
                "run_id": "feature_store_write",
                "ts_min": ts_min,
                "ts_max": ts_max,
                "count": len(batch),
                "status": EventStatus.SUCCESS.value,
            }
            if self.publisher is not None:
                self.publisher.publish(topic, payload)
        except Exception:
            logger.debug("FeatureStore publish failed", exc_info=True)

    def _publish_single_row_event(self, instrument_id: str, ts_event: int) -> None:
        """
        Publish a single-row write event to the message bus.

        Args:
            instrument_id: Instrument identifier
            ts_event: Event timestamp in nanoseconds

        """
        try:
            stage = Stage.FEATURE_COMPUTED
            topic = build_topic_for_stage(
                stage,
                instrument_id,
                scheme=self.config.topic_scheme,
                prefix=self.config.topic_prefix,
            )
            payload: dict[str, Any] = {
                "dataset_id": "features",
                "instrument_id": instrument_id,
                "stage": stage.value,
                "source": "computed",
                "run_id": "feature_store_write",
                "ts_min": ts_event,
                "ts_max": ts_event,
                "count": 1,
                "status": EventStatus.SUCCESS.value,
            }
            if self.publisher is not None:
                self.publisher.publish(topic, payload)
        except Exception:
            logger.debug("FeatureStore publish failed", exc_info=True)

    def _execute_write(
        self,
        row: dict[str, Any],
    ) -> None:
        """
        Upsert a single feature row (patchable in tests).

        Performs timestamp normalization, circuit breaker gating, and optional
        per-row publishing to the message bus.

        Args:
            row: Dictionary containing feature row data with keys:
                - feature_set_id: Feature set identifier
                - instrument_id: Instrument identifier
                - ts_event: Event timestamp in nanoseconds
                - ts_init: Initialization timestamp in nanoseconds
                - values: Feature name to value mapping
                - is_live: Whether this is live data
                - source: Data source identifier

        """
        # Local import to avoid circular deps
        from sqlalchemy.dialects.postgresql import insert

        # Track stage boundary for observability (cold path only)
        ts_stage_start = time.time_ns()

        # Optional audit logging (sampled)
        self._audit_log_sampled(row)

        # Final guard: normalize any incoming timestamps
        if "ts_event" in row:
            row["ts_event"] = sanitize_timestamp_ns(
                int(row["ts_event"]),
                logger=logger,
                context="FeatureWriter._execute_write",
            )
        if "ts_init" in row:
            row["ts_init"] = sanitize_timestamp_ns(
                int(row["ts_init"]),
                logger=logger,
                context="FeatureWriter._execute_write",
            )

        stmt = insert(self.table).values(row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["feature_set_id", "instrument_id", "ts_event"],
            set_={
                "values": stmt.excluded["values"],
                "ts_init": stmt.excluded.ts_init,
                "source": stmt.excluded.source,
            },
        )

        cb = self.circuit_breaker
        if cb is not None and not cb.can_execute():
            return

        try:
            with self.engine.begin() as conn:
                conn.execute(stmt)
        except Exception:
            if cb is not None:
                try:
                    cb.record_failure()
                except Exception:
                    pass
            raise
        else:
            if cb is not None:
                try:
                    cb.record_success()
                except Exception:
                    pass

        # Optional per-row publish when enabled
        if (
            self.config.enable_publishing
            and self.publisher is not None
            and self.config.publish_mode in ("row", "both")
        ):
            self._publish_per_row_event(row)

        # Record observability data (off hot path - background processing only)
        ts_stage_end = time.time_ns()
        self._record_observability_stage_boundary(
            stage="feature_storage",
            instrument_id=str(row.get("instrument_id", "unknown")),
            ts_stage_start=ts_stage_start,
            ts_stage_end=ts_stage_end,
            row_count=1,
        )

    def _audit_log_sampled(self, row: dict[str, Any]) -> None:
        """
        Optionally log audit information for the row (sampled by ML_AUDIT env var).

        Args:
            row: Feature row being written

        """
        try:
            import os
            import random

            sample = int(os.getenv("ML_AUDIT", "0"))
            if sample > 0 and random.randint(1, sample) == 1:
                logger.info("AUDIT FeatureWriter._execute_write: keys=%s", list(row.keys()))
        except Exception:
            logger.debug(
                "Audit logging skipped due to error",
                exc_info=True,
            )

    def _publish_per_row_event(self, row: dict[str, Any]) -> None:
        """
        Publish a per-row event to the message bus.

        Args:
            row: Feature row that was written

        """
        try:
            stage = Stage.FEATURE_COMPUTED
            inst = str(row.get("instrument_id", "UNKNOWN"))
            topic = build_topic_for_stage(
                stage,
                inst,
                scheme=self.config.topic_scheme,
                prefix=self.config.topic_prefix,
            )
            ts_e = int(row.get("ts_event", 0))
            payload: dict[str, Any] = {
                "dataset_id": "features",
                "instrument_id": inst,
                "stage": stage.value,
                "source": str(row.get("source", "computed")),
                "run_id": "feature_store_row",
                "ts_min": ts_e,
                "ts_max": ts_e,
                "count": 1,
                "status": EventStatus.SUCCESS.value,
            }
            if self.publisher is not None:
                self.publisher.publish(topic, payload)
        except Exception:
            logger.debug("FeatureStore per-row publish failed", exc_info=True)

    def _record_observability_stage_boundary(
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
            stage: Processing stage name
            instrument_id: Instrument identifier
            ts_stage_start: Stage start timestamp in nanoseconds
            ts_stage_end: Stage end timestamp in nanoseconds
            row_count: Number of rows processed

        """
        try:
            from ml.common.observability_utils import record_stage_boundary as _rec

            obs_service = self._observability_service
            _rec(
                obs_service,
                component="feature_store",
                instrument_id=instrument_id,
                stage=stage,
                ts_stage_start=ts_stage_start,
                ts_stage_end=ts_stage_end,
                row_count=row_count,
            )
        except Exception:
            logger.debug(
                "Failed to record observability stage boundary",
                exc_info=True,
            )

    def write_batch(self, data: list[object]) -> None:
        """
        Write a batch of FeatureData rows (compat shim).

        Args:
            data: List of FeatureData-like objects to upsert. Accepts objects
                with attributes feature_set_id, instrument_id, ts_event, ts_init,
                feature_values.

        Example
        -------
        >>> writer.write_batch([feature_data_1, feature_data_2])
        >>> assert len(writer._write_buffer) == 0  # Buffer cleared after write

        """
        if not data:
            return

        # Append to buffer for visibility during the call (tests assert
        # the buffer is cleared after write_batch returns)
        self._write_buffer.extend(data)

        for item in list(data):
            fs_id = getattr(item, "feature_set_id", None)
            inst = getattr(item, "instrument_id", None)
            tse = int(getattr(item, "ts_event", 0))
            tsi = int(getattr(item, "ts_init", tse))
            # Use feature_values to avoid colliding with mapping API on objects
            try:
                vals = getattr(item, "feature_values")
            except Exception:
                vals = {}
            row = {
                "feature_set_id": fs_id,
                "instrument_id": inst,
                "ts_event": tse,
                "ts_init": tsi,
                "values": dict(vals or {}),
                "is_live": False,
                "source": "computed",
            }
            self._execute_write(row)

        # Clear buffer after successful write
        self._write_buffer.clear()

        # Publish batch summary event if enabled
        if self.config.enable_publishing and self.publisher is not None and data:
            self._publish_batch_summary_event(list(data))

    def store_features(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible alias for write_features with relaxed argument requirements.

        Accepts minimal explicit args used in integration tests: instrument_id,
        ts_event, and features. Fills feature_set_id from current pipeline/config and
        ts_init with ts_event when not provided.

        Example
        -------
        >>> writer.store_features(
        ...     instrument_id="SPY.DATABENTO",
        ...     ts_event=1700000000000000000,
        ...     features={"close_return": 0.01},
        ... )

        """
        if args or set(kwargs.keys()) & {"feature_set_id", "data"}:
            # Delegate when full signature or batch data is supplied
            self.write_features(*args, **kwargs)
            return

        instrument_id = kwargs.get("instrument_id")
        ts_event = kwargs.get("ts_event")
        features = kwargs.get("features")
        ts_init = kwargs.get("ts_init", ts_event)
        if instrument_id is None or ts_event is None or features is None:
            # Fallback to strict path
            self.write_features(*args, **kwargs)
            return

        self.write_features(
            feature_set_id=self.get_feature_set_id(),
            instrument_id=str(instrument_id),
            features=features,
            ts_event=int(ts_event),
            ts_init=int(ts_init) if ts_init is not None else int(ts_event),
        )


__all__ = [
    "FeatureWriterComponent",
    "FeatureWriterConfig",
    "FeatureWriterProtocol",
    "MessagePublisherProtocol",
]
