"""
Feature persistence operations for FeatureStore.

This module handles feature write operations to storage, implementing Pattern 1 (4-Store
Integration), Pattern 3 (Cold Path), Pattern 4 (Progressive Fallback), and Pattern 5
(Centralized Metrics).

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    from collections.abc import Mapping

    from ml.stores.protocols import CircuitBreakerProtocol


logger = logging.getLogger(__name__)


class FeaturePersistenceProtocol(Protocol):
    """
    Protocol for feature write operations.
    """

    def write_features(
        self,
        feature_set_id: str,
        instrument_id: str,
        ts_event: int,
        ts_init: int,
        features: Mapping[str, float],
        *,
        publish_bus: bool = True,
    ) -> None:
        """
        Write features to storage.
        """
        ...

    def write_batch(
        self,
        data: list[Any],
        *,
        publish_bus: bool = True,
    ) -> None:
        """
        Write batch of feature records.
        """
        ...


class FeaturePersistence:
    """
    Handles feature storage write operations.

    Implements Pattern 1 (4-Store Integration), Pattern 3 (Cold Path),
    Pattern 4 (Progressive Fallback), and Pattern 5 (Centralized Metrics).

    IMPORTANT: This is COLD PATH only. Write operations are synchronous
    and may include database I/O.

    """

    # Pattern 5: Centralized Metrics Bootstrap
    _WRITE_COUNTER = get_counter(
        "ml_feature_writes_total",
        "Total feature write operations",
    )
    _WRITE_ERRORS = get_counter(
        "ml_feature_write_errors_total",
        "Total feature write errors",
    )
    _WRITE_LATENCY = get_histogram(
        "ml_feature_write_duration_seconds",
        "Feature write operation duration",
    )

    def __init__(
        self,
        engine: Engine,
        table: Table,
        circuit_breaker: CircuitBreakerProtocol | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize feature persistence.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for database operations
        table : Table
            Feature storage table
        circuit_breaker : CircuitBreakerProtocol | None
            Circuit breaker for fault tolerance (optional)
        logger : logging.Logger | None
            Logger for operations (default: creates module logger)

        """
        self._engine = engine
        self._table = table
        self._circuit_breaker = circuit_breaker
        self._logger = logger or logging.getLogger(__name__)

    def _execute_write(
        self,
        row: dict[str, Any],
    ) -> bool:
        """
        Execute write statement with circuit breaker protection.

        Parameters
        ----------
        row : dict[str, Any]
            Row data to write

        Returns
        -------
        bool
            True if write successful, False otherwise

        """
        # Normalize timestamps before write
        from ml.common.timestamps import sanitize_timestamp_ns

        if "ts_event" in row:
            row["ts_event"] = sanitize_timestamp_ns(
                int(row["ts_event"]),
                logger=self._logger,
                context="FeaturePersistence._execute_write.ts_event",
            )
        if "ts_init" in row:
            row["ts_init"] = sanitize_timestamp_ns(
                int(row["ts_init"]),
                logger=self._logger,
                context="FeaturePersistence._execute_write.ts_init",
            )

        # Check circuit breaker
        if self._circuit_breaker is not None and not self._circuit_breaker.can_execute():
            self._logger.warning("Circuit breaker open, skipping write")
            return False

        # Execute write with upsert
        stmt = insert(self._table).values(row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["feature_set_id", "instrument_id", "ts_event"],
            set_={
                "values": stmt.excluded["values"],
                "ts_init": stmt.excluded.ts_init,
                "source": stmt.excluded.source,
            },
        )

        try:
            with self._engine.begin() as conn:
                conn.execute(stmt)

            # Record success
            if self._circuit_breaker is not None:
                try:
                    self._circuit_breaker.record_success()
                except Exception:
                    pass

            return True

        except Exception as e:
            # Record failure
            if self._circuit_breaker is not None:
                try:
                    self._circuit_breaker.record_failure()
                except Exception:
                    pass

            self._WRITE_ERRORS.inc()
            self._logger.error(
                "Failed to write feature row for instrument %s: %s",
                row.get("instrument_id", "unknown"),
                e,
                exc_info=True,
            )
            return False

    def write_features(
        self,
        feature_set_id: str,
        instrument_id: str,
        ts_event: int,
        ts_init: int,
        features: Mapping[str, float],
        *,
        publish_bus: bool = True,
    ) -> None:
        """
        Write features to storage.

        Parameters
        ----------
        feature_set_id : str
            Feature set identifier
        instrument_id : str
            Instrument identifier (e.g., "AAPL.NASDAQ")
        ts_event : int
            Event timestamp in nanoseconds since epoch
        ts_init : int
            Initialization timestamp in nanoseconds since epoch
        features : Mapping[str, float]
            Feature name-value pairs
        publish_bus : bool, keyword-only, default True
            Whether to publish to message bus (if enabled)

        """
        with self._WRITE_LATENCY.time():
            # Prepare row
            features_payload: dict[str, float] = {
                str(k): float(v) for k, v in dict(features or {}).items()
            }

            row = {
                "feature_set_id": feature_set_id,
                "instrument_id": instrument_id,
                "ts_event": int(ts_event),
                "ts_init": int(ts_init),
                "values": features_payload,
                "is_live": False,
                "source": "computed",
            }

            # Execute write
            success = self._execute_write(row)

            # Update metrics
            if success:
                self._WRITE_COUNTER.inc()
                self._logger.debug(
                    "Wrote features for %s at ts_event=%d",
                    instrument_id,
                    ts_event,
                )

    def write_batch(
        self,
        data: list[Any],
        *,
        publish_bus: bool = True,
    ) -> None:
        """
        Write batch of feature records.

        Parameters
        ----------
        data : list[Any]
            List of feature records to write.
            Each record must have attributes: feature_set_id, instrument_id,
            ts_event, ts_init, feature_values
        publish_bus : bool, keyword-only, default True
            Whether to publish to message bus (if enabled)

        """
        if not data:
            return

        with self._WRITE_LATENCY.time():
            success_count = 0

            for item in data:
                try:
                    # Extract attributes
                    fs_id = getattr(item, "feature_set_id", None)
                    inst = getattr(item, "instrument_id", None)
                    tse = int(getattr(item, "ts_event", 0))
                    tsi = int(getattr(item, "ts_init", tse))

                    # Get feature values
                    try:
                        vals = getattr(item, "feature_values")
                    except Exception:
                        vals = {}

                    # Prepare row
                    row = {
                        "feature_set_id": fs_id,
                        "instrument_id": inst,
                        "ts_event": tse,
                        "ts_init": tsi,
                        "values": dict(vals or {}),
                        "is_live": False,
                        "source": "computed",
                    }

                    # Write
                    if self._execute_write(row):
                        success_count += 1

                except Exception as e:
                    self._logger.warning(
                        "Failed to write batch item: %s",
                        e,
                        exc_info=True,
                    )
                    continue

            # Update metrics
            self._WRITE_COUNTER.inc(amount=success_count)
            self._logger.info(
                "Batch write completed: %d/%d successful",
                success_count,
                len(data),
            )

    def store_features(
        self,
        instrument_id: str,
        timestamp: int,
        features: Mapping[str, float],
        metadata: dict[str, Any] | None = None,
        feature_set_id: str | None = None,
    ) -> bool:
        """
        Store features with optional metadata.

        Higher-level method that may delegate to write_features.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        timestamp : int
            Timestamp in nanoseconds
        features : Mapping[str, float]
            Features to store
        metadata : dict[str, Any] | None
            Optional metadata
        feature_set_id : str | None
            Optional feature set ID (if None, must be provided)

        Returns
        -------
        bool
            True if storage successful, False otherwise

        """
        if feature_set_id is None:
            self._logger.error("feature_set_id required for store_features")
            return False

        try:
            self.write_features(
                feature_set_id=feature_set_id,
                instrument_id=instrument_id,
                ts_event=timestamp,
                ts_init=timestamp,
                features=features,
            )
            return True
        except Exception as e:
            self._logger.error(
                "Failed to store features for %s: %s",
                instrument_id,
                e,
                exc_info=True,
            )
            return False
