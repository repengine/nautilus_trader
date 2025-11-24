"""
Feature retrieval operations for FeatureStore.

This module handles feature read/query operations from storage.

"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd
from sqlalchemy import Table
from sqlalchemy import bindparam
from sqlalchemy import desc
from sqlalchemy import select
from sqlalchemy.engine import Engine

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class FeatureRetrievalProtocol(Protocol):
    """
    Protocol for feature read operations.
    """

    def get_training_data(
        self,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        feature_names: list[str] | None = None,
    ) -> tuple[Any, Any, list[str]]:
        """
        Retrieve features for training.
        """
        ...

    def read_range(
        self,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Read feature range.
        """
        ...


class FeatureRetrieval:
    """
    Handles feature storage read operations.

    Implements Pattern 3 (Cold Path) and Pattern 5 (Centralized Metrics).

    """

    # Pattern 5: Centralized Metrics
    _READ_COUNTER = get_counter(
        "ml_feature_reads_total",
        "Total feature read operations",
    )
    _READ_LATENCY = get_histogram(
        "ml_feature_read_duration_seconds",
        "Feature read operation duration",
    )

    def __init__(
        self,
        engine: Engine,
        table: Table,
        feature_set_id: str,
        catalog_path: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize feature retrieval.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine
        table : Table
            Feature storage table
        feature_set_id : str
            Feature set identifier for queries
        catalog_path : str | None
            Path to Nautilus data catalog for bar loading
        logger : logging.Logger | None
            Logger for operations

        """
        self._engine = engine
        self._table = table
        self._feature_set_id = feature_set_id
        self._catalog_path = catalog_path
        self._logger = logger or logging.getLogger(__name__)

    def get_training_data(
        self,
        instrument_id: str,
        start: Any,
        end: Any,
        feature_set_id: str,
        feature_names: list[str] | None = None,
        include_bars: bool = True,
    ) -> tuple[Any, Any, list[str]]:
        """
        Retrieve features for training.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        start : datetime
            Start time
        end : datetime
            End time
        feature_set_id : str
            Feature set identifier
        feature_names : list[str] | None
            Specific features to retrieve (None = all)
        include_bars : bool, default True
            Whether to join with bar data (currently unused)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Features array, timestamps array, and feature names

        """
        import numpy as np

        from ml.common.timestamps import sanitize_timestamp_ns

        with self._READ_LATENCY.time():
            try:
                # Convert datetime to nanoseconds
                start_ts = (
                    int(start.timestamp() * 1e9) if hasattr(start, "timestamp") else int(start)
                )
                end_ts = int(end.timestamp() * 1e9) if hasattr(end, "timestamp") else int(end)

                # Normalize timestamps
                start_ns = sanitize_timestamp_ns(
                    start_ts,
                    context="feature_store.get_training_data.start",
                )
                end_ns = sanitize_timestamp_ns(
                    end_ts,
                    context="feature_store.get_training_data.end",
                )

                # Query features (use passed feature_set_id, not instance variable)
                query = (
                    select(
                        self._table.c.ts_event,
                        self._table.c["values"],
                    )
                    .where(
                        (self._table.c.feature_set_id == feature_set_id)
                        & (self._table.c.instrument_id == instrument_id)
                        & (self._table.c.ts_event >= start_ns)
                        & (self._table.c.ts_event <= end_ns),
                    )
                    .order_by(self._table.c.ts_event)
                )

                with self._engine.connect() as conn:
                    result = conn.execute(query)
                    rows = result.fetchall()

                if not rows:
                    self._logger.warning(
                        "No features found for %s in range [%d, %d]",
                        instrument_id,
                        start_ns,
                        end_ns,
                    )
                    return (
                        np.array([]),
                        np.array([]),
                        [],
                    )

                # Extract timestamps
                timestamps = np.array([row[0] for row in rows], dtype=np.int64)

                # Extract feature names from first row if not provided
                first_values = rows[0][1]
                if isinstance(first_values, str):
                    first_values = json.loads(first_values)

                all_feature_names = list(first_values.keys()) if first_values else []

                # Filter to requested features if specified
                if feature_names is None:
                    feature_names = all_feature_names
                else:
                    # Ensure requested features exist
                    feature_names = [fn for fn in feature_names if fn in all_feature_names]

                # Reconstruct arrays in feature_names order
                feature_arrays: list[list[float]] = []
                for _, values_json in rows:
                    mapping = values_json
                    if isinstance(mapping, str):
                        try:
                            mapping = json.loads(mapping)
                        except Exception:
                            mapping = {}
                    feature_arrays.append(
                        [float(mapping.get(name, 0.0)) for name in feature_names],
                    )

                features = np.array(feature_arrays, dtype=np.float64)

                self._READ_COUNTER.inc()
                self._logger.debug(
                    "Retrieved %d feature rows for %s",
                    len(features),
                    instrument_id,
                )

                return features, timestamps, feature_names

            except Exception as e:
                self._logger.error(
                    "Failed to get training data for %s: %s",
                    instrument_id,
                    e,
                    exc_info=True,
                )
                return (
                    np.array([]),
                    np.array([]),
                    [],
                )

    def get_latest_at_or_before(
        self,
        instrument_id: str,
        ts_event: int,
        feature_set_id: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, float] | None:
        """
        Get latest feature values at or before timestamp.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp in nanoseconds
        feature_set_id : str
            Feature set identifier
        feature_names : list[str] | None
            Specific features to retrieve (None = all)

        Returns
        -------
        dict[str, float] | None
            Feature name-value mapping, or None if not found

        """
        from ml.common.timestamps import sanitize_timestamp_ns

        try:
            ts_norm = sanitize_timestamp_ns(
                ts_event,
                context="feature_store.get_latest_at_or_before",
            )

            query = (
                select(self._table.c["values"])
                .where(
                    (self._table.c.feature_set_id == feature_set_id)
                    & (self._table.c.instrument_id == instrument_id)
                    & (self._table.c.ts_event <= ts_norm),
                )
                .order_by(desc(self._table.c.ts_event))
                .limit(1)
            )

            with self._engine.connect() as conn:
                row = conn.execute(query).fetchone()

            if not row:
                return None

            mapping: Any = row[0]
            if isinstance(mapping, str):
                try:
                    mapping = json.loads(mapping)
                except Exception:
                    mapping = {}

            # Filter to requested features if specified
            if feature_names is not None:
                mapping = {k: v for k, v in mapping.items() if k in feature_names}

            return {str(k): float(v) for k, v in (mapping or {}).items()}

        except Exception as e:
            self._logger.error(
                "Failed to get latest features for %s at %d: %s",
                instrument_id,
                ts_event,
                e,
                exc_info=True,
            )
            return None

    def read_range(
        self,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Read features in time range.

        Parameters
        ----------
        start_ns : int
            Start timestamp (nanoseconds, inclusive)
        end_ns : int
            End timestamp (nanoseconds, exclusive)
        instrument_id : str | None
            Optional instrument filter

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature_set_id, instrument_id,
            values, ts_event, ts_init

        """
        try:
            params: dict[str, Any] = {
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
            }

            conditions = [
                self._table.c.ts_event >= bindparam("start_ns"),
                self._table.c.ts_event < bindparam("end_ns"),
            ]

            if instrument_id is not None:
                conditions.append(self._table.c.instrument_id == bindparam("instrument_id"))
                params["instrument_id"] = instrument_id

            stmt = (
                select(
                    self._table.c.feature_set_id,
                    self._table.c.instrument_id,
                    self._table.c["values"],
                    self._table.c.ts_event,
                    self._table.c.ts_init,
                )
                .where(*conditions)
                .order_by(self._table.c.ts_event)
            )

            with self._engine.connect() as conn:
                features_data = pd.read_sql_query(stmt, conn, params=params)

            self._READ_COUNTER.inc()
            return features_data

        except Exception as e:
            self._logger.error(
                "Failed to read feature range [%d, %d): %s",
                start_ns,
                end_ns,
                e,
                exc_info=True,
            )
            return pd.DataFrame()

    def _load_bars_from_nautilus(
        self,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
    ) -> Any:
        """
        Load bars from Nautilus data catalog.

        Pattern 4: Progressive fallback (catalog → database → None)

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        start_ts : int
            Start timestamp (nanoseconds)
        end_ts : int
            End timestamp (nanoseconds)

        Returns
        -------
        Any
            Bars dataframe with Nautilus schema (pl.DataFrame)

        """
        from typing import Any, cast

        from sqlalchemy import text as _text

        from ml._imports import pl
        from ml.common.timestamps import sanitize_timestamp_ns

        pl = cast(Any, pl)

        try:
            start_ns = sanitize_timestamp_ns(
                start_ts,
                context="feature_store._load_bars_from_nautilus.start",
            )
            end_ns = sanitize_timestamp_ns(
                end_ts,
                context="feature_store._load_bars_from_nautilus.end",
            )

            sql = _text(
                """
                SELECT ts_event, open, high, low, close, volume
                FROM public.bar
                WHERE instrument_id = :instrument_id
                  AND ts_event >= :start_ns
                  AND ts_event <= :end_ns
                ORDER BY ts_event
                """,
            )

            with self._engine.connect() as conn:
                from collections.abc import Mapping

                params = cast(
                    Mapping[str, Any],
                    {
                        "instrument_id": instrument_id,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                    },
                )
                pdf = pd.read_sql_query(sql, conn, params=params)

            result: Any = pl.from_pandas(pdf)
            return result

        except Exception as e:
            self._logger.error(
                "Failed to load bars for %s: %s",
                instrument_id,
                e,
                exc_info=True,
            )
            empty_result: Any = pl.DataFrame()
            return empty_result

    def _features_exist(
        self,
        instrument_id: str,
        ts_event: int,
    ) -> bool:
        """
        Check if features exist for instrument at timestamp.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        ts_event : int
            Event timestamp (nanoseconds)

        Returns
        -------
        bool
            True if features exist, False otherwise

        """
        from ml.common.timestamps import sanitize_timestamp_ns

        try:
            ts_norm = sanitize_timestamp_ns(
                ts_event,
                context="feature_store._features_exist",
            )

            query = (
                select(self._table.c.ts_event)
                .where(
                    (self._table.c.feature_set_id == self._feature_set_id)
                    & (self._table.c.instrument_id == instrument_id)
                    & (self._table.c.ts_event == ts_norm),
                )
                .limit(1)
            )

            with self._engine.connect() as conn:
                result = conn.execute(query)
                return result.fetchone() is not None

        except Exception as e:
            self._logger.error(
                "Failed to check feature existence for %s at %d: %s",
                instrument_id,
                ts_event,
                e,
                exc_info=True,
            )
            return False
