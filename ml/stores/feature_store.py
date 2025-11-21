"""
Feature store facade delegating to specialized components.

This module provides a unified interface for computing, storing, and retrieving
ML features from the same PostgreSQL instance used by Nautilus Trader.

The facade delegates to 5 specialized components:
- FeatureTableManager: Schema and table management
- FeatureVersioning: Configuration hashing and feature set IDs
- FeaturePersistence: Write operations
- FeatureRetrieval: Read operations
- FeatureComputation: Historical and realtime feature computation

Key principles:
- Single PostgreSQL container (Nautilus's existing one)
- FeatureEngineer provides all computation logic (training/inference parity)
- Features stored alongside Nautilus market data for unified access
- Efficient batch computation for historical data
- Real-time computation for live trading

Feature Flag:
  Set ML_USE_LEGACY_FEATURE_STORE=1 to use the original god class implementation.
  Default (0 or unset) uses this component-based facade.

"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from sqlalchemy import MetaData
from sqlalchemy.engine import Engine

from ml.common.db_utils import get_or_create_engine
from ml.common.message_bus import BusPublisherMixin
from ml.common.message_bus import MessagePublisherProtocol
from ml.config.base import MLFeatureConfig
from ml.core.db_engine import EngineManager
from ml.features.config import FeatureConfig
from ml.features.facade import FeatureEngineer
from ml.features.indicators import IndicatorManager
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.stores.mixins import DataRegistryMixin
from ml.stores.mixins import HealthMixin


if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from nautilus_trader.model.data import Bar

    from ml.registry.protocols import RegistryProtocol
    from ml.stores.services.feature_services import CrossAssetFeatureService


def _should_use_component_impl() -> bool:
    """
    Determine whether to enable the component-based FeatureStore facade.

    Precedence (highest to lowest):
    1. ML_USE_COMPONENT_FEATURE_STORE=1 explicitly opts in.
    2. ML_USE_COMPONENT_FEATURE_STORE=0 explicitly opts out.
    3. Historical flag ML_USE_LEGACY_FEATURE_STORE keeps working:
       - "1" => legacy implementation
       - "0" => component implementation
       - unset => legacy (default)
    """
    legacy_flag = os.getenv("ML_USE_LEGACY_FEATURE_STORE")
    if legacy_flag is not None:
        return legacy_flag.strip() == "0"

    component_flag = os.getenv("ML_USE_COMPONENT_FEATURE_STORE")
    if component_flag is not None:
        return component_flag.strip() == "1"

    return False


USE_COMPONENT_FEATURE_STORE = _should_use_component_impl()
USE_LEGACY_FEATURE_STORE = not USE_COMPONENT_FEATURE_STORE


logger = logging.getLogger(__name__)


class ComponentFeatureStore(HealthMixin, BusPublisherMixin, DataRegistryMixin):
    """
    Unified feature computation and storage facade.

    This facade delegates to 5 specialized components while maintaining
    backward compatibility with the original FeatureStore API.

    Components:
    - FeatureTableManager: Database schema and table management
    - FeatureVersioning: Configuration hashing and feature set identification
    - FeaturePersistence: Write operations to storage
    - FeatureRetrieval: Read operations from storage
    - FeatureComputation: Feature calculation (historical and realtime)

    """

    def __init__(
        self,
        connection_string: str,
        feature_config: FeatureConfig | MLFeatureConfig | None = None,
        pipeline_spec: PipelineSpec | None = None,
        persistence_manager: object | None = None,
        enable_publishing: bool = False,
        publisher: MessagePublisherProtocol | None = None,
        publish_mode: Literal["batch", "row", "both"] = "batch",
        **_: Any,
    ) -> None:
        """
        Initialize the feature store facade.

        Parameters
        ----------
        connection_string : str
            PostgreSQL connection string (same as Nautilus uses).
            Example: "postgresql://postgres:postgres@localhost:5432/nautilus"
        feature_config : FeatureConfig | MLFeatureConfig, optional
            Configuration for feature engineering.
        pipeline_spec : PipelineSpec, optional
            Pipeline specification for feature computation.
        persistence_manager : object | None
            Optional persistence/session provider (used by tests for mocking).
        enable_publishing : bool, optional
            When True, publish store events to the optional message bus.
        publisher : MessagePublisherProtocol | None, optional
            Publisher implementation used when `enable_publishing` is True.
        publish_mode : {"batch", "row", "both"}, optional
            Controls whether to publish batch summaries, per-row events, or both.

        """
        # Store configuration
        self.connection_string = connection_string
        self._data_registry: RegistryProtocol | None = None
        self.persistence: object | None = persistence_manager

        # Normalize feature config (accept both types)
        if isinstance(feature_config, FeatureConfig):
            self.feature_config: FeatureConfig = feature_config
        elif isinstance(feature_config, MLFeatureConfig):
            try:
                import msgspec as _msgspec

                self.feature_config = FeatureConfig(**_msgspec.to_builtins(feature_config))
            except Exception:
                self.feature_config = FeatureConfig(**getattr(feature_config, "__dict__", {}))
        else:
            self.feature_config = FeatureConfig()

        self.pipeline_spec = pipeline_spec

        # Create engine and metadata
        self.engine: Engine = get_or_create_engine(connection_string)
        self.metadata = MetaData()

        # Component 1: Table Manager
        from ml.stores.feature_table_manager import FeatureTableManager

        self._table_mgr = FeatureTableManager(
            engine=self.engine,
            metadata=self.metadata,
            logger=logger,
        )
        self.feature_values_table = self._table_mgr.setup_tables()

        # Create pipeline runners
        self.pipeline_runner_offline: PipelineRunner | None = None
        self.pipeline_runner_online: PipelineRunner | None = None

        if self.pipeline_spec:
            from ml.registry.base import DataRequirements

            # Offline (batch/teacher): allow L1_L2 to include microstructure/trade-flow
            self.pipeline_runner_offline = PipelineRunner(
                self.pipeline_spec,
                DataRequirements.L1_L2,
            )
            # Online (student/runtime): limit to L1 until actors are available
            self.pipeline_runner_online = PipelineRunner(
                self.pipeline_spec,
                DataRequirements.L1_ONLY,
            )

        # Component 2: Versioning
        from ml.stores.feature_versioning import FeatureVersioning

        self._versioning = FeatureVersioning(
            feature_config=self.feature_config,
            pipeline_runner_offline=self.pipeline_runner_offline,
            pipeline_runner_online=self.pipeline_runner_online,
            logger=logger,
        )
        self.pipeline_hash = self._versioning.compute_config_hash()

        # Component 3: Persistence
        from ml.stores.feature_persistence import FeaturePersistence
        from ml.stores.protocols import CircuitBreakerProtocol as _CBP

        self._circuit_breaker: _CBP | None = None
        self._persistence = FeaturePersistence(
            engine=self.engine,
            table=self.feature_values_table,
            circuit_breaker=self._circuit_breaker,
            logger=logger,
        )

        # Component 4: Retrieval
        from ml.stores.feature_retrieval import FeatureRetrieval

        feature_set_id = self._versioning.get_feature_set_id()
        self._retrieval = FeatureRetrieval(
            engine=self.engine,
            table=self.feature_values_table,
            feature_set_id=feature_set_id,
            catalog_path=None,
            logger=logger,
        )

        # Component 5: Computation
        from ml.stores.feature_computation import FeatureComputation

        self.feature_engineer = FeatureEngineer(self.feature_config)
        self._indicator_managers: dict[str, IndicatorManager] = {}

        self._computation = FeatureComputation(
            feature_engineer=self.feature_engineer,
            feature_versioning=self._versioning,
            persistence=self._persistence,
            retrieval=self._retrieval,
            indicator_manager=None,
            data_registry=None,
            logger=logger,
        )

        # Backwards compatibility: write buffer for tests
        from ml.stores.base import FeatureData

        self._write_buffer: list[FeatureData] = []
        self._buffer: list[FeatureData] = self._write_buffer

        # Cross-asset service (lazy initialization)
        self._cross_asset_service: CrossAssetFeatureService | None = None

        # Optional message publishing
        self._init_bus_publishing(
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode=publish_mode,
        )

    def _get_data_registry(self) -> RegistryProtocol | None:
        """
        Delegate initialization to shared mixin (unified across stores).

        Returns
        -------
        RegistryProtocol | None
            The data registry instance, or None if not available.

        """
        return DataRegistryMixin._get_data_registry(self)

    def set_data_registry(self, registry: RegistryProtocol) -> None:
        """
        Set the DataRegistry instance used for event emission and watermarks.

        Parameters
        ----------
        registry : RegistryProtocol
            The shared registry instance to use.

        """
        self._data_registry = registry
        # Update computation component's registry
        self._computation._data_registry = registry

    @property
    def cross_asset(self) -> CrossAssetFeatureService:
        """
        Access cross-asset feature operations (beta, spreads, correlations).

        This property provides lazy initialization of the CrossAssetFeatureService,
        which enables storage and retrieval of cross-asset relationship features
        using the existing ml_feature_values table with namespaced feature_set_ids.

        Returns
        -------
        CrossAssetFeatureService
            Service instance for cross-asset feature operations.

        Example
        -------
        >>> store = ComponentFeatureStore(connection_string="postgresql://...")
        >>> store.cross_asset.write_beta(
        ...     asset_id="AAPL",
        ...     benchmark_id="SPY",
        ...     ts_event=1234567890000000000,
        ...     ts_init=1234567890000000000,
        ...     beta=1.25,
        ...     lookback_periods=60,
        ...     ewma_span=30,
        ... )
        >>> history = store.cross_asset.get_beta_history(
        ...     asset_id="AAPL",
        ...     benchmark_id="SPY",
        ...     start_ts=1234567890000000000,
        ...     end_ts=1234567891000000000,
        ... )

        """
        if self._cross_asset_service is None:
            from ml.stores.services.feature_services import CrossAssetFeatureService

            self._cross_asset_service = CrossAssetFeatureService(deps=self)

        return self._cross_asset_service

    # -------------------------------------------------------------------------
    # Public API: Computation Methods
    # -------------------------------------------------------------------------

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        force_recompute: bool = False,
    ) -> int:
        """
        Compute and store features for historical data.

        Delegates to FeatureComputation component.

        Parameters
        ----------
        instrument_id : str
            Instrument to compute features for.
        start : datetime
            Start time for historical computation.
        end : datetime
            End time for historical computation.
        force_recompute : bool, default False
            If True, recompute even if features exist.

        Returns
        -------
        int
            Number of feature rows computed and stored.

        """
        feature_set_id = self._versioning.get_feature_set_id()
        feature_names = self._versioning.get_feature_names(online=False)

        return self._computation.compute_and_store_historical(
            instrument_id=instrument_id,
            start=start,
            end=end,
            feature_set_id=feature_set_id,
            feature_names=feature_names,
            force_recompute=force_recompute,
        )

    def compute_realtime(
        self,
        bar: Bar,
        store: bool = True,
        indicator_manager: IndicatorManager | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Compute features for real-time inference.

        Delegates to FeatureComputation component.

        Parameters
        ----------
        bar : Bar
            Current bar from Nautilus.
        store : bool, default True
            Whether to store computed features for future training.
        indicator_manager : IndicatorManager | None, default None
            Optional indicator manager for stateful indicator computation.

        Returns
        -------
        npt.NDArray[np.float32]
            Computed feature vector.

        """
        # Prepare indicator manager (prefer provided from actor for shared state)
        instrument_key = str(
            getattr(
                bar,
                "instrument_id",
                getattr(bar, "bar_type", getattr(bar, "instrument_id", None)),
            ),
        )
        instrument_key = (
            str(bar.bar_type.instrument_id)
            if hasattr(bar, "bar_type") and hasattr(bar.bar_type, "instrument_id")
            else str(getattr(bar, "instrument_id", "unknown"))
        )

        if indicator_manager is None:
            indicator_manager = self._indicator_managers.get(instrument_key)
            if indicator_manager is None:
                indicator_manager = IndicatorManager(self.feature_engineer.config)
                self._indicator_managers[instrument_key] = indicator_manager

        feature_set_id = self._versioning.get_feature_set_id()
        feature_names_online = self._versioning.get_feature_names(online=True)

        return self._computation.compute_realtime(
            bar=bar,
            indicator_manager=indicator_manager,
            feature_set_id=feature_set_id,
            feature_names_online=feature_names_online,
            store=store,
        )

    def compute_historical_parallel(
        self,
        instrument_ids: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        force_recompute: bool = False,
        max_workers: int = 4,
    ) -> dict[str, int]:
        """
        Compute-and-store historical features for multiple instruments in parallel.

        Delegates to FeatureComputation component.

        Parameters
        ----------
        instrument_ids : list[str]
            Instruments to compute.
        start : datetime, optional
            Start time (inclusive).
        end : datetime, optional
            End time (exclusive).
        force_recompute : bool, default False
            Recompute even if features exist.
        max_workers : int, default 4
            Maximum concurrent workers (bounded to avoid pool exhaustion).

        Returns
        -------
        dict[str, int]
            Mapping instrument_id -> rows written (0 on failure).

        """
        return self._computation.compute_historical_parallel(
            instrument_ids=instrument_ids,
            start=start,
            end=end,
            force_recompute=force_recompute,
            max_workers=max_workers,
        )

    # -------------------------------------------------------------------------
    # Public API: Read Methods
    # -------------------------------------------------------------------------

    def get_training_data(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        include_bars: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
        """
        Load features for training.

        Delegates to FeatureRetrieval component.

        Parameters
        ----------
        instrument_id : str
            Instrument to load features for.
        start : datetime
            Start time.
        end : datetime
            End time.
        include_bars : bool, default True
            Whether to join with bar data for labels.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]
            Features array, timestamps array, and feature names.

        """
        feature_set_id = self._versioning.get_feature_set_id()
        feature_names = self._versioning.get_feature_names(online=False)

        return self._retrieval.get_training_data(
            instrument_id=instrument_id,
            start=start,
            end=end,
            feature_set_id=feature_set_id,
            feature_names=feature_names,
            include_bars=include_bars,
        )

    def get_latest_at_or_before(
        self,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return the latest feature row at or before the given timestamp.

        Delegates to FeatureRetrieval component.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier.
        ts_event : int
            Event timestamp in nanoseconds.

        Returns
        -------
        dict[str, float] | None
            Mapping of feature name to value, or None when not found.

        """
        feature_set_id = self._versioning.get_feature_set_id()

        return self._retrieval.get_latest_at_or_before(
            instrument_id=instrument_id,
            ts_event=ts_event,
            feature_set_id=feature_set_id,
            feature_names=None,
        )

    def read_range(
        self,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Read features in a time range (inclusive start, exclusive end).

        Delegates to FeatureRetrieval component.

        Parameters
        ----------
        start_ns : int
            Start timestamp in nanoseconds (inclusive).
        end_ns : int
            End timestamp in nanoseconds (exclusive).
        instrument_id : str | None
            Optional instrument filter.

        Returns
        -------
        pd.DataFrame
            A DataFrame of rows with columns: feature_set_id, instrument_id,
            values, ts_event, ts_init.

        """
        return self._retrieval.read_range(
            start_ns=start_ns,
            end_ns=end_ns,
            instrument_id=instrument_id,
        )

    # -------------------------------------------------------------------------
    # Public API: Write Methods
    # -------------------------------------------------------------------------

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

        Delegates to FeaturePersistence component.

        Supports both the explicit-args signature and a backwards-compatible
        form where callers pass a FeatureData or list[FeatureData].

        Parameters
        ----------
        feature_set_id : str | None
            Feature set identifier (explicit mode)
        instrument_id : str | None
            Instrument identifier (explicit mode)
        features : dict[str, float] | None
            Feature name to value mapping (explicit mode)
        ts_event : int | None
            Event timestamp in nanoseconds (explicit mode)
        ts_init : int | None
            Initialization timestamp in nanoseconds (explicit mode)
        data : Any | None
            Backwards-compat: a FeatureData or list[FeatureData]
        publish_bus : bool, keyword-only, default True
            When True and publishing is enabled, publish a summary payload.

        """
        # Backwards compatibility: support write_features([FeatureData])
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
            self._persistence.write_batch(data=batch_data, publish_bus=publish_bus)
            return

        # Explicit-args mode
        if feature_set_id is None or instrument_id is None or features is None or ts_event is None:
            raise TypeError(
                "write_features requires explicit arguments or a FeatureData batch",
            )

        ts_init_val = int(ts_init) if ts_init is not None else int(ts_event)

        self._persistence.write_features(
            feature_set_id=feature_set_id,
            instrument_id=instrument_id,
            ts_event=int(ts_event),
            ts_init=ts_init_val,
            features=features,
            publish_bus=publish_bus,
        )

    def write_batch(self, data: list[object]) -> None:
        """
        Write a batch of FeatureData rows.

        Delegates to FeaturePersistence component.

        Parameters
        ----------
        data : list[FeatureData]
            Rows to upsert. Accepts objects with attributes
            feature_set_id, instrument_id, ts_event, ts_init, feature_values.

        """
        if not data:
            return

        # Append to buffer for visibility during the call (tests assert
        # the buffer is cleared after write_batch returns)
        from typing import cast

        from ml.stores.base import FeatureData as _FeatureData

        self._write_buffer.extend(cast(list[_FeatureData], data))

        self._persistence.write_batch(data=data, publish_bus=True)

        # Clear buffer after successful write
        self._write_buffer.clear()

    def store_features(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible alias for write_features.

        Accepts minimal explicit args used in integration tests: instrument_id,
        ts_event, and features. Fills feature_set_id from current pipeline/config
        and ts_init with ts_event when not provided.

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

        from typing import cast

        self.write_features(
            feature_set_id=self._versioning.get_feature_set_id(),
            instrument_id=str(instrument_id),
            features=features,
            ts_event=int(cast(int, ts_event)),
            ts_init=int(cast(int, ts_init)),
        )

    # -------------------------------------------------------------------------
    # Public API: Management Methods
    # -------------------------------------------------------------------------

    def clear_features(
        self,
        instrument_id: str | None = None,
        feature_version: str | None = None,
    ) -> None:
        """
        Clear stored features.

        Delegates to FeatureTableManager component.

        Parameters
        ----------
        instrument_id : str, optional
            Clear only for specific instrument.
        feature_version : str, optional
            Clear only specific version.

        """
        self._table_mgr.clear_features(
            instrument_id=instrument_id,
            feature_version=feature_version,
        )

    def flush(self) -> None:
        """
        Flush any pending writes to storage.

        Note: FeatureStore currently writes synchronously, so this is a no-op.
        Future versions may implement write buffering for performance.

        """
        # Currently a no-op as writes are synchronous

    def is_healthy(self) -> bool:
        """
        Check if the feature store is healthy and accessible.

        Returns
        -------
        bool
            True if store is healthy, False otherwise

        """
        try:
            # Try a simple query to verify connection
            with self.engine.connect() as conn:
                from sqlalchemy import text

                result = conn.execute(text("SELECT 1"))
                return result is not None
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Private/Internal API: Versioning and Config
    # -------------------------------------------------------------------------

    def _compute_config_hash(self) -> str:
        """
        Compute hash of feature configuration for versioning.

        Delegates to FeatureVersioning component.

        Returns
        -------
        str
            16-character hex hash of configuration.

        """
        return self._versioning.compute_config_hash()

    def _get_feature_set_id(self) -> str:
        """
        Derive a stable feature_set_id for storage.

        Delegates to FeatureVersioning component.

        Returns
        -------
        str
            Feature set identifier.

        """
        return self._versioning.get_feature_set_id()

    def _get_feature_names(self) -> list[str]:
        """
        Get OFFLINE feature names from pipeline or config.

        Delegates to FeatureVersioning component.

        Returns
        -------
        list[str]
            Offline feature names.

        """
        return self._versioning.get_feature_names(online=False)

    def _get_feature_names_online(self) -> list[str]:
        """
        Get ONLINE (hot-path) feature names with L1_ONLY gating.

        Delegates to FeatureVersioning component.

        Returns
        -------
        list[str]
            Online feature names.

        """
        return self._versioning.get_feature_names(online=True)

    # -------------------------------------------------------------------------
    # Test Hooks and Compatibility
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_ts_ns(ts_value: int) -> tuple[int, bool]:
        """
        Delegate to centralized timestamp normalization utility.

        Parameters
        ----------
        ts_value : int
            Timestamp value to normalize.

        Returns
        -------
        tuple[int, bool]
            Normalized timestamp and whether it was modified.

        """
        from ml.common.timestamps import normalize_timestamp_ns

        return normalize_timestamp_ns(ts_value)

    def _store_to_postgres(self, *args: Any, **kwargs: Any) -> None:
        """
        Store computed features (placeholder for test monkeypatching).

        Tests may monkeypatch this method. In production, storage is handled directly by
        FeaturePersistence component.

        """
        return None

    def _execute_write(self, row: dict[str, Any]) -> None:
        """
        Upsert a single feature row (patchable in tests).

        Delegates to FeaturePersistence component.

        Parameters
        ----------
        row : dict[str, Any]
            Row data to write.

        """
        self._persistence._execute_write(row)

    def _execute_query(self, sql: str) -> list[Any]:
        """
        Execute a SQL query and return rows (patchable in tests).

        Parameters
        ----------
        sql : str
            SQL query to execute.

        Returns
        -------
        list[Any]
            Query results.

        """
        from sqlalchemy import text as _text

        with self.engine.connect() as conn:
            result = conn.execute(_text(sql))
            try:
                return [dict(row) for row in result.mappings().all()]
            except Exception:
                return list(result.fetchall())

    def _get_connection(self) -> Any:
        """
        Return a connection context manager (patchable in tests).

        Returns
        -------
        Any
            Database connection context manager.

        """
        return self.engine.connect()


def create_engine(connection_string: str, **kwargs: Any) -> Engine:
    """
    Return the shared SQLAlchemy engine for ``connection_string``.

    This helper mirrors the historical module-level function used throughout the
    tests. It simply delegates to :class:`~ml.core.db_engine.EngineManager` so
    all store modules share the same engine cache.

    Args:
        connection_string: Database URL (e.g. ``postgresql://...``).
        **kwargs: Optional SQLAlchemy engine configuration overrides.

    Returns:
        Engine: The shared SQLAlchemy engine instance.
    """
    return EngineManager.get_engine(connection_string, **kwargs)


# During type checking, expose the legacy type to satisfy existing annotations.
if TYPE_CHECKING:
    from ml.stores.feature_store_legacy import FeatureStoreLegacy as FeatureStore
elif USE_COMPONENT_FEATURE_STORE:
    FeatureStore = ComponentFeatureStore
else:
    from ml.stores.feature_store_legacy import FeatureStoreLegacy

    FeatureStore = FeatureStoreLegacy


__all__ = [
    "ComponentFeatureStore",
    "EngineManager",
    "FeatureStore",
    "FeatureStoreLegacy",
    "create_engine",
]
