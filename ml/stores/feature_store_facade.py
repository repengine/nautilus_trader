#!/usr/bin/env python3

"""
FeatureStore facade integrating all 6 decomposed components.

This module provides the final facade layer for Phase 3.7.7, wiring together:
- FeatureWriterComponent (Phase 3.7.1) - Write operations with circuit breaker
- FeatureReaderComponent (Phase 3.7.2) - Read and training data operations
- FeatureComputationComponent (Phase 3.7.3) - Real-time and historical computation
- FeatureSchemaComponent (Phase 3.7.4) - Table setup and config hashing
- FeatureEventComponent (Phase 3.7.5) - Event emission and DataRegistry integration
- FeatureHealthComponent (Phase 3.7.6) - Health checks and clearing operations

The facade maintains 100% backward compatibility with the legacy FeatureStore API
while delegating all operations to specialized components.

Phase 3.7.7 - Final Facade Integration

"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from ml.common.db_utils import get_or_create_engine
from ml.config.base import MLFeatureConfig
from ml.core.db_engine import EngineManager
from ml.features import FeatureConfig
from ml.features import FeatureEngineer
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.stores.common.feature_computation import FeatureComputationComponent
from ml.stores.common.feature_computation import FeatureComputationConfig
from ml.stores.common.feature_event import FeatureEventComponent
from ml.stores.common.feature_event import FeatureEventConfig
from ml.stores.common.feature_health import FeatureHealthComponent
from ml.stores.common.feature_health import FeatureHealthConfig
from ml.stores.common.feature_reader import FeatureReaderComponent
from ml.stores.common.feature_reader import FeatureReaderConfig
from ml.stores.common.feature_schema import FeatureSchemaComponent
from ml.stores.common.feature_schema import FeatureSchemaConfig
from ml.stores.common.feature_writer import FeatureWriterComponent
from ml.stores.common.feature_writer import FeatureWriterConfig
from ml.stores.common.feature_writer import MessagePublisherProtocol
from ml.stores.mixins import DataRegistryMixin


if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from nautilus_trader.model.data import Bar
    from sqlalchemy import MetaData
    from sqlalchemy import Table
    from sqlalchemy.engine import Engine

    from ml.features import IndicatorManager
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.protocols import CircuitBreakerProtocol
    from ml.stores.services.cross_asset_service import CrossAssetFeatureService


logger = logging.getLogger(__name__)


class FeatureStoreFacade(DataRegistryMixin):
    """
    Thin facade that wires all 6 FeatureStore components together.

    Preserves the exact public API of legacy FeatureStore while
    delegating to decomposed components.

    Component Delegation:
    - Write operations -> FeatureWriterComponent
    - Read operations -> FeatureReaderComponent
    - Computation -> FeatureComputationComponent
    - Schema/setup -> FeatureSchemaComponent
    - Events/registry -> FeatureEventComponent
    - Health/clear/flush -> FeatureHealthComponent

    All 5 Universal ML Architecture Patterns enforced:
    1. 4-Store + 4-Registry Integration (via components)
    2. Protocol-First Interface Design (component protocols)
    3. Hot/Cold Path Separation (maintained from components)
    4. Progressive Fallback Chains (via components)
    5. Centralized Metrics Bootstrap (all components use metrics_bootstrap)

    Example:
        >>> store = FeatureStoreFacade(
        ...     connection_string="postgresql://...",
        ...     feature_config=FeatureConfig(),
        ... )
        >>> features = store.compute_realtime(bar, store=True)
        >>> training_data = store.get_training_data(
        ...     instrument_id="SPY.DATABENTO",
        ...     start=datetime(2024, 1, 1),
        ...     end=datetime(2024, 1, 2),
        ... )

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
        # Accept extra kwargs for compatibility
        **_: Any,
    ) -> None:
        """
        Initialize the feature store facade.

        Parameters
        ----------
        connection_string : str
            PostgreSQL connection string (same as Nautilus uses).
            Example: "postgresql://postgres:postgres@localhost:5432/nautilus"
        feature_config : FeatureConfig | MLFeatureConfig | None
            Configuration for feature engineering.
        pipeline_spec : PipelineSpec | None
            Pipeline specification for feature computation.
        persistence_manager : object | None
            Optional persistence/session provider (used by tests for mocking).
        enable_publishing : bool, default False
            When True, publish store events to the optional message bus.
        publisher : MessagePublisherProtocol | None
            Publisher implementation used when `enable_publishing` is True.
        publish_mode : {"batch", "row", "both"}, default "batch"
            Controls whether to publish batch summaries, per-row events, or both.

        """
        # Store connection string and create engine
        self.connection_string = connection_string
        self.engine: Engine = get_or_create_engine(connection_string)

        # Normalize feature_config
        self.feature_config = self._normalize_feature_config(feature_config)
        self.pipeline_spec = pipeline_spec

        # Store persistence manager (mock-friendly)
        self.persistence: object | None = persistence_manager

        # Create FeatureEngineer
        self.feature_engineer = FeatureEngineer(self.feature_config)

        # Initialize pipeline runners (before schema component)
        self.pipeline_runner_offline: PipelineRunner | None = None
        self.pipeline_runner_online: PipelineRunner | None = None
        self.pipeline_hash: str = ""
        self._init_pipeline_runners_early()

        # Initialize schema component (must be after pipeline runners)
        self._schema_component = FeatureSchemaComponent(
            engine=self.engine,
            feature_config=self.feature_config,
            pipeline_spec=pipeline_spec,
            pipeline_runner_offline=self.pipeline_runner_offline,
            pipeline_runner_online=self.pipeline_runner_online,
            pipeline_hash=self.pipeline_hash,
            config=FeatureSchemaConfig(),
        )
        self._schema_component.set_feature_engineer(self.feature_engineer)

        # Setup tables via schema component
        self.feature_values_table: Table = self._schema_component.setup_tables()
        self.metadata: MetaData = self._schema_component.metadata

        # Re-sync pipeline_hash from schema component
        self.pipeline_hash = self._schema_component.pipeline_hash

        # Resolve message bus config (topic scheme + prefix)
        try:
            from ml.config.bus import MessageBusConfig

            bus_config = MessageBusConfig.from_env()
            topic_scheme = str(bus_config.scheme)
            topic_prefix = str(bus_config.topic_prefix)
        except Exception:  # pragma: no cover - defensive fallback
            topic_scheme = "domain_op"
            topic_prefix = "events.ml"

        # Initialize writer component
        self._writer_config = FeatureWriterConfig(
            enable_publishing=enable_publishing,
            publish_mode=publish_mode,
            topic_scheme=topic_scheme,
            topic_prefix=topic_prefix,
        )
        self._writer_component = FeatureWriterComponent(
            engine=self.engine,
            table=self.feature_values_table,
            get_feature_set_id=self._get_feature_set_id,
            publisher=publisher,
            config=self._writer_config,
        )

        # Store the original component execute method and redirect through facade
        # This enables mocking store._execute_write in tests (backward compat)
        self._component_execute_write = self._writer_component._execute_write
        self._writer_component._execute_write = self._facade_execute_write_redirect  # type: ignore[method-assign]

        # Initialize reader component
        self._reader_component = FeatureReaderComponent(
            engine=self.engine,
            table=self.feature_values_table,
            get_feature_set_id=self._get_feature_set_id,
            get_feature_names=self._get_feature_names,
            config=FeatureReaderConfig(),
            persistence=persistence_manager,
        )

        # Initialize health component
        self._health_component = FeatureHealthComponent(
            engine=self.engine,
            table=self.feature_values_table,
            config=FeatureHealthConfig(),
        )

        # Initialize event component
        self._event_component = FeatureEventComponent(
            config=FeatureEventConfig(),
            get_registry=self._get_data_registry,
            get_feature_set_id=self._get_feature_set_id,
        )

        # Initialize computation component (last, depends on others)
        self._computation_component = FeatureComputationComponent(
            engine=self.engine,
            table=self.feature_values_table,
            feature_engineer=self.feature_engineer,
            feature_writer=self._writer_component,
            feature_reader=self._reader_component,
            get_feature_set_id=self._get_feature_set_id,
            get_feature_names=self._get_feature_names,
            get_feature_names_online=self._get_feature_names_online,
            feature_config=self.feature_config,
            config=FeatureComputationConfig(),
        )

        # Internal indicator managers (fallback for online computation)
        self._indicator_managers: dict[str, IndicatorManager] = {}

        # Lightweight write buffer for backward compatibility
        from ml.stores.base import FeatureData as _FeatureData

        self._write_buffer: list[_FeatureData] = []
        self._buffer: list[_FeatureData] = self._write_buffer  # Back-compat alias

        # Data registry (shared via mixin)
        self._data_registry: RegistryProtocol | None = None

        # Circuit breaker (injected by actors/services)
        self._circuit_breaker: CircuitBreakerProtocol | None = None

        # Store publisher reference for attribute access
        self.publisher = publisher
        self._enable_publishing = enable_publishing
        self._publish_mode = publish_mode

        # Message bus config (topic scheme and prefix)
        self._topic_scheme = topic_scheme
        self._topic_prefix = topic_prefix

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _normalize_feature_config(
        self,
        feature_config: FeatureConfig | MLFeatureConfig | None,
    ) -> FeatureConfig:
        """
        Normalize feature config to FeatureConfig type.

        Args:
            feature_config: Config to normalize

        Returns:
            Normalized FeatureConfig instance

        """
        if isinstance(feature_config, FeatureConfig):
            return feature_config
        if isinstance(feature_config, MLFeatureConfig):
            try:
                import msgspec as _msgspec

                return FeatureConfig(**_msgspec.to_builtins(feature_config))
            except Exception:
                return FeatureConfig(**getattr(feature_config, "__dict__", {}))
        return FeatureConfig()

    def _init_pipeline_runners_early(self) -> None:
        """
        Initialize pipeline runners from pipeline_spec (early phase).

        This is called before _schema_component is created, so it computes
        the pipeline hash directly without depending on the schema component.

        """
        if self.pipeline_spec:
            from ml.registry.base import DataRequirements

            # Offline (batch/teacher): allow L1_L2 to include microstructure
            self.pipeline_runner_offline = PipelineRunner(
                self.pipeline_spec,
                DataRequirements.L1_L2,
            )
            # Online (student/runtime): limit to L1 until actors are available
            self.pipeline_runner_online = PipelineRunner(
                self.pipeline_spec,
                DataRequirements.L1_ONLY,
            )
            self.pipeline_hash = self.pipeline_runner_offline.compute_signature()
        else:
            self.pipeline_runner_offline = None
            self.pipeline_runner_online = None
            # Compute hash directly without schema component
            self.pipeline_hash = self._compute_config_hash_early()

    def _compute_config_hash_early(self) -> str:
        """
        Compute hash of feature configuration (early phase, before schema component).

        Returns:
            SHA256 hash string (first 16 characters)

        """
        import hashlib
        import json

        if self.feature_config is None:
            return hashlib.sha256(b"{}").hexdigest()[:16]

        # Handle both dict-like and msgspec Struct objects
        if hasattr(self.feature_config, "__dict__"):
            config_dict = self.feature_config.__dict__
        else:
            try:
                import msgspec
                config_dict = msgspec.to_builtins(self.feature_config)
            except Exception:
                config_dict = {}

        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_feature_set_id(self) -> str:
        """
        Derive a stable feature_set_id for storage.

        Delegates to schema component.

        Returns:
            Feature set identifier string

        """
        return self._schema_component.get_feature_set_id()

    def _get_feature_names(self) -> list[str]:
        """
        Get OFFLINE feature names from pipeline or config.

        Delegates to schema component.

        Returns:
            List of feature name strings

        """
        return self._schema_component.get_feature_names()

    def _get_feature_names_online(self) -> list[str]:
        """
        Get ONLINE (hot-path) feature names with L1_ONLY gating.

        Delegates to schema component.

        Returns:
            List of online feature name strings

        """
        return self._schema_component.get_feature_names_online()

    def _compute_config_hash(self) -> str:
        """
        Compute hash of feature configuration for versioning.

        Delegates to schema component.

        Returns:
            SHA256 hash string (first 16 characters)

        """
        return self._schema_component.compute_config_hash()

    @staticmethod
    def _normalize_ts_ns(ts_value: int) -> tuple[int, bool]:
        """
        Delegate to centralized timestamp normalization utility.

        Args:
            ts_value: Timestamp value to normalize

        Returns:
            Tuple of (normalized_value, was_normalized)

        """
        return FeatureSchemaComponent.normalize_ts_ns(ts_value)

    # =========================================================================
    # DataRegistry Integration (from mixin)
    # =========================================================================

    def _get_data_registry(self) -> RegistryProtocol | None:
        """
        Delegate initialization to shared mixin.

        Returns:
            The data registry or None

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
        # Also update computation component
        self._computation_component.set_data_registry(registry)
        # Update event component getter
        self._event_component.set_registry_getter(lambda: registry)

    # =========================================================================
    # Write Operations (delegate to FeatureWriterComponent)
    # =========================================================================

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
        self._writer_component.write_features(
            feature_set_id=feature_set_id,
            instrument_id=instrument_id,
            features=features,
            ts_event=ts_event,
            ts_init=ts_init,
            data=data,
            publish_bus=publish_bus,
        )

    def write_batch(self, data: list[object]) -> None:
        """
        Write a batch of FeatureData rows (compat shim).

        Parameters
        ----------
        data : list[FeatureData]
            Rows to upsert. Accepts objects with attributes
            feature_set_id, instrument_id, ts_event, ts_init, feature_values.

        """
        self._writer_component.write_batch(data)

    def store_features(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible alias for write_features with relaxed argument requirements.

        Accepts minimal explicit args used in integration tests: instrument_id,
        ts_event, and features.

        """
        self._writer_component.store_features(*args, **kwargs)

    # =========================================================================
    # Read Operations (delegate to FeatureReaderComponent)
    # =========================================================================

    def get_training_data(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        include_bars: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
        """
        Load features for training.

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
        return self._reader_component.get_training_data(
            instrument_id=instrument_id,
            start=start,
            end=end,
            include_bars=include_bars,
        )

    def get_latest_at_or_before(
        self,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return the latest feature row at or before the given timestamp.

        HOT PATH: P99 < 5ms requirement.

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
        return self._reader_component.get_latest_at_or_before(
            instrument_id=instrument_id,
            ts_event=ts_event,
        )

    def read_range(
        self,
        start_ns: int,
        end_ns: int,
        instrument_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Read features in a time range (inclusive start, exclusive end).

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
        return self._reader_component.read_range(
            start_ns=start_ns,
            end_ns=end_ns,
            instrument_id=instrument_id,
        )

    # =========================================================================
    # Computation Operations (delegate to FeatureComputationComponent)
    # =========================================================================

    def compute_realtime(
        self,
        bar: Bar,
        store: bool = True,
        indicator_manager: IndicatorManager | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Compute features for real-time inference.

        HOT PATH: P99 < 5ms requirement.

        Uses the SAME FeatureEngineer as historical computation to ensure
        perfect parity between training and inference.

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
        return self._computation_component.compute_realtime(
            bar=bar,
            store=store,
            indicator_manager=indicator_manager,
        )

    def compute_and_store_historical(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
        force_recompute: bool = False,
    ) -> int:
        """
        Compute and store features for historical data.

        This method:
        1. Queries bars from Nautilus PostgreSQL tables
        2. Computes features using FeatureEngineer (same logic as live)
        3. Stores features in ml_feature_values table

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
        return self._computation_component.compute_and_store_historical(
            instrument_id=instrument_id,
            start=start,
            end=end,
            force_recompute=force_recompute,
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
        return self._computation_component.compute_historical_parallel(
            instrument_ids=instrument_ids,
            start=start,
            end=end,
            force_recompute=force_recompute,
            max_workers=max_workers,
        )

    # =========================================================================
    # Health Operations (delegate to FeatureHealthComponent)
    # =========================================================================

    def is_healthy(self) -> bool:
        """
        Check if the feature store is healthy and accessible.

        Returns
        -------
        bool
            True if store is healthy, False otherwise

        """
        return self._health_component.is_healthy()

    def clear_features(
        self,
        instrument_id: str | None = None,
        feature_version: str | None = None,
    ) -> None:
        """
        Clear stored features.

        Parameters
        ----------
        instrument_id : str, optional
            Clear only for specific instrument.
        feature_version : str, optional
            Clear only specific version.

        """
        self._health_component.clear_features(
            instrument_id=instrument_id,
            feature_version=feature_version,
        )

    def flush(self) -> None:
        """
        Flush any pending writes to storage.

        Note: FeatureStore currently writes synchronously, so this is a no-op.
        Future versions may implement write buffering for performance.

        """
        self._health_component.flush()

    # =========================================================================
    # Legacy Compatibility Methods (patchable hooks)
    # =========================================================================

    def _setup_tables(self) -> None:
        """
        Reflect or create ml_feature_values table.

        Legacy compatibility hook - delegates to schema component.

        """
        self.feature_values_table = self._schema_component.setup_tables()

    def _store_to_postgres(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """
        Store computed features (placeholder).

        Tests may monkeypatch this method. In production, storage is handled
        directly in compute_realtime/compute_and_store_historical.

        """
        return None

    def _facade_execute_write_redirect(self, row: dict[str, Any]) -> None:
        """
        Redirect from component to facade's _execute_write.

        This method is assigned to the component's _execute_write to enable
        mocking store._execute_write in tests while still affecting the
        component's internal calls.

        """
        self._execute_write(row)

    def _execute_write(self, row: dict[str, Any]) -> None:  # pragma: no cover
        """
        Upsert a single feature row (patchable in tests).

        Calls the original component execute method. When this method is mocked
        in tests, the mock will be called instead of the real SQL execution.

        """
        self._component_execute_write(row)

    def _execute_query(self, sql: str) -> list[Any]:  # pragma: no cover (test hook)
        """
        Execute a SQL query and return rows (patchable).

        """
        from sqlalchemy import text as _text

        with self.engine.connect() as conn:
            result = conn.execute(_text(sql))
            try:
                return [dict(row) for row in result.mappings().all()]
            except Exception:
                return list(result.fetchall())

    def _get_connection(self) -> Any:  # pragma: no cover (test hook for patching)
        """
        Return a connection context manager (patchable in tests).

        """
        return self.engine.connect()

    def _features_exist(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """
        Check if features already exist for the given range.

        Delegates to reader component.

        """
        return self._reader_component.features_exist(
            instrument_id=instrument_id,
            start=start,
            end=end,
        )

    def _load_bars_from_nautilus(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
    ) -> Any:
        """
        Load bars from Nautilus PostgreSQL tables.

        Delegates to computation component.

        """
        return self._computation_component._load_bars_from_nautilus(
            instrument_id=instrument_id,
            start=start,
            end=end,
        )

    # =========================================================================
    # Event Emission (delegate to FeatureEventComponent)
    # =========================================================================

    def _emit_historical_event(
        self,
        instrument_id: str,
        timestamps: npt.NDArray[np.int64],
        row_count: int,
    ) -> None:
        """
        Emit FEATURE_COMPUTED event for historical computation.

        Non-blocking operation.

        """
        self._event_component.emit_historical_event(
            instrument_id=instrument_id,
            timestamps=timestamps,
            row_count=row_count,
        )

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

        """
        self._event_component.record_observability_stage_boundary(
            stage=stage,
            instrument_id=instrument_id,
            ts_stage_start=ts_stage_start,
            ts_stage_end=ts_stage_end,
            row_count=row_count,
        )

    # =========================================================================
    # Cross-Asset Service (lazy-initialized)
    # =========================================================================

    @property
    def cross_asset(self) -> CrossAssetFeatureService:
        """
        Access the cross-asset feature service for beta, spread, and correlation storage.

        Lazy-initialized on first access.

        Returns
        -------
        CrossAssetFeatureService
            Service for cross-asset relationship features.

        Example:
            >>> store = FeatureStoreFacade(connection_string="postgresql://...")
            >>> store.cross_asset.write_beta(
            ...     asset_id="AAPL",
            ...     benchmark_id="SPY",
            ...     ts_event=1234567890000000000,
            ...     ts_init=1234567890000000000,
            ...     beta=1.25,
            ...     lookback_periods=60,
            ...     ewma_span=30,
            ... )

        """
        if not hasattr(self, "_cross_asset_service"):
            from ml.stores.services.cross_asset_service import CrossAssetFeatureService

            self._cross_asset_service = CrossAssetFeatureService(deps=self)
        return self._cross_asset_service


# Module-level delegation function for EngineManager integration
def create_engine(connection_string: str) -> Any:
    """
    Create database engine delegating to EngineManager.

    This function ensures all stores share the same engine pool,
    preventing connection exhaustion in parallel tests.

    Parameters
    ----------
    connection_string : str
        Database connection string

    Returns
    -------
    Engine
        SQLAlchemy engine instance

    """
    return EngineManager.get_engine(connection_string)


# Backwards-compatible alias for facade-only deployment.
FeatureStore = FeatureStoreFacade


__all__ = [
    "FeatureStore",
    "FeatureStoreFacade",
    "create_engine",
]
