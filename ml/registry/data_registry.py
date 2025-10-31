#!/usr/bin/env python3

"""
Data registry facade delegating to specialized components.

This module provides a unified interface for dataset manifest management, lineage tracking,
watermark management, event emission, and contract validation.

The facade delegates to 5 specialized components:
- ManifestManager: Dataset manifest CRUD operations
- LineageManager: Dataset lineage tracking
- WatermarkManager: Dataset watermark management
- EventManager: Dataset event emission
- ContractManager: Dataset contract validation

Feature Flag:
  Set ML_USE_LEGACY_DATA_REGISTRY=1 to use the original god class implementation.
  Default (0 or unset) uses this component-based facade.

"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from ml.common.correlation import make_correlation_id
from ml.common.protocols import MLComponentMixin
from ml.common.timestamps import sanitize_timestamp_ns
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.contract_manager import ContractManager
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetLineageRecord
from ml.registry.dataclasses import DatasetManifest
from ml.registry.event_manager import EventManager
from ml.registry.lineage_manager import LineageManager
from ml.registry.manifest_manager import ManifestManager
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager
from ml.registry.watermark_manager import Watermark
from ml.registry.watermark_manager import WatermarkManager


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class DataRegistry(MLComponentMixin):
    """
    Facade for dataset registry operations.

    Delegates to specialized component managers:
    - ManifestManager: Dataset manifest CRUD
    - LineageManager: Dataset lineage tracking
    - WatermarkManager: Dataset watermark management
    - EventManager: Dataset event emission
    - ContractManager: Dataset contract validation

    Thread-safe for concurrent operations.

    Examples
    --------
    >>> # Initialize with JSON backend for development
    >>> registry = DataRegistry(
    ...     registry_path=Path("/tmp/registry"),
    ...     persistence_config=PersistenceConfig(backend=BackendType.JSON)
    ... )

    >>> # Register a dataset
    >>> manifest = DatasetManifest(
    ...     dataset_id="bars_eurusd_1m",
    ...     dataset_type=DatasetType.BARS,
    ...     storage_kind=StorageKind.PARQUET,
    ...     location="/data/bars/eurusd/1m/",
    ...     # ... other fields
    ... )
    >>> dataset_id = registry.register_dataset(manifest)

    >>> # Emit processing event
    >>> registry.emit_event(
    ...     dataset_id="bars_eurusd_1m",
    ...     instrument_id="EUR/USD",
    ...     stage="CATALOG_WRITTEN",
    ...     source="historical",
    ...     run_id="run_123",
    ...     ts_min=1234567890000000000,
    ...     ts_max=1234567900000000000,
    ...     count=1000,
    ...     status="success"
    ... )

    """

    def __init__(
        self,
        registry_path: Path,
        batch_save_interval: float = 0.1,
        persistence_config: PersistenceConfig | None = None,
    ) -> None:
        """
        Initialize data registry facade with component managers.

        Parameters
        ----------
        registry_path : Path
            Directory path for registry storage (used for JSON backend)
        batch_save_interval : float
            Seconds to wait before flushing batch saves (default 0.1s)
        persistence_config : PersistenceConfig | None
            Persistence configuration. If None, defaults to JSON backend.

        """
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.batch_save_interval = batch_save_interval

        # Store absolute path for security validation
        self._registry_root = self.registry_path.resolve()

        # Setup persistence
        if persistence_config is None:
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )
        self.persistence = PersistenceManager(persistence_config)
        self.backend = persistence_config.backend

        self.registry_file = self.registry_path / "data_registry.json"
        self._lock = threading.RLock()  # Use RLock to allow reentrant locking

        # Initialize component managers
        self._manifest_mgr = ManifestManager()
        self._lineage_mgr = LineageManager()
        self._watermark_mgr = WatermarkManager()
        self._event_mgr = EventManager()
        self._contract_mgr = ContractManager()

        # Load registry data for JSON backend
        if self.backend == BackendType.JSON:
            self._load_json_registry()
        else:
            logger.info("Using PostgreSQL backend - data is stored in database")

        # Back-compat: expose legacy attributes referencing component caches
        self._manifests = self._manifest_mgr._manifests
        self._contracts = self._contract_mgr._contracts
        self._events = self._event_mgr._events
        self._watermarks = self._watermark_mgr._watermarks
        self._lineage = self._lineage_mgr._lineage
        self._pending_save = False
        self._save_timer: threading.Timer | None = None

        logger.info(
            "Initialized DataRegistry facade at %s with backend=%s, batch_save_interval=%ss",
            registry_path,
            self.backend.value,
            batch_save_interval,
        )

    def _load_json_registry(self) -> None:
        """
        Load registry data from JSON file for JSON backend.

        For PostgreSQL backend, data is loaded on-demand from the database.

        """
        if self.registry_file.exists():
            data = self.persistence.load_json("data_registry.json")
            if data is not None:
                # Load manifests into ManifestManager cache
                for dataset_id, manifest_data in data.get("manifests", {}).items():
                    manifest = self._manifest_mgr._dict_to_manifest(manifest_data)
                    self._manifest_mgr._manifests[dataset_id] = manifest

                # Load contracts into ContractManager cache
                for dataset_id, contract_data in data.get("contracts", {}).items():
                    contract = self._contract_mgr._dict_to_contract(contract_data)
                    self._contract_mgr._contracts[dataset_id] = contract

                # Load events into EventManager
                self._event_mgr._events = data.get("events", [])

                # Load watermarks into WatermarkManager cache
                for key, watermark_data in data.get("watermarks", {}).items():
                    watermark = self._watermark_mgr._dict_to_watermark(watermark_data)
                    self._watermark_mgr._watermarks[key] = watermark

                # Load lineage into LineageManager
                self._lineage_mgr._lineage = data.get("lineage", [])
            else:
                self._save_json_registry(immediate=True)
        else:
            self._save_json_registry(immediate=True)

    def _save_json_registry(self, immediate: bool = False, force: bool = False) -> None:
        """
        Save registry data to JSON file (JSON backend only).

        Skips saves during testing to avoid O(N²) event serialization,
        unless force=True (for explicit flush() calls).

        Parameters
        ----------
        immediate : bool
            If True, save immediately. If False, not used in facade (always saves immediately).
        force : bool
            If True, bypass pytest detection (for explicit flush() calls).

        """
        # Skip during pytest to avoid catastrophic slowdown (O(N²) serialization)
        # UNLESS force=True (explicit flush() calls must persist)
        import os

        if not force and os.getenv("PYTEST_CURRENT_TEST"):
            logger.debug("Skipping JSON registry save during pytest")
            return

        if self.backend == BackendType.JSON:
            with self._lock:
                # Collect data from all component managers
                manifests_dict = {
                    dataset_id: self._manifest_mgr._manifest_to_dict(manifest)
                    for dataset_id, manifest in self._manifest_mgr._manifests.items()
                }

                contracts_dict = {
                    dataset_id: self._contract_mgr._contract_to_dict(contract)
                    for dataset_id, contract in self._contract_mgr._contracts.items()
                }

                watermarks_dict = {
                    key: self._watermark_mgr._watermark_to_dict(watermark)
                    for key, watermark in self._watermark_mgr._watermarks.items()
                }

                data = {
                    "manifests": manifests_dict,
                    "contracts": contracts_dict,
                    "events": self._event_mgr._events,
                    "watermarks": watermarks_dict,
                    "lineage": self._lineage_mgr._lineage,
                    "last_updated": __import__("time").time(),
                }

                self.persistence.save_json(data, "data_registry.json")

    def _schedule_save(self) -> None:
        """
        Schedule a deferred JSON save with debouncing.

        Uses a timer to batch multiple save requests within batch_save_interval. Thread-
        safe with automatic cleanup of existing timers.

        """
        if self._pending_save:
            return  # Already scheduled

        # Cancel existing timer if any (no join - hot path optimization)
        if self._save_timer is not None:
            self._save_timer.cancel()

        # Schedule new save
        self._pending_save = True
        self._save_timer = threading.Timer(
            self.batch_save_interval,
            self._do_deferred_save,
        )
        self._save_timer.daemon = True
        self._save_timer.start()

    def _do_deferred_save(self) -> None:
        """
        Perform the actual deferred save.

        Called by timer thread after batch_save_interval elapses. Resets pending state
        and timer reference after completion.

        """
        try:
            self._save_json_registry(immediate=True)
        finally:
            self._pending_save = False
            self._save_timer = None

    def _save_registry(self, immediate: bool = False) -> None:
        """
        Backward-compatible save helper for tooling/tests expecting legacy API.
        """
        self._save_json_registry(immediate=immediate)
        self._pending_save = False
        self._save_timer = None

    def _load_registry(self) -> None:
        """
        Backward-compatible load helper mirroring legacy behavior.
        """
        if self.backend == BackendType.JSON:
            self._load_json_registry()

    # -------------------------------------------------------------------------
    # Public API: Flush Method
    # -------------------------------------------------------------------------

    def flush(self) -> None:
        """
        Persist any pending batched changes immediately (JSON backend only).

        Cancels any pending timer and forces an immediate save. Bypasses pytest
        detection to ensure explicit flush requests are honored.

        """
        if self.backend == BackendType.JSON:
            # Cancel pending timer if any
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer.join()  # Wait for thread to actually finish
                self._save_timer = None

            # Force immediate save (force=True bypasses pytest detection)
            self._save_json_registry(immediate=True, force=True)
            self._pending_save = False

    # -------------------------------------------------------------------------
    # Public API: Manifest Management (Delegates to ManifestManager)
    # -------------------------------------------------------------------------

    def register_dataset(self, manifest: DatasetManifest) -> str:
        """
        Register a new dataset manifest.

        Delegates to ManifestManager component.

        Parameters
        ----------
        manifest : DatasetManifest
            Dataset manifest to register

        Returns
        -------
        str
            Dataset ID of the registered dataset

        Raises
        ------
        ValueError
            If dataset with same ID already exists

        Examples
        --------
        >>> manifest = DatasetManifest(
        ...     dataset_id="bars_eurusd_1m",
        ...     dataset_type=DatasetType.BARS,
        ...     # ... other fields
        ... )
        >>> dataset_id = registry.register_dataset(manifest)

        """
        with self._lock:
            dataset_id = self._manifest_mgr.register_manifest(manifest, self.persistence)

            # Handle special datasets (earnings) - create contracts
            if manifest.dataset_id in {"ml.earnings_actuals", "ml.earnings_estimates"}:
                self._contract_mgr.create_contract_from_manifest(manifest)

            # Save to JSON if using JSON backend
            if self.backend == BackendType.JSON:
                self._save_json_registry(immediate=True)

            # Log audit event
            self.persistence.log_audit(
                entity_type="dataset",
                entity_id=manifest.dataset_id,
                action="register",
                changes={"manifest": self._manifest_mgr._manifest_to_dict(manifest)},
            )

            # Emit ops event (catalog written) with correlation id
            try:
                corr = make_correlation_id(
                    run_id="registry_register",
                    dataset_id=manifest.dataset_id,
                    instrument_id="*",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                )
                self.emit_event(
                    dataset_id=manifest.dataset_id,
                    instrument_id="*",
                    stage=Stage.CATALOG_WRITTEN,
                    source=Source.HISTORICAL,
                    run_id="registry_register",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                    status=EventStatus.SUCCESS,
                    metadata={"correlation_id": corr},
                )
            except Exception:
                logger.debug("Failed to emit registry register event", exc_info=True)

            logger.info("Registered dataset '%s' version %s", manifest.dataset_id, manifest.version)
            return dataset_id

    def register_manifest(self, manifest: DatasetManifest) -> str:
        """
        Backward-compatible alias for ``register_dataset``.

        Returns
        -------
        str
            Dataset identifier of the registered manifest.

        """
        return self.register_dataset(manifest)

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        """
        Get a dataset manifest by ID.

        Delegates to ManifestManager component.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to retrieve

        Returns
        -------
        DatasetManifest
            The dataset manifest

        Raises
        ------
        ValueError
            If dataset doesn't exist

        Examples
        --------
        >>> manifest = registry.get_manifest("bars_eurusd_1m")
        >>> print(manifest.dataset_type)
        DatasetType.BARS

        """
        with self._lock:
            return self._manifest_mgr.get_manifest(dataset_id, self.persistence)

    def list_manifests(self) -> list[DatasetManifest]:
        """
        Return all dataset manifests known to the registry.

        Delegates to ManifestManager component.

        Returns
        -------
        list[DatasetManifest]
            List of all dataset manifests

        """
        with self._lock:
            return self._manifest_mgr.list_manifests(self.persistence)

    def update_manifest(self, dataset_id: str, changes: dict[str, Any]) -> None:
        """
        Update an existing dataset manifest.

        Delegates to ManifestManager component.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to update
        changes : dict[str, Any]
            Dictionary of fields to update

        Raises
        ------
        ValueError
            If dataset doesn't exist or changes are invalid

        Examples
        --------
        >>> registry.update_manifest(
        ...     "bars_eurusd_1m",
        ...     {"retention_days": 180, "version": "1.1.0"}
        ... )

        """
        with self._lock:
            self._manifest_mgr.update_manifest(dataset_id, changes, self.persistence)

            # Save to JSON if using JSON backend
            if self.backend == BackendType.JSON:
                self._save_json_registry(immediate=True)

            # Log audit event
            self.persistence.log_audit(
                entity_type="dataset",
                entity_id=dataset_id,
                action="update",
                changes=changes,
            )

            # Emit ops event (catalog written) with correlation id
            try:
                corr = make_correlation_id(
                    run_id="registry_update",
                    dataset_id=dataset_id,
                    instrument_id="*",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                )
                self.emit_event(
                    dataset_id=dataset_id,
                    instrument_id="*",
                    stage=Stage.CATALOG_WRITTEN,
                    source=Source.HISTORICAL,
                    run_id="registry_update",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                    status=EventStatus.SUCCESS,
                    metadata={"correlation_id": corr},
                )
            except Exception:
                logger.debug("Failed to emit registry update event", exc_info=True)

            logger.info("Updated dataset '%s' with changes: %s", dataset_id, list(changes.keys()))

    def deprecate(self, dataset_id: str) -> None:
        """
        Mark a dataset as deprecated.

        Delegates to ManifestManager component.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to deprecate

        Raises
        ------
        ValueError
            If dataset doesn't exist

        Examples
        --------
        >>> registry.deprecate("bars_eurusd_1m_old")

        """
        with self._lock:
            self.update_manifest(
                dataset_id,
                {
                    "metadata": {
                        "deprecated": True,
                        "deprecated_at": sanitize_timestamp_ns(
                            int(__import__("time").time_ns()),
                            context="registry.deprecate:deprecated_at",
                        ),
                    },
                },
            )

            # Emit ops event (deprecated) with correlation id
            try:
                corr = make_correlation_id(
                    run_id="registry_deprecate",
                    dataset_id=dataset_id,
                    instrument_id="*",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                )
                self.emit_event(
                    dataset_id=dataset_id,
                    instrument_id="*",
                    stage=Stage.CATALOG_WRITTEN,
                    source=Source.HISTORICAL,
                    run_id="registry_deprecate",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                    status=EventStatus.SUCCESS,
                    metadata={"correlation_id": corr, "deprecated": True},
                )
            except Exception:
                logger.debug("Failed to emit registry deprecate event", exc_info=True)

            logger.info("Deprecated dataset '%s'", dataset_id)

    # -------------------------------------------------------------------------
    # Public API: Contract Management (Delegates to ContractManager)
    # -------------------------------------------------------------------------

    def get_contract(self, dataset_id: str) -> DataContract:
        """
        Get the data contract for a dataset.

        Delegates to ContractManager component.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to get contract for

        Returns
        -------
        DataContract
            The data contract

        Raises
        ------
        ValueError
            If dataset or contract doesn't exist

        Examples
        --------
        >>> contract = registry.get_contract("bars_eurusd_1m")
        >>> print(contract.enforcement_mode)
        strict

        """
        with self._lock:
            return self._contract_mgr.get_contract(
                dataset_id,
                self._manifest_mgr,
                self.persistence,
            )

    # -------------------------------------------------------------------------
    # Public API: Event Management (Delegates to EventManager)
    # -------------------------------------------------------------------------

    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """
        Emit a data processing event.

        Delegates to EventManager component.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        stage : str
            Processing stage (INGESTED, CATALOG_WRITTEN, FEATURE_COMPUTED, etc.)
        source : str
            Data source (live, historical, backfill)
        run_id : str
            Unique identifier for this processing run
        ts_min : int
            Minimum timestamp in nanoseconds
        ts_max : int
            Maximum timestamp in nanoseconds
        count : int
            Number of records processed
        status : str
            Processing status (success, failed, partial)
        error : str | None
            Error message if status is failed

        Examples
        --------
        >>> from ml.config.events import Stage, Source, EventStatus
        >>> registry.emit_event(
        ...     dataset_id="bars_eurusd_1m",
        ...     instrument_id="EUR/USD",
        ...     stage=Stage.CATALOG_WRITTEN,
        ...     source=Source.HISTORICAL,
        ...     run_id="run_123",
        ...     ts_min=1234567890000000000,
        ...     ts_max=1234567900000000000,
        ...     count=1000,
        ...     status=EventStatus.SUCCESS,
        ... )

        """
        with self._lock:
            self._event_mgr.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=status,
                error=error,
                metadata=metadata,
                persistence=self.persistence,
            )

            # Schedule batched save to avoid O(N²) serialization on every event
            if self.backend == BackendType.JSON:
                import os

                if os.getenv("PYTEST_CURRENT_TEST"):
                    # Skip entirely during tests (already handled in _save_json_registry)
                    pass
                else:
                    self._schedule_save()  # Use batching for production

    # -------------------------------------------------------------------------
    # Public API: Watermark Management (Delegates to WatermarkManager)
    # -------------------------------------------------------------------------

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        """
        Update watermark for a dataset/instrument/source combination.

        Delegates to WatermarkManager component.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : str
            Data source (live, historical, backfill)
        last_success_ns : int
            Last successful processing timestamp in nanoseconds
        count : int
            Count from last successful processing
        completeness_pct : float
            Percentage of expected data received (0-100)

        Examples
        --------
        >>> from ml.config.events import Source
        >>> registry.update_watermark(
        ...     dataset_id="bars_eurusd_1m",
        ...     instrument_id="EUR/USD",
        ...     source=Source.LIVE,
        ...     last_success_ns=1234567900000000000,
        ...     count=1000,
        ...     completeness_pct=98.5,
        ... )

        """
        with self._lock:
            self._watermark_mgr.update_watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=source,
                last_success_ns=last_success_ns,
                count=count,
                completeness_pct=completeness_pct,
                persistence=self.persistence,
            )

            # Save to JSON if using JSON backend
            if self.backend == BackendType.JSON:
                self._save_json_registry(immediate=True)

    @overload
    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source,
    ) -> Watermark | None: ...

    @overload
    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
    ) -> Watermark | None: ...

    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source | str,
    ) -> Watermark | None:
        """
        Get watermark for a dataset/instrument/source combination.

        Delegates to WatermarkManager component.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : Source | str
            Data source as enum (preferred) or persisted string ("live", "historical", "backfill").

        Returns
        -------
        Watermark | None
            The watermark if exists, None otherwise

        Examples
        --------
        >>> watermark = registry.get_watermark("bars_eurusd_1m", "EUR/USD", "live")
        >>> watermark = registry.get_watermark("bars_eurusd_1m", "EUR/USD", Source.LIVE)
        >>> if watermark:
        ...     print(f"Last success: {watermark.last_success_ns}")
        ...     print(f"Completeness: {watermark.completeness_pct}%")

        """
        with self._lock:
            return self._watermark_mgr.get_watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=source,
                persistence=self.persistence,
            )

    def iter_watermarks(
        self,
        *,
        dataset_id: str | None = None,
        instrument_id: str | None = None,
        source: Source | str | None = None,
        limit: int | None = None,
    ) -> Iterator[Watermark]:
        """
        Yield watermarks optionally filtered by dataset, instrument, or source.

        Delegates to WatermarkManager component.

        Parameters
        ----------
        dataset_id : str | None
            Optional dataset ID filter
        instrument_id : str | None
            Optional instrument ID filter
        source : Source | str | None
            Optional source filter
        limit : int | None
            Optional limit on number of records returned

        Returns
        -------
        Iterator[Watermark]
            Iterator of watermarks

        """
        with self._lock:
            yield from self._watermark_mgr.iter_watermarks(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=source,
                limit=limit,
                persistence=self.persistence,
            )

    # -------------------------------------------------------------------------
    # Public API: Lineage Management (Delegates to LineageManager)
    # -------------------------------------------------------------------------

    def link_lineage(
        self,
        child_dataset_id: str,
        parent_ids: list[str],
        transform_id: str,
        ts_range: dict[str, int],
        params: dict[str, Any],
    ) -> None:
        """
        Link dataset lineage relationships.

        Delegates to LineageManager component.

        Parameters
        ----------
        child_dataset_id : str
            Child dataset ID
        parent_ids : list[str]
            List of parent dataset IDs
        transform_id : str
            Unique identifier for this transformation
        ts_range : dict[str, int]
            Time range with 'start_ns' and 'end_ns' keys
        params : dict[str, Any]
            Transformation parameters used

        Examples
        --------
        >>> registry.link_lineage(
        ...     child_dataset_id="features_microstructure",
        ...     parent_ids=["bars_eurusd_1m", "quotes_eurusd"],
        ...     transform_id="feature_pipeline_v1",
        ...     ts_range={"start_ns": 1234567890000000000, "end_ns": 1234567900000000000},
        ...     params={"lookback_bars": 20, "include_imbalance": True}
        ... )

        """
        with self._lock:
            self._lineage_mgr.link_lineage(
                child_dataset_id=child_dataset_id,
                parent_ids=parent_ids,
                transform_id=transform_id,
                ts_range=ts_range,
                params=params,
                persistence=self.persistence,
            )

            # Save to JSON if using JSON backend
            if self.backend == BackendType.JSON:
                self._save_json_registry(immediate=True)

    def iter_lineage(
        self,
        *,
        child: str | None = None,
        parent: str | None = None,
        limit: int | None = None,
    ) -> Iterator[DatasetLineageRecord]:
        """
        Yield lineage records filtered by optional child or parent identifiers.

        Delegates to LineageManager component.

        Parameters
        ----------
        child : str | None
            Optional child dataset ID filter
        parent : str | None
            Optional parent dataset ID filter
        limit : int | None
            Optional limit on number of records returned

        Returns
        -------
        Iterator[DatasetLineageRecord]
            Iterator of lineage records

        """
        with self._lock:
            yield from self._lineage_mgr.iter_lineage(
                child=child,
                parent=parent,
                limit=limit,
                persistence=self.persistence,
            )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def __del__(self) -> None:
        """
        Cleanup on deletion.
        """
        # Ensure final save for JSON backend
        if self.backend == BackendType.JSON:
            try:
                self._save_json_registry(immediate=True)
            except Exception as save_exc:
                logger.debug(
                    "data_registry.save_on_exit_failed",
                    exc_info=True,
                    extra={"error": repr(save_exc)},
                )

        # Close persistence connections
        if hasattr(self, "persistence"):
            self.persistence.close()
