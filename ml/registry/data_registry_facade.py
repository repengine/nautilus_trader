#!/usr/bin/env python3

"""
DataRegistryFacade - Unified interface for data registry operations.

This facade wires all 5 components extracted from the DataRegistry god class:
- DataPersistenceComponent
- ManifestManagerComponent
- EventEmissionComponent
- WatermarkManagerComponent
- LineageTrackerComponent

Thread-safety: All operations are thread-safe via component locks.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from ml.common.protocols import MLComponentMixin
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.common.data_persistence import DataPersistenceComponent
from ml.registry.common.event_emission import EventEmissionComponent
from ml.registry.common.lineage_tracker import LineageTrackerComponent
from ml.registry.common.manifest_manager import ManifestManagerComponent
from ml.registry.common.watermark_manager import WatermarkManagerComponent
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetLineageRecord
from ml.registry.dataclasses import DatasetManifest
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.watermark import Watermark


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class DataRegistryFacade(MLComponentMixin):
    """
    Unified facade for data registry operations.

    This facade delegates to specialized components for different
    registry operations while maintaining backward compatibility
    with the legacy DataRegistry API.

    Attributes
    ----------
    registry_path : Path
        Directory path for registry storage.
    batch_save_interval : float
        Seconds to wait before flushing batch saves.

    Thread Safety
    -------------
    All public methods are thread-safe through component locks.

    Example
    -------
    >>> config = PersistenceConfig(backend=BackendType.JSON, json_path=Path("/tmp/registry"))
    >>> facade = DataRegistryFacade(
    ...     registry_path=Path("/tmp/registry"),
    ...     persistence_config=config,
    ... )
    >>> dataset_id = facade.register_dataset(manifest)
    """

    def __init__(
        self,
        registry_path: Path,
        batch_save_interval: float = 0.1,
        persistence_config: PersistenceConfig | None = None,
    ) -> None:
        """
        Initialize data registry facade with configurable persistence backend.

        Parameters
        ----------
        registry_path : Path
            Directory path for registry storage (used for JSON backend).
        batch_save_interval : float
            Seconds to wait before flushing batch saves (default 0.1s).
        persistence_config : PersistenceConfig | None
            Persistence configuration. If None, defaults to JSON backend.
        """
        self.registry_path = registry_path
        self.batch_save_interval = batch_save_interval

        # Setup persistence
        if persistence_config is None:
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )

        # Initialize components
        self._persistence = DataPersistenceComponent(
            registry_path=registry_path,
            persistence_config=persistence_config,
            batch_save_interval=batch_save_interval,
        )
        self._manifest_manager = ManifestManagerComponent(self._persistence)
        self._event_emission = EventEmissionComponent(self._persistence)
        self._watermark_manager = WatermarkManagerComponent(self._persistence)
        self._lineage_tracker = LineageTrackerComponent(self._persistence)

        # Store reference for backward compatibility
        self.persistence = self._persistence.persistence
        self.backend = persistence_config.backend

        logger.info(
            "Initialized DataRegistryFacade at %s with backend=%s, batch_save_interval=%ss",
            registry_path,
            self.backend.value,
            batch_save_interval,
        )

    # -------------------------------------------------------------------------
    # Manifest Operations (delegate to ManifestManagerComponent)
    # -------------------------------------------------------------------------

    def register_dataset(self, manifest: DatasetManifest) -> str:
        """
        Register a new dataset manifest.

        Parameters
        ----------
        manifest : DatasetManifest
            Dataset manifest to register.

        Returns
        -------
        str
            Dataset ID of the registered dataset.

        Raises
        ------
        ValueError
            If dataset with same ID already exists.
        """
        dataset_id = self._manifest_manager.register_dataset(manifest)

        # Emit ops event (catalog written) with correlation id
        try:
            from ml.common.correlation import make_correlation_id

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

        return dataset_id

    def update_manifest(self, dataset_id: str, changes: dict[str, Any]) -> None:
        """
        Update an existing dataset manifest.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to update.
        changes : dict[str, Any]
            Dictionary of fields to update.

        Raises
        ------
        ValueError
            If dataset doesn't exist or changes are invalid.
        """
        self._manifest_manager.update_manifest(dataset_id, changes)

        # Emit ops event (catalog written)
        try:
            from ml.common.correlation import make_correlation_id

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

    def deprecate(self, dataset_id: str) -> None:
        """
        Mark a dataset as deprecated.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to deprecate.

        Raises
        ------
        ValueError
            If dataset doesn't exist.
        """
        self._manifest_manager.deprecate(dataset_id)

        # Emit ops event (deprecated)
        try:
            from ml.common.correlation import make_correlation_id

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

    def list_manifests(self) -> list[DatasetManifest]:
        """
        Return all dataset manifests known to the registry.

        Returns
        -------
        list[DatasetManifest]
            List of all manifests sorted by dataset_id.
        """
        return self._manifest_manager.list_manifests()

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        """
        Get a dataset manifest by ID.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to retrieve.

        Returns
        -------
        DatasetManifest
            The dataset manifest.

        Raises
        ------
        ValueError
            If dataset doesn't exist.
        """
        return self._manifest_manager.get_manifest(dataset_id)

    def get_contract(self, dataset_id: str) -> DataContract:
        """
        Get the data contract for a dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset ID to get contract for.

        Returns
        -------
        DataContract
            The data contract.

        Raises
        ------
        ValueError
            If dataset or contract doesn't exist.
        """
        return self._manifest_manager.get_contract(dataset_id)

    # -------------------------------------------------------------------------
    # Event Operations (delegate to EventEmissionComponent)
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

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        instrument_id : str
            Instrument identifier.
        stage : Stage
            Processing stage.
        source : Source
            Data source.
        run_id : str
            Unique identifier for this processing run.
        ts_min : int
            Minimum timestamp in nanoseconds.
        ts_max : int
            Maximum timestamp in nanoseconds.
        count : int
            Number of records processed.
        status : EventStatus
            Processing status.
        error : str | None
            Error message if status is failed.
        metadata : dict[str, object] | None
            Additional metadata.
        """
        self._event_emission.emit_event(
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
        )

    # -------------------------------------------------------------------------
    # Watermark Operations (delegate to WatermarkManagerComponent)
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

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        instrument_id : str
            Instrument identifier.
        source : Source
            Data source.
        last_success_ns : int
            Last successful processing timestamp in nanoseconds.
        count : int
            Count from last successful processing.
        completeness_pct : float
            Percentage of expected data received (0-100).
        """
        self._watermark_manager.update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source,
            last_success_ns=last_success_ns,
            count=count,
            completeness_pct=completeness_pct,
        )

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

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        instrument_id : str
            Instrument identifier.
        source : Source | str
            Data source.

        Returns
        -------
        Watermark | None
            The watermark if exists, None otherwise.
        """
        return self._watermark_manager.get_watermark(dataset_id, instrument_id, source)

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

        Parameters
        ----------
        dataset_id : str | None
            Filter by dataset ID.
        instrument_id : str | None
            Filter by instrument ID.
        source : Source | str | None
            Filter by data source.
        limit : int | None
            Maximum number of watermarks to return.

        Yields
        ------
        Watermark
            Matching watermarks.
        """
        yield from self._watermark_manager.iter_watermarks(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source,
            limit=limit,
        )

    # -------------------------------------------------------------------------
    # Lineage Operations (delegate to LineageTrackerComponent)
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

        Parameters
        ----------
        child_dataset_id : str
            Child dataset ID.
        parent_ids : list[str]
            List of parent dataset IDs.
        transform_id : str
            Unique identifier for this transformation.
        ts_range : dict[str, int]
            Time range with 'start_ns' and 'end_ns' keys.
        params : dict[str, Any]
            Transformation parameters used.
        """
        self._lineage_tracker.link_lineage(
            child_dataset_id=child_dataset_id,
            parent_ids=parent_ids,
            transform_id=transform_id,
            ts_range=ts_range,
            params=params,
        )

    def iter_lineage(
        self,
        *,
        child: str | None = None,
        parent: str | None = None,
        limit: int | None = None,
    ) -> Iterator[DatasetLineageRecord]:
        """
        Yield lineage records filtered by optional child or parent identifiers.

        Parameters
        ----------
        child : str | None
            Filter by child dataset ID.
        parent : str | None
            Filter by parent dataset ID.
        limit : int | None
            Maximum number of records to return.

        Yields
        ------
        DatasetLineageRecord
            Matching lineage records.
        """
        yield from self._lineage_tracker.iter_lineage(
            child=child,
            parent=parent,
            limit=limit,
        )

    def get_pipeline_signature(self, dataset_id: str) -> str | None:
        """
        Get the pipeline signature for a dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.

        Returns
        -------
        str | None
            Pipeline signature or None if not found.
        """
        return self._lineage_tracker.get_pipeline_signature(dataset_id)

    def set_pipeline_signature(self, dataset_id: str, signature: str) -> None:
        """
        Set the pipeline signature for a dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        signature : str
            Pipeline signature (SHA256 hex digest).

        Raises
        ------
        ValueError
            If signature is empty or dataset doesn't exist.
        """
        self._lineage_tracker.set_pipeline_signature(dataset_id, signature)

    # -------------------------------------------------------------------------
    # Persistence Operations
    # -------------------------------------------------------------------------

    def flush(self) -> None:
        """
        Persist any pending batched changes immediately.
        """
        self._persistence.flush()

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            if hasattr(self, "_persistence"):
                self._persistence.flush()
        except Exception as exc:
            logger.debug("DataRegistryFacade cleanup failed: %s", exc)


def create_data_registry(
    registry_path: Path,
    batch_save_interval: float = 0.1,
    persistence_config: PersistenceConfig | None = None,
) -> DataRegistryFacade:
    """
    Factory function to create a data registry.

    Parameters
    ----------
    registry_path : Path
        Directory path for registry storage.
    batch_save_interval : float
        Seconds to wait before flushing batch saves.
    persistence_config : PersistenceConfig | None
        Persistence configuration.

    Returns
    -------
    DataRegistryFacade
        The data registry instance.
    """
    logger.info("Using DataRegistryFacade")
    return DataRegistryFacade(
        registry_path=registry_path,
        batch_save_interval=batch_save_interval,
        persistence_config=persistence_config,
    )
