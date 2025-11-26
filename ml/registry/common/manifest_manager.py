#!/usr/bin/env python3

"""
ManifestManagerComponent - Handles dataset manifest CRUD operations.

This component is extracted from the DataRegistry god class following the
established TDD decomposition pattern. It handles:
- Dataset registration
- Manifest updates
- Dataset deprecation
- Manifest listing and retrieval
- Contract creation from manifests

Thread-safety: Uses the persistence component's lock for thread-safe operations.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType


if TYPE_CHECKING:
    from ml.registry.common.data_persistence import DataPersistenceComponent
    from ml.registry.common.event_emission import EventEmissionComponent


logger = logging.getLogger(__name__)


class ManifestManagerComponent:
    """
    Handles dataset manifest CRUD operations.

    This component manages the registration, update, deprecation, and retrieval
    of dataset manifests in the data registry.

    Attributes
    ----------
    persistence : DataPersistenceComponent
        The persistence component for storage operations.

    Thread Safety
    -------------
    All operations use the persistence component's lock for thread-safe access.

    Example
    -------
    >>> from ml.registry.common.data_persistence import DataPersistenceComponent
    >>> persistence = DataPersistenceComponent(config, path)
    >>> manager = ManifestManagerComponent(persistence)
    >>> dataset_id = manager.register_dataset(manifest)
    """

    def __init__(
        self,
        persistence: DataPersistenceComponent,
        event_emitter: EventEmissionComponent | None = None,
    ) -> None:
        """
        Initialize manifest manager component.

        Parameters
        ----------
        persistence : DataPersistenceComponent
            The persistence component for storage operations.
        event_emitter : EventEmissionComponent | None
            Optional event emitter for emitting CATALOG_WRITTEN events.
        """
        self._persistence = persistence
        self._event_emitter = event_emitter

    # -------------------------------------------------------------------------
    # Dataset Registration
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

        Example
        -------
        >>> dataset_id = manager.register_dataset(manifest)
        >>> print(f"Registered: {dataset_id}")
        """
        with self._persistence._lock:
            if manifest.dataset_id in self._persistence._manifests:
                raise ValueError(f"Dataset '{manifest.dataset_id}' already exists")

            # Store manifest
            self._persistence._manifests[manifest.dataset_id] = manifest
            self._persistence._save_registry()

            # Auto-create contract for earnings datasets
            if manifest.dataset_id in {"ml.earnings_actuals", "ml.earnings_estimates"}:
                contract = self._create_contract_from_manifest(manifest)
                self._persistence._contracts[manifest.dataset_id] = contract
                self._persistence._save_registry()

            # Log audit event
            self._persistence.persistence.log_audit(
                entity_type="dataset",
                entity_id=manifest.dataset_id,
                action="register",
                changes={"manifest": self._persistence._manifest_to_dict(manifest)},
            )

            # Emit CATALOG_WRITTEN event (if event emitter is available)
            self._emit_catalog_event(
                dataset_id=manifest.dataset_id,
                run_id="registry_register",
                metadata={"action": "register"},
            )

            logger.info(
                "Registered dataset '%s' version %s",
                manifest.dataset_id,
                manifest.version,
            )
            return manifest.dataset_id

    # -------------------------------------------------------------------------
    # Manifest Updates
    # -------------------------------------------------------------------------

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

        Example
        -------
        >>> manager.update_manifest("my_dataset", {"retention_days": 180})
        """
        with self._persistence._lock:
            if dataset_id not in self._persistence._manifests:
                raise ValueError(f"Dataset '{dataset_id}' not found")

            manifest = self._persistence._manifests[dataset_id]

            # Create new manifest with updates
            manifest_dict = self._persistence._manifest_to_dict(manifest)

            # Validate that at least one valid field is being updated
            valid_fields = {
                "location", "partitioning", "retention_days", "schema",
                "ts_field", "seq_field", "primary_keys", "schema_hash",
                "constraints", "lineage", "pipeline_signature", "version",
                "metadata",
            }

            if not any(k in valid_fields for k in changes):
                raise ValueError("No valid fields to update")

            manifest_dict.update(changes)
            from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

            manifest_dict["last_modified"] = _sanitize(
                int(time.time_ns()),
                context="registry.update_manifest:json.last_modified",
            )

            # Convert back to manifest object
            updated_manifest = self._persistence._dict_to_manifest(manifest_dict)
            self._persistence._manifests[dataset_id] = updated_manifest

            self._persistence._save_registry()

            # Log audit event
            self._persistence.persistence.log_audit(
                entity_type="dataset",
                entity_id=dataset_id,
                action="update",
                changes=changes,
            )

            logger.info(
                "Updated dataset '%s' with changes: %s",
                dataset_id,
                list(changes.keys()),
            )

    # -------------------------------------------------------------------------
    # Dataset Deprecation
    # -------------------------------------------------------------------------

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

        Example
        -------
        >>> manager.deprecate("old_dataset")
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        self.update_manifest(
            dataset_id,
            {
                "metadata": {
                    "deprecated": True,
                    "deprecated_at": _sanitize(
                        int(time.time_ns()),
                        context="registry.deprecate:deprecated_at",
                    ),
                },
            },
        )

        # Emit CATALOG_WRITTEN event with deprecated metadata
        self._emit_catalog_event(
            dataset_id=dataset_id,
            run_id="registry_deprecate",
            metadata={"action": "deprecate", "deprecated": True},
        )

        logger.info("Deprecated dataset '%s'", dataset_id)

    # -------------------------------------------------------------------------
    # Manifest Listing and Retrieval
    # -------------------------------------------------------------------------

    def list_manifests(self) -> list[DatasetManifest]:
        """
        Return all dataset manifests known to the registry.

        Returns
        -------
        list[DatasetManifest]
            List of all manifests sorted by dataset_id.
        """
        with self._persistence._lock:
            return [
                self._persistence._manifests[dataset_id]
                for dataset_id in sorted(self._persistence._manifests)
            ]

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

        Example
        -------
        >>> manifest = manager.get_manifest("my_dataset")
        >>> print(manifest.dataset_type)
        """
        with self._persistence._lock:
            if dataset_id not in self._persistence._manifests:
                raise ValueError(f"Dataset '{dataset_id}' not found")
            return self._persistence._manifests[dataset_id]

    # -------------------------------------------------------------------------
    # Contract Management
    # -------------------------------------------------------------------------

    def _create_contract_from_manifest(self, manifest: DatasetManifest) -> DataContract:
        """
        Create a data contract from a dataset manifest.

        Parameters
        ----------
        manifest : DatasetManifest
            Manifest to create contract from.

        Returns
        -------
        DataContract
            Created contract with validation rules.
        """
        rules = []
        constraints = manifest.constraints or {}

        # Convert constraints to validation rules
        if "ranges" in constraints:
            for field, range_spec in constraints["ranges"].items():
                if "min" in range_spec or "max" in range_spec:
                    rules.append(
                        ValidationRule(
                            rule_type=ValidationRuleType.RANGE,
                            field_name=field,
                            parameters=range_spec,
                            severity=QualityFlag.FAIL,
                            description=f"Range validation for {field}",
                        ),
                    )

        if "nullability" in constraints:
            for field, nullable in constraints["nullability"].items():
                if not nullable:
                    rules.append(
                        ValidationRule(
                            rule_type=ValidationRuleType.NULLABILITY,
                            field_name=field,
                            parameters={"nullable": False},
                            severity=QualityFlag.FAIL,
                            description=f"{field} cannot be null",
                        ),
                    )

        # Regex constraints
        if "regex" in constraints:
            for field, pattern in constraints["regex"].items():
                if isinstance(pattern, str) and pattern:
                    rules.append(
                        ValidationRule(
                            rule_type=ValidationRuleType.REGEX,
                            field_name=str(field),
                            parameters={"pattern": pattern},
                            severity=QualityFlag.FAIL,
                            description=f"Regex validation for {field}",
                        ),
                    )

        # Per-dataset null-rate threshold
        quality_thresholds: dict[str, float] = {}
        try:
            thr = constraints.get("null_rate_threshold")
            if isinstance(thr, (int, float)) and 0.0 <= float(thr) <= 1.0:
                quality_thresholds["null_rate"] = float(thr)
        except Exception:
            quality_thresholds = {}

        # Create default rule if no rules defined
        if not rules:
            rules.append(
                ValidationRule(
                    rule_type=ValidationRuleType.TYPE_CHECK,
                    field_name="*",
                    parameters={},
                    severity=QualityFlag.WARN,
                    description="Type validation for all fields",
                ),
            )

        return DataContract(
            contract_id=f"{manifest.dataset_id}_contract",
            dataset_id=manifest.dataset_id,
            version="1.0.0",
            validation_rules=rules,
            quality_thresholds=quality_thresholds,
            enforcement_mode="strict",
            created_at=manifest.created_at,
            last_modified=manifest.last_modified,
        )

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

        Example
        -------
        >>> contract = manager.get_contract("my_dataset")
        >>> print(contract.enforcement_mode)
        """
        with self._persistence._lock:
            if dataset_id not in self._persistence._contracts:
                # Create a default contract from manifest if not exists
                manifest = self.get_manifest(dataset_id)
                contract = self._create_contract_from_manifest(manifest)
                self._persistence._contracts[dataset_id] = contract
                return contract
            return self._persistence._contracts[dataset_id]

    # -------------------------------------------------------------------------
    # Event Emission Helper
    # -------------------------------------------------------------------------

    def _emit_catalog_event(
        self,
        dataset_id: str,
        run_id: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """
        Emit a CATALOG_WRITTEN event for manifest operations.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier.
        run_id : str
            Run identifier for the event.
        metadata : dict[str, object] | None
            Additional metadata for the event.
        """
        # Build event
        event = {
            "dataset_id": dataset_id,
            "instrument_id": "*",
            "stage": Stage.CATALOG_WRITTEN.value,
            "source": Source.HISTORICAL.value,
            "run_id": run_id,
            "ts_min": 0,
            "ts_max": 0,
            "ts_event": 0,
            "count": 0,
            "status": EventStatus.SUCCESS.value,
            "error": None,
            "created_at": time.time(),
            "metadata": metadata or {},
        }

        # Store in persistence's events list
        self._persistence._events.append(event)

        # Trim old events to prevent unbounded growth (keep last 10000)
        if len(self._persistence._events) > 10000:
            self._persistence._events = self._persistence._events[-10000:]

        # Save immediately
        self._persistence._save_registry(immediate=True)

        # Also use event emitter if available
        if self._event_emitter is not None:
            try:
                self._event_emitter.emit_event(
                    dataset_id=dataset_id,
                    instrument_id="*",
                    stage=Stage.CATALOG_WRITTEN,
                    source=Source.HISTORICAL,
                    run_id=run_id,
                    ts_min=0,
                    ts_max=0,
                    count=0,
                    status=EventStatus.SUCCESS,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.debug("Failed to emit event via emitter: %s", exc)
