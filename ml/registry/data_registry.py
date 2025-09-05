#!/usr/bin/env python3

"""
Data registry with self-describing manifests, contracts, lineage tracking, and
watermarks.

This module provides a registry for dataset manifests with lifecycle management, data
contracts, lineage tracking, event recording, and watermark management. It supports both
JSON (for development) and PostgreSQL (for production) backends.

"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ml.common.correlation import make_correlation_id
from ml.common.protocols import MLComponentMixin
from ml.config.events import Stage
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)


# Watermark dataclass for tracking data processing progress
@dataclass(frozen=True)
class Watermark:
    """
    Watermark tracking data processing progress for a dataset.

    Attributes
    ----------
    dataset_id : str
        Dataset identifier
    instrument_id : str
        Instrument identifier
    source : str
        Data source ('live', 'historical', 'backfill')
    last_success_ns : int
        Last successful processing timestamp in nanoseconds
    last_attempt_ns : int
        Last attempted processing timestamp in nanoseconds
    last_count : int
        Count from last successful processing
    completeness_pct : float
        Percentage of expected data received (0-100)
    updated_at : float
        Unix timestamp of last update

    """

    dataset_id: str
    instrument_id: str
    source: str
    last_success_ns: int
    last_attempt_ns: int
    last_count: int
    completeness_pct: float
    updated_at: float


class DataRegistry(MLComponentMixin):
    """
    Registry for dataset manifests with configurable persistence backend.

    This registry manages dataset metadata, contracts, lineage relationships, processing
    events, and watermarks. It supports both JSON files (development) and PostgreSQL
    (production) for persistence.

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
        Initialize data registry with configurable persistence backend.

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

        # Batch save management
        self._pending_save = False
        self._save_timer: threading.Timer | None = None

        # Initialize or load registry
        self._load_registry()

        logger.info(
            "Initialized DataRegistry at %s with backend=%s, batch_save_interval=%ss",
            registry_path,
            self.backend.value,
            batch_save_interval,
        )

    def _load_registry(self) -> None:
        """
        Load registry from persistence backend or create new one.
        """
        if self.backend == BackendType.JSON:
            if self.registry_file.exists():
                data = self.persistence.load_json("data_registry.json")
                if data is not None:
                    # Load manifests
                    self._manifests: dict[str, DatasetManifest] = {}
                    for dataset_id, manifest_data in data.get("manifests", {}).items():
                        self._manifests[dataset_id] = self._dict_to_manifest(manifest_data)

                    # Load contracts
                    self._contracts: dict[str, DataContract] = {}
                    for dataset_id, contract_data in data.get("contracts", {}).items():
                        self._contracts[dataset_id] = self._dict_to_contract(contract_data)

                    # Load events, watermarks, and lineage
                    self._events: list[dict[str, Any]] = data.get("events", [])
                    self._watermarks: dict[str, Watermark] = {}
                    for key, watermark_data in data.get("watermarks", {}).items():
                        self._watermarks[key] = self._dict_to_watermark(watermark_data)
                    self._lineage: list[dict[str, Any]] = data.get("lineage", [])
                else:
                    self._initialize_empty_registry()
            else:
                self._initialize_empty_registry()
        elif self.backend == BackendType.POSTGRES:
            # For PostgreSQL, we query the database directly
            # The tables are created by the migration script
            self._initialize_empty_registry()
            logger.info("Using PostgreSQL backend - data is stored in database")

    def _initialize_empty_registry(self) -> None:
        """
        Initialize empty registry structures.
        """
        self._manifests = {}
        self._contracts = {}
        self._events = []
        self._watermarks = {}
        self._lineage = []
        if self.backend == BackendType.JSON:
            # Save immediately to ensure the backing file exists for callers/tests
            # which verify presence directly after initialization.
            self._save_registry(immediate=True)

    # Public flush for tests and tooling
    def flush(self) -> None:
        """
        Persist any pending batched changes immediately (JSON backend only).
        """
        if self.backend == BackendType.JSON:
            self._save_registry(immediate=True)

    def _dict_to_manifest(self, data: dict[str, Any]) -> DatasetManifest:
        """
        Convert dictionary to DatasetManifest.
        """
        # Convert string enum values back to enum types
        data["dataset_type"] = DatasetType(data["dataset_type"])
        data["storage_kind"] = StorageKind(data["storage_kind"])
        return DatasetManifest(**data)

    def _manifest_to_dict(self, manifest: DatasetManifest) -> dict[str, Any]:
        """
        Convert DatasetManifest to dictionary.
        """
        data = {
            "dataset_id": manifest.dataset_id,
            "dataset_type": manifest.dataset_type.value,
            "storage_kind": manifest.storage_kind.value,
            "location": manifest.location,
            "partitioning": manifest.partitioning,
            "retention_days": manifest.retention_days,
            "schema": manifest.schema,
            "ts_field": manifest.ts_field,
            "seq_field": manifest.seq_field,
            "primary_keys": manifest.primary_keys,
            "schema_hash": manifest.schema_hash,
            "constraints": manifest.constraints,
            "lineage": manifest.lineage,
            "pipeline_signature": manifest.pipeline_signature,
            "version": manifest.version,
            "created_at": manifest.created_at,
            "last_modified": manifest.last_modified,
            "metadata": manifest.metadata,
        }
        return data

    def _dict_to_contract(self, data: dict[str, Any]) -> DataContract:
        """
        Convert dictionary to DataContract.
        """
        from ml.registry.dataclasses import QualityFlag
        from ml.registry.dataclasses import ValidationRule
        from ml.registry.dataclasses import ValidationRuleType

        # Convert validation rules
        rules = []
        for rule_data in data.get("validation_rules", []):
            rule_data["rule_type"] = ValidationRuleType(rule_data["rule_type"])
            rule_data["severity"] = QualityFlag(rule_data["severity"])
            rules.append(ValidationRule(**rule_data))

        data["validation_rules"] = rules
        return DataContract(**data)

    def _contract_to_dict(self, contract: DataContract) -> dict[str, Any]:
        """
        Convert DataContract to dictionary.
        """
        rules = []
        for rule in contract.validation_rules:
            rules.append(
                {
                    "rule_type": rule.rule_type.value,
                    "field_name": rule.field_name,
                    "parameters": rule.parameters,
                    "severity": rule.severity.value,
                    "description": rule.description,
                }
            )

        return {
            "contract_id": contract.contract_id,
            "dataset_id": contract.dataset_id,
            "version": contract.version,
            "validation_rules": rules,
            "quality_thresholds": contract.quality_thresholds,
            "enforcement_mode": contract.enforcement_mode,
            "created_at": contract.created_at,
            "last_modified": contract.last_modified,
            "metadata": contract.metadata,
        }

    def _dict_to_watermark(self, data: dict[str, Any]) -> Watermark:
        """
        Convert dictionary to Watermark.
        """
        return Watermark(**data)

    def _watermark_to_dict(self, watermark: Watermark) -> dict[str, Any]:
        """
        Convert Watermark to dictionary.
        """
        return {
            "dataset_id": watermark.dataset_id,
            "instrument_id": watermark.instrument_id,
            "source": watermark.source,
            "last_success_ns": watermark.last_success_ns,
            "last_attempt_ns": watermark.last_attempt_ns,
            "last_count": watermark.last_count,
            "completeness_pct": watermark.completeness_pct,
            "updated_at": watermark.updated_at,
        }

    def _save_registry(self, immediate: bool = False) -> None:
        """
        Save registry to disk with optional batching.

        Parameters
        ----------
        immediate : bool
            If True, save immediately. If False, batch the save.

        """
        with self._lock:
            # For very small batch intervals (e.g., tests), flush immediately
            if immediate or (self.backend == BackendType.JSON and self.batch_save_interval <= 0.02):
                # Cancel any pending batch save
                if self._save_timer is not None:
                    self._save_timer.cancel()
                    self._save_timer = None
                self._pending_save = False

                # Save immediately
                self._do_save()
            else:
                # Schedule batch save if not already pending
                if not self._pending_save:
                    self._pending_save = True

                    # Cancel existing timer if any
                    if self._save_timer is not None:
                        self._save_timer.cancel()

                    # Schedule new save
                    self._save_timer = threading.Timer(
                        self.batch_save_interval,
                        self._do_save,
                    )
                    self._save_timer.start()

    def _do_save(self) -> None:
        """
        Perform the actual save operation.
        """
        with self._lock:
            if self.backend == BackendType.JSON:
                # Convert all data to serializable format
                manifests_dict = {
                    dataset_id: self._manifest_to_dict(manifest)
                    for dataset_id, manifest in self._manifests.items()
                }

                contracts_dict = {
                    dataset_id: self._contract_to_dict(contract)
                    for dataset_id, contract in self._contracts.items()
                }

                watermarks_dict = {
                    key: self._watermark_to_dict(watermark)
                    for key, watermark in self._watermarks.items()
                }

                data = {
                    "manifests": manifests_dict,
                    "contracts": contracts_dict,
                    "events": self._events,
                    "watermarks": watermarks_dict,
                    "lineage": self._lineage,
                    "last_updated": time.time(),
                }

                self.persistence.save_json(data, "data_registry.json")

            self._pending_save = False
            self._save_timer = None

    def register_dataset(self, manifest: DatasetManifest) -> str:
        """
        Register a new dataset manifest.

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
            if manifest.dataset_id in self._manifests:
                raise ValueError(f"Dataset '{manifest.dataset_id}' already exists")

            # Store in appropriate backend
            if self.backend == BackendType.JSON:
                self._manifests[manifest.dataset_id] = manifest
                self._save_registry()
            elif self.backend == BackendType.POSTGRES:
                session = self.persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    # Execute SQL function to register dataset
                    query = text(
                        """
                        INSERT INTO ml_dataset_registry
                        (dataset_id, name, version, dataset_type, storage_kind, location,
                         partitioning, retention_days, schema, schema_hash, constraints,
                         parents, pipeline_signature, metadata)
                        VALUES
                        (:dataset_id, :name, :version, :dataset_type, :storage_kind, :location,
                         :partitioning, :retention_days, :schema, :schema_hash, :constraints,
                         :parents, :pipeline_signature, :metadata)
                    """
                    )

                    session.execute(
                        query,
                        {
                            "dataset_id": manifest.dataset_id,
                            "name": manifest.metadata.get("name", manifest.dataset_id),
                            "version": manifest.version,
                            "dataset_type": manifest.dataset_type.value,
                            "storage_kind": manifest.storage_kind.value,
                            "location": manifest.location,
                            "partitioning": json.dumps(manifest.partitioning),
                            "retention_days": manifest.retention_days,
                            "schema": json.dumps(manifest.schema),
                            "schema_hash": manifest.schema_hash,
                            "constraints": json.dumps(manifest.constraints),
                            "parents": json.dumps(manifest.lineage),
                            "pipeline_signature": manifest.pipeline_signature,
                            "metadata": json.dumps(manifest.metadata),
                        },
                    )
                    session.commit()

                    # Cache locally
                    self._manifests[manifest.dataset_id] = manifest

                except Exception as e:
                    session.rollback()
                    logger.error("Failed to register dataset: %s", e)
                    raise
                finally:
                    session.close()

            # Log audit event
            self.persistence.log_audit(
                entity_type="dataset",
                entity_id=manifest.dataset_id,
                action="register",
                changes={"manifest": self._manifest_to_dict(manifest)},
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
                    stage=Stage.CATALOG_WRITTEN.value,
                    source="historical",
                    run_id="registry_register",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                    status="success",
                    metadata={"correlation_id": corr},
                )
            except Exception:
                logger.debug("Failed to emit registry register event", exc_info=True)

            logger.info("Registered dataset '%s' version %s", manifest.dataset_id, manifest.version)
            return manifest.dataset_id

    def update_manifest(self, dataset_id: str, changes: dict[str, Any]) -> None:
        """
        Update an existing dataset manifest.

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
            if self.backend == BackendType.JSON:
                if dataset_id not in self._manifests:
                    raise ValueError(f"Dataset '{dataset_id}' not found")

                manifest = self._manifests[dataset_id]

                # Create new manifest with updates
                manifest_dict = self._manifest_to_dict(manifest)
                manifest_dict.update(changes)
                manifest_dict["last_modified"] = time.time_ns()

                # Convert back to manifest object
                updated_manifest = self._dict_to_manifest(manifest_dict)
                self._manifests[dataset_id] = updated_manifest

                self._save_registry()

            elif self.backend == BackendType.POSTGRES:
                session = self.persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    # Build UPDATE query safely with all possible fields
                    # This avoids dynamic SQL construction
                    all_fields = [
                        "name",
                        "version",
                        "dataset_type",
                        "storage_kind",
                        "location",
                        "partitioning",
                        "retention_days",
                        "schema",
                        "schema_hash",
                        "constraints",
                        "parents",
                        "pipeline_signature",
                        "metadata",
                    ]

                    # Build the update dict
                    update_data = {"dataset_id": dataset_id}
                    set_parts = []

                    for field in all_fields:
                        if field in changes:
                            value = changes[field]
                            if field in [
                                "partitioning",
                                "schema",
                                "constraints",
                                "metadata",
                                "parents",
                            ]:
                                update_data[field] = (
                                    json.dumps(value) if value is not None else "{}"
                                )
                            else:
                                update_data[field] = value
                            set_parts.append(f"{field} = :{field}")

                    if not set_parts:
                        raise ValueError("No valid fields to update")

                    # Safe query with parameterized values
                    query = text(
                        f"""
                        UPDATE ml_dataset_registry
                        SET {', '.join(set_parts)}, last_modified = NOW()
                        WHERE dataset_id = :dataset_id
                    """
                    )

                    result = session.execute(query, update_data)
                    # Check if any rows were affected
                    row_count = getattr(result, "rowcount", None)
                    if row_count is not None and row_count == 0:
                        raise ValueError(f"Dataset '{dataset_id}' not found")

                    session.commit()

                    # Update cache if present
                    if dataset_id in self._manifests:
                        manifest_dict = self._manifest_to_dict(self._manifests[dataset_id])
                        manifest_dict.update(changes)
                        manifest_dict["last_modified"] = time.time_ns()
                        self._manifests[dataset_id] = self._dict_to_manifest(manifest_dict)

                except Exception as e:
                    session.rollback()
                    logger.error("Failed to update dataset: %s", e)
                    raise
                finally:
                    session.close()

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
                    stage=Stage.CATALOG_WRITTEN.value,
                    source="historical",
                    run_id="registry_update",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                    status="success",
                    metadata={"correlation_id": corr},
                )
            except Exception:
                logger.debug("Failed to emit registry update event", exc_info=True)

            logger.info("Updated dataset '%s' with changes: %s", dataset_id, list(changes.keys()))

    def deprecate(self, dataset_id: str) -> None:
        """
        Mark a dataset as deprecated.

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
                {"metadata": {"deprecated": True, "deprecated_at": time.time_ns()}},
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
                    stage=Stage.CATALOG_WRITTEN.value,
                    source="historical",
                    run_id="registry_deprecate",
                    ts_min=0,
                    ts_max=0,
                    count=0,
                    status="deprecated",
                    metadata={"correlation_id": corr},
                )
            except Exception:
                logger.debug("Failed to emit registry deprecate event", exc_info=True)

            logger.info("Deprecated dataset '%s'", dataset_id)

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        """
        Get a dataset manifest by ID.

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
            if self.backend == BackendType.JSON:
                if dataset_id not in self._manifests:
                    raise ValueError(f"Dataset '{dataset_id}' not found")
                return self._manifests[dataset_id]

            elif self.backend == BackendType.POSTGRES:
                # Check cache first
                if dataset_id in self._manifests:
                    return self._manifests[dataset_id]

                session = self.persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    query = text(
                        """
                        SELECT dataset_id, dataset_type, storage_kind, location,
                               partitioning, retention_days, schema, schema_hash,
                               constraints, parents as lineage, pipeline_signature,
                               version, EXTRACT(EPOCH FROM created_at) * 1000000000 as created_at,
                               EXTRACT(EPOCH FROM last_modified) * 1000000000 as last_modified,
                               metadata
                        FROM ml_dataset_registry
                        WHERE dataset_id = :dataset_id
                    """
                    )

                    result = session.execute(query, {"dataset_id": dataset_id}).fetchone()
                    if result is None:
                        raise ValueError(f"Dataset '{dataset_id}' not found")

                    # Convert to manifest
                    manifest_data = dict(result)
                    manifest_data["seq_field"] = manifest_data.get("metadata", {}).get("seq_field")
                    manifest_data["ts_field"] = manifest_data.get("metadata", {}).get(
                        "ts_field", "ts_event"
                    )
                    manifest_data["primary_keys"] = manifest_data.get("metadata", {}).get(
                        "primary_keys", ["instrument_id", "ts_event"]
                    )

                    manifest = self._dict_to_manifest(manifest_data)

                    # Cache for future use
                    self._manifests[dataset_id] = manifest

                    return manifest

                finally:
                    session.close()

    def _create_contract_from_manifest(self, manifest: DatasetManifest) -> DataContract:
        """
        Create a data contract from a dataset manifest.
        """
        from ml.registry.dataclasses import QualityFlag
        from ml.registry.dataclasses import ValidationRule
        from ml.registry.dataclasses import ValidationRuleType

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
            quality_thresholds={},
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
            if self.backend == BackendType.JSON:
                if dataset_id not in self._contracts:
                    # Create a default contract from manifest if not exists
                    manifest = self.get_manifest(dataset_id)
                    contract = self._create_contract_from_manifest(manifest)
                    self._contracts[dataset_id] = contract
                    return contract
                return self._contracts[dataset_id]

            elif self.backend == BackendType.POSTGRES:
                # Check if we have a cached contract
                if dataset_id in self._contracts:
                    return self._contracts[dataset_id]

                # For PostgreSQL, contracts are stored in the manifest's constraints field
                manifest = self.get_manifest(dataset_id)
                contract = self._create_contract_from_manifest(manifest)

                # Cache for future use
                self._contracts[dataset_id] = contract

                return contract

    def emit_event(
        self,
        dataset_id: str,
        instrument_id: str,
        stage: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: str,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """
        Emit a data processing event.

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
        with self._lock:
            # Anchor event time within the provided window (start of period)
            ts_event = ts_min

            event = {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage,
                "source": source,
                "run_id": run_id,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "ts_event": ts_event,
                "count": count,
                "status": status,
                "error": error,
                "created_at": time.time(),
                "metadata": metadata or {},
            }

            if self.backend == BackendType.JSON:
                self._events.append(event)

                # Trim old events to prevent unbounded growth (keep last 10000)
                if len(self._events) > 10000:
                    self._events = self._events[-10000:]

                # Tests expect persistence immediately after emit_event
                self._save_registry(immediate=True)

            elif self.backend == BackendType.POSTGRES:
                session = self.persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    # Prefer extended function with metadata if available; else fallback
                    try:
                        query_ext = text(
                            """
                            SELECT emit_data_event_ext(
                                :dataset_id, :instrument_id, :stage, :source, :run_id,
                                :ts_min, :ts_max, :count, :status, :error, :metadata
                            )
                        """
                        )
                        session.execute(
                            query_ext,
                            {
                                "dataset_id": dataset_id,
                                "instrument_id": instrument_id,
                                "stage": stage,
                                "source": source,
                                "run_id": run_id,
                                "ts_min": ts_min,
                                "ts_max": ts_max,
                                "count": count,
                                "status": status,
                                "error": error,
                                "metadata": json.dumps(metadata or {}),
                            },
                        )
                    except Exception:
                        query = text(
                            """
                            SELECT emit_data_event(
                                :dataset_id, :instrument_id, :stage, :source, :run_id,
                                :ts_min, :ts_max, :count, :status, :error
                            )
                        """
                        )
                        session.execute(
                            query,
                            {
                                "dataset_id": dataset_id,
                                "instrument_id": instrument_id,
                                "stage": stage,
                                "source": source,
                                "run_id": run_id,
                                "ts_min": ts_min,
                                "ts_max": ts_max,
                                "count": count,
                                "status": status,
                                "error": error,
                            },
                        )
                    session.commit()

                except Exception as e:
                    session.rollback()
                    logger.error("Failed to emit event: %s", e)
                    raise
                finally:
                    session.close()

            logger.debug(
                "Emitted event: dataset=%s, instrument=%s, stage=%s, status=%s, count=%d",
                dataset_id,
                instrument_id,
                stage,
                status,
                count,
            )

    def update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        """
        Update watermark for a dataset/instrument/source combination.

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
        >>> registry.update_watermark(
        ...     dataset_id="bars_eurusd_1m",
        ...     instrument_id="EUR/USD",
        ...     source="live",
        ...     last_success_ns=1234567900000000000,
        ...     count=1000,
        ...     completeness_pct=98.5
        ... )

        """
        with self._lock:
            watermark_key = f"{dataset_id}:{instrument_id}:{source}"

            watermark = Watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=source,
                last_success_ns=last_success_ns,
                last_attempt_ns=last_success_ns,
                last_count=count,
                completeness_pct=completeness_pct,
                updated_at=time.time(),
            )

            if self.backend == BackendType.JSON:
                self._watermarks[watermark_key] = watermark
                # Persist immediately to satisfy conformance tests
                self._save_registry(immediate=True)

            elif self.backend == BackendType.POSTGRES:
                session = self.persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    # Use the SQL function to update watermark
                    query = text(
                        """
                        SELECT update_watermark(
                            :dataset_id, :instrument_id, :source,
                            :last_success_ns, :count, :completeness_pct
                        )
                    """
                    )

                    session.execute(
                        query,
                        {
                            "dataset_id": dataset_id,
                            "instrument_id": instrument_id,
                            "source": source,
                            "last_success_ns": last_success_ns,
                            "count": count,
                            "completeness_pct": completeness_pct,
                        },
                    )
                    session.commit()

                    # Update cache
                    self._watermarks[watermark_key] = watermark

                except Exception as e:
                    session.rollback()
                    logger.error("Failed to update watermark: %s", e)
                    raise
                finally:
                    session.close()

            logger.debug(
                "Updated watermark: dataset=%s, instrument=%s, source=%s, completeness=%.1f%%",
                dataset_id,
                instrument_id,
                source,
                completeness_pct,
            )

    def get_watermark(self, dataset_id: str, instrument_id: str, source: str) -> Watermark | None:
        """
        Get watermark for a dataset/instrument/source combination.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : str
            Data source (live, historical, backfill)

        Returns
        -------
        Watermark | None
            The watermark if exists, None otherwise

        Examples
        --------
        >>> watermark = registry.get_watermark("bars_eurusd_1m", "EUR/USD", "live")
        >>> if watermark:
        ...     print(f"Last success: {watermark.last_success_ns}")
        ...     print(f"Completeness: {watermark.completeness_pct}%")

        """
        with self._lock:
            watermark_key = f"{dataset_id}:{instrument_id}:{source}"

            if self.backend == BackendType.JSON:
                return self._watermarks.get(watermark_key)

            elif self.backend == BackendType.POSTGRES:
                # Check cache first
                if watermark_key in self._watermarks:
                    return self._watermarks[watermark_key]

                session = self.persistence.get_session()
                if session is None:
                    raise RuntimeError("Failed to get database session")

                try:
                    query = text(
                        """
                        SELECT dataset_id, instrument_id, source,
                               last_success_ns, last_attempt_ns, last_count,
                               completeness_pct, EXTRACT(EPOCH FROM updated_at) as updated_at
                        FROM ml_data_watermarks
                        WHERE dataset_id = :dataset_id
                          AND instrument_id = :instrument_id
                          AND source = :source
                    """
                    )

                    result = session.execute(
                        query,
                        {
                            "dataset_id": dataset_id,
                            "instrument_id": instrument_id,
                            "source": source,
                        },
                    ).fetchone()

                    if result is None:
                        return None

                    watermark = Watermark(**dict(result))

                    # Cache for future use
                    self._watermarks[watermark_key] = watermark

                    return watermark

                finally:
                    session.close()

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
            for parent_id in parent_ids:
                lineage_entry = {
                    "transform_id": transform_id,
                    "child_dataset_id": child_dataset_id,
                    "parent_dataset_id": parent_id,
                    "ts_range": ts_range,
                    "parameters": params,
                    "created_at": time.time(),
                }

                if self.backend == BackendType.JSON:
                    self._lineage.append(lineage_entry)

                    # Trim old lineage entries to prevent unbounded growth (keep last 5000)
                    if len(self._lineage) > 5000:
                        self._lineage = self._lineage[-5000:]

                    self._save_registry()

                elif self.backend == BackendType.POSTGRES:
                    session = self.persistence.get_session()
                    if session is None:
                        raise RuntimeError("Failed to get database session")

                    try:
                        query = text(
                            """
                            INSERT INTO ml_data_lineage
                            (transform_id, child_dataset_id, parent_dataset_id, ts_range, parameters)
                            VALUES
                            (:transform_id, :child_dataset_id, :parent_dataset_id, :ts_range, :parameters)
                        """
                        )

                        session.execute(
                            query,
                            {
                                "transform_id": transform_id,
                                "child_dataset_id": child_dataset_id,
                                "parent_dataset_id": parent_id,
                                "ts_range": json.dumps(ts_range),
                                "parameters": json.dumps(params),
                            },
                        )

                    except Exception as e:
                        session.rollback()
                        logger.error("Failed to link lineage: %s", e)
                        raise

                    # Commit after all parents are linked
                    if parent_id == parent_ids[-1]:
                        session.commit()
                        session.close()

            logger.info(
                "Linked lineage: child=%s, parents=%s, transform=%s",
                child_dataset_id,
                parent_ids,
                transform_id,
            )

    def __del__(self) -> None:
        """
        Cleanup on deletion.
        """
        # Cancel any pending save timer
        if hasattr(self, "_save_timer") and self._save_timer is not None:
            self._save_timer.cancel()

        # Ensure final save
        if hasattr(self, "_pending_save") and self._pending_save:
            self._do_save()

        # Close persistence connections
        if hasattr(self, "persistence"):
            self.persistence.close()
