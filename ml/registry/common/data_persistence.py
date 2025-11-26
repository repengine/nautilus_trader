#!/usr/bin/env python3

"""
DataPersistenceComponent - Handles persistence operations for data registry.

This component is extracted from the DataRegistry god class following the
established TDD decomposition pattern. It handles:
- JSON/PostgreSQL persistence
- Serialization/deserialization of DatasetManifest, DataContract, Watermark
- Batch save management with timers
- Row conversion for PostgreSQL backend

Thread-safety: All operations are protected by an RLock.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from ml.registry.data_registry import Watermark
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetLineageRecord
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import StorageKind
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)


class DataPersistenceComponent:
    """
    Handles persistence operations for data registry.

    This component manages the storage and retrieval of dataset manifests,
    contracts, events, watermarks, and lineage using either JSON files
    or PostgreSQL as the backend.

    Attributes
    ----------
    persistence : PersistenceManager
        The persistence manager for backend operations.
    registry_path : Path
        The directory path for registry storage.
    batch_save_interval : float
        Seconds to wait before flushing batch saves.

    Thread Safety
    -------------
    All public methods are protected by an RLock for thread-safe concurrent access.

    Example
    -------
    >>> config = PersistenceConfig(backend=BackendType.JSON, json_path=Path("/tmp/registry"))
    >>> component = DataPersistenceComponent(config, Path("/tmp/registry"))
    >>> component._load_registry()
    >>> component._save_registry(immediate=True)
    """

    def __init__(
        self,
        registry_path: Path,
        persistence_config: PersistenceConfig,
        batch_save_interval: float = 0.1,
    ) -> None:
        """
        Initialize persistence component.

        Parameters
        ----------
        registry_path : Path
            Directory path for registry storage.
        persistence_config : PersistenceConfig
            Configuration for persistence backend (JSON or PostgreSQL).
        batch_save_interval : float
            Seconds to wait before flushing batch saves (default 0.1s).

        Raises
        ------
        ValueError
            If persistence_config is invalid.
        """
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.persistence = PersistenceManager(persistence_config)
        self.batch_save_interval = batch_save_interval
        self.registry_file = self.registry_path / "data_registry.json"

        # Thread safety
        self._lock = threading.RLock()

        # In-memory state
        self._manifests: dict[str, DatasetManifest] = {}
        self._contracts: dict[str, DataContract] = {}
        self._events: list[dict[str, Any]] = []
        self._watermarks: dict[str, Watermark] = {}
        self._lineage: list[dict[str, Any]] = []

        # Batch save management
        self._pending_save = False
        self._save_timer: threading.Timer | None = None

        # Initialize or load registry
        self._load_registry()

        logger.debug(
            "Initialized DataPersistenceComponent at %s with backend=%s",
            registry_path,
            self.backend.value,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def backend(self) -> BackendType:
        """
        Get the persistence backend type.

        Returns
        -------
        BackendType
            The backend type (JSON or POSTGRES).
        """
        return self.persistence.config.backend

    # -------------------------------------------------------------------------
    # Registry Loading
    # -------------------------------------------------------------------------

    def _load_registry(self) -> None:
        """
        Load registry from persistence backend or create new one.

        For JSON backend, loads from data_registry.json file if it exists,
        otherwise initializes empty state.

        For PostgreSQL backend, queries the database directly.

        Thread-safe: Acquires lock during loading.
        """
        with self._lock:
            if self.backend == BackendType.JSON:
                self._load_from_json()
            elif self.backend == BackendType.POSTGRES:
                self._initialize_empty_registry()
                logger.info("Using PostgreSQL backend - data is stored in database")

    def _load_from_json(self) -> None:
        """Load registry state from JSON file."""
        if self.registry_file.exists():
            try:
                data = self.persistence.load_json("data_registry.json")
                if data is not None:
                    # Load manifests
                    self._manifests = {}
                    for dataset_id, manifest_data in data.get("manifests", {}).items():
                        self._manifests[dataset_id] = self._dict_to_manifest(manifest_data)

                    # Load contracts
                    self._contracts = {}
                    for dataset_id, contract_data in data.get("contracts", {}).items():
                        self._contracts[dataset_id] = self._dict_to_contract(contract_data)

                    # Load events, watermarks, and lineage
                    self._events = data.get("events", [])
                    self._watermarks = {}
                    for key, watermark_data in data.get("watermarks", {}).items():
                        self._watermarks[key] = self._dict_to_watermark(watermark_data)
                    self._lineage = data.get("lineage", [])

                    logger.debug(
                        "Loaded registry from JSON with %d manifests",
                        len(self._manifests),
                    )
                else:
                    self._initialize_empty_registry()
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Failed to load registry from JSON, starting fresh: %s",
                    exc,
                    exc_info=True,
                )
                self._initialize_empty_registry()
        else:
            self._initialize_empty_registry()

    def _initialize_empty_registry(self) -> None:
        """Initialize empty registry structures."""
        self._manifests = {}
        self._contracts = {}
        self._events = []
        self._watermarks = {}
        self._lineage = []
        if self.backend == BackendType.JSON:
            # Save immediately to ensure the backing file exists
            self._save_registry(immediate=True)

    # -------------------------------------------------------------------------
    # Registry Saving
    # -------------------------------------------------------------------------

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
                        self._flush_batch_save,
                    )
                    self._save_timer.start()

    def _do_save(self) -> None:
        """Perform the actual save to disk."""
        try:
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
            logger.debug("Registry saved with %d manifests", len(self._manifests))
        except Exception as exc:
            logger.error(
                "Failed to save registry: %s",
                exc,
                exc_info=True,
            )
            raise

    def _flush_batch_save(self) -> None:
        """Flush pending batch saves."""
        with self._lock:
            if self._pending_save:
                try:
                    self._do_save()
                except FileNotFoundError as exc:
                    logger.debug(
                        "Batch save flush: registry path missing (ignored): %s",
                        exc,
                    )
                except Exception as exc:
                    logger.error(
                        "Error during batch save flush: %s",
                        exc,
                        exc_info=True,
                    )
                finally:
                    self._pending_save = False
                    self._save_timer = None

    def flush(self) -> None:
        """
        Flush any pending batch saves immediately.

        Call this before shutdown or when immediate persistence is needed.
        """
        with self._lock:
            if self.backend == BackendType.JSON:
                self._save_registry(immediate=True)

    # -------------------------------------------------------------------------
    # Serialization: Manifest
    # -------------------------------------------------------------------------

    def _dict_to_manifest(self, data: dict[str, Any]) -> DatasetManifest:
        """
        Convert dictionary to DatasetManifest.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary to deserialize.

        Returns
        -------
        DatasetManifest
            Deserialized manifest.
        """
        # Convert string enum values back to enum types (case-insensitive)
        dataset_type_val = data.get("dataset_type")
        if isinstance(dataset_type_val, str):
            dataset_type_val = dataset_type_val.lower()
        data["dataset_type"] = DatasetType(dataset_type_val)

        storage_kind_val = data.get("storage_kind")
        if isinstance(storage_kind_val, str):
            storage_kind_val = storage_kind_val.lower()
        data["storage_kind"] = StorageKind(storage_kind_val)

        return DatasetManifest(**data)

    def _manifest_to_dict(self, manifest: DatasetManifest) -> dict[str, Any]:
        """
        Convert DatasetManifest to dictionary.

        Parameters
        ----------
        manifest : DatasetManifest
            Manifest to serialize.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON.
        """
        return {
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

    def _manifest_from_row(self, row: Any) -> DatasetManifest:
        """
        Convert a database row to a dataset manifest.

        Parameters
        ----------
        row : Any
            SQLAlchemy Row with _mapping attribute.

        Returns
        -------
        DatasetManifest
            Converted manifest.
        """
        manifest_data = dict(getattr(row, "_mapping", row))

        metadata = self._ensure_json(manifest_data.get("metadata") or {}) or {}
        manifest_data["metadata"] = metadata
        manifest_data["partitioning"] = self._ensure_json(manifest_data.get("partitioning") or {}) or {}
        manifest_data["constraints"] = self._ensure_json(manifest_data.get("constraints") or {}) or {}
        manifest_data["schema"] = self._ensure_json(manifest_data.get("schema") or {}) or {}
        manifest_data["lineage"] = self._ensure_json(manifest_data.get("lineage") or []) or []

        manifest_data["seq_field"] = metadata.get("seq_field")
        manifest_data["ts_field"] = metadata.get(
            "ts_field",
            manifest_data.get("ts_field", "ts_event"),
        )
        manifest_data["primary_keys"] = metadata.get(
            "primary_keys",
            manifest_data.get("primary_keys", ["instrument_id", "ts_event"]),
        )

        return self._dict_to_manifest(manifest_data)

    # -------------------------------------------------------------------------
    # Serialization: Contract
    # -------------------------------------------------------------------------

    def _dict_to_contract(self, data: dict[str, Any]) -> DataContract:
        """
        Convert dictionary to DataContract.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary to deserialize.

        Returns
        -------
        DataContract
            Deserialized contract.
        """
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

        Parameters
        ----------
        contract : DataContract
            Contract to serialize.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON.
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
                },
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

    # -------------------------------------------------------------------------
    # Serialization: Watermark
    # -------------------------------------------------------------------------

    def _dict_to_watermark(self, data: dict[str, Any]) -> Watermark:
        """
        Convert dictionary to Watermark.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary to deserialize.

        Returns
        -------
        Watermark
            Deserialized watermark.
        """
        return Watermark(**data)

    def _watermark_to_dict(self, watermark: Watermark) -> dict[str, Any]:
        """
        Convert Watermark to dictionary.

        Parameters
        ----------
        watermark : Watermark
            Watermark to serialize.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON.
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

    @staticmethod
    def _watermark_from_row(row: Any) -> Watermark:
        """
        Convert a database row to a watermark instance.

        Parameters
        ----------
        row : Any
            SQLAlchemy Row with _mapping attribute.

        Returns
        -------
        Watermark
            Converted watermark.
        """
        data = dict(getattr(row, "_mapping", row))
        return Watermark(
            dataset_id=str(data.get("dataset_id", "")),
            instrument_id=str(data.get("instrument_id", "")),
            source=str(data.get("source", "")),
            last_success_ns=int(data.get("last_success_ns", 0)),
            last_attempt_ns=int(data.get("last_attempt_ns", 0)),
            last_count=int(data.get("last_count", 0)),
            completeness_pct=float(data.get("completeness_pct", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )

    # -------------------------------------------------------------------------
    # Serialization: Lineage
    # -------------------------------------------------------------------------

    @staticmethod
    def _lineage_from_row(row: Any) -> DatasetLineageRecord:
        """
        Convert a database row to a dataset lineage record.

        Parameters
        ----------
        row : Any
            SQLAlchemy Row with _mapping attribute.

        Returns
        -------
        DatasetLineageRecord
            Converted lineage record.
        """
        data = dict(getattr(row, "_mapping", row))

        ts_range_raw = data.get("ts_range") or {}
        if isinstance(ts_range_raw, str):
            try:
                ts_range = json.loads(ts_range_raw)
            except json.JSONDecodeError:
                ts_range = {}
        else:
            ts_range = dict(ts_range_raw)

        params_raw = data.get("parameters") or {}
        if isinstance(params_raw, str):
            try:
                parameters = json.loads(params_raw)
            except json.JSONDecodeError:
                parameters = {}
        else:
            parameters = dict(params_raw)

        created_at_raw = data.get("created_at", 0.0)
        created_at = float(created_at_raw) if created_at_raw is not None else 0.0

        return DatasetLineageRecord(
            transform_id=str(data.get("transform_id", "")),
            child_dataset_id=str(data.get("child_dataset_id", "")),
            parent_dataset_id=str(data.get("parent_dataset_id", "")),
            ts_range={str(k): int(v) for k, v in ts_range.items()},
            parameters={str(k): v for k, v in parameters.items()},
            created_at=created_at,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _ensure_json(self, value: Any) -> Any:
        """
        Ensure value is parsed JSON.

        Parameters
        ----------
        value : Any
            Value to parse (may be string or already parsed).

        Returns
        -------
        Any
            Parsed JSON or original value if not valid JSON string.
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def __del__(self) -> None:
        """Ensure pending saves are flushed on cleanup."""
        try:
            # Cancel any pending save timer
            if hasattr(self, "_save_timer") and self._save_timer is not None:
                self._save_timer.cancel()

            # Ensure final save
            if hasattr(self, "_pending_save") and self._pending_save:
                self._do_save()

            # Close persistence connections
            if hasattr(self, "persistence"):
                self.persistence.close()
        except Exception as exc:
            logger.debug("DataPersistenceComponent cleanup failed: %s", exc)
