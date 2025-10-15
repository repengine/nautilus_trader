#!/usr/bin/env python3

"""
Contract enforcement for ML data operations.

This module enforces data contracts and manages schema versions, including
manifest/contract retrieval, caching, preflight validation, and schema
migration window management.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, cast

from ml._imports import HAS_PROMETHEUS
from ml._imports import pd
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.stores.schema_validator import SchemaValidator
from ml.stores.validation_types import QualityReport


logger = logging.getLogger(__name__)


# ========================================================================
# Prometheus Metrics (using centralized bootstrap pattern)
# ========================================================================

class _CounterLike(Protocol):
    def labels(self, **kwargs: object) -> _CounterLike: ...
    def inc(self, *args: object, **kwargs: object) -> None: ...


class _NoOpMetric:
    def labels(self, **_: object) -> _NoOpMetric:
        return self

    def inc(self, *_: object, **__: object) -> None:
        return None


# Declare metric variables
schema_mismatch_counter: Any = _NoOpMetric()

try:
    from ml.common.metrics import schema_mismatch_counter as _smc

    schema_mismatch_counter = _smc
except Exception:
    logger.debug("Metrics import failed; using no-op counter", exc_info=True)


# ========================================================================
# Protocol Definition
# ========================================================================


class ContractEnforcerProtocol(Protocol):
    """Protocol for contract enforcement operations."""

    def preflight_check(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Perform preflight schema validation before processing.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        data : DataFrame | list[dict]
            Data to validate
        strict : bool
            Require exact schema match

        Returns
        -------
        tuple[bool, str | None, dict[str, Any]]
            (success, error_message, validation_details)
        """
        ...

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        """
        Get dataset manifest with caching.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        DatasetManifest
            Dataset manifest
        """
        ...

    def get_contract(self, dataset_id: str) -> DataContract:
        """
        Get data contract with caching.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        DataContract
            Data contract
        """
        ...


# ========================================================================
# ContractEnforcer Implementation
# ========================================================================


class ContractEnforcer:
    """
    Enforces data contracts and manages schema versions.

    Retrieves and caches manifests/contracts, validates data against contracts,
    performs preflight checks, and manages schema migration windows. This component
    is stateless except for caching and is designed for dependency injection.

    This component is extracted from the DataStore god class to provide focused,
    testable contract enforcement following the Strangler Fig pattern.

    Parameters
    ----------
    registry : Any
        Data registry for manifest/contract retrieval (RegistryProtocol)
    schema_validator : SchemaValidator
        Schema validation component
    allow_schema_migration : bool
        Allow dual-write during schema migration
    schema_migration_window_hours : int
        Hours to allow dual-write during migration
    """

    def __init__(
        self,
        *,
        registry: Any,
        schema_validator: SchemaValidator,
        allow_schema_migration: bool = False,
        schema_migration_window_hours: int = 24,
    ) -> None:
        """
        Initialize contract enforcer.

        Parameters
        ----------
        registry : Any
            Data registry for manifest/contract retrieval
        schema_validator : SchemaValidator
            Schema validation component
        allow_schema_migration : bool
            Allow dual-write during schema migration
        schema_migration_window_hours : int
            Hours to allow dual-write during migration
        """
        self.registry = registry
        self.schema_validator = schema_validator
        self.allow_schema_migration = allow_schema_migration
        self.schema_migration_window_hours = schema_migration_window_hours

        # Caches
        self._manifest_cache: dict[str, DatasetManifest] = {}
        self._contract_cache: dict[str, DataContract] = {}
        self._schema_migration_state: dict[str, dict[str, Any]] = {}

        logger.debug(
            "Initialized ContractEnforcer (migration=%s, window=%dh)",
            allow_schema_migration,
            schema_migration_window_hours,
        )

    def preflight_check(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Perform preflight schema validation before processing data.

        Validates schema compatibility, type matching, primary keys, and required
        fields before allowing data writes. Handles schema migration windows when
        configured.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        data : DataFrame | list[dict]
            Data to validate
        strict : bool
            If True, require exact schema match; if False, allow type coercion

        Returns
        -------
        tuple[bool, str | None, dict[str, Any]]
            - success: True if validation passed
            - error_message: Error description if failed, None otherwise
            - validation_details: Dict with validation metadata

        Examples
        --------
        >>> enforcer = ContractEnforcer(registry=registry, schema_validator=validator)
        >>> success, error, details = enforcer.preflight_check("bars_eurusd_1m", df)
        >>> if not success:
        ...     print(f"Validation failed: {error}")
        """
        validation_details: dict[str, Any] = {
            "dataset_id": dataset_id,
            "strict_mode": strict,
            "checks_performed": [],
            "warnings": [],
        }

        try:
            # Get manifest and contract
            manifest = self.get_manifest(dataset_id)
            validation_details["manifest_version"] = manifest.version
            validation_details["manifest_schema_hash"] = manifest.schema_hash

            # Convert to DataFrame if needed
            data_frame: DataFrameLike = self._to_dataframe(data)
            data_frame_any = cast(Any, data_frame)

            # Check 1: Required columns present
            if hasattr(data_frame, "columns"):
                validation_details["checks_performed"].append("required_columns")
                actual_columns = set(data_frame_any.columns)
                required_columns = set(manifest.schema.keys())
                missing_cols = required_columns - actual_columns

                if missing_cols:
                    error_msg = f"Missing required columns: {missing_cols}"
                    return False, error_msg, validation_details

            # Check 2: Type compatibility
            type_mismatches = []
            if hasattr(data_frame, "columns"):
                validation_details["checks_performed"].append("type_compatibility")
                for col_name, expected_type in manifest.schema.items():
                    if col_name in data_frame_any.columns:
                        actual_type = str(data_frame_any[col_name].dtype)
                        if not self._types_compatible(actual_type, expected_type):
                            type_mismatches.append(
                                {
                                    "column": col_name,
                                    "expected": expected_type,
                                    "actual": actual_type,
                                },
                            )

            if type_mismatches:
                validation_details["type_mismatches"] = type_mismatches
                if strict:
                    error_msg = f"Type mismatches found: {type_mismatches}"
                    return False, error_msg, validation_details
                else:
                    validation_details["warnings"].append(
                        f"Type coercion will be attempted for {len(type_mismatches)} columns",
                    )

            # Check 3: Schema hash compatibility
            actual_schema_hash = self._compute_schema_hash(data_frame, manifest)
            validation_details["actual_schema_hash"] = actual_schema_hash
            validation_details["checks_performed"].append("schema_hash")

            if actual_schema_hash != manifest.schema_hash:
                # Check if we're in a migration window
                if self._is_in_migration_window(dataset_id):
                    validation_details["migration_mode"] = True
                    validation_details["warnings"].append(
                        "Schema migration in progress - dual-write enabled",
                    )
                else:
                    error_msg = (
                        f"Schema hash mismatch. Expected: {manifest.schema_hash}, "
                        f"Got: {actual_schema_hash}. Version bump required."
                    )
                    # Record schema mismatch metric
                    if HAS_PROMETHEUS:
                        schema_mismatch_counter.labels(
                            dataset=dataset_id,
                            mismatch_type="hash_mismatch",
                        ).inc()

                    if strict:
                        return False, error_msg, validation_details
                    else:
                        validation_details["warnings"].append(error_msg)

            # Check 4: Primary key fields present and not null
            if manifest.primary_keys:
                validation_details["checks_performed"].append("primary_keys")
                for pk_field in manifest.primary_keys:
                    if hasattr(data_frame_any, "columns") and pk_field in data_frame_any.columns:
                        # Handle both Polars and pandas
                        if hasattr(data_frame_any[pk_field], "is_null"):
                            # Polars
                            null_count = data_frame_any[pk_field].is_null().sum()
                        elif hasattr(data_frame_any[pk_field], "isna"):
                            # pandas
                            null_count = data_frame_any[pk_field].isna().sum()
                        else:
                            null_count = 0

                        if null_count > 0:
                            error_msg = (
                                f"Primary key field '{pk_field}' contains {null_count} null values"
                            )
                            return False, error_msg, validation_details

            # Check 5: Required fields (from constraints)
            if manifest.constraints and "nullability" in manifest.constraints:
                validation_details["checks_performed"].append("required_fields")
                for field, nullable in manifest.constraints["nullability"].items():
                    if not nullable and hasattr(data_frame, "columns") and field in data_frame.columns:
                        # Handle both Polars and pandas
                        if hasattr(data_frame[field], "is_null"):
                            # Polars
                            null_count = data_frame[field].is_null().sum()
                        elif hasattr(data_frame[field], "isna"):
                            # pandas
                            null_count = data_frame[field].isna().sum()
                        else:
                            null_count = 0

                        if null_count > 0:
                            error_msg = (
                                f"Required field '{field}' contains {null_count} null values"
                            )
                            return False, error_msg, validation_details

            validation_details["preflight_passed"] = True
            return True, None, validation_details

        except Exception:
            error_msg = "Preflight check failed"
            logger.error(
                error_msg,
                exc_info=True,
            )
            return False, error_msg, {}

    def validate_batch(
        self,
        dataset_id: str,
        data: DataFrameLike,
        strict_mode: bool = False,
    ) -> QualityReport:
        """
        Validate a batch of data against contract using SchemaValidator.

        This is a convenience method that retrieves the manifest/contract and
        delegates to the SchemaValidator.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        data : DataFrameLike
            Data to validate
        strict_mode : bool
            If True, treat warnings as failures

        Returns
        -------
        QualityReport
            Validation results with quality score and violations
        """
        manifest = self.get_manifest(dataset_id)
        contract = self.get_contract(dataset_id)
        return self.schema_validator.validate_batch(
            data,
            manifest,
            contract,
            strict_mode=strict_mode,
        )

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        """
        Get dataset manifest with caching and version check.

        Manifests are cached to avoid repeated registry lookups. Version changes
        trigger migration window logic if configured.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        DatasetManifest
            Dataset manifest
        """
        if dataset_id not in self._manifest_cache:
            manifest = self.registry.get_manifest(dataset_id)
            self._manifest_cache[dataset_id] = manifest

            # Check for schema version changes
            if dataset_id in self._schema_migration_state:
                old_version = self._schema_migration_state[dataset_id].get("version")
                if old_version and old_version != manifest.version:
                    logger.info(
                        "Schema version change detected for %s: %s -> %s",
                        dataset_id,
                        old_version,
                        manifest.version,
                    )
                    # Start migration window if configured
                    if self.allow_schema_migration:
                        self._start_migration_window(dataset_id, manifest)

        return self._manifest_cache[dataset_id]

    def get_contract(self, dataset_id: str) -> DataContract:
        """
        Get data contract with caching and version logging.

        Contracts are cached to avoid repeated registry lookups.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        DataContract
            Data contract with validation rules
        """
        if dataset_id not in self._contract_cache:
            contract = self.registry.get_contract(dataset_id)
            self._contract_cache[dataset_id] = contract

            # Log contract version for tracking
            logger.debug(
                "Loaded contract for %s: version=%s, mode=%s, rules=%d",
                dataset_id,
                contract.version,
                contract.enforcement_mode,
                len(contract.validation_rules),
            )

        return self._contract_cache[dataset_id]

    def ensure_dataset_registered(
        self,
        dataset_id: str,
        dataset_type: DatasetType,
        instrument_id: str,
    ) -> None:
        """
        Ensure dataset is registered in the registry.

        Creates a basic manifest if dataset doesn't exist. Used for dynamic
        dataset creation (e.g., predictions, signals).

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        dataset_type : DatasetType
            Type of dataset
        instrument_id : str
            Instrument identifier
        """
        try:
            self.get_manifest(dataset_id)
        except Exception:
            # Dataset not registered, create basic manifest
            logger.info("Auto-registering dataset %s (type=%s)", dataset_id, dataset_type)

            # Create minimal manifest
            from ml.registry.dataclasses import StorageKind
            from ml.registry.utils import compute_dataset_schema_hash

            basic_schema = {
                "instrument_id": "str",
                "ts_event": "int64",
                "ts_init": "int64",
            }

            schema_hash = compute_dataset_schema_hash(
                schema=basic_schema,
                primary_keys=["instrument_id", "ts_event"],
                ts_field="ts_event",
                seq_field=None,
                pipeline_signature=None,
            )

            manifest = DatasetManifest(
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                storage_kind=StorageKind.POSTGRES,
                location=f"ml.{dataset_id}",
                partitioning={},
                retention_days=90,
                version="1.0.0",
                schema=basic_schema,
                schema_hash=schema_hash,
                primary_keys=["instrument_id", "ts_event"],
                ts_field="ts_event",
                seq_field=None,
                constraints={},
                lineage=[],
                pipeline_signature="auto_generated",
            )

            # Register with registry
            self.registry.register_manifest(manifest)
            self._manifest_cache[dataset_id] = manifest

    def _compute_schema_hash(self, data_frame: DataFrameLike, manifest: DatasetManifest) -> str:
        """
        Compute schema hash for actual data.

        Parameters
        ----------
        data_frame : DataFrameLike
            Data to compute hash from
        manifest : DatasetManifest
            Dataset manifest for reference schema

        Returns
        -------
        str
            Schema hash
        """
        from ml.registry.utils import compute_dataset_schema_hash

        data_frame_any = cast(Any, data_frame)
        if not hasattr(data_frame_any, "columns"):
            # For non-DataFrame data, use manifest hash
            return manifest.schema_hash

        # Build schema dict from actual data
        actual_schema: dict[str, str] = {}
        for col in data_frame_any.columns:
            if col in manifest.schema:
                actual_schema[col] = manifest.schema[col]
            else:
                dtype = str(data_frame_any[col].dtype)
                lower = dtype.lower()
                if "int" in lower:
                    actual_schema[col] = "int64"
                elif "float" in lower:
                    actual_schema[col] = "float64"
                elif "bool" in lower:
                    actual_schema[col] = "bool"
                else:
                    actual_schema[col] = "str"

        for manifest_column, manifest_dtype in manifest.schema.items():
            actual_schema.setdefault(manifest_column, manifest_dtype)

        return compute_dataset_schema_hash(
            schema=actual_schema,
            primary_keys=manifest.primary_keys,
            ts_field=manifest.ts_field,
            seq_field=manifest.seq_field,
            pipeline_signature=manifest.pipeline_signature,
        )

    def _is_in_migration_window(self, dataset_id: str) -> bool:
        """
        Check if dataset is in schema migration window.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        bool
            True if in migration window
        """
        if not self.allow_schema_migration:
            return False

        if dataset_id not in self._schema_migration_state:
            return False

        migration_info = self._schema_migration_state[dataset_id]
        migration_start = migration_info.get("start_time", 0)
        window_ns = self.schema_migration_window_hours * 3600 * 1e9

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        current_time = _sanitize(
            int(time.time_ns()),
            context="contract_enforcer._is_in_migration_window:now",
        )
        if current_time - migration_start < window_ns:
            return True
        else:
            # Migration window expired, clear state
            del self._schema_migration_state[dataset_id]
            logger.info("Schema migration window expired for %s", dataset_id)
            return False

    def _start_migration_window(self, dataset_id: str, manifest: DatasetManifest) -> None:
        """
        Start a schema migration window for dual-write.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        manifest : DatasetManifest
            New manifest version
        """
        self._schema_migration_state[dataset_id] = {
            "start_time": time.time_ns(),
            "version": manifest.version,
            "schema_hash": manifest.schema_hash,
        }

        logger.info(
            "Started schema migration window for %s (version %s, %d hours)",
            dataset_id,
            manifest.version,
            self.schema_migration_window_hours,
        )

        # Record migration start metric
        if HAS_PROMETHEUS:
            schema_mismatch_counter.labels(
                dataset=dataset_id,
                mismatch_type="migration_started",
            ).inc()

    def _to_dataframe(
        self,
        data: DataFrameLike | list[dict[str, Any]],
    ) -> DataFrameLike:
        """
        Convert various data formats to DataFrame-like or pass-through list.

        Parameters
        ----------
        data : DataFrameLike | list[dict]
            Input data

        Returns
        -------
        DataFrameLike | list[dict]
            DataFrame or list of dicts
        """
        # Import here to avoid circular dependency
        from ml._imports import HAS_POLARS
        from ml._imports import pl

        if isinstance(data, list):
            if HAS_POLARS and pl is not None:
                return cast(DataFrameLike, pl.DataFrame(data))
            if pd is not None:
                return cast(DataFrameLike, pd.DataFrame(data))
            raise RuntimeError("No DataFrame backend available to materialize list input")

        if not hasattr(data, "columns"):
            raise TypeError("Unsupported data type for dataframe conversion")

        return data

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """
        Check if actual type is compatible with expected type.

        Parameters
        ----------
        actual : str
            Actual data type
        expected : str
            Expected data type

        Returns
        -------
        bool
            True if types are compatible
        """
        type_map = {
            "int64": ["int", "int64", "i8", "Int64"],
            "float64": ["float", "float64", "f8", "Float64"],
            "str": ["str", "string", "object", "Utf8"],
            "bool": ["bool", "boolean", "Boolean"],
        }

        # Check both directions - actual could be in a group and expected could be in same group
        actual_lower = actual.lower()
        expected_lower = expected.lower()

        for expected_base, compatible_types in type_map.items():
            lower_compatible = [t.lower() for t in compatible_types]
            if expected_lower in lower_compatible and actual_lower in lower_compatible:
                return True

        return actual_lower == expected_lower
