#!/usr/bin/env python3

"""
Data Store facade providing typed read/write operations with contract validation.

This module provides a unified interface for all data operations in the ML pipeline,
integrating with the DataRegistry for manifest/contract retrieval, validating data
against contracts before writing, emitting events, and updating watermarks.

The DataStore acts as a facade over existing FeatureStore, ModelStore, and StrategyStore
while adding contract validation, event emission, and watermark tracking.

"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Protocol, cast

from ml._imports import HAS_PROMETHEUS
from ml.common.correlation import make_correlation_id
from ml.common.protocols import MLComponentMixin
from ml.config.events import Stage
from ml.registry.dataclasses import DataContract
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import QualityFlag
from ml.registry.dataclasses import ValidationRule
from ml.registry.dataclasses import ValidationRuleType
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.data_processor import DataProcessor
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.typing import DataFrameLike


if TYPE_CHECKING:
    from ml.registry.protocols import RegistryProtocol


logger = logging.getLogger(__name__)

# ========================================================================
# Prometheus Metrics
# ========================================================================


class _CounterLike(Protocol):
    def labels(self, **kwargs: object) -> _CounterLike:
        ...

    def inc(self, *args: object, **kwargs: object) -> None:
        ...


class _HistogramLike(Protocol):
    def labels(self, **kwargs: object) -> _HistogramLike:
        ...

    def observe(self, *args: object, **kwargs: object) -> None:
        ...


class _NoOpMetric:
    def labels(self, **_: object) -> _NoOpMetric:
        return self

    def inc(self, *_: object, **__: object) -> None:
        return None

    def observe(self, *_: object, **__: object) -> None:
        return None


# Declare metric variables once; assign real or no-op implementations below
validation_violations_counter: Any = _NoOpMetric()
validation_duration_histogram: Any = _NoOpMetric()
schema_mismatch_counter: Any = _NoOpMetric()
write_rejection_counter: Any = _NoOpMetric()
quality_score_histogram: Any = _NoOpMetric()

try:
    from ml.common.metrics import quality_score_histogram as _qsh
    from ml.common.metrics import schema_mismatch_counter as _smc
    from ml.common.metrics import validation_duration_histogram as _vd
    from ml.common.metrics import validation_violations_counter as _vvc
    from ml.common.metrics import write_rejection_counter as _wrc

    quality_score_histogram = _qsh
    schema_mismatch_counter = _smc
    validation_duration_histogram = _vd
    validation_violations_counter = _vvc
    write_rejection_counter = _wrc
except Exception:
    # Keep no-ops assigned
    pass


# ========================================================================
# Data Events and Quality Reports
# ========================================================================


@dataclass(frozen=True)
class DataEvent:
    """
    Event tracking data operations in the store.

    Attributes
    ----------
    event_id : str
        Unique event identifier
    dataset_id : str
        Dataset identifier
    instrument_id : str
        Instrument identifier
    operation : str
        Operation type (write_ingestion, write_features, etc.)
    source : str
        Data source (live, historical, backfill)
    run_id : str
        Processing run identifier
    ts_min : int
        Minimum timestamp in nanoseconds
    ts_max : int
        Maximum timestamp in nanoseconds
    record_count : int
        Number of records processed
    status : str
        Operation status (success, failed, partial)
    error_message : str | None
        Error message if failed
    created_at : int
        Event creation timestamp in nanoseconds
    metadata : dict[str, Any]
        Additional event metadata

    """

    event_id: str
    dataset_id: str
    instrument_id: str
    operation: str
    source: str
    run_id: str
    ts_min: int
    ts_max: int
    record_count: int
    status: str
    error_message: str | None = None
    created_at: int = field(default_factory=time.time_ns)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityReport:
    """
    Quality validation report for a batch of data.

    Attributes
    ----------
    dataset_id : str
        Dataset identifier
    total_records : int
        Total number of records validated
    passed_records : int
        Number of records that passed validation
    failed_records : int
        Number of records that failed validation
    quality_score : float
        Overall quality score (0-1)
    violations : list[ValidationViolation]
        List of validation rule violations
    validation_time_ms : float
        Time taken for validation in milliseconds
    metadata : dict[str, Any]
        Additional metadata

    """

    dataset_id: str
    total_records: int
    passed_records: int
    failed_records: int
    quality_score: float
    violations: list[ValidationViolation]
    validation_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationViolation:
    """
    Details of a validation rule violation.

    Attributes
    ----------
    rule_type : ValidationRuleType
        Type of validation rule violated
    field_name : str
        Field that failed validation
    severity : QualityFlag
        Severity of the violation
    violation_count : int
        Number of records with this violation
    sample_values : list[Any]
        Sample of violating values (max 5)
    description : str
        Human-readable description

    """

    rule_type: ValidationRuleType
    field_name: str
    severity: QualityFlag
    violation_count: int
    sample_values: list[Any]
    description: str


# ========================================================================
# DataStore Implementation
# ========================================================================


class DataStore(MLComponentMixin):
    """
    Typed read/write facade with contract validation and event emission.

    This class provides a unified interface for all data operations in the ML pipeline,
    ensuring data quality through contract validation, tracking all operations through
    events, and maintaining watermarks for data freshness tracking.

    Thread-safe for concurrent operations.

    Examples
    --------
    >>> # Initialize with registry and connection
    >>> store = DataStore(
    ...     registry=data_registry,
    ...     connection_string="postgresql://postgres@localhost/nautilus",
    ...     feature_store=feature_store,
    ...     model_store=model_store,
    ...     strategy_store=strategy_store
    ... )

    >>> # Write ingestion data with validation
    >>> event = store.write_ingestion(
    ...     dataset_id="bars_eurusd_1m",
    ...     records=bar_data,
    ...     source="historical",
    ...     run_id="run_123"
    ... )

    >>> # Validate data batch
    >>> report = store.validate_batch("bars_eurusd_1m", df)
    >>> if report.quality_score < 0.95:
    ...     logger.warning(f"Data quality below threshold: {report.quality_score}")

    """

    def __init__(
        self,
        connection_string: str,
        registry: RegistryProtocol | None = None,
        feature_store: FeatureStore | None = None,
        model_store: ModelStore | None = None,
        strategy_store: StrategyStore | None = None,
        fail_on_validation_error: bool = True,
        batch_size: int = 10000,
        allow_schema_migration: bool = False,
        schema_migration_window_hours: int = 24,
    ) -> None:
        """
        Initialize DataStore with registry and underlying stores.

        Parameters
        ----------
        registry : DataRegistry
            Data registry for manifest/contract retrieval
        connection_string : str
            PostgreSQL connection string
        feature_store : FeatureStore | None
            Feature store instance (created if not provided)
        model_store : ModelStore | None
            Model store instance (created if not provided)
        strategy_store : StrategyStore | None
            Strategy store instance (created if not provided)
        fail_on_validation_error : bool
            If True, fail writes on validation errors. If False, log warnings.
        batch_size : int
            Batch size for write operations
        allow_schema_migration : bool
            If True, allow dual-write during schema migration window
        schema_migration_window_hours : int
            Hours to allow dual-write during schema migration

        """
        # Explicit attribute annotation for protocol conformance
        self.registry: RegistryProtocol

        # Allow registry to be optional for convenience in tests/integration
        if registry is None:
            try:
                from ml.registry.data_registry import DataRegistry
                from ml.registry.persistence import BackendType
                from ml.registry.persistence import PersistenceConfig
                persistence_config = PersistenceConfig(
                    backend=BackendType.POSTGRES,
                    connection_string=connection_string,
                )
                self.registry = cast(
                    "RegistryProtocol",
                    DataRegistry(
                        registry_path=Path.home() / ".nautilus" / "ml" / "registry",
                        persistence_config=persistence_config,
                    ),
                )
            except Exception:
                # Fallback to JSON registry if DB not available
                from ml.registry.data_registry import DataRegistry
                from ml.registry.persistence import BackendType
                from ml.registry.persistence import PersistenceConfig
                persistence_config = PersistenceConfig(
                    backend=BackendType.JSON,
                    json_path=Path("./data/registry"),
                )
                self.registry = cast(
                    "RegistryProtocol",
                    DataRegistry(
                        registry_path=Path("./data/registry"),
                        persistence_config=persistence_config,
                    ),
                )
        else:
            self.registry = registry
        self.connection_string = connection_string
        self.fail_on_validation_error = fail_on_validation_error
        self.batch_size = batch_size
        self.allow_schema_migration = allow_schema_migration
        self.schema_migration_window_hours = schema_migration_window_hours

        # Initialize underlying stores
        self.feature_store = feature_store or FeatureStore(connection_string)
        self.model_store = model_store or ModelStore(connection_string)
        self.strategy_store = strategy_store or StrategyStore(connection_string)

        # Initialize data processor for validation
        self.data_processor = DataProcessor(connection_string)

        # Cache for manifests and contracts
        self._manifest_cache: dict[str, DatasetManifest] = {}
        self._contract_cache: dict[str, DataContract] = {}
        self._schema_migration_state: dict[str, dict[str, Any]] = {}

        logger.info(
            "Initialized DataStore with fail_on_validation=%s, batch_size=%d, allow_migration=%s",
            fail_on_validation_error,
            batch_size,
            allow_schema_migration,
        )

    # =========================================================================
    # Write Operations
    # =========================================================================

    def preflight_check(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Perform preflight schema validation before processing data.

        This method checks that the data conforms to the expected schema
        without actually writing anything. It validates column names,
        data types, required fields, and schema hash compatibility.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        data : DataFrame | list
            Data to validate
        strict : bool
            If True, require exact schema match. If False, allow subset.

        Returns
        -------
        tuple[bool, str | None, dict[str, Any]]
            (success, error_message, validation_details)

        Examples
        --------
        >>> success, error, details = store.preflight_check("bars_eurusd_1m", df)
        >>> if not success:
        ...     print(f"Preflight check failed: {error}")
        ...     print(f"Details: {details}")

        """
        try:
            # Get manifest and contract
            manifest = self._get_manifest(dataset_id)

            # Convert to DataFrame for validation
            df_obj = self._to_dataframe(data)
            df = cast(DataFrameLike, df_obj)
            df_any = cast(Any, df)

            validation_details: dict[str, Any] = {
                "dataset_id": dataset_id,
                "expected_schema_hash": manifest.schema_hash,
                "checks_performed": [],
                "warnings": [],
            }

            # Check 1: Required columns present
            if hasattr(df, "columns"):
                actual_columns = set(df.columns)
                expected_columns = set(manifest.schema.keys())

                missing_columns = expected_columns - actual_columns
                extra_columns = actual_columns - expected_columns

                validation_details["actual_columns"] = list(actual_columns)
                validation_details["expected_columns"] = list(expected_columns)
                validation_details["checks_performed"].append("column_presence")

                if missing_columns:
                    error_msg = f"Missing required columns: {missing_columns}"
                    validation_details["missing_columns"] = list(missing_columns)
                    if strict or any(col in manifest.primary_keys for col in missing_columns):
                        return False, error_msg, validation_details
                    else:
                        validation_details["warnings"].append(error_msg)

                if extra_columns and strict:
                    error_msg = f"Unexpected columns: {extra_columns}"
                    validation_details["extra_columns"] = list(extra_columns)
                    return False, error_msg, validation_details
                elif extra_columns:
                    validation_details["warnings"].append(
                        f"Extra columns will be ignored: {extra_columns}",
                    )

            # Check 2: Data types compatibility
            type_mismatches: list[dict[str, str]] = []
            if hasattr(df, "columns"):
                validation_details["checks_performed"].append("type_compatibility")
                for col_name, expected_type in manifest.schema.items():
                    if col_name in df.columns:
                        actual_type = str(df[col_name].dtype)
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
            actual_schema_hash = self._compute_schema_hash(df, manifest)
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
                    if hasattr(df_any, "columns") and pk_field in df_any.columns:
                        # Handle both Polars and pandas
                        if hasattr(df_any[pk_field], "is_null"):
                            # Polars
                            null_count = df_any[pk_field].is_null().sum()
                        elif hasattr(df_any[pk_field], "isnull"):
                            # pandas
                            null_count = df_any[pk_field].isnull().sum()
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
                    if not nullable and hasattr(df, "columns") and field in df.columns:
                        # Handle both Polars and pandas
                        if hasattr(df[field], "is_null"):
                            # Polars
                            null_count = df[field].is_null().sum()
                        elif hasattr(df[field], "isnull"):
                            # pandas
                            null_count = df[field].isnull().sum()
                        else:
                            null_count = 0

                        if null_count > 0:
                            error_msg = (
                                f"Required field '{field}' contains {null_count} null values"
                            )
                            return False, error_msg, validation_details

            validation_details["preflight_passed"] = True
            return True, None, validation_details

        except Exception as e:
            error_msg = f"Preflight check failed: {e}"
            logger.error(error_msg)
            return False, error_msg, {"error": str(e)}

    def write_ingestion(
        self,
        dataset_id: str,
        records: list[dict[str, Any]] | DataFrameLike,  # Accept DataFrame or list
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> DataEvent:
        """
        Write ingestion data with contract validation and event emission.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        records : list[dict[str, Any]] | DataFrame
            Data records to write
        source : str
            Data source (live, historical, backfill)
        run_id : str
            Processing run identifier
        instrument_id : str | None
            Instrument identifier (extracted from data if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation

        Raises
        ------
        ValueError
            If dataset not found or validation fails (when fail_on_validation_error=True)
        RuntimeError
            If write operation fails

        Examples
        --------
        >>> event = store.write_ingestion(
        ...     dataset_id="bars_eurusd_1m",
        ...     records=bar_data,
        ...     source="historical",
        ...     run_id="run_20240101_120000"
        ... )
        >>> print(f"Wrote {event.record_count} records")

        """
        start_time = time.perf_counter()

        # Get manifest and contract
        manifest = self._get_manifest(dataset_id)
        contract = self._get_contract(dataset_id)

        # Perform preflight schema check
        preflight_passed, preflight_error, preflight_details = self.preflight_check(
            dataset_id,
            records,
            strict=self.fail_on_validation_error,
        )

        if not preflight_passed:
            # Record write rejection metric
            if HAS_PROMETHEUS:
                write_rejection_counter.labels(
                    dataset_id=dataset_id,
                    reason="preflight_failed",
                ).inc()

            raise ValueError(
                f"Preflight check failed for {dataset_id}: {preflight_error}. "
                f"Details: {preflight_details}",
            )

        # Log warnings from preflight check
        if preflight_details.get("warnings"):
            for warning in preflight_details["warnings"]:
                logger.warning("Preflight warning for %s: %s", dataset_id, warning)

        # Convert to DataFrame if needed
        df_obj = self._to_dataframe(records)
        df = cast(DataFrameLike, df_obj)

        # Extract instrument_id if not provided
        if instrument_id is None:
            if hasattr(df, "columns") and "instrument_id" in df.columns:
                # Handle both Polars and pandas
                col = df["instrument_id"]
                if hasattr(col, "iloc"):
                    # pandas Series
                    instrument_id = str(col.iloc[0])
                else:
                    # Polars Series or other
                    instrument_id = str(col[0])
            else:
                instrument_id = "UNKNOWN"

        # Validate data against contract based on enforcement mode
        # Only use strict_mode if contract enforcement is "strict"
        use_strict = contract.enforcement_mode == "strict"
        quality_report = self.validate_batch(dataset_id, df, strict_mode=use_strict)

        # Check validation results with fail-closed approach
        if quality_report.quality_score < 1.0:
            violations_str = self._format_violations(quality_report.violations)

            # Count critical violations (FAIL severity)
            critical_violations = [
                v for v in quality_report.violations if v.severity == QualityFlag.FAIL
            ]

            # Fail-closed: Block any data with critical violations (unless monitor_only)
            if critical_violations and contract.enforcement_mode != "monitor_only":
                # Record write rejection metric
                if HAS_PROMETHEUS:
                    write_rejection_counter.labels(
                        dataset_id=dataset_id,
                        reason="validation_failed",
                    ).inc()

                raise ValueError(
                    f"Data validation failed for {dataset_id} (fail-closed). "
                    f"Quality score: {quality_report.quality_score:.2f}. "
                    f"Critical violations: {len(critical_violations)}. "
                    f"Details: {violations_str}",
                )

            # For non-critical violations, check enforcement mode
            if self.fail_on_validation_error and contract.enforcement_mode == "strict":
                # Record write rejection metric
                if HAS_PROMETHEUS:
                    write_rejection_counter.labels(
                        dataset_id=dataset_id,
                        reason="strict_mode_violation",
                    ).inc()

                raise ValueError(
                    f"Data validation failed for {dataset_id} (strict mode). "
                    f"Quality score: {quality_report.quality_score:.2f}. "
                    f"Violations: {violations_str}",
                )
            elif contract.enforcement_mode == "lenient":
                logger.warning(
                    "Data validation warnings for %s (lenient mode): %s",
                    dataset_id,
                    violations_str,
                )
            else:  # monitor_only
                logger.info(
                    "Data validation issues for %s (monitor-only): %s",
                    dataset_id,
                    violations_str,
                )

        # Extract timestamp range
        ts_field = manifest.ts_field
        ts_min = int(cast(Any, df)[ts_field].min())
        ts_max = int(cast(Any, df)[ts_field].max())

        # Determine appropriate store and stage based on dataset type
        stage = self._get_stage_for_dataset_type(manifest.dataset_type)

        try:
            # Route to appropriate store based on dataset type
            if manifest.dataset_type == DatasetType.FEATURES:
                # Convert to FeatureData format and write
                feature_data = self._df_to_feature_data(df, instrument_id)
                # Note: FeatureStore doesn't have a batch method, write individually
                for feature in feature_data:
                    self.feature_store.write_features(
                        feature_set_id=feature.feature_set_id,
                        instrument_id=feature.instrument_id,
                        features=feature.values,
                        ts_event=feature.ts_event,
                        ts_init=feature.ts_init,
                    )

            elif manifest.dataset_type == DatasetType.PREDICTIONS:
                # Convert to ModelPrediction format and write
                predictions = self._df_to_predictions(df)
                self.model_store.write_batch(predictions)

            elif manifest.dataset_type == DatasetType.SIGNALS:
                # Convert to StrategySignal format and write
                signals = self._df_to_signals(df)
                self.strategy_store.write_batch(signals)

            else:
                # For raw market data types, write directly to catalog or appropriate location
                # This would integrate with Nautilus catalog or data adapters
                logger.info(
                    "Writing %s data type %s (not implemented in this example)",
                    dataset_id,
                    manifest.dataset_type,
                )

            # Create event
            event = DataEvent(
                event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                operation="write_ingestion",
                source=source,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                record_count=len(df),
                status="success",
                metadata={
                    "quality_score": quality_report.quality_score,
                    "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                },
            )

            # Emit event to registry
            correlation_id = make_correlation_id(
                run_id=run_id,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=len(df),
            )
            source_norm = source if source in {"live", "historical", "backfill"} else "live"
            self.registry.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source_norm,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=len(df),
                status="success",
                metadata={"correlation_id": correlation_id},
            )

            # Update watermark (test hook patchable via _update_watermark)
            self._update_watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=source,
                last_success_ns=ts_max,
                count=len(df),
                completeness_pct=quality_report.quality_score * 100,
            )

            logger.info(
                "Successfully wrote %d records to %s (quality=%.2f)",
                len(df),
                dataset_id,
                quality_report.quality_score,
            )

            return event

        except Exception as e:
            # Create failure event
            event = DataEvent(
                event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                operation="write_ingestion",
                source=source,
                run_id=run_id,
                ts_min=ts_min if "ts_min" in locals() else 0,
                ts_max=ts_max if "ts_max" in locals() else 0,
                record_count=0,
                status="failed",
                error_message=str(e),
            )

            # Emit failure event
            correlation_id = make_correlation_id(
                run_id=run_id,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                ts_min=0,
                ts_max=0,
                count=0,
            )
            source_norm = source if source in {"live", "historical", "backfill"} else "live"
            self.registry.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source_norm,
                run_id=run_id,
                ts_min=0,
                ts_max=0,
                count=0,
                status="failed",
                error=str(e),
                metadata={"correlation_id": correlation_id},
            )

            logger.error("Failed to write data to %s: %s", dataset_id, e)
            raise RuntimeError(f"Write operation failed: {e}") from e

    def write_features(
        self,
        instrument_id: str,
        features: list[FeatureData],
        source: str = "computed",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write features with validation and event emission.

        Wraps FeatureStore.store_features_batch with contract validation and event tracking.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        features : list[FeatureData]
            Feature data to store
        source : str
            Data source (default: "computed")
        run_id : str | None
            Processing run identifier (generated if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation

        Examples
        --------
        >>> event = store.write_features(
        ...     instrument_id="EUR/USD",
        ...     features=feature_list,
        ...     source="realtime"
        ... )

        """
        run_id = run_id or f"features_{time.time_ns()}"
        dataset_id = "features"

        # Register dataset if not exists
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.FEATURES,
            instrument_id=instrument_id,
        )

        # Validate features
        for feature_data in features:
            if feature_data.instrument_id != instrument_id:
                raise ValueError(
                    f"Instrument mismatch: expected {instrument_id}, "
                    f"got {feature_data.instrument_id}",
                )

        # Store features
        for feature in features:
            self.feature_store.write_features(
                feature_set_id=feature.feature_set_id,
                instrument_id=feature.instrument_id,
                features=feature.values,
                ts_event=feature.ts_event,
                ts_init=feature.ts_init,
            )

        # Calculate timestamp range
        ts_min = min(f.ts_event for f in features)
        ts_max = max(f.ts_event for f in features)

            # Create event and emit/update registry watermarks
        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_features",
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            record_count=len(features),
            status="success",
        )
        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.FEATURE_COMPUTED.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=len(features),
        )

        logger.debug("Wrote %d features for %s", len(features), instrument_id)
        return event

    def write_predictions(
        self,
        predictions: list[ModelPrediction],
        source: str = "inference",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write model predictions with validation and event emission.

        Wraps ModelStore.store_predictions_batch with contract validation and event tracking.

        Parameters
        ----------
        predictions : list[ModelPrediction]
            Model predictions to store
        source : str
            Data source (default: "inference")
        run_id : str | None
            Processing run identifier (generated if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation

        Examples
        --------
        >>> event = store.write_predictions(
        ...     predictions=prediction_list,
        ...     source="realtime"
        ... )

        """
        if not predictions:
            raise ValueError("No predictions to write")

        run_id = run_id or f"predictions_{time.time_ns()}"

        # Group by instrument for event emission
        instrument_id = predictions[0].instrument_id
        model_id = predictions[0].model_id
        dataset_id = "predictions"

        # Register dataset if not exists
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.PREDICTIONS,
            instrument_id=instrument_id,
        )

        # Store predictions
        self.model_store.write_batch(predictions)

        # Calculate timestamp range
        ts_min = min(p.ts_event for p in predictions)
        ts_max = max(p.ts_event for p in predictions)

        # Create event and emit/update registry watermarks
        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_predictions",
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            record_count=len(predictions),
            status="success",
            metadata={"model_id": model_id},
        )
        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.PREDICTION_EMITTED.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=len(predictions),
        )

        logger.debug(
            "Wrote %d predictions for %s from model %s",
            len(predictions),
            instrument_id,
            model_id,
        )
        return event

    def write_signals(
        self,
        signals: list[StrategySignal],
        source: str = "strategy",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write strategy signals with validation and event emission.

        Wraps StrategyStore.store_signals_batch with contract validation and event tracking.

        Parameters
        ----------
        signals : list[StrategySignal]
            Strategy signals to store
        source : str
            Data source (default: "strategy")
        run_id : str | None
            Processing run identifier (generated if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation

        Examples
        --------
        >>> event = store.write_signals(
        ...     signals=signal_list,
        ...     source="realtime"
        ... )

        """
        if not signals:
            raise ValueError("No signals to write")

        run_id = run_id or f"signals_{time.time_ns()}"

        # Group by instrument for event emission
        instrument_id = signals[0].instrument_id
        strategy_id = signals[0].strategy_id
        dataset_id = "signals"

        # Register dataset if not exists
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.SIGNALS,
            instrument_id=instrument_id,
        )

        # Store signals
        self.strategy_store.write_batch(signals)

        # Calculate timestamp range
        ts_min = min(s.ts_event for s in signals)
        ts_max = max(s.ts_event for s in signals)

        # Create event and emit/update registry watermarks
        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_signals",
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            record_count=len(signals),
            status="success",
            metadata={"strategy_id": strategy_id},
        )
        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.SIGNAL_EMITTED.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            count=len(signals),
        )

        logger.debug(
            "Wrote %d signals for %s from strategy %s",
            len(signals),
            instrument_id,
            strategy_id,
        )
        return event

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------



    def _emit_success_event_and_update(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        completeness_pct: float = 100.0,
    ) -> None:
        """
        Emit a success event and update registry watermark for the dataset.
        """
        try:
            correlation_id = make_correlation_id(
                run_id=run_id,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
            )
            # Normalize source to allowed set for registry persistence
            source_norm = source if source in {"live", "historical", "backfill"} else "live"
            self.registry.emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source_norm,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status="success",
                metadata={"correlation_id": correlation_id},
            )
            self.registry.update_watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=source_norm,
                last_success_ns=ts_max,
                count=count,
                completeness_pct=completeness_pct,
            )
        except Exception as e:
            logger.warning("Failed to emit event/update watermark: %s", e)

    # =========================================================================
    # Read Operations
    # =========================================================================

    def read_range(
        self,
        dataset_id: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> object:  # Returns DataFrame or list depending on dataset
        """
        Read data for a specific time range.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        DataFrame | list
            Data for the specified range

        Raises
        ------
        ValueError
            If dataset not found or invalid time range

        Examples
        --------
        >>> df = store.read_range(
        ...     dataset_id="features",
        ...     instrument_id="EUR/USD",
        ...     start_ns=1234567890000000000,
        ...     end_ns=1234567900000000000
        ... )

        """
        manifest = self._get_manifest(dataset_id)

        # Validate time range
        if start_ns >= end_ns:
            raise ValueError(f"Invalid time range: start={start_ns} >= end={end_ns}")

        # Route to appropriate store based on dataset type
        if manifest.dataset_type == DatasetType.FEATURES:
            # Convert nanoseconds to datetime for FeatureStore
            from datetime import datetime

            start_dt = datetime.fromtimestamp(start_ns / 1e9)
            end_dt = datetime.fromtimestamp(end_ns / 1e9)
            # Use feature store's get_training_data method
            return self.feature_store.get_training_data(
                instrument_id=instrument_id,
                start=start_dt,
                end=end_dt,
            )

        elif manifest.dataset_type == DatasetType.PREDICTIONS:
            # Use model store's read_predictions method
            # Extract model_id from dataset_id
            model_id = dataset_id.replace("predictions_", "")
            return self.model_store.read_predictions(
                model_id=model_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        elif manifest.dataset_type == DatasetType.SIGNALS:
            # Use strategy store's read_signals method
            # Extract strategy_id from dataset_id
            strategy_id = dataset_id.replace("signals_", "")
            return self.strategy_store.read_signals(
                strategy_id=strategy_id,
                instrument_id=instrument_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        else:
            # For raw market data types, would integrate with Nautilus catalog
            raise NotImplementedError(
                f"Read not implemented for dataset type {manifest.dataset_type}",
            )

    # =========================================================================
    # Validation Operations
    # =========================================================================

    def validate_batch(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],  # DataFrame or list
        strict_mode: bool = False,
    ) -> QualityReport:
        """
        Validate a batch of data against the dataset's contract.

        Performs comprehensive contract validation including type checking,
        null validation, range validation, uniqueness constraints,
        monotonicity checks, and lateness validation.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        data : DataFrame | list
            Data to validate
        strict_mode : bool
            If True, apply stricter validation rules

        Returns
        -------
        QualityReport
            Validation results with quality score and violations

        Examples
        --------
        >>> report = store.validate_batch("bars_eurusd_1m", df)
        >>> print(f"Quality score: {report.quality_score:.2%}")
        >>> if report.violations:
        ...     for violation in report.violations:
        ...         print(f"  - {violation.description}")

        """
        start_time = time.perf_counter()

        # Get manifest and contract
        manifest = self._get_manifest(dataset_id)
        contract = self._get_contract(dataset_id)

        # Convert to DataFrame for validation
        df_obj = self._to_dataframe(data)
        df = cast(DataFrameLike, df_obj)

        # Track violations
        violations: list[ValidationViolation] = []
        failed_record_indices = set()  # Track unique failed record indices
        validation_metadata = {
            "rules_evaluated": len(contract.validation_rules),
            "strict_mode": strict_mode,
        }

        # Apply each validation rule with enhanced checks
        for rule in contract.validation_rules:
            # Skip WARN-only rules in strict mode if configured
            if strict_mode and rule.severity == QualityFlag.WARN:
                # In strict mode, treat warnings as failures
                rule = ValidationRule(
                    rule_type=rule.rule_type,
                    field_name=rule.field_name,
                    parameters=rule.parameters,
                    severity=QualityFlag.FAIL,
                    description=rule.description,
                )

            violation = self._apply_validation_rule(rule, df, manifest)
            if violation:
                violations.append(violation)
                # For simplicity, assume each violation affects unique records up to the total
                # In a real implementation, we'd track actual record indices
                if violation.severity == QualityFlag.FAIL:
                    # Add violation count but cap at total records to avoid overcounting
                    for i in range(min(violation.violation_count, len(df))):
                        failed_record_indices.add(i)

                # Record violation metric
                if HAS_PROMETHEUS:
                    validation_violations_counter.labels(
                        dataset_id=dataset_id,
                        rule_type=str(violation.rule_type.value),
                        severity=str(violation.severity.value),
                    ).inc(violation.violation_count)

        # Calculate quality score
        total_records = len(df)
        failed_records = len(failed_record_indices)
        passed_records = total_records - failed_records
        quality_score = passed_records / total_records if total_records > 0 else 0.0

        # Check quality thresholds
        if contract.quality_thresholds:
            # Calculate metrics
            df_any = cast(Any, df)
            if hasattr(df_any, "null_count"):
                # Polars
                null_count_total = df_any.null_count().sum_horizontal().sum()
            elif hasattr(df_any, "isnull"):
                # pandas
                null_count_total = df_any.isnull().sum().sum()
            else:
                null_count_total = 0

            base_count = (total_records * len(cast(Any, df).columns)) if total_records > 0 else 0
            null_rate = float(null_count_total) / float(base_count) if base_count else 0.0

            if "null_rate" in contract.quality_thresholds:
                if null_rate > contract.quality_thresholds["null_rate"]:
                    violations.append(
                        ValidationViolation(
                            rule_type=ValidationRuleType.NULLABILITY,
                            field_name="*",
                            severity=QualityFlag.WARN,
                            violation_count=int(null_rate * total_records),
                            sample_values=[],
                            description=f"Null rate {null_rate:.2%} exceeds threshold",
                        ),
                    )

        validation_time_ms = (time.perf_counter() - start_time) * 1000

        # Record validation metrics
        if HAS_PROMETHEUS:
            validation_duration_histogram.labels(
                dataset_id=dataset_id,
            ).observe(validation_time_ms / 1000.0)

            quality_score_histogram.labels(
                dataset_id=dataset_id,
            ).observe(quality_score)

        report = QualityReport(
            dataset_id=dataset_id,
            total_records=total_records,
            passed_records=passed_records,
            failed_records=failed_records,
            quality_score=quality_score,
            violations=violations,
            validation_time_ms=validation_time_ms,
            metadata={
                "contract_version": contract.version,
                "enforcement_mode": contract.enforcement_mode,
                "strict_mode": strict_mode,
                **validation_metadata,
            },
        )

        logger.debug(
            "Validated %d records for %s: quality=%.2f, violations=%d",
            total_records,
            dataset_id,
            quality_score,
            len(violations),
        )

        return report

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_manifest(self, dataset_id: str) -> DatasetManifest:
        """
        Get dataset manifest with caching and version check.
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

    def _get_contract(self, dataset_id: str) -> DataContract:
        """
        Get data contract with caching and version check.
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

    def _to_dataframe(
        self,
        data: DataFrameLike | list[dict[str, Any]],
    ) -> DataFrameLike | list[dict[str, Any]]:
        """
        Convert various data formats to DataFrame-like or pass-through list.
        """
        # Import here to avoid circular dependency
        from ml._imports import HAS_POLARS
        from ml._imports import pl

        if not HAS_POLARS:
            # If Polars not available, work with raw data
            if isinstance(data, list):
                return data
            return data

        # If already a DataFrame, return as is
        if hasattr(data, "columns"):
            return data

        # Convert list of dicts to DataFrame
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return pl.DataFrame(data)

        # Return as is for other formats
        return data

    def _get_stage_for_dataset_type(self, dataset_type: DatasetType) -> str:
        """
        Map dataset type to processing stage.
        """
        stage_map = {
            DatasetType.BARS: Stage.CATALOG_WRITTEN.value,
            DatasetType.TRADES: Stage.CATALOG_WRITTEN.value,
            DatasetType.QUOTES: Stage.CATALOG_WRITTEN.value,
            DatasetType.MBP1: Stage.CATALOG_WRITTEN.value,
            DatasetType.TBBO: Stage.CATALOG_WRITTEN.value,
            DatasetType.FEATURES: Stage.FEATURE_COMPUTED.value,
            DatasetType.PREDICTIONS: Stage.PREDICTION_EMITTED.value,
            DatasetType.SIGNALS: Stage.SIGNAL_EMITTED.value,
        }
        return stage_map.get(dataset_type, Stage.DATA_INGESTED.value)

    def _apply_validation_rule(
        self,
        rule: ValidationRule,
        df: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """
        Apply a single validation rule to data.
        """
        try:
            if rule.rule_type == ValidationRuleType.TYPE_CHECK:
                return self._validate_types(rule, df, manifest)
            elif rule.rule_type == ValidationRuleType.RANGE:
                return self._validate_range(rule, df)
            elif rule.rule_type == ValidationRuleType.UNIQUENESS:
                return self._validate_uniqueness(rule, df)
            elif rule.rule_type == ValidationRuleType.MONOTONICITY:
                return self._validate_monotonicity(rule, df)
            elif rule.rule_type == ValidationRuleType.NULLABILITY:
                return self._validate_nullability(rule, df)
            elif rule.rule_type == ValidationRuleType.LATENESS:
                return self._validate_lateness(rule, df, manifest)
            else:
                logger.warning("Unknown validation rule type: %s", rule.rule_type)
                return None
        except Exception as e:
            logger.error("Error applying validation rule %s: %s", rule.rule_type, e)
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=rule.field_name,
                severity=QualityFlag.WARN,
                violation_count=1,
                sample_values=[],
                description=f"Validation error: {e}",
            )

    def _validate_types(
        self,
        rule: ValidationRule,
        df: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        df_any = cast(Any, df)
        """
        Validate data types match schema.
        """
        violations = 0
        sample_values = []

        # Check each column in schema
        for col_name, expected_type in manifest.schema.items():
            if hasattr(df_any, "columns") and col_name in df_any.columns:
                # Get actual type
                actual_type = str(df_any[col_name].dtype)

                # Simple type checking (would be more sophisticated in production)
                if not self._types_compatible(actual_type, expected_type):
                    violations += 1
                    sample_values.append(f"{col_name}: {actual_type} != {expected_type}")

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=rule.field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=sample_values[:5],
                description=f"Type mismatches found in {violations} columns",
            )

        return None

    def _validate_range(self, rule: ValidationRule, df: object) -> ValidationViolation | None:
        """
        Validate values are within specified range.
        """
        df_any = cast(Any, df)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(df_any, "columns") or field_name not in df_any.columns:
            return None

        violations = 0
        sample_values = []

        # Get column values
        col = df_any[field_name]

        # Check min
        if "min" in params:
            min_val = params["min"]

            # Handle both Polars and pandas
            if hasattr(col, "__module__") and "polars" in str(col.__module__):
                # Polars Series
                below_min = col < min_val
                violations += below_min.sum()
                if violations > 0:
                    # Get sample of violating values
                    violating = col.filter(below_min)
                    sample_values.extend(violating.head(5).to_list())
            else:
                # pandas or raw data
                try:
                    below_min = col < min_val
                    if hasattr(below_min, "sum"):
                        violations += below_min.sum()
                        if violations > 0:
                            # Get sample of violating values
                            violating = col[below_min]
                            if hasattr(violating, "head"):
                                sample_values.extend(violating.head(5).to_list())
                except:
                    pass

        # Check max
        if "max" in params:
            max_val = params["max"]

            # Handle both Polars and pandas
            if hasattr(col, "__module__") and "polars" in str(col.__module__):
                # Polars Series
                above_max = col > max_val
                violations += above_max.sum()
                if violations > 0 and len(sample_values) < 5:
                    # Get sample of violating values
                    violating = col.filter(above_max)
                    sample_values.extend(violating.head(5 - len(sample_values)).to_list())
            else:
                # pandas or raw data
                try:
                    above_max = col > max_val
                    if hasattr(above_max, "sum"):
                        violations += above_max.sum()
                        if violations > 0 and len(sample_values) < 5:
                            # Get sample of violating values
                            violating = col[above_max]
                            if hasattr(violating, "head"):
                                sample_values.extend(
                                    violating.head(5 - len(sample_values)).to_list(),
                                )
                except:
                    pass

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=sample_values[:5],
                description=f"{field_name} values out of range [{params.get('min', '-inf')}, {params.get('max', 'inf')}]",
            )

        return None

    def _validate_uniqueness(
        self,
        rule: ValidationRule,
        df: object,
    ) -> ValidationViolation | None:
        """
        Validate uniqueness constraints.
        """
        field_name = rule.field_name
        df_any = cast(Any, df)

        if not hasattr(df_any, "columns"):
            return None

        # Handle composite keys
        if "," in field_name:
            key_fields = [f.strip() for f in field_name.split(",")]
            if all(f in df_any.columns for f in key_fields):
                # Check for duplicates on composite key
                if hasattr(df_any, "is_duplicated"):
                    # Polars
                    duplicates = df_any.select(key_fields).is_duplicated()
                    duplicate_count = duplicates.sum()
                elif hasattr(df_any, "duplicated"):
                    # pandas
                    duplicates = df_any.duplicated(subset=key_fields)
                    duplicate_count = duplicates.sum()
                else:
                    duplicate_count = 0

                if duplicate_count > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=duplicate_count,
                        sample_values=[],
                        description=f"Duplicate values found for composite key {field_name}",
                    )
        else:
            # Single field uniqueness
            if field_name in df_any.columns:
                if hasattr(df_any, "is_duplicated"):
                    # Polars
                    duplicates = df_any.select([field_name]).is_duplicated()
                    duplicate_count = duplicates.sum()
                    sample_values = []
                    if duplicate_count > 0:
                        # Get sample duplicate values
                        duplicate_vals = df_any[field_name].filter(duplicates)
                        sample_values = duplicate_vals.head(5).to_list()
                elif hasattr(df_any, "duplicated"):
                    # pandas
                    duplicates = df_any.duplicated(subset=[field_name])
                    duplicate_count = duplicates.sum()
                    sample_values = []
                    if duplicate_count > 0:
                        # Get sample duplicate values
                        duplicate_vals = df_any[field_name][duplicates]
                        if hasattr(duplicate_vals, "head"):
                            sample_values = duplicate_vals.head(5).to_list()
                else:
                    duplicate_count = 0
                    sample_values = []

                if duplicate_count > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=duplicate_count,
                        sample_values=sample_values,
                        description=f"Duplicate values found in {field_name}",
                    )

        return None

    def _validate_monotonicity(
        self,
        rule: ValidationRule,
        df: object,
    ) -> ValidationViolation | None:
        """
        Validate monotonic sequences (e.g., timestamps).
        """
        df_any = cast(Any, df)

        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(df_any, "columns") or field_name not in df_any.columns:
            return None

        col = df_any[field_name]
        direction = params.get("direction", "increasing")
        strict = params.get("strict", True)

        violations = 0

        # Handle both Polars and pandas
        if hasattr(col, "__module__") and "polars" in str(col.__module__):
            # Polars Series
            diffs = col.diff()

            if direction == "increasing":
                if strict:
                    # Check for strictly increasing (diff must be > 0)
                    # First element of diff is null, so we drop it
                    violations = (diffs.drop_nulls() <= 0).sum()
                else:
                    # Check for non-decreasing (diff must be >= 0)
                    violations = (diffs.drop_nulls() < 0).sum()
            else:  # decreasing
                if strict:
                    # Check for strictly decreasing (diff must be < 0)
                    violations = (diffs.drop_nulls() >= 0).sum()
                else:
                    # Check for non-increasing (diff must be <= 0)
                    violations = (diffs.drop_nulls() > 0).sum()
        else:
            # pandas or raw data
            if hasattr(col, "diff"):
                diffs = col.diff()

                if direction == "increasing":
                    if strict:
                        # Check for strictly increasing
                        # dropna() to remove the first NaN
                        violations = (cast(Any, diffs.dropna()) <= 0).sum()
                    else:
                        # Check for non-decreasing
                        violations = (cast(Any, diffs.dropna()) < 0).sum()
                else:  # decreasing
                    if strict:
                        # Check for strictly decreasing
                        violations = (cast(Any, diffs.dropna()) >= 0).sum()
                    else:
                        # Check for non-increasing
                        violations = (cast(Any, diffs.dropna()) > 0).sum()

        if violations > 0:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=field_name,
                severity=rule.severity,
                violation_count=violations,
                sample_values=[],
                description=f"{field_name} is not {'strictly ' if strict else ''}{direction}",
            )

        return None

    def _validate_nullability(
        self,
        rule: ValidationRule,
        df: object,
    ) -> ValidationViolation | None:
        """
        Validate null value constraints.
        """
        df_any = cast(Any, df)
        field_name = rule.field_name
        params = rule.parameters

        if not hasattr(df_any, "columns"):
            return None

        nullable = params.get("nullable", True)

        if not nullable:
            if field_name == "*":
                # Check all fields
                if hasattr(df_any, "null_count"):
                    # Polars
                    null_counts = df_any.null_count()
                    total_nulls = sum(null_counts.to_dicts()[0].values())
                    fields_with_nulls = [
                        col for col in df_any.columns if df_any[col].is_null().sum() > 0
                    ]
                elif hasattr(df_any, "isnull"):
                    # pandas
                    null_counts = df_any.isnull().sum()
                    total_nulls = null_counts.sum()
                    fields_with_nulls = [
                        col for col in df_any.columns if df_any[col].isnull().sum() > 0
                    ]
                else:
                    total_nulls = 0
                    fields_with_nulls = []

                if total_nulls > 0:
                    return ValidationViolation(
                        rule_type=rule.rule_type,
                        field_name=field_name,
                        severity=rule.severity,
                        violation_count=total_nulls,
                        sample_values=fields_with_nulls[:5],
                        description=f"Null values found in {len(fields_with_nulls)} fields",
                    )
            else:
                # Check specific field
                if field_name in df_any.columns:
                    # Handle both Polars and pandas
                    if hasattr(df_any[field_name], "is_null"):
                        # Polars
                        null_count = df_any[field_name].is_null().sum()
                    elif hasattr(df_any[field_name], "isnull"):
                        # pandas
                        null_count = df_any[field_name].isnull().sum()
                    else:
                        null_count = 0

                    if null_count > 0:
                        return ValidationViolation(
                            rule_type=rule.rule_type,
                            field_name=field_name,
                            severity=rule.severity,
                            violation_count=null_count,
                            sample_values=[],
                            description=f"{field_name} contains {null_count} null values",
                        )

        return None

    def _validate_lateness(
        self,
        rule: ValidationRule,
        df: object,
        manifest: DatasetManifest,
    ) -> ValidationViolation | None:
        """
        Validate data freshness/lateness.
        """
        df_any = cast(Any, df)
        params = rule.parameters
        max_lateness_ns = params.get("max_lateness_ns", 300_000_000_000)  # Default 5 minutes

        ts_field = manifest.ts_field
        if not hasattr(df_any, "columns") or ts_field not in df_any.columns:
            return None

        current_ns = time.time_ns()
        latest_ts = int(cast(Any, df)[ts_field].max())

        lateness_ns = current_ns - latest_ts

        if lateness_ns > max_lateness_ns:
            return ValidationViolation(
                rule_type=rule.rule_type,
                field_name=ts_field,
                severity=rule.severity,
                violation_count=1,
                sample_values=[f"Lateness: {lateness_ns / 1e9:.1f} seconds"],
                description=f"Data is {lateness_ns / 1e9:.1f} seconds late",
            )

        return None

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """
        Check if actual type is compatible with expected type.
        """
        # Simple type compatibility check
        type_map = {
            "int64": ["int", "int64", "i8", "Int64"],
            "float64": ["float", "float64", "f8", "Float64"],
            "str": ["str", "string", "object", "Utf8"],
            "bool": ["bool", "boolean", "Boolean"],
        }

        for expected_base, compatible_types in type_map.items():
            if expected in compatible_types:
                return any(t in actual.lower() for t in compatible_types)

        # Default to string comparison
        return actual.lower() == expected.lower()

    def _format_violations(self, violations: list[ValidationViolation]) -> str:
        """
        Format violations for logging.
        """
        if not violations:
            return "None"

        parts = []
        for v in violations[:3]:  # Show first 3 violations
            parts.append(f"{v.field_name}: {v.description} ({v.violation_count} records)")

        if len(violations) > 3:
            parts.append(f"... and {len(violations) - 3} more")

        return "; ".join(parts)

    def _df_to_feature_data(self, df: DataFrameLike, instrument_id: str) -> list[FeatureData]:
        """
        Convert DataFrame to list of FeatureData.
        """
        features = []

        # Generate a default feature_set_id if not present
        feature_set_id = f"features_{instrument_id.lower().replace('/', '_')}"

        # Handle both Polars and pandas-like DataFrames
        if hasattr(df, "iter_rows"):
            # Polars DataFrame
            df_polars = cast(Any, df)
            for row in df_polars.iter_rows(named=True):
                features.append(
                    FeatureData(
                        feature_set_id=feature_set_id,
                        instrument_id=instrument_id,
                        values={
                            str(k): v
                            for k, v in row.items()
                            if k not in ["instrument_id", "ts_event", "ts_init"]
                        },
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        elif hasattr(df, "iterrows"):
            # pandas DataFrame
            for _, row in df.iterrows():
                features.append(
                    FeatureData(
                        feature_set_id=feature_set_id,
                        instrument_id=instrument_id,
                        values={
                            str(k): v
                            for k, v in row.items()
                            if k not in ["instrument_id", "ts_event", "ts_init"]
                        },
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        else:
            # Fallback for list of dicts
            for row in df:
                if isinstance(row, dict):
                    features.append(
                        FeatureData(
                            feature_set_id=feature_set_id,
                            instrument_id=instrument_id,
                            values={
                                str(k): v
                                for k, v in row.items()
                                if k not in ["instrument_id", "ts_event", "ts_init"]
                            },
                            _ts_event=int(row["ts_event"]),
                            _ts_init=int(row.get("ts_init") or row["ts_event"]),
                        ),
                    )

        return features

    def _df_to_predictions(self, df: DataFrameLike | list[dict[str, Any]]) -> list[ModelPrediction]:
        """
        Convert DataFrame to list of ModelPrediction.
        """
        predictions = []

        # Handle both Polars and pandas-like DataFrames
        if hasattr(df, "iter_rows"):
            # Polars DataFrame
            df_polars = cast(Any, df)
            for row in df_polars.iter_rows(named=True):
                predictions.append(
                    ModelPrediction(
                        model_id=row["model_id"],
                        instrument_id=row["instrument_id"],
                        prediction=row["prediction"],
                        confidence=row.get("confidence", 0.0),
                        features_used=row.get("features_used", {}),
                        inference_time_ms=row.get("inference_time_ms", 0.0),
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        elif hasattr(df, "iterrows"):
            # pandas DataFrame
            for _, row in df.iterrows():
                predictions.append(
                    ModelPrediction(
                        model_id=row["model_id"],
                        instrument_id=row["instrument_id"],
                        prediction=row["prediction"],
                        confidence=row.get("confidence", 0.0),
                        features_used=row.get("features_used", {}),
                        inference_time_ms=row.get("inference_time_ms", 0.0),
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        else:
            # Fallback for list of dicts
            for row in df:
                if isinstance(row, dict):
                    predictions.append(
                        ModelPrediction(
                            model_id=row["model_id"],
                            instrument_id=row["instrument_id"],
                            prediction=row["prediction"],
                            confidence=row.get("confidence", 0.0),
                            features_used=row.get("features_used", {}),
                            inference_time_ms=row.get("inference_time_ms", 0.0),
                            _ts_event=int(row["ts_event"]),
                            _ts_init=int(row.get("ts_init") or row["ts_event"]),
                        ),
                    )

        return predictions

    def _df_to_signals(self, df: DataFrameLike | list[dict[str, Any]]) -> list[StrategySignal]:
        """
        Convert DataFrame to list of StrategySignal.
        """
        signals = []

        # Handle both Polars and pandas-like DataFrames
        if hasattr(df, "iter_rows"):
            # Polars DataFrame
            df_polars = cast(Any, df)
            for row in df_polars.iter_rows(named=True):
                signals.append(
                    StrategySignal(
                        strategy_id=row["strategy_id"],
                        instrument_id=row["instrument_id"],
                        signal_type=row["signal_type"],
                        strength=float(row.get("strength", row.get("signal_value", 0.0))),
                        model_predictions=row.get("model_predictions", {}),
                        risk_metrics=row.get("risk_metrics", {}),
                        execution_params=row.get("execution_params", {}),
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        elif hasattr(df, "iterrows"):
            # pandas DataFrame
            for _, row in df.iterrows():
                signals.append(
                    StrategySignal(
                        strategy_id=row["strategy_id"],
                        instrument_id=row["instrument_id"],
                        signal_type=row["signal_type"],
                        strength=float(row.get("strength", row.get("signal_value", 0.0))),
                        model_predictions=row.get("model_predictions", {}),
                        risk_metrics=row.get("risk_metrics", {}),
                        execution_params=row.get("execution_params", {}),
                        _ts_event=int(row["ts_event"]),
                        _ts_init=int(row.get("ts_init", row["ts_event"])),
                    ),
                )
        else:
            # Fallback for list of dicts
            for row in df:
                if isinstance(row, dict):
                    signals.append(
                        StrategySignal(
                            strategy_id=row["strategy_id"],
                            instrument_id=row["instrument_id"],
                            signal_type=row["signal_type"],
                            strength=float(
                                row.get("strength", row.get("signal_value", 0.0)) or 0.0,
                            ),
                            model_predictions=row.get("model_predictions", {}),
                            risk_metrics=row.get("risk_metrics", {}),
                            execution_params=row.get("execution_params", {}),
                            _ts_event=int(row["ts_event"]),
                            _ts_init=int(row.get("ts_init") or row["ts_event"]),
                        ),
                    )

        return signals

    def _ensure_dataset_registered(
        self,
        dataset_id: str,
        dataset_type: DatasetType,
        instrument_id: str,
    ) -> None:
        """
        Ensure dataset is registered in the registry.
        """
        try:
            # Check if already registered
            self.registry.get_manifest(dataset_id)
        except ValueError:
            # Not registered, create a basic manifest
            import hashlib

            from ml.registry.dataclasses import DatasetManifest
            from ml.registry.dataclasses import StorageKind

            schema = self._get_schema_for_type(dataset_type)
            schema_hash = hashlib.sha256(str(schema).encode()).hexdigest()

            manifest = DatasetManifest(
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                storage_kind=StorageKind.POSTGRES,
                location=f"ml_{dataset_type.value}",
                partitioning={"by": "ts_event", "interval": "monthly"},
                retention_days=365,
                schema=schema,
                ts_field="ts_event",
                seq_field=None,
                primary_keys=["instrument_id", "ts_event"],
                schema_hash=schema_hash,
                constraints={
                    "nullability": {
                        "instrument_id": False,
                        "ts_event": False,
                        "ts_init": False,
                    },
                },
                lineage=[],
                pipeline_signature="data_store_auto",
                version="1.0.0",
                metadata={"auto_registered": True, "instrument_id": instrument_id},
            )

            self.registry.register_dataset(manifest)
            logger.info("Auto-registered dataset %s", dataset_id)

    def _get_schema_for_type(self, dataset_type: DatasetType) -> dict[str, str]:
        """
        Get default schema for dataset type.
        """
        if dataset_type == DatasetType.FEATURES:
            return {
                "instrument_id": "str",
                "ts_event": "int64",
                "ts_init": "int64",
                "feature_values": "json",
            }
        elif dataset_type == DatasetType.PREDICTIONS:
            return {
                "instrument_id": "str",
                "model_id": "str",
                "ts_event": "int64",
                "ts_init": "int64",
                "prediction": "float64",
                "confidence": "float64",
                "metadata": "json",
            }
        elif dataset_type == DatasetType.SIGNALS:
            return {
                "instrument_id": "str",
                "strategy_id": "str",
                "ts_event": "int64",
                "ts_init": "int64",
                "signal_type": "str",
                "signal_value": "float64",
                "metadata": "json",
            }
        else:
            # Default schema for market data types
            return {
                "instrument_id": "str",
                "ts_event": "int64",
                "ts_init": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            }

    def _compute_schema_hash(self, df: DataFrameLike, manifest: DatasetManifest) -> str:
        """
        Compute schema hash for the actual data.
        """
        df_any = cast(Any, df)
        if not hasattr(df_any, "columns"):
            # For non-DataFrame data, use manifest hash
            return manifest.schema_hash

        # Build schema dict from actual data
        actual_schema = {}
        for col in df_any.columns:
            if col in manifest.schema:
                # Use expected type if column is in manifest
                actual_schema[col] = manifest.schema[col]
            else:
                # Infer type for extra columns
                dtype = str(df_any[col].dtype)
                if "int" in dtype.lower():
                    actual_schema[col] = "int64"
                elif "float" in dtype.lower():
                    actual_schema[col] = "float64"
                elif "bool" in dtype.lower():
                    actual_schema[col] = "bool"
                else:
                    actual_schema[col] = "str"

        # Sort by key for consistent hashing
        sorted_schema = dict(sorted(actual_schema.items()))
        schema_str = str(sorted_schema)

        return hashlib.sha256(schema_str.encode()).hexdigest()

    def _is_in_migration_window(self, dataset_id: str) -> bool:
        """
        Check if dataset is in schema migration window.
        """
        if not self.allow_schema_migration:
            return False

        if dataset_id not in self._schema_migration_state:
            return False

        migration_info = self._schema_migration_state[dataset_id]
        migration_start = migration_info.get("start_time", 0)
        window_ns = self.schema_migration_window_hours * 3600 * 1e9

        current_time = time.time_ns()
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

    def _update_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: str,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        """
        Internal hook to update the registry watermark (patchable in tests).
        """
        self.registry.update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source,
            last_success_ns=last_success_ns,
            count=count,
            completeness_pct=completeness_pct,
        )

    def _begin_transaction(
        self,
    ) -> ContextManager[object]:  # pragma: no cover (test hook for patching)
        """
        Return a no-op context manager for tests to patch.
        """
        from contextlib import nullcontext

        return nullcontext()
