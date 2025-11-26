#!/usr/bin/env python3

"""
Protocol definitions for DataStore components.

Defines structural interfaces for schema validation, data writing,
data reading, event emission, contract enforcement, and store operations.

Phase 2.4.1 - Component Decomposition

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from ml.ml_types import DataFrameLike
    from ml.registry.dataclasses import DatasetManifest
    from ml.registry.dataclasses import ValidationRule


class SchemaValidatorProtocol(Protocol):
    """
    Protocol for schema validation component.

    Provides pre-write schema validation, batch validation against contracts, and
    individual validation rules for types, ranges, uniqueness, monotonicity,
    nullability, and data freshness.

    """

    def preflight_check(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict: bool = True,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Perform preflight schema validation before processing data.

        Args:
            dataset_id: Dataset identifier
            data: Data to validate
            strict: If True, require exact schema match. If False, allow subset.

        Returns:
            (success, error_message, validation_details)

        """
        ...

    def validate_batch(
        self,
        dataset_id: str,
        data: DataFrameLike | list[dict[str, Any]],
        strict_mode: bool = False,
    ) -> Any:  # QualityReport
        """
        Validate a batch of data against the dataset's contract.

        Args:
            dataset_id: Dataset identifier
            data: Data to validate
            strict_mode: If True, apply stricter validation rules

        Returns:
            QualityReport with quality score and violations

        """
        ...

    def _validate_types(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> Any | None:  # ValidationViolation | None
        """
        Validate data types match schema.
        """
        ...

    def _validate_regex(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> Any | None:  # ValidationViolation | None
        """
        Validate a column against a regex pattern.
        """
        ...

    def _validate_range(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> Any | None:  # ValidationViolation | None
        """
        Validate values are within specified range.
        """
        ...

    def _validate_uniqueness(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> Any | None:  # ValidationViolation | None
        """
        Validate uniqueness constraints.
        """
        ...

    def _validate_monotonicity(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> Any | None:  # ValidationViolation | None
        """
        Validate monotonic sequences (e.g., timestamps).
        """
        ...

    def _validate_nullability(
        self,
        rule: ValidationRule,
        data_frame: object,
    ) -> Any | None:  # ValidationViolation | None
        """
        Validate null value constraints.
        """
        ...

    def _validate_lateness(
        self,
        rule: ValidationRule,
        data_frame: object,
        manifest: DatasetManifest,
    ) -> Any | None:  # ValidationViolation | None
        """
        Validate data freshness/lateness.
        """
        ...


class DataWriterProtocol(Protocol):
    """
    Protocol for data writer component.

    Provides write operations for ingestion data, features, predictions,
    signals, and earnings data with validation and event emission.

    """

    def write_ingestion(
        self,
        dataset_id: str,
        records: Any,  # list[dict[str, Any]] | DataFrameLike
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Write ingestion data with contract validation and event emission.

        Args:
            dataset_id: Dataset identifier
            records: Data records to write
            source: Data source (live, historical, backfill)
            run_id: Processing run identifier
            instrument_id: Instrument identifier (extracted if not provided)

        Returns:
            DataEvent tracking the write operation

        """
        ...

    def write_features(
        self,
        instrument_id: str,
        features: list[Any],  # list[FeatureData]
        source: str = "computed",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Write features with validation and event emission.

        Args:
            instrument_id: Instrument identifier
            features: Feature data to store
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        ...

    def write_predictions(
        self,
        predictions: list[Any],  # list[ModelPrediction]
        source: str = "inference",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Write model predictions with validation and event emission.

        Args:
            predictions: Model predictions to store
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        ...

    def write_signals(
        self,
        signals: list[Any],  # list[StrategySignal]
        source: str = "strategy",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Write strategy signals with validation and event emission.

        Args:
            signals: Strategy signals to store
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        ...

    def write_earnings_actual(
        self,
        *,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        eps_basic: float | None = None,
        net_income: float | None = None,
        operating_income: float | None = None,
        shares_outstanding: int | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        fiscal_quarter: int | None = None,
        source: str = "historical",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Persist an earnings actual record with contract validation.

        Args:
            ticker: Stock ticker symbol
            period_end: Period end date
            filing_date: Filing date
            eps_diluted: Diluted earnings per share
            revenue: Revenue amount
            ts_event: Event timestamp (nanoseconds)
            ts_init: Initialization timestamp (nanoseconds)
            eps_basic: Basic earnings per share
            net_income: Net income amount
            operating_income: Operating income amount
            shares_outstanding: Shares outstanding
            filing_type: Filing type (10-Q, 10-K, etc.)
            fiscal_year: Fiscal year
            fiscal_quarter: Fiscal quarter
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        ...

    def write_earnings_estimate(
        self,
        *,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        revenue_consensus: float | None = None,
        num_analysts: int | None = None,
        source: str = "historical",
        run_id: str | None = None,
    ) -> Any:  # DataEvent
        """
        Persist an earnings estimate record with contract validation.

        Args:
            ticker: Stock ticker symbol
            estimate_date: Estimate date
            period_end: Period end date
            eps_consensus: Consensus EPS estimate
            ts_event: Event timestamp (nanoseconds)
            ts_init: Initialization timestamp (nanoseconds)
            revenue_consensus: Consensus revenue estimate
            num_analysts: Number of analysts in consensus
            source: Data source
            run_id: Processing run identifier

        Returns:
            DataEvent tracking the write operation

        """
        ...


class DataReaderProtocol(Protocol):
    """
    Protocol for data reader component.

    Provides read operations for ingestion data, features, predictions,
    signals, and earnings data with time-travel queries.

    HOT PATH: get_features_at_or_before() must achieve P99 < 5ms
    COLD PATH: All other methods (async acceptable)

    """

    def get_features_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
    ) -> dict[str, float] | None:
        """
        Return latest feature values at or before the given timestamp.

        HOT PATH: P99 < 5ms requirement

        Args:
            instrument_id: Instrument identifier
            ts_event: Timestamp in nanoseconds (point-in-time query)

        Returns:
            Dictionary mapping feature names to values (all features),
            or None if no features exist before timestamp.

        """
        ...

    def read_ingestion_data(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        dataset_type: Any = None,  # DatasetType | None
    ) -> Any:  # pl.DataFrame
        """
        Read ingestion data for a specific time range.

        COLD PATH: Bulk read operation

        Args:
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)
            dataset_type: Optional dataset type filter

        Returns:
            DataFrame with ingestion data

        """
        ...

    def read_features(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        feature_names: list[str] | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read feature data for a specific time range.

        COLD PATH: Bulk read operation

        Args:
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)
            feature_names: Optional list of feature names to retrieve

        Returns:
            DataFrame with feature data

        """
        ...

    def read_predictions(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        model_id: str | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read model predictions for a specific time range.

        COLD PATH: Bulk read operation

        Args:
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)
            model_id: Optional model identifier filter

        Returns:
            DataFrame with prediction data

        """
        ...

    def read_signals(
        self,
        *,
        instrument_id: str,
        start_ts: int,
        end_ts: int,
        strategy_id: str | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read strategy signals for a specific time range.

        COLD PATH: Bulk read operation

        Args:
            instrument_id: Instrument identifier
            start_ts: Start timestamp in nanoseconds (inclusive)
            end_ts: End timestamp in nanoseconds (exclusive)
            strategy_id: Optional strategy identifier filter

        Returns:
            DataFrame with signal data

        """
        ...

    def get_latest_prediction_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        model_id: str | None = None,
    ) -> Any | None:  # PredictionRecord | None
        """
        Return latest prediction at or before ts_event.

        Args:
            instrument_id: Instrument identifier
            ts_event: Timestamp in nanoseconds (point-in-time query)
            model_id: Optional model identifier filter

        Returns:
            Prediction record or None when not found

        """
        ...

    def get_latest_signal_at_or_before(
        self,
        *,
        instrument_id: str,
        ts_event: int,
        strategy_id: str | None = None,
    ) -> Any | None:  # SignalRecord | None
        """
        Return latest strategy signal at or before ts_event.

        Args:
            instrument_id: Instrument identifier
            ts_event: Timestamp in nanoseconds (point-in-time query)
            strategy_id: Optional strategy identifier filter

        Returns:
            Signal record or None when not found

        """
        ...

    def read_earnings_actual(
        self,
        *,
        symbol: str,
        start_date: str | None,
        end_date: str | None,
        as_of_ts: int | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read earnings actuals for a ticker within date range.

        COLD PATH: Bulk read operation

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None
            as_of_ts: Optional point-in-time timestamp (nanoseconds)

        Returns:
            DataFrame with earnings actuals data

        """
        ...

    def read_earnings_estimate(
        self,
        *,
        symbol: str,
        start_date: str | None,
        end_date: str | None,
        as_of_ts: int | None = None,
    ) -> Any:  # pl.DataFrame
        """
        Read earnings estimates for a ticker within date range.

        COLD PATH: Bulk read operation

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None
            as_of_ts: Optional point-in-time timestamp (nanoseconds)

        Returns:
            DataFrame with earnings estimates data

        """
        ...


class EventEmitterProtocol(Protocol):
    """
    Protocol for event emitter component.

    Provides event emission and message bus integration for dataset operations.
    All methods are COLD path (async operations acceptable, non-blocking).

    CRITICAL: All event emission must be NON-BLOCKING to ensure data operations
    are not disrupted by event tracking failures.

    """

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Any,  # Stage | str
        source: Any,  # Source | str
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: str = "success",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit a dataset processing event via centralized registry.

        COLD PATH: Event emission is async, non-blocking
        CRITICAL: Non-blocking - failures are logged but don't raise

        Args:
            dataset_id: Dataset identifier
            instrument_id: Instrument identifier
            stage: Processing stage (DATA_INGESTED, FEATURES_COMPUTED, etc.)
            source: Event source (LIVE, HISTORICAL, BACKFILL)
            run_id: Unique identifier for this processing run
            ts_min: Minimum timestamp (ns) for covered data
            ts_max: Maximum timestamp (ns) for covered data
            count: Number of records processed
            status: Status string (EventStatus.value)
            error: Error message if status is failed
            metadata: Additional metadata to attach to the event

        """
        ...

    def emit_dataset_event(
        self,
        *,
        dataset_id: str,
        status: Any,  # EventStatus | str
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit a dataset-specific event with simplified parameter set.

        COLD PATH: Event emission is async, non-blocking
        CRITICAL: Non-blocking - failures are logged but don't raise

        Args:
            dataset_id: Dataset identifier
            status: Event status (SUCCESS, FAILED, PARTIAL)
            metadata: Additional event metadata

        """
        ...

    def _emit_partial_event(
        self,
        *,
        operation: str,
        details: dict[str, Any],
    ) -> None:
        """
        Emit a partial success event for incomplete operations.

        COLD PATH: Event emission is async, non-blocking
        CRITICAL: Non-blocking - failures are logged but don't raise

        Args:
            operation: Operation name (write_ingestion, write_features, etc.)
            details: Details about the partial success

        """
        ...

    def _emit_failed_event(
        self,
        *,
        operation: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """
        Emit a failure event for operations that completely failed.

        COLD PATH: Event emission is async, non-blocking
        CRITICAL: Non-blocking - failures are logged but don't raise

        Args:
            operation: Operation name
            error: Exception that caused the failure
            context: Context about the failed operation

        """
        ...


class ContractEnforcerProtocol(Protocol):
    """
    Protocol for contract enforcer component.

    Provides contract retrieval, schema migration management, and quality enforcement
    for dataset operations. All methods are COLD path (contract management is async acceptable).

    """

    def get_manifest(self, dataset_id: str) -> Any:  # DatasetManifest
        """
        Get dataset manifest with caching and version check.

        COLD PATH: Manifest retrieval is async acceptable

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dataset manifest with schema, constraints, and metadata

        """
        ...

    def get_contract(self, dataset_id: str) -> Any:  # DataContract
        """
        Get data contract with caching and version check.

        COLD PATH: Contract retrieval is async acceptable

        Args:
            dataset_id: Dataset identifier

        Returns:
            Data contract with validation rules and enforcement mode

        """
        ...

    def compute_schema_hash(self, data_frame: Any, manifest: Any) -> str:  # DataFrameLike, DatasetManifest
        """
        Compute schema hash for the actual data.

        COLD PATH: Schema hash computation is one-time overhead

        Args:
            data_frame: Data to compute schema hash from
            manifest: Dataset manifest with expected schema

        Returns:
            Hex-encoded schema hash (deterministic)

        """
        ...

    def is_in_migration_window(self, dataset_id: str) -> bool:
        """
        Check if dataset is in schema migration window.

        COLD PATH: Migration window check is rare

        Args:
            dataset_id: Dataset identifier

        Returns:
            True if dataset is in active migration window

        """
        ...

    def start_migration_window(self, dataset_id: str, manifest: Any) -> None:  # DatasetManifest
        """
        Start a schema migration window for dual-write.

        COLD PATH: Migration window start is rare

        Args:
            dataset_id: Dataset identifier
            manifest: Updated manifest with new schema version

        """
        ...

    def enforce_quality(
        self,
        quality_report: Any,  # QualityReport
        contract: Any,  # DataContract
        dataset_id: str,
    ) -> None:
        """
        Apply contract enforcement logic for quality reports.

        COLD PATH: Quality enforcement happens once per write batch

        Args:
            quality_report: Quality report from batch validation
            contract: Data contract with enforcement mode and thresholds
            dataset_id: Dataset identifier for error messages

        Raises:
            ValueError: When quality score < 1.0 and enforcement mode requires failure

        """
        ...

    def format_violations(self, violations: list[Any]) -> str:  # list[ValidationViolation]
        """
        Format violations for logging.

        COLD PATH: Violation formatting is rare

        Args:
            violations: List of validation violations from quality report

        Returns:
            Formatted violation summary

        """
        ...

    def apply_validation_rule(
        self,
        rule: Any,  # ValidationRule
        data_frame: object,
        manifest: Any | None = None,  # DatasetManifest | None
    ) -> Any | None:  # ValidationViolation | None
        """
        Apply a single validation rule to data.

        COLD PATH: Rule application is part of batch validation

        Args:
            rule: Validation rule to apply
            data_frame: Data to validate
            manifest: Optional manifest for schema context

        Returns:
            Violation if rule fails, None if validation passes

        """
        ...

    def migrate_schema(
        self,
        from_version: str,
        to_version: str,
        data: Any,  # pl.DataFrame
    ) -> Any:  # pl.DataFrame
        """
        Migrate data schema from one version to another.

        COLD PATH: Schema migration is rare

        Args:
            from_version: Source schema version
            to_version: Target schema version
            data: Data to migrate

        Returns:
            Migrated data with target schema

        """
        ...

    def register_contract(self, contract_id: str, schema: dict[str, Any], version: str) -> None:
        """
        Register a new data contract with the registry.

        COLD PATH: Contract registration is infrequent

        Args:
            contract_id: Unique contract identifier
            schema: Contract schema with validation rules
            version: Contract version

        """
        ...

    def update_contract(self, contract_id: str, schema: dict[str, Any], version: str) -> None:
        """
        Update an existing data contract.

        COLD PATH: Contract updates are infrequent

        Args:
            contract_id: Contract identifier
            schema: Updated contract schema
            version: New contract version

        """
        ...

    def register_manifest(self, manifest_id: str, metadata: dict[str, Any]) -> None:
        """
        Register a new dataset manifest with the registry.

        COLD PATH: Manifest registration is infrequent

        Args:
            manifest_id: Unique manifest identifier (dataset_id)
            metadata: Manifest metadata (schema, constraints, etc.)

        """
        ...

    def validate_contract(self, contract_id: str, data: Any) -> bool:  # pl.DataFrame
        """
        Validate data against a contract.

        COLD PATH: Contract validation is part of write path

        Args:
            contract_id: Contract identifier
            data: Data to validate

        Returns:
            True if validation passes, False otherwise

        """
        ...


class StoreOperationsProtocol(Protocol):
    """
    Protocol for store operations component.

    Provides store lifecycle management, health monitoring, metrics collection,
    progressive fallback chains, and circuit breaker logic for unstable dependencies.

    All methods are COLD path (infrastructure operations, no hot path constraints).

    """

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check across all 4 stores + 4 registries.

        COLD PATH: Health monitoring is infrastructure operation

        Returns:
            Health status with component-level details

        """
        ...

    def get_metrics(self) -> dict[str, float]:
        """
        Aggregate performance metrics from all components.

        COLD PATH: Metrics collection is infrastructure operation

        Returns:
            Aggregated metrics for observability

        """
        ...

    def close(self) -> None:
        """
        Gracefully shutdown all stores and clean up resources.

        COLD PATH: Shutdown is infrastructure operation

        """
        ...

    def _initialize_stores(self) -> None:
        """
        Initialize all stores with progressive fallback chains.

        COLD PATH: Store initialization happens at startup

        """
        ...

    def _initialize_fallback_chain(self) -> None:
        """
        Initialize progressive fallback chain: PRIMARY → CACHED → FILE → DUMMY.

        COLD PATH: Fallback chain initialization happens at startup

        """
        ...

    def _activate_fallback(self, reason: str) -> None:
        """
        Activate fallback chain and emit metrics.

        COLD PATH: Fallback activation is rare (only on failures)

        Args:
            reason: Reason for fallback activation

        """
        ...

    def _restore_primary(self) -> bool:
        """
        Attempt to restore primary store connection.

        COLD PATH: Primary restoration is rare (only after failures)

        Returns:
            True if primary restored successfully, False otherwise

        """
        ...

    def _emit_health_metric(self, status: str, component: str) -> None:
        """
        Emit health check metric for component.

        COLD PATH: Metric recording is infrastructure operation

        Args:
            status: Health status (healthy, degraded, unhealthy)
            component: Component name

        """
        ...

    def _record_operation_latency(self, operation: str, duration_ms: float) -> None:
        """
        Record operation latency for performance tracking.

        COLD PATH: Metric recording is infrastructure operation

        Args:
            operation: Operation name
            duration_ms: Operation duration in milliseconds

        """
        ...
