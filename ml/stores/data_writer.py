#!/usr/bin/env python3

"""
Write operations for ML data stores.

This module provides focused write operations with validation, event emission,
and watermark updates for all ML data types. Extracted from the monolithic
DataStore class for better maintainability and testability.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, cast

from ml._imports import HAS_PROMETHEUS
from ml._imports import pd
from ml.common.message_bus import MessagePublisherProtocol
from ml.common.metrics_bootstrap import get_counter
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetManifest
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.utils import compute_dataset_schema_hash
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.contract_enforcer import ContractEnforcer
from ml.stores.feature_dataset_store import FeatureDatasetStore
from ml.stores.protocols import EarningsStoreProtocol
from ml.stores.schema_validator import SchemaValidator
from ml.stores.validation_types import DataEvent


logger = logging.getLogger(__name__)

# Constants for earnings datasets
EARNINGS_ACTUALS_DATASET_ID = "earnings_actuals"
EARNINGS_ESTIMATES_DATASET_ID = "earnings_estimates"


# Get metrics via bootstrap (returns dummy metrics if Prometheus unavailable)
write_rejection_counter = get_counter(
    "ml_write_rejections_total",
    "Total number of write rejections",
    labelnames=["dataset_id", "reason"],
)


# ========================================================================
# Protocol Definition
# ========================================================================


class DataWriterProtocol(Protocol):
    """Protocol for data write operations."""

    def write_ingestion(
        self,
        dataset_id: str,
        records: list[dict[str, Any]] | DataFrameLike,
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> DataEvent:
        """
        Write ingestion data with validation and event emission.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        records : list[dict] | DataFrame
            Data records to write
        source : str
            Data source (live, historical, backfill)
        run_id : str
            Processing run identifier
        instrument_id : str | None
            Instrument identifier (extracted if not provided)

        Returns
        -------
        DataEvent
            Event tracking the write operation
        """
        ...

    def write_features(
        self,
        instrument_id: str,
        features: list[FeatureData],
        source: str = "computed",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write features with validation and event emission.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier
        features : list[FeatureData]
            Feature data to store
        source : str
            Data source (default: "computed")
        run_id : str | None
            Processing run identifier

        Returns
        -------
        DataEvent
            Event tracking the write operation
        """
        ...

    def write_predictions(
        self,
        predictions: list[ModelPrediction],
        source: str = "inference",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write model predictions with validation and event emission.

        Parameters
        ----------
        predictions : list[ModelPrediction]
            Model predictions to store
        source : str
            Data source (default: "inference")
        run_id : str | None
            Processing run identifier

        Returns
        -------
        DataEvent
            Event tracking the write operation
        """
        ...

    def write_signals(
        self,
        signals: list[StrategySignal],
        source: str = "strategy",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write strategy signals with validation and event emission.

        Parameters
        ----------
        signals : list[StrategySignal]
            Strategy signals to store
        source : str
            Data source (default: "strategy")
        run_id : str | None
            Processing run identifier

        Returns
        -------
        DataEvent
            Event tracking the write operation
        """
        ...


# ========================================================================
# DataWriter Implementation
# ========================================================================


class DataWriter:
    """
    Performs write operations with validation and event emission.

    Wraps underlying stores with contract validation, quality enforcement,
    event emission, and watermark tracking. This component is extracted from
    the DataStore god class to provide focused, testable write functionality
    following the Strangler Fig pattern.

    Parameters
    ----------
    feature_store : Any
        Feature store instance
    model_store : Any
        Model store instance
    strategy_store : Any
        Strategy store instance
    earnings_store : EarningsStoreProtocol
        Earnings store instance
    contract_enforcer : ContractEnforcer
        Contract enforcement component
    schema_validator : SchemaValidator
        Schema validation component
    registry : Any
        Data registry for manifest/contract retrieval
    publisher : MessagePublisherProtocol | None
        Message bus publisher (optional)
    enable_publishing : bool
        Enable event publishing
    fail_on_validation_error : bool
        If True, fail writes on validation errors
    batch_size : int
        Batch size for write operations
    raw_writer : Any | None
        Optional raw data writer
    topic_scheme : str
        Topic naming scheme for message bus
    topic_prefix : str
        Topic prefix for message bus
    """

    def __init__(
        self,
        *,
        feature_store: Any,
        model_store: Any,
        strategy_store: Any,
        earnings_store: EarningsStoreProtocol,
        feature_dataset_store: FeatureDatasetStore | None = None,
        contract_enforcer: ContractEnforcer,
        schema_validator: SchemaValidator,
        registry: Any,
        publisher: MessagePublisherProtocol | None = None,
        enable_publishing: bool = False,
        fail_on_validation_error: bool = True,
        batch_size: int = 10000,
        raw_writer: Any | None = None,
        topic_scheme: str = "hierarchical",
        topic_prefix: str = "nautilus",
    ) -> None:
        """
        Initialize data writer with dependencies.

        Parameters
        ----------
        feature_store : Any
            Feature store instance
        model_store : Any
            Model store instance
        strategy_store : Any
            Strategy store instance
        earnings_store : EarningsStoreProtocol
            Earnings store instance
        feature_dataset_store : FeatureDatasetStore | None
            Store used for macro/events/micro/L2 datasets
        contract_enforcer : ContractEnforcer
            Contract enforcement component
        schema_validator : SchemaValidator
            Schema validation component
        registry : Any
            Data registry for manifest/contract retrieval
        publisher : MessagePublisherProtocol | None
            Message bus publisher (optional)
        enable_publishing : bool
            Enable event publishing
        fail_on_validation_error : bool
            If True, fail writes on validation errors
        batch_size : int
            Batch size for write operations
        raw_writer : Any | None
            Optional raw data writer
        topic_scheme : str
            Topic naming scheme for message bus
        topic_prefix : str
            Topic prefix for message bus
        """
        self.feature_store = feature_store
        self.model_store = model_store
        self.strategy_store = strategy_store
        self.earnings_store = earnings_store
        self.contract_enforcer = contract_enforcer
        self.schema_validator = schema_validator
        self.registry = registry
        self.feature_dataset_store = feature_dataset_store
        self.publisher: MessagePublisherProtocol | None = publisher
        self.enable_publishing = enable_publishing
        self.fail_on_validation_error = fail_on_validation_error
        self.batch_size = batch_size
        self.raw_writer = raw_writer
        self.topic_scheme = topic_scheme
        self.topic_prefix = topic_prefix

        logger.debug(
            "Initialized DataWriter (publishing=%s, fail_on_error=%s)",
            enable_publishing,
            fail_on_validation_error,
        )

    def write_ingestion(
        self,
        dataset_id: str,
        records: list[dict[str, Any]] | DataFrameLike,
        source: str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> DataEvent:
        """
        Write ingestion data with contract validation and event emission.

        This is the main entry point for data ingestion with full validation,
        quality enforcement, and event emission.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        records : list[dict] | DataFrame
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
            If dataset not found or validation fails
        RuntimeError
            If write operation fails
        """
        start_time = time.perf_counter()

        # Get manifest and contract
        manifest = self.contract_enforcer.get_manifest(dataset_id)
        contract = self.contract_enforcer.get_contract(dataset_id)

        # Perform preflight schema check
        preflight_passed, preflight_error, preflight_details = self.contract_enforcer.preflight_check(
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
        data_frame: DataFrameLike = self._to_dataframe(records)

        # Extract metadata from DataFrame
        extra_metadata: dict[str, object] = {}
        if pd is not None:
            data_frame_for_meta: Any = None
            if isinstance(data_frame, pd.DataFrame):
                data_frame_for_meta = data_frame
            else:
                to_pandas = getattr(data_frame, "to_pandas", None)
                if callable(to_pandas):
                    try:
                        data_frame_for_meta = to_pandas()
                    except Exception:
                        data_frame_for_meta = None

            if data_frame_for_meta is not None:
                extra_metadata = self._extract_ingestion_metadata_from_dataframe(data_frame_for_meta)

        # Extract instrument_id if not provided
        if instrument_id is None:
            if hasattr(data_frame, "columns") and "instrument_id" in cast(Any, data_frame).columns:
                # Handle both Polars and pandas
                col = cast(Any, data_frame)["instrument_id"]
                if hasattr(col, "iloc"):
                    # pandas Series
                    instrument_id = str(col.iloc[0])
                else:
                    # Polars Series or other
                    instrument_id = str(col[0])
            else:
                instrument_id = "UNKNOWN"

        # Validate data against contract
        use_strict = contract.enforcement_mode == "strict"
        quality_report = self.contract_enforcer.validate_batch(dataset_id, data_frame, strict_mode=use_strict)

        monitor_message: str | None = None
        if (
            quality_report.quality_score < 1.0
            and contract.enforcement_mode == "monitor_only"
        ):
            monitor_message = self.schema_validator.format_violations(quality_report.violations)

        # Enforce quality report
        if quality_report.quality_score < 1.0:
            self.schema_validator.enforce_quality_report(
                dataset_id=dataset_id,
                contract=contract,
                quality_report=quality_report,
                fail_on_validation_error=self.fail_on_validation_error,
            )

        # Extract timestamp range
        ts_field = manifest.ts_field
        ts_min = int(cast(Any, data_frame)[ts_field].min())
        ts_max = int(cast(Any, data_frame)[ts_field].max())

        # Determine appropriate stage based on dataset type
        stage = self._get_stage_for_dataset_type(manifest.dataset_type)

        try:
            # Route to appropriate store based on dataset type
            if manifest.dataset_type == DatasetType.FEATURES:
                # Convert to FeatureData format and write
                feature_data = self._data_frame_to_feature_data(data_frame, instrument_id)
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
                predictions = self._data_frame_to_predictions(data_frame)
                try:
                    self.model_store.write_batch(predictions, emit_events=False, publish_bus=False)
                except TypeError:
                    self.model_store.write_batch(predictions)

            elif manifest.dataset_type == DatasetType.SIGNALS:
                # Convert to StrategySignal format and write
                signals = self._data_frame_to_signals(data_frame)
                self.strategy_store.write_batch(signals, emit_events=False, publish_bus=False)

            elif manifest.dataset_type in {
                DatasetType.MACRO_RELEASES,
                DatasetType.MACRO_OBSERVATIONS,
                DatasetType.EVENTS_CALENDAR,
                DatasetType.MICRO_MINUTE_FEATURES,
                DatasetType.L2_MINUTE_FEATURES,
            }:
                self._write_feature_dataset(manifest.dataset_type, data_frame)

            else:
                # Raw dataset types: delegate to optional writer if configured
                if self.raw_writer is not None:
                    try:
                        written = self.raw_writer.write(
                            dataset_type=manifest.dataset_type,
                            data=data_frame,
                        )
                        if written <= 0:
                            logger.warning("Raw writer reported 0 records written for %s", dataset_id)
                            return self._create_partial_event(
                                dataset_id=dataset_id,
                                instrument_id=instrument_id,
                                source=source,
                                run_id=run_id,
                                ts_min=ts_min,
                                ts_max=ts_max,
                                record_count=len(cast(Any, data_frame)),
                                reason="no_records_written",
                                metadata=extra_metadata,
                            )
                    except Exception as exc:
                        logger.error("Raw writer failed for %s", dataset_id, exc_info=True)
                        return self._create_failed_event(
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            record_count=0,
                            error=str(exc),
                        )
                else:
                    # No raw writer configured
                    if self.fail_on_validation_error:
                        logger.info("Raw writer not configured; proceeding with success for %s", dataset_id)
                    else:
                        logger.warning("Raw writer not configured; skipping persistence for %s", dataset_id)
                        return self._create_partial_event(
                            dataset_id=dataset_id,
                            instrument_id=instrument_id,
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min,
                            ts_max=ts_max,
                            record_count=len(cast(Any, data_frame)),
                            reason="raw_writer_missing",
                            metadata=extra_metadata,
                        )

            # Create success event
            from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

            ts_min_s = _sanitize(int(ts_min), context="data_writer.write_ingestion:ts_min")
            ts_max_s = _sanitize(int(ts_max), context="data_writer.write_ingestion:ts_max")

            metadata: dict[str, Any] = {
                "quality_score": quality_report.quality_score,
                "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                **extra_metadata,
            }
            if monitor_message is not None:
                metadata["monitor_only_details"] = monitor_message

            event = DataEvent(
                event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                operation="write_ingestion",
                source=source,
                run_id=run_id,
                ts_min=ts_min_s,
                ts_max=ts_max_s,
                record_count=len(cast(Any, data_frame)),
                status=EventStatus.SUCCESS.value,
                metadata=metadata,
            )

            # Emit event and update watermark
            self._emit_success_event_and_update(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source,
                run_id=run_id,
                ts_min=ts_min_s,
                ts_max=ts_max_s,
                count=len(cast(Any, data_frame)),
                dataset_type=manifest.dataset_type,
            )

            logger.info(
                "Successfully wrote %d records to %s (quality=%.2f)",
                len(cast(Any, data_frame)),
                dataset_id,
                quality_report.quality_score,
            )

            return event

        except Exception as exc:
            logger.error("Failed to write data to %s", dataset_id, exc_info=True)
            raise RuntimeError(f"Write operation failed: {exc}") from exc

    def write_features(
        self,
        instrument_id: str,
        features: list[FeatureData],
        source: str = "computed",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write features with validation and event emission.

        Wraps FeatureStore.write_features with event tracking.

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

        stage = Stage.FEATURE_COMPUTED

        try:
            for feature in features:
                self.feature_store.write_features(
                    feature_set_id=feature.feature_set_id,
                    instrument_id=feature.instrument_id,
                    features=feature.values,
                    ts_event=feature.ts_event,
                    ts_init=feature.ts_init,
                    publish_bus=False,
                )
        except Exception as exc:
            logger.exception("Feature store write failed", exc_info=True)
            raise RuntimeError(f"Feature write failed: {exc}") from exc

        # Calculate timestamp range
        ts_min = min(f.ts_event for f in features)
        ts_max = max(f.ts_event for f in features)

        # Create event
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_writer.write_features:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_writer.write_features:ts_max")

        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_features",
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            record_count=len(features),
            status=EventStatus.SUCCESS.value,
        )

        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(features),
            dataset_type=DatasetType.FEATURES,
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

        Wraps ModelStore.write_batch with event tracking.

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
        try:
            self.model_store.write_batch(predictions, emit_events=False, publish_bus=False)
        except TypeError:
            self.model_store.write_batch(predictions)

        # Calculate timestamp range
        ts_min = min(p.ts_event for p in predictions)
        ts_max = max(p.ts_event for p in predictions)

        # Create event
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_writer.write_predictions:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_writer.write_predictions:ts_max")

        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_predictions",
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            record_count=len(predictions),
            status=EventStatus.SUCCESS.value,
            metadata={"model_id": model_id},
        )

        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.PREDICTION_EMITTED.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(predictions),
            dataset_type=DatasetType.PREDICTIONS,
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

        Wraps StrategyStore.write_batch with event tracking.

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
        self.strategy_store.write_batch(signals, emit_events=False, publish_bus=False)

        # Calculate timestamp range
        ts_min = min(s.ts_event for s in signals)
        ts_max = max(s.ts_event for s in signals)

        # Create event
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_writer.write_signals:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_writer.write_signals:ts_max")

        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_signals",
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            record_count=len(signals),
            status=EventStatus.SUCCESS.value,
            metadata={"strategy_id": strategy_id},
        )

        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=Stage.SIGNAL_EMITTED.value,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(signals),
            dataset_type=DatasetType.SIGNALS,
        )

        logger.debug(
            "Wrote %d signals for %s from strategy %s",
            len(signals),
            instrument_id,
            strategy_id,
        )
        return event

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
        source: str = Source.HISTORICAL.value,
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Persist an earnings actual record with contract validation.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period_end : str
            Period end date (ISO format)
        filing_date : str
            Filing date (ISO format)
        eps_diluted : float | None
            Diluted EPS
        revenue : float | None
            Revenue
        ts_event : int
            Event timestamp in nanoseconds
        ts_init : int
            Init timestamp in nanoseconds
        eps_basic : float | None
            Basic EPS
        net_income : float | None
            Net income
        operating_income : float | None
            Operating income
        shares_outstanding : int | None
            Shares outstanding
        filing_type : str | None
            Filing type (10-K, 10-Q, etc.)
        fiscal_year : int | None
            Fiscal year
        fiscal_quarter : int | None
            Fiscal quarter
        source : str
            Data source
        run_id : str | None
            Processing run identifier

        Returns
        -------
        DataEvent
            Event tracking the write operation
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        dataset_id = EARNINGS_ACTUALS_DATASET_ID
        run_id_local = run_id or f"earnings_actual_{time.time_ns()}"
        ts_event_s = _sanitize_ts(int(ts_event), context="data_writer.write_earnings_actual:ts_event")
        ts_init_s = _sanitize_ts(int(ts_init), context="data_writer.write_earnings_actual:ts_init")

        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.EARNINGS_ACTUALS,
            instrument_id=ticker,
        )

        record: dict[str, Any] = {
            "ticker": ticker,
            "period_end": period_end,
            "filing_date": filing_date,
            "ts_event": ts_event_s,
            "ts_init": ts_init_s,
            "eps_basic": eps_basic,
            "eps_diluted": eps_diluted,
            "revenue": revenue,
            "net_income": net_income,
            "operating_income": operating_income,
            "shares_outstanding": shares_outstanding,
            "filing_type": filing_type,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "data_source": "EDGAR",
        }

        contract = self.contract_enforcer.get_contract(dataset_id)
        use_strict = contract.enforcement_mode == "strict"
        record_frame: DataFrameLike = self._to_dataframe([record])
        quality_report = self.contract_enforcer.validate_batch(
            dataset_id,
            record_frame,
            strict_mode=use_strict,
        )
        self.schema_validator.enforce_quality_report(
            dataset_id=dataset_id,
            contract=contract,
            quality_report=quality_report,
            fail_on_validation_error=self.fail_on_validation_error,
        )

        try:
            self.earnings_store.write_actuals(
                ticker=ticker,
                period_end=period_end,
                filing_date=filing_date,
                eps_diluted=eps_diluted,
                revenue=revenue,
                ts_event=ts_event_s,
                ts_init=ts_init_s,
                eps_basic=eps_basic,
                net_income=net_income,
                operating_income=operating_income,
                shares_outstanding=shares_outstanding,
                filing_type=filing_type,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
            )
        except Exception as exc:
            logger.exception("Earnings actual write failed for %s", ticker)
            raise RuntimeError(f"Earnings actual write failed: {exc}") from exc

        raw_writer_status = "not_configured"
        if self.raw_writer is not None:
            raw_writer_status = "ok"
            try:
                self.raw_writer.write(
                    dataset_type=DatasetType.EARNINGS_ACTUALS,
                    data=[record],
                )
            except Exception as exc:  # pragma: no cover - mirror failures are non-critical
                raw_writer_status = "failed"
                logger.warning(
                    "Raw earnings mirror write failed for %s: %s",
                    ticker,
                    exc,
                    exc_info=True,
                )

        event = DataEvent(
            event_id=f"{run_id_local}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=ticker,
            operation="write_earnings_actual",
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            record_count=1,
            status=EventStatus.SUCCESS.value,
            metadata=self._build_earnings_metadata(
                quality_score=quality_report.quality_score,
                raw_writer_status=raw_writer_status,
            ),
        )

        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=ticker,
            stage=Stage.DATA_INGESTED.value,
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            count=1,
            dataset_type=DatasetType.EARNINGS_ACTUALS,
        )

        return event

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
        source: str = Source.HISTORICAL.value,
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Persist an earnings estimate record with contract validation.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        estimate_date : str
            Estimate date (ISO format)
        period_end : str
            Period end date (ISO format)
        eps_consensus : float | None
            EPS consensus estimate
        ts_event : int
            Event timestamp in nanoseconds
        ts_init : int
            Init timestamp in nanoseconds
        revenue_consensus : float | None
            Revenue consensus estimate
        num_analysts : int | None
            Number of analysts
        source : str
            Data source
        run_id : str | None
            Processing run identifier

        Returns
        -------
        DataEvent
            Event tracking the write operation
        """
        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize_ts

        dataset_id = EARNINGS_ESTIMATES_DATASET_ID
        run_id_local = run_id or f"earnings_estimate_{time.time_ns()}"
        ts_event_s = _sanitize_ts(int(ts_event), context="data_writer.write_earnings_estimate:ts_event")
        ts_init_s = _sanitize_ts(int(ts_init), context="data_writer.write_earnings_estimate:ts_init")

        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.EARNINGS_ESTIMATES,
            instrument_id=ticker,
        )

        record: dict[str, Any] = {
            "ticker": ticker,
            "estimate_date": estimate_date,
            "period_end": period_end,
            "ts_event": ts_event_s,
            "ts_init": ts_init_s,
            "eps_consensus": eps_consensus,
            "revenue_consensus": revenue_consensus,
            "num_analysts": num_analysts,
            "data_source": "YAHOO",
        }

        contract = self.contract_enforcer.get_contract(dataset_id)
        use_strict = contract.enforcement_mode == "strict"
        record_frame: DataFrameLike = self._to_dataframe([record])
        quality_report = self.contract_enforcer.validate_batch(
            dataset_id,
            record_frame,
            strict_mode=use_strict,
        )
        self.schema_validator.enforce_quality_report(
            dataset_id=dataset_id,
            contract=contract,
            quality_report=quality_report,
            fail_on_validation_error=self.fail_on_validation_error,
        )

        try:
            self.earnings_store.write_estimates(
                ticker=ticker,
                estimate_date=estimate_date,
                period_end=period_end,
                eps_consensus=eps_consensus,
                ts_event=ts_event_s,
                ts_init=ts_init_s,
                revenue_consensus=revenue_consensus,
                num_analysts=num_analysts,
            )
        except Exception as exc:
            logger.exception("Earnings estimate write failed for %s", ticker)
            raise RuntimeError(f"Earnings estimate write failed: {exc}") from exc

        raw_writer_status = "not_configured"
        if self.raw_writer is not None:
            raw_writer_status = "ok"
            try:
                self.raw_writer.write(
                    dataset_type=DatasetType.EARNINGS_ESTIMATES,
                    data=[record],
                )
            except Exception as exc:  # pragma: no cover - mirror failures are non-critical
                raw_writer_status = "failed"
                logger.warning(
                    "Raw earnings estimate mirror failed for %s: %s",
                    ticker,
                    exc,
                    exc_info=True,
                )

        event = DataEvent(
            event_id=f"{run_id_local}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=ticker,
            operation="write_earnings_estimate",
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            record_count=1,
            status=EventStatus.SUCCESS.value,
            metadata=self._build_earnings_metadata(
                quality_score=quality_report.quality_score,
                raw_writer_status=raw_writer_status,
            ),
        )

        self._emit_success_event_and_update(
            dataset_id=dataset_id,
            instrument_id=ticker,
            stage=Stage.DATA_INGESTED.value,
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            count=1,
            dataset_type=DatasetType.EARNINGS_ESTIMATES,
        )

        return event

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    @staticmethod
    def _build_earnings_metadata(
        *,
        quality_score: float,
        raw_writer_status: str,
    ) -> dict[str, object]:
        """Build metadata payload for earnings write events."""
        metadata: dict[str, object] = {"quality_score": quality_score}
        if raw_writer_status != "not_configured":
            metadata["raw_mirror_status"] = raw_writer_status
        return metadata

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
        dataset_type: DatasetType,
        completeness_pct: float = 100.0,
    ) -> None:
        """
        Emit a success event and update registry watermark.

        This is a best-effort operation - failures are logged but not raised.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        stage : str
            Processing stage
        source : str
            Data source
        run_id : str
            Processing run identifier
        ts_min : int
            Minimum timestamp
        ts_max : int
            Maximum timestamp
        count : int
            Record count
        dataset_type : DatasetType
            Dataset type
        completeness_pct : float
            Completeness percentage (default: 100.0)
        """
        try:
            from ml.common.correlation import make_correlation_id
            from ml.common.event_emitter import emit_dataset_event_and_watermark
            from ml.common.events_util import build_bus_payload
            from ml.common.events_util import to_source_enum
            from ml.common.events_util import to_stage_enum
            from ml.common.message_topics import build_topic_for_stage

            # Build correlation id for observability
            correlation_id = make_correlation_id(
                run_id=run_id,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
            )

            # Normalize source to the canonical enum
            src_enum = to_source_enum(source)
            stage_enum = to_stage_enum(stage)

            # Emit event and watermark
            emit_dataset_event_and_watermark(
                self.registry,
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage_enum,
                source=src_enum,
                run_id=run_id,
                ts_min=ts_min,
                ts_max=ts_max,
                count=count,
                status=EventStatus.SUCCESS,
                dataset_type=str(dataset_type.value if hasattr(dataset_type, "value") else dataset_type),
                component=self.__class__.__name__,
            )

            # Optionally publish to message bus
            if self.enable_publishing and self.publisher is not None:
                try:
                    topic = build_topic_for_stage(
                        stage_enum,
                        instrument_id,
                        scheme=self.topic_scheme,
                        prefix=self.topic_prefix,
                    )
                except Exception:
                    topic = build_topic_for_stage(
                        Stage.CATALOG_WRITTEN,
                        instrument_id,
                        scheme=self.topic_scheme,
                        prefix=self.topic_prefix,
                    )

                payload = build_bus_payload(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    stage=stage_enum,
                    source=src_enum,
                    run_id=run_id,
                    ts_min=ts_min,
                    ts_max=ts_max,
                    count=count,
                    status=EventStatus.SUCCESS,
                    metadata={"correlation_id": correlation_id},
                )

                try:
                    self.publisher.publish(topic, payload)
                except Exception:
                    logger.exception("Message bus publish failed for topic %s", topic)

        except Exception:
            logger.warning("Failed to emit event/update watermark", exc_info=True)

    def _ensure_dataset_registered(
        self,
        dataset_id: str,
        dataset_type: DatasetType,
        instrument_id: str,
    ) -> None:
        """
        Ensure dataset is registered in the registry.

        Creates a basic manifest if dataset doesn't exist.

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
            self.contract_enforcer.get_manifest(dataset_id)
        except Exception:
            # Dataset not registered, create basic manifest
            logger.info("Auto-registering dataset %s (type=%s)", dataset_id, dataset_type)

            # Create minimal manifest
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

            # Register with registry, falling back to legacy API when needed
            registrar = getattr(self.registry, "register_manifest", None)
            if callable(registrar):
                registrar(manifest)
            else:
                legacy_registrar = getattr(self.registry, "register_dataset", None)
                if callable(legacy_registrar):
                    legacy_registrar(manifest)
                else:  # pragma: no cover - defensive for unexpected stubs
                    raise AttributeError(
                        "Registry missing register_manifest/register_dataset methods",
                    )

    def _get_stage_for_dataset_type(self, dataset_type: DatasetType) -> str:
        """
        Map dataset type to processing stage.

        Parameters
        ----------
        dataset_type : DatasetType
            Dataset type

        Returns
        -------
        str
            Processing stage value
        """
        stage_map = {
            DatasetType.FEATURES: Stage.FEATURE_COMPUTED.value,
            DatasetType.PREDICTIONS: Stage.PREDICTION_EMITTED.value,
            DatasetType.SIGNALS: Stage.SIGNAL_EMITTED.value,
            DatasetType.EARNINGS_ACTUALS: Stage.DATA_INGESTED.value,
            DatasetType.EARNINGS_ESTIMATES: Stage.DATA_INGESTED.value,
            DatasetType.MACRO_RELEASES: Stage.DATA_INGESTED.value,
            DatasetType.MACRO_OBSERVATIONS: Stage.DATA_INGESTED.value,
            DatasetType.EVENTS_CALENDAR: Stage.DATA_INGESTED.value,
            DatasetType.MICRO_MINUTE_FEATURES: Stage.FEATURE_COMPUTED.value,
            DatasetType.L2_MINUTE_FEATURES: Stage.FEATURE_COMPUTED.value,
        }
        return stage_map.get(dataset_type, Stage.DATA_INGESTED.value)

    def _write_feature_dataset(self, dataset_type: DatasetType, frame: DataFrameLike) -> None:
        """
        Delegate persistence of feature-adjacent datasets to FeatureDatasetStore.
        """
        if self.feature_dataset_store is None:
            raise RuntimeError("FeatureDatasetStore is not configured for dataset ingestion")
        if dataset_type == DatasetType.MACRO_RELEASES:
            self.feature_dataset_store.write_macro_releases(frame)
        elif dataset_type == DatasetType.MACRO_OBSERVATIONS:
            self.feature_dataset_store.write_macro_observations(frame)
        elif dataset_type == DatasetType.EVENTS_CALENDAR:
            self.feature_dataset_store.write_events_calendar(frame)
        elif dataset_type == DatasetType.MICRO_MINUTE_FEATURES:
            self.feature_dataset_store.write_micro_features(frame)
        elif dataset_type == DatasetType.L2_MINUTE_FEATURES:
            self.feature_dataset_store.write_l2_features(frame)
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported feature dataset type {dataset_type}")

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
        from ml._imports import HAS_POLARS
        from ml._imports import pl

        if isinstance(data, list):
            if HAS_POLARS and pl is not None:
                return cast(DataFrameLike, pl.DataFrame(data))
            if pd is not None:
                return cast(DataFrameLike, pd.DataFrame(data))
            raise RuntimeError("No DataFrame backend available to materialize list input")

        # Already DataFrame-like (pandas or polars)
        return data

    @staticmethod
    def _extract_ingestion_metadata_from_dataframe(data_frame: Any) -> dict[str, object]:
        """
        Extract metadata from DataFrame for event enrichment.

        Parameters
        ----------
        data_frame : Any
            pandas DataFrame

        Returns
        -------
        dict[str, object]
            Metadata dictionary
        """
        if pd is None or data_frame is None:
            return {}

        metadata: dict[str, object] = {}
        if data_frame.empty:
            return metadata

        if "source_dataset" in data_frame.columns:
            values = data_frame["source_dataset"].dropna().astype(str).unique().tolist()
            normalized = [value for value in values if value]
            if normalized:
                metadata["source_datasets"] = sorted(dict.fromkeys(normalized))

        return metadata

    def _data_frame_to_feature_data(
        self,
        data_frame: DataFrameLike,
        instrument_id: str,
    ) -> list[FeatureData]:
        """
        Convert DataFrame to list of FeatureData.

        Parameters
        ----------
        data_frame : DataFrameLike
            DataFrame with feature data
        instrument_id : str
            Instrument identifier

        Returns
        -------
        list[FeatureData]
            List of FeatureData objects
        """
        features = []
        data_frame_any = cast(Any, data_frame)

        # Handle both Polars and pandas DataFrames
        if hasattr(data_frame_any, "iter_rows"):
            # Polars DataFrame
            for row in data_frame_any.iter_rows(named=True):
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in feature row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init", ts_event_raw)
                ts_init = int(ts_init_raw)
                features.append(
                    FeatureData(
                        feature_set_id=row.get("feature_set_id", "default"),
                        instrument_id=instrument_id,
                        values=row.get("values", {}),
                        ts_event=ts_event,
                        ts_init=ts_init,
                    ),
                )
        elif hasattr(data_frame_any, "iterrows"):
            # pandas DataFrame
            for _, row in data_frame_any.iterrows():
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in feature row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init", ts_event_raw)
                ts_init = int(ts_init_raw)
                features.append(
                    FeatureData(
                        feature_set_id=row.get("feature_set_id", "default"),
                        instrument_id=instrument_id,
                        values=row.get("values", {}),
                        ts_event=ts_event,
                        ts_init=ts_init,
                    ),
                )
        else:
            # Fallback for list of dicts
            for row in data_frame_any:
                if isinstance(row, dict):
                    ts_event_raw = row.get("ts_event")
                    if ts_event_raw is None:
                        raise ValueError("Missing ts_event in feature row")
                    ts_event = int(ts_event_raw)
                    ts_init_raw = row.get("ts_init", ts_event_raw)
                    ts_init = int(ts_init_raw)
                    features.append(
                        FeatureData(
                            feature_set_id=row.get("feature_set_id", "default"),
                            instrument_id=instrument_id,
                            values=row.get("values", {}),
                            ts_event=ts_event,
                            ts_init=ts_init,
                        ),
                    )

        return features

    def _data_frame_to_predictions(
        self,
        data_frame: DataFrameLike,
    ) -> list[ModelPrediction]:
        """
        Convert DataFrame to list of ModelPrediction.

        Parameters
        ----------
        data_frame : DataFrameLike | list[dict]
            DataFrame with prediction data

        Returns
        -------
        list[ModelPrediction]
            List of ModelPrediction objects
        """
        predictions = []
        data_frame_any = cast(Any, data_frame)

        # Handle both Polars and pandas DataFrames
        if hasattr(data_frame_any, "iter_rows"):
            # Polars DataFrame
            for row in data_frame_any.iter_rows(named=True):
                features_used = cast(dict[str, float], row.get("features_used", row.get("features", {})) or {})
                inference_time_ms = float(row.get("inference_time_ms", 0.0))
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in prediction row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init", ts_event_raw)
                ts_init = int(ts_init_raw)
                predictions.append(
                    ModelPrediction(
                        model_id=row["model_id"],
                        instrument_id=row["instrument_id"],
                        prediction=float(row.get("prediction", row.get("value", 0.0))),
                        confidence=float(row.get("confidence", 0.0)),
                        features_used=features_used,
                        inference_time_ms=inference_time_ms,
                        _ts_event=ts_event,
                        _ts_init=ts_init,
                        is_live=bool(row.get("is_live", False)),
                    ),
                )
        elif hasattr(data_frame_any, "iterrows"):
            # pandas DataFrame
            for _, row in data_frame_any.iterrows():
                features_used = cast(dict[str, float], row.get("features_used", row.get("features", {})) or {})
                inference_time_ms = float(row.get("inference_time_ms", 0.0))
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in prediction row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init", ts_event_raw)
                ts_init = int(ts_init_raw)
                predictions.append(
                    ModelPrediction(
                        model_id=row["model_id"],
                        instrument_id=row["instrument_id"],
                        prediction=float(row.get("prediction", row.get("value", 0.0))),
                        confidence=float(row.get("confidence", 0.0)),
                        features_used=features_used,
                        inference_time_ms=inference_time_ms,
                        _ts_event=ts_event,
                        _ts_init=ts_init,
                        is_live=bool(row.get("is_live", False)),
                    ),
                )

        return predictions

    def _data_frame_to_signals(
        self,
        data_frame: DataFrameLike | list[dict[str, Any]],
    ) -> list[StrategySignal]:
        """
        Convert DataFrame to list of StrategySignal.

        Parameters
        ----------
        data_frame : DataFrameLike | list[dict]
            DataFrame with signal data

        Returns
        -------
        list[StrategySignal]
            List of StrategySignal objects
        """
        signals = []
        data_frame_any = cast(Any, data_frame)

        # Handle both Polars and pandas DataFrames
        if hasattr(data_frame_any, "iter_rows"):
            # Polars DataFrame
            for row in data_frame_any.iter_rows(named=True):
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in signal row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init", ts_event_raw)
                ts_init = int(ts_init_raw)
                signals.append(
                    StrategySignal(
                        strategy_id=row["strategy_id"],
                        instrument_id=row["instrument_id"],
                        signal_type=row["signal_type"],
                        strength=float(row.get("strength", row.get("signal_value", 0.0))),
                        model_predictions=row.get("model_predictions", {}),
                        risk_metrics=row.get("risk_metrics", {}),
                        execution_params=row.get("execution_params", {}),
                        _ts_event=ts_event,
                        _ts_init=ts_init,
                    ),
                )
        elif hasattr(data_frame_any, "iterrows"):
            # pandas DataFrame
            for _, row in data_frame_any.iterrows():
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in signal row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init", ts_event_raw)
                ts_init = int(ts_init_raw)
                signals.append(
                    StrategySignal(
                        strategy_id=row["strategy_id"],
                        instrument_id=row["instrument_id"],
                        signal_type=row["signal_type"],
                        strength=float(row.get("strength", row.get("signal_value", 0.0))),
                        model_predictions=row.get("model_predictions", {}),
                        risk_metrics=row.get("risk_metrics", {}),
                        execution_params=row.get("execution_params", {}),
                        _ts_event=ts_event,
                        _ts_init=ts_init,
                    ),
                )
        else:
            # Fallback for list of dicts
            for row in data_frame_any:
                if isinstance(row, dict):
                    ts_event_raw = row.get("ts_event")
                    if ts_event_raw is None:
                        raise ValueError("Missing ts_event in signal row")
                    ts_event = int(ts_event_raw)
                    ts_init_raw = row.get("ts_init") or ts_event_raw
                    ts_init = int(ts_init_raw)
                    signals.append(
                        StrategySignal(
                            strategy_id=row["strategy_id"],
                            instrument_id=row["instrument_id"],
                            signal_type=row["signal_type"],
                            strength=float(row.get("strength", row.get("signal_value", 0.0)) or 0.0),
                            model_predictions=row.get("model_predictions", {}),
                            risk_metrics=row.get("risk_metrics", {}),
                            execution_params=row.get("execution_params", {}),
                            _ts_event=ts_event,
                            _ts_init=ts_init,
                        ),
                    )

        return signals

    def _create_partial_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        record_count: int,
        reason: str,
        metadata: dict[str, object] | None = None,
    ) -> DataEvent:
        """
        Create a PARTIAL status event.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : str
            Data source
        run_id : str
            Processing run identifier
        ts_min : int
            Minimum timestamp
        ts_max : int
            Maximum timestamp
        record_count : int
            Record count
        reason : str
            Reason for partial status
        metadata : dict[str, object] | None
            Additional metadata

        Returns
        -------
        DataEvent
            Partial status event
        """
        event_metadata = metadata or {}
        event_metadata["reason"] = reason
        event_metadata["no_write"] = True

        return DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_ingestion",
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            record_count=record_count,
            status=EventStatus.PARTIAL.value,
            metadata=event_metadata,
        )

    def _create_failed_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        record_count: int,
        error: str,
    ) -> DataEvent:
        """
        Create a FAILED status event.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        instrument_id : str
            Instrument identifier
        source : str
            Data source
        run_id : str
            Processing run identifier
        ts_min : int
            Minimum timestamp
        ts_max : int
            Maximum timestamp
        record_count : int
            Record count
        error : str
            Error message

        Returns
        -------
        DataEvent
            Failed status event
        """
        return DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            operation="write_ingestion",
            source=source,
            run_id=run_id,
            ts_min=ts_min,
            ts_max=ts_max,
            record_count=record_count,
            status=EventStatus.FAILED.value,
            error_message=error,
        )
