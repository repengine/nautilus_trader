#!/usr/bin/env python3

"""
Data writer component for DataStore.

Extracted from DataStore (Phase 2.4.2). Provides write operations for ingestion,
features, predictions, signals, and earnings data.

All methods are COLD path (async operations acceptable).

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

from ml._imports import HAS_PROMETHEUS
from ml.common.events_util import stage_for_dataset_type
from ml.common.metrics_bootstrap import get_counter
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.ml_types import DataFrameLike
from ml.registry.dataclasses import DatasetType
from ml.stores.validation_types import DataEvent


if TYPE_CHECKING:
    from ml.actors.types import FeatureData
    from ml.actors.types import ModelPrediction
    from ml.actors.types import StrategySignal
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.common.protocols import EventEmitterProtocol
    from ml.stores.common.protocols import SchemaValidatorProtocol
    from ml.stores.earnings_store import EarningsStore
    from ml.stores.feature_store_facade import FeatureStore
    from ml.stores.io_raw import RawIngestionWriterProtocol
    from ml.stores.model_store import ModelStore
    from ml.stores.strategy_store import StrategyStore

logger = logging.getLogger(__name__)

__all__ = [
    "DataEvent",
    "DataWriterComponent",
]


# =========================================================================
# Constants
# =========================================================================

EARNINGS_ACTUALS_DATASET_ID = "earnings_actuals"
EARNINGS_ESTIMATES_DATASET_ID = "earnings_estimates"


# =========================================================================
# Prometheus Metrics (using centralized bootstrap - CLAUDE.md Pattern 5)
# =========================================================================


# Get metrics via bootstrap (returns dummy metrics if Prometheus unavailable)
write_rejection_counter = get_counter(
    "ml_write_rejections_total",
    "Total number of write rejections",
    labelnames=["dataset_id", "reason"],
)


# =========================================================================
# DataWriterComponent
# =========================================================================


class DataWriterComponent:
    """
    Data writing operations for DataStore.

    Extracted from DataStore (Phase 2.4.2).
    All methods are COLD path (async operations acceptable).

    Provides:
    - Write ingestion data with validation
    - Write features to FeatureStore
    - Write predictions to ModelStore
    - Write signals to StrategyStore
    - Write earnings actuals
    - Write earnings estimates

    Example
    -------
    >>> from ml.stores.common.data_writer import DataWriterComponent
    >>> writer = DataWriterComponent(
    ...     feature_store=feature_store,
    ...     model_store=model_store,
    ...     strategy_store=strategy_store,
    ...     earnings_store=earnings_store,
    ...     validator=validator,
    ...     registry=registry,
    ... )
    >>> event = writer.write_features(
    ...     instrument_id="EURUSD.SIM",
    ...     features=feature_list,
    ...     source="computed",
    ... )

    """

    def __init__(
        self,
        feature_store: FeatureStore,
        model_store: ModelStore,
        strategy_store: StrategyStore,
        earnings_store: EarningsStore,
        validator: SchemaValidatorProtocol,
        registry: RegistryProtocol,
        *,
        event_emitter: EventEmitterProtocol | None = None,
        raw_writer: RawIngestionWriterProtocol | None = None,
        fail_on_validation_error: bool = True,
    ) -> None:
        """
        Initialize data writer with store dependencies.

        Args:
            feature_store: FeatureStore for feature data
            model_store: ModelStore for prediction data
            strategy_store: StrategyStore for signal data
            earnings_store: EarningsStore for earnings data
            validator: Schema validator component
            registry: Data registry for manifest/contract lookup
            event_emitter: Optional event emitter for registry/bus events
            raw_writer: Raw ingestion writer for Parquet backup (dual-write)
            fail_on_validation_error: If True, raise on validation failures

        """
        self._feature_store = feature_store
        self._model_store = model_store
        self._strategy_store = strategy_store
        self._earnings_store = earnings_store
        self._event_emitter = event_emitter
        self._raw_writer = raw_writer
        self._validator = validator
        self._registry = registry
        self._fail_on_validation_error = fail_on_validation_error

    # =========================================================================
    # Public API
    # =========================================================================

    def write_ingestion(
        self,
        dataset_id: str,
        records: list[dict[str, Any]] | DataFrameLike,
        source: Source | str,
        run_id: str,
        instrument_id: str | None = None,
    ) -> DataEvent:
        """
        Write ingestion data with contract validation and event emission.

        EXTRACTED FROM: ml/stores/data_store_facade.py:1204

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
        >>> event = writer.write_ingestion(
        ...     dataset_id="bars_eurusd_1m",
        ...     records=bar_data,
        ...     source="historical",
        ...     run_id="run_20240101_120000"
        ... )
        >>> print(f"Wrote {event.record_count} records")

        """
        start_time = time.perf_counter()

        # Get manifest and contract
        manifest = self._registry.get_manifest(dataset_id)
        contract = self._registry.get_contract(dataset_id)

        # Perform preflight schema check
        preflight_passed, preflight_error, preflight_details = self._validator.preflight_check(
            dataset_id,
            records,
            strict=self._fail_on_validation_error,
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

        # Convert to DataFrame for validation
        data_frame_obj = self._to_dataframe(records)
        data_frame = cast(DataFrameLike, data_frame_obj)

        # Extract instrument_id if not provided
        if instrument_id is None:
            instrument_id = self._extract_instrument_id(data_frame)

        # Validate batch
        use_strict = contract.enforcement_mode == "strict"
        quality_report = self._validator.validate_batch(dataset_id, data_frame, strict_mode=use_strict)

        # Enforce quality thresholds
        self._enforce_quality_report(dataset_id, contract, quality_report)

        # Calculate timestamp range and stage
        ts_min, ts_max = self._extract_timestamp_range(data_frame, manifest)
        dataset_type = manifest.dataset_type
        stage = stage_for_dataset_type(dataset_type)

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_writer.write_ingestion:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_writer.write_ingestion:ts_max")

        try:
            record_count = len(data_frame)
        except Exception:
            record_count = 0

        base_metadata: dict[str, Any] = {"quality_score": quality_report.quality_score}

        if dataset_type == DatasetType.FEATURES:
            try:
                self._write_to_feature_store(data_frame, instrument_id)
            except Exception as exc:
                self._emit_event(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id or "UNKNOWN",
                    stage=stage,
                    source=source,
                    run_id=run_id,
                    ts_min=ts_min_s,
                    ts_max=ts_max_s,
                    count=0,
                    status=EventStatus.FAILED,
                    error=str(exc),
                    metadata=base_metadata,
                )
                raise RuntimeError(f"Write operation failed for {dataset_id}: {exc}") from exc

        elif dataset_type == DatasetType.PREDICTIONS:
            try:
                self._write_to_model_store(data_frame, instrument_id)
            except Exception as exc:
                self._emit_event(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id or "UNKNOWN",
                    stage=stage,
                    source=source,
                    run_id=run_id,
                    ts_min=ts_min_s,
                    ts_max=ts_max_s,
                    count=0,
                    status=EventStatus.FAILED,
                    error=str(exc),
                    metadata=base_metadata,
                )
                raise RuntimeError(f"Write operation failed for {dataset_id}: {exc}") from exc

        elif dataset_type == DatasetType.SIGNALS:
            try:
                self._write_to_strategy_store(data_frame, instrument_id)
            except Exception as exc:
                self._emit_event(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id or "UNKNOWN",
                    stage=stage,
                    source=source,
                    run_id=run_id,
                    ts_min=ts_min_s,
                    ts_max=ts_max_s,
                    count=0,
                    status=EventStatus.FAILED,
                    error=str(exc),
                    metadata=base_metadata,
                )
                raise RuntimeError(f"Write operation failed for {dataset_id}: {exc}") from exc

        else:
            if self._raw_writer is not None:
                try:
                    written = self._raw_writer.write(
                        dataset_type=dataset_type,
                        data=data_frame,
                    )
                    if written <= 0:
                        event = self._create_partial_event(
                            dataset_id=dataset_id,
                            instrument_id=instrument_id or "UNKNOWN",
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min_s,
                            ts_max=ts_max_s,
                            record_count=record_count,
                            reason="no_records_written",
                            metadata=base_metadata,
                        )
                        self._emit_event(
                            dataset_id=dataset_id,
                            instrument_id=instrument_id or "UNKNOWN",
                            stage=stage,
                            source=source,
                            run_id=run_id,
                            ts_min=ts_min_s,
                            ts_max=ts_max_s,
                            count=record_count,
                            status=EventStatus.PARTIAL,
                            metadata=event.metadata,
                        )
                        return event
                except Exception as exc:
                    event = self._create_failed_event(
                        dataset_id=dataset_id,
                        instrument_id=instrument_id or "UNKNOWN",
                        source=source,
                        run_id=run_id,
                        ts_min=ts_min_s,
                        ts_max=ts_max_s,
                        record_count=0,
                        error=str(exc),
                    )
                    self._emit_event(
                        dataset_id=dataset_id,
                        instrument_id=instrument_id or "UNKNOWN",
                        stage=stage,
                        source=source,
                        run_id=run_id,
                        ts_min=ts_min_s,
                        ts_max=ts_max_s,
                        count=0,
                        status=EventStatus.FAILED,
                        error=str(exc),
                        metadata=event.metadata,
                    )
                    return event
            else:
                if not self._fail_on_validation_error:
                    event = self._create_partial_event(
                        dataset_id=dataset_id,
                        instrument_id=instrument_id or "UNKNOWN",
                        source=source,
                        run_id=run_id,
                        ts_min=ts_min_s,
                        ts_max=ts_max_s,
                        record_count=record_count,
                        reason="raw_writer_missing",
                        metadata=base_metadata,
                    )
                    self._emit_event(
                        dataset_id=dataset_id,
                        instrument_id=instrument_id or "UNKNOWN",
                        stage=stage,
                        source=source,
                        run_id=run_id,
                        ts_min=ts_min_s,
                        ts_max=ts_max_s,
                        count=record_count,
                        status=EventStatus.PARTIAL,
                        metadata=event.metadata,
                    )
                    return event
                logger.info("Raw writer not configured; proceeding with success for %s", dataset_id)

        # Calculate write duration
        write_duration_ms = (time.perf_counter() - start_time) * 1000.0

        event = DataEvent(
            event_id=f"{run_id}_{dataset_id}_{time.time_ns()}",
            dataset_id=dataset_id,
            instrument_id=instrument_id or "UNKNOWN",
            operation="write_ingestion",
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            record_count=record_count,
            status=EventStatus.SUCCESS.value,
            metadata={
                **base_metadata,
                "write_duration_ms": write_duration_ms,
            },
        )

        self._emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id or "UNKNOWN",
            stage=stage,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=record_count,
            status=EventStatus.SUCCESS,
            metadata=event.metadata,
        )
        self._update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id or "UNKNOWN",
            source=source,
            last_success_ns=ts_max_s,
            count=record_count,
        )

        logger.debug(
            "Wrote %d records for %s (quality=%.2f, duration=%.1fms)",
            record_count,
            dataset_id,
            quality_report.quality_score,
            write_duration_ms,
        )

        return event

    def write_features(
        self,
        instrument_id: str,
        features: list[FeatureData],
        source: str = "computed",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write features with validation and event emission.

        EXTRACTED FROM: ml/stores/data_store_facade.py:1709

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
        >>> event = writer.write_features(
        ...     instrument_id="EUR/USD",
        ...     features=feature_list,
        ...     source="realtime"
        ... )

        """
        run_id = run_id or f"features_{time.time_ns()}"
        dataset_id = "features"

        # Ensure dataset is registered
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

        stage = stage_for_dataset_type(DatasetType.FEATURES)

        # Calculate timestamp range
        ts_min = min(f.ts_event for f in features)
        ts_max = max(f.ts_event for f in features)

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_writer.write_features:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_writer.write_features:ts_max")

        try:
            for feature in features:
                self._feature_store.write_features(
                    feature_set_id=feature.feature_set_id,
                    instrument_id=feature.instrument_id,
                    features=feature.values,
                    ts_event=feature.ts_event,
                    ts_init=feature.ts_init,
                    publish_bus=False,
                )
        except Exception as exc:
            logger.error("Feature store write failed", exc_info=True)
            self._emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source,
                run_id=run_id,
                ts_min=ts_min_s,
                ts_max=ts_max_s,
                count=0,
                status=EventStatus.FAILED,
                error=str(exc),
            )
            raise RuntimeError(f"Feature write failed: {exc}") from exc

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
            metadata={},
        )

        self._emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(features),
            status=EventStatus.SUCCESS,
            metadata=event.metadata,
        )
        self._update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source,
            last_success_ns=ts_max_s,
            count=len(features),
        )

        logger.debug(
            "Wrote %d features for %s",
            len(features),
            instrument_id,
        )

        return event

    def write_predictions(
        self,
        predictions: list[ModelPrediction],
        source: str = "inference",
        run_id: str | None = None,
    ) -> DataEvent:
        """
        Write model predictions with validation and event emission.

        EXTRACTED FROM: ml/stores/data_store_facade.py:1827

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
        >>> event = writer.write_predictions(
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

        # Ensure dataset is registered
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.PREDICTIONS,
            instrument_id=instrument_id,
        )

        stage = stage_for_dataset_type(DatasetType.PREDICTIONS)

        # Calculate timestamp range
        ts_min = min(p.ts_event for p in predictions)
        ts_max = max(p.ts_event for p in predictions)

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_writer.write_predictions:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_writer.write_predictions:ts_max")

        # Store predictions
        try:
            try:
                self._model_store.write_batch(predictions, emit_events=False, publish_bus=False)
            except TypeError:
                # Fallback for legacy API
                self._model_store.write_batch(predictions)
        except Exception as exc:
            logger.error("Model store write failed", exc_info=True)
            self._emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source,
                run_id=run_id,
                ts_min=ts_min_s,
                ts_max=ts_max_s,
                count=0,
                status=EventStatus.FAILED,
                error=str(exc),
                metadata={"model_id": model_id},
            )
            raise RuntimeError(f"Prediction write failed: {exc}") from exc

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

        self._emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(predictions),
            status=EventStatus.SUCCESS,
            metadata=event.metadata,
        )
        self._update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source,
            last_success_ns=ts_max_s,
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

        EXTRACTED FROM: ml/stores/data_store_facade.py:1924

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
        >>> event = writer.write_signals(
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

        # Ensure dataset is registered
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.SIGNALS,
            instrument_id=instrument_id,
        )

        stage = stage_for_dataset_type(DatasetType.SIGNALS)

        # Calculate timestamp range
        ts_min = min(s.ts_event for s in signals)
        ts_max = max(s.ts_event for s in signals)

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

        ts_min_s = _sanitize(int(ts_min), context="data_writer.write_signals:ts_min")
        ts_max_s = _sanitize(int(ts_max), context="data_writer.write_signals:ts_max")

        # Store signals
        try:
            self._strategy_store.write_batch(signals, emit_events=False, publish_bus=False)
        except Exception as exc:
            logger.error("Strategy store write failed", exc_info=True)
            self._emit_event(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                stage=stage,
                source=source,
                run_id=run_id,
                ts_min=ts_min_s,
                ts_max=ts_max_s,
                count=0,
                status=EventStatus.FAILED,
                error=str(exc),
                metadata={"strategy_id": strategy_id},
            )
            raise RuntimeError(f"Signal write failed: {exc}") from exc

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

        self._emit_event(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            stage=stage,
            source=source,
            run_id=run_id,
            ts_min=ts_min_s,
            ts_max=ts_max_s,
            count=len(signals),
            status=EventStatus.SUCCESS,
            metadata=event.metadata,
        )
        self._update_watermark(
            dataset_id=dataset_id,
            instrument_id=instrument_id,
            source=source,
            last_success_ns=ts_max_s,
            count=len(signals),
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

        EXTRACTED FROM: ml/stores/data_store_facade.py:2018

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        period_end : str
            Period end date (YYYY-MM-DD)
        filing_date : str
            Filing date (YYYY-MM-DD)
        eps_diluted : float | None
            Diluted earnings per share
        revenue : float | None
            Revenue amount
        ts_event : int
            Event timestamp (nanoseconds)
        ts_init : int
            Initialization timestamp (nanoseconds)
        eps_basic : float | None
            Basic earnings per share
        net_income : float | None
            Net income amount
        operating_income : float | None
            Operating income amount
        shares_outstanding : int | None
            Number of shares outstanding
        filing_type : str | None
            Filing type (10-Q, 10-K, etc.)
        fiscal_year : int | None
            Fiscal year
        fiscal_quarter : int | None
            Fiscal quarter (1-4)
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
        stage = stage_for_dataset_type(DatasetType.EARNINGS_ACTUALS)

        # Ensure dataset is registered
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.EARNINGS_ACTUALS,
            instrument_id=ticker,
        )

        # Build record for validation
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
        }

        # Validate record
        contract = self._registry.get_contract(dataset_id)
        use_strict = contract.enforcement_mode == "strict"
        quality_report = self._validator.validate_batch(dataset_id, [record], strict_mode=use_strict)
        self._enforce_quality_report(
            dataset_id=dataset_id,
            contract=contract,
            quality_report=quality_report,
        )

        # Write to earnings store (SQL)
        try:
            self._earnings_store.write_actuals(
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
            logger.error("Earnings actual write failed for %s", ticker, exc_info=True)
            self._emit_event(
                dataset_id=dataset_id,
                instrument_id=ticker,
                stage=stage,
                source=source,
                run_id=run_id_local,
                ts_min=ts_event_s,
                ts_max=ts_event_s,
                count=0,
                status=EventStatus.FAILED,
                error=str(exc),
            )
            raise RuntimeError(f"Earnings actual write failed: {exc}") from exc

        # Dual-write to raw_writer (Parquet backup) if available
        raw_writer_status = "skipped"
        if self._raw_writer is not None:
            try:
                self._raw_writer.write(
                    dataset_type=DatasetType.EARNINGS_ACTUALS,
                    data=[record],
                )
                raw_writer_status = "ok"
            except Exception as exc:
                logger.warning(
                    "Raw writer backup failed for earnings actual %s (non-fatal): %s",
                    ticker,
                    exc,
                    exc_info=True,
                )
                raw_writer_status = "failed"

        # Create success event
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
            metadata={
                "quality_score": quality_report.quality_score,
                "raw_writer_status": raw_writer_status,
            },
        )

        self._emit_event(
            dataset_id=dataset_id,
            instrument_id=ticker,
            stage=stage,
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            count=1,
            status=EventStatus.SUCCESS,
            metadata=event.metadata,
        )
        self._update_watermark(
            dataset_id=dataset_id,
            instrument_id=ticker,
            source=source,
            last_success_ns=ts_event_s,
            count=1,
        )

        logger.debug("Wrote earnings actual for %s", ticker)

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

        EXTRACTED FROM: ml/stores/data_store_facade.py:2128

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        estimate_date : str
            Estimate date (YYYY-MM-DD)
        period_end : str
            Period end date (YYYY-MM-DD)
        eps_consensus : float | None
            Consensus EPS estimate
        ts_event : int
            Event timestamp (nanoseconds)
        ts_init : int
            Initialization timestamp (nanoseconds)
        revenue_consensus : float | None
            Consensus revenue estimate
        num_analysts : int | None
            Number of analysts in consensus
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
        stage = stage_for_dataset_type(DatasetType.EARNINGS_ESTIMATES)

        # Ensure dataset is registered
        self._ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=DatasetType.EARNINGS_ESTIMATES,
            instrument_id=ticker,
        )

        # Build record for validation
        record: dict[str, Any] = {
            "ticker": ticker,
            "estimate_date": estimate_date,
            "period_end": period_end,
            "ts_event": ts_event_s,
            "ts_init": ts_init_s,
            "eps_consensus": eps_consensus,
            "revenue_consensus": revenue_consensus,
            "num_analysts": num_analysts,
        }

        # Validate record
        contract = self._registry.get_contract(dataset_id)
        use_strict = contract.enforcement_mode == "strict"
        quality_report = self._validator.validate_batch(dataset_id, [record], strict_mode=use_strict)
        self._enforce_quality_report(
            dataset_id=dataset_id,
            contract=contract,
            quality_report=quality_report,
        )

        # Write to earnings store (SQL)
        try:
            self._earnings_store.write_estimates(
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
            logger.error("Earnings estimate write failed for %s", ticker, exc_info=True)
            self._emit_event(
                dataset_id=dataset_id,
                instrument_id=ticker,
                stage=stage,
                source=source,
                run_id=run_id_local,
                ts_min=ts_event_s,
                ts_max=ts_event_s,
                count=0,
                status=EventStatus.FAILED,
                error=str(exc),
            )
            raise RuntimeError(f"Earnings estimate write failed: {exc}") from exc

        # Dual-write to raw_writer (Parquet backup) if available
        raw_writer_status = "skipped"
        if self._raw_writer is not None:
            try:
                self._raw_writer.write(
                    dataset_type=DatasetType.EARNINGS_ESTIMATES,
                    data=[record],
                )
                raw_writer_status = "ok"
            except Exception as exc:
                logger.warning(
                    "Raw writer backup failed for earnings estimate %s (non-fatal): %s",
                    ticker,
                    exc,
                    exc_info=True,
                )
                raw_writer_status = "failed"

        # Create success event
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
            metadata={
                "quality_score": quality_report.quality_score,
                "raw_writer_status": raw_writer_status,
            },
        )

        self._emit_event(
            dataset_id=dataset_id,
            instrument_id=ticker,
            stage=stage,
            source=source,
            run_id=run_id_local,
            ts_min=ts_event_s,
            ts_max=ts_event_s,
            count=1,
            status=EventStatus.SUCCESS,
            metadata=event.metadata,
        )
        self._update_watermark(
            dataset_id=dataset_id,
            instrument_id=ticker,
            source=source,
            last_success_ns=ts_event_s,
            count=1,
        )

        logger.debug("Wrote earnings estimate for %s", ticker)

        return event

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _to_dataframe(
        self,
        data: DataFrameLike | list[dict[str, Any]],
    ) -> DataFrameLike | list[dict[str, Any]]:
        """
        Convert various data formats to DataFrame-like or pass-through list.
        """
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
            if HAS_POLARS and pl is not None:
                return cast(DataFrameLike, pl.DataFrame(data))
            return data

        # Return as is for other formats
        return data

    def _extract_instrument_id(self, data_frame: DataFrameLike) -> str:
        """
        Extract instrument_id from data frame.
        """
        data_frame_any = cast(Any, data_frame)

        if hasattr(data_frame_any, "columns") and "instrument_id" in data_frame_any.columns:
            # Get first non-null instrument_id
            instrument_ids = data_frame_any["instrument_id"]
            if hasattr(instrument_ids, "drop_nulls"):
                # Polars
                non_null = instrument_ids.drop_nulls()
                if len(non_null) > 0:
                    return str(non_null[0])
            elif hasattr(instrument_ids, "dropna"):
                # pandas
                non_null = instrument_ids.dropna()
                if len(non_null) > 0:
                    return str(non_null.iloc[0])

        if isinstance(data_frame_any, list) and data_frame_any:
            for row in data_frame_any:
                if isinstance(row, dict):
                    instrument_value = row.get("instrument_id")
                    if instrument_value is not None:
                        return str(instrument_value)

        return "UNKNOWN"

    def _extract_timestamp_range(self, data_frame: DataFrameLike, manifest: Any) -> tuple[int, int]:
        """
        Extract min/max timestamps from data frame.
        """
        data_frame_any = cast(Any, data_frame)
        ts_field = manifest.ts_field

        if hasattr(data_frame_any, "columns") and ts_field in data_frame_any.columns:
            ts_col = data_frame_any[ts_field]
            ts_min = int(ts_col.min())
            ts_max = int(ts_col.max())
            return ts_min, ts_max

        if isinstance(data_frame_any, list) and data_frame_any:
            ts_values: list[int] = []
            for row in data_frame_any:
                if isinstance(row, dict) and ts_field in row:
                    ts_value = row.get(ts_field)
                    if ts_value is not None:
                        ts_values.append(int(ts_value))
            if ts_values:
                return min(ts_values), max(ts_values)

        # Default to current time if no timestamp field
        current_ns = time.time_ns()
        return current_ns, current_ns

    def _enforce_quality_report(
        self,
        dataset_id: str,
        contract: Any,
        quality_report: Any,
    ) -> None:
        """
        Enforce quality thresholds from contract.
        """
        if contract.enforcement_mode == "strict" and quality_report.quality_score < 1.0:
            # Record write rejection metric
            if HAS_PROMETHEUS:
                write_rejection_counter.labels(
                    dataset_id=dataset_id,
                    reason="validation_failed",
                ).inc()

            raise ValueError(
                f"Data validation failed for {dataset_id} (fail-closed mode). "
                f"Quality score: {quality_report.quality_score:.2f}, "
                f"Violations: {len(quality_report.violations)}",
            )

        # Log advisory issues for non-strict modes
        if quality_report.quality_score < 1.0:
            if contract.enforcement_mode == "monitor_only":
                logger.info(
                    "Quality issues for %s (monitor-only): score=%.2f, violations=%d",
                    dataset_id,
                    quality_report.quality_score,
                    len(quality_report.violations),
                )
            else:
                logger.warning(
                    "Quality issues for %s: score=%.2f, violations=%d",
                    dataset_id,
                    quality_report.quality_score,
                    len(quality_report.violations),
                )

    def _ensure_dataset_registered(
        self,
        dataset_id: str,
        dataset_type: DatasetType,
        instrument_id: str,
    ) -> None:
        """
        Ensure dataset is registered in registry.
        """
        try:
            self._registry.get_manifest(dataset_id)
        except Exception:
            # Dataset not registered, log warning
            logger.debug(
                "Dataset %s not registered (type=%s, instrument=%s)",
                dataset_id,
                dataset_type,
                instrument_id,
                exc_info=True,
            )

    def _emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Stage | str,
        source: str,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus | str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit a dataset event using the configured event emitter (non-blocking).
        """
        if self._event_emitter is None:
            logger.debug(
                "Event emitter not configured; skipping event emission",
                extra={"dataset_id": dataset_id, "instrument_id": instrument_id},
            )
            return
        try:
            self._event_emitter.emit_event(
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
        except Exception as exc:
            logger.warning(
                "Event emission failed for %s: %s",
                dataset_id,
                exc,
                exc_info=True,
            )

    def _update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: Source | str,
        last_success_ns: int,
        count: int,
    ) -> None:
        """
        Update registry watermark (best-effort, non-blocking).
        """
        from ml.common.events_util import to_source_enum

        try:
            self._registry.update_watermark(
                dataset_id=dataset_id,
                instrument_id=instrument_id,
                source=to_source_enum(source),
                last_success_ns=last_success_ns,
                count=count,
                completeness_pct=100.0,
            )
        except Exception as exc:
            logger.warning(
                "Watermark update failed for %s: %s",
                dataset_id,
                exc,
                exc_info=True,
            )

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
        metadata: dict[str, Any] | None = None,
    ) -> DataEvent:
        """
        Create a PARTIAL status event for ingestion.
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
        Create a FAILED status event for ingestion.
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

    def _write_to_feature_store(self, data_frame: DataFrameLike, instrument_id: str) -> None:
        """
        Write data to FeatureStore.
        """
        from ml.stores.common.data_frame_converters import data_frame_to_feature_data

        logger.debug("Writing features for %s", instrument_id)
        features = data_frame_to_feature_data(data_frame, instrument_id)
        for feature in features:
            try:
                self._feature_store.write_features(
                    feature_set_id=feature.feature_set_id,
                    instrument_id=feature.instrument_id,
                    features=feature.values,
                    ts_event=feature.ts_event,
                    ts_init=feature.ts_init,
                    publish_bus=False,
                )
            except TypeError:
                self._feature_store.write_features(
                    feature_set_id=feature.feature_set_id,
                    instrument_id=feature.instrument_id,
                    features=feature.values,
                    ts_event=feature.ts_event,
                    ts_init=feature.ts_init,
                )

    def _write_to_model_store(self, data_frame: DataFrameLike, instrument_id: str) -> None:
        """
        Write data to ModelStore.
        """
        from ml.stores.common.data_frame_converters import data_frame_to_predictions

        logger.debug("Writing predictions for %s", instrument_id)
        predictions = data_frame_to_predictions(data_frame)
        try:
            self._model_store.write_batch(predictions, emit_events=False, publish_bus=False)
        except TypeError:
            self._model_store.write_batch(predictions)

    def _write_to_strategy_store(self, data_frame: DataFrameLike, instrument_id: str) -> None:
        """
        Write data to StrategyStore.
        """
        from ml.stores.common.data_frame_converters import data_frame_to_signals

        logger.debug("Writing signals for %s", instrument_id)
        signals = data_frame_to_signals(data_frame)
        self._strategy_store.write_batch(signals, emit_events=False, publish_bus=False)
