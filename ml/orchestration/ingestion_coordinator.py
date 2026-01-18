#!/usr/bin/env python3

"""
Ingestion coordination for ML pipeline orchestrator.

This module provides comprehensive ingestion coordination including backfill
management, auto-fill universe population, pre-ingestion tasks, and integration
with ingestion services.

This component is extracted from the MLPipelineOrchestrator god class to provide
focused, testable ingestion coordination functionality.

"""

from __future__ import annotations

import importlib
import json
import logging
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.coverage import CoveragePolicy
from ml.config.coverage import get_max_lookback_days
from ml.data.dataset_manifest_defaults import build_auto_dataset_manifest
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.orchestrator import BackfillWindowList
from ml.data.ingest.orchestrator import DomainWindowLoaderProtocol
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.service import DatabentoIngestionService
from ml.orchestration.config_types import AutoFillUniverseConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import EarningsCoordinatorConfig
from ml.orchestration.config_types import MacroIngestionConfig
from ml.orchestration.config_types import PreIngestionOptions
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.schema import map_schema_to_dataset_type


__all__ = ["IngestionCoordinator", "IngestionOrchestrator"]
from ml.registry.protocols import RegistryProtocol
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.providers import DAY_NS
from ml.stores.raw_protocols import RawIngestionWriterProtocol
from ml.tasks.ingest import PopulateL2TaskConfig
from ml.tasks.ingest import populate_l2_efficient


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ml.config.scheduler_config import SchedulerConfig
    from ml.data.ingest.orchestrator import IngestionOrchestrator
    from ml.orchestration.discovery_client import DiscoveryClient


logger = logging.getLogger(__name__)


# ========================================================================
# Metrics Dataclasses
# ========================================================================


class _AutoFillMetrics:
    """Auto-fill universe operation metrics."""

    def __init__(self) -> None:
        self.operations_total = get_counter(
            "ml_auto_fill_operations_total",
            "Total auto-fill operations by schema and status",
            labelnames=("schema", "status"),
        )
        self.latency_seconds = get_histogram(
            "ml_auto_fill_latency_seconds",
            "Auto-fill operation latency in seconds",
            labelnames=("schema",),
        )

    @staticmethod
    def default() -> _AutoFillMetrics:
        return _AutoFillMetrics()


# ========================================================================
# Protocol Definition
# ========================================================================


class IngestionCoordinatorProtocol(Protocol):
    """
    Protocol for ingestion coordination operations.
    """

    def run_pre_ingestion(
        self,
        *,
        catalog_path: Path,
        scheduler_cfg: SchedulerConfig,
        options: PreIngestionOptions | None = None,
    ) -> None:
        """
        Run pre-ingestion tasks.

        Parameters
        ----------
        catalog_path : Path
            Path to catalog
        scheduler_cfg : SchedulerConfig
            Scheduler configuration
        options : PreIngestionOptions | None
            Pre-ingestion options

        """
        ...

    def backfill(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> BackfillWindowList:
        """
        Backfill market data for a single instrument.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Data schema
        instrument_id : str
            Instrument identifier
        lookback_days : int
            Days to backfill

        Returns
        -------
        BackfillWindowList
            Backfill results

        """
        ...

    def backfill_binding(
        self,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
    ) -> dict[str, BackfillWindowList]:
        """
        Backfill market data for a binding.

        Parameters
        ----------
        binding : ResolvedMarketBinding
            Market binding
        lookback_days : int
            Days to backfill

        Returns
        -------
        dict[str, BackfillWindowList]
            Backfill results by instrument

        """
        ...

    def backfill_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        policy: CoveragePolicy | None = None,
    ) -> list[tuple[int, int]]:
        """
        Backfill coverage gaps.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Data schema
        instrument_id : str
            Instrument identifier
        policy : CoveragePolicy | None
            Coverage policy

        Returns
        -------
        list[tuple[int, int]]
            Coverage gaps

        """
        ...


# ========================================================================
# IngestionCoordinator Implementation
# ========================================================================


class IngestionCoordinator:
    """
    Coordinates ingestion pipelines and backfill operations.

    Manages pre-ingestion tasks, backfill scheduling, auto-fill universe
    population, and integration with ingestion services.

    This component is extracted from the MLPipelineOrchestrator god class to
    provide focused, testable ingestion coordination functionality.

    """

    def __init__(
        self,
        *,
        coverage: CoverageProviderProtocol | None = None,
        writer: MarketDataWriterProtocol | None = None,
        registry: RegistryProtocol | None = None,
        ingestor: DatabentoIngestor | None = None,
        service: DatabentoIngestionService | None = None,
        raw_writer: RawIngestionWriterProtocol | None = None,
        domain_loader: DomainWindowLoaderProtocol | None = None,
        discovery_client: DiscoveryClient | None = None,
        write_mode_tokens: tuple[str, ...] = (),
        data_store: object | None = None,
        data_registry: object | None = None,
        macro_config: MacroIngestionConfig | None = None,
        earnings_config: EarningsCoordinatorConfig | None = None,
        message_bus: object | None = None,
    ) -> None:
        """
        Initialize ingestion coordinator.

        Parameters
        ----------
        coverage : CoverageProviderProtocol | None
            Coverage provider for gap analysis
        writer : MarketDataWriterProtocol | None
            Market data writer
        registry : RegistryProtocol | None
            Registry for dataset registration
        ingestor : DatabentoIngestor | None
            Direct ingestor for backfill operations
        service : DatabentoIngestionService | None
            Ingestion service for dataset operations
        raw_writer : RawIngestionWriterProtocol | None
            Raw data writer
        domain_loader : DomainWindowLoaderProtocol | None
            Domain window loader
        discovery_client : DiscoveryClient | None
            Discovery client for binding discovery
        write_mode_tokens : tuple[str, ...]
            Write mode tokens for storage decisions
        macro_config : MacroIngestionConfig | None
            Configuration for FRED/ALFRED macro data ingestion
        earnings_config : EarningsCoordinatorConfig | None
            Configuration for earnings data ingestion
        message_bus : object | None
            Message bus for event emission

        """
        self.coverage = coverage
        self.writer = writer
        self.registry = registry
        self.ingestor = ingestor
        self.service = service
        self.raw_writer = raw_writer
        self.domain_loader = domain_loader
        self.discovery_client = discovery_client
        self.write_mode_tokens = write_mode_tokens
        self._data_store = data_store
        self._data_registry = data_registry
        self._macro_config = macro_config or MacroIngestionConfig()
        self._earnings_config = earnings_config or EarningsCoordinatorConfig()
        self._message_bus = message_bus

        # Initialize IngestState for state management (from ml.data.ingest.resume)
        from ml.data.ingest.resume import IngestState

        self._ingest_state = IngestState()

        logger.debug("Initialized IngestionCoordinator")

    # -------------------------------------------------------------------------
    # Pre-ingestion
    # -------------------------------------------------------------------------

    def run_pre_ingestion(
        self,
        *,
        catalog_path: Path,
        scheduler_cfg: SchedulerConfig,
        options: PreIngestionOptions | None = None,
    ) -> None:
        """
        Run a data ingestion pre-stage using DataScheduler in orchestrator mode.

        Always cold path. Dual-write (SQL + Parquet) can be enabled in options to
        guarantee the dataset builder has catalog data while preserving SQL coverage.

        Parameters
        ----------
        catalog_path : Path
            Path to catalog
        scheduler_cfg : SchedulerConfig
            Scheduler configuration
        options : PreIngestionOptions | None
            Pre-ingestion options

        """
        from ml.data.scheduler import DataScheduler
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

        opts = options or PreIngestionOptions()
        catalog = ParquetDataCatalog(str(catalog_path))
        scheduler = DataScheduler(
            catalog=catalog,
            config=scheduler_cfg,
            use_orchestrator=opts.use_orchestrator,
            dual_write=opts.dual_write,
            dual_write_dataset_types=opts.dual_write_dataset_types(),
            start_metrics_server=opts.start_metrics_server,
            metrics_port=opts.metrics_port,
        )
        scheduler.run_daily_update()

    # -------------------------------------------------------------------------
    # Backfill operations
    # -------------------------------------------------------------------------

    def _create_ingestion_orchestrator(self) -> IngestionOrchestrator:
        """Create ingestion orchestrator instance."""
        if self.ingestor is None:
            raise RuntimeError("Ingestor is not configured for ingestion coordinator")
        if self.coverage is None:
            raise RuntimeError("Coverage provider is required for ingestion orchestration")
        if self.writer is None:
            raise RuntimeError("Market data writer is required for ingestion orchestration")
        if self.registry is None:
            raise RuntimeError("Data registry is required for ingestion orchestration")
        try:
            from ml.orchestration import pipeline_orchestrator as _pipeline

            orchestrator_cls = cast(
                type[IngestionOrchestrator],
                getattr(_pipeline, "IngestionOrchestrator"),
            )
        except Exception:  # pragma: no cover - defensive fallback
            orchestrator_cls = IngestionOrchestrator

        return orchestrator_cls(
            coverage=self.coverage,
            writer=self.writer,
            registry=self.registry,
            ingestor=self.ingestor,
            raw_writer=self.raw_writer,
            domain_loader=self.domain_loader,
            service=self.service,
        )

    def backfill(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> BackfillWindowList:
        """
        Backfill market data for a single instrument.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Data schema
        instrument_id : str
            Instrument identifier
        lookback_days : int
            Days to backfill

        Returns
        -------
        BackfillWindowList
            Backfill results

        """
        orchestrator = self._create_ingestion_orchestrator()
        return orchestrator.backfill_gaps(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            lookback_days=lookback_days,
        )

    def backfill_binding(
        self,
        *,
        binding: ResolvedMarketBinding,
        lookback_days: int,
    ) -> dict[str, BackfillWindowList]:
        """
        Backfill market data for a binding.

        Parameters
        ----------
        binding : ResolvedMarketBinding
            Market binding
        lookback_days : int
            Days to backfill

        Returns
        -------
        dict[str, BackfillWindowList]
            Backfill results by instrument

        """
        orchestrator = self._create_ingestion_orchestrator()
        return orchestrator.backfill_binding(
            binding=binding,
            lookback_days=lookback_days,
        )

    def backfill_coverage(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        policy: CoveragePolicy | None = None,
    ) -> list[tuple[int, int]]:
        """
        Backfill gaps bounded by subscription coverage policy.

        Determines the maximum lookback window from CoveragePolicy for the given dataset
        identifier and delegates to backfill.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Data schema
        instrument_id : str
            Instrument identifier
        policy : CoveragePolicy | None
            Coverage policy

        Returns
        -------
        list[tuple[int, int]]
            Coverage gaps

        """
        days = get_max_lookback_days(dataset_id, policy)
        return self.backfill(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            lookback_days=days,
        )

    # -------------------------------------------------------------------------
    # Auto-fill universe
    # -------------------------------------------------------------------------

    def auto_fill_universe(
        self,
        dataset_cfg: DatasetBuildConfig,
        auto_fill_cfg: AutoFillUniverseConfig,
        resolve_instrument_ids_fn: Any,
    ) -> None:
        """
        Auto-fill universe with market data.

        Parameters
        ----------
        dataset_cfg : DatasetBuildConfig
            Dataset configuration
        auto_fill_cfg : AutoFillUniverseConfig
            Auto-fill configuration
        resolve_instrument_ids_fn : Callable
            Function to resolve instrument IDs from config

        """
        if not auto_fill_cfg.enabled:
            return

        instruments = resolve_instrument_ids_fn(
            dataset_cfg,
            auto_fill_cfg.instrument_ids,
        )
        if not instruments:
            logger.info("Auto-fill universe requested but no instruments resolved; skipping")
            return

        metrics = _AutoFillMetrics.default()
        policy = CoveragePolicy.from_env()
        schema_aliases = {
            "bars": "ohlcv-1m",
            "tbbo": "tbbo",
            "trades": "trades",
        }
        processed_binding_keys: set[tuple[str, str]] = set()

        if auto_fill_cfg.include_bars:
            lookback = get_max_lookback_days("bars", policy)
            for instrument_id in instruments:
                self._auto_fill_schema(
                    dataset_id=auto_fill_cfg.dataset_id,
                    schema=schema_aliases.get("bars", "bars"),
                    instrument_id=instrument_id,
                    lookback_days=lookback,
                    metrics=metrics,
                    dataset_cfg=dataset_cfg,
                    processed_bindings=processed_binding_keys,
                )

        if auto_fill_cfg.include_tbbo:
            lookback = get_max_lookback_days("quotes", policy)
            for instrument_id in instruments:
                self._auto_fill_schema(
                    dataset_id=auto_fill_cfg.dataset_id,
                    schema=schema_aliases.get("tbbo", "tbbo"),
                    instrument_id=instrument_id,
                    lookback_days=lookback,
                    metrics=metrics,
                    dataset_cfg=dataset_cfg,
                    processed_bindings=processed_binding_keys,
                )

        if auto_fill_cfg.include_trades:
            lookback = get_max_lookback_days("trades", policy)
            for instrument_id in instruments:
                self._auto_fill_schema(
                    dataset_id=auto_fill_cfg.dataset_id,
                    schema=schema_aliases.get("trades", "trades"),
                    instrument_id=instrument_id,
                    lookback_days=lookback,
                    metrics=metrics,
                    dataset_cfg=dataset_cfg,
                    processed_bindings=processed_binding_keys,
                )

        if auto_fill_cfg.include_l2:
            self._auto_fill_l2(
                dataset_cfg=dataset_cfg,
                auto_fill_cfg=auto_fill_cfg,
                instruments=instruments,
                metrics=metrics,
                policy=policy,
            )

        if auto_fill_cfg.include_l3:
            self._auto_fill_l3(
                dataset_cfg=dataset_cfg,
                auto_fill_cfg=auto_fill_cfg,
                instruments=instruments,
                metrics=metrics,
                policy=policy,
            )

    def _auto_fill_schema(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
        metrics: _AutoFillMetrics,
        dataset_cfg: DatasetBuildConfig,
        processed_bindings: set[tuple[str, str]] | None = None,
    ) -> None:
        """
        Auto-fill a specific schema for an instrument.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Data schema
        instrument_id : str
            Instrument identifier
        lookback_days : int
            Days to backfill
        metrics : _AutoFillMetrics
            Metrics collector
        dataset_cfg : DatasetBuildConfig
            Dataset configuration
        processed_bindings : set[tuple[str, str]] | None
            Set of processed bindings to avoid duplicates

        """
        if lookback_days <= 0:
            logger.debug(
                "Auto-fill %s skipped for %s (non-positive lookback)",
                schema,
                instrument_id,
            )
            metrics.operations_total.labels(schema=schema, status="skipped").inc()
            return

        if self.ingestor is None and self.service is None:
            logger.info(
                "Auto-fill %s skipped for %s (ingestor/service unavailable)",
                schema,
                instrument_id,
            )
            metrics.operations_total.labels(schema=schema, status="skipped").inc()
            return

        binding_used: ResolvedMarketBinding | None = None
        effective_dataset_id = dataset_id
        effective_schema = schema
        binding_key: tuple[str, str] | None = None
        now_ns = int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)
        lookback_ns = max(int(lookback_days), 1) * DAY_NS

        # Check if we should use market bindings from config
        if dataset_cfg.market_inputs or dataset_cfg.market_dataset_id:
            base_symbol = instrument_id.split(".")[0].upper()
            resolved = IngestionOrchestrator.resolve_market_bindings(
                symbols=[base_symbol],
                instrument_ids=(instrument_id,),
                market_dataset_id=dataset_cfg.market_dataset_id,
                market_inputs=dataset_cfg.market_inputs,
            )
            if resolved:
                target_schema = (effective_schema or "").lower()
                matched_binding = next(
                    (
                        candidate
                        for candidate in resolved
                        if candidate.schema is not None
                        and candidate.schema.lower() == target_schema
                    ),
                    None,
                )
                if matched_binding is not None:
                    binding_used = matched_binding
                    effective_dataset_id = binding_used.dataset_id
                    if binding_used.schema:
                        effective_schema = binding_used.schema
                    if processed_bindings is not None:
                        binding_key = (binding_used.binding_id, effective_schema)
                        if binding_key in processed_bindings:
                            logger.debug(
                                "Auto-fill binding already processed; skipping instrument",
                                extra={
                                    "binding_id": binding_used.binding_id,
                                    "schema": effective_schema,
                                    "instrument_id": instrument_id,
                                },
                            )
                            metrics.operations_total.labels(
                                schema=effective_schema,
                                status="skipped",
                            ).inc()
                            return
                        processed_bindings.add(binding_key)
            elif dataset_cfg.market_inputs:
                logger.warning(
                    "No binding resolved for instrument %s; falling back to dataset %s",
                    instrument_id,
                    dataset_id,
                )

        start_time = time.perf_counter()
        status = "success"
        try:
            if binding_used is not None:

                def _run_binding(
                    binding_to_use: ResolvedMarketBinding,
                ) -> dict[str, BackfillWindowList]:
                    nonlocal effective_dataset_id
                    nonlocal effective_schema
                    effective_dataset_id = binding_to_use.dataset_id
                    if binding_to_use.schema:
                        effective_schema = binding_to_use.schema
                    dataset_type_local = self._map_schema_to_dataset_type(effective_schema)
                    storage_kind_local = binding_to_use.storage_kind or StorageKind.PARQUET
                    self._ensure_dataset_registered(
                        dataset_id=effective_dataset_id,
                        dataset_type=dataset_type_local,
                        location=dataset_cfg.data_dir,
                        storage_kind=storage_kind_local,
                    )
                    return self.backfill_binding(
                        binding=binding_to_use,
                        lookback_days=lookback_days,
                    )

                def _aggregate_binding_results(
                    results: dict[str, BackfillWindowList],
                ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]:
                    persisted: list[tuple[int, int]] = []
                    requested: list[tuple[int, int]] = []
                    frames_total = 0
                    rows_total = 0
                    for window_list in results.values():
                        persisted.extend(window_list)
                        requested.extend(window_list.requested_windows)
                        frames_total += window_list.frames_written
                        rows_total += window_list.rows_written
                    return persisted, requested, frames_total, rows_total

                binding_results = _run_binding(binding_used)
                (
                    persisted_windows,
                    requested_windows,
                    frames_total,
                    rows_total,
                ) = _aggregate_binding_results(binding_results)
                binding_result = binding_results.get(instrument_id)
                if binding_result is not None:
                    instrument_attempted = binding_result.attempted_window_count
                    instrument_rows = binding_result.rows_written
                else:
                    instrument_attempted = len(requested_windows)
                    instrument_rows = rows_total

                # Fallback to discovery if binding produced zero frames
                if (
                    instrument_attempted > 0
                    and instrument_rows == 0
                    and binding_used.source != "discovered"
                ):
                    base_symbol = instrument_id.split(".")[0].upper()
                    logger.warning(
                        "Binding produced zero frames; attempting discovery fallback",
                        extra={
                            "binding_id": binding_used.binding_id,
                            "dataset_id": effective_dataset_id,
                            "schema": effective_schema,
                            "instrument_id": instrument_id,
                        },
                    )
                    if self.discovery_client is not None:
                        fallback_binding = self.discovery_client.discover_binding_for_symbol(
                            symbol=base_symbol,
                            instrument_ids=(instrument_id,),
                            schema=effective_schema,
                            start_ns=now_ns - lookback_ns,
                            end_ns=now_ns,
                        )
                        if (
                            fallback_binding is not None
                            and fallback_binding.binding_id != binding_used.binding_id
                        ):
                            binding_used = fallback_binding
                            binding_results = _run_binding(binding_used)
                            (
                                persisted_windows,
                                requested_windows,
                                frames_total,
                                rows_total,
                            ) = _aggregate_binding_results(binding_results)
                            binding_result = binding_results.get(instrument_id)
                            if binding_result is not None:
                                instrument_attempted = binding_result.attempted_window_count
                                instrument_rows = binding_result.rows_written
                            else:
                                instrument_attempted = len(requested_windows)
                                instrument_rows = rows_total
                            if processed_bindings is not None:
                                processed_bindings.add((binding_used.binding_id, effective_schema))
                            logger.info(
                                "Discovery fallback applied | binding=%s dataset=%s schema=%s windows=%d",
                                binding_used.binding_id,
                                effective_dataset_id,
                                effective_schema,
                                len(persisted_windows),
                            )
                        else:
                            logger.warning(
                                "Discovery fallback unavailable or unchanged",
                                extra={
                                    "instrument_id": instrument_id,
                                    "dataset_id": effective_dataset_id,
                                    "schema": effective_schema,
                                    "attempted_windows": instrument_attempted,
                                },
                            )

                if binding_result is not None:
                    gaps: BackfillWindowList | list[tuple[int, int]] = binding_result
                else:
                    gaps = BackfillWindowList(
                        persisted=tuple(persisted_windows),
                        requested=tuple(requested_windows),
                        frames_written=frames_total,
                        rows_written=rows_total,
                    )
                logger.info(
                    "Auto-fill %s using binding %s | instrument=%s dataset=%s gaps=%d lookback_days=%d",
                    effective_schema,
                    binding_used.binding_id,
                    instrument_id,
                    effective_dataset_id,
                    len(gaps),
                    lookback_days,
                )
            else:
                # Direct backfill without binding
                dataset_type = self._map_schema_to_dataset_type(effective_schema)
                self._ensure_dataset_registered(
                    dataset_id=effective_dataset_id,
                    dataset_type=dataset_type,
                    location=dataset_cfg.data_dir,
                )
                gaps = self.backfill(
                    dataset_id=effective_dataset_id,
                    schema=effective_schema,
                    instrument_id=instrument_id,
                    lookback_days=lookback_days,
                )

            # Check for unresolved coverage gaps
            unresolved: list[tuple[int, int]] = []
            if gaps:
                unresolved = self._remaining_coverage_gaps(
                    dataset_id=effective_dataset_id,
                    schema=effective_schema,
                    instrument_id=instrument_id,
                    lookback_days=lookback_days,
                )

            if binding_used is None and not unresolved:
                logger.info(
                    "Auto-fill %s complete | instrument=%s dataset=%s gaps=%d lookback_days=%d",
                    effective_schema,
                    instrument_id,
                    effective_dataset_id,
                    len(gaps),
                    lookback_days,
                )

            if unresolved:
                status = "partial"
                logger.warning(
                    "Auto-fill %s completed with unresolved coverage gaps",
                    effective_schema,
                    extra={
                        "instrument_id": instrument_id,
                        "dataset_id": effective_dataset_id,
                        "gap_count": len(unresolved),
                        "lookback_days": lookback_days,
                    },
                )

        except Exception as exc:  # pragma: no cover - defensive guard
            status = "error"
            logger.error(
                "Auto-fill %s failed for %s (dataset=%s lookback=%s): %s",
                effective_schema,
                instrument_id,
                effective_dataset_id,
                lookback_days,
                exc,
                exc_info=True,
            )
            raise
        finally:
            metrics.operations_total.labels(schema=effective_schema, status=status).inc()
            metrics.latency_seconds.labels(schema=effective_schema).observe(
                max(time.perf_counter() - start_time, 0.0),
            )

    def _remaining_coverage_gaps(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> list[tuple[int, int]]:
        """
        Calculate remaining coverage gaps after backfill.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Data schema
        instrument_id : str
            Instrument identifier
        lookback_days : int
            Days to check

        Returns
        -------
        list[tuple[int, int]]
            List of coverage gaps as (start_ns, end_ns) tuples

        """
        if self.coverage is None:
            return []

        now_ns = int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)
        start_ns = now_ns - int(lookback_days) * DAY_NS
        covered = self.coverage.read_bucket_coverage(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            start_ns=start_ns,
            end_ns=now_ns,
        )
        start_bucket = start_ns // DAY_NS
        end_bucket_candidate = ((now_ns - DAY_NS) // DAY_NS) - 1
        end_bucket = (
            int(start_bucket) if end_bucket_candidate < start_bucket else int(end_bucket_candidate)
        )
        gaps: list[tuple[int, int]] = []
        for bucket in range(int(start_bucket), end_bucket + 1):
            if bucket not in covered:
                gaps.append((bucket * DAY_NS, (bucket + 1) * DAY_NS))
        return gaps

    def _auto_fill_l2(
        self,
        *,
        dataset_cfg: DatasetBuildConfig,
        auto_fill_cfg: AutoFillUniverseConfig,
        instruments: tuple[str, ...],
        metrics: _AutoFillMetrics,
        policy: CoveragePolicy,
    ) -> None:
        """
        Auto-fill L2 market data.

        Parameters
        ----------
        dataset_cfg : DatasetBuildConfig
            Dataset configuration
        auto_fill_cfg : AutoFillUniverseConfig
            Auto-fill configuration
        instruments : tuple[str, ...]
            Instrument identifiers
        metrics : _AutoFillMetrics
            Metrics collector
        policy : CoveragePolicy
            Coverage policy

        """
        data_dir = Path(dataset_cfg.data_dir).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        progress_path = (
            Path(auto_fill_cfg.l2_progress_file).expanduser()
            if auto_fill_cfg.l2_progress_file
            else data_dir / ".auto_fill_l2_progress.json"
        )
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        l2_days = auto_fill_cfg.l2_days
        if l2_days is None:
            l2_days = get_max_lookback_days("l2", policy)

        symbols_iter = [
            symbol.split(".")[0].upper() if symbol and "." in symbol else str(symbol).upper()
            for symbol in instruments
            if symbol
        ]
        symbol_roots = tuple(dict.fromkeys(symbols_iter))
        if not symbol_roots:
            logger.info("Auto-fill L2 skipped: no symbols resolved")
            metrics.operations_total.labels(schema="l2", status="skipped").inc()
            return

        start_time = time.perf_counter()
        status = "success"
        try:
            self._ensure_dataset_registered(
                dataset_id=auto_fill_cfg.l2_dataset_id,
                dataset_type=DatasetType.MBP1,
                location=dataset_cfg.data_dir,
            )
            config = PopulateL2TaskConfig(
                data_dir=data_dir,
                progress_file=progress_path,
                symbols=symbol_roots,
                tier=None,
                days=int(l2_days),
                dataset=auto_fill_cfg.l2_dataset_id,
                schema=auto_fill_cfg.l2_schema,
            )
            logger.info(
                "Auto-fill L2 start | symbols=%d days=%d dataset=%s schema=%s",
                len(symbol_roots),
                l2_days,
                auto_fill_cfg.l2_dataset_id,
                auto_fill_cfg.l2_schema,
            )
            populate_l2_efficient(config)
        except Exception as exc:  # pragma: no cover - defensive guard
            status = "error"
            logger.error("Auto-fill L2 failed: %s", exc, exc_info=True)
        finally:
            metrics.operations_total.labels(schema="l2", status=status).inc()
            metrics.latency_seconds.labels(schema="l2").observe(
                time.perf_counter() - start_time,
            )

    def _auto_fill_l3(
        self,
        *,
        dataset_cfg: DatasetBuildConfig,
        auto_fill_cfg: AutoFillUniverseConfig,
        instruments: tuple[str, ...],
        metrics: _AutoFillMetrics,
        policy: CoveragePolicy,
    ) -> None:
        """
        Auto-fill L3 market data.

        Parameters
        ----------
        dataset_cfg : DatasetBuildConfig
            Dataset configuration
        auto_fill_cfg : AutoFillUniverseConfig
            Auto-fill configuration
        instruments : tuple[str, ...]
            Instrument identifiers
        metrics : _AutoFillMetrics
            Metrics collector
        policy : CoveragePolicy
            Coverage policy

        """
        try:
            module = importlib.import_module("ml.tasks.ingest.l3")
        except Exception:  # pragma: no cover - optional dependency
            logger.info("Auto-fill L3 requested but task helpers are unavailable; skipping")
            metrics.operations_total.labels(schema="l3", status="skipped").inc()
            return

        PopulateL3TaskConfig = getattr(module, "PopulateL3TaskConfig", None)
        populate_l3_efficient = getattr(module, "populate_l3_efficient", None)
        if PopulateL3TaskConfig is None or populate_l3_efficient is None:
            logger.info("Auto-fill L3 requested but helpers missing from module; skipping")
            metrics.operations_total.labels(schema="l3", status="skipped").inc()
            return

        data_dir = Path(dataset_cfg.data_dir).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        progress_path = data_dir / ".auto_fill_l3_progress.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        l3_days = auto_fill_cfg.l3_days
        if l3_days is None:
            l3_days = get_max_lookback_days("l3", policy)

        symbols_iter = [
            symbol.split(".")[0].upper() if symbol and "." in symbol else str(symbol).upper()
            for symbol in instruments
            if symbol
        ]
        symbol_roots = tuple(dict.fromkeys(symbols_iter))
        if not symbol_roots:
            logger.info("Auto-fill L3 skipped: no symbols resolved")
            metrics.operations_total.labels(schema="l3", status="skipped").inc()
            return

        start_time = time.perf_counter()
        status = "success"
        try:
            dataset_identifier = auto_fill_cfg.l3_dataset_id or auto_fill_cfg.l2_dataset_id
            schema_identifier = auto_fill_cfg.l3_schema or "mbo"
            self._ensure_dataset_registered(
                dataset_id=dataset_identifier,
                dataset_type=self._map_schema_to_dataset_type(schema_identifier),
                location=dataset_cfg.data_dir,
            )
            dataset = auto_fill_cfg.l3_dataset_id or auto_fill_cfg.l2_dataset_id
            schema = auto_fill_cfg.l3_schema or "mbo"
            config = PopulateL3TaskConfig(
                data_dir=data_dir,
                progress_file=progress_path,
                symbols=symbol_roots,
                days=int(l3_days),
                dataset=dataset,
                schema=schema,
            )
            populate_l3_efficient(config)
        except Exception as exc:  # pragma: no cover - defensive guard
            status = "error"
            logger.error("Auto-fill L3 failed: %s", exc, exc_info=True)
        finally:
            metrics.operations_total.labels(schema="l3", status=status).inc()
            metrics.latency_seconds.labels(schema="l3").observe(
                time.perf_counter() - start_time,
            )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _map_schema_to_dataset_type(self, schema: str) -> DatasetType:
        """
        Map schema string to dataset type.

        Parameters
        ----------
        schema : str
            Schema identifier

        Returns
        -------
        DatasetType
            Dataset type

        """
        return map_schema_to_dataset_type(schema)

    def _ensure_dataset_registered(
        self,
        *,
        dataset_id: str,
        dataset_type: DatasetType,
        location: str,
        storage_kind: StorageKind | None = None,
    ) -> None:
        """
        Ensure dataset is registered in registry.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        dataset_type : DatasetType
            Dataset type
        location : str
            Dataset location
        storage_kind : StorageKind | None
            Storage kind

        """
        registry = self.registry
        if registry is None:
            return
        try:
            registry.get_manifest(dataset_id)
            return
        except Exception as exc:
            logger.debug(
                "Dataset manifest lookup failed during ingestion coordinator registration",
                exc_info=True,
                extra={"dataset_id": dataset_id, "reason": str(exc)},
            )

        # Determine storage_kind based on write_mode if not provided
        if storage_kind is None:
            if "sql" in self.write_mode_tokens:
                storage_kind = StorageKind.POSTGRES
            else:
                storage_kind = StorageKind.PARQUET

        manifest = build_auto_dataset_manifest(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            location=location,
            storage_kind=storage_kind,
            pipeline_signature="auto_fill_orchestrator",
            metadata={
                "auto_registered": True,
                "storage_path": str(Path(location).expanduser()),
            },
        )

        try:
            registry.register_dataset(manifest)
            try:
                registry.get_manifest(dataset_id)
            except Exception as verify_exc:  # pragma: no cover - defensive log
                logger.warning(
                    "Dataset registration verification failed",
                    extra={
                        "dataset_id": dataset_id,
                        "dataset_type": dataset_type.value,
                        "reason": str(verify_exc),
                    },
                )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug(
                "Dataset registration skipped",
                exc_info=True,
                extra={
                    "dataset_id": dataset_id,
                    "dataset_type": dataset_type.value,
                    "reason": str(exc),
                },
            )

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def coordinate_ingestion(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_ids: list[str],
        lookback_days: int,
        policy: CoveragePolicy | None = None,
    ) -> dict[str, int | str]:
        if not instrument_ids:
            return {
                "rows_written": 0,
                "fallback_level": "dummy",
                "error": "No instrument_ids provided",
            }

        effective_lookback = get_max_lookback_days(dataset_id, policy)
        if effective_lookback <= 0:
            effective_lookback = lookback_days

        try:
            rows_written = 0
            for instrument_id in instrument_ids:
                result = self.backfill(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    lookback_days=effective_lookback,
                )
                rows_written += result.rows_written
            return {"rows_written": rows_written, "fallback_level": "primary"}
        except Exception:
            logger.debug("Primary ingestion failed; attempting fallback", exc_info=True)
            return self._handle_ingestion_fallback(
                dataset_id=dataset_id,
                schema=schema,
                instrument_ids=instrument_ids,
                lookback_days=effective_lookback,
                level="cached",
            )

    def ingest_from_databento(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> BackfillWindowList:
        return self.backfill(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            lookback_days=lookback_days,
        )

    def ingest_from_yahoo(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> int:
        return 0

    def ingest_from_fred(
        self,
        *,
        series_ids: list[str],
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Refresh FRED/ALFRED macro data if stale.

        Uses the configured macro paths and staleness settings to ensure
        macro indicator data is fresh. The start_date and end_date parameters
        are accepted for API compatibility but the refresh is staleness-based.

        Parameters
        ----------
        series_ids : list[str]
            FRED series identifiers to refresh (e.g., ["DGS10", "FEDFUNDS"]).
        start_date : str
            Start date (for API compatibility; refresh is staleness-based).
        end_date : str
            End date (for API compatibility; refresh is staleness-based).

        Returns
        -------
        int
            1 if any data was refreshed, 0 otherwise.

        """
        # Import here to avoid circular imports and keep cold path
        from datetime import timedelta

        from ml.data.ingest.macro_refresh import MacroRefreshResult
        from ml.data.ingest.macro_refresh import ensure_macro_ready

        # Log the request (start_date/end_date for observability, not used for staleness)
        logger.info(
            "Refreshing FRED/ALFRED macro data",
            extra={
                "series_ids": series_ids,
                "start_date": start_date,
                "end_date": end_date,
                "fred_path": self._macro_config.fred_path,
                "vintage_dir": self._macro_config.vintage_dir,
                "max_staleness_hours": self._macro_config.max_staleness_hours,
            },
        )

        try:
            # Merge series_ids from method call with config (method takes precedence)
            effective_series: tuple[str, ...] | None = None
            if series_ids:
                effective_series = tuple(series_ids)
            elif self._macro_config.series_ids:
                effective_series = self._macro_config.series_ids

            # Call the real implementation
            result: MacroRefreshResult = ensure_macro_ready(
                fred_path=Path(self._macro_config.fred_path),
                vintage_dir=(
                    Path(self._macro_config.vintage_dir)
                    if self._macro_config.vintage_dir
                    else None
                ),
                max_age=timedelta(hours=self._macro_config.max_staleness_hours),
                series_ids=effective_series,
            )

            # Log result
            if result.fred_error or result.alfred_error:
                logger.warning(
                    "Macro refresh completed with errors",
                    extra={
                        "fred_refreshed": result.fred_refreshed,
                        "alfred_refreshed": result.alfred_refreshed,
                        "fred_error": str(result.fred_error) if result.fred_error else None,
                        "alfred_error": str(result.alfred_error) if result.alfred_error else None,
                    },
                )
            else:
                logger.info(
                    "Macro refresh completed",
                    extra={
                        "fred_refreshed": result.fred_refreshed,
                        "alfred_refreshed": result.alfred_refreshed,
                    },
                )

            # Return 1 if any refresh happened
            return 1 if (result.fred_refreshed or result.alfred_refreshed) else 0

        except Exception:
            logger.warning("FRED/ALFRED macro refresh failed", exc_info=True)
            return 0

    def ingest_earnings_data(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Ingest earnings data for a single symbol.

        Delegates to EarningsIngestionService with the DataStore as writer.
        The service fetches earnings actuals from SEC EDGAR and consensus
        estimates from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Stock ticker symbol (e.g., 'AAPL')
        start_date : str
            Start date (currently unused - service uses edgar_quarters config)
        end_date : str
            End date (currently unused - service uses edgar_quarters config)

        Returns
        -------
        int
            Total count of actuals + estimates written

        """
        # Note: start_date and end_date are kept for API compatibility but
        # the EarningsIngestionService uses edgar_quarters for date range
        del start_date, end_date

        if self._data_store is None:
            logger.warning(
                "No DataStore available for earnings ingestion",
                extra={"symbol": symbol},
            )
            return 0

        try:
            from ml.config.earnings_ingestion import DEFAULT_SKIP_ACTUALS_TICKERS
            from ml.config.earnings_ingestion import EarningsIngestionConfig
            from ml.features.earnings.ingestion.service import EarningsIngestionResult
            from ml.features.earnings.ingestion.service import EarningsIngestionService

            # Build skip set: default ETFs + any custom skips from coordinator config
            skip_set: tuple[str, ...] = DEFAULT_SKIP_ACTUALS_TICKERS
            if self._earnings_config.skip_tickers:
                combined = set(skip_set) | set(self._earnings_config.skip_tickers)
                skip_set = tuple(sorted(combined))

            # Use empty DSN since we're providing our own writer (DataStore).
            # The universe resolver will use override_symbols directly.
            config = EarningsIngestionConfig(
                postgres_dsn="",  # Not used when override_symbols is set
                override_symbols=(symbol.upper(),),
                skip_actuals=skip_set,
                edgar_quarters=self._earnings_config.edgar_quarters,
                enable_yahoo=self._earnings_config.enable_yahoo,
                edgar_rate_limit=self._earnings_config.edgar_rate_limit,
                yahoo_rate_limit=self._earnings_config.yahoo_rate_limit,
                sec_identity=self._earnings_config.sec_identity,
            )

            # DataStore implements the writer protocol expected by EarningsIngestionService
            service = EarningsIngestionService(
                config=config,
                writer=self._data_store,  # type: ignore[arg-type]
            )

            result: EarningsIngestionResult = service.run()

            logger.info(
                "Earnings ingestion completed",
                extra={
                    "symbol": symbol,
                    "actuals_written": result.actuals_written,
                    "estimates_written": result.estimates_written,
                    "duration_seconds": result.duration_seconds,
                    "failures": result.failures,
                },
            )

            return result.actuals_written + result.estimates_written

        except Exception:
            logger.warning(
                "Earnings ingestion failed",
                extra={"symbol": symbol},
                exc_info=True,
            )
            return 0

    def _handle_ingestion_fallback(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_ids: list[str],
        lookback_days: int,
        level: str,
    ) -> dict[str, int | str]:
        """
        Handle ingestion fallback using Pattern 4 (PRIMARY → CACHED → FILE → DUMMY).

        Emits fallback activation metrics and attempts recovery at the specified level.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Data schema
        instrument_ids : list[str]
            List of instrument identifiers
        lookback_days : int
            Number of days to look back
        level : str
            Fallback level ('primary', 'cached', 'file', 'dummy')

        Returns
        -------
        dict[str, int | str]
            Result dict with rows_written, fallback_level, and optional error

        """
        # Emit fallback activation metric
        try:
            from ml.common.metrics_bootstrap import get_counter

            fallback_counter = get_counter(
                "ml_fallback_activations_total",
                "Total fallback activations by level and component",
                ["level", "component"],
            )
            fallback_counter.labels(level=level, component="ingestion").inc()
        except Exception:
            # Pattern 4: Never raise from metrics
            logger.debug("Failed to emit fallback metric", exc_info=True)

        # Log fallback activation
        logger.info(
            "Activating ingestion fallback",
            extra={
                "level": level,
                "dataset_id": dataset_id,
                "instrument_count": len(instrument_ids),
                "lookback_days": lookback_days,
            },
        )

        # Attempt fallback based on level
        fallback_levels = ["primary", "cached", "file", "dummy"]
        try:
            current_idx = fallback_levels.index(level)
        except ValueError:
            current_idx = len(fallback_levels) - 1  # Default to dummy

        rows_written = 0
        error_msg: str | None = None

        # PRIMARY level: Try component backfill
        if current_idx == 0:
            try:
                for instrument_id in instrument_ids:
                    result = self.backfill(
                        dataset_id=dataset_id,
                        schema=schema,
                        instrument_id=instrument_id,
                        lookback_days=lookback_days,
                    )
                    rows_written += result.rows_written
                return {"rows_written": rows_written, "fallback_level": "primary"}
            except Exception as exc:
                error_msg = str(exc)
                logger.debug("PRIMARY fallback failed", exc_info=True)

        # CACHED level: Try component backfill_coverage
        if current_idx <= 1:
            try:
                for instrument_id in instrument_ids:
                    windows = self.backfill_coverage(
                        dataset_id=dataset_id,
                        schema=schema,
                        instrument_id=instrument_id,
                    )
                    rows_written += len(windows)
                if rows_written > 0:
                    return {"rows_written": rows_written, "fallback_level": "cached"}
            except Exception as exc:
                error_msg = str(exc)
                logger.debug("CACHED fallback failed", exc_info=True)

        # FILE level: Try local file lookup
        if current_idx <= 2:
            try:
                # Attempt to find local data files
                from pathlib import Path

                data_dir = Path("data/tier1")
                if data_dir.exists():
                    for instrument_id in instrument_ids:
                        symbol = instrument_id.split(".")[0] if "." in instrument_id else instrument_id
                        parquet_files = list(data_dir.glob(f"*{symbol}*.parquet"))
                        if parquet_files:
                            rows_written += 1  # Found local data
                    if rows_written > 0:
                        return {"rows_written": rows_written, "fallback_level": "file"}
            except Exception as exc:
                error_msg = str(exc)
                logger.debug("FILE fallback failed", exc_info=True)

        # DUMMY level: Return safe default
        return {
            "rows_written": 0,
            "fallback_level": "dummy",
            "error": error_msg or "Fallback activated - no data available",
        }

    def _create_ingestion_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        rows_written: int,
        current_instrument_index: int,
        progress: float,
    ) -> None:
        checkpoint_data = {
            "rows_written": rows_written,
            "current_instrument_index": current_instrument_index,
            "progress": progress,
        }
        try:
            checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
        except Exception:
            logger.debug("Failed to write ingestion checkpoint", exc_info=True)

    def _restore_from_checkpoint(
        self,
        *,
        checkpoint_path: Path,
    ) -> dict[str, int | float]:
        if not checkpoint_path.exists():
            return {"rows_written": 0, "current_instrument_index": 0, "progress": 0.0}
        try:
            return cast(
                dict[str, int | float],
                json.loads(checkpoint_path.read_text()),
            )
        except Exception:
            logger.debug("Failed to restore ingestion checkpoint", exc_info=True)
            return {"rows_written": 0, "current_instrument_index": 0, "progress": 0.0}

    def _validate_ingestion_data(
        self,
        *,
        data: object,
        instrument_id: str,
    ) -> tuple[bool, list[str]]:
        """
        Validate ingestion data using schema validators.

        Wired to ml.stores.common.schema_validator.SchemaValidatorComponent for validation.

        Parameters
        ----------
        data : object
            Data to validate (DataFrame, dict, or other)
        instrument_id : str
            Instrument identifier for context

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, error_messages) tuple

        """
        errors: list[str] = []

        # Handle None data
        if data is None:
            return False, ["Data is None"]

        # Helper to check if data is empty without triggering DataFrame ambiguity
        def _is_empty(obj: object) -> bool:
            """Check if data is empty without triggering DataFrame ValueError."""
            if obj is None:
                return True
            if hasattr(obj, "__len__"):
                try:
                    return len(obj) == 0
                except (TypeError, ValueError):
                    return False
            return False

        # Validate using SchemaValidatorComponent if available
        try:
            from ml.stores.common.schema_validator import SchemaValidatorComponent

            # SchemaValidatorComponent requires data_registry
            # Use type guard for mypy compatibility
            if self._data_registry is None:
                # Fall through to basic validation
                raise ImportError("No data_registry available")
            validator = SchemaValidatorComponent(
                data_registry=cast(RegistryProtocol, self._data_registry),
            )

            # Check for required fields based on data type
            if hasattr(data, "columns"):
                # DataFrame-like: check for required columns
                columns = set(getattr(data, "columns", []))
                required = {"ts_event", "instrument_id"}
                missing = required - columns

                if missing:
                    errors.append(f"Missing required columns: {sorted(missing)}")

                # Check for ts_event validity if present
                if "ts_event" in columns:
                    ts_col = data["ts_event"]  # type: ignore[index]
                    if hasattr(ts_col, "isna") and ts_col.isna().any():
                        errors.append("ts_event contains null values")
                    if hasattr(ts_col, "min") and hasattr(ts_col, "max"):
                        min_ts = ts_col.min()
                        max_ts = ts_col.max()
                        if min_ts > max_ts:
                            errors.append("ts_event values are not monotonic")

                # Use preflight_check with correct signature: (dataset_id, data, strict)
                # Build dataset_id from instrument_id
                preflight_dataset_id = f"ingestion.{instrument_id.replace('.', '_')}"
                try:
                    # Cast data to DataFrameLike for type safety
                    from ml.ml_types import DataFrameLike

                    success, error_msg, _details = validator.preflight_check(
                        dataset_id=preflight_dataset_id,
                        data=cast(DataFrameLike, data),
                        strict=False,  # Allow subset for flexible validation
                    )
                    if not success and error_msg:
                        errors.append(f"Preflight check failed: {error_msg}")
                except Exception:
                    logger.debug(
                        "Preflight check raised exception",
                        exc_info=True,
                        extra={"instrument_id": instrument_id, "dataset_id": preflight_dataset_id},
                    )
                    # Pattern 4: Log but don't fail on preflight errors
                    # The basic column checks above are sufficient fallback

            elif isinstance(data, dict):
                # Dict-like: check for required keys
                required_keys = {"ts_event", "instrument_id"}
                missing_keys = required_keys - set(data.keys())
                if missing_keys:
                    errors.append(f"Missing required keys: {sorted(missing_keys)}")

                # Check ts_event validity
                ts_event = data.get("ts_event")
                if ts_event is not None:
                    if isinstance(ts_event, list):
                        if any(v is None for v in ts_event):
                            errors.append("ts_event contains null values")
                    elif ts_event is None:
                        errors.append("ts_event is null")

            else:
                # Unknown data type: basic validation using safe empty check
                if _is_empty(data):
                    errors.append("Data is empty")

        except ImportError:
            # SchemaValidatorComponent not available, use basic validation
            logger.debug("SchemaValidatorComponent not available; using basic validation")
            if _is_empty(data):
                errors.append("Data is empty")
        except Exception:
            # Pattern 4: Log and continue with basic validation
            logger.debug(
                "Schema validation error",
                exc_info=True,
                extra={"instrument_id": instrument_id},
            )
            if _is_empty(data):
                errors.append("Data is empty")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _emit_ingestion_event(
        self,
        *,
        event_type: str,
        dataset_id: str,
        rows_written: int,
        instrument_id: str | None = None,
        status: str = "success",
    ) -> None:
        """
        Emit ingestion event to message bus using build_topic_for_stage.

        Wired to ml.common.message_topics.build_topic_for_stage() for topic generation
        and ml.config.events.Stage for event type mapping.

        Parameters
        ----------
        event_type : str
            Event type string (mapped to Stage enum)
        dataset_id : str
            Dataset identifier
        rows_written : int
            Number of rows written
        instrument_id : str | None
            Optional instrument identifier
        status : str
            Event status ('success', 'failed', 'partial')

        """
        if self._message_bus is None:
            logger.debug(
                "No message bus configured; skipping event emission",
                extra={"event_type": event_type, "dataset_id": dataset_id},
            )
            return

        try:
            from ml.common.message_topics import build_topic_for_stage
            from ml.config.bus import MessageBusConfig
            from ml.config.events import EventStatus
            from ml.config.events import Stage

            # Map event_type string to Stage enum
            # Use DATASET_PLANNED for start, DATA_INGESTED for completion
            stage_map: dict[str, Stage] = {
                "ingestion_started": Stage.DATASET_PLANNED,
                "ingestion_completed": Stage.DATA_INGESTED,
                "catalog_written": Stage.CATALOG_WRITTEN,
                "feature_computed": Stage.FEATURE_COMPUTED,
            }
            stage = stage_map.get(event_type, Stage.DATA_INGESTED)

            # Map status string to EventStatus enum
            status_map: dict[str, EventStatus] = {
                "success": EventStatus.SUCCESS,
                "failed": EventStatus.FAILED,
                "partial": EventStatus.PARTIAL,
            }
            event_status = status_map.get(status, EventStatus.SUCCESS)

            # Get topic config from environment (scheme, prefix)
            bus_config = MessageBusConfig.from_env()

            # Build topic using centralized topic builder with config-driven scheme/prefix
            topic_instrument = instrument_id or "SYSTEM"
            topic = build_topic_for_stage(
                stage,
                topic_instrument,
                scheme=bus_config.scheme,
                prefix=bus_config.topic_prefix,
            )

            # Build event payload
            from datetime import UTC
            from datetime import datetime

            payload: dict[str, object] = {
                "event_type": event_type,
                "stage": stage.value,
                "status": event_status.value,
                "dataset_id": dataset_id,
                "rows_written": rows_written,
                "ts_event": int(datetime.now(tz=UTC).timestamp() * 1_000_000_000),
            }
            if instrument_id is not None:
                payload["instrument_id"] = instrument_id

            # Publish to message bus (duck-typed)
            publish = getattr(self._message_bus, "publish", None)
            if publish is not None and callable(publish):
                publish(topic, payload)
                logger.debug(
                    "Emitted ingestion event",
                    extra={"topic": topic, "event_type": event_type, "rows_written": rows_written},
                )
            else:
                logger.debug(
                    "Message bus has no publish method",
                    extra={"event_type": event_type},
                )
        except Exception:
            # Pattern 4: Never raise from event emission
            logger.debug(
                "Failed to emit ingestion event",
                exc_info=True,
                extra={"event_type": event_type, "dataset_id": dataset_id},
            )

    def _get_ingestion_state(self) -> dict[str, object]:
        """
        Get current ingestion state from IngestState.

        Returns dict with last timestamp per instrument for resume support.

        Returns
        -------
        dict[str, object]
            State dict with 'last_ts_ns_by_instrument' mapping

        """
        return {
            "last_ts_ns_by_instrument": dict(self._ingest_state.last_ts_ns_by_instrument),
        }

    def _update_ingestion_state(
        self,
        *,
        rows_written: int,
        current_instrument: str,
        ts_ns: int | None = None,
    ) -> None:
        """
        Update ingestion state for resume support.

        Wired to ml.data.ingest.resume.IngestState for consistent state tracking.

        Parameters
        ----------
        rows_written : int
            Number of rows written (logged for observability)
        current_instrument : str
            Current instrument being processed
        ts_ns : int | None
            Latest timestamp in nanoseconds for this instrument

        """
        if ts_ns is not None:
            self._ingest_state.update_last_ts(current_instrument, ts_ns)
            logger.debug(
                "Updated ingestion state",
                extra={
                    "instrument": current_instrument,
                    "ts_ns": ts_ns,
                    "rows_written": rows_written,
                },
            )
        else:
            logger.debug(
                "Ingestion state update skipped (no ts_ns)",
                extra={"instrument": current_instrument, "rows_written": rows_written},
            )
