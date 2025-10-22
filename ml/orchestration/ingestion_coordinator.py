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
from ml.orchestration.config_types import PreIngestionOptions
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind


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
        normalized = schema.lower()
        if normalized.startswith("ohlcv"):
            return DatasetType.BARS
        if normalized in {"tbbo", "bbo-1s", "bbo-1m", "tcbbo"}:
            return DatasetType.TBBO
        if normalized.startswith("trade"):
            return DatasetType.TRADES
        if normalized.startswith(("mbp", "mbo")):
            return DatasetType.MBP1
        return DatasetType.BARS

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
