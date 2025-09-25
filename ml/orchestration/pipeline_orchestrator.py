#!/usr/bin/env python3

"""
High-level ML pipeline orchestrator (cold path only).

Composes existing ingestion, dataset build, HPO, and training CLIs into a typed,
testable interface suitable for a single long-running service or a nightly batch job.
All heavy work (DataFrames, file I/O, GPU training) remains strictly off the actor hot
paths.

"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import time
import uuid as _uuid
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ml.common.logging_config import bind_log_context
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.coverage import CoveragePolicy
from ml.config.coverage import get_max_lookback_days
from ml.data import DatasetMetadata
from ml.data import DatasetMetadataExpectations
from ml.data import DatasetValidationConfig
from ml.data import compute_dataset_pipeline_signature
from ml.data import load_dataset_metadata
from ml.data import validate_dataset_metadata_expectations
from ml.data.dataset_manifest_defaults import build_auto_dataset_manifest
from ml.data.ingest.databento_adapter import DatabentoAPIClient
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.vintage import VintagePolicy
from ml.data.vintage import format_dt
from ml.data.vintage import parse_dt
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.registry.protocols import RegistryProtocol
from ml.stores.io_raw import ParquetCatalogRawWriter
from ml.stores.io_raw import RawIngestionWriterProtocol
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import DataStoreFacadeProtocol
from ml.stores.protocols import MarketDataWriterProtocol
from ml.stores.providers import DAY_NS
from ml.stores.providers import CatalogCoverageProvider
from ml.stores.providers import SqlCoverageProvider
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.writers import FanoutMarketDataWriter
from ml.stores.writers import ParquetCatalogMarketDataWriter
from ml.tasks.ingest import PopulateL2TaskConfig
from ml.tasks.ingest import populate_l2_efficient


logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ml.config.scheduler_config import SchedulerConfig


class _CliMain(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


class IntegrationManagerProtocol(Protocol):
    data_registry: object | None
    feature_registry: object | None
    model_registry: object | None
    strategy_registry: object | None
    data_store: object | None
    feature_store: object | None
    model_store: object | None
    strategy_store: object | None
    partition_manager: object | None


@dataclass(slots=True)
class BuildArtifacts:
    out_dir: Path
    feature_registry_dir: str | None
    feature_set_id: str | None
    feature_names: tuple[str, ...] = ()
    dataset_metadata: DatasetMetadata | None = None


class _EmptyDatasetError(RuntimeError):
    """Raised when the dataset build produces zero rows."""

    def __init__(self, message: str, *, row_count: int | None = None) -> None:
        super().__init__(message)
        self.row_count = row_count


@dataclass(slots=True, frozen=True)
class DatasetBuildConfig:
    data_dir: str
    symbols: str
    out_dir: str
    dataset_id: str = "tft_dataset"
    market_dataset_id: str | None = None
    instrument_ids: tuple[str, ...] | None = None
    include_macro: bool = False
    macro_lag_days: int = 1
    include_micro: bool = False
    include_l2: bool = False
    include_events: bool = False
    include_calendar: bool = False
    fred_vintage_dir: str | None = None
    events_dir: str | None = None
    student_mode: bool = False
    horizon_minutes: int = 15
    threshold: float = 0.001
    lookback_periods: int = 30
    emit_dataset_events: bool = False
    # Optional time window and chunking for memory/perf control
    start_iso: str | None = None
    end_iso: str | None = None
    chunk_days: int = 0
    # Optional feature registration
    register_features: bool = False
    feature_registry_dir: str | None = None
    feature_role: str = "teacher"
    auto_refresh_macro: bool = True
    macro_staleness_hours: int = 24
    macro_series_ids: tuple[str, ...] | None = None
    macro_fred_path: str | None = None
    validation: DatasetValidationConfig | None = None
    vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME
    vintage_as_of: str | None = None


@dataclass(slots=True, frozen=True)
class AutoFillUniverseConfig:
    enabled: bool = False
    dataset_id: str = "EQUS.MINI"
    include_bars: bool = True
    include_tbbo: bool = True
    include_trades: bool = True
    include_l2: bool = False
    include_l3: bool = False
    l2_dataset_id: str = "DBEQ.BASIC"
    l2_schema: str = "mbp-10"
    l2_days: int | None = None
    l2_progress_file: str | None = None
    disable_dataset_l2_ingest: bool = True
    instrument_ids: tuple[str, ...] | None = None
    l3_dataset_id: str | None = None
    l3_schema: str | None = None
    l3_days: int | None = None


@dataclass(slots=True, frozen=True)
class HPOConfig:
    enabled: bool = False
    epochs: int = 2
    batch_size: int = 32
    tail_rows: int = 5000
    limit_groups: int = 50
    workers: int = 2
    backend: str = "optuna"
    metric: str = "prx"
    direction: str | None = None
    optuna_trials: int = 20
    optuna_timeout: int | None = None
    loss: str = "bce"
    pos_weight: str = "auto"


@dataclass(slots=True, frozen=True)
class TeacherTrainConfig:
    enabled: bool = True
    model_id: str = "teacher_model"
    feature_registry_dir: str | None = None
    feature_set_id: str | None = None
    max_epochs: int = 5


@dataclass(slots=True, frozen=True)
class StudentDistillConfig:
    enabled: bool = False
    model_id: str = "student_model"
    parent_model_id: str | None = None
    model_registry_dir: str | None = None
    feature_registry_dir: str | None = None
    feature_set_id: str | None = None
    objective: str = "logit_mse"
    kd_lambda: float = 0.5
    early_stopping: int = 200
    opset: int | None = None
    use_val_for_distill: bool = False


@dataclass(slots=True, frozen=True)
class IntegrationConfig:
    enabled: bool = False
    db_connection: str | None = None
    auto_start_postgres: bool = False
    auto_migrate: bool = False
    ensure_healthy: bool = True
    strict_protocol_validation: bool | None = None
    run_validators: bool = True


@dataclass(slots=True, frozen=True)
class _AutoFillMetrics:
    operations_total: Any
    latency_seconds: Any

    @staticmethod
    def default() -> _AutoFillMetrics:
        return _AutoFillMetrics(
            operations_total=get_counter(
                "nautilus_ml_auto_fill_operations_total",
                "Auto-fill ingestion operations",
                ("schema", "status"),
            ),
            latency_seconds=get_histogram(
                "nautilus_ml_auto_fill_latency_seconds",
                "Auto-fill ingestion latency",
                ("schema",),
                buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            ),
        )

@dataclass(slots=True, frozen=True)
class OrchestratorConfig:
    dataset: DatasetBuildConfig
    hpo: HPOConfig
    teacher: TeacherTrainConfig
    student: StudentDistillConfig = StudentDistillConfig()
    # Optional promotions/feature refresh settings (used by config-driven scheduler)
    promotions: PromotionsConfig | None = None
    # Optional data ingestion pre-stage before dataset build
    pre_ingestion: SchedulerConfig | None = None
    pre_ingestion_options: PreIngestionOptions | None = None
    auto_fill: AutoFillUniverseConfig | None = None
    integration: IntegrationConfig | None = None


@dataclass(slots=True, frozen=True)
class PromotionsConfig:
    # Model promotions
    auto_register_model: bool = False
    gates_json: str | None = None
    auto_promote: bool = False
    deploy_target: str | None = None
    # Feature registration/refresh
    auto_register_features: bool = False
    feature_metrics_json: str | None = None
    refresh_features: bool = False


@dataclass(slots=True)
class MLPipelineOrchestrator:
    coverage: CoverageProviderProtocol
    writer: MarketDataWriterProtocol
    build_main: _CliMain
    teacher_main: _CliMain
    registry: object | None = None  # Backwards compatibility alias
    data_registry: object | None = (
        None  # RegistryProtocol at runtime; kept lax to avoid import cycles
    )
    ingestor: object | None = None  # DatabentoIngestor or similar
    hpo_main: _CliMain | None = None
    raw_writer: RawIngestionWriterProtocol | None = None
    service: DatabentoIngestionService | None = None
    model_registry: object | None = None
    feature_registry: object | None = None
    strategy_registry: object | None = None
    feature_store: object | None = None
    model_store: object | None = None
    strategy_store: object | None = None
    data_store: object | None = None
    partition_manager: object | None = None
    integration_manager_factory: Callable[..., IntegrationManagerProtocol] | None = None
    _integration_manager: IntegrationManagerProtocol | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _build_artifacts: BuildArtifacts | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.data_registry is None and self.registry is not None:
            object.__setattr__(self, "data_registry", self.registry)

    @staticmethod
    def _infer_dataset_row_count(result: object) -> int | None:
        """Best-effort row count inference for API build results."""
        metadata = getattr(result, "metadata", None)
        if metadata is not None:
            overall_window = getattr(metadata, "overall_window", None)
            ts_start = getattr(metadata, "ts_event_start", None)
            ts_end = getattr(metadata, "ts_event_end", None)
            if overall_window is None and ts_start is None and ts_end is None:
                return 0

        dataset_parquet = getattr(result, "dataset_parquet", None)
        if isinstance(dataset_parquet, Path) and dataset_parquet.exists():
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
                logger.debug(
                    "pyarrow unavailable for row count inference",
                    extra={"dataset_parquet": str(dataset_parquet)},
                )
            else:
                try:
                    return int(pq.ParquetFile(str(dataset_parquet)).metadata.num_rows)
                except Exception:  # pragma: no cover - defensive best effort
                    logger.debug(
                        "Unable to infer row count from dataset parquet",
                        exc_info=True,
                        extra={"dataset_parquet": str(dataset_parquet)},
                    )
            # fall through when inference fails
        dataset_csv = getattr(result, "dataset_csv", None)
        if isinstance(dataset_csv, Path) and dataset_csv.exists():
            try:
                with dataset_csv.open("r", encoding="utf-8") as handle:
                    next(handle, None)  # header (if any)
                    has_data = next(handle, None)
                return 0 if has_data is None else None
            except Exception:  # pragma: no cover - defensive best effort
                logger.debug(
                    "Unable to infer row count from dataset CSV",
                    exc_info=True,
                    extra={"dataset_csv": str(dataset_csv)},
                )

        return None

    @staticmethod
    def _resolve_instrument_ids(
        dataset_cfg: DatasetBuildConfig,
        override: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        if override:
            return tuple(item.strip() for item in override if item.strip())
        if dataset_cfg.instrument_ids:
            return tuple(item.strip() for item in dataset_cfg.instrument_ids if item.strip())
        symbols_raw = dataset_cfg.symbols.split(",")
        return tuple(item.strip().upper() for item in symbols_raw if item.strip())

    def _auto_fill_universe(
        self,
        dataset_cfg: DatasetBuildConfig,
        auto_fill_cfg: AutoFillUniverseConfig,
    ) -> None:
        if not auto_fill_cfg.enabled:
            return

        instruments = self._resolve_instrument_ids(
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
    ) -> None:
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

        start_time = time.perf_counter()
        status = "success"
        try:
            dataset_type = self._map_schema_to_dataset_type(schema)
            self._ensure_dataset_registered(
                dataset_id=dataset_id,
                dataset_type=dataset_type,
                location=dataset_cfg.data_dir,
            )
            gaps = self.backfill(
                dataset_id=dataset_id,
                schema=schema,
                instrument_id=instrument_id,
                lookback_days=lookback_days,
            )
            if gaps:
                unresolved = self._remaining_coverage_gaps(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    lookback_days=lookback_days,
                )
                if unresolved:
                    status = "error"
                    raise RuntimeError(
                        "Auto-fill completed with unresolved coverage gaps "
                        f"(dataset={dataset_id} schema={schema} instrument={instrument_id} gaps={len(unresolved)})",
                    )
            logger.info(
                "Auto-fill %s complete | instrument=%s dataset=%s gaps=%d lookback_days=%d",
                schema,
                instrument_id,
                dataset_id,
                len(gaps),
                lookback_days,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            status = "error"
            logger.error(
                "Auto-fill %s failed for %s (dataset=%s lookback=%s): %s",
                schema,
                instrument_id,
                dataset_id,
                lookback_days,
                exc,
                exc_info=True,
            )
            raise
        finally:
            metrics.operations_total.labels(schema=schema, status=status).inc()
            metrics.latency_seconds.labels(schema=schema).observe(
                time.perf_counter() - start_time,
            )

    def _remaining_coverage_gaps(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> list[tuple[int, int]]:
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
        end_bucket = int(start_bucket) if end_bucket_candidate < start_bucket else int(end_bucket_candidate)
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
            symbol.split(".")[0].upper()
            if symbol and "." in symbol
            else str(symbol).upper()
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
            symbol.split(".")[0].upper()
            if symbol and "." in symbol
            else str(symbol).upper()
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

    def _map_schema_to_dataset_type(self, schema: str) -> DatasetType:
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
    ) -> None:
        registry_obj = self.data_registry
        if registry_obj is None:
            return
        registry = cast(RegistryProtocol, registry_obj)
        try:
            registry.get_manifest(dataset_id)
            return
        except Exception:
            pass

        manifest = build_auto_dataset_manifest(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            location=location,
            storage_kind=StorageKind.PARQUET,
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

    # ---------------------------------------------------------------------
    # Pre-ingestion (unified orchestrator path)
    # ---------------------------------------------------------------------

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

    def backfill(
        self,
        *,
        dataset_id: str,
        schema: str,
        instrument_id: str,
        lookback_days: int,
    ) -> list[tuple[int, int]]:
        orchestrator = IngestionOrchestrator(
            coverage=self.coverage,
            writer=self.writer,
            registry=self.data_registry,  # type: ignore[arg-type]
            ingestor=self.ingestor,  # type: ignore[arg-type]
            raw_writer=self.raw_writer,
            service=self.service,
        )
        return orchestrator.backfill_gaps(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
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
        identifier and delegates to IngestionOrchestrator.

        """
        days = get_max_lookback_days(dataset_id, policy)
        return self.backfill(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            lookback_days=days,
        )

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        # Prefer the public API to capture BuildResult (feature_set_id). Fallback to CLI if it fails.
        try:
            from ml.data import DatasetBuildConfig as APICfg
            from ml.data import build_tft_dataset as api_build

            symbols_list = [s.strip().upper() for s in cfg.symbols.split(",") if s.strip()]
            instrument_ids_list = (
                [item.strip() for item in cfg.instrument_ids] if cfg.instrument_ids else None
            )
            vintage_as_of_dt = parse_dt(cfg.vintage_as_of)

            api_cfg = APICfg(
                data_dir=Path(cfg.data_dir),
                out_dir=Path(cfg.out_dir),
                symbols=symbols_list,
                instrument_ids=instrument_ids_list,
                include_macro=cfg.include_macro,
                macro_lag_days=cfg.macro_lag_days,
                include_micro=cfg.include_micro,
                include_l2=cfg.include_l2,
                include_events=cfg.include_events,
                include_calendar=cfg.include_calendar,
                fred_vintage_dir=(
                    Path(cfg.fred_vintage_dir).expanduser() if cfg.fred_vintage_dir else None
                ),
                events_base_dir=(Path(cfg.events_dir).expanduser() if cfg.events_dir else None),
                student_mode=cfg.student_mode,
                horizon_minutes=cfg.horizon_minutes,
                threshold=cfg.threshold,
                lookback_periods=cfg.lookback_periods,
                start=(
                    None
                    if not cfg.start_iso
                    else __import__("datetime").datetime.fromisoformat(cfg.start_iso)
                ),
                end=(
                    None
                    if not cfg.end_iso
                    else __import__("datetime").datetime.fromisoformat(cfg.end_iso)
                ),
                chunk_days=int(cfg.chunk_days or 0),
                emit_dataset_events=cfg.emit_dataset_events,
                register_features=cfg.register_features,
                feature_registry_dir=(
                    None if cfg.feature_registry_dir is None else Path(cfg.feature_registry_dir)
                ),
                feature_role=cfg.feature_role,
                market_dataset_id=cfg.market_dataset_id,
                auto_refresh_macro=cfg.auto_refresh_macro,
                macro_staleness_hours=cfg.macro_staleness_hours,
                macro_series_ids=cfg.macro_series_ids,
                macro_fred_path=(
                    Path(cfg.macro_fred_path).expanduser() if cfg.macro_fred_path else None
                ),
                validation=cfg.validation,
                vintage_policy=cfg.vintage_policy,
                vintage_as_of=vintage_as_of_dt,
            )
            logger.info(
                "Dataset readiness | macro=%s events=%s student_mode=%s vintages=%s events_dir=%s",
                cfg.include_macro,
                cfg.include_events,
                getattr(cfg, "student_mode", False),
                bool(cfg.fred_vintage_dir),
                bool(cfg.events_dir),
            )
            result = api_build(
                api_cfg,
                data_store=cast("DataStoreFacadeProtocol | None", self.data_store),
            )
            if not result.feature_names:
                row_count = self._infer_dataset_row_count(result)
                metadata = getattr(result, "metadata", None)
                dataset_empty = row_count == 0
                if not dataset_empty and metadata is not None:
                    overall_window = getattr(metadata, "overall_window", None)
                    ts_start = getattr(metadata, "ts_event_start", None)
                    ts_end = getattr(metadata, "ts_event_end", None)
                    dataset_empty = overall_window is None and ts_start is None and ts_end is None
                if dataset_empty:
                    raise _EmptyDatasetError(
                        "Dataset build via API returned zero rows",
                        row_count=row_count,
                    )
                raise ValueError("API dataset build returned no features; falling back to CLI")
            manifest_id = self._export_feature_manifest(cfg, result)
            # Persist feature registration metadata for HPO
            try:
                meta_path = Path(cfg.out_dir) / "feature_registration.json"
                import json as _json

                payload = {
                    "feature_set_id": result.feature_set_id,
                    "feature_registry_dir": cfg.feature_registry_dir,
                    "feature_role": cfg.feature_role,
                }
                if manifest_id:
                    payload["manifest_id"] = manifest_id
                meta_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
            except Exception:
                pass
            dataset_metadata = getattr(result, "metadata", None)
            if dataset_metadata is not None:
                try:
                    self._guard_dataset_metadata(cfg=cfg, metadata=dataset_metadata)
                except Exception as exc:
                    raise ValueError(f"Dataset metadata guardrail violation: {exc}") from exc
                self._synchronize_dataset_manifest(cfg=cfg, metadata=dataset_metadata)
                try:
                    logger.info(
                        "Dataset metadata recorded | vintage_policy=%s vintage_cutoff=%s train_window=%s validation_window=%s",
                        dataset_metadata.vintage_policy.value,
                        dataset_metadata.vintage_cutoff,
                        dataset_metadata.train_window,
                        dataset_metadata.validation_window,
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug("Failed to log dataset metadata", exc_info=True)
            self._record_build_artifacts(
                cfg=cfg,
                feature_set_id=getattr(result, "feature_set_id", None),
                feature_names=list(getattr(result, "feature_names", [])),
                feature_registry_dir=cfg.feature_registry_dir,
                dataset_metadata=dataset_metadata,
            )
            return 0
        except _EmptyDatasetError as empty_exc:
            row_info = (
                f" rows={empty_exc.row_count}"
                if empty_exc.row_count is not None
                else ""
            )
            logger.error(
                "Dataset build produced no rows%s; extend the build window or ensure catalog coverage before rerunning.",
                row_info,
                extra={
                    "dataset_id": cfg.dataset_id,
                    "symbols": cfg.symbols,
                    "start_iso": cfg.start_iso,
                    "end_iso": cfg.end_iso,
                },
            )
            return 1
        except Exception as exc:  # pragma: no cover - defensive fallback to CLI path
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "API-based dataset build failed; falling back to CLI: %s",
                exc,
                exc_info=True,
            )

        # Fallback to invoking the CLI main with assembled args
        args: list[str] = [
            "--data_dir",
            cfg.data_dir,
            "--symbols",
            cfg.symbols,
            "--out_dir",
            cfg.out_dir,
            "--horizon_minutes",
            str(cfg.horizon_minutes),
            "--threshold",
            str(cfg.threshold),
            "--lookback_periods",
            str(cfg.lookback_periods),
        ]
        if cfg.include_macro:
            args += ["--include_macro", "--macro_lag_days", str(cfg.macro_lag_days)]
        if cfg.include_micro:
            args += ["--include_micro"]
        if cfg.include_l2:
            args += ["--include_l2"]
        if getattr(cfg, "include_events", False):
            args += ["--include_events"]
        if getattr(cfg, "include_calendar", False):
            args += ["--include_calendar"]
        if cfg.fred_vintage_dir:
            args += ["--fred_vintage_dir", cfg.fred_vintage_dir]
        if cfg.events_dir:
            args += ["--events_dir", cfg.events_dir]
        if cfg.student_mode:
            args += ["--student_mode"]
        if cfg.instrument_ids:
            args += ["--instrument_ids", ",".join(cfg.instrument_ids)]
        if cfg.market_dataset_id:
            args += ["--market_dataset_id", cfg.market_dataset_id]
        if not cfg.auto_refresh_macro:
            args += ["--skip_macro_refresh"]
        if cfg.macro_staleness_hours != 24:
            args += ["--macro_freshness_hours", str(cfg.macro_staleness_hours)]
        if cfg.macro_series_ids:
            args += ["--macro_series_ids", ",".join(cfg.macro_series_ids)]
        if cfg.macro_fred_path:
            args += ["--macro_fred_path", cfg.macro_fred_path]
        if cfg.vintage_policy:
            args += ["--vintage_policy", cfg.vintage_policy.value]
        if cfg.vintage_as_of:
            args += ["--vintage_as_of", cfg.vintage_as_of]
        if cfg.validation is not None:
            args += ["--validation_min_rows", str(cfg.validation.min_rows)]
            if cfg.validation.min_positive_rate is not None:
                args += ["--validation_min_positive_rate", str(cfg.validation.min_positive_rate)]
            if cfg.validation.max_positive_rate is not None:
                args += ["--validation_max_positive_rate", str(cfg.validation.max_positive_rate)]
            if cfg.validation.min_feature_coverage is not None:
                args += [
                    "--validation_min_feature_coverage",
                    str(cfg.validation.min_feature_coverage),
                ]
        if getattr(cfg, "start_iso", None):
            args += ["--start", str(cfg.start_iso)]
        if getattr(cfg, "end_iso", None):
            args += ["--end", str(cfg.end_iso)]
        if int(getattr(cfg, "chunk_days", 0) or 0) > 0:
            args += ["--chunk_days", str(int(cfg.chunk_days))]
        if cfg.emit_dataset_events:
            args += ["--emit_dataset_events"]
        if cfg.register_features:
            args += ["--register_features"]
            reg_dir = cfg.feature_registry_dir or str(Path.home() / ".nautilus" / "ml" / "features")
            args += ["--feature_registry_dir", reg_dir]
        rc = self.build_main(args)
        if rc == 0:
            self._capture_cli_build_artifacts(cfg)
        return rc

    @staticmethod
    def _export_feature_manifest(
        cfg: DatasetBuildConfig,
        result: object,
    ) -> str | None:
        """Export a feature manifest when registry configuration is provided."""
        if not cfg.register_features or not cfg.feature_registry_dir:
            return None

        try:
            feature_names = getattr(result, "feature_names")
        except AttributeError:
            logger.warning("Feature manifest export skipped: result missing feature_names")
            return None
        if not feature_names:
            logger.warning("Feature manifest export skipped: no feature names returned")
            return None

        try:
            from ml.data.feature_manifest_export import FeatureExportConfig
            from ml.data.feature_manifest_export import export_feature_manifest
            from ml.registry.base import DataRequirements
            from ml.registry.feature_registry import FeatureRole
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("Feature manifest export unavailable: %s", exc)
            return None

        try:
            role = FeatureRole(cfg.feature_role)
        except ValueError:
            logger.warning("Unknown feature_role '%s'; defaulting to TEACHER", cfg.feature_role)
            role = FeatureRole.TEACHER

        data_requirements = (
            DataRequirements.L1_L2 if cfg.include_l2 else DataRequirements.L1_ONLY
        )
        flags = {
            "include_macro": cfg.include_macro,
            "macro_lag_days": cfg.macro_lag_days,
            "include_events": cfg.include_events,
            "include_l2": cfg.include_l2,
            "student_mode": cfg.student_mode,
            "fred_vintages": bool(cfg.fred_vintage_dir),
            "events_dir": bool(cfg.events_dir),
        }

        export_cfg = FeatureExportConfig(
            registry_path=Path(cfg.feature_registry_dir),
            role=role,
            data_requirements=data_requirements,
        )

        try:
            manifest_id = export_feature_manifest(
                feature_names=list(feature_names),
                flags=flags,
                cfg=export_cfg,
            )
            logger.info("Exported feature manifest %s", manifest_id)
            return manifest_id
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Feature manifest export failed: %s", exc, exc_info=True)
            return None

    def _record_build_artifacts(
        self,
        *,
        cfg: DatasetBuildConfig,
        feature_set_id: str | None,
        feature_names: Sequence[str] | None,
        feature_registry_dir: str | None,
        dataset_metadata: DatasetMetadata | None = None,
    ) -> None:
        names_tuple = tuple(feature_names or [])
        self._build_artifacts = BuildArtifacts(
            out_dir=Path(cfg.out_dir),
            feature_registry_dir=feature_registry_dir,
            feature_set_id=feature_set_id,
            feature_names=names_tuple,
            dataset_metadata=dataset_metadata,
        )

    def _guard_dataset_metadata(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        """Validate dataset metadata against configuration guardrails."""
        def _normalize(value: str | None) -> str | None:
            if not value:
                return None
            try:
                dt_value = parse_dt(value)
            except ValueError:
                return value
            if dt_value is None:
                return value
            formatted = format_dt(dt_value)
            return formatted or value

        expectations = DatasetMetadataExpectations(
            dataset_id=cfg.dataset_id,
            vintage_policy=cfg.vintage_policy,
            vintage_cutoff=_normalize(cfg.vintage_as_of),
            ts_event_start=_normalize(cfg.start_iso),
            ts_event_end=_normalize(cfg.end_iso),
        )
        validate_dataset_metadata_expectations(
            metadata,
            expectations,
            context="orchestrator.dataset",
        )
        if cfg.include_macro and cfg.macro_series_ids:
            missing = []
            for series in cfg.macro_series_ids:
                key = str(series)
                if metadata.macro_observation_counts.get(key, 0) <= 0:
                    missing.append(key)
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"Missing macro observations for series: {missing_str}",
                )

    @staticmethod
    def _compute_dataset_pipeline_signature(
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> str:
        """Derive a stable pipeline signature covering vintage policy and scope."""
        return compute_dataset_pipeline_signature(
            dataset_id=cfg.dataset_id,
            symbols=cfg.symbols,
            instrument_ids=cfg.instrument_ids,
            macro_series_ids=cfg.macro_series_ids,
            include_macro=cfg.include_macro,
            macro_lag_days=cfg.macro_lag_days,
            vintage_policy=metadata.vintage_policy,
            vintage_cutoff=metadata.vintage_cutoff,
            ts_event_start=metadata.ts_event_start,
            ts_event_end=metadata.ts_event_end,
        )

    def _synchronize_dataset_manifest(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        registry_obj = self.data_registry
        if registry_obj is None or not cfg.dataset_id:
            return
        registry = cast(RegistryProtocol, registry_obj)

        try:
            manifest = registry.get_manifest(cfg.dataset_id)
        except Exception:
            logger.debug(
                "Data registry manifest missing for dataset_id=%s; skipping metadata sync",
                cfg.dataset_id,
            )
            return

        manifest_metadata = dict(getattr(manifest, "metadata", {}) or {})
        manifest_metadata.update(
            {
                "dataset_id": cfg.dataset_id,
                "vintage": {
                    "policy": metadata.vintage_policy.value,
                    "cutoff": metadata.vintage_cutoff,
                    "build_ts": metadata.build_ts,
                },
                "windows": {
                    "overall": metadata.overall_window,
                    "train": metadata.train_window,
                    "validation": metadata.validation_window,
                    "test": metadata.test_window,
                    "ts_event_start": metadata.ts_event_start,
                    "ts_event_end": metadata.ts_event_end,
                },
            },
        )

        try:
            registry.update_manifest(
                cfg.dataset_id,
                {
                    "metadata": manifest_metadata,
                    "pipeline_signature": self._compute_dataset_pipeline_signature(cfg, metadata),
                },
            )
        except Exception as exc:  # pragma: no cover - registry backend failures
            logger.debug(
                "Failed to update dataset manifest metadata: %s",
                exc,
                exc_info=True,
            )

    @staticmethod
    def _infer_feature_names(out_dir: Path) -> tuple[str, ...]:
        dataset_path = out_dir / "dataset.parquet"
        if not dataset_path.exists():
            logger.debug("Dataset parquet missing after CLI build: %s", dataset_path)
            return ()
        try:
            from ml._imports import HAS_PANDAS
            from ml._imports import HAS_POLARS
            from ml._imports import check_ml_dependencies
            from ml._imports import pd
            from ml._imports import pl
        except Exception as exc:  # pragma: no cover - defensive import guard
            logger.debug("Failed to import dataset engines: %s", exc)
            return ()

        exclude = {"y", "time_index", "timestamp", "instrument_id", "ts_event"}
        try:
            if HAS_POLARS and pl is not None:
                frame = pl.read_parquet(str(dataset_path))
                return tuple(col for col in frame.columns if col not in exclude)
            if HAS_PANDAS and pd is not None:
                frame_pd = pd.read_parquet(str(dataset_path))
                return tuple(col for col in frame_pd.columns if col not in exclude)
        except Exception as exc:  # pragma: no cover - io errors
            logger.warning("Failed to inspect dataset parquet: %s", exc)
            return ()
        try:
            check_ml_dependencies(["polars"])
        except Exception:
            pass
        return ()

    def _capture_cli_build_artifacts(self, cfg: DatasetBuildConfig) -> None:
        out_dir = Path(cfg.out_dir)
        feature_registry_dir = cfg.feature_registry_dir
        feature_set_id: str | None = None
        feature_names: tuple[str, ...] = ()
        for candidate in (
            out_dir / "feature_set.json",
            out_dir / "feature_registration.json",
        ):
            if not candidate.exists():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("Failed to parse feature metadata %s: %s", candidate, exc)
                continue
            feature_registry_dir = feature_registry_dir or data.get("feature_registry_dir")
            feature_set_id = feature_set_id or data.get("feature_set_id")
            names = data.get("feature_names")
            if isinstance(names, list):
                feature_names = tuple(str(name) for name in names)

        if not feature_names:
            feature_names = self._infer_feature_names(out_dir)

        manifest_id: str | None = None
        dataset_metadata: DatasetMetadata | None = None
        metadata_path = out_dir / "dataset_metadata.json"
        if metadata_path.exists():
            try:
                dataset_metadata = load_dataset_metadata(metadata_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load dataset metadata from %s: %s",
                    metadata_path,
                    exc,
                )
        else:
            logger.debug("Dataset metadata not found at %s; continuing without it", metadata_path)

        if cfg.register_features and feature_names:
            sentinel = type("_Result", (), {"feature_names": feature_names})
            manifest_id = self._export_feature_manifest(cfg, sentinel)
            if manifest_id:
                feature_set_id = feature_set_id or manifest_id
                feature_registry_dir = feature_registry_dir or cfg.feature_registry_dir
                payload = {
                    "feature_set_id": feature_set_id,
                    "feature_registry_dir": feature_registry_dir,
                    "feature_names": list(feature_names),
                    "manifest_id": manifest_id,
                }
                try:
                    (out_dir / "feature_registration.json").write_text(
                        json.dumps(payload, indent=2),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to persist feature registration metadata: %s",
                        exc,
                    )

        if dataset_metadata is not None:
            try:
                self._guard_dataset_metadata(cfg=cfg, metadata=dataset_metadata)
            except Exception as exc:
                raise ValueError(f"Dataset metadata guardrail violation: {exc}") from exc
            self._synchronize_dataset_manifest(cfg=cfg, metadata=dataset_metadata)

        self._record_build_artifacts(
            cfg=cfg,
            feature_set_id=feature_set_id,
            feature_names=feature_names,
            feature_registry_dir=feature_registry_dir,
            dataset_metadata=dataset_metadata,
        )

    def run_hpo(self, cfg: HPOConfig, dataset_csv: Path, out_dir: Path) -> int:
        if not cfg.enabled or self.hpo_main is None:
            return 0
        artifacts = self._build_artifacts
        args = [
            "--dataset_csv",
            str(dataset_csv),
            "--out_dir",
            str(out_dir),
            "--epochs",
            str(cfg.epochs),
            "--batch_size",
            str(cfg.batch_size),
            "--tail_rows",
            str(cfg.tail_rows),
            "--limit_groups",
            str(cfg.limit_groups),
            "--workers",
            str(cfg.workers),
            "--backend",
            cfg.backend,
            "--metric",
            cfg.metric,
            "--optuna_trials",
            str(cfg.optuna_trials),
            "--loss",
            cfg.loss,
            "--pos_weight",
            cfg.pos_weight,
        ]
        if cfg.direction:
            args += ["--direction", cfg.direction]
        if cfg.optuna_timeout is not None:
            args += ["--optuna_timeout", str(cfg.optuna_timeout)]
        if artifacts and artifacts.feature_registry_dir:
            args += ["--feature_registry_dir", artifacts.feature_registry_dir]
        if artifacts and artifacts.feature_set_id:
            args += ["--feature_set_id", artifacts.feature_set_id]
        return self.hpo_main(args)

    def train_teacher(self, cfg: TeacherTrainConfig, dataset_csv: Path, out_dir: Path) -> int:
        if not cfg.enabled:
            return 0
        artifacts = self._build_artifacts
        feature_registry_dir = cfg.feature_registry_dir or (
            artifacts.feature_registry_dir if artifacts else None
        )
        feature_set_id = cfg.feature_set_id or (artifacts.feature_set_id if artifacts else None)

        metadata_path = dataset_csv.parent / "dataset_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata missing at {metadata_path}")

        metadata_source: DatasetMetadata | None = None
        if artifacts and artifacts.dataset_metadata is not None:
            metadata_source = artifacts.dataset_metadata
        else:
            try:
                metadata_source = load_dataset_metadata(metadata_path)
            except Exception as exc:
                logger.debug("Failed to load dataset metadata prior to training: %s", exc)

        if metadata_source is None or metadata_source.dataset_id is None:
            raise ValueError("Dataset metadata must include dataset_id before teacher training")

        args: list[str] = [
            "--train_data_csv",
            str(dataset_csv),
            "--out_dir",
            str(out_dir),
            "--model_id",
            cfg.model_id,
            "--max_epochs",
            str(cfg.max_epochs),
            "--dataset_metadata",
            str(metadata_path),
            "--expected_dataset_id",
            metadata_source.dataset_id,
            "--expected_vintage_policy",
            metadata_source.vintage_policy.value,
        ]
        if metadata_source.vintage_cutoff:
            args += ["--expected_vintage_cutoff", metadata_source.vintage_cutoff]
        if feature_registry_dir is not None:
            args += ["--feature_registry_dir", feature_registry_dir]
        if feature_set_id is not None:
            args += ["--feature_set_id", feature_set_id]
        return self.teacher_main(args)

    def distill_student(
        self,
        cfg: StudentDistillConfig,
        *,
        dataset_dir: Path,
        teacher_cfg: TeacherTrainConfig,
    ) -> int:
        if not cfg.enabled:
            return 0

        features_npz = dataset_dir / "features_npz.npz"
        teacher_npz = dataset_dir / "teacher_preds.npz"
        if not features_npz.exists():
            logger.error("Distillation enabled but missing features NPZ at %s", features_npz)
            return 1
        if not teacher_npz.exists():
            logger.error("Distillation enabled but missing teacher predictions NPZ at %s", teacher_npz)
            return 1

        artifacts = self._build_artifacts
        feature_registry_dir = cfg.feature_registry_dir or (
            artifacts.feature_registry_dir if artifacts else None
        )
        feature_set_id = cfg.feature_set_id or (
            artifacts.feature_set_id if artifacts else None
        )
        if feature_registry_dir is None or feature_set_id is None:
            logger.error(
                "Feature registry metadata required for distillation (have dir=%s id=%s)",
                feature_registry_dir,
                feature_set_id,
            )
            return 1

        model_registry_dir = cfg.model_registry_dir
        if model_registry_dir is None:
            logger.error("model_registry_dir is required for student registration")
            return 1

        parent_model_id = cfg.parent_model_id or teacher_cfg.model_id
        args: list[str] = [
            "--features_npz",
            str(features_npz),
            "--teacher_npz",
            str(teacher_npz),
            "--out_dir",
            str(dataset_dir),
            "--model_id",
            cfg.model_id,
            "--parent_id",
            parent_model_id,
            "--registry_dir",
            model_registry_dir,
            "--feature_registry_dir",
            feature_registry_dir,
            "--feature_set_id",
            feature_set_id,
            "--objective",
            cfg.objective,
            "--kd_lambda",
            str(cfg.kd_lambda),
            "--early_stopping",
            str(cfg.early_stopping),
        ]
        if cfg.opset is not None:
            args += ["--opset", str(cfg.opset)]
        if cfg.use_val_for_distill:
            args += ["--use_val_for_distill"]

        from ml.training.distillation.cli import main as distill_main

        return distill_main(args)

    def run(self, cfg: OrchestratorConfig) -> int:
        # 0) Optional pre-ingestion stage (unified orchestrator path)
        if cfg.pre_ingestion is not None:
            # Prefer environment CATALOG_PATH to keep configs portable
            import os

            catalog_path_env = os.getenv("CATALOG_PATH")
            if catalog_path_env:
                self.run_pre_ingestion(
                    catalog_path=Path(catalog_path_env),
                    scheduler_cfg=cfg.pre_ingestion,
                    options=cfg.pre_ingestion_options,
                )

        if cfg.auto_fill is not None and cfg.auto_fill.enabled:
            self._auto_fill_universe(cfg.dataset, cfg.auto_fill)

        # 1) Build dataset
        rc = self.build_dataset(cfg.dataset)
        if rc != 0:
            return rc
        out_dir = Path(cfg.dataset.out_dir)
        dataset_csv = out_dir / "dataset.csv"

        # 2) HPO (optional)
        rc = self.run_hpo(cfg.hpo, dataset_csv=dataset_csv, out_dir=out_dir)
        if rc != 0:
            return rc

        # 3) Train teacher / calibration
        rc = self.train_teacher(cfg.teacher, dataset_csv=dataset_csv, out_dir=out_dir)
        if rc != 0:
            return rc

        rc = self.distill_student(cfg.student, dataset_dir=out_dir, teacher_cfg=cfg.teacher)
        if rc != 0:
            return rc

        self._handle_promotions(cfg.promotions, out_dir=out_dir, dataset_csv=dataset_csv)
        self._attach_runtime(cfg.integration, dataset_out_dir=out_dir)
        return 0

    def _handle_promotions(
        self,
        promotions: PromotionsConfig | None,
        *,
        out_dir: Path,
        dataset_csv: Path,
    ) -> None:
        if promotions is None:
            return

        from ml.orchestration.promotions import register_and_promote_model
        from ml.orchestration.promotions import register_or_refresh_features
        from ml.registry.dataclasses import QualityGate

        logger = logging.getLogger(__name__)

        feature_registry = self.feature_registry
        feature_metrics_path = Path(
            promotions.feature_metrics_json or (Path(out_dir) / "feature_metrics.json"),
        )
        if promotions.refresh_features or promotions.auto_register_features:
            if not feature_metrics_path.exists():
                feature_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "feature_set_id": f"auto_refresh_{_uuid.uuid4().hex[:8]}",
                    "generated_ts": int(time.time()),
                    "dataset_csv": str(dataset_csv),
                }
                feature_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            succeeded = False
            if feature_registry is not None:
                try:
                    register_or_refresh_features(
                        feature_metrics_path=str(feature_metrics_path),
                        feature_registry=feature_registry,
                        auto_register=bool(promotions.auto_register_features),
                    )
                    succeeded = True
                except Exception as exc:
                    logger.warning("Feature refresh failed: %s", exc)
            if not succeeded:
                self._emit_feature_refresh_event(feature_metrics_path)

        should_promote_model = bool(
            promotions.auto_register_model or promotions.auto_promote or promotions.deploy_target,
        )

        if not should_promote_model or self.model_registry is None:
            return

        metrics_path = Path(out_dir) / "model_metrics.json"
        if not metrics_path.exists():
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_id": f"auto_model_{_uuid.uuid4().hex[:8]}",
                "model_path": str(Path(out_dir) / "model.onnx"),
                "architecture": "unknown",
                "feature_schema_hash": f"auto_{_uuid.uuid4().hex[:8]}",
                "serveable": True,
            }
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        gates: list[QualityGate] = []
        if promotions.gates_json:
            try:
                data = json.loads(Path(promotions.gates_json).read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        metric_name = item.get("metric") or item.get("metric_name")
                        if not metric_name:
                            continue
                        comparison_value = item.get("comparison")
                        comparison: str = (
                            str(comparison_value)
                            if comparison_value is not None
                            else "gte"
                        )
                        gates.append(
                            QualityGate(
                                metric_name=str(metric_name),
                                threshold=float(item.get("threshold", 0.0)),
                                comparison=comparison,
                                required=bool(item.get("required", True)),
                            ),
                        )
            except Exception as exc:
                logger.warning("Failed to parse gates JSON %s: %s", promotions.gates_json, exc)
                gates = []

        try:
            register_and_promote_model(
                model_metrics_path=str(metrics_path),
                out_dir=str(out_dir),
                registry=self.model_registry,
                feature_registry=self.feature_registry,
                gates=gates,
                auto_promote=bool(promotions.auto_promote),
                deploy_target=promotions.deploy_target,
            )
        except Exception as exc:
            logger.warning("Model promotion failed: %s", exc)

    def _attach_runtime(
        self,
        integration_cfg: IntegrationConfig | None,
        *,
        dataset_out_dir: Path,
    ) -> None:
        if integration_cfg is None or not integration_cfg.enabled:
            return

        logger.info(
            "Attaching ML integration runtime (validators=%s, out_dir=%s)",
            integration_cfg.run_validators,
            dataset_out_dir,
        )

        if self._integration_manager is None:
            factory = self.integration_manager_factory
            if factory is None:
                from ml.core.integration import MLIntegrationManager as _MLIntegrationManager

                factory = cast(
                    Callable[..., IntegrationManagerProtocol],
                    _MLIntegrationManager,
                )

            kwargs: dict[str, Any] = {
                "auto_start_postgres": integration_cfg.auto_start_postgres,
                "auto_migrate": integration_cfg.auto_migrate,
                "ensure_healthy": integration_cfg.ensure_healthy,
            }
            if integration_cfg.db_connection is not None:
                kwargs["db_connection"] = integration_cfg.db_connection
            if integration_cfg.strict_protocol_validation is not None:
                kwargs["strict_protocol_validation"] = integration_cfg.strict_protocol_validation

            manager = factory(**kwargs)
            object.__setattr__(self, "_integration_manager", manager)
        else:
            manager = self._integration_manager

        for attr in (
            "data_registry",
            "feature_registry",
            "model_registry",
            "strategy_registry",
            "feature_store",
            "model_store",
            "strategy_store",
            "data_store",
            "partition_manager",
        ):
            if getattr(self, attr, None) is None:
                object.__setattr__(self, attr, getattr(manager, attr, None))

        if integration_cfg.run_validators:
            self._run_validators()

    def _run_validators(self) -> None:
        from tools import validate_event_constants as event_mod
        from tools import validate_metrics_bootstrap as metrics_mod

        metrics_rc = metrics_mod.main()
        if metrics_rc != 0:
            raise RuntimeError("metrics bootstrap validation failed")

        events_rc = event_mod.main()
        if events_rc != 0:
            raise RuntimeError("event constants validation failed")

        logger.info("Runtime validators succeeded")

    def _emit_feature_refresh_event(self, metrics_path: Path) -> None:
        try:
            from ml.common.event_emitter import emit_dataset_event
            from ml.config.events import EventStatus
            from ml.config.events import Source
            from ml.config.events import Stage
        except Exception:
            return

        feature_set_id = "unknown"
        metadata: dict[str, object] = {}
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    feature_set_id = str(data.get("feature_set_id", feature_set_id))
                    metadata = {k: v for k, v in data.items() if isinstance(k, str)}
            except Exception:
                pass

        meta_payload = dict(metadata)
        meta_payload["feature_set_id"] = feature_set_id

        try:
            registry_obj = self.data_registry
            if registry_obj is None:
                return
            data_registry = cast(RegistryProtocol, registry_obj)
            emit_dataset_event(
                data_registry,
                dataset_id="features",
                instrument_id="GLOBAL",
                stage=Stage.FEATURE_COMPUTED,
                source=Source.HISTORICAL,
                run_id=f"refresh_{feature_set_id}",
                ts_min=0,
                ts_max=0,
                count=1,
                status=EventStatus.SUCCESS,
                metadata=meta_payload,
                dataset_type="features",
                component="pipeline_orchestrator.refresh_features",
            )
        except Exception:
            pass


@dataclass(slots=True, frozen=True)
class PreIngestionOptions:
    """
    Options for the pre-ingestion scheduler stage.
    """

    use_orchestrator: bool = True
    dual_write: bool = True
    start_metrics_server: bool = False
    metrics_port: int | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    import os

    parser = argparse.ArgumentParser(description="Run end-to-end ML pipeline (cold path)")

    # Ingestion/backfill
    parser.add_argument("--ingest", action="store_true", help="Run ingestion backfill first")
    parser.add_argument("--dataset_id", default="EQUS.MINI")
    parser.add_argument("--schema", default="bars", choices=["bars", "tbbo", "trades"])
    parser.add_argument("--instruments", default="SPY.NYSE")
    parser.add_argument("--lookback_days", type=int, default=7)
    parser.add_argument("--coverage_mode", default="catalog", choices=["catalog", "sql"])
    parser.add_argument("--catalog_path", default=os.getenv("CATALOG_PATH", ""))
    parser.add_argument(
        "--db",
        default=os.getenv("NAUTILUS_DB", "postgresql://postgres:postgres@localhost:5432/nautilus"),
    )

    # Writer mode for ingestion
    parser.add_argument(
        "--write_mode",
        default="parquet",
        choices=["parquet", "datastore"],
        help="Mirror DataStore writes to Parquet (parquet) or keep datastore-only persistence",
    )

    # Dataset build
    parser.add_argument("--data_dir", default="data/tier1")
    parser.add_argument("--symbols", default="SPY.NYSE")
    parser.add_argument("--out_dir", default="ml_out")
    parser.add_argument("--include_macro", action="store_true")
    parser.add_argument("--macro_lag_days", type=int, default=1)
    parser.add_argument("--include_micro", action="store_true")
    parser.add_argument("--include_l2", action="store_true")
    parser.add_argument("--include_events", action="store_true")
    parser.add_argument("--include_calendar", action="store_true")
    parser.add_argument(
        "--instrument_ids",
        default=None,
        help="Comma-separated instrument identifiers (symbol.exchange)",
    )
    parser.add_argument(
        "--market_dataset_id",
        default=None,
        help="Identifier for the canonical market data dataset (defaults to auto-fill dataset)",
    )
    parser.add_argument(
        "--skip_macro_refresh",
        action="store_true",
        help="Skip automatic macro refresh even when macro features are included",
    )
    parser.add_argument(
        "--macro_freshness_hours",
        type=int,
        default=24,
        help="Maximum age (hours) before macro artifacts are refreshed",
    )
    parser.add_argument(
        "--macro_series_ids",
        default=None,
        help="Comma-separated list of macro series ids to refresh (defaults to loader configuration)",
    )
    parser.add_argument(
        "--macro_fred_path",
        default=None,
        help="Explicit target path for FRED ML parquet (defaults to data/fred/fred_indicators_ml_format.parquet)",
    )
    parser.add_argument(
        "--vintage_policy",
        default=VintagePolicy.REAL_TIME.value,
        choices=[policy.value for policy in VintagePolicy],
        help="Vintage policy for macro features (real_time or final)",
    )
    parser.add_argument(
        "--vintage_as_of",
        default=None,
        help="ISO8601 timestamp (UTC) limiting macro revisions (optional)",
    )
    parser.add_argument("--validation_min_rows", type=int, default=None)
    parser.add_argument("--validation_min_positive_rate", type=float, default=None)
    parser.add_argument("--validation_max_positive_rate", type=float, default=None)
    parser.add_argument("--validation_min_feature_coverage", type=float, default=None)
    parser.add_argument(
        "--skip_l2_ingest",
        action="store_true",
        help="Skip automatic L2 ingestion even when include_l2 is enabled",
    )
    parser.add_argument(
        "--l2_days",
        type=int,
        default=30,
        help="Number of calendar days to ingest depth data when include_l2 is enabled",
    )
    parser.add_argument(
        "--l2_progress_file",
        default=None,
        help="Optional path for tracking L2 ingestion progress (defaults to <data_dir>/.l2_progress.json)",
    )
    parser.add_argument(
        "--l2_symbols",
        default=None,
        help="Comma-separated list of symbols for L2 ingestion (defaults to Tier 1 universe)",
    )
    parser.add_argument(
        "--l2_tier",
        type=int,
        default=1,
        help="Tier to use for automatic L2 ingestion when symbols are not provided",
    )
    parser.add_argument(
        "--fred_vintage_dir",
        default=None,
        help="Optional ALFRED vintage directory",
    )
    parser.add_argument("--events_dir", default=None, help="Optional normalized events directory")
    parser.add_argument(
        "--student_mode",
        action="store_true",
        help="Build student-mode (L1-only) dataset",
    )
    parser.add_argument(
        "--emit_dataset_events",
        action="store_true",
        help="Emit dataset events via DataRegistry for the TFT build",
    )
    parser.add_argument("--horizon_minutes", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--lookback_periods", type=int, default=30)
    parser.add_argument("--start_iso", default=None, help="Optional start date ISO (YYYY-MM-DD)")
    parser.add_argument("--end_iso", default=None, help="Optional end date ISO (YYYY-MM-DD)")
    parser.add_argument(
        "--chunk_days",
        type=int,
        default=0,
        help="Chunk build by N days (0=disabled)",
    )
    parser.add_argument(
        "--auto_fill_universe",
        action="store_true",
        help="Automatically backfill market data coverage before dataset build",
    )
    parser.add_argument(
        "--auto_fill_dataset_id",
        default=None,
        help="Dataset identifier for auto-fill (defaults to --dataset_id)",
    )
    parser.add_argument(
        "--auto_fill_instrument_ids",
        default=None,
        help="Comma-separated instrument IDs overriding dataset config for auto-fill",
    )
    parser.add_argument(
        "--auto_fill_l2_days",
        type=int,
        default=None,
        help="Override L2 lookback window for auto-fill (days)",
    )
    parser.add_argument(
        "--auto_fill_skip_l2",
        action="store_true",
        help="Skip L2 ingestion during auto-fill",
    )
    parser.add_argument(
        "--auto_fill_l2_dataset_id",
        default=None,
        help="Dataset identifier for auto-fill L2 ingestion (default DBEQ.BASIC)",
    )
    parser.add_argument(
        "--auto_fill_l2_schema",
        default=None,
        help="Schema to use for auto-fill L2 ingestion (default mbp-10)",
    )
    parser.add_argument(
        "--auto_fill_l2_progress_file",
        default=None,
        help="Progress file path for auto-fill L2 ingestion",
    )
    parser.add_argument(
        "--auto_fill_include_l3",
        action="store_true",
        help="Attempt L3 auto-fill when helpers are available",
    )
    parser.add_argument(
        "--auto_fill_l3_dataset_id",
        default=None,
        help="Dataset identifier for auto-fill L3 ingestion",
    )
    parser.add_argument(
        "--auto_fill_l3_schema",
        default=None,
        help="Schema to use for auto-fill L3 ingestion",
    )
    parser.add_argument(
        "--auto_fill_l3_days",
        type=int,
        default=None,
        help="Override L3 lookback window for auto-fill (days)",
    )
    parser.add_argument(
        "--auto_fill_allow_dataset_l2_ingest",
        action="store_true",
        help="Allow dataset-stage L2 ingestion even when auto-fill runs",
    )
    parser.add_argument(
        "--attach-runtime",
        action="store_true",
        help="Attach MLIntegrationManager after pipeline completion",
    )
    parser.add_argument(
        "--runtime-db-connection",
        default=None,
        help="Override DB connection string for runtime attachment",
    )
    parser.add_argument(
        "--runtime-auto-start-db",
        action="store_true",
        help="Automatically start PostgreSQL when attaching runtime",
    )
    parser.add_argument(
        "--runtime-auto-migrate",
        action="store_true",
        help="Run database migrations when attaching runtime",
    )
    parser.add_argument(
        "--runtime-no-ensure-healthy",
        action="store_true",
        help="Skip health checks during runtime attachment",
    )
    parser.add_argument(
        "--runtime-strict-protocol-validation",
        action="store_true",
        help="Enable strict protocol validation when attaching runtime",
    )
    parser.add_argument(
        "--runtime-skip-validators",
        action="store_true",
        help="Skip metrics/events validators during runtime attachment",
    )

    # HPO
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--hpo_epochs", type=int, default=2)
    parser.add_argument("--hpo_batch_size", type=int, default=32)
    parser.add_argument("--hpo_tail_rows", type=int, default=5000)
    parser.add_argument("--hpo_limit_groups", type=int, default=50)

    # Teacher training
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--teacher_model_id", default="teacher_model")
    parser.add_argument("--feature_registry_dir", default=None)
    parser.add_argument(
        "--dataset_register_features",
        action="store_true",
        help="Register features during dataset build using feature_registry_dir",
    )
    parser.add_argument("--feature_set_id", default=None)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--distill_student", action="store_true")
    parser.add_argument("--student_model_id", default="student_model")
    parser.add_argument("--student_parent_model_id", default=None)
    parser.add_argument("--student_model_registry_dir", default=None)
    parser.add_argument("--student_feature_registry_dir", default=None)
    parser.add_argument("--student_feature_set_id", default=None)
    parser.add_argument(
        "--student_objective",
        default="logit_mse",
        choices=["logit_mse", "soft_ce", "hybrid"],
    )
    parser.add_argument("--student_kd_lambda", type=float, default=0.5)
    parser.add_argument("--student_early_stopping", type=int, default=200)
    parser.add_argument("--student_opset", type=int, default=None)
    parser.add_argument("--student_use_val_for_distill", action="store_true")

    # Optional promotions and feature registration
    parser.add_argument("--auto_register_model", action="store_true")
    parser.add_argument("--gates_json", default=None)
    parser.add_argument("--auto_promote", action="store_true")
    parser.add_argument("--deploy_target", default=None)

    parser.add_argument("--auto_register_features", action="store_true")
    parser.add_argument("--feature_metrics_json", default=None)

    # Optional small feature refresh phase
    parser.add_argument("--refresh_features", action="store_true")

    # Promotion stage 2 (walk-forward + cost-aware backtest)
    parser.add_argument("--promote_stage2", action="store_true")
    parser.add_argument("--stage2_gates_json", default=None)
    parser.add_argument("--stage2_cost_bps", type=float, default=0.0)
    parser.add_argument(
        "--stage2_engine",
        choices=["returns", "backtest"],
        default="returns",
        help="Stage 2 engine: returns (default) or backtest (advisory)",
    )
    parser.add_argument("--stage2_commission_bps", type=float, default=0.0)
    parser.add_argument("--stage2_slippage_bps", type=float, default=0.0)
    parser.add_argument(
        "--final_model_id",
        default=None,
        help="Model ID to promote in stage 2 (optional)",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _run_id: str = f"orch_{_uuid.uuid4().hex[:12]}"
    bind_log_context(run_id=_run_id, component="ml.pipeline_orchestrator")

    from ml.core.integration import MLIntegrationManager

    mgr = MLIntegrationManager(
        db_connection=args.db,
        auto_start_postgres=False,
        auto_migrate=False,
        ensure_healthy=False,
    )
    data_store = getattr(mgr, "data_store", None)
    if data_store is None:
        logger.info(
            "DataStore unavailable; falling back to catalog-only runtime attachment",
        )
    if mgr.data_registry is None:
        raise SystemExit("DataRegistry unavailable; configure ML_DB_CONNECTION for pipeline orchestration")

    registry = mgr.data_registry
    manifest_resolver = None
    if registry is not None and hasattr(registry, "get_manifest"):
        manifest_resolver = cast(RegistryProtocol, registry).get_manifest

    parquet_catalog: Any | None = None
    raw_writer: RawIngestionWriterProtocol | None = None
    if args.catalog_path:
        try:
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

            parquet_catalog = ParquetDataCatalog(args.catalog_path)
        except Exception as exc:  # pragma: no cover - import env issue
            raise SystemExit(f"ParquetDataCatalog unavailable: {exc}")
        raw_writer = ParquetCatalogRawWriter(parquet_catalog)

    coverage: CoverageProviderProtocol
    if args.coverage_mode == "catalog":
        if parquet_catalog is None:
            raise SystemExit("catalog_path is required for catalog coverage mode")
        coverage = CatalogCoverageProvider(catalog_path=args.catalog_path)
    else:
        coverage = SqlCoverageProvider(connection_string=args.db)

    primary_writer: MarketDataWriterProtocol
    mirror_writers: tuple[MarketDataWriterProtocol, ...]
    if data_store is not None:
        from ml.stores.data_store import DataStore as _DataStore

        primary_writer = DataStoreMarketDataWriter(
            store=cast(_DataStore, data_store),
        )
        if args.write_mode == "parquet":
            if parquet_catalog is None:
                raise SystemExit("catalog_path is required when write_mode=parquet")
            parquet_writer = ParquetCatalogMarketDataWriter(
                catalog=parquet_catalog,
                manifest_resolver=manifest_resolver,
            )
            mirror_writers = (parquet_writer,)
        else:
            mirror_writers = ()
    else:
        if parquet_catalog is None:
            raise SystemExit(
                "DataStore unavailable and no catalog_path provided; cannot attach runtime writer",
            )
        primary_writer = ParquetCatalogMarketDataWriter(
            catalog=parquet_catalog,
            manifest_resolver=manifest_resolver,
        )
        mirror_writers = ()

    writer = FanoutMarketDataWriter(primary=primary_writer, mirrors=mirror_writers)
    integration_factory: Callable[..., IntegrationManagerProtocol] | None = cast(
        Callable[..., IntegrationManagerProtocol],
        MLIntegrationManager,
    )

    ingestor: object | None = None
    ingestion_service: DatabentoIngestionService | None = None
    need_databento = bool(args.ingest or getattr(args, "auto_fill_universe", False))
    if need_databento:
        api_key = os.getenv("DATABENTO_API_KEY", "").strip()
        if api_key:
            client = DatabentoAPIClient(api_key=api_key)
            ingestor = DatabentoIngestor(client=client)
            try:
                ingestion_service = DatabentoIngestionService.from_env()
            except Exception as exc:  # pragma: no cover - runtime warning only
                logging.getLogger(__name__).warning(
                    "Failed to initialise ingestion service: %s",
                    exc,
                )
        elif args.ingest:
            logging.getLogger(__name__).warning(
                "--ingest requested but DATABENTO_API_KEY is missing; skipping",
            )

    from ml.scripts.build_tft_dataset import main as build_main
    from ml.training.teacher.tft_cli import main as teacher_main

    try:
        from ml.cli.hpo_tft import main as _hpo_main

        hpo_main_cli: _CliMain | None = _hpo_main
    except Exception:  # pragma: no cover - optional dependency
        hpo_main_cli = None

    orch = MLPipelineOrchestrator(
        coverage=coverage,
        writer=writer,
        data_registry=registry,
        ingestor=ingestor if ingestor is not None else None,
        build_main=build_main,
        hpo_main=hpo_main_cli,
        teacher_main=teacher_main,
        raw_writer=raw_writer,
        service=ingestion_service,
        model_registry=getattr(mgr, "model_registry", None),
        feature_registry=getattr(mgr, "feature_registry", None),
        strategy_registry=getattr(mgr, "strategy_registry", None),
        feature_store=getattr(mgr, "feature_store", None),
        model_store=getattr(mgr, "model_store", None),
        strategy_store=getattr(mgr, "strategy_store", None),
        data_store=getattr(mgr, "data_store", None),
        partition_manager=getattr(mgr, "partition_manager", None),
        integration_manager_factory=integration_factory,
    )

    if args.ingest and ingestor is not None:
        if ingestion_service is None:
            logging.getLogger(__name__).warning(
                "Ingestion requested but DatabentoIngestionService unavailable; skipping",
            )
        else:
            schema_map = {"bars": "ohlcv-1m", "tbbo": "tbbo", "trades": "trades"}
            provider_schema = schema_map.get(str(args.schema).lower(), str(args.schema))
            instruments = [s.strip() for s in str(args.instruments).split(",") if s.strip()]
            for inst in instruments:
                orch.backfill(
                    dataset_id=args.dataset_id,
                    schema=provider_schema,
                    instrument_id=inst,
                    lookback_days=int(args.lookback_days),
                )
    data_dir_effective = Path(args.data_dir)
    if args.catalog_path and str(args.data_dir) == "data/tier1":
        data_dir_effective = Path(args.catalog_path)

    raw_macro_series_ids = tuple(
        item.strip()
        for item in (str(args.macro_series_ids).split(",") if args.macro_series_ids else [])
        if item.strip()
    )
    macro_series_ids: tuple[str, ...] | None = raw_macro_series_ids or None

    raw_instrument_ids = tuple(
        item.strip()
        for item in (str(args.instrument_ids).split(",") if args.instrument_ids else [])
        if item.strip()
    )
    instrument_ids: tuple[str, ...] | None = raw_instrument_ids or None

    validation_cfg = _build_validation_config_from_args(
        args,
        macro_series_ids,
    )

    auto_fill_enabled = bool(getattr(args, "auto_fill_universe", False))
    auto_fill_blocks_l2 = auto_fill_enabled and not bool(
        getattr(args, "auto_fill_allow_dataset_l2_ingest", False),
    )

    if args.include_l2 and not args.skip_l2_ingest and not auto_fill_blocks_l2:
        l2_symbols = None
        if args.l2_symbols:
            l2_symbols = tuple(s.strip().upper() for s in str(args.l2_symbols).split(",") if s.strip())
        l2_tier = None if l2_symbols else args.l2_tier
        progress_file = (
            Path(args.l2_progress_file)
            if args.l2_progress_file
            else data_dir_effective / ".l2_progress.json"
        )
        try:
            l2_config = PopulateL2TaskConfig(
                data_dir=data_dir_effective,
                progress_file=progress_file,
                symbols=l2_symbols,
                tier=l2_tier,
                days=int(args.l2_days),
            )
            symbols_desc = (
                f"custom:{len(l2_symbols)}" if l2_symbols else f"tier:{l2_tier}"
            )
            logger.info(
                "Starting L2 ingestion (symbols=%s, days=%s)",
                symbols_desc,
                args.l2_days,
            )
            populate_l2_efficient(l2_config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("L2 ingestion failed: %s", exc, exc_info=True)
            raise

    try:
        effective_vintage_policy = VintagePolicy(str(args.vintage_policy))
    except ValueError as exc:
        raise SystemExit(f"Invalid vintage_policy: {args.vintage_policy}") from exc

    market_dataset_id = (
        args.market_dataset_id
        or getattr(args, "auto_fill_dataset_id", None)
        or getattr(args, "dataset_id", "")
    )

    ds_cfg = DatasetBuildConfig(
        data_dir=str(data_dir_effective),
        symbols=str(args.symbols),
        out_dir=str(args.out_dir),
        dataset_id=str(getattr(args, "dataset_id", "tft_dataset")),
        market_dataset_id=str(market_dataset_id) if market_dataset_id else None,
        include_macro=bool(args.include_macro),
        macro_lag_days=int(args.macro_lag_days),
        include_micro=bool(args.include_micro),
        include_l2=bool(args.include_l2),
        include_events=bool(getattr(args, "include_events", False)),
        include_calendar=bool(getattr(args, "include_calendar", False)),
        instrument_ids=instrument_ids,
        auto_refresh_macro=not bool(args.skip_macro_refresh),
        macro_staleness_hours=int(args.macro_freshness_hours),
        macro_series_ids=macro_series_ids,
        macro_fred_path=str(args.macro_fred_path) if args.macro_fred_path else None,
        fred_vintage_dir=str(args.fred_vintage_dir) if args.fred_vintage_dir else None,
        events_dir=str(args.events_dir) if args.events_dir else None,
        student_mode=bool(args.student_mode),
        emit_dataset_events=bool(getattr(args, "emit_dataset_events", False)),
        horizon_minutes=int(args.horizon_minutes),
        threshold=float(args.threshold),
        lookback_periods=int(args.lookback_periods),
        start_iso=args.start_iso,
        end_iso=args.end_iso,
        chunk_days=int(args.chunk_days),
        register_features=bool(args.dataset_register_features),
        feature_registry_dir=args.feature_registry_dir,
        feature_role="teacher",
        validation=validation_cfg,
        vintage_policy=effective_vintage_policy,
        vintage_as_of=args.vintage_as_of,
    )

    auto_fill_cfg = _build_auto_fill_config_from_args(args, ds_cfg)

    hpo_cfg = HPOConfig(
        enabled=bool(args.hpo),
        epochs=int(args.hpo_epochs),
        batch_size=int(args.hpo_batch_size),
        tail_rows=int(args.hpo_tail_rows),
        limit_groups=int(args.hpo_limit_groups),
    )

    teacher_cfg = TeacherTrainConfig(
        enabled=bool(args.train),
        model_id=str(args.teacher_model_id),
        feature_registry_dir=args.feature_registry_dir,
        feature_set_id=args.feature_set_id,
        max_epochs=int(args.max_epochs),
    )

    student_cfg = StudentDistillConfig(
        enabled=bool(args.distill_student),
        model_id=str(args.student_model_id),
        parent_model_id=args.student_parent_model_id,
        model_registry_dir=args.student_model_registry_dir,
        feature_registry_dir=args.student_feature_registry_dir,
        feature_set_id=args.student_feature_set_id,
        objective=str(args.student_objective),
        kd_lambda=float(args.student_kd_lambda),
        early_stopping=int(args.student_early_stopping),
        opset=None if args.student_opset is None else int(args.student_opset),
        use_val_for_distill=bool(args.student_use_val_for_distill),
    )

    promotions_cfg = PromotionsConfig(
        auto_register_model=bool(args.auto_register_model),
        gates_json=args.gates_json,
        auto_promote=bool(args.auto_promote),
        deploy_target=args.deploy_target,
        auto_register_features=bool(args.auto_register_features),
        feature_metrics_json=args.feature_metrics_json,
        refresh_features=bool(args.refresh_features),
    )

    integration_cfg = IntegrationConfig(
        enabled=bool(args.attach_runtime),
        db_connection=(args.runtime_db_connection or args.db),
        auto_start_postgres=bool(args.runtime_auto_start_db),
        auto_migrate=bool(args.runtime_auto_migrate),
        ensure_healthy=not bool(args.runtime_no_ensure_healthy),
        strict_protocol_validation=(
            True if args.runtime_strict_protocol_validation else None
        ),
        run_validators=not bool(args.runtime_skip_validators),
    )

    orchestrator_cfg = OrchestratorConfig(
        dataset=ds_cfg,
        hpo=hpo_cfg,
        teacher=teacher_cfg,
        student=student_cfg,
        promotions=promotions_cfg,
        integration=integration_cfg if integration_cfg.enabled else None,
        auto_fill=auto_fill_cfg if auto_fill_cfg.enabled else None,
    )

    orch.run(orchestrator_cfg)
    return 0


def _build_validation_config_from_args(
    args: argparse.Namespace,
    macro_series_ids: tuple[str, ...] | None,
) -> DatasetValidationConfig | None:
    config = DatasetValidationConfig()
    modified = False
    if args.validation_min_rows is not None:
        config = replace(config, min_rows=int(args.validation_min_rows))
        modified = True
    if args.validation_min_positive_rate is not None:
        config = replace(config, min_positive_rate=float(args.validation_min_positive_rate))
        modified = True
    if args.validation_max_positive_rate is not None:
        config = replace(config, max_positive_rate=float(args.validation_max_positive_rate))
        modified = True
    if args.validation_min_feature_coverage is not None:
        config = replace(
            config,
            min_feature_coverage=float(args.validation_min_feature_coverage),
        )
        modified = True
    if macro_series_ids and config.require_macro_series is None:
        config = replace(config, require_macro_series=macro_series_ids)
        modified = True
    return config if modified else None


def _build_auto_fill_config_from_args(
    args: argparse.Namespace,
    _dataset_cfg: DatasetBuildConfig,
) -> AutoFillUniverseConfig:
    enabled = bool(getattr(args, "auto_fill_universe", False))
    instrument_override: tuple[str, ...] | None = None
    raw_override = getattr(args, "auto_fill_instrument_ids", None)
    if raw_override:
        instrument_override = tuple(
            item.strip()
            for item in str(raw_override).split(",")
            if item.strip()
        )
    dataset_id = str(
        getattr(args, "auto_fill_dataset_id", None)
        or getattr(args, "dataset_id", "EQUS.MINI")
    )
    include_l2 = bool(getattr(args, "include_l2", False)) and not bool(
        getattr(args, "auto_fill_skip_l2", False),
    )
    l2_dataset_id = str(
        getattr(args, "auto_fill_l2_dataset_id", None) or "DBEQ.BASIC"
    )
    l2_schema = str(
        getattr(args, "auto_fill_l2_schema", None) or "mbp-10"
    )
    l2_days_raw = getattr(args, "auto_fill_l2_days", None)
    l2_days = int(l2_days_raw) if l2_days_raw is not None else None
    l2_progress_file_raw = getattr(args, "auto_fill_l2_progress_file", None)
    l2_progress_file = (
        str(l2_progress_file_raw) if l2_progress_file_raw else None
    )
    allow_dataset_l2 = bool(getattr(args, "auto_fill_allow_dataset_l2_ingest", False))
    include_l3 = bool(getattr(args, "auto_fill_include_l3", False))
    l3_dataset_id_raw = getattr(args, "auto_fill_l3_dataset_id", None)
    l3_schema_raw = getattr(args, "auto_fill_l3_schema", None)
    l3_days_raw = getattr(args, "auto_fill_l3_days", None)
    l3_days = int(l3_days_raw) if l3_days_raw is not None else None

    return AutoFillUniverseConfig(
        enabled=enabled,
        dataset_id=dataset_id,
        include_bars=True,
        include_tbbo=True,
        include_trades=True,
        include_l2=include_l2,
        include_l3=include_l3,
        l2_dataset_id=l2_dataset_id,
        l2_schema=l2_schema,
        l2_days=l2_days,
        l2_progress_file=l2_progress_file,
        disable_dataset_l2_ingest=not allow_dataset_l2,
        instrument_ids=instrument_override,
        l3_dataset_id=str(l3_dataset_id_raw) if l3_dataset_id_raw else None,
        l3_schema=str(l3_schema_raw) if l3_schema_raw else None,
        l3_days=l3_days,
    )
