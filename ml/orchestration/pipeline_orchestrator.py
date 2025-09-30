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
import sys
import time
import uuid as _uuid
from calendar import monthrange
from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import UTC
from datetime import date
from datetime import datetime
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from ml.common.db_connections import ConnectionRole as _DbConnectionRole
from ml.common.db_connections import collect_postgres_candidates as _collect_db_candidates
from ml.common.logging_config import bind_log_context
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.coverage import CoveragePolicy
from ml.config.coverage import get_max_lookback_days
from ml.config.market_data import MarketDatasetInput
from ml.config.market_data import coerce_storage_kind
from ml.config.market_data import load_market_feed_descriptors as _load_market_feed_descriptors
from ml.data import DatasetMetadata
from ml.data import DatasetMetadataExpectations
from ml.data import DatasetValidationConfig
from ml.data import compute_dataset_pipeline_signature
from ml.data import load_dataset_metadata
from ml.data import validate_dataset_metadata_expectations
from ml.data.dataset_manifest_defaults import build_auto_dataset_manifest
from ml.data.ingest.databento_adapter import DatabentoAPIClient
from ml.data.ingest.discovery import DatasetDiscoveryError
from ml.data.ingest.discovery import DatasetDiscoveryService
from ml.data.ingest.discovery import DiscoveryPolicy
from ml.data.ingest.discovery import DiscoveryRequest
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.orchestrator import BackfillWindowList
from ml.data.ingest.orchestrator import DomainWindowLoaderProtocol
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionError
from ml.data.ingest.service import SymbolDatasetDiscovery
from ml.data.ingest.symbology import DatabentoSymbologyResolver
from ml.data.vintage import VintagePolicy
from ml.data.vintage import format_dt
from ml.data.vintage import parse_dt
from ml.orchestration.config_loader import IngestionStageConfig
from ml.orchestration.config_loader import Stage
from ml.orchestration.config_loader import load_orchestrator_run_config
from ml.orchestration.config_loader import to_pipeline_args
from ml.orchestration.config_types import DEFAULT_LOOKBACK_YEARS
from ml.orchestration.config_types import DEFAULT_MACRO_SERIES
from ml.orchestration.config_types import AutoFillUniverseConfig
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import PreIngestionOptions
from ml.orchestration.config_types import PromotionsConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
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
from ml.stores.providers import SqlMarketDataWriter
from ml.stores.writers import DataStoreMarketDataWriter
from ml.stores.writers import FanoutMarketDataWriter
from ml.stores.writers import ParquetCatalogMarketDataWriter
from ml.tasks.ingest import PopulateL2TaskConfig
from ml.tasks.ingest import populate_l2_efficient


load_market_feed_descriptors = _load_market_feed_descriptors


logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ml.config.scheduler_config import SchedulerConfig


@lru_cache(maxsize=1)
def _get_allowed_databento_datasets() -> frozenset[str] | None:
    try:
        from ml.config.databento_policy import load_databento_safety_config

        cfg = load_databento_safety_config(None)
        datasets = cfg.datasets if hasattr(cfg, "datasets") else None
        return frozenset(datasets) if datasets else None
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("databento_safety_config_unavailable", exc_info=True)
        return None


_WRITE_MODE_TOKEN_MAP: Final[dict[str, tuple[str, ...]]] = {
    "datastore": ("datastore",),
    "parquet": ("datastore", "parquet"),
    "datastore+parquet": ("datastore", "parquet"),
    "sql": ("sql",),
    "sql+datastore": ("sql", "datastore"),
    "sql+parquet": ("sql", "parquet"),
    "sql+datastore+parquet": ("sql", "datastore", "parquet"),
}

_WRITE_MODE_ALLOWED_TOKENS: Final[frozenset[str]] = frozenset({"sql", "datastore", "parquet"})

_SCHEMA_ALIASES: Final[dict[str, str]] = {
    "bars": "ohlcv-1m",
    "ohlcv": "ohlcv-1m",
    "tbbo": "tbbo",
    "quotes": "tbbo",
    "trades": "trades",
}


def _resolve_write_mode_tokens(raw_mode: str) -> tuple[str, ...]:
    """
    Normalize write-mode token strings to ordered mode tuples.
    """
    normalized = raw_mode.strip().lower()
    mapped = _WRITE_MODE_TOKEN_MAP.get(normalized)
    if mapped is not None:
        return mapped
    if normalized:
        tokens = tuple(token for token in normalized.split("+") if token)
        if tokens:
            invalid = [token for token in tokens if token not in _WRITE_MODE_ALLOWED_TOKENS]
            if invalid:
                raise SystemExit(
                    f"Unsupported write_mode tokens {invalid}; allowed tokens are "
                    f"{sorted(_WRITE_MODE_ALLOWED_TOKENS)}",
                )
            ordered = tuple(dict.fromkeys(tokens))
            return ordered
    raise SystemExit(f"Unsupported write_mode '{raw_mode}'")


def _apply_default_market_inputs(cfg: DatasetBuildConfig) -> DatasetBuildConfig:
    """
    Seed dataset configs with descriptor-driven market inputs when ``market_dataset_id``
    is explicitly provided.
    """
    if cfg.market_inputs or not cfg.market_dataset_id:
        return cfg

    descriptors = load_market_feed_descriptors().as_mapping()
    descriptor = descriptors.get(cfg.market_dataset_id)

    if descriptor is None:
        return cfg

    symbols: list[str] = []
    for raw_symbol in str(cfg.symbols).split(","):
        token = raw_symbol.strip().upper()
        if not token:
            continue
        base = token.split(".", maxsplit=1)[0]
        if base and base not in symbols:
            symbols.append(base)
    if not symbols:
        return cfg

    inputs = tuple(
        MarketDatasetInput(
            descriptor_id=descriptor.descriptor_id,
            dataset_id=descriptor.dataset_id,
            symbols=(symbol,),
            schema_override=descriptor.schema,
            storage_kind_override=descriptor.storage_kind,
        )
        for symbol in symbols
    )

    return replace(
        cfg,
        market_inputs=inputs,
        market_dataset_id=cfg.market_dataset_id,
    )


def _collect_symbol_map(
    *,
    ds_cfg: DatasetBuildConfig | None,
    ingestion_cfg: IngestionStageConfig,
) -> dict[str, tuple[str, ...]]:
    symbol_to_instruments: dict[str, list[str]] = {}

    def _register(symbol: str, instrument_id: str | None = None) -> None:
        symbol_norm = symbol.strip().upper()
        if not symbol_norm:
            return
        bucket = symbol_to_instruments.setdefault(symbol_norm, [])
        if instrument_id is None:
            return
        inst_norm = instrument_id.strip().upper()
        if inst_norm and inst_norm not in bucket:
            bucket.append(inst_norm)

    def _extract_symbol(token: str) -> str:
        stripped = token.strip()
        if not stripped:
            return ""
        upper = stripped.upper()
        if "." in upper:
            return upper.split(".")[0]
        return upper

    for symbol in ingestion_cfg.symbols or ():
        symbol_to_instruments.setdefault(symbol.strip().upper(), [])
    for instrument in ingestion_cfg.instruments:
        base = _extract_symbol(instrument)
        if base:
            _register(base, instrument)
    for instrument in ingestion_cfg.instrument_ids or ():
        base = _extract_symbol(instrument)
        if base:
            _register(base, instrument)

    if ds_cfg is not None:
        for raw_symbol in str(ds_cfg.symbols).split(","):
            symbol_norm = raw_symbol.strip().upper()
            if symbol_norm:
                symbol_to_instruments.setdefault(symbol_norm, [])
        for instrument in ds_cfg.instrument_ids or ():
            base = _extract_symbol(instrument)
            if base:
                _register(base, instrument)

    market_inputs = ingestion_cfg.market_inputs
    if market_inputs is None and ds_cfg is not None:
        market_inputs = ds_cfg.market_inputs
    if market_inputs:
        for item in market_inputs:
            for symbol in item.symbols or ():
                symbol_to_instruments.setdefault(symbol.strip().upper(), [])

    if not symbol_to_instruments:
        for instrument in ingestion_cfg.instruments:
            base = _extract_symbol(instrument)
            if base:
                _register(base, instrument)

    return {symbol: tuple(values) for symbol, values in symbol_to_instruments.items()}


def _compute_window_start_iso(*, end_iso: str, lookback_years: int = DEFAULT_LOOKBACK_YEARS) -> str:
    """
    Compute ISO8601 start date by subtracting ``lookback_years`` from ``end_iso``.
    """
    end_date = date.fromisoformat(end_iso)
    target_year = end_date.year - lookback_years
    days_in_month = monthrange(target_year, end_date.month)[1]
    day = min(end_date.day, days_in_month)
    start_date = date(target_year, end_date.month, day)
    return start_date.isoformat()


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
    """
    Raised when the dataset build produces zero rows.
    """

    def __init__(self, message: str, *, row_count: int | None = None) -> None:
        super().__init__(message)
        self.row_count = row_count


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
class _IngestionMetrics:
    """
    Instrumentation bundle for ingestion-stage bookkeeping.
    """

    runs_total: Any
    latency_seconds: Any
    fallback_total: Any

    @staticmethod
    def default() -> _IngestionMetrics:
        """
        Initialise lazily to ensure metrics bootstrap occurs once per process.
        """
        return _IngestionMetrics(
            runs_total=get_counter(
                "nautilus_ml_ingestion_stage_runs_total",
                "Pipeline ingestion stage executions",
                labelnames=("component", "status"),
            ),
            latency_seconds=get_histogram(
                "nautilus_ml_ingestion_stage_latency_seconds",
                "Pipeline ingestion stage latency",
                labelnames=("component", "status"),
                buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            ),
            fallback_total=get_counter(
                "ml_fallback_activations_total",
                "Fallback activations",
                labelnames=("component", "level"),
            ),
        )


@dataclass(slots=True, frozen=True)
class _IngestionAttemptReport:
    """
    Structured outcome for an ingestion attempt.
    """

    success: bool
    context: dict[str, object]
    reason: str | None = None


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
    domain_loader: DomainWindowLoaderProtocol | None = None
    integration_manager_factory: Callable[..., IntegrationManagerProtocol] | None = None
    dataset_discovery: DatasetDiscoveryService | None = None
    write_mode_tokens: tuple[str, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
    )
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
        """
        Best-effort row count inference for API build results.
        """
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

    def _prepare_dataset_config(self, cfg: DatasetBuildConfig) -> DatasetBuildConfig:
        base_cfg = _apply_default_market_inputs(cfg)
        resolved_inputs, bindings = self._resolve_market_inputs(base_cfg)
        if resolved_inputs:
            instrument_ids = self._collect_instrument_ids(bindings, base_cfg.instrument_ids)
            base_cfg = replace(
                base_cfg,
                market_inputs=resolved_inputs,
                instrument_ids=instrument_ids,
            )
        logger.info(
            "Dataset config prepared",
            extra={
                "symbols": base_cfg.symbols,
                "instrument_ids": base_cfg.instrument_ids,
                "market_inputs": (
                    0 if base_cfg.market_inputs is None else len(base_cfg.market_inputs)
                ),
            },
        )
        return base_cfg

    def _resolve_market_inputs(
        self,
        cfg: DatasetBuildConfig,
    ) -> tuple[tuple[MarketDatasetInput, ...] | None, tuple[ResolvedMarketBinding, ...]]:
        coverage = self.coverage
        if coverage is None:
            return None, ()

        symbol_map = self._symbol_to_instruments(cfg)
        if not symbol_map:
            return None, ()

        try:
            start_ns, end_ns = self._resolve_window_bounds_ns(cfg)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Unable to resolve coverage window; skipping market input resolution")
            return None, ()

        default_schema = self._infer_default_schema(cfg)
        effective_inputs: tuple[MarketDatasetInput, ...] | None = cfg.market_inputs
        if (not effective_inputs) and self.dataset_discovery is not None:
            discovery_inputs = self._discover_market_inputs(
                symbol_map=symbol_map,
                schema=default_schema,
                start_ns=start_ns,
                end_ns=end_ns,
                dataset_hint=cfg.market_dataset_id,
            )
            if discovery_inputs:
                logger.info(
                    "Dataset discovery inputs applied",
                    extra={
                        "stage": Stage.DATASET.value,
                        "symbol_count": len(discovery_inputs),
                    },
                )
                effective_inputs = discovery_inputs

        resolved_inputs: list[MarketDatasetInput] = []
        resolved_bindings: list[ResolvedMarketBinding] = []
        for symbol, instrument_ids in symbol_map.items():
            candidates = IngestionOrchestrator.resolve_market_bindings(
                symbols=[symbol],
                instrument_ids=instrument_ids or None,
                market_dataset_id=cfg.market_dataset_id,
                market_inputs=effective_inputs,
            )
            binding: ResolvedMarketBinding | None
            if candidates:
                candidates = self._filter_candidate_bindings(
                    candidates,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    symbol=symbol,
                    default_schema=default_schema,
                )
                if candidates:
                    binding = self._select_binding_with_coverage(
                        candidates=candidates,
                        start_ns=start_ns,
                        end_ns=end_ns,
                    )
                else:
                    binding = None
                if binding is None and candidates:
                    discovered = self._discover_binding_for_symbol(
                        symbol=symbol,
                        instrument_ids=instrument_ids or None,
                        schema=default_schema,
                        start_ns=start_ns,
                        end_ns=end_ns,
                    )
                    if discovered is not None:
                        candidates = (discovered,)
                        binding = discovered
            else:
                binding = self._discover_binding_for_symbol(
                    symbol=symbol,
                    instrument_ids=instrument_ids or None,
                    schema=default_schema,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
                if binding is None:
                    logger.warning(
                        "No binding resolved for symbol",
                        extra={"symbol": symbol, "market_dataset_id": cfg.market_dataset_id},
                    )
                    continue
                candidates = (binding,)
            if binding is not None and not self._binding_allowed(
                binding=binding,
                start_ns=start_ns,
                end_ns=end_ns,
                symbol=symbol,
                default_schema=default_schema,
            ):
                logger.info(
                    "Binding rejected after validation",
                    extra={
                        "dataset_id": binding.dataset_id,
                        "schema": binding.schema,
                        "symbol": symbol,
                    },
                )
                continue
            if binding is None:
                continue
            resolved_inputs.append(
                MarketDatasetInput(
                    descriptor_id=binding.descriptor_id,
                    dataset_id=binding.dataset_id,
                    symbols=(symbol,),
                    schema_override=binding.schema or default_schema,
                    storage_kind_override=binding.storage_kind,
                ),
            )
            resolved_bindings.append(binding)
            logger.info(
                "Binding selected",
                extra={
                    "symbol": symbol,
                    "dataset_id": binding.dataset_id,
                    "schema": binding.schema or default_schema,
                    "instrument_ids": binding.instrument_ids,
                    "source": binding.source,
                },
            )

        if not resolved_inputs:
            return None, ()
        return tuple(resolved_inputs), tuple(resolved_bindings)

    def _discover_market_inputs(
        self,
        *,
        symbol_map: Mapping[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None,
    ) -> tuple[MarketDatasetInput, ...]:
        service = self.dataset_discovery
        if service is None or start_ns >= end_ns:
            return ()
        start_dt = self._ns_to_datetime(start_ns)
        end_dt = self._ns_to_datetime(end_ns)
        requests = tuple(
            DiscoveryRequest(
                symbol=symbol,
                schema=schema,
                start=start_dt,
                end=end_dt,
            )
            for symbol in symbol_map
        )
        if not requests:
            return ()
        try:
            inputs = service.discover(requests=requests, dataset_hint=dataset_hint)
            coverage_policy = None
            try:
                coverage_policy = service.policy.coverage
            except AttributeError:
                coverage_policy = None
            if coverage_policy is not None:
                for market_input in inputs:
                    coverage_policy.allow_dataset(market_input.dataset_id or "")
            return inputs
        except DatasetDiscoveryError as exc:
            logger.warning(
                "Dataset discovery unavailable",
                extra={
                    "stage": Stage.DATASET.value,
                    "reason": str(exc),
                    "symbol_count": len(requests),
                },
            )
            return ()

    @staticmethod
    def _infer_default_schema(cfg: DatasetBuildConfig) -> str:
        """
        Infer a reasonable default schema for discovery lookups.
        """
        return "ohlcv-1m"

    @staticmethod
    def _ns_to_datetime(value: int) -> datetime:
        """
        Convert nanoseconds since epoch to an aware UTC datetime.
        """
        seconds = value / 1_000_000_000
        return datetime.fromtimestamp(seconds, tz=UTC)

    def _symbol_to_instruments(
        self,
        cfg: DatasetBuildConfig,
    ) -> OrderedDict[str, tuple[str, ...]]:
        symbols: OrderedDict[str, None] = OrderedDict()
        raw_symbols = str(cfg.symbols or "").split(",")
        for raw in raw_symbols:
            token = raw.strip()
            if not token:
                continue
            symbols.setdefault(token.split(".")[0].upper(), None)

        instrument_mapping: dict[str, list[str]] = {}
        for inst in cfg.instrument_ids or ():
            token = inst.strip()
            if not token:
                continue
            upper = token.upper()
            base = upper.split(".")[0]
            instrument_mapping.setdefault(base, []).append(upper)
            symbols.setdefault(base, None)

        ordered: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        for symbol in symbols.keys():
            ordered[symbol] = tuple(instrument_mapping.get(symbol, ()))
        return ordered

    @staticmethod
    def _collect_instrument_ids(
        bindings: tuple[ResolvedMarketBinding, ...],
        existing: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        collected: OrderedDict[str, None] = OrderedDict()
        if existing:
            for inst in existing:
                token = inst.strip()
                if token:
                    collected.setdefault(token.upper(), None)

        for binding in bindings:
            for inst in binding.instrument_ids or (binding.symbol,):
                token = inst.strip().upper()
                if token:
                    collected.setdefault(token, None)

        return tuple(collected.keys())

    def _filter_candidate_bindings(
        self,
        candidates: tuple[ResolvedMarketBinding, ...],
        *,
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> tuple[ResolvedMarketBinding, ...]:
        if not candidates:
            return ()
        filtered: list[ResolvedMarketBinding] = []
        for binding in candidates:
            if self._binding_allowed(
                binding=binding,
                start_ns=start_ns,
                end_ns=end_ns,
                symbol=symbol,
                default_schema=default_schema,
            ):
                filtered.append(binding)
        if filtered:
            filtered.sort(key=self._binding_priority_key)
        return tuple(filtered)

    @staticmethod
    def _binding_priority_key(binding: ResolvedMarketBinding) -> tuple[int, str]:
        dataset_id = binding.dataset_id.upper()
        if dataset_id == "EQUS.MINI":
            return (0, dataset_id)
        if dataset_id == "XNAS.ITCH":
            return (1, dataset_id)
        return (2, dataset_id)

    def _binding_allowed(
        self,
        *,
        binding: ResolvedMarketBinding,
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> bool:
        service = self.service
        schema = binding.schema or default_schema
        if not schema:
            return False

        if service is not None and binding.dataset_id:
            try:
                available_start_ns, available_end_ns = service.get_available_range_ns(
                    dataset=binding.dataset_id,
                    schema=schema,
                )
            except IngestionError as exc:
                logger.info(
                    "Binding rejected by ingestion service",
                    extra={
                        "dataset_id": binding.dataset_id,
                        "schema": schema,
                        "symbol": symbol,
                        "reason": str(exc),
                    },
                )
                return False
            except Exception:  # pragma: no cover - defensive guard
                logger.debug(
                    "Binding availability check failed",
                    exc_info=True,
                    extra={
                        "dataset_id": binding.dataset_id,
                        "schema": schema,
                        "symbol": symbol,
                    },
                )
            else:
                if available_start_ns is not None and end_ns <= available_start_ns:
                    logger.info(
                        "Binding outside provider coverage",
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "symbol": symbol,
                            "available_start_ns": available_start_ns,
                            "requested_end_ns": end_ns,
                        },
                    )
                    return False
                if available_end_ns is not None and start_ns >= available_end_ns:
                    logger.info(
                        "Binding outside provider coverage",
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "symbol": symbol,
                            "available_end_ns": available_end_ns,
                            "requested_start_ns": start_ns,
                        },
                    )
                    return False
                try:
                    cost_usd = service.estimate_cost_usd(
                        dataset=binding.dataset_id,
                        schema=schema,
                        symbols=(symbol,),
                        start=self._ns_to_datetime(start_ns),
                        end=self._ns_to_datetime(end_ns),
                    )
                except IngestionError as exc:
                    logger.info(
                        "Binding rejected by cost policy",
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "symbol": symbol,
                            "reason": str(exc),
                        },
                    )
                    return False
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug(
                        "Binding cost estimation failed",
                        exc_info=True,
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "symbol": symbol,
                        },
                    )
                else:
                    if cost_usd > 0.0:
                        logger.info(
                            "Binding rejected due to non-zero cost",
                            extra={
                                "dataset_id": binding.dataset_id,
                                "schema": schema,
                                "symbol": symbol,
                                "cost_usd": cost_usd,
                            },
                        )
                        return False

        return True

    def _select_binding_with_coverage(
        self,
        *,
        candidates: tuple[ResolvedMarketBinding, ...],
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None:
        coverage = self.coverage
        if coverage is None:
            return None

        for binding in candidates:
            schema = binding.schema or ""
            if not schema:
                continue
            instruments = binding.instrument_ids or (binding.symbol,)
            for instrument in instruments:
                try:
                    buckets = coverage.read_bucket_coverage(
                        dataset_id=binding.dataset_id,
                        schema=schema,
                        instrument_id=instrument,
                        start_ns=start_ns,
                        end_ns=end_ns,
                    )
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug(
                        "Coverage lookup failed",
                        exc_info=True,
                        extra={
                            "dataset_id": binding.dataset_id,
                            "schema": schema,
                            "instrument_id": instrument,
                        },
                    )
                    buckets = set()
                if buckets:
                    return binding
        return None

    def _discover_binding_for_symbol(
        self,
        *,
        symbol: str,
        instrument_ids: tuple[str, ...] | None,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None:
        service = self.service
        if service is None:
            return None

        schema_token = schema.strip()
        if not schema_token:
            return None

        discovery_func = getattr(service, "discover_symbol_dataset", None)
        dataset_service = self.dataset_discovery
        if (discovery_func is None or not callable(discovery_func)) and dataset_service is not None:
            def _dataset_service_wrapper(
                *,
                symbol: str,
                schema: str,
                start_ns: int,
                end_ns: int,
            ) -> SymbolDatasetDiscovery | None:
                return self._discover_symbol_via_dataset_service(
                    dataset_service=dataset_service,
                    symbol=symbol,
                    schema=schema,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )

            discovery_func = _dataset_service_wrapper

        if discovery_func is None or not callable(discovery_func):
            return None

        try:
            discovery = discovery_func(
                symbol=symbol,
                schema=schema_token,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Dataset discovery failed",
                exc_info=True,
                extra={
                    "symbol": symbol,
                    "schema": schema_token,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                },
            )
            return None

        if discovery is None:
            return None

        resolved_symbol = getattr(discovery, "symbol", symbol)
        instrument_tuple = instrument_ids or (resolved_symbol,)
        binding_id = f"discovered:{discovery.dataset_id}:{resolved_symbol}"
        return ResolvedMarketBinding(
            binding_id=binding_id,
            symbol=resolved_symbol,
            instrument_ids=tuple(instrument_tuple),
            dataset_id=discovery.dataset_id,
            descriptor_id=None,
            schema=discovery.schema,
            storage_kind=discovery.storage_kind,
            license_start=None,
            license_end=None,
            start=None,
            end=None,
            source="discovered",
        )

    def _discover_symbol_via_dataset_service(
        self,
        *,
        dataset_service: DatasetDiscoveryService,
        symbol: str,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> SymbolDatasetDiscovery | None:
        if start_ns >= end_ns:
            return None
        request = DiscoveryRequest(
            symbol=symbol,
            schema=schema,
            start=self._ns_to_datetime(start_ns),
            end=self._ns_to_datetime(end_ns),
        )
        try:
            discovered = dataset_service.discover_one(request=request)
        except DatasetDiscoveryError as exc:
            logger.debug(
                "Dataset discovery service rejected symbol",
                extra={
                    "symbol": symbol,
                    "schema": schema,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "reason": str(exc),
                },
            )
            return None
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Dataset discovery service failed",
                exc_info=True,
                extra={
                    "symbol": symbol,
                    "schema": schema,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                },
            )
            return None

        storage_kind = discovered.storage_kind or StorageKind.POSTGRES
        resolved_symbol = discovered.symbol or symbol
        return SymbolDatasetDiscovery(
            dataset_id=discovered.dataset_id,
            schema=discovered.schema,
            storage_kind=storage_kind,
            symbol=resolved_symbol,
            requested_symbol=discovered.requested_symbol,
            available_start_ns=discovered.available_start_ns,
            available_end_ns=discovered.available_end_ns,
            cost_usd=discovered.cost_usd,
            instrument_id=discovered.instrument_id,
        )

    def _resolve_window_bounds_ns(self, cfg: DatasetBuildConfig) -> tuple[int, int]:
        end_dt = parse_dt(cfg.end_iso) if cfg.end_iso else None
        if end_dt is None:
            end_dt = datetime.now(tz=UTC)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)
        start_iso = cfg.start_iso
        if start_iso is None:
            start_iso = _compute_window_start_iso(end_iso=end_dt.date().isoformat())
        start_dt = parse_dt(start_iso)
        if start_dt is None:
            start_dt = datetime.fromisoformat(start_iso).replace(tzinfo=UTC)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)
        if end_ns <= start_ns:
            end_ns = start_ns + DAY_NS
        return start_ns, end_ns

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
                    fallback_binding = self._discover_binding_for_symbol(
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
        storage_kind: StorageKind | None = None,
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

        # Determine storage_kind based on write_mode if not provided
        if storage_kind is None:
            write_mode_tokens = getattr(self, "write_mode_tokens", ())
            if "sql" in write_mode_tokens:
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

    def _create_ingestion_orchestrator(self) -> IngestionOrchestrator:
        if self.ingestor is None:
            raise RuntimeError("Ingestor is not configured for pipeline orchestrator")
        return IngestionOrchestrator(
            coverage=self.coverage,
            writer=self.writer,
            registry=cast(RegistryProtocol, self.data_registry),
            ingestor=cast(DatabentoIngestor, self.ingestor),
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
                dataset_id=cfg.dataset_id,
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
            row_info = f" rows={empty_exc.row_count}" if empty_exc.row_count is not None else ""
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
            logger.warning("API dataset build failed; falling back to CLI: %s", exc, exc_info=True)

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
        if cfg.market_dataset_id:
            args += ["--market_dataset_id", cfg.market_dataset_id]
        if cfg.market_inputs:
            inputs_payload: list[object] = []
            for item in cfg.market_inputs:
                entry: dict[str, object] = {}
                if item.descriptor_id is not None:
                    entry["descriptor_id"] = item.descriptor_id
                if item.dataset_id is not None:
                    entry["dataset_id"] = item.dataset_id
                if item.symbols is not None:
                    entry["symbols"] = list(item.symbols)
                if item.schema_override is not None:
                    entry["schema"] = item.schema_override
                if item.storage_kind_override is not None:
                    entry["storage_kind"] = item.storage_kind_override.value
                if item.start is not None:
                    entry["start"] = item.start
                if item.end is not None:
                    entry["end"] = item.end
                inputs_payload.append(entry or (item.descriptor_id or item.dataset_id or ""))
            args += ["--market_inputs_json", json.dumps(inputs_payload)]
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
        """
        Export a feature manifest when registry configuration is provided.
        """
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

        data_requirements = DataRequirements.L1_L2 if cfg.include_l2 else DataRequirements.L1_ONLY
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
        """
        Validate dataset metadata against configuration guardrails.
        """

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
        if metadata.market_bindings:
            for binding in metadata.market_bindings:
                if (binding.dataset_id or "").upper() != "EQUS.MINI":
                    continue
                if not binding.source_datasets or not binding.aggregation_modes:
                    raise ValueError(
                        "EQUS.MINI metadata missing provenance fields (source_datasets/aggregation_modes)",
                    )
                if "scaled_volume" in binding.aggregation_modes and not binding.scaling_factors:
                    raise ValueError(
                        "EQUS.MINI scaling fallback lacks recorded scaling_factors",
                    )

    @staticmethod
    def _compute_dataset_pipeline_signature(
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> str:
        """
        Derive a stable pipeline signature covering vintage policy and scope.
        """
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
            market_bindings=metadata.market_bindings,
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
                "market_bindings": [
                    {
                        "binding_id": binding.binding_id,
                        "dataset_id": binding.dataset_id,
                        "descriptor_id": binding.descriptor_id,
                        "source": binding.source,
                        "storage_kind": binding.storage_kind,
                        "symbols": list(binding.symbols),
                        "instrument_ids": list(binding.instrument_ids),
                    }
                    for binding in (metadata.market_bindings or ())
                ],
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
            logger.error(
                "Distillation enabled but missing teacher predictions NPZ at %s", teacher_npz
            )
            return 1

        artifacts = self._build_artifacts
        feature_registry_dir = cfg.feature_registry_dir or (
            artifacts.feature_registry_dir if artifacts else None
        )
        feature_set_id = cfg.feature_set_id or (artifacts.feature_set_id if artifacts else None)
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
        dataset_cfg = self._prepare_dataset_config(cfg.dataset)
        cfg = replace(cfg, dataset=dataset_cfg)

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

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        """
        Run HPO, teacher training, and student distillation without rebuilding data.
        """
        dataset_cfg = self._prepare_dataset_config(cfg.dataset)
        cfg = replace(cfg, dataset=dataset_cfg)

        dataset_dir = Path(dataset_cfg.out_dir)
        dataset_csv = dataset_dir / "dataset.csv"
        if not dataset_csv.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found at {dataset_csv}; run dataset stage first",
            )

        metadata_path = dataset_dir / "dataset_metadata.json"
        dataset_metadata = load_dataset_metadata(metadata_path)

        feature_registry_dir = (
            cfg.teacher.feature_registry_dir
            or dataset_cfg.feature_registry_dir
            or (self._build_artifacts.feature_registry_dir if self._build_artifacts else None)
        )
        feature_set_id = (
            cfg.teacher.feature_set_id
            or getattr(dataset_metadata, "feature_set_id", None)
            or (self._build_artifacts.feature_set_id if self._build_artifacts else None)
        )

        self._build_artifacts = BuildArtifacts(
            out_dir=dataset_dir,
            feature_registry_dir=feature_registry_dir,
            feature_set_id=feature_set_id,
            dataset_metadata=dataset_metadata,
        )

        rc = self.run_hpo(cfg.hpo, dataset_csv=dataset_csv, out_dir=dataset_dir)
        if rc != 0:
            return rc

        rc = self.train_teacher(cfg.teacher, dataset_csv=dataset_csv, out_dir=dataset_dir)
        if rc != 0:
            return rc

        rc = self.distill_student(cfg.student, dataset_dir=dataset_dir, teacher_cfg=cfg.teacher)
        if rc != 0:
            return rc

        self._handle_promotions(cfg.promotions, out_dir=dataset_dir, dataset_csv=dataset_csv)
        self._attach_runtime(cfg.integration, dataset_out_dir=dataset_dir)
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
                            str(comparison_value) if comparison_value is not None else "gte"
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    import os

    parser = argparse.ArgumentParser(description="Run end-to-end ML pipeline (cold path)")
    parser.add_argument("--config", default=None, help="Path to orchestrator JSON/TOML config")
    parser.add_argument(
        "--stage",
        default=None,
        choices=[member.value for member in Stage],
        help="Pipeline stage to run (ingest, dataset, train, full)",
    )

    # Ingestion/backfill
    parser.add_argument("--ingest", action="store_true", help="Run ingestion backfill first")
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--schema", default="bars", choices=["bars", "tbbo", "trades"])
    parser.add_argument("--instruments", default="SPY.NYSE")
    parser.add_argument("--lookback_days", type=int, default=7)
    parser.add_argument("--coverage_mode", default="catalog", choices=["catalog", "sql"])
    parser.add_argument("--catalog_path", default=os.getenv("CATALOG_PATH", ""))
    default_db_candidates = _collect_db_candidates(_DbConnectionRole.PRIMARY)
    default_db_url = (
        default_db_candidates.urls[0]
        if default_db_candidates.urls
        else "postgresql://postgres:postgres@localhost:5433/nautilus"
    )
    parser.add_argument(
        "--db",
        default=default_db_url,
    )

    # Writer mode for ingestion
    parser.add_argument(
        "--write_mode",
        default="parquet",
        choices=tuple(sorted(_WRITE_MODE_TOKEN_MAP.keys())),
        help=(
            "Ingestion writer fanout: parquet (DataStore+Parquet), datastore, sql, "
            "sql+datastore, sql+parquet, or sql+datastore+parquet"
        ),
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
        help="Identifier for the canonical market data dataset (defaults to auto-fill dataset when provided)",
    )
    parser.add_argument(
        "--market_inputs_json",
        default=None,
        help="JSON payload describing market feed inputs",
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


def _extract_config_args(
    raw_args: Sequence[str],
) -> tuple[str | None, str | None, list[str]]:
    """
    Split ``raw_args`` into config path, stage override, and remaining args.
    """
    config_path: str | None = None
    stage_override: str | None = None
    passthrough: list[str] = []
    idx = 0
    while idx < len(raw_args):
        token = raw_args[idx]
        if token == "--config":
            if idx + 1 >= len(raw_args):
                raise SystemExit("--config requires a file path")
            config_path = raw_args[idx + 1]
            idx += 2
            continue
        if token.startswith("--config="):
            config_path = token.split("=", 1)[1]
            idx += 1
            continue
        if token == "--stage":
            if idx + 1 >= len(raw_args):
                raise SystemExit("--stage requires a value")
            stage_override = raw_args[idx + 1]
            idx += 2
            continue
        if token.startswith("--stage="):
            stage_override = token.split("=", 1)[1]
            idx += 1
            continue
        passthrough.append(token)
        idx += 1
    return config_path, stage_override, passthrough


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    config_path, stage_override, passthrough = _extract_config_args(raw_args)
    stage_default: Stage | None = None

    if config_path is not None:
        run_cfg = load_orchestrator_run_config(config_path)
        stage_default = run_cfg.stage
        if stage_override is not None:
            try:
                stage_for_args = Stage(stage_override)
            except ValueError as exc:  # pragma: no cover - defensive
                raise SystemExit(f"Unsupported stage '{stage_override}'") from exc
        else:
            stage_for_args = run_cfg.stage
        ingestion_cfg = run_cfg.ingestion if stage_for_args in {Stage.FULL, Stage.INGEST} else None
        config_args: list[str]
        if run_cfg.dataset is None:
            if stage_for_args is not Stage.INGEST:
                raise SystemExit("Dataset configuration is required for non-ingestion stages")
            effective_ingestion = ingestion_cfg or IngestionStageConfig(enabled=True)
            config_args = _ingestion_config_to_args(effective_ingestion)
        else:
            orchestrator_cfg = run_cfg.compose_orchestrator_config()
            config_args = to_pipeline_args(orchestrator_cfg, ingestion=ingestion_cfg)
        combined_args = config_args + passthrough
        if stage_override is not None:
            combined_args += ["--stage", stage_override]
        args = parse_args(combined_args)
        if args.stage is None:
            args.stage = stage_for_args.value
    else:
        if stage_override is not None:
            passthrough += ["--stage", stage_override]
        args = parse_args(passthrough)

    return _execute_with_namespace(args, stage_default=stage_default)


def _execute_with_namespace(
    args: argparse.Namespace,
    *,
    stage_default: Stage | None = None,
) -> int:
    _run_id: str = f"orch_{_uuid.uuid4().hex[:12]}"
    bind_log_context(run_id=_run_id, component="ml.pipeline_orchestrator")

    from ml.core.integration import MLIntegrationManager

    mgr = MLIntegrationManager(
        db_connection=args.db,
        auto_start_postgres=False,
        auto_migrate=True,
        ensure_healthy=False,
    )
    data_store = getattr(mgr, "data_store", None)
    if data_store is None:
        logger.info(
            "DataStore unavailable; falling back to catalog-only runtime attachment",
        )
    if mgr.data_registry is None:
        raise SystemExit(
            "DataRegistry unavailable; configure ML_DB_CONNECTION for pipeline orchestration"
        )

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

    mode_tokens = _resolve_write_mode_tokens(args.write_mode)
    writer_chain: list[MarketDataWriterProtocol] = []

    if "sql" in mode_tokens:
        writer_chain.append(SqlMarketDataWriter(connection_string=args.db))

    if "datastore" in mode_tokens:
        if data_store is None:
            logger.warning(
                "write_mode requested DataStore persistence but DataStore is unavailable; "
                "skipping datastore writer",
            )
        else:
            from ml.stores.data_store import DataStore as _DataStore

            writer_chain.append(
                DataStoreMarketDataWriter(
                    store=cast(_DataStore, data_store),
                ),
            )

    if "parquet" in mode_tokens:
        if parquet_catalog is None:
            raise SystemExit("catalog_path is required when write_mode includes parquet")
        writer_chain.append(
            ParquetCatalogMarketDataWriter(
                catalog=parquet_catalog,
                manifest_resolver=manifest_resolver,
            ),
        )

    if not writer_chain:
        if data_store is not None:
            from ml.stores.data_store import DataStore as _DataStore

            writer_chain.append(
                DataStoreMarketDataWriter(
                    store=cast(_DataStore, data_store),
                ),
            )
        elif parquet_catalog is not None:
            writer_chain.append(
                ParquetCatalogMarketDataWriter(
                    catalog=parquet_catalog,
                    manifest_resolver=manifest_resolver,
                ),
            )
        else:
            raise SystemExit("No ingestion writers available; configure DataStore or catalog")

    primary_writer = writer_chain[0]
    mirror_writers = tuple(writer_chain[1:])
    writer = FanoutMarketDataWriter(primary=primary_writer, mirrors=mirror_writers)
    integration_factory: Callable[..., IntegrationManagerProtocol] | None = cast(
        Callable[..., IntegrationManagerProtocol],
        MLIntegrationManager,
    )

    ingestor: object | None = None
    ingestion_service: DatabentoIngestionService | None = None
    dataset_discovery: DatasetDiscoveryService | None = None
    need_databento = bool(args.ingest or getattr(args, "auto_fill_universe", False))
    if need_databento:
        api_key = os.getenv("DATABENTO_API_KEY", "").strip()
        if api_key:
            client = DatabentoAPIClient(api_key=api_key)
            ingestor = DatabentoIngestor(client=client)
            discovery_policy = DiscoveryPolicy.from_env(os.environ)
            resolver = DatabentoSymbologyResolver(
                client=client.symbology_client,
            )
            dataset_discovery = DatasetDiscoveryService(
                metadata=client.metadata_client,
                policy=discovery_policy,
                resolver=resolver,
            )
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
        dataset_discovery=dataset_discovery,
    )

    # Store write_mode_tokens for determining storage_kind
    orch.write_mode_tokens = mode_tokens

    # Deferred ingestion block runs after dataset config is prepared
    data_dir_effective = Path(args.data_dir)
    if args.catalog_path and str(args.data_dir) == "data/tier1":
        data_dir_effective = Path(args.catalog_path)

    raw_macro_series_ids = tuple(
        item.strip()
        for item in (str(args.macro_series_ids).split(",") if args.macro_series_ids else [])
        if item.strip()
    )
    macro_series_ids: tuple[str, ...] | None = raw_macro_series_ids or None
    if bool(args.include_macro) and macro_series_ids is None:
        macro_series_ids = DEFAULT_MACRO_SERIES

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
            l2_symbols = tuple(
                s.strip().upper() for s in str(args.l2_symbols).split(",") if s.strip()
            )
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
            symbols_desc = f"custom:{len(l2_symbols)}" if l2_symbols else f"tier:{l2_tier}"
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

    market_inputs_tuple = _parse_market_inputs_json(getattr(args, "market_inputs_json", None))

    end_iso = args.end_iso
    start_iso = args.start_iso
    if start_iso is None and end_iso:
        start_iso = _compute_window_start_iso(end_iso=end_iso)

    ds_cfg = DatasetBuildConfig(
        data_dir=str(data_dir_effective),
        symbols=str(args.symbols),
        out_dir=str(args.out_dir),
        dataset_id=str(getattr(args, "dataset_id", "tft_dataset")),
        market_dataset_id=str(market_dataset_id) if market_dataset_id else None,
        market_inputs=market_inputs_tuple,
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
        start_iso=start_iso,
        end_iso=end_iso,
        chunk_days=int(args.chunk_days),
        register_features=bool(args.dataset_register_features),
        feature_registry_dir=args.feature_registry_dir,
        feature_role="teacher",
        validation=validation_cfg,
        vintage_policy=effective_vintage_policy,
        vintage_as_of=args.vintage_as_of,
    )

    ds_cfg = orch._prepare_dataset_config(ds_cfg)

    auto_fill_cfg = _build_auto_fill_config_from_args(args, ds_cfg)
    ingestion_cfg = _build_ingestion_config_from_args(args, ds_cfg)

    stage_token = args.stage or (stage_default.value if stage_default is not None else None)
    stage = Stage(stage_token) if stage_token is not None else Stage.FULL

    ingestion_requested = bool(ingestion_cfg.enabled or auto_fill_cfg.enabled)
    if stage in {Stage.FULL, Stage.INGEST} and ingestion_requested:
        rc = _run_ingestion_stage(
            orch=orch,
            ds_cfg=ds_cfg,
            auto_fill_cfg=auto_fill_cfg,
            ingestion_cfg=ingestion_cfg,
            ingestor=ingestor,
            ingestion_service=ingestion_service,
        )
        if rc != 0:
            return rc
        if stage is Stage.INGEST:
            return 0
    elif stage is Stage.INGEST:
        logger.info("Ingestion stage requested but ingestion inputs are disabled")
        return 0

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
        strict_protocol_validation=(True if args.runtime_strict_protocol_validation else None),
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

    return _execute_stage(
        orch=orch,
        orchestrator_cfg=orchestrator_cfg,
        stage=stage,
        ds_cfg=ds_cfg,
        auto_fill_cfg=auto_fill_cfg,
        args=args,
        ingestor=ingestor,
        ingestion_service=ingestion_service,
    )


def _run_ingestion_stage(
    *,
    orch: MLPipelineOrchestrator,
    ds_cfg: DatasetBuildConfig | None,
    auto_fill_cfg: AutoFillUniverseConfig,
    ingestion_cfg: IngestionStageConfig,
    ingestor: object | None,
    ingestion_service: DatabentoIngestionService | None,
) -> int:
    """
    Run ingestion/backfill operations prior to dataset construction.
    """
    metrics = _IngestionMetrics.default()
    component_label = "pipeline_orchestrator_ingestion"
    stage_status = "skipped"
    work_performed = False
    stage_start = time.perf_counter()
    fallback_reports: list[dict[str, object]] = []
    coverage_metric_emitted = False
    file_metric_emitted = False

    def _finalize() -> None:
        elapsed = time.perf_counter() - stage_start
        metrics.runs_total.labels(component=component_label, status=stage_status).inc()
        metrics.latency_seconds.labels(component=component_label, status=stage_status).observe(
            elapsed
        )

    def _normalise_schema_for_lookback(raw_schema: str | None) -> str:
        token = (raw_schema or "bars").lower()
        if "ohlcv" in token or "bar" in token:
            return "bars"
        if "tbbo" in token or "bbo" in token or "quote" in token:
            return "quotes"
        if "trade" in token:
            return "trades"
        if "mbp" in token or token.startswith(("l2", "l3")):
            return "mbp"
        return token

    def _attempt_primary_ingestion(
        plan_items: tuple[_IngestionPlanItem, ...],
        *,
        policy: CoveragePolicy,
    ) -> _IngestionAttemptReport:
        bindings = tuple(item.binding for item in plan_items if item.binding is not None)
        context: dict[str, object] = {
            "stage": Stage.INGEST.value,
            "attempt": "primary",
            "binding_count": len(bindings),
            "datasets": sorted({item.dataset_id for item in plan_items}),
        }
        rows_written = 0
        attempted_windows = 0
        try:
            for item in plan_items:
                if item.binding is None:
                    continue
                binding = item.binding
                schema_token = _normalise_schema_for_lookback(binding.schema or item.schema)
                lookback_days = get_max_lookback_days(schema_token, policy)
                results = orch.backfill_binding(
                    binding=binding,
                    lookback_days=lookback_days,
                )
                for window_list in results.values():
                    rows_written += window_list.rows_written
                    attempted_windows += window_list.attempted_window_count
            context["rows_written"] = rows_written
            context["attempted_windows"] = attempted_windows
            return _IngestionAttemptReport(success=True, context=context)
        except Exception as exc:  # pragma: no cover - defensive guard
            context["error_type"] = exc.__class__.__name__
            return _IngestionAttemptReport(
                success=False,
                context=context,
                reason=str(exc),
            )

    def _attempt_coverage_ingestion(
        plan_items: tuple[_IngestionPlanItem, ...],
        *,
        policy: CoveragePolicy,
    ) -> _IngestionAttemptReport:
        context: dict[str, object] = {
            "stage": Stage.INGEST.value,
            "attempt": "coverage",
            "plan_items": len(plan_items),
            "datasets": sorted({item.dataset_id for item in plan_items}),
            "instrument_total": sum(len(item.instrument_ids) for item in plan_items),
        }
        window_count = 0
        try:
            for item in plan_items:
                if not item.instrument_ids:
                    continue
                for instrument_id in item.instrument_ids:
                    windows = orch.backfill_coverage(
                        dataset_id=item.dataset_id,
                        schema=item.schema,
                        instrument_id=instrument_id,
                        policy=policy,
                    )
                    window_count += len(windows)
            context["window_count"] = window_count
            return _IngestionAttemptReport(success=True, context=context)
        except Exception as exc:  # pragma: no cover - defensive guard
            context["error_type"] = exc.__class__.__name__
            return _IngestionAttemptReport(
                success=False,
                context=context,
                reason=str(exc),
            )

    def _attempt_manual_ingestion(
        plan_items: tuple[_IngestionPlanItem, ...],
        *,
        lookback_days: int,
        policy: CoveragePolicy,
    ) -> _IngestionAttemptReport:
        context: dict[str, object] = {
            "stage": Stage.INGEST.value,
            "attempt": "manual",
            "lookback_days": lookback_days,
            "plan_items": len(plan_items),
            "datasets": sorted({item.dataset_id for item in plan_items}),
            "instrument_total": sum(len(item.instrument_ids) for item in plan_items),
        }
        if context["instrument_total"] == 0:
            return _IngestionAttemptReport(
                success=False,
                context=context,
                reason="no_instruments",
            )
        rows_written = 0
        attempted_windows = 0
        try:
            for item in plan_items:
                if not item.instrument_ids:
                    continue
                schema_token = _normalise_schema_for_lookback(item.schema)
                effective_lookback = lookback_days or get_max_lookback_days(schema_token, policy)
                for instrument_id in item.instrument_ids:
                    windows = orch.backfill(
                        dataset_id=item.dataset_id,
                        schema=item.schema,
                        instrument_id=instrument_id,
                        lookback_days=effective_lookback,
                    )
                    rows_written += windows.rows_written
                    attempted_windows += windows.attempted_window_count
            context["rows_written"] = rows_written
            context["attempted_windows"] = attempted_windows
            return _IngestionAttemptReport(success=True, context=context)
        except Exception as exc:  # pragma: no cover - defensive guard
            context["error_type"] = exc.__class__.__name__
            return _IngestionAttemptReport(
                success=False,
                context=context,
                reason=str(exc),
            )

    def _find_existing_artifact() -> Path | None:
        if ds_cfg is None:
            return None
        candidates: list[Path] = []
        out_dir = Path(ds_cfg.out_dir)
        data_dir = Path(ds_cfg.data_dir)
        dataset_id_local = ds_cfg.dataset_id
        candidates.append(out_dir / "dataset_metadata.json")
        if dataset_id_local:
            candidates.append(out_dir / dataset_id_local / "dataset_metadata.json")
            candidates.append(data_dir / dataset_id_local / "dataset_metadata.json")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    ingestion_requested = ingestion_cfg.enabled or auto_fill_cfg.enabled
    if not ingestion_requested:
        logger.info(
            "Ingestion stage skipped (disabled)",
            extra={"stage": Stage.INGEST.value, "status": stage_status},
        )
        return 0

    symbol_map_for_ingestion = _collect_symbol_map(ds_cfg=ds_cfg, ingestion_cfg=ingestion_cfg)

    discovery_inputs: tuple[MarketDatasetInput, ...] | None = None
    discover_method = getattr(orch, "_discover_market_inputs", None)
    discovery_service = getattr(orch, "dataset_discovery", None)
    if (
        ingestion_cfg.market_inputs is None
        and callable(discover_method)
        and discovery_service is not None
        and symbol_map_for_ingestion
    ):
        schema_token = _SCHEMA_ALIASES.get(ingestion_cfg.schema.lower(), ingestion_cfg.schema)
        end_ns = time.time_ns()
        lookback_days = max(int(ingestion_cfg.lookback_days or 1), 1)
        start_ns = end_ns - lookback_days * DAY_NS
        discovery_inputs = discover_method(
            symbol_map=symbol_map_for_ingestion,
            schema=schema_token,
            start_ns=start_ns,
            end_ns=end_ns,
            dataset_hint=ingestion_cfg.market_dataset_id or ingestion_cfg.dataset_id,
        )
    if discovery_inputs:
        dataset_id_hint = ingestion_cfg.dataset_id or discovery_inputs[0].dataset_id
        ingestion_cfg = replace(
            ingestion_cfg,
            market_inputs=discovery_inputs,
            dataset_id=dataset_id_hint,
        )

    try:
        plan_items = _build_ingestion_plan(ds_cfg=ds_cfg, ingestion_cfg=ingestion_cfg)
        binding_count = sum(1 for item in plan_items if item.binding is not None)
        datasets_in_plan = sorted({item.dataset_id for item in plan_items})
        schema_set = sorted({item.schema for item in plan_items})

        ingestion_policy = getattr(ingestion_service, "_policy", None)
        if ingestion_policy is not None:
            for item in plan_items:
                try:
                    ingestion_policy.allow_dataset(item.dataset_id)
                except Exception:
                    logger.debug(
                        "Unable to extend ingestion coverage policy",
                        exc_info=True,
                        extra={
                            "dataset_id": item.dataset_id,
                            "stage": Stage.INGEST.value,
                        },
                    )

        should_register = getattr(orch, "data_store", None) is not None and hasattr(
            orch, "_ensure_dataset_registered"
        )
        if should_register and plan_items:
            location_root = Path(
                ds_cfg.data_dir if ds_cfg is not None else (ingestion_cfg.catalog_path or "ml_out"),
            )
            for item in plan_items:
                try:
                    orch._ensure_dataset_registered(
                        dataset_id=item.dataset_id,
                        dataset_type=orch._map_schema_to_dataset_type(item.schema),
                        location=str(location_root),
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.debug(
                        "Dataset auto-registration skipped",
                        exc_info=True,
                        extra={
                            "stage": Stage.INGEST.value,
                            "dataset_id": item.dataset_id,
                            "schema": item.schema,
                            "reason": str(exc),
                        },
                    )

        logger.info(
            "Ingestion stage starting",
            extra={
                "stage": Stage.INGEST.value,
                "plan_items": len(plan_items),
                "binding_count": binding_count,
                "datasets": datasets_in_plan,
                "schemas": schema_set,
            },
        )

        if auto_fill_cfg.enabled:
            if ds_cfg is None:
                logger.warning(
                    "Auto-fill requested but dataset configuration missing; skipping",
                    extra={"stage": Stage.INGEST.value},
                )
            else:
                work_performed = True
                logger.info(
                    "Executing auto-fill ingestion",
                    extra={
                        "stage": Stage.INGEST.value,
                        "symbol_count": len([s for s in ds_cfg.symbols.split(",") if s.strip()]),
                        "instrument_count": (
                            0 if ds_cfg.instrument_ids is None else len(ds_cfg.instrument_ids)
                        ),
                    },
                )
                orch._auto_fill_universe(ds_cfg, auto_fill_cfg)

        if not ingestion_cfg.enabled:
            stage_status = "success" if work_performed else "skipped"
            logger.info(
                "Ingestion stage skipped (disabled)",
                extra={"stage": Stage.INGEST.value, "status": stage_status},
            )
            return 0

        work_performed = True

        if ingestor is None or ingestion_service is None:
            stage_status = "degraded"
            missing_key = not bool(os.getenv("DATABENTO_API_KEY", "").strip())
            detail = (
                "missing_databento_api_key" if missing_key else "ingestion_components_unavailable"
            )
            logger.error(
                "Databento ingestion unavailable; running in degraded mode",
                extra={"stage": Stage.INGEST.value, "detail": detail},
            )
            metrics.fallback_total.labels(component=component_label, level="dummy").inc()
            return 0

        policy = CoveragePolicy.from_env()
        primary_bindings = tuple(item.binding for item in plan_items if item.binding is not None)
        if primary_bindings:
            primary_report = _attempt_primary_ingestion(plan_items, policy=policy)
            if primary_report.success:
                stage_status = "success"
                logger.info(
                    "Ingestion completed via primary bindings",
                    extra={**primary_report.context},
                )
                return 0
            fallback_reports.append(
                {
                    "level": "primary",
                    "reason": primary_report.reason or "unknown",
                    **primary_report.context,
                },
            )
        else:
            fallback_reports.append(
                {
                    "level": "primary",
                    "reason": "no_bindings",
                    "stage": Stage.INGEST.value,
                },
            )

        coverage_candidates = tuple(item for item in plan_items if item.instrument_ids)
        if coverage_candidates:
            metrics.fallback_total.labels(component=component_label, level="cached").inc()
            coverage_metric_emitted = True
            coverage_report = _attempt_coverage_ingestion(
                coverage_candidates,
                policy=policy,
            )
            if coverage_report.success:
                stage_status = "success"
                logger.info(
                    "Ingestion fallback succeeded via cached coverage",
                    extra={**coverage_report.context},
                )
                return 0
            fallback_reports.append(
                {
                    "level": "cached",
                    "reason": coverage_report.reason or "unknown",
                    **coverage_report.context,
                },
            )

        metrics.fallback_total.labels(component=component_label, level="file").inc()
        file_metric_emitted = True
        manual_report = _attempt_manual_ingestion(
            plan_items,
            lookback_days=int(ingestion_cfg.lookback_days),
            policy=policy,
        )
        if manual_report.success:
            stage_status = "success"
            logger.info(
                "Ingestion fallback succeeded via manual lookback",
                extra={**manual_report.context},
            )
            return 0
        fallback_reports.append(
            {
                "level": "file",
                "reason": manual_report.reason or "unknown",
                **manual_report.context,
            },
        )

        artifact_path = _find_existing_artifact()
        if artifact_path is not None:
            stage_status = "degraded"
            if not file_metric_emitted:
                metrics.fallback_total.labels(component=component_label, level="file").inc()
            logger.warning(
                "Using existing dataset artifacts as ingestion fallback",
                extra={
                    "stage": Stage.INGEST.value,
                    "artifact": str(artifact_path),
                },
            )
            return 0

        stage_status = "error"
        if not coverage_metric_emitted:
            metrics.fallback_total.labels(component=component_label, level="cached").inc()
        metrics.fallback_total.labels(component=component_label, level="dummy").inc()
        logger.error(
            "Ingestion fallback exhausted; no viable data sources",
            extra={
                "stage": Stage.INGEST.value,
                "datasets": datasets_in_plan,
                "schemas": schema_set,
                "reports": fallback_reports,
            },
        )
        return 1
    except IngestionError as exc:
        stage_status = "error"
        logger.error(
            "Ingestion stage failed",
            extra={"stage": Stage.INGEST.value, "error": str(exc)},
            exc_info=True,
        )
        if not coverage_metric_emitted:
            metrics.fallback_total.labels(component=component_label, level="cached").inc()
        metrics.fallback_total.labels(component=component_label, level="dummy").inc()
        return 1
    except Exception as exc:  # pragma: no cover - defensive guard
        stage_status = "error"
        logger.exception(
            "Unexpected ingestion stage failure",
            extra={"stage": Stage.INGEST.value, "error": str(exc)},
        )
        if not coverage_metric_emitted:
            metrics.fallback_total.labels(component=component_label, level="cached").inc()
        metrics.fallback_total.labels(component=component_label, level="dummy").inc()
        return 1
    finally:
        _finalize()


def _dataset_only_config(cfg: OrchestratorConfig) -> OrchestratorConfig:
    """
    Return a copy of ``cfg`` with training/promotions disabled.
    """
    hpo_disabled = replace(cfg.hpo, enabled=False)
    teacher_disabled = replace(cfg.teacher, enabled=False)
    student_disabled = replace(cfg.student, enabled=False)
    return replace(
        cfg,
        hpo=hpo_disabled,
        teacher=teacher_disabled,
        student=student_disabled,
        promotions=None,
        integration=None,
    )


def _execute_stage(
    *,
    orch: MLPipelineOrchestrator,
    orchestrator_cfg: OrchestratorConfig,
    stage: Stage,
    ds_cfg: DatasetBuildConfig,
    auto_fill_cfg: AutoFillUniverseConfig,
    args: argparse.Namespace,
    ingestor: object | None,
    ingestion_service: DatabentoIngestionService | None,
) -> int:
    """
    Execute the requested pipeline ``stage`` using the prepared orchestrator.
    """
    if stage is Stage.DATASET:
        dataset_only_cfg = _dataset_only_config(orchestrator_cfg)
        return orch.run(dataset_only_cfg)
    if stage is Stage.TRAIN:
        return orch.run_training_only(orchestrator_cfg)
    if stage is Stage.FULL:
        return orch.run(orchestrator_cfg)
    # Stage.INGEST handled earlier; reaching here implies nothing to do.
    return 0


def _parse_market_inputs_json(
    value: str | None,
) -> tuple[MarketDatasetInput, ...] | None:
    """
    Parse CLI-provided JSON payload into MarketDatasetInput entries.
    """
    if value is None:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"market_inputs_json must be valid JSON: {exc}") from exc

    items: list[object]
    if isinstance(payload, str | dict):
        items = [payload]
    elif isinstance(payload, list):
        items = list(payload)
    else:
        raise SystemExit("market_inputs_json must encode a list, object, or descriptor string")

    inputs: list[MarketDatasetInput] = []
    for entry in items:
        if isinstance(entry, str):
            inputs.append(MarketDatasetInput(descriptor_id=entry))
            continue
        if isinstance(entry, dict):
            descriptor_id = entry.get("descriptor_id")
            dataset_id = entry.get("dataset_id")
            if descriptor_id is None and dataset_id is None:
                raise SystemExit("market_inputs_json entries require descriptor_id or dataset_id")

            symbols_field = entry.get("symbols")
            symbols_tuple: tuple[str, ...] | None
            if symbols_field is None:
                symbols_tuple = None
            elif isinstance(symbols_field, str):
                symbols_tuple = (
                    tuple(
                        token.strip().upper() for token in symbols_field.split(",") if token.strip()
                    )
                    or None
                )
            elif isinstance(symbols_field, list | tuple):
                symbols_tuple = (
                    tuple(
                        str(token).strip().upper() for token in symbols_field if str(token).strip()
                    )
                    or None
                )
            else:
                raise SystemExit("market_inputs_json symbols must be string or iterable")

            schema_override = entry.get("schema") or entry.get("schema_override")
            storage_raw = entry.get("storage_kind") or entry.get("storage_kind_override")
            storage_kind = None
            if storage_raw is not None:
                try:
                    storage_kind = coerce_storage_kind(storage_raw)
                except ValueError as exc:  # pragma: no cover - defensive guard
                    raise SystemExit(
                        f"Invalid storage_kind '{storage_raw}' in market_inputs_json",
                    ) from exc

            inputs.append(
                MarketDatasetInput(
                    descriptor_id=str(descriptor_id) if descriptor_id is not None else None,
                    dataset_id=str(dataset_id) if dataset_id is not None else None,
                    symbols=symbols_tuple,
                    schema_override=str(schema_override) if schema_override is not None else None,
                    storage_kind_override=storage_kind,
                    start=str(entry.get("start")) if entry.get("start") is not None else None,
                    end=str(entry.get("end")) if entry.get("end") is not None else None,
                ),
            )
            continue
        raise SystemExit("market_inputs_json entries must be strings or objects")

    return tuple(inputs) if inputs else None


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


def _build_ingestion_config_from_args(
    args: argparse.Namespace,
    ds_cfg: DatasetBuildConfig | None,
) -> IngestionStageConfig:
    """
    Construct an ingestion stage config from CLI arguments.
    """
    default_cfg = IngestionStageConfig()
    raw_dataset_id = getattr(args, "dataset_id", None)
    dataset_id = str(raw_dataset_id).strip() if raw_dataset_id else None
    if dataset_id is None and ds_cfg is not None and ds_cfg.market_dataset_id:
        dataset_id = ds_cfg.market_dataset_id
    schema = str(getattr(args, "schema", default_cfg.schema))

    raw_instruments = getattr(args, "instruments", None)
    instruments: tuple[str, ...]
    if raw_instruments:
        tokens = [token.strip() for token in str(raw_instruments).split(",") if token.strip()]
        instruments = tuple(tokens) if tokens else default_cfg.instruments
    elif ds_cfg is not None and ds_cfg.instrument_ids:
        instruments = ds_cfg.instrument_ids
    else:
        instruments = default_cfg.instruments

    raw_symbol_override = getattr(args, "symbols", None)
    if raw_symbol_override:
        symbol_tokens = tuple(
            token.strip().upper() for token in str(raw_symbol_override).split(",") if token.strip()
        )
    elif ds_cfg is not None:
        symbol_tokens = tuple(
            token.strip().upper() for token in str(ds_cfg.symbols).split(",") if token.strip()
        )
    else:
        symbol_tokens = tuple(inst.split(".")[0].upper() for inst in instruments if inst)
    symbols: tuple[str, ...] | None = symbol_tokens or None

    raw_instrument_ids = getattr(args, "instrument_ids", None)
    if raw_instrument_ids:
        instrument_ids_override = tuple(
            token.strip() for token in str(raw_instrument_ids).split(",") if token.strip()
        )
    elif ds_cfg is not None and ds_cfg.instrument_ids:
        instrument_ids_override = ds_cfg.instrument_ids
    else:
        instrument_ids_override = tuple(instruments)
    instrument_ids: tuple[str, ...] | None = instrument_ids_override or None

    lookback_days = int(getattr(args, "lookback_days", default_cfg.lookback_days))
    coverage_mode = str(getattr(args, "coverage_mode", default_cfg.coverage_mode))
    write_mode = str(getattr(args, "write_mode", default_cfg.write_mode))
    catalog_path_raw = getattr(args, "catalog_path", None)
    catalog_path = str(catalog_path_raw) if catalog_path_raw else None

    market_dataset_id_raw = getattr(args, "market_dataset_id", None)
    market_dataset_id = str(market_dataset_id_raw).strip() if market_dataset_id_raw else None
    if market_dataset_id is None and ds_cfg is not None:
        market_dataset_id = ds_cfg.market_dataset_id

    market_inputs = _parse_market_inputs_json(
        getattr(args, "market_inputs_json", None),
    )
    if not market_inputs:
        market_inputs = ds_cfg.market_inputs if ds_cfg is not None else None

    return IngestionStageConfig(
        enabled=bool(getattr(args, "ingest", False)),
        dataset_id=dataset_id,
        schema=schema,
        instruments=instruments,
        lookback_days=lookback_days,
        coverage_mode=coverage_mode,
        write_mode=write_mode,
        catalog_path=catalog_path,
        symbols=symbols,
        instrument_ids=instrument_ids,
        market_dataset_id=market_dataset_id,
        market_inputs=market_inputs,
    )


def _ingestion_config_to_args(cfg: IngestionStageConfig) -> list[str]:
    """
    Convert an ingestion stage config into CLI arguments.
    """
    args: list[str] = []
    if cfg.enabled:
        args.append("--ingest")
    if cfg.dataset_id:
        args += ["--dataset_id", cfg.dataset_id]
    args += ["--schema", cfg.schema]
    if cfg.instruments:
        args += ["--instruments", ",".join(cfg.instruments)]
    if cfg.symbols:
        args += ["--symbols", ",".join(cfg.symbols)]
    if cfg.instrument_ids:
        args += ["--instrument_ids", ",".join(cfg.instrument_ids)]
    args += ["--lookback_days", str(cfg.lookback_days)]
    args += ["--coverage_mode", cfg.coverage_mode]
    args += ["--write_mode", cfg.write_mode]
    if cfg.catalog_path:
        args += ["--catalog_path", cfg.catalog_path]
    if cfg.market_dataset_id:
        args += ["--market_dataset_id", cfg.market_dataset_id]
    if cfg.market_inputs:
        payload: list[dict[str, object]] = []
        for item in cfg.market_inputs:
            entry: dict[str, object] = {}
            if item.descriptor_id is not None:
                entry["descriptor_id"] = item.descriptor_id
            if item.dataset_id is not None:
                entry["dataset_id"] = item.dataset_id
            if item.symbols is not None:
                entry["symbols"] = list(item.symbols)
            if item.schema_override is not None:
                entry["schema"] = item.schema_override
            if item.storage_kind_override is not None:
                entry["storage_kind"] = item.storage_kind_override.value
            if item.start is not None:
                entry["start"] = item.start
            if item.end is not None:
                entry["end"] = item.end
        payload.append(entry)
        args += ["--market_inputs_json", json.dumps(payload)]
    return args


@dataclass(slots=True, frozen=True)
class _IngestionPlanItem:
    """
    Resolved ingestion work unit derived from configuration inputs.
    """

    binding: ResolvedMarketBinding | None
    dataset_id: str
    schema: str
    instrument_ids: tuple[str, ...]


def _build_ingestion_plan(
    *,
    ds_cfg: DatasetBuildConfig | None,
    ingestion_cfg: IngestionStageConfig,
) -> tuple[_IngestionPlanItem, ...]:
    """
    Construct per-binding ingestion plan items from configuration.
    """
    symbol_to_instruments = _collect_symbol_map(ds_cfg=ds_cfg, ingestion_cfg=ingestion_cfg)

    market_inputs = ingestion_cfg.market_inputs
    if market_inputs is None and ds_cfg is not None:
        market_inputs = ds_cfg.market_inputs

    symbols_tuple = tuple(symbol_to_instruments.keys())
    instrument_ids_all = tuple(
        dict.fromkeys(chain.from_iterable(symbol_to_instruments.values())),
    )

    market_dataset_id = (
        ingestion_cfg.market_dataset_id
        or (ds_cfg.market_dataset_id if ds_cfg is not None else None)
        or ingestion_cfg.dataset_id
    )

    fallback_candidates: list[str | None] = [ingestion_cfg.dataset_id, market_dataset_id]
    if market_inputs:
        fallback_candidates.extend(item.dataset_id for item in market_inputs if item.dataset_id)

    resolved_bindings: tuple[ResolvedMarketBinding, ...] = ()
    if symbols_tuple:
        try:
            resolved_bindings = IngestionOrchestrator.resolve_market_bindings(
                symbols=symbols_tuple,
                instrument_ids=instrument_ids_all or None,
                market_dataset_id=market_dataset_id,
                market_inputs=market_inputs,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Ingestion binding resolution failed", exc_info=True)

    def _select_fallback_dataset() -> str | None:
        allowed = _get_allowed_databento_datasets()
        candidates_ordered: list[str] = []
        candidates_ordered.extend(candidate for candidate in fallback_candidates if candidate)
        candidates_ordered.extend(
            binding.dataset_id for binding in resolved_bindings if binding.dataset_id
        )
        if ds_cfg is not None and ds_cfg.market_dataset_id:
            candidates_ordered.append(ds_cfg.market_dataset_id)

        if allowed:
            for candidate in candidates_ordered:
                if candidate in allowed:
                    return candidate
        for candidate in candidates_ordered:
            if candidate:
                return candidate
        if allowed:
            # Deterministic fallback so ingest runs without manual dataset wiring.
            ordered_allowed = sorted(allowed)
            if ordered_allowed:
                return ordered_allowed[0]
        return None

    fallback_dataset_id = _select_fallback_dataset()
    if fallback_dataset_id is None:
        raise ValueError("Ingestion configuration requires a dataset identifier")

    fallback_schema = _SCHEMA_ALIASES.get(ingestion_cfg.schema.lower(), ingestion_cfg.schema)
    if not fallback_schema:
        raise ValueError("Ingestion configuration requires a schema value")

    plan_items: list[_IngestionPlanItem] = []
    for binding in resolved_bindings:
        dataset_id = binding.dataset_id or fallback_dataset_id
        schema = binding.schema or fallback_schema
        schema = _SCHEMA_ALIASES.get(schema.lower(), schema)
        binding_instruments = tuple(
            dict.fromkeys(
                binding.instrument_ids or symbol_to_instruments.get(binding.symbol.upper(), ()),
            ),
        )
        if not binding_instruments:
            fallback_symbol = binding.symbol.strip().upper()
            binding_instruments = (fallback_symbol,) if fallback_symbol else ()
        plan_items.append(
            _IngestionPlanItem(
                binding=binding,
                dataset_id=dataset_id,
                schema=schema,
                instrument_ids=binding_instruments,
            ),
        )

    if not plan_items:
        manual_instruments = instrument_ids_all
        if not manual_instruments and ingestion_cfg.instrument_ids:
            manual_instruments = tuple(
                dict.fromkeys(
                    instrument.strip().upper()
                    for instrument in ingestion_cfg.instrument_ids
                    if instrument.strip()
                ),
            )
        if not manual_instruments and ingestion_cfg.instruments:
            manual_instruments = tuple(
                dict.fromkeys(
                    instrument.strip().upper()
                    for instrument in ingestion_cfg.instruments
                    if instrument.strip()
                ),
            )
        if not manual_instruments:
            manual_instruments = tuple(symbol_to_instruments.keys())
        if not manual_instruments:
            manual_instruments = tuple(
                instrument.strip().upper().split(".")[0]
                for instrument in ingestion_cfg.instruments
                if instrument.strip()
            )
        manual_instruments = tuple(dict.fromkeys(filter(None, manual_instruments)))
        if not manual_instruments:
            raise ValueError(
                "Ingestion configuration requires at least one instrument for manual fallback",
            )
        plan_items.append(
            _IngestionPlanItem(
                binding=None,
                dataset_id=fallback_dataset_id,
                schema=fallback_schema,
                instrument_ids=manual_instruments,
            ),
        )

    return tuple(plan_items)


def _build_auto_fill_config_from_args(
    args: argparse.Namespace,
    _dataset_cfg: DatasetBuildConfig,
) -> AutoFillUniverseConfig:
    enabled = bool(getattr(args, "auto_fill_universe", False))
    instrument_override: tuple[str, ...] | None = None
    raw_override = getattr(args, "auto_fill_instrument_ids", None)
    if raw_override:
        instrument_override = tuple(
            item.strip() for item in str(raw_override).split(",") if item.strip()
        )
    dataset_id_arg = getattr(args, "auto_fill_dataset_id", None)
    dataset_id = str(dataset_id_arg or getattr(args, "dataset_id", "EQUS.MINI"))
    include_l2 = bool(getattr(args, "include_l2", False)) and not bool(
        getattr(args, "auto_fill_skip_l2", False),
    )
    l2_dataset_id = str(
        getattr(args, "auto_fill_l2_dataset_id", None) or "DBEQ.BASIC",
    )
    l2_schema = str(
        getattr(args, "auto_fill_l2_schema", None) or "mbp-10",
    )
    l2_days_raw = getattr(args, "auto_fill_l2_days", None)
    l2_days = int(l2_days_raw) if l2_days_raw is not None else None
    l2_progress_file_raw = getattr(args, "auto_fill_l2_progress_file", None)
    l2_progress_file = str(l2_progress_file_raw) if l2_progress_file_raw else None
    allow_dataset_l2 = bool(getattr(args, "auto_fill_allow_dataset_l2_ingest", False))
    include_l3 = bool(getattr(args, "auto_fill_include_l3", False))
    l3_dataset_id_raw = getattr(args, "auto_fill_l3_dataset_id", None)
    l3_schema_raw = getattr(args, "auto_fill_l3_schema", None)
    l3_days_raw = getattr(args, "auto_fill_l3_days", None)
    l3_days = int(l3_days_raw) if l3_days_raw is not None else None

    include_bars = True
    include_tbbo = True
    include_trades = True
    if dataset_id_arg and dataset_id != getattr(args, "dataset_id", dataset_id):
        include_bars = False
        include_tbbo = False
        include_trades = False

    return AutoFillUniverseConfig(
        enabled=enabled,
        dataset_id=dataset_id,
        include_bars=include_bars,
        include_tbbo=include_tbbo,
        include_trades=include_trades,
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
