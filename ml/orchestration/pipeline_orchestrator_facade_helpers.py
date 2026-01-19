"""
Helper methods for `MLPipelineOrchestratorFacade`.

This module keeps `ml/orchestration/pipeline_orchestrator_facade.py` thin by hosting
small parity helpers used by unit tests and legacy shims.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.orchestration.binding_resolver import BindingResolver
from ml.orchestration.common.stage_controller import IntegrationManagerProtocol
from ml.orchestration.common.utils import parse_symbols
from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.dataset_builder import DatasetBuilder
from ml.orchestration.discovery_client import DiscoveryClient
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol


if TYPE_CHECKING:
    from ml.config.market_data import MarketDatasetInput
    from ml.data import DatasetMetadata
    from ml.data.ingest.discovery import DatasetDiscoveryService
    from ml.data.ingest.orchestrator import BackfillWindowList
    from ml.data.ingest.service import DatabentoIngestionService
    from ml.orchestration.config_types import AutoFillUniverseConfig
    from ml.orchestration.ingestion_coordinator import IngestionCoordinator
    from ml.orchestration.ingestion_coordinator import _AutoFillMetrics
    from ml.orchestration.registry_synchronizer import RegistrySynchronizer


logger = logging.getLogger(__name__)


class OrchestratorFacadeHelpers:
    """Shared helper methods used by `MLPipelineOrchestratorFacade`."""

    coverage: CoverageProviderProtocol
    writer: MarketDataWriterProtocol
    build_main: Callable[..., int] | None
    teacher_main: Callable[..., int] | None
    registry: object | None
    data_registry: object | None
    ingestor: object | None
    service: DatabentoIngestionService | None
    model_registry: object | None
    feature_registry: object | None
    strategy_registry: object | None
    integration_manager_factory: Callable[..., IntegrationManagerProtocol] | None
    dataset_discovery: object | None
    _config_resolver: ConfigResolver | None
    _discovery_client: DiscoveryClient | None
    _ingestion_coordinator: IngestionCoordinator | None
    _registry_synchronizer: RegistrySynchronizer | None
    _dataset_builder: DatasetBuilder | None
    _ingestion_backfill: Callable[..., BackfillWindowList] | None
    _ingestion_backfill_binding: Callable[..., dict[str, BackfillWindowList]] | None
    _ingestion_backfill_coverage: Callable[..., list[tuple[int, int]]] | None
    _ingestion_ensure_dataset_registered: Callable[..., None] | None
    backfill: Callable[..., BackfillWindowList]
    backfill_binding: Callable[..., dict[str, BackfillWindowList]]
    backfill_coverage: Callable[..., list[tuple[int, int]]]
    _ensure_dataset_registered: Callable[..., None]

    def apply_default_market_inputs(self, cfg: DatasetBuildConfig) -> DatasetBuildConfig:
        """Apply default market inputs via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.apply_default_market_inputs(cfg)

    def _build_health_status(self) -> dict[str, Any]:
        """Build a health status summary for facade components."""
        binding_available = bool(self.coverage or self.service or self._discovery_client)
        return {
            "implementation": "component-based",
            "coverage_provider": "healthy" if self.coverage else "unavailable",
            "writer": "healthy" if self.writer else "unavailable",
            "build_main": "healthy" if self.build_main is not None else "unavailable",
            "teacher_main": "healthy" if self.teacher_main is not None else "unavailable",
            "config_resolver": "healthy" if self._config_resolver is not None else "unavailable",
            "discovery_client": "healthy" if self._discovery_client is not None else "unavailable",
            "binding_resolver": "healthy" if binding_available else "unavailable",
            "ingestion_coordinator": (
                "healthy" if self._ingestion_coordinator is not None else "unavailable"
            ),
            "dataset_builder": "healthy" if self._dataset_builder is not None else "unavailable",
            "has_registry": self.registry is not None,
            "has_data_registry": self.data_registry is not None,
            "has_ingestor": self.ingestor is not None,
            "has_model_registry": self.model_registry is not None,
            "has_feature_registry": self.feature_registry is not None,
            "has_strategy_registry": self.strategy_registry is not None,
            "has_integration_manager_factory": self.integration_manager_factory is not None,
        }

    def collect_symbol_map(
        self,
        *,
        ds_cfg: DatasetBuildConfig | None = None,
        symbols: tuple[str, ...] | None = None,
        instruments: tuple[str, ...] | None = None,
        instrument_ids: tuple[str, ...] | None = None,
        market_inputs: tuple[Any, ...] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Collect the normalized symbol map via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.collect_symbol_map(
            ds_cfg=ds_cfg,
            symbols=symbols,
            instruments=instruments,
            instrument_ids=instrument_ids,
            market_inputs=market_inputs,
        )

    def compute_window_start_iso(self, end_iso: str, lookback_years: int = 1) -> str:
        """Compute an ISO start date from end date and lookback via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.compute_window_start_iso(end_iso=end_iso, lookback_years=lookback_years)

    def resolve_window_bounds_ns(self, cfg: DatasetBuildConfig) -> tuple[int, int]:
        """Resolve dataset time window bounds (ns) via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.resolve_window_bounds_ns(cfg)

    def prepare_dataset_config(
        self,
        *,
        cfg: DatasetBuildConfig,
        resolved_inputs: tuple[Any, ...] | None,
        bindings: tuple[Any, ...],
    ) -> DatasetBuildConfig:
        """Prepare a dataset config for build stage via ConfigResolver."""
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.prepare_dataset_config(
            cfg=cfg,
            resolved_inputs=resolved_inputs,
            bindings=bindings,
        )

    def discover_market_inputs(
        self,
        *,
        symbol_map: dict[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...] | None:
        """Discover market inputs via DiscoveryClient when available."""
        discovery_service = cast("DatasetDiscoveryService | None", getattr(self, "dataset_discovery", None))
        discovery_client = self._discovery_client or (
            DiscoveryClient(dataset_discovery=discovery_service) if discovery_service else None
        )
        if discovery_client is None:
            return None
        try:
            return discovery_client.discover_market_inputs(
                symbol_map=symbol_map,
                schema=schema,
                start_ns=start_ns,
                end_ns=end_ns,
                dataset_hint=dataset_hint,
            )
        except Exception:
            logger.debug("discover_market_inputs failed (facade)", exc_info=True)
            return None

    def resolve_market_inputs(
        self,
        *,
        cfg: DatasetBuildConfig,
        symbol_map: dict[str, tuple[str, ...]],
        start_ns: int,
        end_ns: int,
    ) -> tuple[tuple[MarketDatasetInput, ...] | None, tuple[ResolvedMarketBinding, ...]]:
        """Resolve market bindings and inputs via BindingResolver."""
        resolver = BindingResolver(
            coverage_provider=self.coverage,
            ingestion_service=self.service,
            discovery_client=self._discovery_client,
        )
        return resolver.resolve_market_inputs(
            cfg=cfg,
            symbol_map=symbol_map,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    def filter_candidate_bindings(
        self,
        *,
        candidates: tuple[ResolvedMarketBinding, ...],
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> tuple[ResolvedMarketBinding, ...]:
        """Filter candidate bindings via BindingResolver."""
        resolver = BindingResolver(
            coverage_provider=self.coverage,
            ingestion_service=self.service,
            discovery_client=self._discovery_client,
        )
        return resolver.filter_candidate_bindings(
            candidates=candidates,
            start_ns=start_ns,
            end_ns=end_ns,
            symbol=symbol,
            default_schema=default_schema,
        )

    def _parse_symbols(self, symbols: str) -> list[str]:
        """
        Parse a comma-separated symbol list into normalized tokens.

        Args:
            symbols: Raw symbols string.

        Returns:
            List of symbol tokens with whitespace trimmed.
        """
        return parse_symbols(symbols)

    @contextmanager
    def _patched_descriptor_loader(self) -> Iterator[None]:
        """
        Patch feed descriptor loading to honor pipeline orchestrator overrides.
        """
        import sys

        from ml.config import market_data as market_data_module
        from ml.orchestration import config_resolver as config_resolver_module

        loader = market_data_module.load_market_feed_descriptors
        module = sys.modules.get("ml.orchestration.pipeline_orchestrator")
        override = getattr(module, "load_market_feed_descriptors", None) if module is not None else None
        if callable(override):
            loader = override

        original_market = market_data_module.load_market_feed_descriptors
        original_config = getattr(config_resolver_module, "load_market_feed_descriptors", None)
        market_data_module.load_market_feed_descriptors = loader
        if original_config is not None:
            setattr(config_resolver_module, "load_market_feed_descriptors", loader)
        try:
            yield
        finally:
            market_data_module.load_market_feed_descriptors = original_market
            if original_config is not None:
                setattr(config_resolver_module, "load_market_feed_descriptors", original_config)

    @contextmanager
    def _patched_ingestion_backfill(self) -> Iterator[None]:
        """
        Patch ingestion coordinator backfill callables for facade overrides.
        """
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        coordinator = cast(Any, self._ingestion_coordinator)
        original_backfill = coordinator.backfill
        original_backfill_binding = coordinator.backfill_binding
        original_backfill_coverage = coordinator.backfill_coverage
        original_ensure = coordinator._ensure_dataset_registered
        coordinator.backfill = self.backfill
        coordinator.backfill_binding = self.backfill_binding
        coordinator.backfill_coverage = self.backfill_coverage
        coordinator._ensure_dataset_registered = self._ensure_dataset_registered
        try:
            yield
        finally:
            coordinator.backfill = original_backfill
            coordinator.backfill_binding = original_backfill_binding
            coordinator.backfill_coverage = original_backfill_coverage
            coordinator._ensure_dataset_registered = original_ensure

    def _prepare_dataset_config(self, cfg: DatasetBuildConfig) -> DatasetBuildConfig:
        """
        Resolve discovery inputs and bindings for dataset build configuration.
        """
        if not isinstance(self._config_resolver, ConfigResolver):
            logger.debug("ConfigResolver unavailable; skipping dataset config preparation")
            return cfg

        with self._patched_descriptor_loader():
            base_cfg = self.apply_default_market_inputs(cfg)
            symbol_map = self.collect_symbol_map(
                ds_cfg=base_cfg,
                symbols=None,
                instruments=None,
                instrument_ids=None,
                market_inputs=base_cfg.market_inputs,
            )
            if symbol_map:
                augmented: dict[str, tuple[str, ...]] = dict(symbol_map)
                for symbol, instruments in symbol_map.items():
                    if "." not in symbol:
                        continue
                    base = symbol.split(".")[0]
                    if not base:
                        continue
                    current = list(augmented.get(base, ()))
                    candidates = instruments or (symbol,)
                    for inst in candidates:
                        if inst not in current:
                            current.append(inst)
                    augmented[base] = tuple(current)
                symbol_map = augmented
            bounds = self.resolve_window_bounds_ns(base_cfg)
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                logger.debug("Window bounds unavailable; skipping dataset config preparation")
                return base_cfg
            start_ns, end_ns = bounds
            resolved_inputs, bindings = self.resolve_market_inputs(
                cfg=base_cfg,
                symbol_map=symbol_map,
                start_ns=start_ns,
                end_ns=end_ns,
            )

            filtered_bindings = bindings
            if bindings:
                default_schema = self._config_resolver.infer_default_schema(base_cfg)
                selected: list[ResolvedMarketBinding] = []
                for symbol in symbol_map.keys():
                    binding_candidates = tuple(
                        binding for binding in bindings if binding.symbol.upper() == symbol.upper()
                    )
                    if not binding_candidates:
                        continue
                    filtered = self.filter_candidate_bindings(
                        candidates=binding_candidates,
                        start_ns=start_ns,
                        end_ns=end_ns,
                        symbol=symbol,
                        default_schema=default_schema,
                    )
                    if filtered:
                        selected.append(filtered[0])
                if selected:
                    filtered_bindings = tuple(selected)
                    if resolved_inputs:
                        allowed_ids = {
                            binding.dataset_id
                            for binding in filtered_bindings
                            if binding.dataset_id
                        }
                        allowed_ids.update(
                            binding.descriptor_id
                            for binding in filtered_bindings
                            if binding.descriptor_id
                        )
                        resolved_inputs = tuple(
                            item
                            for item in resolved_inputs
                            if (
                                (item.dataset_id is not None and item.dataset_id in allowed_ids)
                                or (item.descriptor_id is not None and item.descriptor_id in allowed_ids)
                            )
                        )

            if resolved_inputs is None and filtered_bindings:
                from ml.config.market_data import MarketDatasetInput

                resolved_inputs = tuple(
                    MarketDatasetInput(
                        descriptor_id=binding.descriptor_id,
                        dataset_id=binding.dataset_id,
                        symbols=(binding.symbol,),
                        schema_override=binding.schema,
                        storage_kind_override=binding.storage_kind,
                    )
                    for binding in filtered_bindings
                )

            return self.prepare_dataset_config(
                cfg=base_cfg,
                resolved_inputs=resolved_inputs,
                bindings=filtered_bindings,
            )

    def _auto_fill_universe(
        self,
        dataset_cfg: DatasetBuildConfig,
        auto_fill_cfg: AutoFillUniverseConfig,
    ) -> None:
        """
        Auto-fill the configured market data universe.
        """
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        resolve_fn = (
            self._config_resolver.resolve_instrument_ids
            if isinstance(self._config_resolver, ConfigResolver)
            else self._resolve_instrument_ids
        )
        with self._patched_descriptor_loader():
            with self._patched_ingestion_backfill():
                self._ingestion_coordinator.auto_fill_universe(
                    dataset_cfg=dataset_cfg,
                    auto_fill_cfg=auto_fill_cfg,
                    resolve_instrument_ids_fn=resolve_fn,
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
        Auto-fill a specific schema/instrument combination via ingestion coordinator.
        """
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        with self._patched_descriptor_loader():
            with self._patched_ingestion_backfill():
                self._ingestion_coordinator._auto_fill_schema(
                    dataset_id=dataset_id,
                    schema=schema,
                    instrument_id=instrument_id,
                    lookback_days=lookback_days,
                    metrics=metrics,
                    dataset_cfg=dataset_cfg,
                    processed_bindings=processed_bindings,
                )

    def _guard_dataset_metadata(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        """
        Validate dataset metadata against orchestrator guardrails.
        """
        if self._registry_synchronizer is None:
            raise RuntimeError("RegistrySynchronizer not initialized")
        guard = getattr(self._registry_synchronizer, "_guard_dataset_metadata", None)
        if guard is None:
            raise AttributeError("RegistrySynchronizer missing _guard_dataset_metadata")
        guard(cfg=cfg, metadata=metadata)

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

    @staticmethod
    def _infer_default_schema(cfg: DatasetBuildConfig) -> str:
        """
        Infer a reasonable default schema for discovery lookups.
        """
        return "ohlcv-1m"

    @staticmethod
    def _binding_priority_key(binding: ResolvedMarketBinding) -> tuple[int, str]:
        dataset_id = binding.dataset_id.upper()
        if dataset_id == "EQUS.MINI":
            return (0, dataset_id)
        if dataset_id == "XNAS.ITCH":
            return (1, dataset_id)
        return (2, dataset_id)

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

    @staticmethod
    def _ns_to_datetime(value: int) -> object:
        """
        Convert nanoseconds since epoch to an aware UTC datetime.
        """
        seconds = value / 1_000_000_000
        return datetime.fromtimestamp(seconds, tz=UTC)
