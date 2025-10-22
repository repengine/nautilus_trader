"""
Runtime wrapper for the ML pipeline orchestrator.

Legacy implementation remains the default while the component-based facade
continues to mature behind an explicit opt-in flag.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import ml.orchestration.ingestion_coordinator as _ingestion_coord_module
from ml.data.ingest.orchestrator import IngestionOrchestrator


def _use_component_impl() -> bool:
    """
    Determine whether the component-based facade should be activated.

    Precedence (highest to lowest):
    1. ML_USE_COMPONENT_PIPELINE_ORCHESTRATOR=1 explicitly opts in.
    2. ML_USE_COMPONENT_PIPELINE_ORCHESTRATOR=0 explicitly opts out.
    3. ML_USE_LEGACY_PIPELINE_ORCHESTRATOR continues to work:
       - "1" => legacy implementation
       - "0" => component implementation
       - unset => legacy (default)
    """
    component_flag = os.getenv("ML_USE_COMPONENT_PIPELINE_ORCHESTRATOR")
    if component_flag is not None:
        return component_flag.strip() == "1"

    legacy_flag = os.getenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR")
    if legacy_flag is not None:
        return legacy_flag.strip() == "0"

    return False


_ActiveOrchestrator: type[Any]

if not _use_component_impl():
    import ml.orchestration.pipeline_orchestrator_legacy as _legacy
    from ml.data.ingest.resume import DatabentoIngestor
    from ml.orchestration.binding_resolver import BindingResolver
    from ml.orchestration.config_resolver import ConfigResolver
    from ml.orchestration.dataset_builder import DatasetBuilder
    from ml.orchestration.discovery_client import DiscoveryClient
    from ml.orchestration.ingestion_coordinator import IngestionCoordinator
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.protocols import CoverageProviderProtocol
    from ml.stores.protocols import DataStoreFacadeProtocol
    from ml.stores.protocols import MarketDataWriterProtocol

    class _NullCoverageProvider:
        def read_bucket_coverage(
            self,
            *,
            dataset_id: str,
            schema: str,
            instrument_id: str,
            start_ns: int,
            end_ns: int,
        ) -> set[int]:
            return set()

    class _NullMarketDataWriter:
        def write(
            self,
            *,
            dataset_id: str,
            schema: str,
            instrument_id: str,
            df: Any,
        ) -> int:
            return 0

    def _noop_main(*args: object, **kwargs: object) -> int:  # pragma: no cover - noop
        return 0

    class _CompatibleLegacyOrchestrator(_legacy.MLPipelineOrchestrator):
        """Legacy orchestrator accepting the newer facade keyword arguments."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            coverage = kwargs.pop("coverage_provider", None) or kwargs.pop("coverage", None)
            if coverage is None:
                coverage = _NullCoverageProvider()

            writer = kwargs.pop("writer", None)
            if writer is None:
                writer = _NullMarketDataWriter()

            build_main = kwargs.pop("build_main", None)
            if build_main is None:
                build_main = _noop_main

            teacher_main = kwargs.pop("teacher_main", None)
            if teacher_main is None:
                teacher_main = _noop_main

            # Legacy facade expects write_mode_tokens as positional omission; ensure
            # dataset discovery kwargs remain for legacy implementation.
            coverage_impl = cast(CoverageProviderProtocol, coverage)
            writer_impl = cast(MarketDataWriterProtocol, writer)
            build_main_impl = cast(_legacy._CliMain, build_main)
            teacher_main_impl = cast(_legacy._CliMain, teacher_main)
            super().__init__(coverage_impl, writer_impl, build_main_impl, teacher_main_impl, **kwargs)

            self._config_resolver = ConfigResolver()
            self._discovery_client = DiscoveryClient(
                dataset_discovery=self.dataset_discovery,
                ingestion_service=self.service,
            )
            effective_coverage = cast(
                CoverageProviderProtocol,
                getattr(self, "coverage", coverage_impl),
            )
            self._binding_resolver = BindingResolver(
                coverage_provider=effective_coverage,
                ingestion_service=self.service,
                discovery_client=self._discovery_client,
            )
            self._ingestion_coordinator = IngestionCoordinator(
                coverage=effective_coverage,
                writer=writer_impl,
                registry=cast(RegistryProtocol | None, self.data_registry),
                ingestor=cast(DatabentoIngestor | None, self.ingestor),
                service=self.service,
                raw_writer=self.raw_writer,
                domain_loader=self.domain_loader,
                discovery_client=self._discovery_client,
                write_mode_tokens=getattr(self, "write_mode_tokens", ()),
            )
            self._dataset_builder = DatasetBuilder(
                data_store=cast(DataStoreFacadeProtocol | None, self.data_store),
                data_registry=cast(RegistryProtocol | None, self.data_registry),
            )

        # ------------------------------------------------------------------
        # Config methods
        # ------------------------------------------------------------------
        def apply_default_market_inputs(self, cfg: Any) -> Any:
            return self._config_resolver.apply_default_market_inputs(cfg)

        def collect_symbol_map(
            self,
            *,
            ds_cfg: Any | None,
            symbols: tuple[str, ...] | None = None,
            instruments: tuple[str, ...] | None = None,
            instrument_ids: tuple[str, ...] | None = None,
            market_inputs: Any = None,
        ) -> dict[str, tuple[str, ...]]:
            return self._config_resolver.collect_symbol_map(
                ds_cfg=ds_cfg,
                symbols=symbols,
                instruments=instruments,
                instrument_ids=instrument_ids,
                market_inputs=market_inputs,
            )

        def resolve_window_bounds_ns(self, cfg: Any) -> tuple[int, int]:
            return self._config_resolver.resolve_window_bounds_ns(cfg)

        def prepare_dataset_config(self, cfg: Any, resolved_inputs: Any, bindings: Any) -> Any:
            return self._config_resolver.prepare_dataset_config(cfg, resolved_inputs, bindings)

        # ------------------------------------------------------------------
        # Discovery + binding
        # ------------------------------------------------------------------
        def discover_market_inputs(self, *args: Any, **kwargs: Any) -> Any:
            return self._discovery_client.discover_market_inputs(*args, **kwargs)

        def resolve_market_inputs(self, *args: Any, **kwargs: Any) -> Any:
            return self._binding_resolver.resolve_market_inputs(*args, **kwargs)

        def filter_candidate_bindings(self, *args: Any, **kwargs: Any) -> Any:
            return self._binding_resolver.filter_candidate_bindings(*args, **kwargs)

        def select_binding_with_coverage(self, *args: Any, **kwargs: Any) -> Any:
            return self._binding_resolver.select_binding_with_coverage(*args, **kwargs)

        # ------------------------------------------------------------------
        # Ingestion coordination
        # ------------------------------------------------------------------
        def run_pre_ingestion(self, *args: Any, **kwargs: Any) -> Any:
            return self._ingestion_coordinator.run_pre_ingestion(*args, **kwargs)

        def backfill(self, *args: Any, **kwargs: Any) -> Any:
            return self._ingestion_coordinator.backfill(*args, **kwargs)

        def backfill_binding(self, *args: Any, **kwargs: Any) -> Any:
            return self._ingestion_coordinator.backfill_binding(*args, **kwargs)

        def backfill_coverage(self, *args: Any, **kwargs: Any) -> Any:
            return self._ingestion_coordinator.backfill_coverage(*args, **kwargs)

        def auto_fill_universe(self, *args: Any, **kwargs: Any) -> Any:
            return self._ingestion_coordinator.auto_fill_universe(*args, **kwargs)

        # ------------------------------------------------------------------
        # Dataset building
        # ------------------------------------------------------------------
        def build_dataset(self, cfg: Any) -> int:
            return self._dataset_builder.build_dataset(cfg)

        def validate_dataset(self, *args: Any, **kwargs: Any) -> Any:
            return self._dataset_builder.validate_dataset(*args, **kwargs)

    _ActiveOrchestrator = _CompatibleLegacyOrchestrator
    _apply_default_market_inputs = _legacy._apply_default_market_inputs
    _build_auto_fill_config_from_args = _legacy._build_auto_fill_config_from_args
    _dataset_only_config = _legacy._dataset_only_config
    _parse_market_inputs_json = _legacy._parse_market_inputs_json
    _resolve_write_mode_tokens = _legacy._resolve_write_mode_tokens
    _run_ingestion_stage = _legacy._run_ingestion_stage
    main = _legacy.main
    parse_args = _legacy.parse_args
else:
    import ml.orchestration.pipeline_orchestrator_component as _component

    _ActiveOrchestrator = _component.MLPipelineOrchestrator
    _apply_default_market_inputs = _component._apply_default_market_inputs
    _build_auto_fill_config_from_args = _component._build_auto_fill_config_from_args
    _dataset_only_config = _component._dataset_only_config
    _parse_market_inputs_json = _component._parse_market_inputs_json
    _resolve_write_mode_tokens = _component._resolve_write_mode_tokens
    _run_ingestion_stage = _component._run_ingestion_stage
    main = _component.main
    parse_args = _component.parse_args


if TYPE_CHECKING:  # pragma: no cover - typing aid
    from ml.orchestration.pipeline_orchestrator_legacy import MLPipelineOrchestrator as _TypingOrchestrator
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.protocols import CoverageProviderProtocol
    from ml.stores.protocols import DataStoreFacadeProtocol
    from ml.stores.protocols import MarketDataWriterProtocol

    MLPipelineOrchestrator = _TypingOrchestrator
else:
    MLPipelineOrchestrator = _ActiveOrchestrator

# Ensure orchestration tests can monkeypatch the orchestrator via this module alias.
setattr(_ingestion_coord_module, "IngestionOrchestrator", IngestionOrchestrator)

__all__ = [
    "IngestionOrchestrator",
    "MLPipelineOrchestrator",
    "_apply_default_market_inputs",
    "_build_auto_fill_config_from_args",
    "_dataset_only_config",
    "_parse_market_inputs_json",
    "_resolve_write_mode_tokens",
    "_run_ingestion_stage",
    "main",
    "parse_args",
]
