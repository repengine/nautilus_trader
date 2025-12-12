#!/usr/bin/env python3
"""
MLPipelineOrchestratorFacade - Thin facade delegating to extracted components.

Maintains 100% API parity with the legacy MLPipelineOrchestrator while
internally delegating to 7 specialized components with minimal business logic.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from ml.data.ingest.subscription import SubscriptionPolicy as CoveragePolicy
from ml.orchestration.binding_resolver import BindingResolver
from ml.orchestration.common.stage_controller import IntegrationManagerProtocol
from ml.orchestration.common.stage_controller import StageController
from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import PreIngestionOptions
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.orchestration.dataset_builder import DatasetBuilder
from ml.orchestration.discovery_client import DiscoveryClient
from ml.orchestration.feature_flags import use_legacy_orchestrator
from ml.orchestration.ingestion_coordinator import IngestionCoordinator
from ml.orchestration.registry_synchronizer import RegistrySynchronizer
from ml.orchestration.runtime_attacher import RuntimeAttacher
from ml.orchestration.training_coordinator import TrainingCoordinator
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol


if TYPE_CHECKING:
    from ml.config.scheduler_config import SchedulerConfig
    from ml.data.ingest.discovery import DatasetDiscoveryService
    from ml.data.ingest.market_bindings import ResolvedMarketBinding
    from ml.data.ingest.orchestrator import BackfillWindowList
    from ml.data.ingest.orchestrator import DomainWindowLoaderProtocol
    from ml.data.ingest.service import DatabentoIngestionService
    from ml.stores.io_raw import RawIngestionWriterProtocol

logger = logging.getLogger(__name__)


class _CliMain(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


@dataclass(slots=True)
class MLPipelineOrchestratorFacade:
    """Thin facade for MLPipelineOrchestrator delegating to extracted components."""

    # Required dependencies
    coverage: CoverageProviderProtocol
    writer: MarketDataWriterProtocol
    build_main: _CliMain
    teacher_main: _CliMain
    # Optional dependencies
    registry: object | None = None
    data_registry: object | None = None
    ingestor: object | None = None
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
    dataset_discovery: object | None = None
    # Internal state
    write_mode_tokens: tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)
    _integration_manager: IntegrationManagerProtocol | None = field(default=None, init=False, repr=False)
    _legacy_orchestrator: object | None = field(default=None, init=False, repr=False)
    # Components
    _ingestion_coordinator: IngestionCoordinator | None = field(default=None, init=False, repr=False)
    _dataset_builder: DatasetBuilder | None = field(default=None, init=False, repr=False)
    _training_coordinator: TrainingCoordinator | None = field(default=None, init=False, repr=False)
    _registry_synchronizer: RegistrySynchronizer | None = field(default=None, init=False, repr=False)
    _runtime_attacher: RuntimeAttacher | None = field(default=None, init=False, repr=False)
    _config_resolver: ConfigResolver | None = field(default=None, init=False, repr=False)
    _discovery_client: DiscoveryClient | None = field(default=None, init=False, repr=False)
    _stage_controller: StageController | None = field(default=None, init=False, repr=False)
    _use_legacy: bool = field(default_factory=use_legacy_orchestrator, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.registry is not None and self.data_registry is None:
            object.__setattr__(self, "data_registry", self.registry)
        if self._use_legacy:
            self._init_legacy_orchestrator()
        self._init_components()
        logger.info("MLPipelineOrchestratorFacade initialized",
                    extra={"has_legacy": self._legacy_orchestrator is not None})

    def _init_legacy_orchestrator(self) -> None:
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        self._legacy_orchestrator = MLPipelineOrchestrator(
            coverage=self.coverage, writer=self.writer, build_main=self.build_main,
            teacher_main=self.teacher_main, registry=self.registry,
            data_registry=self.data_registry, ingestor=self.ingestor, hpo_main=self.hpo_main,
            raw_writer=self.raw_writer, service=self.service, model_registry=self.model_registry,
            feature_registry=self.feature_registry, strategy_registry=self.strategy_registry,
            feature_store=self.feature_store, model_store=self.model_store,
            strategy_store=self.strategy_store, data_store=self.data_store,
            partition_manager=self.partition_manager, domain_loader=self.domain_loader,
            integration_manager_factory=self.integration_manager_factory,
            dataset_discovery=self.dataset_discovery,  # type: ignore[arg-type]
        )

    def _init_components(self) -> None:
        self._config_resolver = ConfigResolver()
        self._discovery_client = DiscoveryClient(
            dataset_discovery=self.dataset_discovery, ingestion_service=self.service)  # type: ignore[arg-type]
        self._runtime_attacher = RuntimeAttacher(
            integration_manager_factory=self.integration_manager_factory, data_registry=self.data_registry)  # type: ignore[arg-type]
        self._registry_synchronizer = RegistrySynchronizer(
            data_registry=self.data_registry, feature_registry=self.feature_registry)  # type: ignore[arg-type]
        self._dataset_builder = DatasetBuilder(
            data_store=self.data_store, data_registry=self.data_registry, build_main=self.build_main)  # type: ignore[arg-type]
        self._training_coordinator = TrainingCoordinator(
            teacher_main=self.teacher_main, hpo_main=self.hpo_main, build_artifacts=None)
        self._ingestion_coordinator = IngestionCoordinator(
            coverage=self.coverage, writer=self.writer,
            registry=self.data_registry,  # type: ignore[arg-type]
            ingestor=self.ingestor,  # type: ignore[arg-type]
            service=self.service, raw_writer=self.raw_writer,
            domain_loader=self.domain_loader, discovery_client=self._discovery_client)
        self._stage_controller = StageController(
            ingestion_coordinator=self._ingestion_coordinator, dataset_builder=self._dataset_builder,
            training_coordinator=self._training_coordinator,
            registry_synchronizer=self._registry_synchronizer, runtime_attacher=self._runtime_attacher,
            feature_registry=self.feature_registry, model_registry=self.model_registry,
            data_registry=self.data_registry, integration_manager_factory=self.integration_manager_factory)
        # Inject helper methods from legacy orchestrator for StageController
        if self._legacy_orchestrator is not None and self._use_legacy:
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
            legacy = self._legacy_orchestrator
            if isinstance(legacy, MLPipelineOrchestrator):
                self._stage_controller._prepare_dataset_config = legacy._prepare_dataset_config
                self._stage_controller._auto_fill_universe = legacy._auto_fill_universe
                self._stage_controller._handle_promotions = legacy._handle_promotions
                self._stage_controller._attach_runtime = legacy._attach_runtime

    def _get_legacy(self) -> object:
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        if not isinstance(self._legacy_orchestrator, MLPipelineOrchestrator):
            raise TypeError("Invalid legacy orchestrator type")
        return self._legacy_orchestrator

    def get_health_status(self) -> dict[str, Any]:
        """
        Get health status of the orchestrator and its components.

        Returns
        -------
        dict[str, Any]
            Health status dictionary with component availability.
        """
        if use_legacy_orchestrator():
            return cast(dict[str, Any], getattr(self._get_legacy(), "get_health_status")())

        # Match legacy structure for parity, add components as additional info
        return {
            "implementation": "component_based",
            "coverage_provider": "healthy" if self.coverage else "unavailable",
            "writer": "healthy" if self.writer else "unavailable",
            "build_main": "healthy" if self.build_main else "unavailable",
            "teacher_main": "healthy" if self.teacher_main else "unavailable",
            "has_registry": self.registry is not None,
            "has_data_registry": self.data_registry is not None,
            "has_ingestor": self.ingestor is not None,
            "has_model_registry": self.model_registry is not None,
            "has_feature_registry": self.feature_registry is not None,
            "has_strategy_registry": self.strategy_registry is not None,
            "has_integration_manager_factory": self.integration_manager_factory is not None,
        }

    # -------------------------------------------------------------------------
    # Public helpers (parity with legacy tests)
    # -------------------------------------------------------------------------

    def apply_default_market_inputs(self, cfg: DatasetBuildConfig) -> DatasetBuildConfig:
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.apply_default_market_inputs(cfg)

    def collect_symbol_map(
        self,
        *,
        ds_cfg: DatasetBuildConfig | None = None,
        symbols: tuple[str, ...] | None = None,
        instruments: tuple[str, ...] | None = None,
        instrument_ids: tuple[str, ...] | None = None,
        market_inputs: tuple[Any, ...] | None = None,
    ) -> dict[str, tuple[str, ...]]:
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.collect_symbol_map(
            ds_cfg=ds_cfg,
            symbols=symbols,
            instruments=instruments,
            instrument_ids=instrument_ids,
            market_inputs=market_inputs,
        )

    def compute_window_start_iso(
        self,
        end_iso: str,
        lookback_years: int = 1,
    ) -> str:
        if self._config_resolver is None:
            raise RuntimeError("ConfigResolver not initialized")
        return self._config_resolver.compute_window_start_iso(
            end_iso=end_iso,
            lookback_years=lookback_years,
        )

    def resolve_window_bounds_ns(self, cfg: DatasetBuildConfig) -> tuple[int, int]:
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
    ) -> tuple[Any, ...] | None:
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
    ) -> tuple[tuple[Any, ...] | None, tuple[Any, ...]]:
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
        candidates: tuple[Any, ...],
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> tuple[Any, ...]:
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

    # -------------------------------------------------------------------------
    # Public API - Pure delegation
    # -------------------------------------------------------------------------

    def run_pre_ingestion(self, *, catalog_path: Path, scheduler_cfg: SchedulerConfig,
                         options: PreIngestionOptions | None = None) -> None:
        if use_legacy_orchestrator():
            getattr(self._get_legacy(), "run_pre_ingestion")(
                catalog_path=catalog_path, scheduler_cfg=scheduler_cfg, options=options)
            return
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        self._ingestion_coordinator.run_pre_ingestion(
            catalog_path=catalog_path, scheduler_cfg=scheduler_cfg, options=options)

    def backfill(self, *, dataset_id: str, schema: str, instrument_id: str,
                lookback_days: int) -> BackfillWindowList:
        if use_legacy_orchestrator():
            result: BackfillWindowList = getattr(self._get_legacy(), "backfill")(
                dataset_id=dataset_id, schema=schema, instrument_id=instrument_id,
                lookback_days=lookback_days)
            return result
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_coordinator.backfill(
            dataset_id=dataset_id, schema=schema, instrument_id=instrument_id,
            lookback_days=lookback_days)

    def backfill_binding(self, *, binding: ResolvedMarketBinding,
                        lookback_days: int) -> dict[str, BackfillWindowList]:
        if use_legacy_orchestrator():
            result: dict[str, BackfillWindowList] = getattr(self._get_legacy(), "backfill_binding")(
                binding=binding, lookback_days=lookback_days)
            return result
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_coordinator.backfill_binding(binding=binding, lookback_days=lookback_days)

    def backfill_coverage(self, *, dataset_id: str, schema: str, instrument_id: str,
                         policy: CoveragePolicy | None = None) -> list[tuple[int, int]]:
        if use_legacy_orchestrator():
            result: list[tuple[int, int]] = getattr(self._get_legacy(), "backfill_coverage")(
                dataset_id=dataset_id, schema=schema, instrument_id=instrument_id, policy=policy)
            return result
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_coordinator.backfill_coverage(
            dataset_id=dataset_id, schema=schema, instrument_id=instrument_id, policy=policy)

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        if use_legacy_orchestrator():
            result: int = getattr(self._get_legacy(), "build_dataset")(cfg)
            return result
        if self._dataset_builder is None:
            raise RuntimeError("DatasetBuilder not initialized")
        return self._dataset_builder.build_dataset(cfg)

    def run_hpo(self, cfg: HPOConfig | None, dataset_csv: Path, out_dir: Path) -> int:
        if use_legacy_orchestrator():
            result: int = getattr(self._get_legacy(), "run_hpo")(cfg, dataset_csv, out_dir)
            return result
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.run_hpo(cfg, dataset_csv, out_dir)

    def train_teacher(self, cfg: TeacherTrainConfig | None, dataset_csv: Path, out_dir: Path) -> int:
        if use_legacy_orchestrator():
            result: int = getattr(self._get_legacy(), "train_teacher")(cfg, dataset_csv, out_dir)
            return result
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.train_teacher(cfg, dataset_csv, out_dir)

    def distill_student(
        self,
        cfg: StudentDistillConfig | None,
        dataset_dir: Path,
        teacher_cfg: TeacherTrainConfig | None,
    ) -> int:
        if use_legacy_orchestrator():
            result: int = getattr(self._get_legacy(), "distill_student")(
                cfg, dataset_dir=dataset_dir, teacher_cfg=teacher_cfg)
            return result
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.distill_student(
            cfg, dataset_dir=dataset_dir, teacher_cfg=teacher_cfg)

    def run(self, cfg: OrchestratorConfig, *, checkpoint_file: Path | None = None,
           resume: bool = False) -> int:
        if use_legacy_orchestrator():
            result: int = getattr(self._get_legacy(), "run")(
                cfg, checkpoint_file=checkpoint_file, resume=resume)
            return result
        if self._stage_controller is None:
            raise RuntimeError("StageController not initialized")
        return self._stage_controller.run_pipeline(cfg, checkpoint_file=checkpoint_file, resume=resume)

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        if use_legacy_orchestrator():
            result: int = getattr(self._get_legacy(), "run_training_only")(cfg)
            return result
        if self._stage_controller is None:
            raise RuntimeError("StageController not initialized")
        return self._stage_controller.run_training_only(cfg)

    # -------------------------------------------------------------------------
    # Legacy API static methods - Pure delegation
    # -------------------------------------------------------------------------

    @staticmethod
    def _infer_dataset_row_count(result: object) -> int | None:
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        return MLPipelineOrchestrator._infer_dataset_row_count(result)

    @staticmethod
    def _resolve_instrument_ids(dataset_cfg: DatasetBuildConfig,
                               override: tuple[str, ...] | None = None) -> tuple[str, ...]:
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        return MLPipelineOrchestrator._resolve_instrument_ids(dataset_cfg, override)

    @staticmethod
    def _infer_default_schema(cfg: DatasetBuildConfig) -> str:
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        return MLPipelineOrchestrator._infer_default_schema(cfg)

    @staticmethod
    def _binding_priority_key(binding: ResolvedMarketBinding) -> tuple[int, str]:
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        return MLPipelineOrchestrator._binding_priority_key(binding)

    @staticmethod
    def _collect_instrument_ids(bindings: tuple[ResolvedMarketBinding, ...],
                               existing: tuple[str, ...] | None) -> tuple[str, ...]:
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        return MLPipelineOrchestrator._collect_instrument_ids(bindings, existing)

    @staticmethod
    def _ns_to_datetime(value: int) -> object:
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        return MLPipelineOrchestrator._ns_to_datetime(value)
