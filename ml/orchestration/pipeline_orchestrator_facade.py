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
from typing import TYPE_CHECKING, Any, Protocol

from ml.data.ingest.subscription import SubscriptionPolicy as CoveragePolicy
from ml.orchestration.common.stage_controller import IntegrationManagerProtocol
from ml.orchestration.common.stage_controller import StageController
from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import PreIngestionOptions
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.orchestration.dataset_builder import DatasetBuilder
from ml.orchestration.discovery_client import DiscoveryClient
from ml.orchestration.ingestion_coordinator import IngestionCoordinator
from ml.orchestration.pipeline_orchestrator_facade_helpers import OrchestratorFacadeHelpers
from ml.orchestration.registry_synchronizer import RegistrySynchronizer
from ml.orchestration.runtime_attacher import RuntimeAttacher
from ml.orchestration.training_coordinator import TrainingCoordinator
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind
from ml.stores.protocols import CoverageProviderProtocol
from ml.stores.protocols import MarketDataWriterProtocol


if TYPE_CHECKING:
    from ml.config.scheduler_config import SchedulerConfig
    from ml.data.ingest.market_bindings import ResolvedMarketBinding
    from ml.data.ingest.orchestrator import BackfillWindowList
    from ml.data.ingest.orchestrator import DomainWindowLoaderProtocol
    from ml.data.ingest.service import DatabentoIngestionService
    from ml.stores.io_raw import RawIngestionWriterProtocol

logger = logging.getLogger(__name__)


class _CliMain(Protocol):
    def __call__(self, argv: list[str] | None = None) -> int: ...


@dataclass(slots=True)
class MLPipelineOrchestratorFacade(OrchestratorFacadeHelpers):
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
    _ingestion_backfill: Callable[..., BackfillWindowList] | None = field(default=None, init=False, repr=False)
    _ingestion_backfill_binding: Callable[..., dict[str, BackfillWindowList]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _ingestion_backfill_coverage: Callable[..., list[tuple[int, int]]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _ingestion_ensure_dataset_registered: Callable[..., None] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    # Components
    _ingestion_coordinator: IngestionCoordinator | None = field(default=None, init=False, repr=False)
    _dataset_builder: DatasetBuilder | None = field(default=None, init=False, repr=False)
    _training_coordinator: TrainingCoordinator | None = field(default=None, init=False, repr=False)
    _registry_synchronizer: RegistrySynchronizer | None = field(default=None, init=False, repr=False)
    _runtime_attacher: RuntimeAttacher | None = field(default=None, init=False, repr=False)
    _config_resolver: ConfigResolver | None = field(default=None, init=False, repr=False)
    _discovery_client: DiscoveryClient | None = field(default=None, init=False, repr=False)
    _stage_controller: StageController | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.registry is not None and self.data_registry is None:
            object.__setattr__(self, "data_registry", self.registry)
        self._init_components()
        logger.info(
            "MLPipelineOrchestratorFacade initialized",
            extra={"implementation": "component-based"},
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
        self._ingestion_backfill = self._ingestion_coordinator.backfill
        self._ingestion_backfill_binding = self._ingestion_coordinator.backfill_binding
        self._ingestion_backfill_coverage = self._ingestion_coordinator.backfill_coverage
        self._ingestion_ensure_dataset_registered = (
            self._ingestion_coordinator._ensure_dataset_registered
        )
        self._stage_controller = StageController(
            ingestion_coordinator=self._ingestion_coordinator, dataset_builder=self._dataset_builder,
            training_coordinator=self._training_coordinator,
            registry_synchronizer=self._registry_synchronizer, runtime_attacher=self._runtime_attacher,
            feature_registry=self.feature_registry, model_registry=self.model_registry,
            data_registry=self.data_registry, integration_manager_factory=self.integration_manager_factory)
        self._stage_controller._attach_runtime = self._attach_runtime
        self._stage_controller._prepare_dataset_config = self._prepare_dataset_config
        self._stage_controller._auto_fill_universe = self._auto_fill_universe

    def _attach_runtime(
        self,
        integration_cfg: IntegrationConfig | None,
        *,
        dataset_out_dir: Path,
    ) -> None:
        """
        Attach runtime stores/registries via the RuntimeAttacher and sync state.

        Args:
            integration_cfg: Integration configuration to apply.
            dataset_out_dir: Dataset output directory for validator context.
        """
        if self._runtime_attacher is None:
            raise RuntimeError("RuntimeAttacher not initialized")
        manager = self._runtime_attacher.attach_runtime(
            integration_cfg,
            dataset_out_dir=dataset_out_dir,
        )
        self._integration_manager = manager
        self._sync_runtime_components()

    def _sync_runtime_components(self) -> None:
        """Sync attached runtime components onto the facade and stage controller."""
        if self._runtime_attacher is None:
            return
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
            value = getattr(self._runtime_attacher, attr, None)
            if value is None:
                continue
            setattr(self, attr, value)
            if self._stage_controller is not None:
                setattr(self._stage_controller, attr, value)

    def get_health_status(self) -> dict[str, Any]:
        """
        Get health status of the orchestrator and its components.

        Returns
        -------
        dict[str, Any]
            Health status dictionary with component availability.
        """
        return self._build_health_status()

    # -------------------------------------------------------------------------
    # Public API - Pure delegation
    # -------------------------------------------------------------------------

    def run_pre_ingestion(self, *, catalog_path: Path, scheduler_cfg: SchedulerConfig,
                         options: PreIngestionOptions | None = None) -> None:
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        self._ingestion_coordinator.run_pre_ingestion(
            catalog_path=catalog_path, scheduler_cfg=scheduler_cfg, options=options)

    def backfill(self, *, dataset_id: str, schema: str, instrument_id: str,
                lookback_days: int) -> BackfillWindowList:
        if self._ingestion_backfill is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_backfill(
            dataset_id=dataset_id, schema=schema, instrument_id=instrument_id,
            lookback_days=lookback_days)

    def backfill_binding(self, *, binding: ResolvedMarketBinding,
                        lookback_days: int) -> dict[str, BackfillWindowList]:
        if self._ingestion_backfill_binding is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_backfill_binding(binding=binding, lookback_days=lookback_days)

    def backfill_coverage(self, *, dataset_id: str, schema: str, instrument_id: str,
                         policy: CoveragePolicy | None = None) -> list[tuple[int, int]]:
        if self._ingestion_backfill_coverage is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_backfill_coverage(
            dataset_id=dataset_id, schema=schema, instrument_id=instrument_id, policy=policy)

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        if self._dataset_builder is None:
            raise RuntimeError("DatasetBuilder not initialized")
        return self._dataset_builder.build_dataset(cfg)

    def run_hpo(self, cfg: HPOConfig | None, dataset_csv: Path, out_dir: Path) -> int:
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.run_hpo(cfg, dataset_csv, out_dir)

    def train_teacher(self, cfg: TeacherTrainConfig | None, dataset_csv: Path, out_dir: Path) -> int:
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.train_teacher(cfg, dataset_csv, out_dir)

    def distill_student(
        self,
        cfg: StudentDistillConfig | None,
        dataset_dir: Path,
        teacher_cfg: TeacherTrainConfig | None,
    ) -> int:
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.distill_student(
            cfg, dataset_dir=dataset_dir, teacher_cfg=teacher_cfg)

    def run(self, cfg: OrchestratorConfig, *, checkpoint_file: Path | None = None,
           resume: bool = False) -> int:
        if self._stage_controller is None:
            raise RuntimeError("StageController not initialized")
        return self._stage_controller.run_pipeline(cfg, checkpoint_file=checkpoint_file, resume=resume)

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        if self._stage_controller is None:
            raise RuntimeError("StageController not initialized")
        return self._stage_controller.run_training_only(cfg)

    def _ensure_dataset_registered(
        self,
        *,
        dataset_id: str,
        dataset_type: DatasetType,
        location: str,
        storage_kind: StorageKind = StorageKind.PARQUET,
    ) -> None:
        """
        Ensure dataset registration is delegated to the ingestion coordinator.

        Args:
            dataset_id: Dataset identifier to register.
            dataset_type: Dataset type metadata.
            location: Storage location string.
            storage_kind: Storage kind enum (default PARQUET).
        """
        if self._ingestion_ensure_dataset_registered is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        self._ingestion_ensure_dataset_registered(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            location=location,
            storage_kind=storage_kind,
        )
