#!/usr/bin/env python3

"""
MLPipelineOrchestratorFacade - Facade for the decomposed MLPipelineOrchestrator.

This facade maintains 100% API parity with the legacy MLPipelineOrchestrator while
internally delegating to 7 specialized components extracted during Phase 2.2:

Components:
- IngestionCoordinator: run_pre_ingestion(), backfill*() methods
- DatasetBuilder: build_dataset() method
- TrainingCoordinator: train_teacher(), distill_student(), run_hpo() methods
- RegistrySynchronizer: Registry operations during pipeline
- RuntimeAttacher: Runtime attachment operations
- ConfigResolver: Configuration parsing and validation
- DiscoveryClient: Service discovery operations

The facade uses the delegation pattern to preserve behavioral parity while enabling
better testability, maintainability, and adherence to Single Responsibility Principle.

Phase 2.2.8 Implementation Notes:
- All public methods delegate to the legacy orchestrator for behavioral parity
- Components are initialized but used through orchestrator delegation
- This ensures zero behavioral regression during gradual migration
- Future phases will migrate logic fully into components

Examples
--------
>>> from ml.orchestration.pipeline_orchestrator_facade import MLPipelineOrchestratorFacade
>>> facade = MLPipelineOrchestratorFacade(
...     coverage=coverage_provider,
...     writer=market_data_writer,
...     build_main=build_main_func,
...     teacher_main=teacher_main_func,
... )
>>> # All legacy APIs work unchanged
>>> facade.build_dataset(config)
>>> facade.run(orchestrator_config)

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ml.data.ingest.subscription import SubscriptionPolicy as CoveragePolicy
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
    """
    Protocol for CLI main callable (matches legacy MLPipelineOrchestrator).
    """

    def __call__(self, argv: list[str] | None = None) -> int:
        """
        Execute CLI with optional arguments.
        """
        ...


class IntegrationManagerProtocol(Protocol):
    """
    Protocol for MLIntegrationManager (matches legacy MLPipelineOrchestrator).
    """

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
class MLPipelineOrchestratorFacade:
    """
    Facade for MLPipelineOrchestrator that delegates to 7 extracted components.

    This facade maintains 100% API parity with the legacy MLPipelineOrchestrator
    while internally using specialized components for better separation of concerns.

    The facade uses the delegation pattern: all public methods delegate to the
    legacy MLPipelineOrchestrator to ensure behavioral parity during migration.
    Components are initialized and available for future full migration.

    Attributes
    ----------
    coverage : CoverageProviderProtocol
        Coverage provider for data availability queries
    writer : MarketDataWriterProtocol
        Market data writer for persisting data
    build_main : _CliMain
        CLI main function for dataset building
    teacher_main : _CliMain
        CLI main function for teacher model training
    registry : object | None
        Backward compatibility alias for data_registry
    data_registry : object | None
        Registry for dataset manifests
    ingestor : object | None
        Databento ingestor or similar
    hpo_main : _CliMain | None
        Optional CLI main function for HPO
    raw_writer : RawIngestionWriterProtocol | None
        Optional raw data writer
    service : DatabentoIngestionService | None
        Optional Databento ingestion service
    model_registry : object | None
        Registry for model metadata
    feature_registry : object | None
        Registry for feature schemas
    strategy_registry : object | None
        Registry for strategy manifests
    feature_store : object | None
        Store for features
    model_store : object | None
        Store for models
    strategy_store : object | None
        Store for strategy state
    data_store : object | None
        Store for data
    partition_manager : object | None
        Partition manager for data partitioning
    domain_loader : DomainWindowLoaderProtocol | None
        Domain window loader for time-based queries
    integration_manager_factory : Callable[..., IntegrationManagerProtocol] | None
        Factory for creating integration managers
    dataset_discovery : DatasetDiscoveryService | None
        Dataset discovery service

    Examples
    --------
    >>> facade = MLPipelineOrchestratorFacade(
    ...     coverage=coverage_provider,
    ...     writer=market_data_writer,
    ...     build_main=build_main_func,
    ...     teacher_main=teacher_main_func,
    ... )
    >>> # Use exactly like legacy MLPipelineOrchestrator
    >>> facade.build_dataset(dataset_config)
    >>> facade.run(orchestrator_config)

    """

    # Required attributes (same as legacy MLPipelineOrchestrator)
    coverage: CoverageProviderProtocol
    writer: MarketDataWriterProtocol
    build_main: _CliMain
    teacher_main: _CliMain

    # Optional attributes with defaults (same as legacy MLPipelineOrchestrator)
    registry: object | None = None  # Backward compatibility alias
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
    dataset_discovery: DatasetDiscoveryService | None = None

    # Internal: write mode tokens (non-init, matches legacy)
    write_mode_tokens: tuple[str, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
    )

    # Internal: integration manager instance (non-init, matches legacy)
    _integration_manager: IntegrationManagerProtocol | None = field(
        default=None,
        init=False,
        repr=False,
    )

    # Internal: legacy orchestrator for delegation (non-init)
    _legacy_orchestrator: object | None = field(
        default=None,
        init=False,
        repr=False,
    )

    # Internal: 7 extracted components (non-init)
    _ingestion_coordinator: IngestionCoordinator | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _dataset_builder: DatasetBuilder | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _training_coordinator: TrainingCoordinator | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _registry_synchronizer: RegistrySynchronizer | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _runtime_attacher: RuntimeAttacher | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _config_resolver: ConfigResolver | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _discovery_client: DiscoveryClient | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Initialize facade with backward compatibility handling and component setup.

        Handles:
        1. registry -> data_registry backward compatibility alias
        2. Legacy orchestrator initialization for delegation
        3. Component initialization for future migration

        """
        # Handle backward compatibility: registry alias -> data_registry
        if self.registry is not None and self.data_registry is None:
            object.__setattr__(self, "data_registry", self.registry)

        # Initialize legacy orchestrator for delegation (ensures behavioral parity)
        self._init_legacy_orchestrator()

        # Initialize components (for future full migration)
        self._init_components()

        logger.info(
            "MLPipelineOrchestratorFacade initialized",
            extra={
                "has_legacy_orchestrator": self._legacy_orchestrator is not None,
                "has_data_registry": self.data_registry is not None,
                "has_feature_registry": self.feature_registry is not None,
                "has_model_registry": self.model_registry is not None,
                "has_data_store": self.data_store is not None,
            },
        )

    def _init_legacy_orchestrator(self) -> None:
        """
        Initialize the legacy MLPipelineOrchestrator for delegation.

        All public methods delegate to legacy orchestrator to ensure 100% behavioral
        parity during the migration period.

        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        self._legacy_orchestrator = MLPipelineOrchestrator(
            coverage=self.coverage,
            writer=self.writer,
            build_main=self.build_main,
            teacher_main=self.teacher_main,
            registry=self.registry,
            data_registry=self.data_registry,
            ingestor=self.ingestor,
            hpo_main=self.hpo_main,
            raw_writer=self.raw_writer,
            service=self.service,
            model_registry=self.model_registry,
            feature_registry=self.feature_registry,
            strategy_registry=self.strategy_registry,
            feature_store=self.feature_store,
            model_store=self.model_store,
            strategy_store=self.strategy_store,
            data_store=self.data_store,
            partition_manager=self.partition_manager,
            domain_loader=self.domain_loader,
            integration_manager_factory=self.integration_manager_factory,
            dataset_discovery=self.dataset_discovery,
        )

    def _init_components(self) -> None:
        """
        Initialize the 7 extracted components.

        All components are always initialized (with placeholder or mock dependencies
        when real ones are not available). This ensures consistent behavior and allows
        future migration to component-based implementation.

        Operations still delegate to the legacy orchestrator for behavioral parity.

        """
        # Initialize ConfigResolver (no dependencies on other components)
        self._config_resolver = ConfigResolver()

        # Initialize DiscoveryClient (no dependencies on other components)
        self._discovery_client = DiscoveryClient(
            dataset_discovery=self.dataset_discovery,
            ingestion_service=self.service,
        )

        # Initialize RuntimeAttacher
        self._runtime_attacher = RuntimeAttacher(
            integration_manager_factory=self.integration_manager_factory,
            data_registry=self.data_registry,  # type: ignore[arg-type]
        )

        # Initialize RegistrySynchronizer
        self._registry_synchronizer = RegistrySynchronizer(
            data_registry=self.data_registry,  # type: ignore[arg-type]
            feature_registry=self.feature_registry,
        )

        # Initialize DatasetBuilder (root module)
        self._dataset_builder = DatasetBuilder(
            data_store=self.data_store,  # type: ignore[arg-type]
            data_registry=self.data_registry,  # type: ignore[arg-type]
            build_main=self.build_main,
        )

        # Initialize TrainingCoordinator
        self._training_coordinator = TrainingCoordinator(
            teacher_main=self.teacher_main,
            hpo_main=self.hpo_main,
            build_artifacts=None,  # Will be set after dataset build
        )

        # Initialize IngestionCoordinator (root module - needs coverage, writer, etc.)
        self._ingestion_coordinator = IngestionCoordinator(
            coverage=self.coverage,
            writer=self.writer,
            registry=self.data_registry,  # type: ignore[arg-type]
            ingestor=self.ingestor,  # type: ignore[arg-type]
            service=self.service,
            raw_writer=self.raw_writer,
            domain_loader=self.domain_loader,
            discovery_client=self._discovery_client,
        )

        logger.debug(
            "Components initialized",
            extra={
                "has_config_resolver": self._config_resolver is not None,
                "has_discovery_client": self._discovery_client is not None,
                "has_registry_synchronizer": self._registry_synchronizer is not None,
                "has_runtime_attacher": self._runtime_attacher is not None,
                "has_dataset_builder": self._dataset_builder is not None,
                "has_training_coordinator": self._training_coordinator is not None,
                "has_ingestion_coordinator": self._ingestion_coordinator is not None,
            },
        )

    # =========================================================================
    # Public API Methods (delegate to legacy orchestrator for parity)
    # =========================================================================

    def run_pre_ingestion(
        self,
        *,
        catalog_path: Path,
        scheduler_cfg: SchedulerConfig,
        options: PreIngestionOptions | None = None,
    ) -> None:
        """
        Run data ingestion pre-stage using DataScheduler in orchestrator mode.

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        catalog_path : Path
            Path to Parquet catalog
        scheduler_cfg : SchedulerConfig
            Scheduler configuration
        options : PreIngestionOptions | None, optional
            Pre-ingestion options (dual-write, metrics, etc.)

        Examples
        --------
        >>> facade.run_pre_ingestion(
        ...     catalog_path=Path("/data/catalog"),
        ...     scheduler_cfg=scheduler_config,
        ...     options=PreIngestionOptions(dual_write=True),
        ... )

        """
        # Use legacy mode for parity testing
        if use_legacy_orchestrator():
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            legacy = self._legacy_orchestrator
            if not isinstance(legacy, MLPipelineOrchestrator):
                raise TypeError("Invalid legacy orchestrator type")
            legacy.run_pre_ingestion(
                catalog_path=catalog_path,
                scheduler_cfg=scheduler_cfg,
                options=options,
            )
            return

        # Use root module (canonical path)
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        self._ingestion_coordinator.run_pre_ingestion(
            catalog_path=catalog_path,
            scheduler_cfg=scheduler_cfg,
            options=options,
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
        Backfill gaps for a single instrument.

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier (e.g., "databento.ohlcv-1s")
        schema : str
            Schema name (e.g., "ohlcv-1m", "tbbo")
        instrument_id : str
            Instrument ID (e.g., "SPY.NASDAQ")
        lookback_days : int
            Number of days to look back

        Returns
        -------
        BackfillWindowList
            Backfill result with rows_written and window_count

        Examples
        --------
        >>> result = facade.backfill(
        ...     dataset_id="databento.ohlcv-1s",
        ...     schema="ohlcv-1s",
        ...     instrument_id="SPY.NASDAQ",
        ...     lookback_days=30,
        ... )

        """
        # Use legacy mode for parity testing
        if use_legacy_orchestrator():
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            legacy = self._legacy_orchestrator
            if not isinstance(legacy, MLPipelineOrchestrator):
                raise TypeError("Invalid legacy orchestrator type")
            return legacy.backfill(
                dataset_id=dataset_id,
                schema=schema,
                instrument_id=instrument_id,
                lookback_days=lookback_days,
            )

        # Use root module (canonical path)
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_coordinator.backfill(
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
        Backfill using resolved market binding.

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        binding : ResolvedMarketBinding
            Resolved market binding
        lookback_days : int
            Number of days to look back

        Returns
        -------
        dict[str, BackfillWindowList]
            Map of instrument_id to backfill results

        Examples
        --------
        >>> result = facade.backfill_binding(
        ...     binding=resolved_binding,
        ...     lookback_days=30,
        ... )

        """
        # Use legacy mode for parity testing
        if use_legacy_orchestrator():
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            legacy = self._legacy_orchestrator
            if not isinstance(legacy, MLPipelineOrchestrator):
                raise TypeError("Invalid legacy orchestrator type")
            return legacy.backfill_binding(
                binding=binding,
                lookback_days=lookback_days,
            )

        # Use root module (canonical path)
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_coordinator.backfill_binding(
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

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Schema name
        instrument_id : str
            Instrument ID
        policy : CoveragePolicy | None, optional
            Coverage policy for determining lookback bounds

        Returns
        -------
        list[tuple[int, int]]
            List of (start_ns, end_ns) windows

        Examples
        --------
        >>> windows = facade.backfill_coverage(
        ...     dataset_id="databento.ohlcv-1s",
        ...     schema="ohlcv-1s",
        ...     instrument_id="SPY.NASDAQ",
        ...     policy=coverage_policy,
        ... )

        """
        # Use legacy mode for parity testing
        if use_legacy_orchestrator():
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            legacy = self._legacy_orchestrator
            if not isinstance(legacy, MLPipelineOrchestrator):
                raise TypeError("Invalid legacy orchestrator type")
            return legacy.backfill_coverage(
                dataset_id=dataset_id,
                schema=schema,
                instrument_id=instrument_id,
                policy=policy,
            )

        # Use root module (canonical path)
        if self._ingestion_coordinator is None:
            raise RuntimeError("IngestionCoordinator not initialized")
        return self._ingestion_coordinator.backfill_coverage(
            dataset_id=dataset_id,
            schema=schema,
            instrument_id=instrument_id,
            policy=policy,
        )

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        """
        Build dataset according to configuration.

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)

        Raises
        ------
        _EmptyDatasetError
            If dataset build produces zero rows

        Examples
        --------
        >>> result = facade.build_dataset(dataset_config)
        >>> assert result == 0  # Success

        """
        # Use legacy mode for parity testing
        if use_legacy_orchestrator():
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            legacy = self._legacy_orchestrator
            if not isinstance(legacy, MLPipelineOrchestrator):
                raise TypeError("Invalid legacy orchestrator type")
            return legacy.build_dataset(cfg)

        # Use root module (canonical path)
        if self._dataset_builder is None:
            raise RuntimeError("DatasetBuilder not initialized")
        return self._dataset_builder.build_dataset(cfg)

    def run_hpo(
        self,
        cfg: HPOConfig | None,
        dataset_csv: Path,
        out_dir: Path,
    ) -> int:
        """
        Run hyperparameter optimization.

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        cfg : HPOConfig | None
            HPO configuration (None or disabled skips HPO)
        dataset_csv : Path
            Path to dataset CSV file
        out_dir : Path
            Output directory for HPO results

        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)

        Examples
        --------
        >>> result = facade.run_hpo(
        ...     cfg=hpo_config,
        ...     dataset_csv=Path("/data/dataset.csv"),
        ...     out_dir=Path("/output"),
        ... )

        """
        # Use legacy mode for parity testing
        if use_legacy_orchestrator():
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            legacy = self._legacy_orchestrator
            if not isinstance(legacy, MLPipelineOrchestrator):
                raise TypeError("Invalid legacy orchestrator type")
            return legacy.run_hpo(cfg, dataset_csv, out_dir)

        # Use root module (canonical path)
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.run_hpo(cfg, dataset_csv, out_dir)

    def train_teacher(
        self,
        cfg: TeacherTrainConfig | None,
        dataset_csv: Path,
        out_dir: Path,
    ) -> int:
        """
        Train teacher model.

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        cfg : TeacherTrainConfig | None
            Teacher training configuration (None or disabled skips training)
        dataset_csv : Path
            Path to dataset CSV file
        out_dir : Path
            Output directory for model artifacts

        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)

        Raises
        ------
        FileNotFoundError
            If dataset metadata is missing

        Examples
        --------
        >>> result = facade.train_teacher(
        ...     cfg=teacher_config,
        ...     dataset_csv=Path("/data/dataset.csv"),
        ...     out_dir=Path("/output"),
        ... )

        """
        # Use legacy mode for parity testing
        if use_legacy_orchestrator():
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            legacy = self._legacy_orchestrator
            if not isinstance(legacy, MLPipelineOrchestrator):
                raise TypeError("Invalid legacy orchestrator type")
            return legacy.train_teacher(cfg, dataset_csv, out_dir)

        # Use root module (canonical path)
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.train_teacher(cfg, dataset_csv, out_dir)

    def distill_student(
        self,
        cfg: StudentDistillConfig | None,
        *,
        dataset_dir: Path,
        teacher_cfg: TeacherTrainConfig | None,
    ) -> int:
        """
        Train student model via knowledge distillation.

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        cfg : StudentDistillConfig | None
            Student distillation configuration (None or disabled skips distillation)
        dataset_dir : Path
            Directory containing dataset artifacts (features_npz.npz, teacher_preds.npz)
        teacher_cfg : TeacherTrainConfig | None
            Teacher configuration for parent model ID

        Returns
        -------
        int
            Exit code (0 for success, 1 for failure)

        Examples
        --------
        >>> result = facade.distill_student(
        ...     cfg=student_config,
        ...     dataset_dir=Path("/data/dataset"),
        ...     teacher_cfg=teacher_config,
        ... )

        """
        # Use legacy mode for parity testing
        if use_legacy_orchestrator():
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            legacy = self._legacy_orchestrator
            if not isinstance(legacy, MLPipelineOrchestrator):
                raise TypeError("Invalid legacy orchestrator type")
            return legacy.distill_student(
                cfg,
                dataset_dir=dataset_dir,
                teacher_cfg=teacher_cfg,
            )

        # Use root module (canonical path)
        if self._training_coordinator is None:
            raise RuntimeError("TrainingCoordinator not initialized")
        return self._training_coordinator.distill_student(
            cfg,
            dataset_dir=dataset_dir,
            teacher_cfg=teacher_cfg,
        )

    def run(
        self,
        cfg: OrchestratorConfig,
        *,
        checkpoint_file: Path | None = None,
        resume: bool = False,
    ) -> int:
        """
        Run the complete ML pipeline with optional checkpoint support.

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Supports both single-symbol and multi-symbol processing with result isolation.

        Parameters
        ----------
        cfg : OrchestratorConfig
            Orchestrator configuration
        checkpoint_file : Path | None, optional
            Path to checkpoint file for saving/loading state
        resume : bool, optional
            If True and checkpoint_file exists, resume from saved checkpoint

        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)

        Examples
        --------
        >>> # Single-symbol run
        >>> result = facade.run(orchestrator_config)
        >>>
        >>> # Multi-symbol run with checkpoint support
        >>> result = facade.run(
        ...     orchestrator_config,
        ...     checkpoint_file=Path("/tmp/checkpoint.json"),
        ...     resume=True,
        ... )

        """
        # Full pipeline orchestration - delegates to legacy for now
        # Individual methods (build_dataset, train_teacher, etc.) use root modules
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        legacy = self._legacy_orchestrator
        if not isinstance(legacy, MLPipelineOrchestrator):
            raise TypeError("Invalid legacy orchestrator type")

        return legacy.run(
            cfg,
            checkpoint_file=checkpoint_file,
            resume=resume,
        )

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        """
        Run training-only pipeline (skips dataset build).

        Delegates to legacy MLPipelineOrchestrator for behavioral parity.

        Parameters
        ----------
        cfg : OrchestratorConfig
            Orchestrator configuration with existing dataset

        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)

        Raises
        ------
        FileNotFoundError
            If dataset CSV is not found in expected location

        Examples
        --------
        >>> result = facade.run_training_only(orchestrator_config)

        """
        # Training pipeline orchestration - delegates to legacy for now
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        legacy = self._legacy_orchestrator
        if not isinstance(legacy, MLPipelineOrchestrator):
            raise TypeError("Invalid legacy orchestrator type")

        return legacy.run_training_only(cfg)

    # =========================================================================
    # Legacy API Helper Methods (delegate for parity)
    # =========================================================================

    @staticmethod
    def _infer_dataset_row_count(result: object) -> int | None:
        """
        Best-effort row count inference for API build results.

        Delegates to legacy MLPipelineOrchestrator static method for parity.

        Parameters
        ----------
        result : object
            Build result object

        Returns
        -------
        int | None
            Inferred row count or None if cannot infer

        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        return MLPipelineOrchestrator._infer_dataset_row_count(result)

    @staticmethod
    def _resolve_instrument_ids(
        dataset_cfg: DatasetBuildConfig,
        override: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """
        Resolve instrument IDs from configuration.

        Delegates to legacy MLPipelineOrchestrator static method for parity.

        Parameters
        ----------
        dataset_cfg : DatasetBuildConfig
            Dataset configuration
        override : tuple[str, ...] | None, optional
            Optional override instrument IDs

        Returns
        -------
        tuple[str, ...]
            Resolved instrument IDs

        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        return MLPipelineOrchestrator._resolve_instrument_ids(dataset_cfg, override)

    @staticmethod
    def _infer_default_schema(cfg: DatasetBuildConfig) -> str:
        """
        Infer default schema for discovery lookups.

        Delegates to legacy MLPipelineOrchestrator static method for parity.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        str
            Inferred default schema (e.g., "ohlcv-1m")

        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        return MLPipelineOrchestrator._infer_default_schema(cfg)

    @staticmethod
    def _binding_priority_key(binding: ResolvedMarketBinding) -> tuple[int, str]:
        """
        Priority key for binding selection.

        Delegates to legacy MLPipelineOrchestrator static method for parity.

        Parameters
        ----------
        binding : ResolvedMarketBinding
            Market binding to compute priority for

        Returns
        -------
        tuple[int, str]
            Priority key (lower is higher priority)

        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        return MLPipelineOrchestrator._binding_priority_key(binding)

    @staticmethod
    def _collect_instrument_ids(
        bindings: tuple[ResolvedMarketBinding, ...],
        existing: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """
        Collect instrument IDs from bindings.

        Delegates to legacy MLPipelineOrchestrator static method for parity.

        Parameters
        ----------
        bindings : tuple[ResolvedMarketBinding, ...]
            Market bindings to collect IDs from
        existing : tuple[str, ...] | None
            Existing instrument IDs to merge

        Returns
        -------
        tuple[str, ...]
            Deduplicated instrument IDs

        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        return MLPipelineOrchestrator._collect_instrument_ids(bindings, existing)

    @staticmethod
    def _ns_to_datetime(value: int) -> object:
        """
        Convert nanoseconds since epoch to datetime.

        Delegates to legacy MLPipelineOrchestrator static method for parity.

        Parameters
        ----------
        value : int
            Nanoseconds since epoch

        Returns
        -------
        datetime
            UTC datetime object

        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        return MLPipelineOrchestrator._ns_to_datetime(value)
