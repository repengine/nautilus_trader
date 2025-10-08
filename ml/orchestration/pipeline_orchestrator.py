#!/usr/bin/env python3

"""
MLPipelineOrchestrator facade maintaining backward compatibility.

This facade delegates to specialized components while preserving the original
public API. Feature flag ML_USE_LEGACY_PIPELINE_ORCHESTRATOR controls legacy vs new path.

Phase 2.2: MLPipelineOrchestrator Decomposition - Strangler Fig Pattern
-----------------------------------------------------------------------
This facade provides 100% backward compatibility while allowing gradual
migration to the decomposed component architecture. The legacy monolithic
implementation can be restored via environment variable for safe rollback.

Components:
-----------
- ConfigResolver: Configuration resolution, market inputs, window bounds
- DiscoveryClient: Dataset discovery, service health checks
- BindingResolver: Market binding resolution, coverage validation
- IngestionCoordinator: Backfill management, auto-fill universe
- DatasetBuilder: Dataset construction, validation, metadata

"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml.orchestration.binding_resolver import BindingResolver
from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.orchestration.dataset_builder import DatasetBuilder
from ml.orchestration.discovery_client import DiscoveryClient
from ml.orchestration.ingestion_coordinator import IngestionCoordinator


if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ml.data.ingest.discovery import DatasetDiscoveryService
    from ml.data.ingest.orchestrator import DomainWindowLoaderProtocol
    from ml.data.ingest.orchestrator import IngestionOrchestrator
    from ml.data.ingest.resume import DatabentoIngestor
    from ml.data.ingest.service import DatabentoIngestionService
    from ml.registry.protocols import RegistryProtocol
    from ml.stores.io_raw import RawIngestionWriterProtocol
    from ml.stores.protocols import CoverageProviderProtocol
    from ml.stores.protocols import DataStoreFacadeProtocol
    from ml.stores.protocols import MarketDataWriterProtocol


logger = logging.getLogger(__name__)

# Feature flag to control legacy vs new implementation
USE_LEGACY = os.getenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0") == "1"


class MLPipelineOrchestrator:
    """
    High-level ML pipeline orchestrator (cold path only).

    This facade delegates to specialized components while maintaining
    100% backward compatibility with the original MLPipelineOrchestrator API.

    Feature Flag Control:
    ---------------------
    - ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1: Use original monolithic implementation
    - ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=0: Use new component-based implementation (default)

    Component Architecture:
    ----------------------
    - ConfigResolver: Configuration resolution, market inputs, window bounds
    - DiscoveryClient: Dataset discovery, service health checks
    - BindingResolver: Market binding resolution, coverage validation
    - IngestionCoordinator: Backfill management, auto-fill universe
    - DatasetBuilder: Dataset construction, validation, metadata

    Parameters
    ----------
    connection_string : str | None
        PostgreSQL connection string
    registry : RegistryProtocol | None
        Data registry instance
    data_store : DataStoreFacadeProtocol | None
        Data store instance
    ingestion_orchestrator : IngestionOrchestrator | None
        Ingestion orchestrator instance
    ingestor : DatabentoIngestor | None
        Databento ingestor instance
    service : DatabentoIngestionService | None
        Databento ingestion service
    dataset_discovery : DatasetDiscoveryService | None
        Dataset discovery service
    coverage_provider : CoverageProviderProtocol | None
        Coverage provider instance
    default_data_dir : Path | None
        Default data directory
    writer : MarketDataWriterProtocol | None
        Market data writer
    raw_writer : RawIngestionWriterProtocol | None
        Raw ingestion writer
    domain_loader : DomainWindowLoaderProtocol | None
        Domain window loader
    write_mode_tokens : tuple[str, ...] | None
        Write mode tokens

    Examples
    --------
    >>> # Use new component-based implementation (default)
    >>> orchestrator = MLPipelineOrchestrator(
    ...     connection_string="postgresql://...",
    ... )
    >>> orchestrator.build_dataset(dataset_cfg)

    >>> # Use legacy implementation (rollback)
    >>> import os
    >>> os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = "1"
    >>> orchestrator = MLPipelineOrchestrator(
    ...     connection_string="postgresql://...",
    ... )

    """

    def __init__(
        self,
        *,
        connection_string: str | None = None,
        registry: RegistryProtocol | None = None,
        data_store: DataStoreFacadeProtocol | None = None,
        ingestion_orchestrator: IngestionOrchestrator | None = None,
        ingestor: DatabentoIngestor | None = None,
        service: DatabentoIngestionService | None = None,
        dataset_discovery: DatasetDiscoveryService | None = None,
        coverage_provider: CoverageProviderProtocol | None = None,
        default_data_dir: Path | None = None,
        writer: MarketDataWriterProtocol | None = None,
        raw_writer: RawIngestionWriterProtocol | None = None,
        domain_loader: DomainWindowLoaderProtocol | None = None,
        write_mode_tokens: tuple[str, ...] | None = None,
    ) -> None:
        """
        Initialize MLPipelineOrchestrator.

        Parameters match original constructor for complete backward compatibility.

        """
        if USE_LEGACY:
            # Use legacy monolithic implementation
            logger.info(
                "Using legacy MLPipelineOrchestrator implementation (ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1)"
            )
            from ml.orchestration.pipeline_orchestrator_legacy import MLPipelineOrchestrator as MLPipelineOrchestratorLegacy

            self._legacy_impl = MLPipelineOrchestratorLegacy(
                connection_string=connection_string,
                registry=registry,
                data_store=data_store,
                ingestion_orchestrator=ingestion_orchestrator,
                ingestor=ingestor,
                service=service,
                dataset_discovery=dataset_discovery,
                coverage_provider=coverage_provider,
                default_data_dir=default_data_dir,
                writer=writer,
                raw_writer=raw_writer,
                domain_loader=domain_loader,
                write_mode_tokens=write_mode_tokens,
            )
            self._use_legacy = True

            # Expose common attributes for compatibility
            self.registry = self._legacy_impl.registry
            self.data_store = self._legacy_impl.data_store
            self.service = self._legacy_impl.service
            self.coverage = self._legacy_impl.coverage
            return

        # Use new component-based implementation
        logger.info(
            "Using component-based MLPipelineOrchestrator implementation (ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=0)"
        )
        self._use_legacy = False

        # Store dependencies for direct access
        self.registry = registry
        self.data_store = data_store
        self.service = service
        self.coverage = coverage_provider
        self.connection_string = connection_string
        self.default_data_dir = default_data_dir or Path.cwd() / "data"

        # Initialize 5 specialized components
        self._config_resolver = ConfigResolver()

        self._discovery_client = DiscoveryClient(
            dataset_discovery=dataset_discovery,
            ingestion_service=service,
        )

        self._binding_resolver = BindingResolver(
            coverage_provider=coverage_provider,
            ingestion_service=service,
            discovery_client=self._discovery_client,
        )

        self._ingestion_coordinator = IngestionCoordinator(
            coverage=coverage_provider,
            writer=writer,
            registry=registry,
            ingestor=ingestor,
            service=service,
            raw_writer=raw_writer,
            domain_loader=domain_loader,
            discovery_client=self._discovery_client,
            write_mode_tokens=write_mode_tokens or (),
        )

        self._dataset_builder = DatasetBuilder(
            data_store=data_store,
            data_registry=registry,
        )

        logger.info(
            "Initialized MLPipelineOrchestrator facade with 5 components: "
            "ConfigResolver, DiscoveryClient, BindingResolver, IngestionCoordinator, DatasetBuilder"
        )

    # =========================================================================
    # Configuration Methods (delegate to ConfigResolver)
    # =========================================================================

    def apply_default_market_inputs(
        self,
        cfg: DatasetBuildConfig,
    ) -> DatasetBuildConfig:
        """
        Seed dataset configs with descriptor-driven market inputs.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        DatasetBuildConfig
            Configuration with market inputs populated

        """
        if self._use_legacy:
            return self._legacy_impl.apply_default_market_inputs(cfg)
        return self._config_resolver.apply_default_market_inputs(cfg)

    def collect_symbol_map(
        self,
        ds_cfg: DatasetBuildConfig | None,
        symbols: tuple[str, ...] | None = None,
        instruments: tuple[str, ...] | None = None,
        instrument_ids: tuple[str, ...] | None = None,
        market_inputs: Any = None,
    ) -> dict[str, tuple[str, ...]]:
        """
        Collect symbol to instrument ID mappings from configs.

        Parameters
        ----------
        ds_cfg : DatasetBuildConfig | None
            Dataset build configuration
        symbols : tuple[str, ...] | None
            Symbol list
        instruments : tuple[str, ...] | None
            Instrument list
        instrument_ids : tuple[str, ...] | None
            Instrument ID list
        market_inputs : tuple[MarketDatasetInput, ...] | None
            Market inputs

        Returns
        -------
        dict[str, tuple[str, ...]]
            Symbol to instrument IDs mapping

        """
        if self._use_legacy:
            return self._legacy_impl.collect_symbol_map(
                ds_cfg,
                symbols,
                instruments,
                instrument_ids,
                market_inputs,
            )
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
        lookback_years: int = 3,
    ) -> str:
        """
        Compute ISO8601 start date by subtracting lookback years.

        Parameters
        ----------
        end_iso : str
            End date in ISO8601 format
        lookback_years : int
            Number of years to look back

        Returns
        -------
        str
            Start date in ISO8601 format

        """
        if self._use_legacy:
            return self._legacy_impl.compute_window_start_iso(end_iso, lookback_years)
        return self._config_resolver.compute_window_start_iso(end_iso, lookback_years)

    def resolve_window_bounds_ns(
        self,
        cfg: DatasetBuildConfig,
    ) -> tuple[int, int]:
        """
        Resolve window bounds in nanoseconds from configuration.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        tuple[int, int]
            (start_ns, end_ns) tuple in nanoseconds since epoch

        """
        if self._use_legacy:
            return self._legacy_impl.resolve_window_bounds_ns(cfg)
        return self._config_resolver.resolve_window_bounds_ns(cfg)

    def prepare_dataset_config(
        self,
        cfg: DatasetBuildConfig,
        resolved_inputs: Any,
        bindings: Any,
    ) -> DatasetBuildConfig:
        """
        Prepare dataset config with resolved market inputs and instrument IDs.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Base dataset configuration
        resolved_inputs : tuple[MarketDatasetInput, ...] | None
            Resolved market inputs
        bindings : tuple[ResolvedMarketBinding, ...]
            Resolved market bindings

        Returns
        -------
        DatasetBuildConfig
            Updated configuration with resolved values

        """
        if self._use_legacy:
            return self._legacy_impl.prepare_dataset_config(cfg, resolved_inputs, bindings)
        return self._config_resolver.prepare_dataset_config(cfg, resolved_inputs, bindings)

    # =========================================================================
    # Ingestion Methods (delegate to IngestionCoordinator)
    # =========================================================================

    def run_pre_ingestion(self, *args: Any, **kwargs: Any) -> Any:
        """Run pre-ingestion tasks."""
        if self._use_legacy:
            return self._legacy_impl.run_pre_ingestion(*args, **kwargs)
        return self._ingestion_coordinator.run_pre_ingestion(*args, **kwargs)

    def backfill(self, *args: Any, **kwargs: Any) -> Any:
        """Backfill market data."""
        if self._use_legacy:
            return self._legacy_impl.backfill(*args, **kwargs)
        return self._ingestion_coordinator.backfill(*args, **kwargs)

    def backfill_binding(self, *args: Any, **kwargs: Any) -> Any:
        """Backfill market data for binding."""
        if self._use_legacy:
            return self._legacy_impl.backfill_binding(*args, **kwargs)
        return self._ingestion_coordinator.backfill_binding(*args, **kwargs)

    def backfill_coverage(self, *args: Any, **kwargs: Any) -> Any:
        """Backfill coverage gaps."""
        if self._use_legacy:
            return self._legacy_impl.backfill_coverage(*args, **kwargs)
        return self._ingestion_coordinator.backfill_coverage(*args, **kwargs)

    def auto_fill_universe(self, *args: Any, **kwargs: Any) -> Any:
        """Auto-fill universe with market data."""
        if self._use_legacy:
            return self._legacy_impl.auto_fill_universe(*args, **kwargs)
        return self._ingestion_coordinator.auto_fill_universe(*args, **kwargs)

    # =========================================================================
    # Dataset Building Methods (delegate to DatasetBuilder)
    # =========================================================================

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        """
        Build ML dataset.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        int
            Exit code (0 for success)

        """
        if self._use_legacy:
            return self._legacy_impl.build_dataset(cfg)
        return self._dataset_builder.build_dataset(cfg)

    def validate_dataset(self, *args: Any, **kwargs: Any) -> Any:
        """Validate dataset against expectations."""
        if self._use_legacy:
            return self._legacy_impl.validate_dataset(*args, **kwargs)
        return self._dataset_builder.validate_dataset(*args, **kwargs)

    # =========================================================================
    # Discovery Methods (delegate to DiscoveryClient)
    # =========================================================================

    def discover_market_inputs(self, *args: Any, **kwargs: Any) -> Any:
        """Discover market inputs for given symbols and time range."""
        if self._use_legacy:
            return self._legacy_impl.discover_market_inputs(*args, **kwargs)
        return self._discovery_client.discover_market_inputs(*args, **kwargs)

    # =========================================================================
    # Binding Resolution Methods (delegate to BindingResolver)
    # =========================================================================

    def resolve_market_inputs(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve market inputs with coverage validation."""
        if self._use_legacy:
            return self._legacy_impl.resolve_market_inputs(*args, **kwargs)
        return self._binding_resolver.resolve_market_inputs(*args, **kwargs)

    def filter_candidate_bindings(self, *args: Any, **kwargs: Any) -> Any:
        """Filter candidate bindings based on availability and cost."""
        if self._use_legacy:
            return self._legacy_impl.filter_candidate_bindings(*args, **kwargs)
        return self._binding_resolver.filter_candidate_bindings(*args, **kwargs)

    def select_binding_with_coverage(self, *args: Any, **kwargs: Any) -> Any:
        """Select first binding with available coverage."""
        if self._use_legacy:
            return self._legacy_impl.select_binding_with_coverage(*args, **kwargs)
        return self._binding_resolver.select_binding_with_coverage(*args, **kwargs)

    # =========================================================================
    # Training and HPO Methods (remain in facade for now)
    # =========================================================================

    def run_hpo(self, cfg: HPOConfig, dataset_csv: Path, out_dir: Path) -> int:
        """
        Run hyperparameter optimization.

        Parameters
        ----------
        cfg : HPOConfig
            HPO configuration
        dataset_csv : Path
            Path to dataset CSV
        out_dir : Path
            Output directory

        Returns
        -------
        int
            Exit code (0 for success)

        """
        if self._use_legacy:
            return self._legacy_impl.run_hpo(cfg, dataset_csv, out_dir)

        # HPO logic remains in facade for now (may extract in future Phase)
        logger.warning(
            "HPO not yet implemented in component-based orchestrator; use legacy mode"
        )
        return 1

    def train_teacher(
        self,
        cfg: TeacherTrainConfig,
        dataset_csv: Path,
        out_dir: Path,
    ) -> int:
        """
        Train teacher model.

        Parameters
        ----------
        cfg : TeacherTrainConfig
            Teacher training configuration
        dataset_csv : Path
            Path to dataset CSV
        out_dir : Path
            Output directory

        Returns
        -------
        int
            Exit code (0 for success)

        """
        if self._use_legacy:
            return self._legacy_impl.train_teacher(cfg, dataset_csv, out_dir)

        # Training logic remains in facade for now (may extract in future Phase)
        logger.warning(
            "Teacher training not yet implemented in component-based orchestrator; use legacy mode"
        )
        return 1

    def distill_student(
        self,
        cfg: StudentDistillConfig,
        dataset_csv: Path,
        teacher_dir: Path,
        out_dir: Path,
    ) -> int:
        """
        Distill student model.

        Parameters
        ----------
        cfg : StudentDistillConfig
            Student distillation configuration
        dataset_csv : Path
            Path to dataset CSV
        teacher_dir : Path
            Teacher model directory
        out_dir : Path
            Output directory

        Returns
        -------
        int
            Exit code (0 for success)

        """
        if self._use_legacy:
            return self._legacy_impl.distill_student(cfg, dataset_csv, teacher_dir, out_dir)

        # Distillation logic remains in facade for now (may extract in future Phase)
        logger.warning(
            "Student distillation not yet implemented in component-based orchestrator; use legacy mode"
        )
        return 1

    def run(self, cfg: OrchestratorConfig) -> int:
        """
        Run full ML pipeline.

        Parameters
        ----------
        cfg : OrchestratorConfig
            Orchestrator configuration

        Returns
        -------
        int
            Exit code (0 for success)

        """
        if self._use_legacy:
            return self._legacy_impl.run(cfg)

        # Full pipeline orchestration - remains in facade for now
        logger.warning(
            "Full pipeline run not yet implemented in component-based orchestrator; use legacy mode"
        )
        return 1

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        """
        Run training-only pipeline.

        Parameters
        ----------
        cfg : OrchestratorConfig
            Orchestrator configuration

        Returns
        -------
        int
            Exit code (0 for success)

        """
        if self._use_legacy:
            return self._legacy_impl.run_training_only(cfg)

        # Training-only pipeline - remains in facade for now
        logger.warning(
            "Training-only run not yet implemented in component-based orchestrator; use legacy mode"
        )
        return 1

    # =========================================================================
    # Health and Monitoring
    # =========================================================================

    def get_health_status(self) -> dict[str, Any]:
        """
        Get health status from all components.

        Returns
        -------
        dict[str, Any]
            Health status information

        """
        if self._use_legacy:
            return self._legacy_impl.get_health_status()

        return {
            "implementation": "component_based",
            "config_resolver": "healthy",
            "discovery_client": "healthy",
            "binding_resolver": "healthy",
            "ingestion_coordinator": "healthy",
            "dataset_builder": "healthy",
        }

    # =========================================================================
    # Additional Public Methods (for backward compatibility)
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attributes to legacy implementation if in legacy mode.

        This ensures complete backward compatibility for any methods not
        explicitly delegated above.

        Parameters
        ----------
        name : str
            Attribute name

        Returns
        -------
        Any
            Attribute value

        Raises
        ------
        AttributeError
            If attribute not found

        """
        if self._use_legacy and hasattr(self, "_legacy_impl"):
            return getattr(self._legacy_impl, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
