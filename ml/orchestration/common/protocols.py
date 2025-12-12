"""
Protocol definitions for orchestration components.

This module defines all protocol interfaces for the decomposed
MLPipelineOrchestrator components. Using protocols enables:
- Structural typing (duck typing) for testing with mocks
- Type safety without implementation coupling
- Clear interface contracts

"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from ml.config.scheduler_config import SchedulerConfig
    from ml.data import DatasetMetadata
    from ml.data.ingest.discovery import MarketDatasetInput
    from ml.data.ingest.market_bindings import ResolvedMarketBinding
    from ml.data.ingest.orchestrator import BackfillWindowList
    from ml.data.ingest.subscription import SubscriptionPolicy as CoveragePolicy
    from ml.orchestration.config_types import DatasetBuildConfig
    from ml.orchestration.config_types import HPOConfig
    from ml.orchestration.config_types import IntegrationConfig
    from ml.orchestration.config_types import OrchestratorConfig
    from ml.orchestration.config_types import PreIngestionOptions
    from ml.orchestration.config_types import StudentDistillConfig
    from ml.orchestration.config_types import TeacherTrainConfig
    from ml.orchestration.dataset_builder import BuildArtifacts


@runtime_checkable
class StageControllerProtocol(Protocol):
    """
    Protocol for pipeline stage orchestration.

    The StageController orchestrates ML pipeline stages in the correct order,
    handles checkpointing for resume capability, and manages multi-symbol
    processing with output isolation.

    """

    def run_pipeline(
        self,
        cfg: OrchestratorConfig,
        *,
        checkpoint_file: Path | None = None,
        resume: bool = False,
    ) -> int:
        """
        Execute full pipeline with checkpoint support.

        Runs all pipeline stages in order:
        PRE_INGEST -> AUTO_FILL -> DATASET -> HPO -> TRAIN -> DISTILL -> PROMOTE -> INTEGRATE

        Parameters
        ----------
        cfg : OrchestratorConfig
            Orchestrator configuration
        checkpoint_file : Path | None, optional
            Path to checkpoint file for state persistence
        resume : bool, optional
            If True and checkpoint exists, resume from checkpoint

        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)

        """
        ...

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        """
        Execute training stages only (skip ingestion/dataset).

        Runs training stages assuming dataset already exists:
        HPO -> TRAIN -> DISTILL -> PROMOTE -> INTEGRATE

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
            If dataset CSV is not found

        """
        ...


@runtime_checkable
class IngestionCoordinatorProtocol(Protocol):
    """
    Protocol for ingestion coordination operations.

    Handles data ingestion, backfill, and auto-fill operations.

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
        Backfill using resolved market binding.

        Parameters
        ----------
        binding : ResolvedMarketBinding
            Resolved market binding
        lookback_days : int
            Days to backfill

        Returns
        -------
        dict[str, BackfillWindowList]
            Map of instrument_id to backfill results

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
        Backfill gaps bounded by coverage policy.

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
            List of (start_ns, end_ns) windows

        """
        ...


@runtime_checkable
class DatasetBuilderProtocol(Protocol):
    """
    Protocol for dataset building operations.

    Handles dataset construction from market data, feature engineering,
    validation against expectations, metadata management, and storage.

    """

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
        ...


@runtime_checkable
class TrainingCoordinatorProtocol(Protocol):
    """
    Protocol for training coordination operations.

    Coordinates HPO, teacher training, and student distillation.

    """

    def run_hpo(
        self,
        cfg: HPOConfig | None,
        dataset_csv: Path,
        out_dir: Path,
    ) -> int:
        """
        Run hyperparameter optimization.

        Parameters
        ----------
        cfg : HPOConfig | None
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
        ...

    def train_teacher(
        self,
        cfg: TeacherTrainConfig | None,
        dataset_csv: Path,
        out_dir: Path,
    ) -> int:
        """
        Train teacher model.

        Parameters
        ----------
        cfg : TeacherTrainConfig | None
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
        ...

    def distill_student(
        self,
        cfg: StudentDistillConfig | None,
        dataset_dir: Path,
        teacher_cfg: TeacherTrainConfig | None,
    ) -> int:
        """
        Distill student model from teacher.

        Parameters
        ----------
        cfg : StudentDistillConfig | None
            Student distillation configuration
        dataset_dir : Path
            Dataset directory
        teacher_cfg : TeacherTrainConfig | None
            Teacher configuration

        Returns
        -------
        int
            Exit code (0 for success)

        """
        ...


@runtime_checkable
class RegistrySynchronizerProtocol(Protocol):
    """
    Protocol for registry synchronization operations.

    Handles dataset manifest synchronization and build artifact capture.

    """

    def synchronize_dataset_manifest(
        self,
        *,
        cfg: DatasetBuildConfig,
        metadata: DatasetMetadata,
    ) -> None:
        """
        Synchronize dataset manifest metadata with the data registry.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration
        metadata : DatasetMetadata
            Dataset metadata to synchronize

        """
        ...

    def capture_cli_build_artifacts(
        self,
        cfg: DatasetBuildConfig,
    ) -> BuildArtifacts | None:
        """
        Capture build artifacts from CLI output directory.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        BuildArtifacts | None
            Captured build artifacts or None if capture failed

        """
        ...


@runtime_checkable
class RuntimeAttacherProtocol(Protocol):
    """
    Protocol for runtime attachment operations.

    Handles integration manager setup and runtime configuration.

    """

    def attach_runtime(
        self,
        integration_cfg: IntegrationConfig | None,
        *,
        dataset_out_dir: Path,
    ) -> object | None:
        """
        Attach integration manager runtime.

        Parameters
        ----------
        integration_cfg : IntegrationConfig | None
            Integration configuration
        dataset_out_dir : Path
            Dataset output directory

        Returns
        -------
        object | None
            Integration manager instance or None if not attached

        """
        ...


@runtime_checkable
class ConfigResolverProtocol(Protocol):
    """
    Protocol for configuration resolution operations.

    Handles config parsing, validation, and preparation.

    """

    def resolve_instrument_ids(
        self,
        cfg: DatasetBuildConfig,
        override: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """
        Resolve instrument IDs from configuration.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration
        override : tuple[str, ...] | None
            Optional override

        Returns
        -------
        tuple[str, ...]
            Resolved instrument IDs

        """
        ...

    def infer_default_schema(self, cfg: DatasetBuildConfig) -> str:
        """
        Infer default schema from configuration.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        str
            Inferred schema

        """
        ...


@runtime_checkable
class DiscoveryClientProtocol(Protocol):
    """
    Protocol for discovery service operations.

    Handles market data discovery and binding resolution.

    """

    def discover_binding_for_symbol(
        self,
        symbol: str,
        instrument_ids: tuple[str, ...] | None,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None:
        """
        Discover market binding for a symbol.

        Parameters
        ----------
        symbol : str
            Symbol to discover
        instrument_ids : tuple[str, ...] | None
            Optional instrument IDs for the symbol
        schema : str
            Data schema
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        ResolvedMarketBinding | None
            Discovered binding or None

        """
        ...

    def discover_market_inputs(
        self,
        symbol_map: Mapping[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...]:
        """
        Discover market inputs for given symbols and time range.

        Parameters
        ----------
        symbol_map : Mapping[str, tuple[str, ...]]
            Symbol to instrument IDs mapping
        schema : str
            Data schema (e.g. 'ohlcv-1m', 'tbbo')
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        dataset_hint : str | None
            Optional dataset ID hint

        Returns
        -------
        tuple[MarketDatasetInput, ...]
            Discovered market inputs

        """
        ...
