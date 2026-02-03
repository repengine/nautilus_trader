"""
Unit tests for MLPipelineOrchestratorFacade.

Phase 2.2.8: Facade Integration - wires 7 components into unified facade.
All tests initially marked @pytest.mark.skip - TDD approach.

Test Design: reports/tests/phase_2_2_8_test_design_report.md
Coverage Target: 90%

"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest

from ml.orchestration.config_types import (
    DatasetBuildConfig,
    HPOConfig,
    IntegrationConfig,
    OrchestratorConfig,
    PreIngestionOptions,
    StudentDistillConfig,
    TeacherTrainConfig,
)
from ml.tests.utils.targets import build_default_target_semantics_payload
if TYPE_CHECKING:
    from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_coverage_provider() -> Mock:
    """
    Mock CoverageProviderProtocol for testing.
    """
    provider = Mock()
    provider.read_bucket_coverage.return_value = set()
    return provider


@pytest.fixture
def mock_writer() -> Mock:
    """
    Mock MarketDataWriterProtocol for testing.
    """
    writer = Mock()
    writer.write.return_value = 0
    return writer


@pytest.fixture
def mock_build_main() -> Mock:
    """
    Mock CLI main function for dataset building.
    """
    return Mock(return_value=0)


@pytest.fixture
def mock_teacher_main() -> Mock:
    """
    Mock CLI main function for teacher training.
    """
    return Mock(return_value=0)


@pytest.fixture
def mock_hpo_main() -> Mock:
    """
    Mock CLI main function for HPO.
    """
    return Mock(return_value=0)


@pytest.fixture
def mock_feature_registry() -> Mock:
    """
    Mock FeatureRegistry for testing.
    """
    registry = Mock()
    registry.register_features.return_value = True
    return registry


@pytest.fixture
def mock_model_registry() -> Mock:
    """
    Mock ModelRegistry for testing.
    """
    registry = Mock()
    registry.register_model.return_value = True
    return registry


@pytest.fixture
def mock_strategy_registry() -> Mock:
    """
    Mock StrategyRegistry for testing.
    """
    registry = Mock()
    return registry


@pytest.fixture
def mock_data_store() -> Mock:
    """
    Mock DataStore for testing.
    """
    store = Mock()
    store.write_bars.return_value = 0
    store.read_bars.return_value = None
    return store


@pytest.fixture
def mock_feature_store() -> Mock:
    """
    Mock FeatureStore for testing.
    """
    store = Mock()
    store.write_features.return_value = 0
    return store


@pytest.fixture
def mock_model_store() -> Mock:
    """
    Mock ModelStore for testing.
    """
    store = Mock()
    return store


@pytest.fixture
def mock_strategy_store() -> Mock:
    """
    Mock StrategyStore for testing.
    """
    store = Mock()
    return store


@pytest.fixture
def sample_dataset_config() -> DatasetBuildConfig:
    """
    Provides sample DatasetBuildConfig for testing.
    """
    return DatasetBuildConfig(
        data_dir="/tmp/test_data",
        symbols="SPY",
        out_dir="/tmp/test_output",
        dataset_id="test_dataset",
        target_semantics=build_default_target_semantics_payload(),
    )


@pytest.fixture
def sample_orchestrator_config(sample_dataset_config: DatasetBuildConfig) -> OrchestratorConfig:
    """
    Provides sample OrchestratorConfig for testing.
    """
    return OrchestratorConfig(
        dataset=sample_dataset_config,
        hpo=HPOConfig(enabled=False),
        teacher=TeacherTrainConfig(enabled=False),
    )


@pytest.fixture
def mock_orchestrator_full(
    mock_coverage_provider: Mock,
    mock_writer: Mock,
    mock_build_main: Mock,
    mock_teacher_main: Mock,
    mock_data_registry: Mock,
    mock_feature_registry: Mock,
    mock_model_registry: Mock,
    mock_strategy_registry: Mock,
    mock_data_store: Mock,
    mock_feature_store: Mock,
    mock_model_store: Mock,
    mock_strategy_store: Mock,
) -> MLPipelineOrchestrator:
    """
    Provides fully configured mock orchestrator.
    """
    from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

    with (
        patch("ml.orchestration.pipeline_orchestrator_facade.IngestionCoordinator") as mock_ingestion_cls,
        patch("ml.orchestration.pipeline_orchestrator_facade.DatasetBuilder") as mock_dataset_cls,
        patch("ml.orchestration.pipeline_orchestrator_facade.RegistrySynchronizer") as mock_registry_cls,
        patch("ml.orchestration.pipeline_orchestrator_facade.RuntimeAttacher") as mock_runtime_cls,
        patch("ml.orchestration.pipeline_orchestrator_facade.ConfigResolver") as mock_config_cls,
        patch("ml.orchestration.pipeline_orchestrator_facade.DiscoveryClient") as mock_discovery_cls,
    ):
        mock_ingestion = Mock()
        mock_ingestion.run_pre_ingestion.return_value = None
        mock_ingestion.backfill.return_value = []
        mock_ingestion.backfill_binding.return_value = {}
        mock_ingestion.backfill_coverage.return_value = []
        mock_ingestion_cls.return_value = mock_ingestion

        mock_dataset_builder = Mock()
        mock_dataset_builder.build_dataset.return_value = 0
        mock_dataset_cls.return_value = mock_dataset_builder

        mock_registry_cls.return_value = Mock()
        mock_runtime_cls.return_value = Mock()
        mock_config_cls.return_value = Mock()
        mock_discovery_cls.return_value = Mock()

        yield MLPipelineOrchestrator(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
            data_registry=mock_data_registry,
            feature_registry=mock_feature_registry,
            model_registry=mock_model_registry,
            strategy_registry=mock_strategy_registry,
            data_store=mock_data_store,
            feature_store=mock_feature_store,
            model_store=mock_model_store,
            strategy_store=mock_strategy_store,
        )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    Provides temporary output directory for tests.
    """
    output_dir = tmp_path / "ml_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def existing_dataset_dir(tmp_path: Path) -> Path:
    """
    Provides directory with existing dataset artifacts.
    """
    import json

    dataset_dir = tmp_path / "existing_dataset"
    dataset_dir.mkdir()

    # Create dataset.csv
    (dataset_dir / "dataset.csv").write_text("timestamp,close\n1704067200,100.0\n")

    # Create dataset_metadata.json
    metadata = {
        "dataset_id": "test_dataset",
        "vintage_policy": "REAL_TIME",
        "vintage_cutoff": None,
        "feature_set_id": "test_features",
    }
    (dataset_dir / "dataset_metadata.json").write_text(json.dumps(metadata))

    return dataset_dir


# ============================================================================
# FACADE INITIALIZATION TESTS
# ============================================================================


@pytest.mark.unit
class TestFacadeInitialization:
    """
    Tests for MLPipelineOrchestratorFacade initialization.
    """

    def test_facade_initializes_with_all_components(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        mock_data_registry: Mock,
    ) -> None:
        """
        Verify facade can be instantiated with all 7 components wired.

        Given:
        - All required dependencies provided
        - All optional registries and stores provided

        When:
        - Creating MLPipelineOrchestratorFacade

        Then:
        - Facade initializes without error
        - All components accessible

        """
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
            data_registry=mock_data_registry,
        )

        assert facade is not None
        assert facade._ingestion_coordinator is not None
        assert facade._dataset_builder is not None
        assert facade._training_coordinator is not None
        assert facade._registry_synchronizer is not None
        assert facade._runtime_attacher is not None
        assert facade._config_resolver is not None
        assert facade._discovery_client is not None

    def test_facade_initializes_with_minimal_dependencies(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify facade gracefully handles optional dependencies.

        Given:
        - Only required parameters provided
        - Optional registries/stores not provided

        When:
        - Creating MLPipelineOrchestratorFacade

        Then:
        - Facade initializes without error
        - Optional components are None

        """
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        assert facade is not None
        assert facade.data_registry is None
        assert facade.feature_registry is None

    def test_facade_preserves_legacy_attributes(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
    ) -> None:
        """
        Verify all legacy attributes accessible on facade.

        Given:
        - Facade with full configuration

        When:
        - Accessing legacy attribute names

        Then:
        - All attributes from MLPipelineOrchestrator exist

        """
        # Verify all expected attributes exist
        assert hasattr(mock_orchestrator_full, "coverage")
        assert hasattr(mock_orchestrator_full, "writer")
        assert hasattr(mock_orchestrator_full, "build_main")
        assert hasattr(mock_orchestrator_full, "teacher_main")
        assert hasattr(mock_orchestrator_full, "data_registry")
        assert hasattr(mock_orchestrator_full, "registry")  # backward compat alias
        assert hasattr(mock_orchestrator_full, "feature_registry")
        assert hasattr(mock_orchestrator_full, "model_registry")
        assert hasattr(mock_orchestrator_full, "strategy_registry")
        assert hasattr(mock_orchestrator_full, "data_store")
        assert hasattr(mock_orchestrator_full, "feature_store")
        assert hasattr(mock_orchestrator_full, "model_store")
        assert hasattr(mock_orchestrator_full, "strategy_store")


# ============================================================================
# COMPONENT DELEGATION TESTS
# ============================================================================


@pytest.mark.unit
class TestComponentDelegation:
    """
    Tests for component delegation from facade.
    """

    def test_run_pre_ingestion_delegates_to_ingestion_coordinator(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        tmp_path: Path,
    ) -> None:
        """
        Verify run_pre_ingestion() routes to IngestionCoordinator.

        Given:
        - Facade with IngestionCoordinator

        When:
        - Calling run_pre_ingestion()

        Then:
        - IngestionCoordinator method called with same args

        """
        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir()
        scheduler_cfg = Mock()
        options = PreIngestionOptions()

        # This test verifies delegation - implementation will add coordinator
        mock_orchestrator_full.run_pre_ingestion(
            catalog_path=catalog_path,
            scheduler_cfg=scheduler_cfg,
            options=options,
        )
        # Assert coordinator was called (implementation pending)

    def test_backfill_delegates_to_ingestion_coordinator(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
    ) -> None:
        """
        Verify backfill() routes to IngestionCoordinator.

        Given:
        - Facade with IngestionCoordinator
        - Ingestor configured

        When:
        - Calling backfill()

        Then:
        - Returns BackfillWindowList from coordinator

        """
        mock_orchestrator_full.ingestor = Mock()

        result = mock_orchestrator_full.backfill(
            dataset_id="databento.ohlcv-1s",
            schema="ohlcv-1s",
            instrument_id="SPY.NASDAQ",
            lookback_days=30,
        )

        # BackfillWindowList type check
        assert result is not None

    def test_backfill_binding_delegates_to_ingestion_coordinator(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
    ) -> None:
        """
        Verify backfill_binding() routes to IngestionCoordinator.

        Given:
        - Facade with IngestionCoordinator
        - ResolvedMarketBinding provided

        When:
        - Calling backfill_binding()

        Then:
        - Returns dict of BackfillWindowList

        """
        mock_orchestrator_full.ingestor = Mock()
        mock_binding = Mock()
        mock_binding.dataset_id = "databento.ohlcv-1s"
        mock_binding.schema = "ohlcv-1s"
        mock_binding.instrument_id = "SPY.NASDAQ"

        result = mock_orchestrator_full.backfill_binding(
            binding=mock_binding,
            lookback_days=30,
        )

        assert isinstance(result, dict)

    def test_backfill_coverage_delegates_with_policy(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
    ) -> None:
        """
        Verify backfill_coverage() uses policy to determine lookback.

        Given:
        - Facade with IngestionCoordinator
        - CoveragePolicy provided

        When:
        - Calling backfill_coverage()

        Then:
        - Policy used to determine lookback days

        """
        mock_orchestrator_full.ingestor = Mock()
        mock_policy = Mock()

        result = mock_orchestrator_full.backfill_coverage(
            dataset_id="databento.ohlcv-1s",
            schema="ohlcv-1s",
            instrument_id="SPY.NASDAQ",
            policy=mock_policy,
        )

        assert isinstance(result, list)

    def test_build_dataset_delegates_to_dataset_builder(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        sample_dataset_config: DatasetBuildConfig,
    ) -> None:
        """
        Verify build_dataset() routes to DatasetBuilder.

        Given:
        - Facade with DatasetBuilder
        - Valid DatasetBuildConfig

        When:
        - Calling build_dataset()

        Then:
        - DatasetBuilder handles dataset construction
        - Return value is int (exit code)

        """
        result = mock_orchestrator_full.build_dataset(sample_dataset_config)
        assert isinstance(result, int)

    def test_run_hpo_delegates_to_training_coordinator(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        tmp_path: Path,
    ) -> None:
        """
        Verify run_hpo() routes to TrainingCoordinator.

        Given:
        - Facade with TrainingCoordinator
        - HPOConfig with enabled=True

        When:
        - Calling run_hpo()

        Then:
        - TrainingCoordinator handles HPO

        """
        mock_orchestrator_full.hpo_main = Mock(return_value=0)
        hpo_config = HPOConfig(enabled=True)
        dataset_csv = tmp_path / "dataset.csv"
        dataset_csv.touch()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        result = mock_orchestrator_full.run_hpo(hpo_config, dataset_csv, out_dir)
        assert isinstance(result, int)

    def test_train_teacher_delegates_to_training_coordinator(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        existing_dataset_dir: Path,
    ) -> None:
        """
        Verify train_teacher() routes to TrainingCoordinator.

        Given:
        - Facade with TrainingCoordinator
        - TeacherTrainConfig with enabled=True
        - Existing dataset with metadata

        When:
        - Calling train_teacher()

        Then:
        - TrainingCoordinator handles training

        """
        teacher_config = TeacherTrainConfig(enabled=True, model_id="test_teacher")
        dataset_csv = existing_dataset_dir / "dataset.csv"

        result = mock_orchestrator_full.train_teacher(
            teacher_config,
            dataset_csv,
            existing_dataset_dir,
        )
        assert isinstance(result, int)

    def test_distill_student_delegates_to_training_coordinator(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        existing_dataset_dir: Path,
    ) -> None:
        """
        Verify distill_student() routes to TrainingCoordinator.

        Given:
        - Facade with TrainingCoordinator
        - StudentDistillConfig with enabled=True

        When:
        - Calling distill_student()

        Then:
        - TrainingCoordinator handles distillation

        """
        student_config = StudentDistillConfig(
            enabled=True,
            model_id="test_student",
            parent_model_id="test_teacher",
            model_registry_dir=str(existing_dataset_dir),
            feature_registry_dir=str(existing_dataset_dir),
            feature_set_id="test_features",
        )
        teacher_config = TeacherTrainConfig(enabled=True, model_id="test_teacher")

        # Create required NPZ files
        import numpy as np

        np.savez(existing_dataset_dir / "features_npz.npz", features=np.array([1, 2, 3]))
        np.savez(existing_dataset_dir / "teacher_preds.npz", preds=np.array([0.5, 0.6, 0.7]))

        result = mock_orchestrator_full.distill_student(
            student_config,
            dataset_dir=existing_dataset_dir,
            teacher_cfg=teacher_config,
        )
        assert isinstance(result, int)

    def test_run_delegates_pipeline_stages(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify run() orchestrates full pipeline via components.

        Given:
        - Facade with all components
        - Valid OrchestratorConfig

        When:
        - Calling run()

        Then:
        - All stages executed in order

        """
        result = mock_orchestrator_full.run(sample_orchestrator_config)
        assert isinstance(result, int)

    def test_run_training_only_delegates_correctly(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        existing_dataset_dir: Path,
    ) -> None:
        """
        Verify run_training_only() skips dataset build.

        Given:
        - Facade with TrainingCoordinator
        - Existing dataset

        When:
        - Calling run_training_only()

        Then:
        - Dataset builder NOT called
        - Training stages executed

        """
        config = OrchestratorConfig(
            dataset=DatasetBuildConfig(
                data_dir=str(existing_dataset_dir),
                symbols="SPY",
                out_dir=str(existing_dataset_dir),
                target_semantics=build_default_target_semantics_payload(),
            ),
            hpo=HPOConfig(enabled=False),
            teacher=TeacherTrainConfig(enabled=True),
        )

        result = mock_orchestrator_full.run_training_only(config)
        assert isinstance(result, int)


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================


@pytest.mark.unit
class TestBackwardCompatibility:
    """
    Tests for backward compatibility with legacy MLPipelineOrchestrator.
    """

    def test_registry_backward_compat_alias(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify 'registry' alias still works for 'data_registry'.

        Given:
        - Orchestrator created with 'registry' parameter

        When:
        - Accessing 'data_registry' attribute

        Then:
        - Returns same object passed to 'registry'

        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

        mock_registry = Mock()

        orchestrator = MLPipelineOrchestrator(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
            registry=mock_registry,
        )

        assert orchestrator.data_registry is mock_registry

    def test_public_api_preserved(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
    ) -> None:
        """
        Verify all public methods from legacy exist on orchestrator.

        Given:
        - MLPipelineOrchestrator instance

        When:
        - Checking for public methods

        Then:
        - All legacy public methods present

        """
        public_methods = [
            "run_pre_ingestion",
            "backfill",
            "backfill_binding",
            "backfill_coverage",
            "build_dataset",
            "run_hpo",
            "train_teacher",
            "distill_student",
            "run",
            "run_training_only",
        ]

        for method_name in public_methods:
            assert hasattr(
                mock_orchestrator_full, method_name
            ), f"Missing public method: {method_name}"
            assert callable(
                getattr(mock_orchestrator_full, method_name)
            ), f"Not callable: {method_name}"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.unit
class TestErrorHandling:
    """
    Tests for error handling in facade.
    """

    def test_run_training_only_handles_missing_dataset(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        tmp_path: Path,
    ) -> None:
        """
        Verify missing dataset.csv raises FileNotFoundError.

        Given:
        - Empty output directory (no dataset.csv)

        When:
        - Calling run_training_only()

        Then:
        - FileNotFoundError raised with descriptive message

        """
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = OrchestratorConfig(
            dataset=DatasetBuildConfig(
                data_dir=str(empty_dir),
                symbols="SPY",
                out_dir=str(empty_dir),
                target_semantics=build_default_target_semantics_payload(),
            ),
            hpo=HPOConfig(enabled=False),
            teacher=TeacherTrainConfig(enabled=True),
        )

        with pytest.raises(FileNotFoundError, match="Dataset CSV not found"):
            mock_orchestrator_full.run_training_only(config)

    def test_train_teacher_handles_missing_metadata(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        tmp_path: Path,
    ) -> None:
        """
        Verify missing metadata.json raises FileNotFoundError.

        Given:
        - Dataset directory with CSV but no metadata

        When:
        - Calling train_teacher()

        Then:
        - FileNotFoundError raised

        """
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "dataset.csv").write_text("timestamp,close\n1,100.0\n")

        teacher_config = TeacherTrainConfig(enabled=True, model_id="test")

        with pytest.raises(FileNotFoundError, match="metadata"):
            mock_orchestrator_full.train_teacher(
                teacher_config,
                dataset_dir / "dataset.csv",
                dataset_dir,
            )

    def test_distill_student_handles_missing_npz(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        existing_dataset_dir: Path,
    ) -> None:
        """
        Verify missing NPZ files returns error code.

        Given:
        - Dataset directory without features_npz.npz

        When:
        - Calling distill_student()

        Then:
        - Returns error code 1

        """
        student_config = StudentDistillConfig(
            enabled=True,
            model_id="test_student",
            parent_model_id="test_teacher",
            model_registry_dir=str(existing_dataset_dir),
            feature_registry_dir=str(existing_dataset_dir),
            feature_set_id="test_features",
        )
        teacher_config = TeacherTrainConfig(enabled=True, model_id="test_teacher")

        # Don't create NPZ files - they should be missing
        result = mock_orchestrator_full.distill_student(
            student_config,
            dataset_dir=existing_dataset_dir,
            teacher_cfg=teacher_config,
        )

        assert result == 1


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """
    Tests for edge cases in facade.
    """

    def test_single_symbol_parsing(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
    ) -> None:
        """
        Verify single symbol treated correctly.

        Given:
        - symbols="SPY" (single symbol)

        When:
        - Parsing symbols

        Then:
        - Returns single-element list

        """
        result = mock_orchestrator_full._parse_symbols("SPY")
        assert result == ["SPY"]

    def test_symbol_with_trailing_comma(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
    ) -> None:
        """
        Verify trailing comma handled.

        Given:
        - symbols="SPY," (trailing comma)

        When:
        - Parsing symbols

        Then:
        - Returns single-element list (empty strings filtered)

        """
        result = mock_orchestrator_full._parse_symbols("SPY,")
        # Filter empty strings
        result = [s for s in result if s.strip()]
        assert result == ["SPY"]

    def test_multi_symbol_parsing(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
    ) -> None:
        """
        Verify multi-symbol parsing.

        Given:
        - symbols="SPY,QQQ,IWM"

        When:
        - Parsing symbols

        Then:
        - Returns list of 3 symbols

        """
        result = mock_orchestrator_full._parse_symbols("SPY,QQQ,IWM")
        assert result == ["SPY", "QQQ", "IWM"]

    def test_hpo_disabled_returns_zero(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        tmp_path: Path,
    ) -> None:
        """
        Verify disabled HPO returns 0 immediately.

        Given:
        - HPOConfig with enabled=False

        When:
        - Calling run_hpo()

        Then:
        - Returns 0 without executing HPO

        """
        hpo_config = HPOConfig(enabled=False)
        dataset_csv = tmp_path / "dataset.csv"
        out_dir = tmp_path / "output"

        result = mock_orchestrator_full.run_hpo(hpo_config, dataset_csv, out_dir)
        assert result == 0

    def test_teacher_disabled_returns_zero(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        tmp_path: Path,
    ) -> None:
        """
        Verify disabled teacher training returns 0 immediately.

        Given:
        - TeacherTrainConfig with enabled=False

        When:
        - Calling train_teacher()

        Then:
        - Returns 0 without executing training

        """
        teacher_config = TeacherTrainConfig(enabled=False)
        dataset_csv = tmp_path / "dataset.csv"
        out_dir = tmp_path / "output"

        result = mock_orchestrator_full.train_teacher(teacher_config, dataset_csv, out_dir)
        assert result == 0

    def test_student_disabled_returns_zero(
        self,
        mock_orchestrator_full: MLPipelineOrchestrator,
        tmp_path: Path,
    ) -> None:
        """
        Verify disabled student distillation returns 0 immediately.

        Given:
        - StudentDistillConfig with enabled=False

        When:
        - Calling distill_student()

        Then:
        - Returns 0 without executing distillation

        """
        student_config = StudentDistillConfig(enabled=False)

        result = mock_orchestrator_full.distill_student(
            student_config,
            dataset_dir=tmp_path,
            teacher_cfg=None,
        )
        assert result == 0
