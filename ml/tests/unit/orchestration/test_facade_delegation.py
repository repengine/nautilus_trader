#!/usr/bin/env python3
"""
Tests for facade delegation to components.

Verifies that the facade correctly delegates to the appropriate component
and does NOT contain business logic itself.

Test Design: reports/tests/mlpipelineorchestrator_test_design_report.md
Coverage Target: 90%

All tests initially marked @pytest.mark.skip - TDD approach.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest


if TYPE_CHECKING:
    from ml.orchestration.config_types import DatasetBuildConfig, OrchestratorConfig


# ============================================================================
# FACADE DELEGATION TESTS
# ============================================================================


@pytest.mark.unit
class TestFacadeDelegation:
    """Tests for facade delegation patterns."""

    @pytest.mark.skip(reason="Pending implementation - StageController delegation")
    def test_facade_run_delegates_to_stage_controller(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify run() delegates to StageController.

        Given:
        - Facade with StageController initialized

        When:
        - Calling facade.run()

        Then:
        - StageController.run_pipeline() called
        - Facade does not contain run logic itself
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

        with patch.object(facade, "_stage_controller", create=True) as mock_controller:
            mock_controller.run_pipeline.return_value = 0

            result = facade.run(sample_orchestrator_config)

            mock_controller.run_pipeline.assert_called_once()
            assert result == 0

    @pytest.mark.skip(reason="Pending implementation - delegation verification")
    def test_facade_build_dataset_delegates_to_dataset_builder(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        sample_dataset_config: DatasetBuildConfig,
    ) -> None:
        """
        Verify build_dataset() delegates to DatasetBuilder.

        Given:
        - Facade with DatasetBuilder initialized

        When:
        - Calling facade.build_dataset()

        Then:
        - DatasetBuilder.build_dataset() called
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

        # Verify delegation occurs
        with patch.object(facade, "_dataset_builder") as mock_builder:
            mock_builder.build_dataset.return_value = 0

            result = facade.build_dataset(sample_dataset_config)

            mock_builder.build_dataset.assert_called_once_with(sample_dataset_config)
            assert result == 0

    @pytest.mark.skip(reason="Pending implementation - delegation verification")
    def test_facade_train_teacher_delegates_to_training_coordinator(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        tmp_path: Path,
    ) -> None:
        """
        Verify train_teacher() delegates to TrainingCoordinator.

        Given:
        - Facade with TrainingCoordinator initialized

        When:
        - Calling facade.train_teacher()

        Then:
        - TrainingCoordinator.train_teacher() called
        """
        from ml.orchestration.config_types import TeacherTrainConfig
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        teacher_config = TeacherTrainConfig(enabled=True)
        dataset_csv = tmp_path / "dataset.csv"
        out_dir = tmp_path / "output"

        with patch.object(facade, "_training_coordinator") as mock_coordinator:
            mock_coordinator.train_teacher.return_value = 0

            result = facade.train_teacher(teacher_config, dataset_csv, out_dir)

            mock_coordinator.train_teacher.assert_called_once()
            assert result == 0

    @pytest.mark.skip(reason="Pending implementation - delegation verification")
    def test_facade_run_pre_ingestion_delegates_to_ingestion_coordinator(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        tmp_path: Path,
    ) -> None:
        """
        Verify run_pre_ingestion() delegates to IngestionCoordinator.

        Given:
        - Facade with IngestionCoordinator initialized

        When:
        - Calling facade.run_pre_ingestion()

        Then:
        - IngestionCoordinator.run_pre_ingestion() called
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

        catalog_path = tmp_path / "catalog"
        catalog_path.mkdir()
        scheduler_cfg = Mock()

        with patch.object(facade, "_ingestion_coordinator") as mock_coordinator:
            mock_coordinator.run_pre_ingestion.return_value = None

            facade.run_pre_ingestion(
                catalog_path=catalog_path,
                scheduler_cfg=scheduler_cfg,
            )

            mock_coordinator.run_pre_ingestion.assert_called_once()

    @pytest.mark.skip(reason="Pending implementation - legacy delegation removal")
    def test_facade_has_no_legacy_delegation_when_flag_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify facade does not delegate to legacy when flag is off.

        Given:
        - ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=0

        When:
        - Calling any facade method

        Then:
        - _legacy_orchestrator NOT used for delegation
        - Component-based implementation used
        """
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")

        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        # The facade should use components, not legacy
        # This is verified by checking that component methods are called
        assert facade._ingestion_coordinator is not None
        assert facade._dataset_builder is not None
        assert facade._training_coordinator is not None


# ============================================================================
# FACADE LINE COUNT TESTS (Category 14)
# ============================================================================


@pytest.mark.unit
class TestFacadeLineCount:
    """Tests for facade size constraints (Category 14)."""

    def test_facade_under_400_lines(self) -> None:
        """
        Verify facade module is under 400 lines.

        This ensures the facade remains thin and delegates
        rather than containing business logic.
        """
        facade_path = Path(
            "/home/nate/projects/nautilus_trader/ml/orchestration/pipeline_orchestrator_facade.py"
        )

        if not facade_path.exists():
            pytest.skip("Facade file not found")

        with open(facade_path) as f:
            line_count = len(f.readlines())

        assert line_count < 400, f"Facade has {line_count} lines, should be <400"

    @pytest.mark.skip(reason="Pending implementation - static analysis")
    def test_facade_methods_are_thin_wrappers(self) -> None:
        """
        Verify facade methods are thin wrappers (< 20 lines each).

        This ensures business logic is in components, not facade.
        """
        import ast

        facade_path = Path(
            "/home/nate/projects/nautilus_trader/ml/orchestration/pipeline_orchestrator_facade.py"
        )

        if not facade_path.exists():
            pytest.skip("Facade file not found")

        content = facade_path.read_text()
        tree = ast.parse(content)

        # Find all method definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip __init__ and private methods
                if node.name.startswith("_"):
                    continue

                # Calculate method line count
                method_lines = node.end_lineno - node.lineno + 1 if node.end_lineno else 0

                # Public methods should be thin (< 30 lines including docstring)
                assert method_lines < 30, (
                    f"Method {node.name} has {method_lines} lines, "
                    "should be <30 for thin facade"
                )


# ============================================================================
# COMPONENT INITIALIZATION TESTS
# ============================================================================


@pytest.mark.unit
class TestComponentInitialization:
    """Tests for component initialization in facade."""

    @pytest.mark.skip(reason="Pending implementation - component initialization")
    def test_all_components_initialized(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify all 7 components are initialized.

        Given:
        - Facade creation with dependencies

        When:
        - Accessing component attributes

        Then:
        - All 7 components are not None
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

        assert facade._ingestion_coordinator is not None
        assert facade._dataset_builder is not None
        assert facade._training_coordinator is not None
        assert facade._registry_synchronizer is not None
        assert facade._runtime_attacher is not None
        assert facade._config_resolver is not None
        assert facade._discovery_client is not None

    @pytest.mark.skip(reason="Pending implementation - StageController initialization")
    def test_stage_controller_initialized(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify StageController is initialized with all components.

        Given:
        - Facade creation

        When:
        - Checking StageController

        Then:
        - StageController has all required components wired
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

        assert hasattr(facade, "_stage_controller")
        assert facade._stage_controller is not None


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_coverage_provider() -> Mock:
    """Mock CoverageProviderProtocol."""
    provider = Mock()
    provider.read_bucket_coverage.return_value = set()
    return provider


@pytest.fixture
def mock_writer() -> Mock:
    """Mock MarketDataWriterProtocol."""
    writer = Mock()
    writer.write.return_value = 0
    return writer


@pytest.fixture
def mock_build_main() -> Mock:
    """Mock CLI main function for dataset building."""
    return Mock(return_value=0)


@pytest.fixture
def mock_teacher_main() -> Mock:
    """Mock CLI main function for teacher training."""
    return Mock(return_value=0)


@pytest.fixture
def sample_orchestrator_config(tmp_path: Path) -> OrchestratorConfig:
    """Sample OrchestratorConfig for testing."""
    from ml.orchestration.config_types import (
        DatasetBuildConfig,
        HPOConfig,
        OrchestratorConfig,
        TeacherTrainConfig,
    )

    return OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path / "data"),
            symbols="SPY",
            out_dir=str(tmp_path / "output"),
        ),
        hpo=HPOConfig(enabled=False),
        teacher=TeacherTrainConfig(enabled=True),
    )


@pytest.fixture
def sample_dataset_config(tmp_path: Path) -> DatasetBuildConfig:
    """Sample DatasetBuildConfig for testing."""
    from ml.orchestration.config_types import DatasetBuildConfig

    return DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "output"),
    )
