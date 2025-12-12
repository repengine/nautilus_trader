#!/usr/bin/env python3
"""
Parity tests ensuring legacy and facade produce identical results.

These tests verify that ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1 and =0
produce identical outputs for all public methods.

Test Design: reports/tests/mlpipelineorchestrator_test_design_report.md
Coverage Target: 90%

All tests initially marked @pytest.mark.skip - TDD approach.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest


if TYPE_CHECKING:
    from ml.orchestration.config_types import (
        DatasetBuildConfig,
        HPOConfig,
        OrchestratorConfig,
        StudentDistillConfig,
        TeacherTrainConfig,
    )


# ============================================================================
# METHOD PARITY TESTS
# ============================================================================


@pytest.mark.unit
class TestMethodParity:
    """Tests for method parity between legacy and facade implementations."""

    def test_build_dataset_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        sample_dataset_config: DatasetBuildConfig,
    ) -> None:
        """
        Verify build_dataset returns identical results in both modes.

        Given:
        - Identical DatasetBuildConfig inputs

        When:
        - Calling with ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1
        - Calling with ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=0

        Then:
        - Return values match
        - build_main called with same arguments
        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")

        legacy = MLPipelineOrchestrator(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        legacy_result = legacy.build_dataset(sample_dataset_config)
        legacy_call_args = mock_build_main.call_args

        mock_build_main.reset_mock()

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        facade_result = facade.build_dataset(sample_dataset_config)
        facade_call_args = mock_build_main.call_args

        assert legacy_result == facade_result
        assert legacy_call_args == facade_call_args

    def test_run_hpo_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        mock_hpo_main: Mock,
        tmp_path: Path,
    ) -> None:
        """
        Verify run_hpo returns identical results in both modes.

        Given:
        - Identical HPOConfig inputs

        When:
        - Calling with ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1
        - Calling with ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=0

        Then:
        - Return values match (both 0 for disabled)
        """
        from ml.orchestration.config_types import HPOConfig
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        hpo_config = HPOConfig(enabled=False)
        dataset_csv = tmp_path / "dataset.csv"
        out_dir = tmp_path / "output"

        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")

        legacy = MLPipelineOrchestrator(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
            hpo_main=mock_hpo_main,
        )
        legacy_result = legacy.run_hpo(hpo_config, dataset_csv, out_dir)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
            hpo_main=mock_hpo_main,
        )
        facade_result = facade.run_hpo(hpo_config, dataset_csv, out_dir)

        assert legacy_result == facade_result

    def test_train_teacher_disabled_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        tmp_path: Path,
    ) -> None:
        """
        Verify train_teacher with enabled=False returns identical results.

        Given:
        - TeacherTrainConfig with enabled=False

        When:
        - Calling in both modes

        Then:
        - Both return 0
        - teacher_main NOT called
        """
        from ml.orchestration.config_types import TeacherTrainConfig
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        teacher_config = TeacherTrainConfig(enabled=False)
        dataset_csv = tmp_path / "dataset.csv"
        out_dir = tmp_path / "output"

        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")

        legacy = MLPipelineOrchestrator(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        legacy_result = legacy.train_teacher(teacher_config, dataset_csv, out_dir)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        facade_result = facade.train_teacher(teacher_config, dataset_csv, out_dir)

        assert legacy_result == facade_result == 0
        assert not mock_teacher_main.called

    def test_distill_student_disabled_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        tmp_path: Path,
    ) -> None:
        """
        Verify distill_student with enabled=False returns identical results.

        Given:
        - StudentDistillConfig with enabled=False

        When:
        - Calling in both modes

        Then:
        - Both return 0
        """
        from ml.orchestration.config_types import StudentDistillConfig
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        student_config = StudentDistillConfig(enabled=False)
        dataset_dir = tmp_path / "dataset"

        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")

        legacy = MLPipelineOrchestrator(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        legacy_result = legacy.distill_student(student_config, dataset_dir, None)

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        facade_result = facade.distill_student(student_config, dataset_dir, None)

        assert legacy_result == facade_result == 0

    def test_get_health_status_structure_parity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify get_health_status returns consistent structure in both modes.

        Given:
        - Orchestrator instance

        When:
        - Calling get_health_status() in both modes

        Then:
        - Both return dict with 'implementation' key
        - Both have same structure (keys match)
        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")

        legacy = MLPipelineOrchestrator(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        legacy_health = legacy.get_health_status()

        # Facade mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        facade_health = facade.get_health_status()

        # Both should have implementation key
        assert "implementation" in legacy_health
        assert "implementation" in facade_health

        # Keys should match (structure parity)
        assert set(legacy_health.keys()) == set(facade_health.keys())


# ============================================================================
# FEATURE FLAG BEHAVIOR TESTS
# ============================================================================


@pytest.mark.unit
class TestFeatureFlagBehavior:
    """Tests for feature flag switching behavior."""

    def test_feature_flag_default_uses_component_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify default (no flag set) uses component mode.

        Given:
        - ML_USE_LEGACY_PIPELINE_ORCHESTRATOR not set

        When:
        - Creating facade

        Then:
        - Component-based implementation used
        """
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        monkeypatch.delenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", raising=False)

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        health = facade.get_health_status()
        # Should indicate component-based implementation
        assert health.get("implementation") == "component_based"

    def test_feature_flag_1_uses_legacy_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify flag=1 uses legacy mode.

        Given:
        - ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1

        When:
        - Creating facade

        Then:
        - Legacy implementation used
        """
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        health = facade.get_health_status()
        # Should indicate legacy implementation
        assert health.get("implementation") == "legacy"

    def test_feature_flag_switching_preserves_behavior(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify runtime feature flag changes don't break behavior.

        Given:
        - Orchestrator created with ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1

        When:
        - Changing flag to 0 mid-session

        Then:
        - Behavior remains consistent within session
        - No exceptions raised
        """
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        # Get initial health
        health1 = facade.get_health_status()

        # Switch flag mid-session
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")

        # Should still work without error
        health2 = facade.get_health_status()

        # Both calls should succeed (not necessarily same implementation)
        assert health1 is not None
        assert health2 is not None


# ============================================================================
# PUBLIC API PARITY TESTS
# ============================================================================


@pytest.mark.unit
class TestPublicAPIParity:
    """Tests for public API preservation between legacy and facade."""

    def test_all_public_methods_exist_on_facade(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
    ) -> None:
        """
        Verify all public methods from legacy exist on facade.

        Given:
        - MLPipelineOrchestrator public methods

        When:
        - Checking facade

        Then:
        - All methods present
        - All methods callable
        """
        from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        # Public methods that must exist
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
            "get_health_status",
        ]

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        for method_name in public_methods:
            assert hasattr(facade, method_name), f"Missing method: {method_name}"
            assert callable(
                getattr(facade, method_name)
            ), f"Not callable: {method_name}"

    def test_all_public_attributes_exist_on_facade(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        mock_data_registry: Mock,
    ) -> None:
        """
        Verify all public attributes from legacy exist on facade.

        Given:
        - MLPipelineOrchestrator public attributes

        When:
        - Checking facade

        Then:
        - All attributes present
        """
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        # Public attributes that must exist
        public_attrs = [
            "coverage",
            "writer",
            "build_main",
            "teacher_main",
            "registry",
            "data_registry",
            "ingestor",
            "hpo_main",
            "model_registry",
            "feature_registry",
            "strategy_registry",
            "feature_store",
            "model_store",
            "strategy_store",
            "data_store",
        ]

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
            registry=mock_data_registry,
        )

        for attr_name in public_attrs:
            assert hasattr(facade, attr_name), f"Missing attribute: {attr_name}"

    def test_registry_backward_compat_alias(
        self,
        mock_coverage_provider: Mock,
        mock_writer: Mock,
        mock_build_main: Mock,
        mock_teacher_main: Mock,
        mock_data_registry: Mock,
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
        from ml.orchestration.pipeline_orchestrator_facade import (
            MLPipelineOrchestratorFacade,
        )

        facade = MLPipelineOrchestratorFacade(
            coverage=mock_coverage_provider,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
            registry=mock_data_registry,
        )

        assert facade.data_registry is mock_data_registry


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
def mock_hpo_main() -> Mock:
    """Mock CLI main function for HPO."""
    return Mock(return_value=0)


@pytest.fixture
def sample_dataset_config(tmp_path: Path) -> DatasetBuildConfig:
    """Sample DatasetBuildConfig for testing."""
    from ml.orchestration.config_types import DatasetBuildConfig

    return DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "output"),
        dataset_id="test_dataset",
    )
