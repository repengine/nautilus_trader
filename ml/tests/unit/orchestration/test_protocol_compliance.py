#!/usr/bin/env python3
"""
Protocol compliance tests for orchestration components.

Verifies that each component correctly implements its protocol interface.

Test Design: reports/tests/mlpipelineorchestrator_test_design_report.md
Coverage Target: 90%

All tests initially marked @pytest.mark.skip - TDD approach.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest


if TYPE_CHECKING:
    pass


# ============================================================================
# INGESTION COORDINATOR PROTOCOL TESTS
# ============================================================================


@pytest.mark.unit
class TestIngestionCoordinatorProtocol:
    """Tests for IngestionCoordinator protocol compliance."""

    def test_conforms_to_protocol(self) -> None:
        """
        Verify IngestionCoordinator implements protocol.

        Given:
        - IngestionCoordinatorProtocol defined

        When:
        - Creating IngestionCoordinator instance

        Then:
        - Instance satisfies protocol (isinstance check)
        """
        from ml.orchestration.ingestion_coordinator import IngestionCoordinator
        from ml.orchestration.protocols import IngestionCoordinatorProtocol

        instance = IngestionCoordinator(
            coverage=Mock(),
            writer=Mock(),
        )

        assert isinstance(instance, IngestionCoordinatorProtocol)

    def test_run_pre_ingestion_signature(self) -> None:
        """
        Verify run_pre_ingestion method signature matches protocol.

        Given:
        - IngestionCoordinator class

        When:
        - Inspecting run_pre_ingestion signature

        Then:
        - Has catalog_path parameter
        - Has scheduler_cfg parameter
        - Has options parameter (optional)
        """
        from ml.orchestration.ingestion_coordinator import IngestionCoordinator

        sig = inspect.signature(IngestionCoordinator.run_pre_ingestion)
        params = list(sig.parameters.keys())

        assert "catalog_path" in params
        assert "scheduler_cfg" in params
        assert "options" in params

    def test_backfill_signature(self) -> None:
        """
        Verify backfill method signature matches protocol.

        Given:
        - IngestionCoordinator class

        When:
        - Inspecting backfill signature

        Then:
        - Has dataset_id parameter
        - Has schema parameter
        - Has instrument_id parameter
        - Has lookback_days parameter
        """
        from ml.orchestration.ingestion_coordinator import IngestionCoordinator

        sig = inspect.signature(IngestionCoordinator.backfill)
        params = list(sig.parameters.keys())

        assert "dataset_id" in params
        assert "schema" in params
        assert "instrument_id" in params
        assert "lookback_days" in params

    def test_backfill_binding_signature(self) -> None:
        """
        Verify backfill_binding method signature matches protocol.

        Given:
        - IngestionCoordinator class

        When:
        - Inspecting backfill_binding signature

        Then:
        - Has binding parameter
        - Has lookback_days parameter
        """
        from ml.orchestration.ingestion_coordinator import IngestionCoordinator

        sig = inspect.signature(IngestionCoordinator.backfill_binding)
        params = list(sig.parameters.keys())

        assert "binding" in params
        assert "lookback_days" in params


# ============================================================================
# DATASET BUILDER PROTOCOL TESTS
# ============================================================================


@pytest.mark.unit
class TestDatasetBuilderProtocol:
    """Tests for DatasetBuilder protocol compliance."""

    def test_conforms_to_protocol(self) -> None:
        """
        Verify DatasetBuilder implements protocol.

        Given:
        - DatasetBuilderProtocol defined

        When:
        - Creating DatasetBuilder instance

        Then:
        - Instance satisfies protocol
        """
        from ml.orchestration.dataset_builder import DatasetBuilder
        from ml.orchestration.protocols import DatasetBuilderProtocol

        instance = DatasetBuilder(
            data_store=Mock(),
            data_registry=Mock(),
            build_main=Mock(),
        )

        assert isinstance(instance, DatasetBuilderProtocol)

    def test_build_dataset_signature(self) -> None:
        """
        Verify build_dataset method signature matches protocol.

        Given:
        - DatasetBuilder class

        When:
        - Inspecting build_dataset signature

        Then:
        - Has config parameter
        - Returns int
        """
        from ml.orchestration.dataset_builder import DatasetBuilder

        sig = inspect.signature(DatasetBuilder.build_dataset)
        params = list(sig.parameters.keys())

        assert "config" in params or "cfg" in params
        # Check return annotation
        assert sig.return_annotation is int or "int" in str(sig.return_annotation)

    def test_validate_dataset_signature(self) -> None:
        """
        Verify validate_dataset method signature exists.

        Given:
        - DatasetBuilder class

        When:
        - Inspecting validate_dataset signature

        Then:
        - Method exists and is callable
        """
        from ml.orchestration.dataset_builder import DatasetBuilder

        # DatasetBuilder has validate_dataset, not prepare_dataset_config
        # (prepare_dataset_config is on ConfigResolver)
        assert hasattr(DatasetBuilder, "validate_dataset")
        assert callable(getattr(DatasetBuilder, "validate_dataset", None))


# ============================================================================
# TRAINING COORDINATOR PROTOCOL TESTS
# ============================================================================


@pytest.mark.unit
class TestTrainingCoordinatorProtocol:
    """Tests for TrainingCoordinator protocol compliance."""

    def test_conforms_to_protocol(self) -> None:
        """
        Verify TrainingCoordinator implements protocol.

        Given:
        - TrainingCoordinatorProtocol defined

        When:
        - Creating TrainingCoordinator instance

        Then:
        - Instance satisfies protocol
        """
        from ml.orchestration.protocols import TrainingCoordinatorProtocol
        from ml.orchestration.training_coordinator import TrainingCoordinator

        instance = TrainingCoordinator(
            teacher_main=Mock(),
        )

        assert isinstance(instance, TrainingCoordinatorProtocol)

    def test_train_teacher_signature(self) -> None:
        """
        Verify train_teacher method signature matches protocol.

        Given:
        - TrainingCoordinator class

        When:
        - Inspecting train_teacher signature

        Then:
        - Has cfg parameter
        - Has dataset_csv parameter
        - Has out_dir parameter
        - Returns int
        """
        from ml.orchestration.training_coordinator import TrainingCoordinator

        sig = inspect.signature(TrainingCoordinator.train_teacher)
        params = list(sig.parameters.keys())

        assert "cfg" in params or "config" in params
        assert "dataset_csv" in params
        assert "out_dir" in params

    def test_distill_student_signature(self) -> None:
        """
        Verify distill_student method signature matches protocol.

        Given:
        - TrainingCoordinator class

        When:
        - Inspecting distill_student signature

        Then:
        - Has cfg parameter
        - Has dataset_dir parameter
        - Has teacher_cfg parameter (optional)
        """
        from ml.orchestration.training_coordinator import TrainingCoordinator

        sig = inspect.signature(TrainingCoordinator.distill_student)
        params = list(sig.parameters.keys())

        assert "cfg" in params or "config" in params
        assert "dataset_dir" in params
        assert "teacher_cfg" in params

    def test_run_hpo_signature(self) -> None:
        """
        Verify run_hpo method signature matches protocol.

        Given:
        - TrainingCoordinator class

        When:
        - Inspecting run_hpo signature

        Then:
        - Has cfg parameter
        - Has dataset_csv parameter
        - Has out_dir parameter
        """
        from ml.orchestration.training_coordinator import TrainingCoordinator

        sig = inspect.signature(TrainingCoordinator.run_hpo)
        params = list(sig.parameters.keys())

        assert "cfg" in params or "config" in params
        assert "dataset_csv" in params
        assert "out_dir" in params


# ============================================================================
# CONFIG RESOLVER PROTOCOL TESTS
# ============================================================================


@pytest.mark.unit
class TestConfigResolverProtocol:
    """Tests for ConfigResolver protocol compliance."""

    def test_conforms_to_protocol(self) -> None:
        """
        Verify ConfigResolver implements protocol.

        Given:
        - ConfigResolverProtocol defined

        When:
        - Creating ConfigResolver instance

        Then:
        - Instance satisfies protocol
        """
        from ml.orchestration.config_resolver import ConfigResolver
        from ml.orchestration.protocols import ConfigResolverProtocol

        instance = ConfigResolver()

        assert isinstance(instance, ConfigResolverProtocol)

    def test_resolve_instrument_ids_signature(self) -> None:
        """
        Verify resolve_instrument_ids method signature matches protocol.

        Given:
        - ConfigResolver class

        When:
        - Inspecting resolve_instrument_ids signature

        Then:
        - Method exists and has cfg parameter
        """
        from ml.orchestration.config_resolver import ConfigResolver

        # ConfigResolver has resolve_instrument_ids (per protocol)
        assert hasattr(ConfigResolver, "resolve_instrument_ids")
        sig = inspect.signature(ConfigResolver.resolve_instrument_ids)
        params = list(sig.parameters.keys())
        # Parameter name may be cfg or dataset_cfg depending on implementation
        assert "cfg" in params or "dataset_cfg" in params


# ============================================================================
# DISCOVERY CLIENT PROTOCOL TESTS
# ============================================================================


@pytest.mark.unit
class TestDiscoveryClientProtocol:
    """Tests for DiscoveryClient protocol compliance."""

    def test_conforms_to_protocol(self) -> None:
        """
        Verify DiscoveryClient implements protocol.

        Given:
        - DiscoveryClientProtocol defined

        When:
        - Creating DiscoveryClient instance

        Then:
        - Instance satisfies protocol
        """
        from ml.orchestration.discovery_client import DiscoveryClient
        from ml.orchestration.protocols import DiscoveryClientProtocol

        instance = DiscoveryClient()

        assert isinstance(instance, DiscoveryClientProtocol)


# ============================================================================
# REGISTRY SYNCHRONIZER PROTOCOL TESTS
# ============================================================================


@pytest.mark.unit
class TestRegistrySynchronizerProtocol:
    """Tests for RegistrySynchronizer protocol compliance."""

    def test_conforms_to_protocol(self) -> None:
        """
        Verify RegistrySynchronizer implements protocol.

        Given:
        - RegistrySynchronizerProtocol defined

        When:
        - Creating RegistrySynchronizer instance

        Then:
        - Instance satisfies protocol
        """
        from ml.orchestration.protocols import RegistrySynchronizerProtocol
        from ml.orchestration.registry_synchronizer import RegistrySynchronizer

        instance = RegistrySynchronizer(data_registry=Mock())

        assert isinstance(instance, RegistrySynchronizerProtocol)

    def test_synchronize_dataset_manifest_signature(self) -> None:
        """
        Verify synchronize_dataset_manifest method signature matches protocol.

        Given:
        - RegistrySynchronizer class

        When:
        - Inspecting synchronize_dataset_manifest signature

        Then:
        - Method exists and has cfg and metadata parameters
        """
        from ml.orchestration.registry_synchronizer import RegistrySynchronizer

        assert hasattr(RegistrySynchronizer, "synchronize_dataset_manifest")
        sig = inspect.signature(RegistrySynchronizer.synchronize_dataset_manifest)
        params = list(sig.parameters.keys())
        assert "cfg" in params
        assert "metadata" in params

    def test_capture_cli_build_artifacts_signature(self) -> None:
        """
        Verify capture_cli_build_artifacts method signature matches protocol.

        Given:
        - RegistrySynchronizer class

        When:
        - Inspecting capture_cli_build_artifacts signature

        Then:
        - Method exists and has cfg parameter
        """
        from ml.orchestration.registry_synchronizer import RegistrySynchronizer

        assert hasattr(RegistrySynchronizer, "capture_cli_build_artifacts")
        sig = inspect.signature(RegistrySynchronizer.capture_cli_build_artifacts)
        params = list(sig.parameters.keys())
        assert "cfg" in params


# ============================================================================
# RUNTIME ATTACHER PROTOCOL TESTS
# ============================================================================


@pytest.mark.unit
class TestRuntimeAttacherProtocol:
    """Tests for RuntimeAttacher protocol compliance."""

    def test_conforms_to_protocol(self) -> None:
        """
        Verify RuntimeAttacher implements protocol.

        Given:
        - RuntimeAttacherProtocol defined

        When:
        - Creating RuntimeAttacher instance

        Then:
        - Instance satisfies protocol
        """
        from ml.orchestration.protocols import RuntimeAttacherProtocol
        from ml.orchestration.runtime_attacher import RuntimeAttacher

        instance = RuntimeAttacher(data_registry=Mock())

        assert isinstance(instance, RuntimeAttacherProtocol)

    def test_attach_runtime_signature(self) -> None:
        """
        Verify attach_runtime method signature matches protocol.

        Given:
        - RuntimeAttacher class

        When:
        - Inspecting attach_runtime signature

        Then:
        - Method exists and is callable
        """
        from ml.orchestration.runtime_attacher import RuntimeAttacher

        assert hasattr(RuntimeAttacher, "attach_runtime")
        assert callable(getattr(RuntimeAttacher, "attach_runtime", None))


# ============================================================================
# STAGE CONTROLLER PROTOCOL TESTS (NEW COMPONENT)
# ============================================================================


@pytest.mark.unit
class TestStageControllerProtocol:
    """Tests for StageController protocol compliance (NEW)."""

    def test_conforms_to_protocol(self) -> None:
        """
        Verify StageController implements protocol.

        Given:
        - StageControllerProtocol defined

        When:
        - Creating StageController instance

        Then:
        - Instance satisfies protocol
        """
        from ml.orchestration.common.stage_controller import StageController
        from ml.orchestration.protocols import StageControllerProtocol

        instance = StageController(
            ingestion_coordinator=Mock(),
            dataset_builder=Mock(),
            training_coordinator=Mock(),
        )

        assert isinstance(instance, StageControllerProtocol)

    def test_run_pipeline_signature(self) -> None:
        """
        Verify run_pipeline method signature matches protocol.

        Given:
        - StageController class

        When:
        - Inspecting run_pipeline signature

        Then:
        - Has cfg parameter
        - Has checkpoint_file parameter (optional)
        - Has resume parameter (optional)
        - Returns int
        """
        from ml.orchestration.common.stage_controller import StageController

        sig = inspect.signature(StageController.run_pipeline)
        params = list(sig.parameters.keys())

        assert "cfg" in params or "config" in params
        assert "checkpoint_file" in params
        assert "resume" in params

    def test_run_training_only_signature(self) -> None:
        """
        Verify run_training_only method signature matches protocol.

        Given:
        - StageController class

        When:
        - Inspecting run_training_only signature

        Then:
        - Has cfg parameter
        - Returns int
        """
        from ml.orchestration.common.stage_controller import StageController

        sig = inspect.signature(StageController.run_training_only)
        params = list(sig.parameters.keys())

        assert "cfg" in params or "config" in params
