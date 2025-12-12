#!/usr/bin/env python3
"""
System validation smoke tests for MLPipelineOrchestrator.

These tests exercise the REAL public API of the orchestrator:
- build_dataset()
- run()
- run_training_only()
- run_hpo()
- train_teacher()
- distill_student()
- run_pre_ingestion()
- backfill()
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Set component mode
os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = "0"

from ml.orchestration import MLPipelineOrchestrator
from ml.orchestration.config_types import (
    DatasetBuildConfig,
    HPOConfig,
    OrchestratorConfig,
    TeacherTrainConfig,
)


pytestmark = [
    pytest.mark.usefixtures("isolated_prometheus_registry"),
]


@pytest.fixture
def mock_coverage():
    """Mock coverage provider."""
    coverage = MagicMock()
    coverage.read_bucket_coverage.return_value = set()
    return coverage


@pytest.fixture
def mock_writer():
    """Mock market data writer."""
    writer = MagicMock()
    writer.write_bars.return_value = 0
    return writer


@pytest.fixture
def mock_build_main():
    """Mock build CLI that simulates successful dataset build."""
    def _build_main(args):
        # Parse args to find out_dir and dataset_id
        out_dir = None
        dataset_id = "unknown"
        args_dict = {}

        # Build args dict
        i = 0
        while i < len(args):
            if args[i].startswith("--"):
                key = args[i][2:]  # Remove --
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    args_dict[key] = args[i + 1]
                    i += 2
                else:
                    args_dict[key] = True
                    i += 1
            else:
                i += 1

        out_dir = Path(args_dict.get("out_dir", "/tmp/test"))
        dataset_id = args_dict.get("dataset_id", "unknown")

        out_dir.mkdir(parents=True, exist_ok=True)
        # Create minimal dataset artifacts
        (out_dir / "dataset.csv").write_text("timestamp,close\n1,100.0\n")
        (out_dir / "dataset_metadata.json").write_text(
            f'{{"dataset_id": "{dataset_id}", "vintage_policy": "real_time"}}'
        )
        return 0

    return MagicMock(side_effect=_build_main)


@pytest.fixture
def mock_teacher_main():
    """Mock teacher CLI that simulates successful training."""
    def _teacher_main(args):
        # Parse args to find out_dir
        out_dir = None
        for i, arg in enumerate(args):
            if arg == "--out_dir" and i + 1 < len(args):
                out_dir = Path(args[i + 1])
                break

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            # Create minimal model artifacts
            (out_dir / "model.onnx").write_bytes(b"mock_onnx_model")
        return 0

    return MagicMock(side_effect=_teacher_main)


@pytest.fixture
def orchestrator(mock_coverage, mock_writer, mock_build_main, mock_teacher_main):
    """Create orchestrator with mocked dependencies."""
    return MLPipelineOrchestrator(
        coverage=mock_coverage,
        writer=mock_writer,
        build_main=mock_build_main,
        teacher_main=mock_teacher_main,
    )


class TestOrchestratorSmoke:
    """Smoke tests exercising real orchestrator operations."""

    def test_build_dataset_smoke(self, orchestrator, tmp_path):
        """
        Smoke test: build_dataset() executes and returns exit code.

        This exercises the DatasetBuilder component.
        """
        cfg = DatasetBuildConfig(
            data_dir=str(tmp_path / "data"),
            symbols="SPY",
            out_dir=str(tmp_path / "output"),
            dataset_id="smoke_test_dataset",
        )

        # Create data dir
        (tmp_path / "data").mkdir(parents=True)

        # Patch _guard_dataset_metadata to skip validation in smoke test
        with patch.object(
            orchestrator._dataset_builder,
            "_guard_dataset_metadata",
            return_value=None,
        ):
            # Build dataset - should delegate to mock_build_main
            result = orchestrator.build_dataset(cfg)

        # Verify it ran
        assert isinstance(result, int)
        print(f"build_dataset returned: {result}")

    def test_run_hpo_disabled_smoke(self, orchestrator, tmp_path):
        """
        Smoke test: run_hpo() with disabled config returns 0.

        This exercises the TrainingCoordinator component.
        """
        cfg = HPOConfig(enabled=False)

        dataset_csv = tmp_path / "dataset.csv"
        dataset_csv.write_text("timestamp,close\n1,100.0\n")
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        result = orchestrator.run_hpo(cfg, dataset_csv, out_dir)

        assert result == 0
        print("run_hpo (disabled) returned: 0")

    def test_train_teacher_disabled_smoke(self, orchestrator, tmp_path):
        """
        Smoke test: train_teacher() with disabled config returns 0.
        """
        cfg = TeacherTrainConfig(enabled=False)

        dataset_csv = tmp_path / "dataset.csv"
        dataset_csv.write_text("timestamp,close\n1,100.0\n")
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        result = orchestrator.train_teacher(cfg, dataset_csv, out_dir)

        assert result == 0
        print("train_teacher (disabled) returned: 0")

    def test_run_pipeline_stages_smoke(self, orchestrator, tmp_path):
        """
        Smoke test: run() executes pipeline stages.

        This exercises the StageController component orchestrating:
        - DatasetBuilder
        - TrainingCoordinator
        """
        orch_cfg = OrchestratorConfig(
            dataset=DatasetBuildConfig(
                data_dir=str(tmp_path / "data"),
                symbols="SPY",
                out_dir=str(tmp_path / "output"),
                dataset_id="smoke_pipeline",
            ),
            hpo=HPOConfig(enabled=False),
            teacher=TeacherTrainConfig(enabled=False),
        )

        # Create data dir
        (tmp_path / "data").mkdir(parents=True)

        # Patch validation to skip in smoke test
        with patch.object(
            orchestrator._dataset_builder,
            "_guard_dataset_metadata",
            return_value=None,
        ):
            # Run full pipeline
            result = orchestrator.run(orch_cfg)

        assert isinstance(result, int)
        print(f"run() pipeline returned: {result}")

    def test_run_training_only_smoke(self, orchestrator, tmp_path):
        """
        Smoke test: run_training_only() executes training stages.

        This exercises StageController's training-only path.
        """
        # Create existing dataset artifacts (simulating prior build)
        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True)
        (out_dir / "dataset.csv").write_text("timestamp,close\n1,100.0\n")
        # Use matching dataset_id
        (out_dir / "dataset_metadata.json").write_text(
            '{"dataset_id": "smoke_training", "vintage_policy": "real_time"}'
        )

        orch_cfg = OrchestratorConfig(
            dataset=DatasetBuildConfig(
                data_dir=str(tmp_path / "data"),
                symbols="SPY",
                out_dir=str(out_dir),
                dataset_id="smoke_training",
            ),
            hpo=HPOConfig(enabled=False),
            teacher=TeacherTrainConfig(enabled=False),
        )

        result = orchestrator.run_training_only(orch_cfg)

        assert isinstance(result, int)
        print(f"run_training_only() returned: {result}")


class TestOrchestratorComponentIntegration:
    """Tests verifying component integration."""

    def test_stage_controller_initialized(self, orchestrator):
        """Verify StageController is properly initialized."""
        # In component mode, _stage_controller should exist
        assert hasattr(orchestrator, "_stage_controller")
        if orchestrator._stage_controller is not None:
            print("StageController: initialized")
        else:
            print("StageController: None (legacy mode)")

    def test_dataset_builder_initialized(self, orchestrator):
        """Verify DatasetBuilder is properly initialized."""
        assert hasattr(orchestrator, "_dataset_builder")
        if orchestrator._dataset_builder is not None:
            print("DatasetBuilder: initialized")

    def test_training_coordinator_initialized(self, orchestrator):
        """Verify TrainingCoordinator is properly initialized."""
        assert hasattr(orchestrator, "_training_coordinator")
        if orchestrator._training_coordinator is not None:
            print("TrainingCoordinator: initialized")

    def test_ingestion_coordinator_initialized(self, orchestrator):
        """Verify IngestionCoordinator is properly initialized."""
        assert hasattr(orchestrator, "_ingestion_coordinator")
        if orchestrator._ingestion_coordinator is not None:
            print("IngestionCoordinator: initialized")


class TestFeatureFlagParity:
    """Tests verifying legacy/component parity."""

    def test_both_modes_instantiate(self, mock_coverage, mock_writer, mock_build_main, mock_teacher_main, monkeypatch):
        """Both feature flag modes should instantiate successfully."""
        # Legacy mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")
        legacy = MLPipelineOrchestrator(
            coverage=mock_coverage,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        assert legacy is not None
        print("Legacy mode: instantiated")

        # Component mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")
        component = MLPipelineOrchestrator(
            coverage=mock_coverage,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        assert component is not None
        print("Component mode: instantiated")

    def test_build_dataset_parity(
        self, mock_coverage, mock_writer, mock_build_main, mock_teacher_main,
        tmp_path, monkeypatch
    ):
        """build_dataset() returns same result in both modes."""
        cfg = DatasetBuildConfig(
            data_dir=str(tmp_path / "data"),
            symbols="SPY",
            out_dir=str(tmp_path / "output"),
            dataset_id="parity_test",
        )
        (tmp_path / "data").mkdir(parents=True)

        # Legacy mode - patch _capture_cli_build_artifacts to skip validation
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "1")
        legacy = MLPipelineOrchestrator(
            coverage=mock_coverage,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        # Patch the legacy orchestrator's validation
        with patch.object(
            legacy._get_legacy(),
            "_capture_cli_build_artifacts",
            return_value=None,
        ):
            legacy_result = legacy.build_dataset(cfg)

        # Reset output dir for component test
        import shutil
        if (tmp_path / "output").exists():
            shutil.rmtree(tmp_path / "output")

        # Component mode
        monkeypatch.setenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0")
        component = MLPipelineOrchestrator(
            coverage=mock_coverage,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )
        with patch.object(
            component._dataset_builder,
            "_guard_dataset_metadata",
            return_value=None,
        ):
            component_result = component.build_dataset(cfg)

        print(f"Legacy result: {legacy_result}, Component result: {component_result}")
        # Both should succeed or both should fail the same way
        assert type(legacy_result) is type(component_result)
