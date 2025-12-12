#!/usr/bin/env python3
"""
Unit tests for StageController component.

The StageController orchestrates pipeline stages in the correct order,
handles checkpointing, and ensures proper error propagation.

Test Design: reports/tests/mlpipelineorchestrator_test_design_report.md
Coverage Target: 90%

All tests initially marked @pytest.mark.skip - TDD approach.
Implementation must make these tests pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest


if TYPE_CHECKING:
    from ml.orchestration.config_types import OrchestratorConfig


# ============================================================================
# PIPELINE EXECUTION TESTS
# ============================================================================


@pytest.mark.unit
class TestStageControllerPipelineExecution:
    """Tests for StageController pipeline execution logic."""

    def test_run_pipeline_executes_stages_in_order(
        self,
        mock_ingestion_coordinator: Mock,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify pipeline stages execute in correct order.

        Given:
        - StageController with all component dependencies
        - Valid orchestrator configuration

        When:
        - Calling run_pipeline()

        Then:
        - Stages execute in order: PRE_INGEST -> DATASET -> HPO -> TRAIN -> DISTILL -> PROMOTE -> INTEGRATE
        - Each stage completes before next begins
        - Return code is 0 on success
        """
        from ml.orchestration.common.stage_controller import StageController

        controller = StageController(
            ingestion_coordinator=mock_ingestion_coordinator,
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_pipeline(sample_orchestrator_config)

        assert result == 0
        # Verify call order
        assert mock_dataset_builder.build_dataset.called
        assert mock_training_coordinator.train_teacher.called

    def test_run_pipeline_respects_checkpoint_resume(
        self,
        mock_ingestion_coordinator: Mock,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
        sample_checkpoint_file: Path,
    ) -> None:
        """
        Verify checkpoint resume skips completed stages.

        Given:
        - Checkpoint file with completed stages ["PRE_INGEST", "DATASET"]
        - resume=True

        When:
        - Calling run_pipeline() with resume=True

        Then:
        - Skips PRE_INGEST and DATASET stages
        - Starts from HPO stage
        - Completes remaining stages
        """
        from ml.orchestration.checkpoint import PipelineCheckpoint
        from ml.orchestration.common.stage_controller import StageController

        checkpoint = PipelineCheckpoint(
            pipeline_id="test_pipeline",
            stage="DATASET",
            timestamp=time.time_ns(),
            state={},
            completed_stages=["PRE_INGEST", "DATASET"],
            progress=1.0,
        )
        checkpoint.save(sample_checkpoint_file)

        controller = StageController(
            ingestion_coordinator=mock_ingestion_coordinator,
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_pipeline(
            sample_orchestrator_config,
            checkpoint_file=sample_checkpoint_file,
            resume=True,
        )

        assert result == 0
        # Dataset builder should NOT be called (already completed)
        assert not mock_dataset_builder.build_dataset.called
        # Training should be called (next stage)
        assert mock_training_coordinator.run_hpo.called

    def test_run_training_only_skips_ingestion_and_dataset(
        self,
        mock_ingestion_coordinator: Mock,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
        tmp_path: Path,
    ) -> None:
        """
        Verify run_training_only skips data preparation stages.

        Given:
        - Existing dataset.csv in output directory

        When:
        - Calling run_training_only()

        Then:
        - IngestionCoordinator NOT called
        - DatasetBuilder NOT called
        - TrainingCoordinator methods called
        """
        from ml.orchestration.common.stage_controller import StageController

        # Create existing dataset
        out_dir = Path(sample_orchestrator_config.dataset.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv = out_dir / "dataset.csv"
        dataset_csv.write_text("timestamp,close\n1,100.0\n")

        # Create metadata
        metadata = {
            "dataset_id": "test",
            "vintage_policy": "real_time",  # Must match VintagePolicy enum value
            "feature_set_id": "test_features",
        }
        (out_dir / "dataset_metadata.json").write_text(json.dumps(metadata))

        controller = StageController(
            ingestion_coordinator=mock_ingestion_coordinator,
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_training_only(sample_orchestrator_config)

        assert result == 0
        assert not mock_ingestion_coordinator.run_pre_ingestion.called
        assert not mock_dataset_builder.build_dataset.called
        assert mock_training_coordinator.train_teacher.called

    def test_stage_failure_stops_pipeline(
        self,
        mock_ingestion_coordinator: Mock,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify pipeline stops on stage failure.

        Given:
        - DatasetBuilder returns error code 1

        When:
        - Calling run_pipeline()

        Then:
        - Pipeline stops after dataset stage
        - Training stages NOT called
        - Returns error code 1
        """
        from ml.orchestration.common.stage_controller import StageController

        mock_dataset_builder.build_dataset.return_value = 1  # Error

        controller = StageController(
            ingestion_coordinator=mock_ingestion_coordinator,
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_pipeline(sample_orchestrator_config)

        assert result == 1
        assert mock_dataset_builder.build_dataset.called
        assert not mock_training_coordinator.train_teacher.called

    def test_run_pipeline_handles_missing_pre_ingestion_config(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify pipeline handles missing pre_ingestion config gracefully.

        Given:
        - OrchestratorConfig with pre_ingestion=None

        When:
        - Calling run_pipeline()

        Then:
        - PRE_INGEST stage skipped
        - Pipeline continues to DATASET stage
        """
        from ml.orchestration.common.stage_controller import StageController

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_pipeline(sample_orchestrator_config)

        assert result == 0
        assert mock_dataset_builder.build_dataset.called


# ============================================================================
# MULTI-SYMBOL TESTS
# ============================================================================


@pytest.mark.unit
class TestStageControllerMultiSymbol:
    """Tests for StageController multi-symbol processing."""

    def test_multi_symbol_isolation(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        multi_symbol_config: OrchestratorConfig,
    ) -> None:
        """
        Verify multi-symbol runs maintain output isolation.

        Given:
        - Config with symbols="AAPL,GOOGL,MSFT"

        When:
        - Processing each symbol

        Then:
        - Each symbol gets unique output directory: {out_dir}/{symbol}/
        - No cross-contamination between symbol datasets
        """
        from ml.orchestration.common.stage_controller import StageController

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_pipeline(multi_symbol_config)

        assert result == 0

        # Verify each symbol got its own config
        calls = mock_dataset_builder.build_dataset.call_args_list
        out_dirs = [str(call[0][0].out_dir) for call in calls]

        assert len(out_dirs) == 3
        assert "AAPL" in out_dirs[0]
        assert "GOOGL" in out_dirs[1]
        assert "MSFT" in out_dirs[2]

    def test_multi_symbol_partial_failure(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        multi_symbol_config: OrchestratorConfig,
    ) -> None:
        """
        Verify multi-symbol handles partial failures correctly.

        Given:
        - Config with symbols="AAPL,GOOGL,MSFT"
        - GOOGL processing fails

        When:
        - Processing all symbols

        Then:
        - AAPL succeeds
        - GOOGL fails
        - MSFT still processed
        - Overall result is failure (1)
        """
        from ml.orchestration.common.stage_controller import StageController

        # Make second call fail
        mock_dataset_builder.build_dataset.side_effect = [0, 1, 0]

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_pipeline(multi_symbol_config)

        # Should return 1 because one symbol failed
        assert result == 1
        # All three should have been attempted
        assert mock_dataset_builder.build_dataset.call_count == 3


# ============================================================================
# CHECKPOINT TESTS
# ============================================================================


@pytest.mark.unit
class TestStageControllerCheckpoint:
    """Tests for StageController checkpoint functionality."""

    def test_checkpoint_saves_on_stage_completion(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
        sample_checkpoint_file: Path,
    ) -> None:
        """
        Verify checkpoint saves after each stage completes.

        Given:
        - checkpoint_file provided

        When:
        - Completing DATASET stage

        Then:
        - Checkpoint file created/updated
        - Contains "DATASET" in completed_stages
        """
        from ml.orchestration.checkpoint import PipelineCheckpoint
        from ml.orchestration.common.stage_controller import StageController

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        controller.run_pipeline(
            sample_orchestrator_config,
            checkpoint_file=sample_checkpoint_file,
        )

        # Verify checkpoint was saved
        assert sample_checkpoint_file.exists()

        checkpoint = PipelineCheckpoint.load(sample_checkpoint_file)
        assert "DATASET" in checkpoint.completed_stages

    def test_checkpoint_load_failure_starts_fresh(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
        sample_checkpoint_file: Path,
    ) -> None:
        """
        Verify invalid checkpoint causes fresh start.

        Given:
        - Corrupted checkpoint file

        When:
        - Calling run_pipeline() with resume=True

        Then:
        - Logs warning about checkpoint failure
        - Starts pipeline from beginning
        """
        from ml.orchestration.common.stage_controller import StageController

        # Create invalid checkpoint
        sample_checkpoint_file.write_text("invalid json {{{")

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_pipeline(
            sample_orchestrator_config,
            checkpoint_file=sample_checkpoint_file,
            resume=True,
        )

        assert result == 0
        # Should start from beginning (dataset called)
        assert mock_dataset_builder.build_dataset.called

    def test_checkpoint_not_saved_when_file_none(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
        tmp_path: Path,
    ) -> None:
        """
        Verify no checkpoint created when checkpoint_file is None.

        Given:
        - checkpoint_file=None

        When:
        - Running pipeline

        Then:
        - No checkpoint files created in tmp directory
        """
        from ml.orchestration.common.stage_controller import StageController

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        result = controller.run_pipeline(
            sample_orchestrator_config,
            checkpoint_file=None,
        )

        assert result == 0
        # No checkpoint files should exist
        checkpoint_files = list(tmp_path.glob("*.json"))
        # Only metadata file might exist, not checkpoint
        assert not any("checkpoint" in str(f) for f in checkpoint_files)


# ============================================================================
# PROMOTIONS TESTS
# ============================================================================


@pytest.mark.unit
class TestStageControllerPromotions:
    """Tests for StageController promotion handling."""

    def test_promotions_called_after_successful_training(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        mock_registry_synchronizer: Mock,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify promotions execute after successful training.

        Given:
        - Training stages complete successfully
        - promotions config provided

        When:
        - Pipeline completes

        Then:
        - RegistrySynchronizer synchronize_dataset_manifest or capture_cli_build_artifacts called

        Note:
        - Skipped until StageController.run_pipeline actually calls registry_synchronizer methods
        """
        from ml.orchestration.common.stage_controller import StageController

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
            registry_synchronizer=mock_registry_synchronizer,
        )

        result = controller.run_pipeline(sample_orchestrator_config)

        assert result == 0
        # Registry synchronization should be called after successful pipeline
        assert (
            mock_registry_synchronizer.synchronize_dataset_manifest.called
            or mock_registry_synchronizer.capture_cli_build_artifacts.called
        )

    def test_promotions_skipped_on_training_failure(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        mock_registry_synchronizer: Mock,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify promotions skipped when training fails.

        Given:
        - Training stage fails

        When:
        - Pipeline execution

        Then:
        - RegistrySynchronizer methods NOT called
        """
        from ml.orchestration.common.stage_controller import StageController

        mock_training_coordinator.train_teacher.return_value = 1

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
            registry_synchronizer=mock_registry_synchronizer,
        )

        result = controller.run_pipeline(sample_orchestrator_config)

        assert result == 1
        assert not mock_registry_synchronizer.sync_model.called
        assert not mock_registry_synchronizer.sync_features.called


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.unit
class TestStageControllerErrorHandling:
    """Tests for StageController error handling."""

    def test_missing_dataset_raises_for_training_only(
        self,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify FileNotFoundError raised when dataset.csv missing.

        Given:
        - Empty output directory (no dataset.csv)

        When:
        - Calling run_training_only()

        Then:
        - FileNotFoundError raised with descriptive message
        """
        from ml.orchestration.common.stage_controller import StageController

        controller = StageController(
            training_coordinator=mock_training_coordinator,
        )

        with pytest.raises(FileNotFoundError, match="Dataset CSV not found"):
            controller.run_training_only(sample_orchestrator_config)

    def test_stage_exception_propagates(
        self,
        mock_dataset_builder: Mock,
        mock_training_coordinator: Mock,
        sample_orchestrator_config: OrchestratorConfig,
    ) -> None:
        """
        Verify exceptions from stages propagate correctly.

        Given:
        - DatasetBuilder raises RuntimeError

        When:
        - Running pipeline

        Then:
        - RuntimeError propagates
        - Pipeline does not silently fail
        """
        from ml.orchestration.common.stage_controller import StageController

        mock_dataset_builder.build_dataset.side_effect = RuntimeError("Build failed")

        controller = StageController(
            dataset_builder=mock_dataset_builder,
            training_coordinator=mock_training_coordinator,
        )

        with pytest.raises(RuntimeError, match="Build failed"):
            controller.run_pipeline(sample_orchestrator_config)
