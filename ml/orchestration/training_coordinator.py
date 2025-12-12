#!/usr/bin/env python3

"""
Training coordination for ML pipeline orchestrator.

This module provides model training orchestration including:
- Hyperparameter optimization (HPO)
- Teacher model training
- Student model distillation

This component is extracted from the MLPipelineOrchestrator god class to provide
focused, testable training coordination functionality.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ml.data import DatasetMetadata
from ml.data import load_dataset_metadata
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.orchestration.dataset_builder import BuildArtifacts


logger = logging.getLogger(__name__)


# ========================================================================
# Protocol Definitions
# ========================================================================


class CliMain(Protocol):
    """Protocol for CLI main entry points."""

    def __call__(self, argv: list[str] | None = None) -> int:
        """Execute CLI with arguments."""
        ...


class TrainingCoordinatorProtocol(Protocol):
    """
    Protocol for training coordination operations.
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
        *,
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
            Teacher configuration (for parent model ID)

        Returns
        -------
        int
            Exit code (0 for success)

        """
        ...


# ========================================================================
# TrainingCoordinator Implementation
# ========================================================================


class TrainingCoordinator:
    """
    Coordinates ML model training workflows.

    Handles hyperparameter optimization, teacher model training,
    and student model distillation.

    This component is extracted from the MLPipelineOrchestrator god class to
    provide focused, testable training coordination functionality.

    Parameters
    ----------
    teacher_main : CliMain
        CLI entry point for teacher training
    hpo_main : CliMain | None
        CLI entry point for HPO (optional)
    build_artifacts : BuildArtifacts | None
        Build artifacts from dataset building phase

    """

    def __init__(
        self,
        *,
        teacher_main: CliMain | None,
        hpo_main: CliMain | None = None,
        build_artifacts: BuildArtifacts | None = None,
        model_store: object | None = None,
        model_registry: object | None = None,
        distill_cli: CliMain | None = None,
    ) -> None:
        """
        Initialize training coordinator.

        Parameters
        ----------
        teacher_main : CliMain
            CLI entry point for teacher training
        hpo_main : CliMain | None
            CLI entry point for HPO (optional)
        build_artifacts : BuildArtifacts | None
            Build artifacts from dataset building phase

        """
        self._teacher_main = teacher_main
        self._hpo_main = hpo_main
        self._build_artifacts = build_artifacts
        self.model_store = model_store
        self.model_registry = model_registry
        self._distill_cli = distill_cli
        self.hpo_main = hpo_main
        self.teacher_main = teacher_main
        self.distill_cli = distill_cli

        logger.debug("Initialized TrainingCoordinator")

    @property
    def build_artifacts(self) -> BuildArtifacts | None:
        """Return the current build artifacts."""
        return self._build_artifacts

    @build_artifacts.setter
    def build_artifacts(self, value: BuildArtifacts | None) -> None:
        """Set the build artifacts."""
        self._build_artifacts = value

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

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
        if (
            cfg is None
            or not isinstance(cfg, HPOConfig)
            or not cfg.enabled
            or self._hpo_main is None
        ):
            return 0

        artifacts = self._build_artifacts
        args = [
            "--dataset_csv",
            str(dataset_csv),
            "--out_dir",
            str(out_dir),
            "--epochs",
            str(cfg.epochs),
            "--batch_size",
            str(cfg.batch_size),
            "--tail_rows",
            str(cfg.tail_rows),
            "--limit_groups",
            str(cfg.limit_groups),
            "--workers",
            str(cfg.workers),
            "--backend",
            cfg.backend,
            "--metric",
            cfg.metric,
            "--optuna_trials",
            str(cfg.optuna_trials),
            "--loss",
            cfg.loss,
            "--pos_weight",
            cfg.pos_weight,
        ]
        if cfg.direction:
            args += ["--direction", cfg.direction]
        if cfg.optuna_timeout is not None:
            args += ["--optuna_timeout", str(cfg.optuna_timeout)]
        if artifacts and artifacts.feature_registry_dir:
            args += ["--feature_registry_dir", artifacts.feature_registry_dir]
        if artifacts and artifacts.feature_set_id:
            args += ["--feature_set_id", artifacts.feature_set_id]

        return self._hpo_main(args)

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

        Raises
        ------
        FileNotFoundError
            If dataset metadata is missing
        ValueError
            If dataset metadata is invalid

        """
        if (
            cfg is None
            or not isinstance(cfg, TeacherTrainConfig)
            or not cfg.enabled
        ):
            return 0
        if self._teacher_main is None:
            return 0

        artifacts = self._build_artifacts
        feature_registry_dir = cfg.feature_registry_dir or (
            artifacts.feature_registry_dir if artifacts else None
        )
        feature_set_id = cfg.feature_set_id or (artifacts.feature_set_id if artifacts else None)

        metadata_path = dataset_csv.parent / "dataset_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata missing at {metadata_path}")

        metadata_source: DatasetMetadata | None = None
        if artifacts and artifacts.dataset_metadata is not None:
            metadata_source = artifacts.dataset_metadata
        else:
            try:
                metadata_source = load_dataset_metadata(metadata_path)
            except Exception as exc:
                logger.debug("Failed to load dataset metadata prior to training: %s", exc)

        if metadata_source is None or metadata_source.dataset_id is None:
            raise ValueError("Dataset metadata must include dataset_id before teacher training")

        args: list[str] = [
            "--train_data_csv",
            str(dataset_csv),
            "--out_dir",
            str(out_dir),
            "--model_id",
            cfg.model_id,
            "--max_epochs",
            str(cfg.max_epochs),
            "--dataset_metadata",
            str(metadata_path),
            "--expected_dataset_id",
            metadata_source.dataset_id,
            "--expected_vintage_policy",
            metadata_source.vintage_policy.value,
        ]
        if metadata_source.vintage_cutoff:
            args += ["--expected_vintage_cutoff", metadata_source.vintage_cutoff]
        if feature_registry_dir is not None:
            args += ["--feature_registry_dir", feature_registry_dir]
        if feature_set_id is not None:
            args += ["--feature_set_id", feature_set_id]

        return self._teacher_main(args)

    def distill_student(
        self,
        cfg: StudentDistillConfig | None,
        *,
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
            Teacher configuration (for parent model ID)

        Returns
        -------
        int
            Exit code (0 for success, 1 for error)

        """
        if (
            cfg is None
            or not isinstance(cfg, StudentDistillConfig)
            or not cfg.enabled
        ):
            return 0

        features_npz = dataset_dir / "features_npz.npz"
        teacher_npz = dataset_dir / "teacher_preds.npz"
        if not features_npz.exists():
            logger.error("Distillation enabled but missing features NPZ at %s", features_npz)
            return 1
        if not teacher_npz.exists():
            logger.error(
                "Distillation enabled but missing teacher predictions NPZ at %s", teacher_npz
            )
            return 1

        artifacts = self._build_artifacts
        feature_registry_dir = cfg.feature_registry_dir or (
            artifacts.feature_registry_dir if artifacts else None
        )
        feature_set_id = cfg.feature_set_id or (artifacts.feature_set_id if artifacts else None)
        if feature_registry_dir is None or feature_set_id is None:
            logger.error(
                "Feature registry metadata required for distillation (have dir=%s id=%s)",
                feature_registry_dir,
                feature_set_id,
            )
            return 1

        model_registry_dir = cfg.model_registry_dir
        if model_registry_dir is None:
            logger.error("model_registry_dir is required for student registration")
            return 1

        parent_model_id = cfg.parent_model_id or (teacher_cfg.model_id if teacher_cfg else None)
        if parent_model_id is None:
            logger.error(
                "parent_model_id is required for student registration "
                "(provide cfg.parent_model_id or teacher_cfg)"
            )
            return 1

        args: list[str] = [
            "--features_npz",
            str(features_npz),
            "--teacher_npz",
            str(teacher_npz),
            "--out_dir",
            str(dataset_dir),
            "--model_id",
            cfg.model_id,
            "--parent_id",
            parent_model_id,
            "--registry_dir",
            model_registry_dir,
            "--feature_registry_dir",
            feature_registry_dir,
            "--feature_set_id",
            feature_set_id,
            "--objective",
            cfg.objective,
            "--kd_lambda",
            str(cfg.kd_lambda),
            "--early_stopping",
            str(cfg.early_stopping),
        ]
        if cfg.opset is not None:
            args += ["--opset", str(cfg.opset)]
        if cfg.use_val_for_distill:
            args += ["--use_val_for_distill"]

        from ml.training.distillation.cli import main as distill_main

        return distill_main(args)

    # ------------------------------------------------------------------
    # Structural compatibility helpers (Phase0 placeholders)
    # ------------------------------------------------------------------

    def run_training_only(self, cfg: object) -> int:
        del cfg
        return 0

    def _handle_promotions(self, promotion_cfg: object) -> None:
        del promotion_cfg
        return None

    def _execute_stage(self, stage: object, cfg: object) -> int:
        del stage, cfg
        return 0
