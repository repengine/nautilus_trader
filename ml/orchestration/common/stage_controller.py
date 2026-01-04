"""
Stage controller for ML pipeline orchestration.

This module provides the StageController component that orchestrates ML pipeline
stages in the correct order with checkpoint support for resume capability.

The StageController is extracted from the MLPipelineOrchestrator god class to
provide focused, testable pipeline orchestration functionality.

"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid as _uuid
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from ml.orchestration.common.protocols import DatasetBuilderProtocol
from ml.orchestration.common.protocols import IngestionCoordinatorProtocol
from ml.orchestration.common.protocols import RegistrySynchronizerProtocol
from ml.orchestration.common.protocols import RuntimeAttacherProtocol
from ml.orchestration.common.protocols import TrainingCoordinatorProtocol
from ml.orchestration.common.types import PipelineCheckpoint
from ml.orchestration.common.utils import parse_symbols


if TYPE_CHECKING:
    from ml.orchestration.config_types import AutoFillUniverseConfig
    from ml.orchestration.config_types import DatasetBuildConfig
    from ml.orchestration.config_types import IntegrationConfig
    from ml.orchestration.config_types import OrchestratorConfig
    from ml.orchestration.config_types import PromotionsConfig


logger = logging.getLogger(__name__)


@runtime_checkable
class IntegrationManagerProtocol(Protocol):
    """Protocol for MLIntegrationManager."""

    data_registry: object | None
    feature_registry: object | None
    model_registry: object | None
    strategy_registry: object | None
    data_store: object | None
    feature_store: object | None
    model_store: object | None
    strategy_store: object | None
    partition_manager: object | None


@dataclass
class StageController:
    """
    Orchestrates ML pipeline stages in correct order.

    The StageController manages the execution of ML pipeline stages including:
    - PRE_INGEST: Pre-ingestion tasks
    - AUTO_FILL: Universe auto-fill
    - DATASET: Dataset building
    - HPO: Hyperparameter optimization
    - TRAIN: Teacher model training
    - DISTILL: Student model distillation
    - PROMOTE: Model/feature promotion
    - INTEGRATE: Runtime attachment

    Supports checkpointing for resume capability and multi-symbol processing
    with output isolation.

    Attributes
    ----------
    ingestion_coordinator : IngestionCoordinatorProtocol | None
        Coordinator for ingestion operations
    dataset_builder : DatasetBuilderProtocol | None
        Builder for dataset construction
    training_coordinator : TrainingCoordinatorProtocol | None
        Coordinator for training operations
    registry_synchronizer : RegistrySynchronizerProtocol | None
        Synchronizer for registry operations
    runtime_attacher : RuntimeAttacherProtocol | None
        Attacher for runtime integration
    feature_registry : object | None
        Feature registry
    model_registry : object | None
        Model registry
    data_registry : object | None
        Data registry
    integration_manager_factory : Callable[..., IntegrationManagerProtocol] | None
        Factory for creating integration managers

    Examples
    --------
    >>> controller = StageController(
    ...     ingestion_coordinator=ingestion,
    ...     dataset_builder=dataset,
    ...     training_coordinator=training,
    ... )
    >>> result = controller.run_pipeline(config)
    >>> assert result == 0

    """

    ingestion_coordinator: IngestionCoordinatorProtocol | None = None
    dataset_builder: DatasetBuilderProtocol | None = None
    training_coordinator: TrainingCoordinatorProtocol | None = None
    registry_synchronizer: RegistrySynchronizerProtocol | None = None
    runtime_attacher: RuntimeAttacherProtocol | None = None
    feature_registry: object | None = None
    model_registry: object | None = None
    data_registry: object | None = None
    integration_manager_factory: Callable[..., IntegrationManagerProtocol] | None = None

    # Internal state (non-init)
    _integration_manager: IntegrationManagerProtocol | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _build_artifacts: object | None = field(
        default=None,
        init=False,
        repr=False,
    )

    # Callables for config preparation (injected from facade)
    _prepare_dataset_config: Callable[[DatasetBuildConfig], DatasetBuildConfig] | None = None
    _auto_fill_universe: Callable[[DatasetBuildConfig, AutoFillUniverseConfig], None] | None = None
    # Helper methods injected from legacy orchestrator (use keyword-only args)
    _handle_promotions: Callable[..., None] | None = None
    _attach_runtime: Callable[..., None] | None = None

    def run_pipeline(
        self,
        cfg: OrchestratorConfig,
        *,
        checkpoint_file: Path | None = None,
        resume: bool = False,
    ) -> int:
        """
        Execute full pipeline with checkpoint support.

        Runs all pipeline stages in order, handling checkpointing and
        multi-symbol processing.

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

        Examples
        --------
        >>> result = controller.run_pipeline(config)
        >>> result = controller.run_pipeline(
        ...     config,
        ...     checkpoint_file=Path("/tmp/checkpoint.json"),
        ...     resume=True,
        ... )

        """
        # Prepare dataset config
        dataset_cfg = self._do_prepare_dataset_config(cfg.dataset)
        cfg = replace(cfg, dataset=dataset_cfg)

        # Parse symbols to determine single vs multi-symbol processing
        symbols = parse_symbols(cfg.dataset.symbols)

        if len(symbols) == 1:
            # Single symbol - use direct processing
            return self._run_single_symbol(cfg, checkpoint_file=checkpoint_file, resume=resume)
        else:
            # Multi-symbol - process each independently with result isolation
            return self._run_multi_symbol(cfg, symbols, checkpoint_file=checkpoint_file, resume=resume)

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        """
        Execute training stages only (skip ingestion/dataset).

        Assumes dataset already exists and runs:
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

        Examples
        --------
        >>> result = controller.run_training_only(config)

        """
        from ml.data import load_dataset_metadata
        from ml.orchestration.dataset_builder import BuildArtifacts

        # Prepare dataset config
        dataset_cfg = self._do_prepare_dataset_config(cfg.dataset)
        cfg = replace(cfg, dataset=dataset_cfg)

        dataset_dir = Path(dataset_cfg.out_dir)
        dataset_csv = dataset_dir / "dataset.csv"
        if not dataset_csv.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found at {dataset_csv}; run dataset stage first",
            )

        metadata_path = dataset_dir / "dataset_metadata.json"
        dataset_metadata = load_dataset_metadata(metadata_path)

        # Resolve feature registry info
        feature_registry_dir = (
            getattr(cfg.teacher, "feature_registry_dir", None)
            or getattr(dataset_cfg, "feature_registry_dir", None)
            or (getattr(self._build_artifacts, "feature_registry_dir", None) if self._build_artifacts else None)
        )
        feature_set_id = (
            getattr(cfg.teacher, "feature_set_id", None)
            or getattr(dataset_metadata, "feature_set_id", None)
            or (getattr(self._build_artifacts, "feature_set_id", None) if self._build_artifacts else None)
        )

        # Create build artifacts for training coordinator
        self._build_artifacts = BuildArtifacts(
            out_dir=dataset_dir,
            feature_registry_dir=feature_registry_dir,
            feature_set_id=feature_set_id,
            dataset_metadata=dataset_metadata,
        )

        # Set build artifacts on training coordinator
        if self.training_coordinator is not None:
            try:
                self.training_coordinator.build_artifacts = self._build_artifacts  # type: ignore[attr-defined]
            except AttributeError:
                pass

        # Run HPO
        rc = self._run_hpo(cfg, dataset_csv, dataset_dir)
        if rc != 0:
            return rc

        # Run teacher training
        rc = self._run_train(cfg, dataset_csv, dataset_dir)
        if rc != 0:
            return rc

        # Run distillation
        rc = self._run_distill(cfg, dataset_dir)
        if rc != 0:
            return rc

        # Run promotions
        self._do_handle_promotions(cfg.promotions, out_dir=dataset_dir, dataset_csv=dataset_csv)

        # Attach runtime
        self._do_attach_runtime(cfg.integration, dataset_out_dir=dataset_dir)

        return 0

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _do_prepare_dataset_config(self, cfg: DatasetBuildConfig) -> DatasetBuildConfig:
        """Prepare dataset config using injected callable or passthrough."""
        if self._prepare_dataset_config is not None:
            return self._prepare_dataset_config(cfg)
        return cfg

    def _do_auto_fill_universe(
        self,
        dataset_cfg: DatasetBuildConfig,
        auto_fill_cfg: AutoFillUniverseConfig,
    ) -> None:
        """Execute auto-fill using injected callable."""
        if self._auto_fill_universe is not None:
            self._auto_fill_universe(dataset_cfg, auto_fill_cfg)

    def _do_handle_promotions(
        self,
        promotions: PromotionsConfig | None,
        *,
        out_dir: Path,
        dataset_csv: Path,
    ) -> None:
        """Execute promotions using injected callable or default implementation."""
        if self._handle_promotions is not None:
            self._handle_promotions(promotions, out_dir=out_dir, dataset_csv=dataset_csv)
            return

        # Default implementation
        if promotions is None:
            return

        self._default_handle_promotions(promotions, out_dir=out_dir, dataset_csv=dataset_csv)

    def _do_attach_runtime(
        self,
        integration_cfg: IntegrationConfig | None,
        *,
        dataset_out_dir: Path,
    ) -> None:
        """Execute runtime attachment using injected callable or default implementation."""
        if self._attach_runtime is not None:
            self._attach_runtime(integration_cfg, dataset_out_dir=dataset_out_dir)
            return

        # Default implementation
        self._default_attach_runtime(integration_cfg, dataset_out_dir=dataset_out_dir)

    def _run_multi_symbol(
        self,
        cfg: OrchestratorConfig,
        symbols: list[str],
        *,
        checkpoint_file: Path | None = None,
        resume: bool = False,
    ) -> int:
        """
        Process multiple symbols independently with result isolation.

        Parameters
        ----------
        cfg : OrchestratorConfig
            Base orchestrator configuration
        symbols : list[str]
            List of symbols to process
        checkpoint_file : Path | None
            Path to checkpoint file
        resume : bool
            If True, resume from checkpoint

        Returns
        -------
        int
            Exit code (0 if all symbols succeed, non-zero if any fail)

        """
        results: dict[str, int] = {}

        logger.info(
            "Starting multi-symbol orchestration",
            extra={
                "symbols": symbols,
                "num_symbols": len(symbols),
                "out_dir": cfg.dataset.out_dir,
            },
        )

        for symbol in symbols:
            logger.info(
                "Processing symbol",
                extra={
                    "symbol": symbol,
                    "progress": f"{len(results)}/{len(symbols)}",
                },
            )

            # Create isolated config for this symbol
            symbol_cfg = self._create_symbol_config(cfg, symbol)

            # Process symbol using single-symbol logic
            exit_code = self._run_single_symbol(
                symbol_cfg,
                checkpoint_file=checkpoint_file,
                resume=resume,
            )

            results[symbol] = exit_code

            if exit_code != 0:
                logger.error(
                    "Symbol processing failed",
                    extra={
                        "symbol": symbol,
                        "exit_code": exit_code,
                    },
                )
            else:
                logger.info(
                    "Symbol processing succeeded",
                    extra={"symbol": symbol},
                )

        # Log summary
        successful = [s for s, code in results.items() if code == 0]
        failed = [s for s, code in results.items() if code != 0]

        logger.info(
            "Multi-symbol orchestration complete",
            extra={
                "total_symbols": len(symbols),
                "successful": len(successful),
                "failed": len(failed),
                "successful_symbols": successful,
                "failed_symbols": failed,
            },
        )

        return 0 if all(code == 0 for code in results.values()) else 1

    def _create_symbol_config(
        self,
        cfg: OrchestratorConfig,
        symbol: str,
    ) -> OrchestratorConfig:
        """
        Create isolated configuration for a single symbol.

        Parameters
        ----------
        cfg : OrchestratorConfig
            Base orchestrator configuration
        symbol : str
            Single symbol to process

        Returns
        -------
        OrchestratorConfig
            Isolated configuration for the symbol

        """
        # Create symbol-specific output directory
        out_dir = Path(cfg.dataset.out_dir) / symbol
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create new dataset config with isolated output directory and single symbol
        dataset_cfg = replace(
            cfg.dataset,
            symbols=symbol,  # Single symbol only
            out_dir=str(out_dir),  # Isolated output directory
        )

        # Create new orchestrator config with symbol-specific dataset config
        return replace(cfg, dataset=dataset_cfg)

    def _run_single_symbol(
        self,
        cfg: OrchestratorConfig,
        *,
        checkpoint_file: Path | None = None,
        resume: bool = False,
    ) -> int:
        """
        Run the complete ML pipeline for a single symbol.

        Parameters
        ----------
        cfg : OrchestratorConfig
            Orchestrator configuration
        checkpoint_file : Path | None
            Path to checkpoint file
        resume : bool
            If True, resume from checkpoint

        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)

        """
        # Load checkpoint if resuming
        completed_stages: list[str] = []
        if resume and checkpoint_file is not None and checkpoint_file.exists():
            try:
                checkpoint = PipelineCheckpoint.load(checkpoint_file)
                completed_stages = checkpoint.completed_stages
                logger.info(
                    "Resuming from checkpoint",
                    extra={
                        "pipeline_id": checkpoint.pipeline_id,
                        "completed_stages": completed_stages,
                        "last_stage": checkpoint.stage,
                    },
                )
            except (FileNotFoundError, ValueError) as e:
                logger.warning(
                    "Failed to load checkpoint, starting from beginning",
                    extra={"checkpoint_file": str(checkpoint_file), "error": str(e)},
                    exc_info=True,
                )
                completed_stages = []

        # Generate pipeline ID for checkpoint tracking
        pipeline_id = f"pipeline_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        def save_checkpoint(stage: str, progress: float = 1.0) -> None:
            """Save checkpoint after completing a stage."""
            if checkpoint_file is None:
                return
            checkpoint = PipelineCheckpoint(
                pipeline_id=pipeline_id,
                stage=stage,
                timestamp=time.time_ns(),
                state={},
                completed_stages=completed_stages.copy(),
                progress=progress,
            )
            try:
                checkpoint.save(checkpoint_file)
            except (OSError, ValueError) as e:
                logger.warning(
                    "Failed to save checkpoint",
                    extra={"stage": stage, "error": str(e)},
                    exc_info=True,
                )

        # 0) Optional pre-ingestion stage
        if "PRE_INGEST" not in completed_stages and cfg.pre_ingestion is not None:
            catalog_path_env = os.getenv("CATALOG_PATH")
            if catalog_path_env and self.ingestion_coordinator is not None:
                self.ingestion_coordinator.run_pre_ingestion(
                    catalog_path=Path(catalog_path_env),
                    scheduler_cfg=cfg.pre_ingestion,
                    options=cfg.pre_ingestion_options,
                )
            completed_stages.append("PRE_INGEST")
            save_checkpoint("PRE_INGEST")

        # Auto-fill universe stage
        if "AUTO_FILL" not in completed_stages and cfg.auto_fill is not None and cfg.auto_fill.enabled:
            self._do_auto_fill_universe(cfg.dataset, cfg.auto_fill)
            completed_stages.append("AUTO_FILL")
            save_checkpoint("AUTO_FILL")

        out_dir = Path(cfg.dataset.out_dir)
        dataset_csv = out_dir / "dataset.csv"

        # 1) Build dataset
        if "DATASET" not in completed_stages:
            rc = self._run_dataset_build(cfg)
            if rc != 0:
                return rc
            completed_stages.append("DATASET")
            save_checkpoint("DATASET")

        # 2) HPO (optional)
        if "HPO" not in completed_stages:
            rc = self._run_hpo(cfg, dataset_csv, out_dir)
            if rc != 0:
                return rc
            completed_stages.append("HPO")
            save_checkpoint("HPO")

        # 3) Train teacher / calibration
        if "TRAIN" not in completed_stages:
            rc = self._run_train(cfg, dataset_csv, out_dir)
            if rc != 0:
                return rc
            completed_stages.append("TRAIN")
            save_checkpoint("TRAIN")

        # 4) Distill student
        if "DISTILL" not in completed_stages:
            rc = self._run_distill(cfg, out_dir)
            if rc != 0:
                return rc
            completed_stages.append("DISTILL")
            save_checkpoint("DISTILL")

        # 5) Handle promotions
        if "PROMOTE" not in completed_stages:
            self._do_handle_promotions(cfg.promotions, out_dir=out_dir, dataset_csv=dataset_csv)
            completed_stages.append("PROMOTE")
            save_checkpoint("PROMOTE")

        # 6) Attach runtime
        if "INTEGRATE" not in completed_stages:
            self._do_attach_runtime(cfg.integration, dataset_out_dir=out_dir)
            completed_stages.append("INTEGRATE")
            save_checkpoint("INTEGRATE")

        return 0

    def _run_dataset_build(self, cfg: OrchestratorConfig) -> int:
        """Run dataset build stage."""
        if self.dataset_builder is None:
            logger.warning("DatasetBuilder not configured; skipping dataset build")
            return 0
        rc = self.dataset_builder.build_dataset(cfg.dataset)
        if rc != 0:
            return rc

        if self.registry_synchronizer is None:
            return 0

        try:
            artifacts = self.registry_synchronizer.capture_cli_build_artifacts(cfg.dataset)
            if artifacts is not None:
                self._build_artifacts = artifacts
                if self.training_coordinator is not None:
                    try:
                        self.training_coordinator.build_artifacts = artifacts  # type: ignore[attr-defined]
                    except AttributeError:
                        pass
                dataset_metadata = getattr(artifacts, "dataset_metadata", None)
                if dataset_metadata is not None:
                    self.registry_synchronizer.synchronize_dataset_manifest(
                        cfg=cfg.dataset,
                        metadata=dataset_metadata,
                    )
        except Exception:
            logger.debug("Failed to synchronize dataset build artifacts", exc_info=True)

        return 0

    def _run_hpo(self, cfg: OrchestratorConfig, dataset_csv: Path, out_dir: Path) -> int:
        """Run HPO stage."""
        if self.training_coordinator is None:
            logger.warning("TrainingCoordinator not configured; skipping HPO")
            return 0
        return self.training_coordinator.run_hpo(cfg.hpo, dataset_csv, out_dir)

    def _run_train(self, cfg: OrchestratorConfig, dataset_csv: Path, out_dir: Path) -> int:
        """Run teacher training stage."""
        if self.training_coordinator is None:
            logger.warning("TrainingCoordinator not configured; skipping training")
            return 0
        return self.training_coordinator.train_teacher(cfg.teacher, dataset_csv, out_dir)

    def _run_distill(self, cfg: OrchestratorConfig, out_dir: Path) -> int:
        """Run distillation stage."""
        if self.training_coordinator is None:
            logger.warning("TrainingCoordinator not configured; skipping distillation")
            return 0
        return self.training_coordinator.distill_student(
            cfg.student,
            dataset_dir=out_dir,
            teacher_cfg=cfg.teacher,
        )

    def _default_handle_promotions(
        self,
        promotions: PromotionsConfig,
        *,
        out_dir: Path,
        dataset_csv: Path,
    ) -> None:
        """Default implementation for handling promotions."""
        try:
            from ml.orchestration.promotions import register_and_promote_model
            from ml.orchestration.promotions import register_or_refresh_features
            from ml.registry.dataclasses import QualityGate
        except ImportError:
            logger.warning("Promotions module not available; skipping promotions")
            return

        feature_registry = self.feature_registry
        feature_metrics_path = Path(
            promotions.feature_metrics_json or (Path(out_dir) / "feature_metrics.json"),
        )

        if promotions.refresh_features or promotions.auto_register_features:
            if not feature_metrics_path.exists():
                feature_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "feature_set_id": f"auto_refresh_{_uuid.uuid4().hex[:8]}",
                    "generated_ts": int(time.time()),
                    "dataset_csv": str(dataset_csv),
                }
                feature_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            succeeded = False
            if feature_registry is not None:
                try:
                    register_or_refresh_features(
                        feature_metrics_path=str(feature_metrics_path),
                        feature_registry=feature_registry,
                        auto_register=bool(promotions.auto_register_features),
                    )
                    succeeded = True
                except Exception as exc:
                    logger.warning("Feature refresh failed: %s", exc)

            if not succeeded:
                self._emit_feature_refresh_event(feature_metrics_path)

        should_promote_model = bool(
            promotions.auto_register_model or promotions.auto_promote or promotions.deploy_target,
        )

        if not should_promote_model or self.model_registry is None:
            return

        metrics_path = Path(out_dir) / "model_metrics.json"
        if not metrics_path.exists():
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model_id": f"auto_model_{_uuid.uuid4().hex[:8]}",
                "model_path": str(Path(out_dir) / "model.onnx"),
                "architecture": "unknown",
                "feature_schema_hash": f"auto_{_uuid.uuid4().hex[:8]}",
                "serveable": True,
            }
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        gates: list[QualityGate] = []
        if promotions.gates_json:
            try:
                data = json.loads(Path(promotions.gates_json).read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        metric_name = item.get("metric") or item.get("metric_name")
                        if not metric_name:
                            continue
                        comparison_value = item.get("comparison")
                        comparison: str = (
                            str(comparison_value) if comparison_value is not None else "gte"
                        )
                        gates.append(
                            QualityGate(
                                metric_name=str(metric_name),
                                threshold=float(item.get("threshold", 0.0)),
                                comparison=comparison,
                                required=bool(item.get("required", True)),
                            ),
                        )
            except Exception as exc:
                logger.warning("Failed to parse gates JSON %s: %s", promotions.gates_json, exc)
                gates = []

        try:
            register_and_promote_model(
                model_metrics_path=str(metrics_path),
                out_dir=str(out_dir),
                registry=self.model_registry,
                feature_registry=self.feature_registry,
                gates=gates,
                auto_promote=bool(promotions.auto_promote),
                deploy_target=promotions.deploy_target,
            )
        except Exception as exc:
            logger.warning("Model promotion failed: %s", exc)

    def _default_attach_runtime(
        self,
        integration_cfg: IntegrationConfig | None,
        *,
        dataset_out_dir: Path,
    ) -> None:
        """Default implementation for attaching runtime."""
        if integration_cfg is None or not integration_cfg.enabled:
            return

        logger.info(
            "Attaching ML integration runtime (validators=%s, out_dir=%s)",
            integration_cfg.run_validators,
            dataset_out_dir,
        )

        if self._integration_manager is None:
            factory = self.integration_manager_factory
            if factory is None:
                try:
                    from ml.core.integration import MLIntegrationManager as _MLIntegrationManager
                    factory = cast(
                        Callable[..., IntegrationManagerProtocol],
                        _MLIntegrationManager,
                    )
                except ImportError:
                    logger.warning("MLIntegrationManager not available; skipping runtime attachment")
                    return

            kwargs: dict[str, Any] = {
                "auto_start_postgres": integration_cfg.auto_start_postgres,
                "auto_migrate": integration_cfg.auto_migrate,
                "ensure_healthy": integration_cfg.ensure_healthy,
            }
            if integration_cfg.db_connection is not None:
                kwargs["db_connection"] = integration_cfg.db_connection
            if integration_cfg.strict_protocol_validation is not None:
                kwargs["strict_protocol_validation"] = integration_cfg.strict_protocol_validation

            manager = factory(**kwargs)
            self._integration_manager = manager
        else:
            manager = self._integration_manager

        # Sync registries and stores from manager
        for attr in (
            "data_registry",
            "feature_registry",
            "model_registry",
            "strategy_registry",
            "feature_store",
            "model_store",
            "strategy_store",
            "data_store",
            "partition_manager",
        ):
            if getattr(self, attr, None) is None:
                setattr(self, attr, getattr(manager, attr, None))

        if integration_cfg.run_validators:
            self._run_validators()

    def _run_validators(self) -> None:
        """Run runtime validators."""
        try:
            from tools import validate_event_constants as event_mod
            from tools import validate_metrics_bootstrap as metrics_mod
        except ImportError:
            logger.warning("Validator modules not available; skipping validation")
            return

        metrics_rc = metrics_mod.main()
        if metrics_rc != 0:
            raise RuntimeError("metrics bootstrap validation failed")

        events_rc = event_mod.main()
        if events_rc != 0:
            raise RuntimeError("event constants validation failed")

        logger.info("Runtime validators succeeded")

    def _emit_feature_refresh_event(self, metrics_path: Path) -> None:
        """Emit feature refresh event."""
        try:
            from ml.common.event_emitter import emit_dataset_event
            from ml.config.events import EventStatus
            from ml.config.events import Source
            from ml.config.events import Stage
            from ml.registry.protocols import RegistryProtocol
        except ImportError:
            return

        feature_set_id = "unknown"
        metadata: dict[str, object] = {}
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    feature_set_id = str(data.get("feature_set_id", feature_set_id))
                    metadata = {k: v for k, v in data.items() if isinstance(k, str)}
            except Exception:
                pass

        meta_payload = dict(metadata)
        meta_payload["feature_set_id"] = feature_set_id

        try:
            registry_obj = self.data_registry
            if registry_obj is None:
                return
            data_registry = cast(RegistryProtocol, registry_obj)
            emit_dataset_event(
                data_registry,
                dataset_id="features",
                instrument_id="GLOBAL",
                stage=Stage.FEATURE_COMPUTED,
                source=Source.HISTORICAL,
                run_id=f"refresh_{feature_set_id}",
                ts_min=0,
                ts_max=0,
                count=1,
                status=EventStatus.SUCCESS,
                metadata=meta_payload,
                dataset_type="features",
                component="stage_controller.refresh_features",
            )
        except Exception:
            pass
