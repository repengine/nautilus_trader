#!/usr/bin/env python3

"""
ModelRegistry facade maintaining backward compatibility.

This facade delegates to specialized components while preserving the original
public API. Feature flag ML_USE_LEGACY_MODEL_REGISTRY controls legacy vs new path.

Phase 2.3: ModelRegistry Decomposition - Strangler Fig Pattern
---------------------------------------------------------------
This facade provides 100% backward compatibility while allowing gradual
migration to the decomposed component architecture. The legacy monolithic
implementation can be restored via environment variable for safe rollback.

Components:
-----------
- ModelPersistence: Model saving/loading, artifact management, file I/O, SHA-256 integrity
- ModelQualityValidator: Quality gates, validation results, gate evaluation
- ModelDeploymentManager: Deployment tracking, version management, rollback, hot reload
- ABTestingManager: A/B test configuration, statistical analysis, metric tracking
- CanaryDeploymentManager: Canary release management, gradual rollout, promotion

"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from ml.config.registry import RegistryPolicyConfig
from ml.config.runtime import OnnxRuntimeConfig
from ml.registry.ab_testing_manager import ABTestingManager
from ml.registry.abstract_registry import AbstractRegistry
from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.canary_deployment_mgr import CanaryDeploymentManager
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment
from ml.registry.dataclasses import QualityGate
from ml.registry.dataclasses import ValidationResult
from ml.registry.model_deployment_mgr import ModelDeploymentManager
from ml.registry.model_persistence import ModelPersistence
from ml.registry.model_quality_validator import ModelQualityValidator
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)

__all__ = [
    "DataRequirements",
    "DeploymentStatus",
    "ModelInfo",
    "ModelManifest",
    "ModelRegistry",
    "ModelRole",
]

# Feature flag to control legacy vs new implementation
USE_LEGACY = os.getenv("ML_USE_LEGACY_MODEL_REGISTRY", "0") == "1"


class ModelRegistry(AbstractRegistry):
    """
    Model registry with configurable persistence backend.

    This facade delegates to specialized components while maintaining
    100% backward compatibility with the original ModelRegistry API.

    Feature Flag Control:
    ---------------------
    - ML_USE_LEGACY_MODEL_REGISTRY=1: Use original monolithic implementation
    - ML_USE_LEGACY_MODEL_REGISTRY=0: Use new component-based implementation (default)

    Component Architecture:
    ----------------------
    - ModelPersistence: Model saving/loading, artifact management, file I/O, SHA-256 integrity
    - ModelQualityValidator: Quality gates, validation results, gate evaluation
    - ModelDeploymentManager: Deployment tracking, version management, rollback, hot reload
    - ABTestingManager: A/B test configuration, statistical analysis, metric tracking
    - CanaryDeploymentManager: Canary release management, gradual rollout, promotion

    Parameters
    ----------
    registry_path : Path
        Directory path for registry storage (used for model files)
    cache_size : int
        Maximum number of models to cache in memory
    batch_save_interval : float
        Seconds to wait before flushing batch saves (default 0.1s)
    persistence_config : PersistenceConfig | None
        Persistence configuration. If None, defaults to JSON backend.
    policy_config : RegistryPolicyConfig | None
        Policy configuration for registry
    onnx_runtime_config : OnnxRuntimeConfig | None
        ONNX runtime configuration

    Examples
    --------
    >>> # Use new component-based implementation (default)
    >>> registry = ModelRegistry(registry_path=Path("models"))
    >>> model_id = registry.register_model(model_path, manifest)

    >>> # Use legacy implementation (rollback)
    >>> import os
    >>> os.environ["ML_USE_LEGACY_MODEL_REGISTRY"] = "1"
    >>> registry = ModelRegistry(registry_path=Path("models"))

    """

    def __init__(
        self,
        registry_path: Path,
        cache_size: int = 10,
        batch_save_interval: float = 0.1,
        persistence_config: PersistenceConfig | None = None,
        policy_config: RegistryPolicyConfig | None = None,
        onnx_runtime_config: OnnxRuntimeConfig | None = None,
    ) -> None:
        """
        Initialize ModelRegistry with configurable backend.

        Parameters match original constructor for compatibility.

        """
        if USE_LEGACY:
            # Use legacy monolithic implementation
            logger.info(
                "Using legacy ModelRegistry implementation (ML_USE_LEGACY_MODEL_REGISTRY=1)",
            )
            from ml.registry.model_registry_legacy import ModelRegistry as ModelRegistryLegacy

            self._legacy_impl = ModelRegistryLegacy(
                registry_path=registry_path,
                cache_size=cache_size,
                batch_save_interval=batch_save_interval,
                persistence_config=persistence_config,
                policy_config=policy_config,
                onnx_runtime_config=onnx_runtime_config,
            )
            self._use_legacy = True
            # Expose persistence for AbstractRegistry
            self.persistence = self._legacy_impl.persistence
            self.registry_path = registry_path
            return

        # Use new component-based implementation
        logger.info(
            "Using component-based ModelRegistry implementation (ML_USE_LEGACY_MODEL_REGISTRY=0)",
        )
        self._use_legacy = False

        # Initialize storage attributes
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.cache_size = cache_size
        self.batch_save_interval = batch_save_interval
        self._policy = policy_config or RegistryPolicyConfig()
        self._onnx_rt = onnx_runtime_config or OnnxRuntimeConfig()

        # Setup persistence
        if persistence_config is None:
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )
        persistence_manager = PersistenceManager(persistence_config)

        # Call parent init
        super().__init__(persistence_manager)

        # Initialize ModelPersistence component
        self._model_persistence = ModelPersistence(
            registry_path=registry_path,
            persistence_manager=persistence_manager,
            cache_size=cache_size,
            batch_save_interval=batch_save_interval,
            onnx_runtime_config=onnx_runtime_config,
        )

        # Load registry data
        self._models, self._ab_tests, self._deployments = (
            self._model_persistence.load_registry()
        )

        # Initialize other components
        self._quality_validator = ModelQualityValidator()

        self._deployment_manager = ModelDeploymentManager(
            models=self._models,
            deployments=self._deployments,
            save_callback=self._save_registry,
        )

        self._ab_testing_manager = ABTestingManager(
            models=self._models,
            deployments=self._deployments,
            ab_models_required=int(self._policy.ab_models_required),
            save_callback=self._save_registry,
        )

        self._canary_deployment_manager = CanaryDeploymentManager(
            models=self._models,
            ab_testing_manager=self._ab_testing_manager,
            deploy_callback=self.deploy_model,
            retire_callback=self.retire_model,
            save_callback=self._save_registry,
        )

        logger.info(
            "Initialized ModelRegistry facade with 5 components: "
            "ModelPersistence, ModelQualityValidator, ModelDeploymentManager, "
            "ABTestingManager, CanaryDeploymentManager",
        )

    def _save_registry(self, immediate: bool = False) -> None:
        """
        Save registry to persistence backend.

        Parameters
        ----------
        immediate : bool
            If True, save immediately; otherwise batch

        """
        if self._use_legacy:
            return self._legacy_impl._save_registry(immediate)
        self._model_persistence.save_registry(
            models=self._models,
            ab_tests=self._ab_tests,
            deployments=self._deployments,
            immediate=immediate,
        )

    # =========================================================================
    # Registration and Core Model Methods
    # =========================================================================

    def register_model(
        self,
        model_path: Path,
        manifest: ModelManifest,
        auto_deploy: bool = False,
        quality_gates: list[QualityGate] | None = None,
        enforce_quality: bool = False,
    ) -> str:
        """
        Register a new model with self-describing manifest.

        Parameters
        ----------
        model_path : Path
            Path to the model file
        manifest : ModelManifest
            Self-describing model manifest
        auto_deploy : bool
            Whether to automatically deploy if validation passes
        quality_gates : list[QualityGate] | None
            Quality gates to validate before registration
        enforce_quality : bool
            If True, raise error on quality gate failure

        Returns
        -------
        str
            Unique model ID

        """
        if self._use_legacy:
            return self._legacy_impl.register_model(
                model_path,
                manifest,
                auto_deploy,
                quality_gates,
                enforce_quality,
            )

        # Validate inputs
        self._validate_registration_inputs(model_path, manifest)

        # Calculate and set artifact SHA-256 digest for integrity verification
        from ml.config.constants import SUFFIX_ONNX
        if model_path.suffix == SUFFIX_ONNX:
            try:
                artifact_digest = self._model_persistence.calculate_file_sha256(model_path)
                manifest.artifact_sha256_digest = artifact_digest
                logger.info(
                    f"Calculated SHA-256 digest for model artifact: {artifact_digest[:16]}...",
                )
            except (OSError, FileNotFoundError) as e:
                logger.error("Failed to calculate artifact digest: %s", e, exc_info=True)
                raise ValueError(
                    f"Cannot calculate SHA-256 digest for model artifact: {e}",
                ) from e

        # Use manifest's model_id or generate new one
        if not manifest.model_id:
            manifest.model_id = f"model_{int(time.time() * 1000000)}"

        # Set timestamps
        manifest.created_at = time.time()
        manifest.last_modified = time.time()

        # Auto-version if needed
        self._auto_version_manifest(manifest)

        # Quality validation if gates provided
        quality_validation_result = self._apply_quality_gates(
            manifest,
            quality_gates,
            enforce_quality,
        )

        # Create model info
        model_info = ModelInfo(
            manifest=manifest,
            model_path=model_path,
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[],
            metadata={},
        )

        # Store quality validation result if performed
        if quality_validation_result:
            model_info.metadata["quality_validation"] = {
                "passed": quality_validation_result.overall_pass,
                "gates_passed": quality_validation_result.gates_passed,
                "gates_failed": quality_validation_result.gates_failed,
                "timestamp": quality_validation_result.timestamp,
            }

        # Handle parent-child relationships
        if manifest.parent_id and manifest.parent_id in self._models:
            parent = self._models[manifest.parent_id]
            if manifest.model_id not in parent.manifest.children_ids:
                parent.manifest.children_ids.append(manifest.model_id)
                parent.manifest.last_modified = time.time()

        # Store and save
        self._models[manifest.model_id] = model_info

        # Persist to backend
        if self.backend == BackendType.POSTGRES:
            self._model_persistence._save_model_to_db(model_info)
        else:
            self._save_registry()

        # Log audit
        self.log_audit(
            entity_type="model",
            entity_id=manifest.model_id,
            action="registered",
            changes={"role": manifest.role.value, "status": model_info.deployment_status.value},
        )

        logger.info(
            f"Registered {manifest.role.value} model {manifest.model_id} "
            f"(version {manifest.version}) at {model_path}",
        )

        # Auto-deploy if requested and validation passes
        if auto_deploy:
            self._maybe_auto_deploy(manifest)

        return manifest.model_id

    def _auto_version_manifest(self, manifest: ModelManifest) -> None:
        """
        Assign a semantic version to the manifest when missing.

        Parameters
        ----------
        manifest : ModelManifest
            Manifest to version

        """
        if manifest.version:
            return
        existing_versions = [
            m.manifest.version
            for m in self._models.values()
            if m.manifest.architecture == manifest.architecture
        ]
        if existing_versions:
            latest = max(existing_versions)
            major, minor, patch = latest.split(".")
            manifest.version = f"{major}.{minor}.{int(patch) + 1}"
        else:
            from ml.config.constants import Versions
            manifest.version = Versions.DEFAULT_MANIFEST_VERSION

    def _apply_quality_gates(
        self,
        manifest: ModelManifest,
        quality_gates: list[QualityGate] | None,
        enforce_quality: bool,
    ) -> ValidationResult | None:
        """
        Run quality gates when provided; optionally enforce.

        Parameters
        ----------
        manifest : ModelManifest
            Model manifest
        quality_gates : list[QualityGate] | None
            Quality gates to check
        enforce_quality : bool
            If True, raise error on failure

        Returns
        -------
        ValidationResult | None
            Validation result or None if no gates

        """
        if not quality_gates:
            return None

        result = self._quality_validator.validate_quality_gates(
            manifest.model_id,
            manifest.performance_metrics,
            quality_gates,
        )
        if not result.overall_pass and enforce_quality:
            failed_gates = [
                name
                for name, gate in result.gate_results.items()
                if not gate["passed"] and gate["required"]
            ]
            raise ValueError(
                f"Quality gates not met for model {manifest.model_id}. "
                f"Failed gates: {failed_gates}",
            )
        return result

    def _validate_registration_inputs(self, model_path: Path, manifest: ModelManifest) -> None:
        """
        Validate model path, format, security, and parity constraints.

        Parameters
        ----------
        model_path : Path
            Path to model file
        manifest : ModelManifest
            Model manifest

        """
        from ml.config.constants import SUFFIX_ONNX

        # Security: Validate model format for serving
        if model_path.suffix != SUFFIX_ONNX:
            # Allow non-ONNX for non-serveable models (e.g., cold-path teachers)
            if getattr(manifest, "serveable", True):
                raise ValueError(
                    f"Only ONNX models are supported for serveable models. Got: {model_path.suffix}.",
                )

        # Security: Validate path is safe
        if not self._model_persistence._validate_model_path(model_path):
            raise ValueError(f"Security: Invalid model path: {model_path}")

        # Validate feature schema linkage
        if not manifest.feature_schema_hash:
            raise ValueError("feature_schema_hash is required for all models")

        # Enforcement for serveable models: validate feature parity where possible
        if getattr(manifest, "serveable", True):
            # Allow relaxed parity in development unless explicitly enforced via env
            strict_parity = os.getenv("ML_STRICT_FEATURE_PARITY", "0") == "1"
            feature_registry_file = self.registry_path / "feature_registry.json"

            if not manifest.feature_set_id:
                msg = "feature_set_id is missing for serveable model; parity validation skipped"
                if strict_parity:
                    raise ValueError(
                        "feature_set_id is required for serveable models to ensure feature parity",
                    )
                logger.warning(msg)
            elif not feature_registry_file.exists():
                msg = (
                    "FeatureRegistry not found alongside ModelRegistry; "
                    "cannot validate feature parity"
                )
                if strict_parity:
                    raise ValueError(msg)
                logger.warning(msg)
            else:
                from ml.registry.feature_registry import FeatureRegistry

                freg = FeatureRegistry(self.registry_path)
                finfo = freg.get_feature_set(manifest.feature_set_id)
                if finfo is None:
                    msg = f"feature_set_id {manifest.feature_set_id} not found in FeatureRegistry"
                    if strict_parity:
                        raise ValueError(msg)
                    logger.warning(msg)
                else:
                    if finfo.manifest.schema_hash != manifest.feature_schema_hash:
                        msg = "feature_schema_hash mismatch between model manifest and feature manifest"
                        if strict_parity:
                            raise ValueError(msg)
                        logger.warning(msg)
                    # Backfill pipeline identity
                    if not manifest.pipeline_signature:
                        manifest.pipeline_signature = finfo.manifest.pipeline_signature
                    if not manifest.pipeline_version:
                        manifest.pipeline_version = finfo.manifest.pipeline_version

    def _maybe_auto_deploy(self, manifest: ModelManifest) -> None:
        """
        Auto-deploy when basic constraints are met.

        Parameters
        ----------
        manifest : ModelManifest
            Model manifest to evaluate for auto-deployment

        """
        # Basic validation
        is_valid = True
        errors: list[str] = []

        # Basic manifest validation
        if not manifest.feature_schema_hash:
            is_valid = False
            errors.append("Missing feature schema hash")

        # Role-specific validation
        if manifest.role == ModelRole.STUDENT:
            if manifest.data_requirements != DataRequirements.L1_ONLY:
                is_valid = False
                errors.append("Student must use L1-only data")
            if not manifest.parent_id:
                is_valid = False
                errors.append("Student must have parent_id")
            # Check latency constraint
            if "inference_latency_ms" in manifest.performance_metrics:
                if manifest.performance_metrics["inference_latency_ms"] > float(
                    self._policy.max_inference_latency_ms,
                ):
                    is_valid = False
                    errors.append(
                        f"Student inference must be under {self._policy.max_inference_latency_ms}ms",
                    )

        if is_valid:
            # Determine deployment target based on role
            if manifest.role == ModelRole.STUDENT:
                target = "ml_signal_actor"  # Students deploy to MLSignalActor
            elif manifest.role == ModelRole.INFERENCE:
                target = "ml_signal_actor"  # Direct inference also to MLSignalActor
            else:
                target = None  # Teachers don't deploy directly (offline only)

            if target:
                self.deploy_model(manifest.model_id, target)
                logger.info(f"Auto-deployed {manifest.model_id} to {target}")
        else:
            logger.warning("Auto-deploy skipped for %s: %s", manifest.model_id, errors)

    def load_model(self, model_id: str) -> object | None:
        """
        Load model from cache or disk.

        SECURITY: Only loads ONNX models to prevent code execution vulnerabilities.

        Parameters
        ----------
        model_id : str
            Model ID to load

        Returns
        -------
        object | None
            Loaded model object (ONNX InferenceSession) or None if not found

        """
        if self._use_legacy:
            return self._legacy_impl.load_model(model_id)
        if model_id not in self._models:
            return None
        return self._model_persistence.load_model(model_id, self._models[model_id])

    def get_artifact_path(self, model_id: str) -> Path | None:
        """
        Return the model artifact path if present and within registry root.

        Parameters
        ----------
        model_id : str
            Model ID

        Returns
        -------
        Path | None
            Artifact path or None if not found

        """
        if self._use_legacy:
            return self._legacy_impl.get_artifact_path(model_id)
        return self._model_persistence.get_artifact_path(model_id, self._models)

    def flush(self) -> None:
        """
        Flush any pending batch saves immediately.

        Call this before shutdown or when immediate persistence is needed.

        """
        if self._use_legacy:
            return self._legacy_impl.flush()
        self._model_persistence.flush()

    # =========================================================================
    # Deployment Methods (delegate to ModelDeploymentManager)
    # =========================================================================

    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deploy a model to a target.

        Parameters
        ----------
        model_id : str
            Model ID to deploy
        target : str
            Deployment target
        config : dict[str, Any] | None
            Deployment configuration

        Returns
        -------
        bool
            True if deployment successful

        """
        if self._use_legacy:
            return self._legacy_impl.deploy_model(model_id, target, config)
        return self._deployment_manager.deploy_model(model_id, target, config)

    def get_active_models(self) -> list[ModelInfo]:
        """Get all currently deployed models."""
        if self._use_legacy:
            return self._legacy_impl.get_active_models()
        return self._deployment_manager.get_active_models()

    def get_all_models(self) -> list[ModelInfo]:
        """Get all registered models."""
        if self._use_legacy:
            return self._legacy_impl.get_all_models()
        return self._deployment_manager.get_all_models()

    def list_compatible(
        self,
        schema_hash: str,
        role: ModelRole | None = None,
        architecture: str | None = None,
    ) -> list[ModelInfo]:
        """List models compatible with a given feature schema hash."""
        if self._use_legacy:
            return self._legacy_impl.list_compatible(schema_hash, role, architecture)
        return self._deployment_manager.list_compatible(schema_hash, role, architecture)

    def resolve_latest(
        self,
        role: ModelRole,
        architecture: str,
        schema_hash: str,
    ) -> ModelInfo | None:
        """Resolve the latest model by version matching criteria."""
        if self._use_legacy:
            return self._legacy_impl.resolve_latest(role, architecture, schema_hash)
        return self._deployment_manager.resolve_latest(role, architecture, schema_hash)

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get information about a specific model."""
        if self._use_legacy:
            return self._legacy_impl.get_model(model_id)
        return self._deployment_manager.get_model(model_id)

    def get_models_by_role(self, role: ModelRole) -> list[ModelInfo]:
        """Get all models with a specific role."""
        if self._use_legacy:
            return self._legacy_impl.get_models_by_role(role)
        return self._deployment_manager.get_models_by_role(role)

    def get_models_by_data_requirements(
        self,
        requirements: DataRequirements,
    ) -> list[ModelInfo]:
        """Get all models with specific data requirements."""
        if self._use_legacy:
            return self._legacy_impl.get_models_by_data_requirements(requirements)
        return self._deployment_manager.get_models_by_data_requirements(requirements)

    def get_model_lineage(self, model_id: str) -> list[ModelInfo]:
        """Get complete lineage of a model (parents and children)."""
        if self._use_legacy:
            return self._legacy_impl.get_model_lineage(model_id)
        return self._deployment_manager.get_model_lineage(model_id)

    def track_performance(
        self,
        model_id: str,
        metrics: dict[str, Any],
    ) -> None:
        """Track model performance metrics."""
        if self._use_legacy:
            return self._legacy_impl.track_performance(model_id, metrics)
        return self._deployment_manager.track_performance(model_id, metrics)

    def update_metadata(self, model_id: str, metadata: dict[str, Any]) -> None:
        """Update arbitrary metadata for a registered model."""
        if self._use_legacy:
            return self._legacy_impl.update_metadata(model_id, metadata)
        return self._deployment_manager.update_metadata(model_id, metadata)

    def get_performance_history(
        self,
        model_id: str,
    ) -> list[dict[str, Any]]:
        """Get performance history for a model."""
        if self._use_legacy:
            return self._legacy_impl.get_performance_history(model_id)
        return self._deployment_manager.get_performance_history(model_id)

    def rollback(
        self,
        target: str,
        to_model_id: str,
    ) -> bool:
        """Rollback to a previous model version."""
        if self._use_legacy:
            return self._legacy_impl.rollback(target, to_model_id)
        return self._deployment_manager.rollback(target, to_model_id)

    def retire_model(self, model_id: str) -> bool:
        """Retire a model from production."""
        if self._use_legacy:
            return self._legacy_impl.retire_model(model_id)
        return self._deployment_manager.retire_model(model_id)

    def hot_reload_model(
        self,
        target: str,
        new_model_id: str,
    ) -> bool:
        """Hot reload a deployment with a new model."""
        if self._use_legacy:
            return self._legacy_impl.hot_reload_model(target, new_model_id)
        return self._deployment_manager.hot_reload_model(target, new_model_id)

    # =========================================================================
    # Quality Validation Methods (delegate to ModelQualityValidator)
    # =========================================================================

    def validate_model_quality(
        self,
        model_id: str,
        gates: list[QualityGate],
    ) -> ValidationResult:
        """
        Validate a registered model against quality gates.

        Parameters
        ----------
        model_id : str
            Model ID to validate
        gates : list[QualityGate]
            Quality gates to check

        Returns
        -------
        ValidationResult
            Validation results

        """
        if self._use_legacy:
            return self._legacy_impl.validate_model_quality(model_id, gates)

        if model_id not in self._models:
            result = ValidationResult(model_id=model_id)
            result.overall_pass = False
            result.gate_results["model_existence"] = {
                "passed": False,
                "reason": "model_not_found",
            }
            return result

        model_info = self._models[model_id]
        return self._quality_validator.validate_quality_gates(
            model_id,
            model_info.manifest.performance_metrics,
            gates,
        )

    # =========================================================================
    # A/B Testing Methods (delegate to ABTestingManager)
    # =========================================================================

    def configure_ab_test(
        self,
        models: list[str],
        split_ratio: float,
        duration_hours: int,
        target: str,
    ) -> dict[str, Any] | None:
        """Configure A/B test between models."""
        if self._use_legacy:
            return self._legacy_impl.configure_ab_test(
                models,
                split_ratio,
                duration_hours,
                target,
            )
        return self._ab_testing_manager.configure_ab_test(
            models,
            split_ratio,
            duration_hours,
            target,
        )

    def compare_models(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """Compare performance between models."""
        if self._use_legacy:
            return self._legacy_impl.compare_models(model_ids, metric)
        return self._ab_testing_manager.compare_models(model_ids, metric)

    def compare_models_statistically(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """Perform statistical comparison using Welch's t-test."""
        if self._use_legacy:
            return self._legacy_impl.compare_models_statistically(model_ids, metric)
        return self._ab_testing_manager.compare_models_statistically(model_ids, metric)

    def run_ab_test(
        self,
        model_a_id: str,
        model_b_id: str,
        split_ratio: float,
        duration_hours: float,
        target: str,
    ) -> str:
        """Start an A/B test between two models."""
        if self._use_legacy:
            return self._legacy_impl.run_ab_test(
                model_a_id,
                model_b_id,
                split_ratio,
                duration_hours,
                target,
            )
        return self._ab_testing_manager.run_ab_test(
            model_a_id,
            model_b_id,
            split_ratio,
            duration_hours,
            target,
        )

    def track_ab_test_metric(
        self,
        test_id: str,
        model_id: str,
        metric_value: float,
    ) -> None:
        """Track metric for A/B test."""
        if self._use_legacy:
            return self._legacy_impl.track_ab_test_metric(test_id, model_id, metric_value)
        return self._ab_testing_manager.track_ab_test_metric(test_id, model_id, metric_value)

    def analyze_ab_test(self, test_id: str) -> dict[str, Any] | None:
        """Analyze A/B test results."""
        if self._use_legacy:
            return self._legacy_impl.analyze_ab_test(test_id)
        return self._ab_testing_manager.analyze_ab_test(test_id)

    # =========================================================================
    # Canary Deployment Methods (delegate to CanaryDeploymentManager)
    # =========================================================================

    def start_canary_deployment(
        self,
        model_id: str,
        target: str,
        config: CanaryConfig,
        baseline_model_id: str | None = None,
    ) -> str:
        """Start a canary deployment for a model."""
        if self._use_legacy:
            return self._legacy_impl.start_canary_deployment(
                model_id,
                target,
                config,
                baseline_model_id,
            )
        return self._canary_deployment_manager.start_canary_deployment(
            model_id,
            target,
            config,
            baseline_model_id,
        )

    def get_canary_deployment(self, deployment_id: str) -> CanaryDeployment | None:
        """Get canary deployment by ID."""
        if self._use_legacy:
            return self._legacy_impl.get_canary_deployment(deployment_id)
        return self._canary_deployment_manager.get_canary_deployment(deployment_id)

    def update_canary_metrics(
        self,
        deployment_id: str,
        metric_value: float,
        latency_ms: float | None = None,
        error_occurred: bool = False,
    ) -> None:
        """Update metrics for a canary deployment."""
        if self._use_legacy:
            return self._legacy_impl.update_canary_metrics(
                deployment_id,
                metric_value,
                latency_ms,
                error_occurred,
            )
        return self._canary_deployment_manager.update_canary_metrics(
            deployment_id,
            metric_value,
            latency_ms,
            error_occurred,
        )

    def evaluate_canary(self, deployment_id: str) -> tuple[bool, str]:
        """Evaluate if canary should be promoted."""
        if self._use_legacy:
            return self._legacy_impl.evaluate_canary(deployment_id)
        return self._canary_deployment_manager.evaluate_canary(deployment_id)

    def evaluate_canary_for_rollback(self, deployment_id: str) -> tuple[bool, str]:
        """Evaluate if canary should be rolled back."""
        if self._use_legacy:
            return self._legacy_impl.evaluate_canary_for_rollback(deployment_id)
        return self._canary_deployment_manager.evaluate_canary_for_rollback(deployment_id)

    def auto_promote_canary(self, deployment_id: str) -> bool:
        """Automatically promote a canary to full production."""
        if self._use_legacy:
            return self._legacy_impl.auto_promote_canary(deployment_id)
        return self._canary_deployment_manager.auto_promote_canary(deployment_id)

    def start_gradual_rollout(
        self,
        current_model_id: str,
        new_model_id: str,
        target: str,
        stages: list[float],
        stage_duration_minutes: int,
    ) -> str:
        """Start gradual rollout of a new model."""
        if self._use_legacy:
            return self._legacy_impl.start_gradual_rollout(
                current_model_id,
                new_model_id,
                target,
                stages,
                stage_duration_minutes,
            )
        return self._canary_deployment_manager.start_gradual_rollout(
            current_model_id,
            new_model_id,
            target,
            stages,
            stage_duration_minutes,
        )

    def get_rollout_status(self, rollout_id: str) -> dict[str, Any] | None:
        """Get rollout status."""
        if self._use_legacy:
            return self._legacy_impl.get_rollout_status(rollout_id)
        return self._canary_deployment_manager.get_rollout_status(rollout_id)

    def advance_rollout_stage(self, rollout_id: str) -> bool:
        """Advance to next rollout stage."""
        if self._use_legacy:
            return self._legacy_impl.advance_rollout_stage(rollout_id)
        return self._canary_deployment_manager.advance_rollout_stage(rollout_id)

    # =========================================================================
    # Health and Monitoring (AbstractRegistry requirement)
    # =========================================================================

    def _health_snapshot(self) -> tuple[int, float | None]:
        """Get health snapshot for AbstractRegistry."""
        if self._use_legacy:
            return self._legacy_impl._health_snapshot()
        try:
            count = len(self._models)
        except AttributeError:
            return 0, None
        if count == 0:
            return 0, None
        try:
            last_modified = max(mi.manifest.last_modified for mi in self._models.values())
        except ValueError:
            last_modified = None
        return count, last_modified

    # =========================================================================
    # Additional Methods for Backward Compatibility
    # =========================================================================

    def __del__(self) -> None:
        """Ensure pending saves are flushed on cleanup."""
        if self._use_legacy and hasattr(self, "_legacy_impl"):
            # Legacy handles its own cleanup
            return
        try:
            self.flush()
        except Exception as exc:  # Best effort on cleanup
            logger.debug("ModelRegistry cleanup flush failed", exc_info=exc)

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
            f"'{type(self).__name__}' object has no attribute '{name}'",
        )
