#!/usr/bin/env python3

"""
ModelRegistryFacade - Facade implementation wiring all ModelRegistry components.

This facade provides identical API to the legacy ModelRegistry while delegating
to focused, decomposed components internally. It follows the established
facade pattern used in TFTDatasetBuilder and MLPipelineOrchestrator decompositions.

Feature flag: ML_USE_LEGACY_MODEL_REGISTRY
- When "1": Use legacy ModelRegistry
- When "0" or unset: Use this facade (default)

Thread-safety: All operations are thread-safe via component locks.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from ml.config.constants import SUFFIX_ONNX
from ml.config.registry import RegistryPolicyConfig
from ml.config.runtime import OnnxRuntimeConfig
from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.common.ab_testing import ABTestingComponent
from ml.registry.common.deployment_manager import DeploymentManagerComponent
from ml.registry.common.model_persistence import ModelPersistenceComponent
from ml.registry.common.version_manager import VersionManagerComponent
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment
from ml.registry.dataclasses import QualityGate
from ml.registry.dataclasses import ValidationResult
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig


logger = logging.getLogger(__name__)


def use_legacy_model_registry() -> bool:
    """
    Check if legacy ModelRegistry should be used.

    Returns
    -------
    bool
        True if ML_USE_LEGACY_MODEL_REGISTRY=1, False otherwise.

    Example
    -------
    >>> import os
    >>> os.environ["ML_USE_LEGACY_MODEL_REGISTRY"] = "1"
    >>> use_legacy_model_registry()
    True
    """
    return os.getenv("ML_USE_LEGACY_MODEL_REGISTRY", "0") == "1"


# Re-export security-related runtime toggles for tests to patch.
try:
    from ml.common.security import HAS_ONNX as HAS_ONNX
    from ml.common.security import check_ml_dependencies as check_ml_dependencies
    from ml.common.security import ort as ort
except Exception:  # pragma: no cover
    HAS_ONNX = False
    ort = None

    def check_ml_dependencies(_deps: list[str]) -> None:
        raise RuntimeError("Dependency check unavailable")


class ModelRegistryFacade:
    """
    Facade wiring all ModelRegistry components.

    Provides identical API to legacy ModelRegistry while delegating
    to focused components internally. This enables progressive decomposition
    while maintaining backward compatibility.

    The facade composes:
    - ModelPersistenceComponent: JSON/PostgreSQL storage, serialization, SHA-256 verification
    - DeploymentManagerComponent: Deployment, canary, rollout operations
    - ABTestingComponent: A/B test configuration, statistical comparison
    - VersionManagerComponent: Auto-versioning, compatibility, lineage

    Attributes
    ----------
    registry_path : Path
        Directory path for registry storage.
    cache_size : int
        Maximum number of models to cache in memory.
    batch_save_interval : float
        Seconds to wait before flushing batch saves.

    Thread Safety
    -------------
    All operations are thread-safe via the underlying component locks.

    Example
    -------
    >>> from pathlib import Path
    >>> from ml.registry.model_registry_facade import ModelRegistryFacade
    >>> registry = ModelRegistryFacade(registry_path=Path("/tmp/registry"))
    >>> model_id = registry.register_model(model_path, manifest)
    >>> registry.deploy_model(model_id, "ml_signal_actor")
    True
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
        Initialize ModelRegistryFacade with all components.

        Parameters
        ----------
        registry_path : Path
            Directory path for registry storage.
        cache_size : int
            Maximum number of models to cache in memory (default 10).
        batch_save_interval : float
            Seconds to wait before flushing batch saves (default 0.1s).
        persistence_config : PersistenceConfig | None
            Persistence configuration. If None, defaults to JSON backend.
        policy_config : RegistryPolicyConfig | None
            Registry policy configuration.
        onnx_runtime_config : OnnxRuntimeConfig | None
            ONNX runtime configuration for model loading.
        """
        # Store config attributes
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.cache_size = cache_size
        self.batch_save_interval = batch_save_interval
        self._policy = policy_config or RegistryPolicyConfig()
        self._onnx_rt = onnx_runtime_config or OnnxRuntimeConfig()

        # Store absolute path for security validation
        self._registry_root = self.registry_path.resolve()

        # Setup persistence config
        if persistence_config is None:
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )

        # Initialize components
        self._persistence = ModelPersistenceComponent(
            persistence_config=persistence_config,
            registry_path=registry_path,
            batch_save_interval=batch_save_interval,
        )
        self._persistence.load_registry()

        self._deployment = DeploymentManagerComponent(self._persistence)
        self._ab_testing = ABTestingComponent(self._persistence, self._policy)
        self._version = VersionManagerComponent(self._persistence)

        # Model cache for performance (hot-path)
        self._model_cache: dict[str, Any] = {}
        self._cache_access_times: dict[str, float] = {}

        logger.info(
            "Initialized ModelRegistryFacade at %s with backend=%s, cache_size=%s",
            registry_path,
            self.backend.value,
            cache_size,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def backend(self) -> BackendType:
        """Get the persistence backend type."""
        return self._persistence.backend

    @property
    def _models(self) -> dict[str, ModelInfo]:
        """
        Access to models dict for backward compatibility.

        Note: Prefer using public methods; this is for internal/test use.
        """
        return self._persistence.models

    @property
    def _ab_tests(self) -> dict[str, dict[str, Any]]:
        """Access to A/B tests for backward compatibility."""
        return self._persistence.ab_tests

    @property
    def _deployments(self) -> dict[str, list[str]]:
        """Access to deployments for backward compatibility."""
        return self._persistence.deployments

    @property
    def _lock(self) -> Any:
        """Access to lock for backward compatibility."""
        return self._persistence._lock

    # =========================================================================
    # Core Model Operations
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
            Path to the model file.
        manifest : ModelManifest
            Self-describing model manifest.
        auto_deploy : bool
            Whether to automatically deploy if validation passes.
        quality_gates : list[QualityGate] | None
            Quality gates to validate before registration.
        enforce_quality : bool
            If True, raise error on quality gate failure.

        Returns
        -------
        str
            Unique model ID.

        Raises
        ------
        ValueError
            If validation fails (invalid path, unsupported format, etc.).
        """
        with self._persistence._lock:
            # Validate inputs
            self._validate_registration_inputs(model_path, manifest)

            # Calculate SHA-256 digest for ONNX files
            if model_path.suffix == SUFFIX_ONNX:
                try:
                    artifact_digest = self._persistence.calculate_file_sha256(model_path)
                    manifest.artifact_sha256_digest = artifact_digest
                    logger.info(
                        "Calculated SHA-256 digest for model artifact: %s...",
                        artifact_digest[:16],
                    )
                except (OSError, FileNotFoundError) as e:
                    logger.error("Failed to calculate artifact digest: %s", e, exc_info=True)
                    raise ValueError(
                        f"Cannot calculate SHA-256 digest for model artifact: {e}",
                    ) from e

            # Generate model_id if not set
            if not manifest.model_id:
                manifest.model_id = self._generate_model_id()

            # Set timestamps
            manifest.created_at = time.time()
            manifest.last_modified = time.time()

            # Auto-version
            self._version.auto_version_manifest(manifest)

            # Quality validation
            quality_validation_result = self._apply_quality_gates(
                manifest, quality_gates, enforce_quality
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

            # Store quality validation result
            if quality_validation_result:
                model_info.metadata["quality_validation"] = {
                    "passed": quality_validation_result.overall_pass,
                    "gates_passed": quality_validation_result.gates_passed,
                    "gates_failed": quality_validation_result.gates_failed,
                    "timestamp": quality_validation_result.timestamp,
                }

            # Handle parent-child relationships
            if manifest.parent_id and manifest.parent_id in self._persistence.models:
                parent = self._persistence.models[manifest.parent_id]
                if manifest.model_id not in parent.manifest.children_ids:
                    parent.manifest.children_ids.append(manifest.model_id)
                    parent.manifest.last_modified = time.time()

            # Store model
            self._persistence.set_model(manifest.model_id, model_info)

            # Persist
            if self.backend == BackendType.POSTGRES:
                self._persistence.save_model_to_db(model_info)
            else:
                self._persistence.save_registry()

            logger.info(
                "Registered %s model %s (version %s) at %s",
                manifest.role.value,
                manifest.model_id,
                manifest.version,
                model_path,
            )

            # Auto-deploy if requested
            if auto_deploy:
                self._maybe_auto_deploy(manifest)

            return manifest.model_id

    def get_model(self, model_id: str) -> ModelInfo | None:
        """
        Get information about a specific model.

        Parameters
        ----------
        model_id : str
            Model ID to retrieve.

        Returns
        -------
        ModelInfo | None
            Model information if found.
        """
        with self._persistence._lock:
            return self._persistence.get_model(model_id)

    def get_all_models(self) -> list[ModelInfo]:
        """
        Get all registered models.

        Returns
        -------
        list[ModelInfo]
            List of all model information.
        """
        with self._persistence._lock:
            return list(self._persistence.models.values())

    def get_active_models(self) -> list[ModelInfo]:
        """
        Get all currently deployed models.

        Returns
        -------
        list[ModelInfo]
            List of active model information.
        """
        with self._persistence._lock:
            return [
                m for m in self._persistence.models.values()
                if m.deployment_status == DeploymentStatus.ACTIVE
            ]

    def get_models_by_role(self, role: ModelRole) -> list[ModelInfo]:
        """
        Get all models with a specific role.

        Parameters
        ----------
        role : ModelRole
            The role to filter by.

        Returns
        -------
        list[ModelInfo]
            Models with the specified role.
        """
        with self._persistence._lock:
            return [
                m for m in self._persistence.models.values()
                if m.manifest.role == role
            ]

    def get_models_by_data_requirements(
        self,
        requirements: DataRequirements,
    ) -> list[ModelInfo]:
        """
        Get all models with specific data requirements.

        Parameters
        ----------
        requirements : DataRequirements
            The data requirements to filter by.

        Returns
        -------
        list[ModelInfo]
            Models with the specified data requirements.
        """
        with self._persistence._lock:
            return [
                m for m in self._persistence.models.values()
                if m.manifest.data_requirements == requirements
            ]

    def load_model(self, model_id: str) -> object | None:
        """
        Load model from cache or disk.

        SECURITY: Only loads ONNX models to prevent code execution vulnerabilities.

        Parameters
        ----------
        model_id : str
            Model ID to load.

        Returns
        -------
        object | None
            Loaded model object (ONNX InferenceSession) or None if not found.

        Raises
        ------
        ValueError
            If artifact integrity verification fails.
        """
        with self._persistence._lock:
            # Check cache first
            if model_id in self._model_cache:
                self._cache_access_times[model_id] = time.time()
                cached_model: object = self._model_cache[model_id]
                return cached_model

            # Get model info
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                return None

            model_path = model_info.model_path

            # Validate path security
            if not self._validate_model_path(model_path):
                logger.error("Security: Invalid model path detected: %s", model_path)
                return None

            if not model_path.exists():
                logger.error("Model file not found: %s", model_path)
                return None

            # Only support ONNX format
            try:
                if model_path.suffix == SUFFIX_ONNX:
                    # Verify artifact integrity before loading
                    expected_digest = model_info.manifest.artifact_sha256_digest
                    self._persistence.verify_artifact_integrity(model_path, expected_digest)

                    if not HAS_ONNX:
                        check_ml_dependencies(["onnxruntime"])

                    if ort is None:
                        logger.error("ONNX runtime not available")
                        return None

                    # Create optimized session
                    from ml.config.runtime import to_session_options

                    session_options, providers = to_session_options(self._onnx_rt)
                    model: object = ort.InferenceSession(
                        str(model_path),
                        sess_options=session_options,
                        providers=providers,
                    )
                else:
                    logger.info(
                        "Non-ONNX artifact requested for load; returning None for security"
                    )
                    return None

                # Update cache with LRU eviction
                if len(self._model_cache) >= self.cache_size:
                    lru_id = min(
                        self._cache_access_times.items(),
                        key=lambda x: x[1],
                    )[0]
                    del self._model_cache[lru_id]
                    del self._cache_access_times[lru_id]

                self._model_cache[model_id] = model
                self._cache_access_times[model_id] = time.time()

                return model

            except ValueError:
                # Re-raise integrity failures
                raise
            except Exception:
                logger.error("Failed to load model %s", model_id, exc_info=True)
                return None

    def track_performance(
        self,
        model_id: str,
        metrics: dict[str, Any],
    ) -> None:
        """
        Track model performance metrics.

        Parameters
        ----------
        model_id : str
            Model ID.
        metrics : dict[str, Any]
            Performance metrics.
        """
        with self._persistence._lock:
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                logger.error("Model %s not found in registry", model_id)
                return

            if "timestamp" not in metrics:
                metrics["timestamp"] = time.time()

            model_info.performance_history.append(metrics)
            model_info.manifest.last_modified = time.time()

            self._persistence.save_registry()
            logger.debug("Tracked performance for model %s: %s", model_id, metrics)

    def update_metadata(self, model_id: str, metadata: dict[str, Any]) -> None:
        """
        Update arbitrary metadata for a registered model.

        Parameters
        ----------
        model_id : str
            Model ID whose metadata to update.
        metadata : dict[str, Any]
            Key/value pairs to merge into the model metadata.
        """
        with self._persistence._lock:
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                logger.error("Model %s not found in registry", model_id)
                return

            try:
                current = model_info.metadata
                if not isinstance(current, dict):
                    current = {}
                    model_info.metadata = current
                current.update(metadata)
                model_info.manifest.last_modified = time.time()
                self._persistence.save_registry()
                logger.debug("Updated metadata for model %s", model_id)
            except Exception as exc:
                logger.error("Failed updating metadata for %s: %s", model_id, exc, exc_info=True)

    def get_performance_history(
        self,
        model_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get performance history for a model.

        Parameters
        ----------
        model_id : str
            Model ID.

        Returns
        -------
        list[dict[str, Any]]
            Performance history.
        """
        with self._persistence._lock:
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                return []
            return model_info.performance_history.copy()

    def get_artifact_path(self, model_id: str) -> Path | None:
        """
        Return the model artifact path if present and within registry root.

        Parameters
        ----------
        model_id : str
            Model ID.

        Returns
        -------
        Path | None
            Model path if valid and exists, None otherwise.
        """
        with self._persistence._lock:
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                return None
            model_path = model_info.model_path
            if not self._validate_model_path(model_path):
                logger.error("Security: Invalid model path detected: %s", model_path)
                return None
            return model_path if model_path.exists() else None

    # =========================================================================
    # Deployment Operations (delegate to DeploymentManagerComponent)
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
            Model ID to deploy.
        target : str
            Deployment target.
        config : dict[str, Any] | None
            Deployment configuration.

        Returns
        -------
        bool
            True if deployment successful.
        """
        return self._deployment.deploy_model(model_id, target, config)

    def rollback(
        self,
        target: str,
        to_model_id: str,
    ) -> bool:
        """
        Rollback to a previous model version.

        Parameters
        ----------
        target : str
            Deployment target.
        to_model_id : str
            Model ID to rollback to.

        Returns
        -------
        bool
            True if rollback successful.
        """
        return self._deployment.rollback(target, to_model_id)

    def retire_model(self, model_id: str) -> bool:
        """
        Retire a model from production.

        Parameters
        ----------
        model_id : str
            Model ID to retire.

        Returns
        -------
        bool
            True if retirement successful.
        """
        return self._deployment.retire_model(model_id)

    def hot_reload_model(
        self,
        target: str,
        new_model_id: str,
    ) -> bool:
        """
        Hot reload a deployment with a new model.

        Parameters
        ----------
        target : str
            Deployment target.
        new_model_id : str
            New model to deploy.

        Returns
        -------
        bool
            True if successful.
        """
        return self._deployment.hot_reload_model(target, new_model_id)

    # =========================================================================
    # Canary Deployment (delegate to DeploymentManagerComponent)
    # =========================================================================

    def start_canary_deployment(
        self,
        model_id: str,
        target: str,
        config: CanaryConfig,
        baseline_model_id: str | None = None,
    ) -> str:
        """
        Start a canary deployment for a model.

        Parameters
        ----------
        model_id : str
            Model to deploy as canary.
        target : str
            Deployment target.
        config : CanaryConfig
            Canary configuration.
        baseline_model_id : str | None
            Baseline model for comparison.

        Returns
        -------
        str
            Canary deployment ID.
        """
        return self._deployment.start_canary_deployment(
            model_id, target, config, baseline_model_id
        )

    def get_canary_deployment(self, deployment_id: str) -> CanaryDeployment | None:
        """
        Get canary deployment by ID.

        Parameters
        ----------
        deployment_id : str
            The canary deployment ID.

        Returns
        -------
        CanaryDeployment | None
            The canary deployment if found.
        """
        return self._deployment.get_canary_deployment(deployment_id)

    def update_canary_metrics(
        self,
        deployment_id: str,
        metric_value: float,
        latency_ms: float | None = None,
        error_occurred: bool = False,
    ) -> None:
        """
        Update metrics for a canary deployment.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID.
        metric_value : float
            Value of the success metric.
        latency_ms : float | None
            Response latency.
        error_occurred : bool
            Whether an error occurred.
        """
        self._deployment.update_canary_metrics(
            deployment_id, metric_value, latency_ms, error_occurred
        )

    def evaluate_canary(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be promoted.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID.

        Returns
        -------
        tuple[bool, str]
            (should_promote, reason).
        """
        return self._deployment.evaluate_canary(deployment_id)

    def evaluate_canary_for_rollback(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be rolled back.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID.

        Returns
        -------
        tuple[bool, str]
            (should_rollback, reason).
        """
        return self._deployment.evaluate_canary_for_rollback(deployment_id)

    def auto_promote_canary(self, deployment_id: str) -> bool:
        """
        Automatically promote a canary to full production.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID.

        Returns
        -------
        bool
            True if promotion successful.
        """
        return self._deployment.auto_promote_canary(deployment_id)

    # =========================================================================
    # Gradual Rollout (delegate to DeploymentManagerComponent)
    # =========================================================================

    def start_gradual_rollout(
        self,
        current_model_id: str,
        new_model_id: str,
        target: str,
        stages: list[float],
        stage_duration_minutes: int,
    ) -> str:
        """
        Start gradual rollout of a new model.

        Parameters
        ----------
        current_model_id : str
            Currently deployed model.
        new_model_id : str
            New model to roll out.
        target : str
            Deployment target.
        stages : list[float]
            Traffic percentages for each stage.
        stage_duration_minutes : int
            Duration of each stage.

        Returns
        -------
        str
            Rollout ID.
        """
        return self._deployment.start_gradual_rollout(
            current_model_id, new_model_id, target, stages, stage_duration_minutes
        )

    def get_rollout_status(self, rollout_id: str) -> dict[str, Any] | None:
        """
        Get rollout status.

        Parameters
        ----------
        rollout_id : str
            The rollout ID.

        Returns
        -------
        dict[str, Any] | None
            Rollout status.
        """
        return self._deployment.get_rollout_status(rollout_id)

    def advance_rollout_stage(self, rollout_id: str) -> bool:
        """
        Advance to next rollout stage.

        Parameters
        ----------
        rollout_id : str
            The rollout ID.

        Returns
        -------
        bool
            True if advanced.
        """
        return self._deployment.advance_rollout_stage(rollout_id)

    # =========================================================================
    # A/B Testing (delegate to ABTestingComponent)
    # =========================================================================

    def configure_ab_test(
        self,
        models: list[str],
        split_ratio: float,
        duration_hours: int,
        target: str,
    ) -> dict[str, Any] | None:
        """
        Configure A/B test between models.

        Parameters
        ----------
        models : list[str]
            List of model IDs to test (expects 2 models).
        split_ratio : float
            Traffic split ratio for first model.
        duration_hours : int
            Test duration in hours.
        target : str
            Deployment target.

        Returns
        -------
        dict[str, Any] | None
            A/B test configuration.
        """
        return self._ab_testing.configure_ab_test(models, split_ratio, duration_hours, target)

    def compare_models(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """
        Compare performance between models.

        Parameters
        ----------
        model_ids : list[str]
            List of model IDs to compare.
        metric : str
            Metric to compare on.

        Returns
        -------
        dict[str, Any] | None
            Comparison results.
        """
        return self._ab_testing.compare_models(model_ids, metric)

    def compare_models_statistically(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """
        Perform statistical comparison between models.

        Parameters
        ----------
        model_ids : list[str]
            Model IDs to compare (exactly 2).
        metric : str
            Metric to compare.

        Returns
        -------
        dict[str, Any] | None
            Statistical comparison results.
        """
        return self._ab_testing.compare_models_statistically(model_ids, metric)

    def run_ab_test(
        self,
        model_a_id: str,
        model_b_id: str,
        split_ratio: float,
        duration_hours: float,
        target: str,
    ) -> str:
        """
        Start an A/B test between two models.

        Parameters
        ----------
        model_a_id : str
            Control model ID.
        model_b_id : str
            Treatment model ID.
        split_ratio : float
            Traffic split for control.
        duration_hours : float
            Test duration.
        target : str
            Deployment target.

        Returns
        -------
        str
            A/B test ID.
        """
        return self._ab_testing.run_ab_test(
            model_a_id, model_b_id, split_ratio, duration_hours, target
        )

    def track_ab_test_metric(
        self,
        test_id: str,
        model_id: str,
        metric_value: float,
    ) -> None:
        """
        Track metric for A/B test.

        Parameters
        ----------
        test_id : str
            A/B test ID.
        model_id : str
            Model ID.
        metric_value : float
            Metric value.
        """
        self._ab_testing.track_ab_test_metric(test_id, model_id, metric_value)

    def analyze_ab_test(self, test_id: str) -> dict[str, Any] | None:
        """
        Analyze A/B test results.

        Parameters
        ----------
        test_id : str
            A/B test ID.

        Returns
        -------
        dict[str, Any] | None
            Analysis results.
        """
        return self._ab_testing.analyze_ab_test(test_id)

    # =========================================================================
    # Version Management (delegate to VersionManagerComponent)
    # =========================================================================

    def list_compatible(
        self,
        schema_hash: str,
        role: ModelRole | None = None,
        architecture: str | None = None,
    ) -> list[ModelInfo]:
        """
        List models compatible with a given feature schema hash.

        Parameters
        ----------
        schema_hash : str
            Feature schema hash to match.
        role : ModelRole | None
            Optional role filter.
        architecture : str | None
            Optional architecture filter.

        Returns
        -------
        list[ModelInfo]
            Compatible models.
        """
        return self._version.list_compatible(schema_hash, role, architecture)

    def resolve_latest(
        self,
        role: ModelRole,
        architecture: str,
        schema_hash: str,
    ) -> ModelInfo | None:
        """
        Resolve the latest model by version matching criteria.

        Parameters
        ----------
        role : ModelRole
            Required model role.
        architecture : str
            Required model architecture.
        schema_hash : str
            Required feature schema hash.

        Returns
        -------
        ModelInfo | None
            The latest matching model.
        """
        return self._version.resolve_latest(role, architecture, schema_hash)

    def get_model_lineage(self, model_id: str) -> list[ModelInfo]:
        """
        Get complete lineage of a model (parents and children).

        Parameters
        ----------
        model_id : str
            Model ID to get lineage for.

        Returns
        -------
        list[ModelInfo]
            Ordered lineage list.
        """
        return self._version.get_model_lineage(model_id)

    # =========================================================================
    # Quality Validation
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
            Model ID to validate.
        gates : list[QualityGate]
            Quality gates to check.

        Returns
        -------
        ValidationResult
            Validation results.
        """
        with self._persistence._lock:
            model_info = self._persistence.get_model(model_id)
            if model_info is None:
                result = ValidationResult(model_id=model_id)
                result.overall_pass = False
                result.gate_results["model_existence"] = {
                    "passed": False,
                    "reason": "model_not_found",
                }
                return result

            return self._validate_quality_gates(
                model_id,
                model_info.manifest.performance_metrics,
                gates,
            )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def flush(self) -> None:
        """
        Flush any pending batch saves immediately.

        Call this before shutdown or when immediate persistence is needed.
        """
        self._persistence.flush()

    def __del__(self) -> None:
        """Ensure pending saves are flushed on cleanup."""
        try:
            self.flush()
        except Exception as exc:
            logger.debug("ModelRegistryFacade cleanup flush failed: %s", exc)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _generate_model_id(self) -> str:
        """Generate unique model ID."""
        timestamp = int(time.time() * 1000000)
        return f"model_{timestamp}"

    def _validate_model_path(self, path: Path) -> bool:
        """
        Validate model path is within registry bounds.

        Prevents path traversal attacks.
        """
        try:
            resolved = path.resolve()
            return str(resolved).startswith(str(self._registry_root))
        except (ValueError, RuntimeError):
            return False

    def _validate_registration_inputs(self, model_path: Path, manifest: ModelManifest) -> None:
        """Validate model path, format, security, and parity constraints."""
        # Security: Validate model format for serving
        if model_path.suffix != SUFFIX_ONNX:
            if getattr(manifest, "serveable", True):
                raise ValueError(
                    f"Only ONNX models are supported for serveable models. Got: {model_path.suffix}."
                )

        # Security: Validate path is safe
        if not self._validate_model_path(model_path):
            raise ValueError(f"Security: Invalid model path: {model_path}")

        # Validate feature schema linkage
        if not manifest.feature_schema_hash:
            raise ValueError("feature_schema_hash is required for all models")

        # Feature parity validation
        if getattr(manifest, "serveable", True):
            strict_parity = os.getenv("ML_STRICT_FEATURE_PARITY", "0") == "1"
            feature_registry_file = self.registry_path / "feature_registry.json"

            if not manifest.feature_set_id:
                msg = "feature_set_id is missing for serveable model"
                if strict_parity:
                    raise ValueError(
                        "feature_set_id is required for serveable models to ensure feature parity"
                    )
                logger.warning(msg)
            elif not feature_registry_file.exists():
                msg = "FeatureRegistry not found alongside ModelRegistry"
                if strict_parity:
                    raise ValueError(msg)
                logger.warning(msg)
            else:
                from ml.registry.feature_registry import FeatureRegistry

                freg = FeatureRegistry(self.registry_path)
                finfo = freg.get_feature_set(manifest.feature_set_id)
                if finfo is None:
                    msg = f"feature_set_id {manifest.feature_set_id} not found"
                    if strict_parity:
                        raise ValueError(msg)
                    logger.warning(msg)
                else:
                    if finfo.manifest.schema_hash != manifest.feature_schema_hash:
                        msg = "feature_schema_hash mismatch"
                        if strict_parity:
                            raise ValueError(msg)
                        logger.warning(msg)
                    if not manifest.pipeline_signature:
                        manifest.pipeline_signature = finfo.manifest.pipeline_signature
                    if not manifest.pipeline_version:
                        manifest.pipeline_version = finfo.manifest.pipeline_version

    def _apply_quality_gates(
        self,
        manifest: ModelManifest,
        quality_gates: list[QualityGate] | None,
        enforce_quality: bool,
    ) -> ValidationResult | None:
        """Run quality gates when provided; optionally enforce."""
        if not quality_gates:
            return None

        result = self._validate_quality_gates(
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
                f"Quality gates not met for model {manifest.model_id}. Failed gates: {failed_gates}"
            )
        return result

    def _validate_quality_gates(
        self,
        model_id: str,
        metrics: dict[str, float],
        gates: list[QualityGate],
    ) -> ValidationResult:
        """Validate model metrics against quality gates."""
        result = ValidationResult(model_id=model_id)

        for gate in gates:
            gate_result = self._evaluate_gate(gate, metrics.get(gate.metric_name))

            if gate_result["passed"]:
                result.gates_passed += 1
            else:
                result.gates_failed += 1
                if gate.required:
                    result.overall_pass = False

            result.gate_results[gate.metric_name] = gate_result

        return result

    def _evaluate_gate(
        self,
        gate: QualityGate,
        actual_value: float | None,
    ) -> dict[str, Any]:
        """Evaluate a single quality gate."""
        if actual_value is None:
            return {
                "threshold": gate.threshold,
                "actual": None,
                "passed": False,
                "required": gate.required,
                "reason": "metric_not_found",
            }

        passed = False
        if gate.comparison == "gte":
            passed = actual_value >= gate.threshold
        elif gate.comparison == "lte":
            passed = actual_value <= gate.threshold
        elif gate.comparison == "gt":
            passed = actual_value > gate.threshold
        elif gate.comparison == "lt":
            passed = actual_value < gate.threshold
        elif gate.comparison == "eq":
            passed = abs(actual_value - gate.threshold) < 1e-10

        return {
            "threshold": gate.threshold,
            "actual": actual_value,
            "passed": passed,
            "required": gate.required,
            "comparison": gate.comparison,
            "margin": (
                actual_value - gate.threshold
                if gate.comparison in ["gte", "gt"]
                else gate.threshold - actual_value
            ),
        }

    def _maybe_auto_deploy(self, manifest: ModelManifest) -> None:
        """Auto-deploy when basic constraints are met."""
        is_valid = True
        errors: list[str] = []

        if not manifest.feature_schema_hash:
            is_valid = False
            errors.append("Missing feature schema hash")

        if manifest.role == ModelRole.STUDENT:
            if manifest.data_requirements != DataRequirements.L1_ONLY:
                is_valid = False
                errors.append("Student must use L1-only data")
            if not manifest.parent_id:
                is_valid = False
                errors.append("Student must have parent_id")
            if "inference_latency_ms" in manifest.performance_metrics:
                if manifest.performance_metrics["inference_latency_ms"] > float(
                    self._policy.max_inference_latency_ms
                ):
                    is_valid = False
                    errors.append(
                        f"Student inference must be under {self._policy.max_inference_latency_ms}ms"
                    )

        if is_valid:
            target = None
            if manifest.role == ModelRole.STUDENT:
                target = "ml_signal_actor"
            elif manifest.role == ModelRole.INFERENCE:
                target = "ml_signal_actor"

            if target:
                self.deploy_model(manifest.model_id, target)
                logger.info("Auto-deployed %s to %s", manifest.model_id, target)
        else:
            logger.warning("Auto-deploy skipped for %s: %s", manifest.model_id, errors)
