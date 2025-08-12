#!/usr/bin/env python3

"""
Local file-based model registry implementation.

This module provides a JSON-based registry for environments without external model
registry services like MLflow.

"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from ml.config.constants import SUFFIX_ONNX
from ml.config.constants import ExportFormats
from ml.config.constants import Providers
from ml.config.constants import Versions
from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRegistry
from ml.registry.base import ModelRole
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment
from ml.registry.dataclasses import QualityGate
from ml.registry.dataclasses import RolloutPlan
from ml.registry.dataclasses import ValidationResult
from ml.registry.statistics import welch_t_test


logger = logging.getLogger(__name__)


class LocalModelRegistry(ModelRegistry):
    """
    Local file-based model registry using JSON for persistence.

    This registry stores all model information in a local JSON file, providing a
    lightweight solution for model lifecycle management without external dependencies.

    Thread-safe for concurrent operations.

    """

    def __init__(
        self,
        registry_path: Path,
        cache_size: int = 10,
        batch_save_interval: float = 0.1,
    ) -> None:
        """
        Initialize local model registry with caching and batch saves.

        Parameters
        ----------
        registry_path : Path
            Directory path for registry storage
        cache_size : int
            Maximum number of models to cache in memory
        batch_save_interval : float
            Seconds to wait before flushing batch saves (default 0.1s)

        """
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.cache_size = cache_size
        self.batch_save_interval = batch_save_interval

        # Store absolute path for security validation
        self._registry_root = self.registry_path.resolve()

        self.registry_file = self.registry_path / "registry.json"
        self._lock = threading.RLock()  # Use RLock to allow reentrant locking

        # In-memory model cache for performance
        self._model_cache: dict[str, Any] = {}
        self._cache_access_times: dict[str, float] = {}

        # Batch save management
        self._pending_save = False
        self._save_timer: threading.Timer | None = None

        # Initialize or load registry
        self._load_registry()

        logger.info(
            f"Initialized LocalModelRegistry at {registry_path} with cache_size={cache_size}, batch_save_interval={batch_save_interval}s",
        )

    def _load_registry(self) -> None:
        """
        Load registry from disk or create new one.
        """
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                self._models: dict[str, ModelInfo] = {
                    model_id: self._dict_to_model_info(model_data)
                    for model_id, model_data in data.get("models", {}).items()
                }
                self._ab_tests: dict[str, dict[str, Any]] = data.get("ab_tests", {})
                self._deployments: dict[str, list[str]] = data.get("deployments", {})
        else:
            self._models = {}
            self._ab_tests = {}
            self._deployments = {}  # target -> model_ids
            self._save_registry()

    def _save_registry(self, immediate: bool = False) -> None:
        """
        Save registry to disk with optional batching.

        Parameters
        ----------
        immediate : bool
            If True, save immediately. If False, batch the save.

        """
        with self._lock:
            if immediate:
                # Cancel any pending batch save
                if self._save_timer is not None:
                    self._save_timer.cancel()
                    self._save_timer = None
                self._pending_save = False

                # Save immediately
                self._do_save()
            else:
                # Schedule batch save if not already pending
                if not self._pending_save:
                    self._pending_save = True

                    # Cancel existing timer if any
                    if self._save_timer is not None:
                        self._save_timer.cancel()

                    # Schedule new save
                    self._save_timer = threading.Timer(
                        self.batch_save_interval,
                        self._flush_batch_save,
                    )
                    self._save_timer.start()

    def _do_save(self) -> None:
        """
        Perform the actual save to disk.
        """
        try:
            data = {
                "models": {
                    model_id: self._model_info_to_dict(model_info)
                    for model_id, model_info in self._models.items()
                },
                "ab_tests": self._ab_tests,
                "deployments": self._deployments,
                "last_updated": time.time(),
            }

            # Ensure directory exists
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Registry saved with {len(self._models)} models")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise

    def _flush_batch_save(self) -> None:
        """
        Flush pending batch saves.
        """
        with self._lock:
            if self._pending_save:
                try:
                    self._do_save()
                except FileNotFoundError:
                    # Directory may have been deleted during cleanup
                    pass
                except Exception as e:
                    logger.error(f"Error during batch save flush: {e}")
                finally:
                    self._pending_save = False
                    self._save_timer = None

    def _model_info_to_dict(self, model_info: ModelInfo) -> dict[str, Any]:
        """
        Convert ModelInfo to dictionary for JSON serialization.
        """
        manifest_dict = {
            "model_id": model_info.manifest.model_id,
            "role": model_info.manifest.role.value,
            "data_requirements": model_info.manifest.data_requirements.value,
            "architecture": model_info.manifest.architecture,
            "feature_schema": model_info.manifest.feature_schema,
            "feature_schema_hash": model_info.manifest.feature_schema_hash,
            "parent_id": model_info.manifest.parent_id,
            "children_ids": model_info.manifest.children_ids,
            "training_config": model_info.manifest.training_config,
            "performance_metrics": model_info.manifest.performance_metrics,
            "deployment_constraints": model_info.manifest.deployment_constraints,
            "version": model_info.manifest.version,
            "created_at": model_info.manifest.created_at,
            "last_modified": model_info.manifest.last_modified,
        }

        return {
            "manifest": manifest_dict,
            "model_path": str(model_info.model_path),
            "deployment_status": model_info.deployment_status.value,
            "deployed_to": model_info.deployed_to,
            "performance_history": model_info.performance_history,
            "metadata": model_info.metadata,
        }

    def _dict_to_model_info(self, data: dict[str, Any]) -> ModelInfo:
        """
        Convert dictionary to ModelInfo.
        """
        # Handle both old and new format
        if "manifest" in data:
            manifest_data = data["manifest"]
            manifest = ModelManifest(
                model_id=manifest_data["model_id"],
                role=ModelRole(manifest_data["role"]),
                data_requirements=DataRequirements(manifest_data["data_requirements"]),
                architecture=manifest_data["architecture"],
                feature_schema=manifest_data["feature_schema"],
                feature_schema_hash=manifest_data["feature_schema_hash"],
                parent_id=manifest_data.get("parent_id"),
                children_ids=manifest_data.get("children_ids", []),
                training_config=manifest_data.get("training_config", {}),
                performance_metrics=manifest_data.get("performance_metrics", {}),
                deployment_constraints=manifest_data.get("deployment_constraints", {}),
                version=manifest_data["version"],
                created_at=manifest_data["created_at"],
                last_modified=manifest_data["last_modified"],
            )
        else:
            # Legacy format - convert to manifest
            manifest = ModelManifest(
                model_id=data["model_id"],
                role=ModelRole.INFERENCE,  # Default for legacy
                data_requirements=DataRequirements.L1_ONLY,  # Default
                architecture="unknown",
                feature_schema={},
                feature_schema_hash="",
                version=data.get("version", Versions.DEFAULT_MANIFEST_VERSION),
                created_at=data.get("created_at", time.time()),
                last_modified=data.get("last_modified", time.time()),
            )

        return ModelInfo(
            manifest=manifest,
            model_path=Path(data["model_path"]),
            deployment_status=DeploymentStatus(data["deployment_status"]),
            deployed_to=data.get("deployed_to", []),
            performance_history=data.get("performance_history", []),
            metadata=data.get("metadata", {}),
        )

    def _generate_model_id(self) -> str:
        """
        Generate unique model ID.
        """
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        return f"model_{timestamp}"

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
        with self._lock:
            # Validate model file exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Security: Validate model format (only ONNX allowed)
            if model_path.suffix != SUFFIX_ONNX:
                raise ValueError(
                    f"Only ONNX models are supported for security reasons. "
                    f"Got: {model_path.suffix}. Please export your model to ONNX format.",
                )

            # Security: Validate path is safe
            if not self._validate_model_path(model_path):
                raise ValueError(f"Security: Invalid model path: {model_path}")

            # Use manifest's model_id or generate new one
            if not manifest.model_id:
                manifest.model_id = self._generate_model_id()

            # Set timestamps
            manifest.created_at = time.time()
            manifest.last_modified = time.time()

            # Auto-version if needed
            if not manifest.version:
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
                    manifest.version = Versions.DEFAULT_MANIFEST_VERSION

            # Quality validation if gates provided
            quality_validation_result = None
            if quality_gates:
                quality_validation_result = self._validate_quality_gates(
                    manifest.model_id,
                    manifest.performance_metrics,
                    quality_gates,
                )

                if not quality_validation_result.overall_pass and enforce_quality:
                    failed_gates = [
                        name
                        for name, result in quality_validation_result.gate_results.items()
                        if not result["passed"] and result["required"]
                    ]
                    raise ValueError(
                        f"Quality gates not met for model {manifest.model_id}. "
                        f"Failed gates: {failed_gates}",
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
            self._save_registry()

            logger.info(
                f"Registered {manifest.role.value} model {manifest.model_id} "
                f"(version {manifest.version}) at {model_path}",
            )

            # Auto-deploy if requested and validation passes
            if auto_deploy:
                # Basic validation (avoid circular import)
                is_valid = True
                errors = []

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
                        if manifest.performance_metrics["inference_latency_ms"] > 5:
                            is_valid = False
                            errors.append("Student inference must be under 5ms")

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
                    logger.warning(f"Auto-deploy skipped for {manifest.model_id}: {errors}")

            return manifest.model_id

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
        config : Optional[dict[str, Any]]
            Deployment configuration

        Returns
        -------
        bool
            True if deployment successful

        """
        with self._lock:
            if model_id not in self._models:
                logger.error(f"Model {model_id} not found in registry")
                return False

            model_info = self._models[model_id]

            # Update deployment status
            model_info.deployment_status = DeploymentStatus.ACTIVE
            model_info.deployed_to.append(target)
            model_info.manifest.last_modified = time.time()

            # Track deployment
            if target not in self._deployments:
                self._deployments[target] = []

            # Remove previous model for this target if exists
            self._deployments[target] = [model_id]

            # Store deployment config in metadata
            if config:
                model_info.metadata["deployment_config"] = config

            self._save_registry()

            logger.info(f"Deployed model {model_id} to {target}")
            return True

    def get_active_models(self) -> list[ModelInfo]:
        """
        Get all currently deployed models.
        """
        with self._lock:
            return [
                model_info
                for model_info in self._models.values()
                if model_info.deployment_status == DeploymentStatus.ACTIVE
            ]

    def get_all_models(self) -> list[ModelInfo]:
        """
        Get all registered models.
        """
        with self._lock:
            return list(self._models.values())

    def get_model(self, model_id: str) -> ModelInfo | None:
        """
        Get information about a specific model.
        """
        with self._lock:
            return self._models.get(model_id)

    def get_models_by_role(self, role: ModelRole) -> list[ModelInfo]:
        """
        Get all models with a specific role.
        """
        with self._lock:
            return [
                model_info
                for model_info in self._models.values()
                if model_info.manifest.role == role
            ]

    def get_models_by_data_requirements(
        self,
        requirements: DataRequirements,
    ) -> list[ModelInfo]:
        """
        Get all models with specific data requirements.
        """
        with self._lock:
            return [
                model_info
                for model_info in self._models.values()
                if model_info.manifest.data_requirements == requirements
            ]

    def get_model_lineage(self, model_id: str) -> list[ModelInfo]:
        """
        Get complete lineage of a model (parents and children).
        """
        with self._lock:
            if model_id not in self._models:
                return []

            lineage: list[ModelInfo] = []
            model = self._models[model_id]

            # Trace parents
            current_id = model.manifest.parent_id
            while current_id and current_id in self._models:
                parent = self._models[current_id]
                lineage.insert(0, parent)  # Add to beginning
                current_id = parent.manifest.parent_id

            # Add the model itself
            lineage.append(model)

            # Add children
            for child_id in model.manifest.children_ids:
                if child_id in self._models:
                    lineage.append(self._models[child_id])

            return lineage

    def load_model(self, model_id: str) -> Any | None:
        """
        Load model from cache or disk.

        SECURITY: Only loads ONNX models to prevent code execution vulnerabilities.

        Parameters
        ----------
        model_id : str
            Model ID to load

        Returns
        -------
        Any | None
            Loaded model object (ONNX InferenceSession) or None if not found

        """
        with self._lock:
            # Check cache first
            if model_id in self._model_cache:
                self._cache_access_times[model_id] = time.time()
                return self._model_cache[model_id]

            # Load from disk
            if model_id not in self._models:
                return None

            model_info = self._models[model_id]
            model_path = model_info.model_path

            # Validate path security
            if not self._validate_model_path(model_path):
                logger.error(f"Security: Invalid model path detected: {model_path}")
                return None

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None

            # Only support ONNX format for security
            try:
                from ml.config.constants import ExportFormats
                from ml.config.constants import Providers

                if model_path.suffix == SUFFIX_ONNX:
                    # Load ONNX model following signal actor pattern
                    from ml._imports import HAS_ONNX
                    from ml._imports import check_ml_dependencies
                    from ml._imports import ort

                    if not HAS_ONNX:
                        check_ml_dependencies(["onnxruntime"])

                    # Create optimized session like in ONNXModelLoader
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

                    # Use CPU provider for predictable latency
                    providers = [Providers.ONNX_PROVIDER_CPU]

                    model = ort.InferenceSession(
                        str(model_path),
                        sess_options=session_options,
                        providers=providers,
                    )
                else:
                    logger.error(
                        f"Unsupported model format: {model_path.suffix}. "
                        f"Only ONNX models are supported for security reasons.",
                    )
                    return None

                # Update cache (with LRU eviction)
                if len(self._model_cache) >= self.cache_size:
                    # Evict least recently used
                    lru_id = min(
                        self._cache_access_times.items(),
                        key=lambda x: x[1],
                    )[0]
                    del self._model_cache[lru_id]
                    del self._cache_access_times[lru_id]

                self._model_cache[model_id] = model
                self._cache_access_times[model_id] = time.time()

                return model

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
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
            Model ID
        metrics : dict[str, Any]
            Performance metrics

        """
        with self._lock:
            if model_id not in self._models:
                logger.error(f"Model {model_id} not found in registry")
                return

            # Add timestamp if not present
            if "timestamp" not in metrics:
                metrics["timestamp"] = time.time()

            # Append to history
            self._models[model_id].performance_history.append(metrics)
            self._models[model_id].manifest.last_modified = time.time()

            self._save_registry()

            logger.debug(f"Tracked performance for model {model_id}: {metrics}")

    def get_performance_history(
        self,
        model_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get performance history for a model.
        """
        with self._lock:
            if model_id not in self._models:
                return []
            return self._models[model_id].performance_history.copy()

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
            Deployment target
        to_model_id : str
            Model ID to rollback to

        Returns
        -------
        bool
            True if rollback successful

        """
        with self._lock:
            if to_model_id not in self._models:
                logger.error(f"Model {to_model_id} not found in registry")
                return False

            # Deactivate current model for target
            if target in self._deployments:
                for current_model_id in self._deployments[target]:
                    if current_model_id in self._models:
                        current_model = self._models[current_model_id]
                        current_model.deployment_status = DeploymentStatus.INACTIVE
                        if target in current_model.deployed_to:
                            current_model.deployed_to.remove(target)

            # Activate rollback model
            rollback_model = self._models[to_model_id]
            rollback_model.deployment_status = DeploymentStatus.ACTIVE
            if target not in rollback_model.deployed_to:
                rollback_model.deployed_to.append(target)
            rollback_model.manifest.last_modified = time.time()

            # Update deployments
            self._deployments[target] = [to_model_id]

            self._save_registry()

            logger.info(f"Rolled back {target} to model {to_model_id}")
            return True

    def flush(self) -> None:
        """
        Flush any pending batch saves immediately.

        Call this before shutdown or when immediate persistence is needed.

        """
        with self._lock:
            if self._pending_save:
                if self._save_timer is not None:
                    self._save_timer.cancel()
                    self._save_timer = None
                self._do_save()
                self._pending_save = False
                logger.debug("Flushed pending batch saves")

    def __del__(self) -> None:
        """
        Ensure pending saves are flushed on cleanup.
        """
        try:
            self.flush()
        except Exception:
            pass  # Best effort on cleanup

    def _validate_model_path(self, path: Path) -> bool:
        """
        Validate model path is within registry bounds.

        Prevents path traversal attacks.

        Parameters
        ----------
        path : Path
            Path to validate

        Returns
        -------
        bool
            True if path is valid and safe

        """
        try:
            resolved = path.resolve()
            # Check if resolved path is within registry root
            return str(resolved).startswith(str(self._registry_root))
        except (ValueError, RuntimeError):
            # Path resolution failed
            return False

    def retire_model(self, model_id: str) -> bool:
        """
        Retire a model from production.

        Parameters
        ----------
        model_id : str
            Model ID to retire

        Returns
        -------
        bool
            True if retirement successful

        """
        with self._lock:
            if model_id not in self._models:
                logger.error(f"Model {model_id} not found in registry")
                return False

            model_info = self._models[model_id]
            model_info.deployment_status = DeploymentStatus.RETIRED
            model_info.manifest.last_modified = time.time()

            # Remove from all deployments
            for target in list(model_info.deployed_to):
                if target in self._deployments and model_id in self._deployments[target]:
                    self._deployments[target].remove(model_id)

            model_info.deployed_to.clear()

            self._save_registry()

            logger.info(f"Retired model {model_id}")
            return True

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
            List of model IDs to test (expects 2 models)
        split_ratio : float
            Traffic split ratio for first model
        duration_hours : int
            Test duration in hours
        target : str
            Deployment target

        Returns
        -------
        Optional[dict[str, Any]]
            A/B test configuration

        """
        with self._lock:
            if len(models) != 2:
                logger.error("A/B test requires exactly 2 models")
                return None

            model_a_id, model_b_id = models

            if model_a_id not in self._models or model_b_id not in self._models:
                logger.error("One or more models not found in registry")
                return None

            # Create A/B test config
            ab_config = {
                "model_a": model_a_id,
                "model_b": model_b_id,
                "split_ratio": split_ratio,
                "duration_hours": duration_hours,
                "target": target,
                "start_time": time.time(),
                "end_time": time.time() + (duration_hours * 3600),
                "status": "active",
            }

            # Deploy both models
            for model_id in models:
                model_info = self._models[model_id]
                model_info.deployment_status = DeploymentStatus.TESTING
                if target not in model_info.deployed_to:
                    model_info.deployed_to.append(target)
                model_info.manifest.last_modified = time.time()

            # Store A/B test config
            test_id = f"ab_test_{int(time.time())}"
            self._ab_tests[test_id] = ab_config

            # Update deployments to include both models
            self._deployments[target] = models

            self._save_registry()

            logger.info(f"Configured A/B test {test_id} for models {models} on {target}")
            return ab_config

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
            List of model IDs to compare
        metric : str
            Metric to compare on

        Returns
        -------
        Optional[dict[str, Any]]
            Comparison results

        """
        with self._lock:
            results = []

            for model_id in model_ids:
                if model_id not in self._models:
                    logger.warning(f"Model {model_id} not found, skipping")
                    continue

                model_info = self._models[model_id]

                # Get latest metric value
                metric_value = None
                for perf in reversed(model_info.performance_history):
                    if metric in perf:
                        metric_value = perf[metric]
                        break

                if metric_value is not None:
                    results.append(
                        {
                            "model_id": model_id,
                            "version": model_info.manifest.version,
                            metric: metric_value,
                        },
                    )

            if not results:
                return None

            # Sort by metric (descending)
            results.sort(key=lambda x: x[metric], reverse=True)

            comparison = {
                "metric": metric,
                "rankings": results,
                "best_model": results[0]["model_id"] if results else None,
            }

            return comparison

    # ========== Enhanced Methods for Quality Validation ==========

    def _validate_quality_gates(
        self,
        model_id: str,
        metrics: dict[str, float],
        gates: list[QualityGate],
    ) -> ValidationResult:
        """
        Validate model metrics against quality gates.

        Parameters
        ----------
        model_id : str
            Model identifier
        metrics : dict[str, float]
            Model metrics to validate
        gates : list[QualityGate]
            Quality gates to check

        Returns
        -------
        ValidationResult
            Validation results with pass/fail status

        """
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
        """
        Evaluate a single quality gate.

        Parameters
        ----------
        gate : QualityGate
            Gate to evaluate
        actual_value : Optional[float]
            Actual metric value

        Returns
        -------
        dict[str, Any]
            Gate evaluation result

        """
        if actual_value is None:
            return {
                "threshold": gate.threshold,
                "actual": None,
                "passed": False,
                "required": gate.required,
                "reason": "metric_not_found",
            }

        # Perform comparison
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
        with self._lock:
            if model_id not in self._models:
                result = ValidationResult(model_id=model_id)
                result.overall_pass = False
                result.gate_results["model_existence"] = {
                    "passed": False,
                    "reason": "model_not_found",
                }
                return result

            model_info = self._models[model_id]
            return self._validate_quality_gates(
                model_id,
                model_info.manifest.performance_metrics,
                gates,
            )

    # ========== Canary Deployment Methods ==========

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
            Model to deploy as canary
        target : str
            Deployment target
        config : CanaryConfig
            Canary configuration
        baseline_model_id : Optional[str]
            Baseline model for comparison (current prod if None)

        Returns
        -------
        str
            Canary deployment ID

        """
        with self._lock:
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not found")

            # Generate deployment ID
            deployment_id = f"canary_{int(time.time())}_{model_id}"

            # Get baseline performance if needed
            baseline_performance = None
            if baseline_model_id:
                if baseline_model_id in self._models:
                    baseline_metrics = self._models[baseline_model_id].manifest.performance_metrics
                    baseline_performance = baseline_metrics.get(config.success_metric)
            else:
                # Find current production model for target
                for m_id, m_info in self._models.items():
                    if (
                        target in m_info.deployed_to
                        and m_info.deployment_status == DeploymentStatus.ACTIVE
                    ):
                        baseline_model_id = m_id
                        baseline_performance = m_info.manifest.performance_metrics.get(
                            config.success_metric,
                        )
                        break

            # Create canary deployment
            canary = CanaryDeployment(
                deployment_id=deployment_id,
                model_id=model_id,
                target=target,
                config=config,
                baseline_model_id=baseline_model_id,
                baseline_performance=baseline_performance,
            )

            # Store canary deployment
            if not hasattr(self, "_canary_deployments"):
                self._canary_deployments: dict[str, CanaryDeployment] = {}
            self._canary_deployments[deployment_id] = canary

            # Update model status
            model_info = self._models[model_id]
            model_info.deployment_status = DeploymentStatus.TESTING
            model_info.metadata["canary_deployment"] = deployment_id

            self._save_registry()
            logger.info(f"Started canary deployment {deployment_id} for model {model_id}")

            return deployment_id

    def get_canary_deployment(self, deployment_id: str) -> CanaryDeployment | None:
        """
        Get canary deployment by ID.
        """
        if not hasattr(self, "_canary_deployments"):
            return None
        return self._canary_deployments.get(deployment_id)

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
            Canary deployment ID
        metric_value : float
            Value of the success metric
        latency_ms : Optional[float]
            Response latency
        error_occurred : bool
            Whether an error occurred

        """
        with self._lock:
            if not hasattr(self, "_canary_deployments"):
                return

            canary = self._canary_deployments.get(deployment_id)
            if canary:
                canary.record_metric(metric_value, latency_ms, error_occurred)

    def evaluate_canary(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be promoted.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID

        Returns
        -------
        tuple[bool, str]
            (should_promote, reason)

        """
        with self._lock:
            if not hasattr(self, "_canary_deployments"):
                return False, "no_canary_deployments"

            canary = self._canary_deployments.get(deployment_id)
            if not canary:
                return False, "deployment_not_found"

            return canary.should_promote()

    def evaluate_canary_for_rollback(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be rolled back.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID

        Returns
        -------
        tuple[bool, str]
            (should_rollback, reason)

        """
        with self._lock:
            if not hasattr(self, "_canary_deployments"):
                return False, "no_canary_deployments"

            canary = self._canary_deployments.get(deployment_id)
            if not canary:
                return False, "deployment_not_found"

            return canary.should_rollback()

    def auto_promote_canary(self, deployment_id: str) -> bool:
        """
        Automatically promote a canary to full production.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID

        Returns
        -------
        bool
            True if promotion successful

        """
        with self._lock:
            if not hasattr(self, "_canary_deployments"):
                return False

            canary = self._canary_deployments.get(deployment_id)
            if not canary:
                return False

            # Promote model to full deployment
            success = self.deploy_model(
                model_id=canary.model_id,
                target=canary.target,
                config={"traffic_percentage": 100.0},
            )

            if success:
                canary.status = "promoted"
                # Retire baseline if exists
                if canary.baseline_model_id:
                    self.retire_model(canary.baseline_model_id)

                logger.info(f"Promoted canary {deployment_id} to full production")

            return success

    # ========== Statistical Comparison Methods ==========

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
            Model IDs to compare
        metric : str
            Metric to compare

        Returns
        -------
        Optional[dict[str, Any]]
            Statistical comparison results

        """
        with self._lock:
            if len(model_ids) != 2:
                logger.error("Statistical comparison requires exactly 2 models")
                return None

            model_a_id, model_b_id = model_ids

            # Collect metric samples
            samples_a = []
            samples_b = []

            if model_a_id in self._models:
                for perf in self._models[model_a_id].performance_history:
                    if metric in perf:
                        samples_a.append(perf[metric])

            if model_b_id in self._models:
                for perf in self._models[model_b_id].performance_history:
                    if metric in perf:
                        samples_b.append(perf[metric])

            if not samples_a or not samples_b:
                return None

            # Perform Welch's t-test
            import numpy as np

            test_result = welch_t_test(
                np.array(samples_a),
                np.array(samples_b),
            )

            test_result["model_a"] = model_a_id
            test_result["model_b"] = model_b_id
            test_result["metric"] = metric

            return test_result

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
            Control model ID
        model_b_id : str
            Treatment model ID
        split_ratio : float
            Traffic split for control (0.0 to 1.0)
        duration_hours : float
            Test duration
        target : str
            Deployment target

        Returns
        -------
        str
            A/B test ID

        """
        with self._lock:
            # Use existing configure_ab_test
            config = self.configure_ab_test(
                models=[model_a_id, model_b_id],
                split_ratio=split_ratio,
                duration_hours=int(duration_hours),
                target=target,
            )

            if config:
                test_id = f"ab_test_{int(time.time())}"
                # Store A/B test metrics tracking
                if not hasattr(self, "_ab_test_metrics"):
                    self._ab_test_metrics: dict[str, dict[str, list[float]]] = {}
                self._ab_test_metrics[test_id] = {
                    model_a_id: [],
                    model_b_id: [],
                }
                return test_id

            return ""

    def track_ab_test_metric(
        self,
        test_id: str,
        model_id: str,
        metric_value: float,
    ) -> None:
        """
        Track metric for A/B test.
        """
        with self._lock:
            if not hasattr(self, "_ab_test_metrics"):
                return

            if test_id in self._ab_test_metrics:
                if model_id in self._ab_test_metrics[test_id]:
                    self._ab_test_metrics[test_id][model_id].append(metric_value)

    def analyze_ab_test(self, test_id: str) -> dict[str, Any] | None:
        """
        Analyze A/B test results.

        Parameters
        ----------
        test_id : str
            A/B test ID

        Returns
        -------
        Optional[dict[str, Any]]
            Analysis results

        """
        with self._lock:
            if not hasattr(self, "_ab_test_metrics"):
                return None

            if test_id not in self._ab_test_metrics:
                return None

            metrics = self._ab_test_metrics[test_id]
            model_ids = list(metrics.keys())

            if len(model_ids) != 2:
                return None

            control_id = model_ids[0]
            treatment_id = model_ids[1]

            control_samples = metrics[control_id]
            treatment_samples = metrics[treatment_id]

            if not control_samples or not treatment_samples:
                return None

            import numpy as np

            control_mean = np.mean(control_samples)
            treatment_mean = np.mean(treatment_samples)

            # Perform statistical test
            test_result = welch_t_test(
                np.array(control_samples),
                np.array(treatment_samples),
            )

            return {
                "test_id": test_id,
                "control_model": control_id,
                "treatment_model": treatment_id,
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "relative_improvement": test_result["relative_improvement"],
                "statistical_significance": test_result["statistically_significant"],
                "p_value": test_result["p_value_approx"],
            }

    # ========== Hot Reload and Gradual Rollout Methods ==========

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
            Deployment target
        new_model_id : str
            New model to deploy

        Returns
        -------
        bool
            True if successful

        """
        with self._lock:
            if new_model_id not in self._models:
                logger.error(f"Model {new_model_id} not found")
                return False

            # Find current model for target
            current_model_id = None
            for model_id, model_info in self._models.items():
                if (
                    target in model_info.deployed_to
                    and model_info.deployment_status == DeploymentStatus.ACTIVE
                ):
                    current_model_id = model_id
                    break

            if not current_model_id:
                # No current model, just deploy new one
                return self.deploy_model(new_model_id, target)

            # Validate feature compatibility
            current_model = self._models[current_model_id]
            new_model = self._models[new_model_id]

            if current_model.manifest.feature_schema_hash != new_model.manifest.feature_schema_hash:
                logger.warning(
                    f"Feature schema mismatch during hot reload: "
                    f"current={current_model.manifest.feature_schema_hash}, "
                    f"new={new_model.manifest.feature_schema_hash}",
                )

            # Deploy new model
            success = self.deploy_model(new_model_id, target)

            if success:
                # Retire old model
                self.retire_model(current_model_id)
                logger.info(f"Hot reloaded {target}: {current_model_id} -> {new_model_id}")

            return success

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
            Currently deployed model
        new_model_id : str
            New model to roll out
        target : str
            Deployment target
        stages : list[float]
            Traffic percentages for each stage
        stage_duration_minutes : int
            Duration of each stage

        Returns
        -------
        str
            Rollout ID

        """
        with self._lock:
            if current_model_id not in self._models or new_model_id not in self._models:
                raise ValueError("One or both models not found")

            rollout_id = f"rollout_{int(time.time())}"

            # Create rollout plan
            rollout = RolloutPlan(
                rollout_id=rollout_id,
                current_model_id=current_model_id,
                new_model_id=new_model_id,
                target=target,
                stages=stages,
                stage_duration_minutes=stage_duration_minutes,
            )

            # Store rollout plan
            if not hasattr(self, "_rollout_plans"):
                self._rollout_plans: dict[str, RolloutPlan] = {}
            self._rollout_plans[rollout_id] = rollout

            # Start first stage
            if stages:
                self.configure_ab_test(
                    models=[current_model_id, new_model_id],
                    split_ratio=1.0 - stages[0],
                    duration_hours=max(
                        1,
                        stage_duration_minutes * 60,
                    ),  # Convert to hours (stage_duration is already in minutes * 60)
                    target=target,
                )

            logger.info(f"Started gradual rollout {rollout_id}")
            return rollout_id

    def get_rollout_status(self, rollout_id: str) -> dict[str, Any] | None:
        """
        Get rollout status.
        """
        with self._lock:
            if not hasattr(self, "_rollout_plans"):
                return None

            rollout = self._rollout_plans.get(rollout_id)
            if not rollout:
                return None

            return {
                "rollout_id": rollout.rollout_id,
                "current_stage": rollout.current_stage,
                "stages": rollout.stages,
                "traffic_split": rollout.get_current_traffic_split(),
                "status": rollout.status,
            }

    def advance_rollout_stage(self, rollout_id: str) -> bool:
        """
        Advance to next rollout stage.
        """
        with self._lock:
            if not hasattr(self, "_rollout_plans"):
                return False

            rollout = self._rollout_plans.get(rollout_id)
            if not rollout:
                return False

            if rollout.advance_stage():
                # Configure next stage
                new_split = rollout.get_current_traffic_split()
                self.configure_ab_test(
                    models=[rollout.current_model_id, rollout.new_model_id],
                    split_ratio=1.0 - new_split,
                    duration_hours=max(
                        1,
                        int(rollout.stage_duration_minutes / 60),
                    ),  # Convert minutes to hours, min 1
                    target=rollout.target,
                )
                logger.info(f"Advanced rollout {rollout_id} to stage {rollout.current_stage}")
                return True

            return False
