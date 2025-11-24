#!/usr/bin/env python3

"""
Local file-based model registry implementation with configurable persistence backends.

This module provides a registry that can use either JSON files or PostgreSQL for
persistence, making it suitable for both development and production environments.

"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, cast

from ml.config.constants import SUFFIX_ONNX
from ml.config.constants import Versions
from ml.config.registry import RegistryPolicyConfig
from ml.config.runtime import OnnxRuntimeConfig
from ml.registry.abstract_registry import AbstractRegistry
from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.dataclasses import CanaryConfig
from ml.registry.dataclasses import CanaryDeployment
from ml.registry.dataclasses import QualityGate
from ml.registry.dataclasses import RolloutPlan
from ml.registry.dataclasses import ValidationResult
from ml.registry.persistence import BackendType
from ml.registry.persistence import ModelTable
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager
from ml.registry.statistics import welch_t_test


logger = logging.getLogger(__name__)
__all__ = [
    "DataRequirements",
    "DeploymentStatus",
    "ModelInfo",
    "ModelManifest",
    "ModelRegistry",
    "ModelRole",
]

# Re-export security-related runtime toggles for tests to patch.
# Tests patch ml.registry.model_registry.HAS_ONNX and .ort directly.
try:  # Lightweight import; symbols are stubs when onnxruntime is unavailable
    from ml.common.security import HAS_ONNX as HAS_ONNX
    from ml.common.security import check_ml_dependencies as check_ml_dependencies
    from ml.common.security import ort as ort
except Exception:  # pragma: no cover - fallback when security module unavailable
    HAS_ONNX = False
    ort = None

    def check_ml_dependencies(_deps: list[str]) -> None:
        raise RuntimeError("Dependency check unavailable")


class ModelRegistry(AbstractRegistry):
    """
    Model registry with configurable persistence backend.

    This registry can use either JSON files or PostgreSQL for persistence, providing a
    flexible solution for model lifecycle management in both development and production
    environments.

    Thread-safe for concurrent operations.

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
        Initialize local model registry with configurable persistence backend.

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

        """
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.cache_size = cache_size
        self.batch_save_interval = batch_save_interval
        self._policy = policy_config or RegistryPolicyConfig()
        self._onnx_rt = onnx_runtime_config or OnnxRuntimeConfig()

        # Store absolute path for security validation
        self._registry_root = self.registry_path.resolve()

        # Setup persistence
        if persistence_config is None:
            persistence_config = PersistenceConfig(
                backend=BackendType.JSON,
                json_path=registry_path,
            )
        persistence = PersistenceManager(persistence_config)
        super().__init__(persistence)

        self.registry_file = self.registry_path / "registry.json"
        # RLock provided by AbstractRegistry

        # In-memory model cache for performance
        self._model_cache: dict[str, Any] = {}
        self._cache_access_times: dict[str, float] = {}

        # Batch save management
        self._pending_save = False
        self._save_timer: threading.Timer | None = None

        # Pre-initialize core dictionaries to ensure they exist even if _load_registry()
        # is mocked or fails. This defensive pattern matches FeatureRegistry and DataRegistry.
        self._models: dict[str, ModelInfo] = {}
        self._ab_tests: dict[str, dict[str, Any]] = {}
        self._deployments: dict[str, list[str]] = {}
        self._rollout_plans: dict[str, RolloutPlan] = {}
        self._canary_deployments: dict[str, CanaryDeployment] = {}
        self._ab_test_metrics: dict[str, dict[str, list[float]]] = {}

        # Initialize or load registry
        self._load_registry()

        logger.info(
            "Initialized ModelRegistry at %s with backend=%s, cache_size=%s, batch_save_interval=%ss",
            registry_path,
            self.backend.value,
            cache_size,
            batch_save_interval,
        )

    def _load_registry(self) -> None:
        """
        Load registry from persistence backend or create new one.
        """
        if self.backend == BackendType.JSON:
            if self.registry_file.exists():
                data = self._json_load("registry.json")
                if data is not None:
                    # Use update() to preserve pre-initialized dicts
                    self._models.update(
                        {
                            model_id: self._dict_to_model_info(model_data)
                            for model_id, model_data in data.get("models", {}).items()
                        }
                    )
                    self._ab_tests.update(data.get("ab_tests", {}))
                    self._deployments.update(data.get("deployments", {}))
                # If data is None, dicts are already initialized, no action needed
            else:
                # Dicts already initialized in __init__, just save empty registry
                self._save_registry()
        elif self.backend == BackendType.POSTGRES:
            # Load all models from PostgreSQL
            session = self.persistence.get_session()
            if session is None:
                # Dicts already initialized in __init__
                return
            try:
                # Dicts already initialized; clear and repopulate from database
                self._models.clear()
                self._ab_tests.clear()
                self._deployments.clear()

                models = session.query(ModelTable).all()
                for model in models:
                    model_info = self._db_to_model_info(model)
                    self._models[model_info.manifest.model_id] = model_info

                    # Reconstruct deployments
                    for target in cast(list[str], model.deployed_to) or []:
                        if target not in self._deployments:
                            self._deployments[target] = []
                        self._deployments[target].append(model_info.manifest.model_id)
            except Exception:
                logger.warning(
                    "Error loading from database. Starting with empty registry.",
                    exc_info=True,
                )
                # Clear dicts on error (already initialized in __init__)
                self._models.clear()
                self._ab_tests.clear()
                self._deployments.clear()
            finally:
                session.close()

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
                # Cancel any pending batch save (no join - hot path optimization)
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

                    # Cancel existing timer if any (no join - hot path optimization)
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
        except Exception:
            logger.error(
                "Failed to save registry",
                exc_info=True,
            )
            raise

    def _flush_batch_save(self) -> None:
        """
        Flush pending batch saves.
        """
        with self._lock:
            if self._pending_save:
                try:
                    self._do_save()
                except FileNotFoundError as exc:
                    # Directory may have been deleted during cleanup
                    import logging as _logging

                    _logging.getLogger(__name__).debug(
                        "Batch save flush: registry path missing (ignored): %s",
                        exc,
                        exc_info=False,
                    )
                except Exception:
                    logger.error(
                        "Error during batch save flush",
                        exc_info=True,
                    )
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
            "serveable": getattr(model_info.manifest, "serveable", True),
            "artifact_format": getattr(model_info.manifest, "artifact_format", "onnx"),
            "feature_set_id": getattr(model_info.manifest, "feature_set_id", None),
            "pipeline_signature": getattr(model_info.manifest, "pipeline_signature", None),
            "pipeline_version": getattr(model_info.manifest, "pipeline_version", None),
            "decision_policy": getattr(model_info.manifest, "decision_policy", None),
            "decision_config": getattr(model_info.manifest, "decision_config", {}),
            "artifact_sha256_digest": getattr(model_info.manifest, "artifact_sha256_digest", None),
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
            manifest_data = cast(dict[str, Any], data["manifest"])
            manifest = ModelManifest(
                model_id=cast(str, manifest_data["model_id"]),
                role=ModelRole(manifest_data["role"]),
                data_requirements=DataRequirements(manifest_data["data_requirements"]),
                architecture=str(manifest_data["architecture"]),
                feature_schema=cast(dict[str, str], manifest_data.get("feature_schema", {})) or {},
                feature_schema_hash=cast(str, manifest_data["feature_schema_hash"]),
                parent_id=cast(str | None, manifest_data.get("parent_id")),
                children_ids=cast(list[str], manifest_data.get("children_ids", [])) or [],
                training_config=cast(dict[str, Any], manifest_data.get("training_config", {}))
                or {},
                performance_metrics=cast(
                    dict[str, float], manifest_data.get("performance_metrics", {})
                )
                or {},
                deployment_constraints=cast(
                    dict[str, Any], manifest_data.get("deployment_constraints", {})
                )
                or {},
                version=cast(str, manifest_data["version"]),
                created_at=float(manifest_data["created_at"]),
                last_modified=float(manifest_data["last_modified"]),
                serveable=bool(manifest_data.get("serveable", True)),
                artifact_format=str(manifest_data.get("artifact_format", "onnx")),
                feature_set_id=cast(str | None, manifest_data.get("feature_set_id")),
                pipeline_signature=cast(str | None, manifest_data.get("pipeline_signature")),
                pipeline_version=cast(str | None, manifest_data.get("pipeline_version")),
                decision_policy=cast(str | None, manifest_data.get("decision_policy")),
                decision_config=cast(dict[str, Any], manifest_data.get("decision_config", {}))
                or {},
                artifact_sha256_digest=cast(
                    str | None, manifest_data.get("artifact_sha256_digest")
                ),
            )
        else:
            # Legacy format - convert to manifest
            manifest = ModelManifest(
                model_id=str(data["model_id"]),
                role=ModelRole.INFERENCE,  # Default for legacy
                data_requirements=DataRequirements.L1_ONLY,  # Default
                architecture=str(data.get("architecture", "unknown")),
                feature_schema=cast(dict[str, str], data.get("feature_schema", {})) or {},
                feature_schema_hash=str(data.get("feature_schema_hash", "")),
                version=str(data.get("version", Versions.DEFAULT_MANIFEST_VERSION)),
                created_at=float(data.get("created_at", time.time())),
                last_modified=float(data.get("last_modified", time.time())),
            )

        status_value = data.get("deployment_status", DeploymentStatus.INACTIVE.value)

        return ModelInfo(
            manifest=manifest,
            model_path=Path(str(data.get("model_path", ""))),
            deployment_status=DeploymentStatus(str(status_value)),
            deployed_to=cast(list[str], data.get("deployed_to", [])) or [],
            performance_history=cast(list[dict[str, Any]], data.get("performance_history", []))
            or [],
            metadata=cast(dict[str, Any], data.get("metadata", {})) or {},
        )

    def _db_to_model_info(self, db_model: ModelTable) -> ModelInfo:
        """
        Convert database model to ModelInfo.

        Parameters
        ----------
        db_model : ModelTable
            Database model record

        Returns
        -------
        ModelInfo
            Model information object

        """
        manifest = ModelManifest(
            model_id=cast(str, db_model.model_id),
            role=ModelRole(db_model.role),
            data_requirements=DataRequirements(db_model.data_requirements),
            architecture=str(db_model.architecture or "unknown"),
            feature_schema=cast(dict[str, str], db_model.feature_schema) or {},
            feature_schema_hash=str(db_model.feature_schema_hash or ""),
            parent_id=db_model.parent_id,
            children_ids=cast(list[str], db_model.children_ids) or [],
            training_config=cast(dict[str, Any], db_model.training_config) or {},
            performance_metrics=cast(dict[str, float], db_model.performance_metrics) or {},
            deployment_constraints=cast(dict[str, Any], db_model.deployment_constraints) or {},
            version=str(db_model.version or Versions.DEFAULT_MANIFEST_VERSION),
            created_at=db_model.created_at.timestamp() if db_model.created_at else time.time(),
            last_modified=(
                db_model.last_modified.timestamp() if db_model.last_modified else time.time()
            ),
            serveable=db_model.serveable == "true" if db_model.serveable else True,
            artifact_format=str(db_model.artifact_format or "onnx"),
            feature_set_id=db_model.feature_set_id,
            pipeline_signature=db_model.pipeline_signature,
            pipeline_version=db_model.pipeline_version,
            artifact_sha256_digest=db_model.artifact_sha256_digest,
        )

        return ModelInfo(
            manifest=manifest,
            model_path=Path(str(db_model.model_path)),
            deployment_status=DeploymentStatus(db_model.deployment_status),
            deployed_to=cast(list[str], db_model.deployed_to) or [],
            performance_history=cast(list[dict[str, Any]], db_model.performance_history) or [],
            metadata=cast(dict[str, Any], db_model.extra_metadata) or {},
        )

    def _save_model_to_db(self, model_info: ModelInfo) -> None:
        """
        Save model to PostgreSQL database.

        Parameters
        ----------
        model_info : ModelInfo
            Model information to save

        """
        session = self.persistence.get_session()
        if session is None:
            return
        try:
            # Check if model exists
            existing = (
                session.query(ModelTable)
                .filter_by(
                    model_id=model_info.manifest.model_id,
                )
                .first()
            )

            if existing:
                # Update existing model
                existing.role = model_info.manifest.role.value
                existing.data_requirements = model_info.manifest.data_requirements.value
                existing.architecture = model_info.manifest.architecture
                existing.feature_schema = cast(Any, model_info.manifest.feature_schema)
                existing.feature_schema_hash = model_info.manifest.feature_schema_hash
                existing.parent_id = model_info.manifest.parent_id
                existing.children_ids = cast(Any, model_info.manifest.children_ids)
                existing.training_config = cast(Any, model_info.manifest.training_config)
                existing.performance_metrics = cast(Any, model_info.manifest.performance_metrics)
                existing.deployment_constraints = cast(
                    Any, model_info.manifest.deployment_constraints
                )
                existing.deployment_status = cast(Any, model_info.deployment_status.value)
                existing.deployed_to = cast(Any, model_info.deployed_to)
                existing.version = model_info.manifest.version
                existing.extra_metadata = cast(Any, model_info.metadata)
                existing.model_path = str(model_info.model_path)
                existing.performance_history = cast(Any, model_info.performance_history)
                existing.serveable = "true" if model_info.manifest.serveable else "false"
                existing.artifact_format = model_info.manifest.artifact_format
                existing.feature_set_id = model_info.manifest.feature_set_id
                existing.pipeline_signature = model_info.manifest.pipeline_signature
                existing.pipeline_version = model_info.manifest.pipeline_version
                existing.artifact_sha256_digest = model_info.manifest.artifact_sha256_digest
            else:
                # Create new model
                new_model = ModelTable(
                    model_id=model_info.manifest.model_id,
                    role=model_info.manifest.role.value,
                    data_requirements=model_info.manifest.data_requirements.value,
                    architecture=model_info.manifest.architecture,
                    feature_schema=cast(Any, model_info.manifest.feature_schema),
                    feature_schema_hash=model_info.manifest.feature_schema_hash,
                    parent_id=model_info.manifest.parent_id,
                    children_ids=cast(Any, model_info.manifest.children_ids),
                    training_config=cast(Any, model_info.manifest.training_config),
                    performance_metrics=cast(Any, model_info.manifest.performance_metrics),
                    deployment_constraints=cast(Any, model_info.manifest.deployment_constraints),
                    deployment_status=cast(Any, model_info.deployment_status.value),
                    deployed_to=cast(Any, model_info.deployed_to),
                    version=model_info.manifest.version,
                    extra_metadata=cast(Any, model_info.metadata),
                    model_path=str(model_info.model_path),
                    performance_history=cast(Any, model_info.performance_history),
                    serveable="true" if model_info.manifest.serveable else "false",
                    artifact_format=model_info.manifest.artifact_format,
                    feature_set_id=model_info.manifest.feature_set_id,
                    pipeline_signature=model_info.manifest.pipeline_signature,
                    pipeline_version=model_info.manifest.pipeline_version,
                    artifact_sha256_digest=model_info.manifest.artifact_sha256_digest,
                )
                session.add(new_model)

            session.commit()
        except Exception:
            session.rollback()
            logger.error(
                "Failed to save model to database",
                exc_info=True,
            )
            raise
        finally:
            session.close()

    def _generate_model_id(self) -> str:
        """
        Generate unique model ID.
        """
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        return f"model_{timestamp}"

    def _calculate_file_sha256(self, file_path: Path) -> str:
        """
        Calculate SHA-256 digest of a file for integrity verification.

        Parameters
        ----------
        file_path : Path
            Path to the file

        Returns
        -------
        str
            Hexadecimal SHA-256 digest of the file

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        IOError
            If the file cannot be read

        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large models efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
        except OSError as e:
            raise OSError(f"Failed to read file {file_path}: {e}") from e

        return sha256_hash.hexdigest()

    def _verify_artifact_integrity(self, file_path: Path, expected_digest: str | None) -> None:
        """
        Verify artifact integrity using SHA-256 digest.

        Parameters
        ----------
        file_path : Path
            Path to the artifact file
        expected_digest : str | None
            Expected SHA-256 digest. If None, skip verification.

        Raises
        ------
        ValueError
            If digest verification fails or artifact is tampered
        FileNotFoundError
            If the file doesn't exist
        IOError
            If the file cannot be read

        """
        if expected_digest is None:
            logger.warning(
                f"No SHA-256 digest available for {file_path.name}, skipping integrity verification",
            )
            return

        if not expected_digest:
            logger.warning(
                f"Empty SHA-256 digest for {file_path.name}, skipping integrity verification",
            )
            return

        try:
            actual_digest = self._calculate_file_sha256(file_path)
        except (OSError, FileNotFoundError) as e:
            raise ValueError(f"Cannot verify artifact integrity: {e}") from e

        if actual_digest != expected_digest:
            # Security: Log the verification failure for auditing
            logger.error(
                f"SECURITY ALERT: Artifact integrity verification failed for {file_path}\n"
                f"Expected SHA-256: {expected_digest}\n"
                f"Actual SHA-256:   {actual_digest}\n"
                f"This indicates the model artifact may have been tampered with!",
            )
            raise ValueError(
                f"Artifact integrity verification failed for {file_path.name}. "
                f"Expected digest: {expected_digest[:16]}..., "
                f"but got: {actual_digest[:16]}... "
                f"The model artifact may have been tampered with and is rejected for security.",
            )

        logger.debug(f"Artifact integrity verified for {file_path.name}: {actual_digest[:16]}...")

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
            # Validate inputs and parity/security constraints
            self._validate_registration_inputs(model_path, manifest)

            # Calculate and set artifact SHA-256 digest for integrity verification
            # Only calculate for ONNX files (serveable models) to harden the supply chain
            if model_path.suffix == SUFFIX_ONNX:
                try:
                    artifact_digest = self._calculate_file_sha256(model_path)
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
                manifest.model_id = self._generate_model_id()

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
                self._save_model_to_db(model_info)
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
                (
                    f"Registered {manifest.role.value} model {manifest.model_id} "
                    f"(version {manifest.version}) at {model_path}"
                ),
            )

            # Auto-deploy if requested and validation passes
            if auto_deploy:
                self._maybe_auto_deploy(manifest)

            return manifest.model_id

    # ------------------------------ health hook ------------------------------
    def _health_snapshot(self) -> tuple[int, float | None]:
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

    # ---------------------------- internal helpers ----------------------------

    def _validate_registration_inputs(self, model_path: Path, manifest: ModelManifest) -> None:
        """
        Validate model path, format, security, and parity constraints.
        """
        # Existence check is deferred to digest calculation to align with
        # error semantics expected by tests (ValueError on missing file).

        # Security: Validate model format for serving
        if model_path.suffix != SUFFIX_ONNX:
            # Allow non-ONNX for non-serveable models (e.g., cold-path teachers)
            if getattr(manifest, "serveable", True):
                raise ValueError(
                    (
                        "Only ONNX models are supported for serveable models. Got: "
                        f"{model_path.suffix}."
                    ),
                )

        # Security: Validate path is safe
        if not self._validate_model_path(model_path):
            raise ValueError(f"Security: Invalid model path: {model_path}")

        # Validate feature schema linkage
        if not manifest.feature_schema_hash:
            raise ValueError("feature_schema_hash is required for all models")

        # Enforcement for serveable models: validate feature parity where possible.
        if getattr(manifest, "serveable", True):
            # Allow relaxed parity in development unless explicitly enforced via env.
            import os

            strict_parity = os.getenv("ML_STRICT_FEATURE_PARITY", "0") == "1"
            feature_registry_file = self.registry_path / "feature_registry.json"

            if not manifest.feature_set_id:
                msg = "feature_set_id is missing for serveable model; " "parity validation skipped"
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

    def _auto_version_manifest(self, manifest: ModelManifest) -> None:
        """
        Assign a semantic version to the manifest when missing.
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
            manifest.version = Versions.DEFAULT_MANIFEST_VERSION

    def _apply_quality_gates(
        self,
        manifest: ModelManifest,
        quality_gates: list[QualityGate] | None,
        enforce_quality: bool,
    ) -> ValidationResult | None:
        """
        Run quality gates when provided; optionally enforce.
        """
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
                (
                    "Quality gates not met for model "
                    f"{manifest.model_id}. Failed gates: {failed_gates}"
                ),
            )
        return result

    def _maybe_auto_deploy(self, manifest: ModelManifest) -> None:
        """
        Auto-deploy when basic constraints are met.
        """
        # Basic validation (avoid circular import)
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
                        (
                            "Student inference must be under "
                            f"{self._policy.max_inference_latency_ms}ms"
                        ),
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
                logger.error("Model %s not found in registry", model_id)
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

    # -------- Compatibility helpers --------
    def list_compatible(
        self,
        schema_hash: str,
        role: ModelRole | None = None,
        architecture: str | None = None,
    ) -> list[ModelInfo]:
        """
        List models compatible with a given feature schema hash.

        Optionally filter by role and architecture.

        """
        with self._lock:
            result = [
                m
                for m in self._models.values()
                if m.manifest.feature_schema_hash == schema_hash
                and (role is None or m.manifest.role == role)
                and (architecture is None or m.manifest.architecture == architecture)
            ]
            return result

    def resolve_latest(
        self,
        role: ModelRole,
        architecture: str,
        schema_hash: str,
    ) -> ModelInfo | None:
        """
        Resolve the latest model by version matching role, architecture, and schema
        hash.

        Uses lexical comparison of version strings.

        """
        candidates = self.list_compatible(
            schema_hash=schema_hash,
            role=role,
            architecture=architecture,
        )
        if not candidates:
            return None
        latest = max(candidates, key=lambda m: m.manifest.version)
        return latest

    def get_artifact_path(self, model_id: str) -> Path | None:
        """
        Return the model artifact path if present and within registry root.

        Does not attempt to load or execute the artifact. Use for cold-path models.

        """
        with self._lock:
            if model_id not in self._models:
                return None
            model_path = self._models[model_id].model_path
            if not self._validate_model_path(model_path):
                logger.error("Security: Invalid model path detected: %s", model_path)
                return None
            return model_path if model_path.exists() else None

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
        Any | None
            Loaded model object (ONNX InferenceSession) or None if not found

        """
        with self._lock:
            # Check cache first
            if model_id in self._model_cache:
                self._cache_access_times[model_id] = time.time()
                from typing import cast as _cast

                return _cast(object, self._model_cache[model_id])

            # Load from disk
            if model_id not in self._models:
                return None

            model_info = self._models[model_id]
            model_path = model_info.model_path

            # Validate path security
            if not self._validate_model_path(model_path):
                logger.error("Security: Invalid model path detected: %s", model_path)
                return None

            if not model_path.exists():
                logger.error("Model file not found: %s", model_path)
                return None

            # Only support ONNX format for security
            try:
                if model_path.suffix == SUFFIX_ONNX:
                    # Verify artifact integrity before loading for security
                    expected_digest = model_info.manifest.artifact_sha256_digest
                    self._verify_artifact_integrity(model_path, expected_digest)

                    # Load ONNX model using module-level re-exported symbols so tests
                    # can patch them directly.
                    if not HAS_ONNX:
                        check_ml_dependencies(["onnxruntime"])

                    # Create optimized session via helper
                    from ml.config.runtime import to_session_options as _to_sess

                    session_options, providers = _to_sess(self._onnx_rt)

                    if ort is not None:
                        session_factory = ort.InferenceSession
                    else:  # pragma: no cover - defensive runtime import
                        import onnxruntime as _ort

                        session_factory = _ort.InferenceSession

                    model = session_factory(
                        str(model_path),
                        sess_options=session_options,
                        providers=providers,
                    )
                else:
                    # Non-serveable artifact format - do not load into process for security
                    logger.info(
                        "Non-ONNX artifact requested for load; returning None to avoid unsafe loads",
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

                from typing import cast as _cast

                return _cast(object, model)

            except Exception as exc:
                # Propagate integrity failures to satisfy security contract tests
                if isinstance(exc, ValueError):
                    raise
                logger.error(
                    "Failed to load model %s",
                    model_id,
                    exc_info=True,
                )
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
                logger.error("Model %s not found in registry", model_id)
                return

            # Add timestamp if not present
            if "timestamp" not in metrics:
                metrics["timestamp"] = time.time()

            # Append to history
            self._models[model_id].performance_history.append(metrics)
            self._models[model_id].manifest.last_modified = time.time()

            self._save_registry()

            logger.debug(f"Tracked performance for model {model_id}: {metrics}")

    # --------------------------- metadata management ---------------------------
    def update_metadata(self, model_id: str, metadata: dict[str, Any]) -> None:
        """
        Update arbitrary metadata for a registered model and persist.

        This method is cold-path only and intended for orchestrator/promotions flows
        to attach auxiliary information such as `training_dataset_id`,
        `universe_instrument_ids`, or `universe_symbols`.

        Parameters
        ----------
        model_id : str
            Model ID whose metadata to update.
        metadata : dict[str, Any]
            Key/value pairs to merge into the model metadata.

        """
        with self._lock:
            if model_id not in self._models:
                logger.error("Model %s not found in registry", model_id)
                return

            try:
                current = self._models[model_id].metadata
                if not isinstance(current, dict):
                    current = {}
                    self._models[model_id].metadata = current
                # Merge shallowly to avoid surprising overwrites
                current.update({k: v for k, v in metadata.items()})
                self._models[model_id].manifest.last_modified = time.time()
                self._save_registry()
                logger.debug(
                    "Updated metadata for model %s: keys=%s",
                    model_id,
                    list(metadata.keys()),
                )
            except Exception as exc:
                logger.error("Failed updating metadata for %s: %s", model_id, exc, exc_info=True)

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
                logger.error("Model %s not found in registry", to_model_id)
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
                    self._save_timer.join()  # Wait for thread to actually finish
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
        except Exception as exc:  # Best effort on cleanup
            logger.debug("ModelRegistry cleanup flush failed", exc_info=exc)

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
                logger.error("Model %s not found in registry", model_id)
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
            required = int(self._policy.ab_models_required)
            if len(models) != required:
                logger.error("A/B test requires exactly %d models", required)
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
                    logger.warning("Model %s not found, skipping", model_id)
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

            # Store canary deployment (dict already initialized in __init__)
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
                # Store A/B test metrics tracking (dict already initialized in __init__)
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
                logger.error("Model %s not found", new_model_id)
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

            # Store rollout plan (dict already initialized in __init__)
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
