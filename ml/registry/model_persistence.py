#!/usr/bin/env python3

"""
Model persistence and artifact management.

This module provides comprehensive model persistence operations including loading,
saving, artifact integrity verification using SHA-256, and model caching with LRU
eviction. Supports both JSON and PostgreSQL backends.

Extracted from ModelRegistry god class as part of Phase 2.3 refactoring.

"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Protocol, cast

from ml.config.constants import SUFFIX_ONNX
from ml.config.constants import Versions
from ml.config.runtime import OnnxRuntimeConfig
from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.persistence import BackendType
from ml.registry.persistence import ModelTable
from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)

# Re-export security-related runtime toggles for tests to patch
try:
    from ml.common.security import HAS_ONNX as HAS_ONNX
    from ml.common.security import check_ml_dependencies as check_ml_dependencies
    from ml.common.security import ort as ort
except Exception:  # pragma: no cover
    HAS_ONNX = False
    ort = None

    def check_ml_dependencies(_deps: list[str]) -> None:
        raise RuntimeError("Dependency check unavailable")


class ModelPersistenceProtocol(Protocol):
    """
    Protocol for model persistence operations.
    """

    def load_registry(self) -> tuple[
        dict[str, ModelInfo],
        dict[str, dict[str, Any]],
        dict[str, list[str]],
    ]: ...

    def save_registry(
        self,
        models: dict[str, ModelInfo],
        ab_tests: dict[str, dict[str, Any]],
        deployments: dict[str, list[str]],
        immediate: bool = False,
    ) -> None: ...

    def load_model(self, model_id: str, model_info: ModelInfo) -> object | None: ...

    def get_artifact_path(self, model_id: str, model_info: ModelInfo) -> Path | None: ...

    def calculate_file_sha256(self, file_path: Path) -> str: ...

    def verify_artifact_integrity(
        self,
        file_path: Path,
        expected_digest: str | None,
    ) -> None: ...

    def flush(self) -> None: ...


class ModelPersistence:
    """
    Manages model persistence, artifact loading, and integrity verification.

    Handles both JSON and PostgreSQL backends, implements model caching with LRU
    eviction, and provides SHA-256 integrity verification for security.

    This component is extracted from ModelRegistry god class to provide focused,
    testable persistence functionality.

    """

    def __init__(
        self,
        registry_path: Path,
        persistence_manager: PersistenceManager,
        cache_size: int = 10,
        batch_save_interval: float = 0.1,
        onnx_runtime_config: OnnxRuntimeConfig | None = None,
    ) -> None:
        """
        Initialize model persistence.

        Parameters
        ----------
        registry_path : Path
            Registry directory path
        persistence_manager : PersistenceManager
            Persistence backend manager
        cache_size : int
            Maximum models to cache
        batch_save_interval : float
            Batch save interval in seconds
        onnx_runtime_config : OnnxRuntimeConfig | None
            ONNX runtime configuration

        """
        self.registry_path = registry_path
        self.persistence = persistence_manager
        self.cache_size = cache_size
        self.batch_save_interval = batch_save_interval
        self._onnx_rt = onnx_runtime_config or OnnxRuntimeConfig()
        self._registry_root = registry_path.resolve()
        self.registry_file = registry_path / "registry.json"

        # Model cache
        self._model_cache: dict[str, Any] = {}
        self._cache_access_times: dict[str, float] = {}

        # Batch save state
        self._lock = threading.RLock()
        self._pending_save = False
        self._save_timer: threading.Timer | None = None
        self._pending_data: (
            tuple[
                dict[str, ModelInfo],
                dict[str, dict[str, Any]],
                dict[str, list[str]],
            ]
            | None
        ) = None

        logger.debug(
            "Initialized ModelPersistence with backend=%s, cache_size=%d",
            persistence_manager.config.backend.value,
            cache_size,
        )

    @property
    def backend(self) -> BackendType:
        """
        Get persistence backend type.
        """
        return self.persistence.config.backend

    def load_registry(self) -> tuple[
        dict[str, ModelInfo],
        dict[str, dict[str, Any]],
        dict[str, list[str]],
    ]:
        """
        Load registry from persistence backend.

        Returns
        -------
        tuple[dict[str, ModelInfo], dict[str, dict[str, Any]], dict[str, list[str]]]
            (models, ab_tests, deployments)

        """
        if self.backend == BackendType.JSON:
            return self._load_from_json()
        elif self.backend == BackendType.POSTGRES:
            return self._load_from_postgres()
        else:
            logger.warning("Unknown backend type: %s", self.backend)
            return {}, {}, {}

    def _load_from_json(self) -> tuple[
        dict[str, ModelInfo],
        dict[str, dict[str, Any]],
        dict[str, list[str]],
    ]:
        """
        Load registry from JSON file.
        """
        if self.registry_file.exists():
            data = self.persistence.load_json("registry.json")
            if data is not None:
                models: dict[str, ModelInfo] = {
                    model_id: self._dict_to_model_info(model_data)
                    for model_id, model_data in data.get("models", {}).items()
                }
                ab_tests: dict[str, dict[str, Any]] = data.get("ab_tests", {})
                deployments: dict[str, list[str]] = data.get("deployments", {})
                return models, ab_tests, deployments

        return {}, {}, {}

    def _load_from_postgres(self) -> tuple[
        dict[str, ModelInfo],
        dict[str, dict[str, Any]],
        dict[str, list[str]],
    ]:
        """
        Load registry from PostgreSQL.
        """
        session = self.persistence.get_session()
        if session is None:
            return {}, {}, {}

        try:
            models: dict[str, ModelInfo] = {}
            ab_tests: dict[str, dict[str, Any]] = {}
            deployments: dict[str, list[str]] = {}

            model_records = session.query(ModelTable).all()
            for model in model_records:
                model_info = self._db_to_model_info(model)
                models[model_info.manifest.model_id] = model_info

                # Reconstruct deployments
                for target in cast(list[str], model.deployed_to) or []:
                    if target not in deployments:
                        deployments[target] = []
                    deployments[target].append(model_info.manifest.model_id)

            return models, ab_tests, deployments
        except Exception:
            logger.warning(
                "Error loading from database. Starting with empty registry.",
                exc_info=True,
            )
            return {}, {}, {}
        finally:
            session.close()

    def save_registry(
        self,
        models: dict[str, ModelInfo],
        ab_tests: dict[str, dict[str, Any]],
        deployments: dict[str, list[str]],
        immediate: bool = False,
    ) -> None:
        """
        Save registry with optional batching.

        Parameters
        ----------
        models : dict[str, ModelInfo]
            Models to save
        ab_tests : dict[str, dict[str, Any]]
            A/B tests to save
        deployments : dict[str, list[str]]
            Deployments to save
        immediate : bool
            If True, save immediately

        """
        with self._lock:
            if immediate:
                # Cancel any pending batch save
                if self._save_timer is not None:
                    self._save_timer.cancel()
                    self._save_timer = None
                self._pending_save = False

                # Save immediately
                self._do_save(models, ab_tests, deployments)
            else:
                # Store pending data
                self._pending_data = (models, ab_tests, deployments)

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

    def _do_save(
        self,
        models: dict[str, ModelInfo],
        ab_tests: dict[str, dict[str, Any]],
        deployments: dict[str, list[str]],
    ) -> None:
        """
        Perform the actual save to disk.
        """
        if self.backend == BackendType.JSON:
            self._save_to_json(models, ab_tests, deployments)
        elif self.backend == BackendType.POSTGRES:
            self._save_to_postgres(models)

    def _save_to_json(
        self,
        models: dict[str, ModelInfo],
        ab_tests: dict[str, dict[str, Any]],
        deployments: dict[str, list[str]],
    ) -> None:
        """
        Save registry to JSON file.
        """
        try:
            data = {
                "models": {
                    model_id: self._model_info_to_dict(model_info)
                    for model_id, model_info in models.items()
                },
                "ab_tests": ab_tests,
                "deployments": deployments,
                "last_updated": time.time(),
            }

            # Ensure directory exists
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug("Registry saved with %d models", len(models))
        except Exception:
            logger.error("Failed to save registry", exc_info=True)
            raise

    def _save_to_postgres(self, models: dict[str, ModelInfo]) -> None:
        """
        Save all models to PostgreSQL.
        """
        for model_info in models.values():
            self._save_model_to_db(model_info)

    def _flush_batch_save(self) -> None:
        """
        Flush pending batch saves.
        """
        with self._lock:
            if self._pending_save and self._pending_data is not None:
                try:
                    models, ab_tests, deployments = self._pending_data
                    self._do_save(models, ab_tests, deployments)
                except FileNotFoundError as exc:
                    logger.debug(
                        "Batch save flush: registry path missing (ignored): %s",
                        exc,
                        exc_info=False,
                    )
                except Exception:
                    logger.error("Error during batch save flush", exc_info=True)
                finally:
                    self._pending_save = False
                    self._save_timer = None
                    self._pending_data = None

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
                training_config=cast(dict[str, Any], manifest_data.get("training_config", {})) or {},
                performance_metrics=cast(dict[str, float], manifest_data.get("performance_metrics", {})) or {},
                deployment_constraints=cast(dict[str, Any], manifest_data.get("deployment_constraints", {})) or {},
                version=cast(str, manifest_data["version"]),
                created_at=float(manifest_data["created_at"]),
                last_modified=float(manifest_data["last_modified"]),
                serveable=bool(manifest_data.get("serveable", True)),
                artifact_format=str(manifest_data.get("artifact_format", "onnx")),
                feature_set_id=cast(str | None, manifest_data.get("feature_set_id")),
                pipeline_signature=cast(str | None, manifest_data.get("pipeline_signature")),
                pipeline_version=cast(str | None, manifest_data.get("pipeline_version")),
                decision_policy=cast(str | None, manifest_data.get("decision_policy")),
                decision_config=cast(dict[str, Any], manifest_data.get("decision_config", {})) or {},
                artifact_sha256_digest=cast(str | None, manifest_data.get("artifact_sha256_digest")),
            )
        else:
            # Legacy format - use defaults
            manifest = ModelManifest(
                model_id=str(data["model_id"]),
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
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
            performance_history=cast(list[dict[str, Any]], data.get("performance_history", [])) or [],
            metadata=cast(dict[str, Any], data.get("metadata", {})) or {},
        )

    def _db_to_model_info(self, db_model: ModelTable) -> ModelInfo:
        """
        Convert database model to ModelInfo.
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
        """
        session = self.persistence.get_session()
        if session is None:
            return

        try:
            # Check if model exists
            existing = (
                session.query(ModelTable).filter_by(model_id=model_info.manifest.model_id).first()
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
                existing.deployment_constraints = cast(Any, model_info.manifest.deployment_constraints)
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
            logger.error("Failed to save model to database", exc_info=True)
            raise
        finally:
            session.close()

    def calculate_file_sha256(self, file_path: Path) -> str:
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
        OSError
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

    def verify_artifact_integrity(self, file_path: Path, expected_digest: str | None) -> None:
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
        OSError
            If the file cannot be read

        """
        if expected_digest is None:
            logger.warning(
                "No SHA-256 digest available for %s, skipping integrity verification",
                file_path.name,
            )
            return

        if not expected_digest:
            logger.warning(
                "Empty SHA-256 digest for %s, skipping integrity verification",
                file_path.name,
            )
            return

        try:
            actual_digest = self.calculate_file_sha256(file_path)
        except (OSError, FileNotFoundError) as e:
            raise ValueError(f"Cannot verify artifact integrity: {e}") from e

        if actual_digest != expected_digest:
            # Security: Log the verification failure for auditing
            logger.error(
                "SECURITY ALERT: Artifact integrity verification failed for %s\n"
                "Expected SHA-256: %s\n"
                "Actual SHA-256:   %s\n"
                "This indicates the model artifact may have been tampered with!",
                file_path,
                expected_digest,
                actual_digest,
            )
            raise ValueError(
                f"Artifact integrity verification failed for {file_path.name}. "
                f"Expected digest: {expected_digest[:16]}..., "
                f"but got: {actual_digest[:16]}... "
                f"The model artifact may have been tampered with and is rejected for security.",
            )

        logger.debug(
            "Artifact integrity verified for %s: %s...", file_path.name, actual_digest[:16]
        )

    def load_model(self, model_id: str, model_info: ModelInfo) -> object | None:
        """
        Load model from cache or disk with integrity verification.

        SECURITY: Only loads ONNX models to prevent code execution vulnerabilities.

        Parameters
        ----------
        model_id : str
            Model ID to load
        model_info : ModelInfo
            Model information

        Returns
        -------
        object | None
            Loaded ONNX InferenceSession or None

        """
        with self._lock:
            # Check cache first
            if model_id in self._model_cache:
                self._cache_access_times[model_id] = time.time()
                return cast(object, self._model_cache[model_id])

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
                    self.verify_artifact_integrity(model_path, expected_digest)

                    # Load ONNX model
                    if not HAS_ONNX:
                        check_ml_dependencies(["onnxruntime"])

                    # Create optimized session via helper
                    from ml.config.runtime import to_session_options

                    session_options, providers = to_session_options(self._onnx_rt)

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
                    # Non-serveable artifact format - do not load for security
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

                return cast(object, model)

            except Exception as exc:
                # Propagate integrity failures to satisfy security contract tests
                if isinstance(exc, ValueError):
                    raise
                logger.error("Failed to load model %s", model_id, exc_info=True)
                return None

    def get_artifact_path(self, model_id: str, model_info: ModelInfo) -> Path | None:
        """
        Return the model artifact path if present and within registry root.

        Parameters
        ----------
        model_id : str
            Model ID
        model_info : ModelInfo
            Model information

        Returns
        -------
        Path | None
            Artifact path or None if invalid/missing

        """
        model_path = model_info.model_path
        if not self._validate_model_path(model_path):
            logger.error("Security: Invalid model path detected: %s", model_path)
            return None
        return model_path if model_path.exists() else None

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

    def flush(self) -> None:
        """
        Flush any pending batch saves immediately.

        Call this before shutdown or when immediate persistence is needed.

        """
        with self._lock:
            if self._pending_save and self._pending_data is not None:
                if self._save_timer is not None:
                    self._save_timer.cancel()
                    self._save_timer = None

                # Execute the pending save
                try:
                    models, ab_tests, deployments = self._pending_data
                    self._do_save(models, ab_tests, deployments)
                except Exception:
                    logger.error("Error during flush", exc_info=True)
                finally:
                    self._pending_save = False
                    self._pending_data = None
                    logger.debug("Flushed pending batch saves")

    def __del__(self) -> None:
        """
        Ensure pending saves are flushed on cleanup.
        """
        try:
            self.flush()
        except Exception as exc:
            logger.debug("ModelPersistence cleanup flush failed", exc_info=exc)
