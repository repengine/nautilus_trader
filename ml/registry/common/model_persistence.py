#!/usr/bin/env python3

"""
ModelPersistenceComponent - Handles persistence operations for model registry.

This component is extracted from the ModelRegistry god class following the
established TDD decomposition pattern. It handles:
- JSON/PostgreSQL persistence
- Serialization/deserialization of ModelInfo
- Batch save management with timers
- SHA-256 integrity verification for model artifacts

Thread-safety: All operations are protected by an RLock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, cast

from ml.config.constants import Versions
from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.persistence import BackendType
from ml.registry.persistence import ModelTable
from ml.registry.persistence import PersistenceConfig
from ml.registry.persistence import PersistenceManager


logger = logging.getLogger(__name__)


class ModelPersistenceComponent:
    """
    Handles persistence operations for model registry.

    This component manages the storage and retrieval of model metadata
    using either JSON files or PostgreSQL as the backend.

    Attributes
    ----------
    persistence : PersistenceManager
        The persistence manager for backend operations.
    registry_path : Path
        The directory path for registry storage.
    batch_save_interval : float
        Seconds to wait before flushing batch saves.

    Thread Safety
    -------------
    All public methods are protected by an RLock for thread-safe concurrent access.

    Example
    -------
    >>> config = PersistenceConfig(backend=BackendType.JSON, json_path=Path("/tmp/registry"))
    >>> component = ModelPersistenceComponent(config, Path("/tmp/registry"))
    >>> component.load_registry()
    >>> component.set_model("model_1", model_info)
    >>> component.save_registry(immediate=True)
    """

    def __init__(
        self,
        persistence_config: PersistenceConfig,
        registry_path: Path,
        batch_save_interval: float = 0.1,
    ) -> None:
        """
        Initialize persistence component.

        Parameters
        ----------
        persistence_config : PersistenceConfig
            Configuration for persistence backend (JSON or PostgreSQL).
        registry_path : Path
            Directory path for registry storage.
        batch_save_interval : float
            Seconds to wait before flushing batch saves (default 0.1s).

        Raises
        ------
        ValueError
            If persistence_config is invalid.
        """
        self.persistence = PersistenceManager(persistence_config)
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.batch_save_interval = batch_save_interval
        self.registry_file = self.registry_path / "registry.json"

        # Thread safety
        self._lock = threading.RLock()

        # In-memory state
        self._models: dict[str, ModelInfo] = {}
        self._ab_tests: dict[str, dict[str, Any]] = {}
        self._deployments: dict[str, list[str]] = {}

        # Batch save management
        self._pending_save = False
        self._save_timer: threading.Timer | None = None

        logger.debug(
            "Initialized ModelPersistenceComponent at %s with backend=%s",
            registry_path,
            self.backend.value,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def models(self) -> dict[str, ModelInfo]:
        """
        Get the models dictionary.

        Returns a reference to the internal models dict.
        Callers should acquire the lock for thread-safe iteration.

        Returns
        -------
        dict[str, ModelInfo]
            Dictionary mapping model_id to ModelInfo.
        """
        return self._models

    @property
    def ab_tests(self) -> dict[str, dict[str, Any]]:
        """
        Get the A/B tests dictionary.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary of A/B test configurations.
        """
        return self._ab_tests

    @property
    def deployments(self) -> dict[str, list[str]]:
        """
        Get the deployments dictionary.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping deployment targets to model IDs.
        """
        return self._deployments

    @property
    def backend(self) -> BackendType:
        """
        Get the persistence backend type.

        Returns
        -------
        BackendType
            The backend type (JSON or POSTGRES).
        """
        return self.persistence.config.backend

    # -------------------------------------------------------------------------
    # Registry Loading
    # -------------------------------------------------------------------------

    def load_registry(self) -> None:
        """
        Load registry from persistence backend or create new one.

        For JSON backend, loads from registry.json file if it exists,
        otherwise initializes empty state.

        For PostgreSQL backend, loads all models from the database.

        Thread-safe: Acquires lock during loading.

        Example
        -------
        >>> component.load_registry()
        >>> print(len(component.models))
        0
        """
        with self._lock:
            if self.backend == BackendType.JSON:
                self._load_from_json()
            elif self.backend == BackendType.POSTGRES:
                self._load_from_postgres()

    def _load_from_json(self) -> None:
        """Load registry state from JSON file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)

                self._models = {
                    model_id: self.dict_to_model_info(model_data)
                    for model_id, model_data in data.get("models", {}).items()
                }
                self._ab_tests = data.get("ab_tests", {})
                self._deployments = data.get("deployments", {})

                logger.debug(
                    "Loaded registry from JSON with %d models",
                    len(self._models),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Failed to load registry from JSON, starting fresh: %s",
                    exc,
                    exc_info=True,
                )
                self._models = {}
                self._ab_tests = {}
                self._deployments = {}
        else:
            self._models = {}
            self._ab_tests = {}
            self._deployments = {}
            logger.debug("No existing registry file, initialized empty state")

    def _load_from_postgres(self) -> None:
        """Load registry state from PostgreSQL database."""
        session = self.persistence.get_session()
        if session is None:
            self._models = {}
            self._ab_tests = {}
            self._deployments = {}
            return

        try:
            self._models = {}
            self._ab_tests = {}
            self._deployments = {}

            models = session.query(ModelTable).all()
            for model in models:
                model_info = self.db_to_model_info(model)
                self._models[model_info.manifest.model_id] = model_info

                # Reconstruct deployments
                for target in cast(list[str], model.deployed_to) or []:
                    if target not in self._deployments:
                        self._deployments[target] = []
                    self._deployments[target].append(model_info.manifest.model_id)

            logger.debug(
                "Loaded registry from PostgreSQL with %d models",
                len(self._models),
            )
        except Exception as exc:
            logger.warning(
                "Error loading from database. Starting with empty registry: %s",
                exc,
                exc_info=True,
            )
            self._models = {}
            self._ab_tests = {}
            self._deployments = {}
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Registry Saving
    # -------------------------------------------------------------------------

    def save_registry(self, immediate: bool = False) -> None:
        """
        Save registry to disk with optional batching.

        Parameters
        ----------
        immediate : bool
            If True, save immediately. If False, batch the save.

        Raises
        ------
        OSError
            If file write fails (only for immediate=True).

        Example
        -------
        >>> component.set_model("model_1", model_info)
        >>> component.save_registry(immediate=True)  # Save now
        >>> component.save_registry(immediate=False)  # Batch save
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

        Raises
        ------
        OSError
            If file write fails.
        """
        try:
            data = {
                "models": {
                    model_id: self.model_info_to_dict(model_info)
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

            logger.debug("Registry saved with %d models", len(self._models))
        except Exception as exc:
            logger.error(
                "Failed to save registry: %s",
                exc,
                exc_info=True,
            )
            raise

    def _flush_batch_save(self) -> None:
        """Flush pending batch saves."""
        with self._lock:
            if self._pending_save:
                try:
                    self._do_save()
                except FileNotFoundError as exc:
                    # Directory may have been deleted during cleanup
                    logger.debug(
                        "Batch save flush: registry path missing (ignored): %s",
                        exc,
                    )
                except Exception as exc:
                    logger.error(
                        "Error during batch save flush: %s",
                        exc,
                        exc_info=True,
                    )
                finally:
                    self._pending_save = False
                    self._save_timer = None

    def flush(self) -> None:
        """
        Flush any pending batch saves immediately.

        Call this before shutdown or when immediate persistence is needed.

        Example
        -------
        >>> component.set_model("model_1", model_info)
        >>> component.save_registry(immediate=False)  # Scheduled
        >>> component.flush()  # Force write now
        """
        with self._lock:
            if self._pending_save:
                if self._save_timer is not None:
                    self._save_timer.cancel()
                    self._save_timer = None
                try:
                    self._do_save()
                except FileNotFoundError:
                    # Directory may have been deleted during cleanup
                    logger.debug("Flush: registry path missing (ignored)")
                except Exception as exc:
                    logger.error(
                        "Error during flush: %s",
                        exc,
                        exc_info=True,
                    )
                finally:
                    self._pending_save = False
                logger.debug("Flushed pending batch saves")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def model_info_to_dict(self, model_info: ModelInfo) -> dict[str, Any]:
        """
        Convert ModelInfo to dictionary for JSON serialization.

        Parameters
        ----------
        model_info : ModelInfo
            Model information to serialize.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON.

        Example
        -------
        >>> data = component.model_info_to_dict(model_info)
        >>> assert "manifest" in data
        >>> assert data["manifest"]["model_id"] == model_info.manifest.model_id
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
            "output_schema": getattr(model_info.manifest, "output_schema", None),
            "calibration": getattr(model_info.manifest, "calibration", None),
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
            "artifact_sha256_digest": getattr(
                model_info.manifest, "artifact_sha256_digest", None
            ),
        }

        return {
            "manifest": manifest_dict,
            "model_path": str(model_info.model_path),
            "deployment_status": model_info.deployment_status.value,
            "deployed_to": model_info.deployed_to,
            "performance_history": model_info.performance_history,
            "metadata": model_info.metadata,
        }

    def dict_to_model_info(self, data: dict[str, Any]) -> ModelInfo:
        """
        Convert dictionary to ModelInfo.

        Handles both new format (with "manifest" key) and legacy format
        (flat structure without manifest wrapper).

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary to deserialize.

        Returns
        -------
        ModelInfo
            Deserialized model information.

        Example
        -------
        >>> data = {"manifest": {...}, "model_path": "/path/to/model.onnx", ...}
        >>> model_info = component.dict_to_model_info(data)
        >>> assert isinstance(model_info, ModelInfo)
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
                output_schema=manifest_data.get("output_schema"),
                calibration=manifest_data.get("calibration"),
                version=manifest_data["version"],
                created_at=manifest_data["created_at"],
                last_modified=manifest_data["last_modified"],
                serveable=manifest_data.get("serveable", True),
                artifact_format=manifest_data.get("artifact_format", "onnx"),
                feature_set_id=manifest_data.get("feature_set_id"),
                pipeline_signature=manifest_data.get("pipeline_signature"),
                pipeline_version=manifest_data.get("pipeline_version"),
                decision_policy=manifest_data.get("decision_policy"),
                decision_config=manifest_data.get("decision_config", {}),
                artifact_sha256_digest=manifest_data.get("artifact_sha256_digest"),
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

    def db_to_model_info(self, db_model: ModelTable) -> ModelInfo:
        """
        Convert database model to ModelInfo.

        Parameters
        ----------
        db_model : ModelTable
            Database model record.

        Returns
        -------
        ModelInfo
            Model information object.

        Example
        -------
        >>> session = component.persistence.get_session()
        >>> db_model = session.query(ModelTable).first()
        >>> model_info = component.db_to_model_info(db_model)
        """
        # Extract values with proper null handling
        model_id = cast(str, db_model.model_id) or ""
        role_str = cast(str, db_model.role)
        data_req_str = cast(str, db_model.data_requirements)
        architecture = cast(str, db_model.architecture) or "unknown"
        feature_schema_hash = cast(str, db_model.feature_schema_hash) or ""
        version = cast(str, db_model.version) or Versions.DEFAULT_MANIFEST_VERSION
        model_path = cast(str, db_model.model_path) or ""
        deployment_status_str = cast(str, db_model.deployment_status)

        manifest = ModelManifest(
            model_id=model_id,
            role=ModelRole(role_str),
            data_requirements=DataRequirements(data_req_str),
            architecture=architecture,
            feature_schema=cast(dict[str, str], db_model.feature_schema) or {},
            feature_schema_hash=feature_schema_hash,
            parent_id=db_model.parent_id,
            children_ids=cast(list[str], db_model.children_ids) or [],
            training_config=cast(dict[str, Any], db_model.training_config) or {},
            performance_metrics=cast(dict[str, float], db_model.performance_metrics) or {},
            deployment_constraints=cast(dict[str, Any], db_model.deployment_constraints) or {},
            output_schema=cast(dict[str, Any] | None, db_model.output_schema),
            calibration=cast(dict[str, Any] | None, db_model.calibration),
            version=version,
            created_at=(
                db_model.created_at.timestamp() if db_model.created_at else time.time()
            ),
            last_modified=(
                db_model.last_modified.timestamp()
                if db_model.last_modified
                else time.time()
            ),
            serveable=db_model.serveable == "true" if db_model.serveable else True,
            artifact_format=db_model.artifact_format if db_model.artifact_format else "onnx",
            feature_set_id=db_model.feature_set_id,
            pipeline_signature=db_model.pipeline_signature,
            pipeline_version=db_model.pipeline_version,
            artifact_sha256_digest=db_model.artifact_sha256_digest,
        )

        return ModelInfo(
            manifest=manifest,
            model_path=Path(model_path),
            deployment_status=DeploymentStatus(deployment_status_str),
            deployed_to=cast(list[str], db_model.deployed_to) or [],
            performance_history=cast(list[dict[str, Any]], db_model.performance_history)
            or [],
            metadata=cast(dict[str, Any], db_model.extra_metadata) or {},
        )

    # -------------------------------------------------------------------------
    # Database Operations
    # -------------------------------------------------------------------------

    def save_model_to_db(self, model_info: ModelInfo) -> None:
        """
        Save model to PostgreSQL database.

        Parameters
        ----------
        model_info : ModelInfo
            Model information to save.

        Raises
        ------
        Exception
            If database operation fails.

        Example
        -------
        >>> component.save_model_to_db(model_info)
        """
        session = self.persistence.get_session()
        if session is None:
            return

        try:
            # Check if model exists
            existing = (
                session.query(ModelTable)
                .filter_by(model_id=model_info.manifest.model_id)
                .first()
            )

            if existing:
                # Update existing model
                # SQLAlchemy accepts direct assignment to Column attributes at runtime
                existing.role = model_info.manifest.role.value
                existing.data_requirements = model_info.manifest.data_requirements.value
                existing.architecture = model_info.manifest.architecture
                existing.feature_schema = model_info.manifest.feature_schema
                existing.feature_schema_hash = model_info.manifest.feature_schema_hash
                existing.parent_id = model_info.manifest.parent_id
                existing.children_ids = model_info.manifest.children_ids  # type: ignore[assignment]
                existing.training_config = model_info.manifest.training_config  # type: ignore[assignment]
                existing.performance_metrics = model_info.manifest.performance_metrics  # type: ignore[assignment]
                existing.deployment_constraints = model_info.manifest.deployment_constraints  # type: ignore[assignment]
                existing.output_schema = model_info.manifest.output_schema  # type: ignore[assignment]
                existing.calibration = model_info.manifest.calibration  # type: ignore[assignment]
                existing.deployment_status = model_info.deployment_status.value  # type: ignore[assignment]
                existing.deployed_to = model_info.deployed_to  # type: ignore[assignment]
                existing.version = model_info.manifest.version
                existing.extra_metadata = model_info.metadata
                existing.model_path = str(model_info.model_path)
                existing.performance_history = model_info.performance_history
                existing.serveable = "true" if model_info.manifest.serveable else "false"
                existing.artifact_format = model_info.manifest.artifact_format
                existing.feature_set_id = model_info.manifest.feature_set_id
                existing.pipeline_signature = model_info.manifest.pipeline_signature
                existing.pipeline_version = model_info.manifest.pipeline_version
                existing.artifact_sha256_digest = model_info.manifest.artifact_sha256_digest
            else:
                # Create new model - SQLAlchemy ORM handles type conversion
                new_model = ModelTable(
                    model_id=model_info.manifest.model_id,
                    role=model_info.manifest.role.value,
                    data_requirements=model_info.manifest.data_requirements.value,
                    architecture=model_info.manifest.architecture,
                    feature_schema=model_info.manifest.feature_schema,
                    feature_schema_hash=model_info.manifest.feature_schema_hash,
                    parent_id=model_info.manifest.parent_id,
                    children_ids=cast(Any, model_info.manifest.children_ids),
                    training_config=cast(Any, model_info.manifest.training_config),
                    performance_metrics=cast(Any, model_info.manifest.performance_metrics),
                    deployment_constraints=cast(Any, model_info.manifest.deployment_constraints),
                    output_schema=cast(Any, model_info.manifest.output_schema),
                    calibration=cast(Any, model_info.manifest.calibration),
                    deployment_status=model_info.deployment_status.value,  # type: ignore[arg-type]
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
            logger.debug("Saved model %s to database", model_info.manifest.model_id)
        except Exception as exc:
            session.rollback()
            logger.error(
                "Failed to save model to database: %s",
                exc,
                exc_info=True,
            )
            raise
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Integrity Verification
    # -------------------------------------------------------------------------

    def calculate_file_sha256(self, file_path: Path) -> str:
        """
        Calculate SHA-256 digest of a file for integrity verification.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        str
            Hexadecimal SHA-256 digest of the file (64 characters).

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        OSError
            If the file cannot be read.

        Example
        -------
        >>> digest = component.calculate_file_sha256(Path("/path/to/model.onnx"))
        >>> assert len(digest) == 64  # SHA-256 produces 64 hex chars
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read file in 8KB chunks to handle large models efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
        except OSError as e:
            raise OSError(f"Failed to read file {file_path}: {e}") from e

        return sha256_hash.hexdigest()

    def verify_artifact_integrity(
        self,
        file_path: Path,
        expected_digest: str | None,
    ) -> None:
        """
        Verify artifact integrity using SHA-256 digest.

        Parameters
        ----------
        file_path : Path
            Path to the artifact file.
        expected_digest : str | None
            Expected SHA-256 digest. If None, skip verification with warning.

        Raises
        ------
        ValueError
            If digest verification fails or artifact is tampered.

        Example
        -------
        >>> component.verify_artifact_integrity(model_path, expected_digest)
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
            "Artifact integrity verified for %s: %s...",
            file_path.name,
            actual_digest[:16],
        )

    # -------------------------------------------------------------------------
    # Model CRUD Operations
    # -------------------------------------------------------------------------

    def get_model(self, model_id: str) -> ModelInfo | None:
        """
        Get model by ID.

        Parameters
        ----------
        model_id : str
            The model identifier.

        Returns
        -------
        ModelInfo | None
            The model info if found, None otherwise.

        Example
        -------
        >>> model_info = component.get_model("model_123")
        >>> if model_info:
        ...     print(model_info.manifest.version)
        """
        with self._lock:
            return self._models.get(model_id)

    def set_model(self, model_id: str, model_info: ModelInfo) -> None:
        """
        Set model in the registry.

        Parameters
        ----------
        model_id : str
            The model identifier.
        model_info : ModelInfo
            The model information to store.

        Example
        -------
        >>> component.set_model("model_123", model_info)
        >>> component.save_registry(immediate=True)
        """
        with self._lock:
            self._models[model_id] = model_info

    def delete_model(self, model_id: str) -> bool:
        """
        Delete model from the registry.

        Parameters
        ----------
        model_id : str
            The model identifier.

        Returns
        -------
        bool
            True if model was deleted, False if not found.

        Example
        -------
        >>> success = component.delete_model("model_123")
        >>> if success:
        ...     component.save_registry(immediate=True)
        """
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def __del__(self) -> None:
        """Ensure pending saves are flushed on cleanup."""
        try:
            self.flush()
        except Exception as exc:
            logger.debug("ModelPersistenceComponent cleanup flush failed: %s", exc)
