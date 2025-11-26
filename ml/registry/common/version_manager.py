#!/usr/bin/env python3

"""
VersionManagerComponent - Handles model versioning, compatibility, and lineage tracking.

This component is extracted from the ModelRegistry god class following the
established TDD decomposition pattern. It handles:
- Auto-versioning of model manifests
- Schema compatibility filtering (list_compatible)
- Latest version resolution (resolve_latest)
- Model lineage tracking (get_model_lineage)

Thread-safety: Uses the persistence component's lock for all operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ml.config.constants import Versions
from ml.registry.base import ModelInfo
from ml.registry.base import ModelRole


if TYPE_CHECKING:
    from ml.registry.base import ModelManifest
    from ml.registry.common.model_persistence import ModelPersistenceComponent


logger = logging.getLogger(__name__)


class VersionManagerComponent:
    """
    Manages model versioning, compatibility, and lineage tracking.

    This component provides:
    - Auto-versioning: Automatically assigns semantic versions to manifests
    - Compatibility filtering: Lists models compatible with given feature schema
    - Latest resolution: Resolves the latest model matching criteria
    - Lineage tracking: Traces parent-child relationships between models

    Attributes
    ----------
    _persistence : ModelPersistenceComponent
        The persistence component for model storage operations.

    Thread Safety
    -------------
    All operations use the persistence component's lock for thread-safe access.

    Example
    -------
    >>> from ml.registry.common import ModelPersistenceComponent, VersionManagerComponent
    >>> persistence = ModelPersistenceComponent(config, path)
    >>> version_mgr = VersionManagerComponent(persistence)
    >>> version_mgr.auto_version_manifest(manifest)  # Sets manifest.version
    >>> latest = version_mgr.resolve_latest(ModelRole.INFERENCE, "XGBoost", "hash123")
    """

    def __init__(self, persistence: ModelPersistenceComponent) -> None:
        """
        Initialize version manager component.

        Parameters
        ----------
        persistence : ModelPersistenceComponent
            The persistence component providing model storage operations.
        """
        self._persistence = persistence
        logger.debug("Initialized VersionManagerComponent")

    # -------------------------------------------------------------------------
    # Auto Versioning
    # -------------------------------------------------------------------------

    def auto_version_manifest(self, manifest: ModelManifest) -> None:
        """
        Auto-assign semantic version to manifest if missing.

        Increments patch version from latest model with same architecture,
        or assigns "1.0.0" if first model of that architecture.

        Parameters
        ----------
        manifest : ModelManifest
            The model manifest to version. Modified in place if version is empty.

        Example
        -------
        >>> manifest = ModelManifest(model_id="m1", version="", architecture="XGBoost", ...)
        >>> version_mgr.auto_version_manifest(manifest)
        >>> print(manifest.version)  # "1.0.0" or "X.Y.Z+1"
        """
        # Skip if version already set
        if manifest.version:
            return

        with self._persistence._lock:
            # Find existing versions for same architecture
            existing_versions = [
                m.manifest.version
                for m in self._persistence.models.values()
                if m.manifest.architecture == manifest.architecture
            ]

            if existing_versions:
                # Increment patch version from latest
                latest = max(existing_versions)
                major, minor, patch = latest.split(".")
                manifest.version = f"{major}.{minor}.{int(patch) + 1}"
                logger.debug(
                    "Auto-versioned manifest %s to %s (incremented from %s)",
                    manifest.model_id,
                    manifest.version,
                    latest,
                )
            else:
                # First model of this architecture
                manifest.version = Versions.DEFAULT_MANIFEST_VERSION
                logger.debug(
                    "Auto-versioned manifest %s to %s (first of architecture %s)",
                    manifest.model_id,
                    manifest.version,
                    manifest.architecture,
                )

    # -------------------------------------------------------------------------
    # Compatibility Filtering
    # -------------------------------------------------------------------------

    def list_compatible(
        self,
        schema_hash: str,
        role: ModelRole | None = None,
        architecture: str | None = None,
    ) -> list[ModelInfo]:
        """
        List models compatible with given feature schema hash.

        Parameters
        ----------
        schema_hash : str
            Feature schema hash to match against.
        role : ModelRole | None
            Optional role filter (INFERENCE, STUDENT, TEACHER, etc.).
        architecture : str | None
            Optional architecture filter (XGBoost, LightGBM, etc.).

        Returns
        -------
        list[ModelInfo]
            List of models matching the compatibility criteria.

        Example
        -------
        >>> compatible = version_mgr.list_compatible(
        ...     schema_hash="abc123",
        ...     role=ModelRole.INFERENCE,
        ...     architecture="XGBoost",
        ... )
        >>> for model in compatible:
        ...     print(f"{model.manifest.model_id}: v{model.manifest.version}")
        """
        with self._persistence._lock:
            result = [
                m
                for m in self._persistence.models.values()
                if m.manifest.feature_schema_hash == schema_hash
                and (role is None or m.manifest.role == role)
                and (architecture is None or m.manifest.architecture == architecture)
            ]
            logger.debug(
                "list_compatible: schema_hash=%s, role=%s, architecture=%s -> %d models",
                schema_hash,
                role.value if role else None,
                architecture,
                len(result),
            )
            return result

    # -------------------------------------------------------------------------
    # Latest Resolution
    # -------------------------------------------------------------------------

    def resolve_latest(
        self,
        role: ModelRole,
        architecture: str,
        schema_hash: str,
    ) -> ModelInfo | None:
        """
        Resolve the latest model by version matching criteria.

        Uses lexical comparison of semantic version strings to determine
        the latest version.

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
            The model with highest version matching criteria, or None if no match.

        Example
        -------
        >>> latest = version_mgr.resolve_latest(
        ...     role=ModelRole.INFERENCE,
        ...     architecture="XGBoost",
        ...     schema_hash="abc123",
        ... )
        >>> if latest:
        ...     print(f"Latest: {latest.manifest.model_id} v{latest.manifest.version}")
        """
        candidates = self.list_compatible(
            schema_hash=schema_hash,
            role=role,
            architecture=architecture,
        )

        if not candidates:
            logger.debug(
                "resolve_latest: No candidates for role=%s, arch=%s, hash=%s",
                role.value,
                architecture,
                schema_hash,
            )
            return None

        latest = max(candidates, key=lambda m: m.manifest.version)
        logger.debug(
            "resolve_latest: Resolved %s v%s from %d candidates",
            latest.manifest.model_id,
            latest.manifest.version,
            len(candidates),
        )
        return latest

    # -------------------------------------------------------------------------
    # Lineage Tracking
    # -------------------------------------------------------------------------

    def get_model_lineage(self, model_id: str) -> list[ModelInfo]:
        """
        Get complete lineage (parents and children) of a model.

        Traces the parent chain backwards to find ancestors, then includes
        the model itself and its direct children.

        The lineage is ordered as: [ancestors...], model, [children...]

        Parameters
        ----------
        model_id : str
            The model ID to get lineage for.

        Returns
        -------
        list[ModelInfo]
            Ordered list of models in the lineage. Empty if model not found.

        Example
        -------
        >>> lineage = version_mgr.get_model_lineage("student_model_v3")
        >>> for model in lineage:
        ...     print(f"{model.manifest.model_id} (parent: {model.manifest.parent_id})")
        """
        with self._persistence._lock:
            if model_id not in self._persistence.models:
                logger.debug("get_model_lineage: Model %s not found", model_id)
                return []

            lineage: list[ModelInfo] = []
            model = self._persistence.models[model_id]

            # Trace parents (ancestors) - insert at beginning to maintain order
            current_id = model.manifest.parent_id
            visited: set[str] = set()  # Cycle detection
            while current_id and current_id in self._persistence.models:
                # Guard against circular references
                if current_id in visited:
                    logger.warning(
                        "Circular parent reference detected in lineage for %s at %s",
                        model_id,
                        current_id,
                    )
                    break
                visited.add(current_id)

                parent = self._persistence.models[current_id]
                lineage.insert(0, parent)  # Add to beginning
                current_id = parent.manifest.parent_id

            # Add the model itself
            lineage.append(model)

            # Add children (only direct children)
            for child_id in model.manifest.children_ids:
                if child_id in self._persistence.models:
                    lineage.append(self._persistence.models[child_id])

            logger.debug(
                "get_model_lineage: %s -> %d models in lineage",
                model_id,
                len(lineage),
            )
            return lineage
