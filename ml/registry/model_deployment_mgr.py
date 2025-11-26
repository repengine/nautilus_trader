#!/usr/bin/env python3

"""
Model deployment management and lifecycle tracking (DEPRECATED).

.. deprecated::
    This is the legacy implementation. Use package-level imports instead:

        # Old (deprecated):
        from ml.registry.model_deployment_mgr import ModelDeploymentManager

        # New (preferred):
        from ml.registry import DeploymentManagerComponent

    The canonical implementation is :class:`ml.registry.common.deployment_manager.DeploymentManagerComponent`
    which provides thread-safety via RLock and uses the persistence component pattern.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelRole


logger = logging.getLogger(__name__)


class ModelDeploymentManagerProtocol(Protocol):
    """
    Protocol for model deployment operations.
    """

    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deploy a model to a target.
        """
        ...

    def rollback(self, target: str, to_model_id: str) -> bool:
        """
        Rollback to a previous model version.
        """
        ...

    def retire_model(self, model_id: str) -> bool:
        """
        Retire a model from production.
        """
        ...

    def hot_reload_model(self, target: str, new_model_id: str) -> bool:
        """
        Hot reload a deployment with a new model.
        """
        ...

    def get_active_models(self) -> list[ModelInfo]:
        """
        Get all currently deployed models.
        """
        ...

    def get_all_models(self) -> list[ModelInfo]:
        """
        Get all registered models.
        """
        ...

    def get_model(self, model_id: str) -> ModelInfo | None:
        """
        Get information about a specific model.
        """
        ...

    def get_models_by_role(self, role: ModelRole) -> list[ModelInfo]:
        """
        Get all models with a specific role.
        """
        ...

    def get_models_by_data_requirements(
        self,
        requirements: DataRequirements,
    ) -> list[ModelInfo]:
        """
        Get all models with specific data requirements.
        """
        ...

    def get_model_lineage(self, model_id: str) -> list[ModelInfo]:
        """
        Get complete lineage of a model (parents and children).
        """
        ...

    def track_performance(
        self,
        model_id: str,
        metrics: dict[str, Any],
    ) -> None:
        """
        Track model performance metrics.
        """
        ...

    def update_metadata(self, model_id: str, metadata: dict[str, Any]) -> None:
        """
        Update arbitrary metadata for a registered model.
        """
        ...

    def get_performance_history(
        self,
        model_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get performance history for a model.
        """
        ...

    def list_compatible(
        self,
        schema_hash: str,
        role: ModelRole | None = None,
        architecture: str | None = None,
    ) -> list[ModelInfo]:
        """
        List models compatible with a given feature schema hash.
        """
        ...

    def resolve_latest(
        self,
        role: ModelRole,
        architecture: str,
        schema_hash: str,
    ) -> ModelInfo | None:
        """
        Resolve the latest model by version matching criteria.
        """
        ...


class ModelDeploymentManager:
    """
    Manages model deployment lifecycle and tracking.

    Handles deployment operations, version management, rollback,
    hot reload, and retirement of models.

    Parameters
    ----------
    models : dict[str, ModelInfo]
        Models registry (reference)
    deployments : dict[str, list[str]]
        Deployments registry (reference)
    save_callback : callable | None
        Callback to save registry state

    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        deployments: dict[str, list[str]],
        save_callback: Any = None,
    ) -> None:
        """
        Initialize deployment manager.

        Parameters
        ----------
        models : dict[str, ModelInfo]
            Models registry (reference)
        deployments : dict[str, list[str]]
            Deployments registry (reference)
        save_callback : callable | None
            Callback to trigger registry save

        """
        self._models = models
        self._deployments = deployments
        self._save_callback = save_callback

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

        # Trigger save
        if self._save_callback:
            self._save_callback()

        logger.info(f"Deployed model {model_id} to {target}")
        return True

    def get_active_models(self) -> list[ModelInfo]:
        """
        Get all currently deployed models.

        Returns
        -------
        list[ModelInfo]
            List of active model info

        """
        return [
            model_info
            for model_info in self._models.values()
            if model_info.deployment_status == DeploymentStatus.ACTIVE
        ]

    def get_all_models(self) -> list[ModelInfo]:
        """
        Get all registered models.

        Returns
        -------
        list[ModelInfo]
            List of all model info

        """
        return list(self._models.values())

    def list_compatible(
        self,
        schema_hash: str,
        role: ModelRole | None = None,
        architecture: str | None = None,
    ) -> list[ModelInfo]:
        """
        List models compatible with a given feature schema hash.

        Optionally filter by role and architecture.

        Parameters
        ----------
        schema_hash : str
            Feature schema hash
        role : ModelRole | None
            Optional role filter
        architecture : str | None
            Optional architecture filter

        Returns
        -------
        list[ModelInfo]
            List of compatible models

        """
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

        Parameters
        ----------
        role : ModelRole
            Model role
        architecture : str
            Model architecture
        schema_hash : str
            Feature schema hash

        Returns
        -------
        ModelInfo | None
            Latest model or None if not found

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

    def get_model(self, model_id: str) -> ModelInfo | None:
        """
        Get information about a specific model.

        Parameters
        ----------
        model_id : str
            Model ID

        Returns
        -------
        ModelInfo | None
            Model info or None if not found

        """
        return self._models.get(model_id)

    def get_models_by_role(self, role: ModelRole) -> list[ModelInfo]:
        """
        Get all models with a specific role.

        Parameters
        ----------
        role : ModelRole
            Model role

        Returns
        -------
        list[ModelInfo]
            List of models with the role

        """
        return [
            model_info for model_info in self._models.values() if model_info.manifest.role == role
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
            Data requirements

        Returns
        -------
        list[ModelInfo]
            List of models with the requirements

        """
        return [
            model_info
            for model_info in self._models.values()
            if model_info.manifest.data_requirements == requirements
        ]

    def get_model_lineage(self, model_id: str) -> list[ModelInfo]:
        """
        Get complete lineage of a model (parents and children).

        Parameters
        ----------
        model_id : str
            Model ID

        Returns
        -------
        list[ModelInfo]
            List of models in lineage

        """
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
        if model_id not in self._models:
            logger.error("Model %s not found in registry", model_id)
            return

        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = time.time()

        # Append to history
        self._models[model_id].performance_history.append(metrics)
        self._models[model_id].manifest.last_modified = time.time()

        # Trigger save
        if self._save_callback:
            self._save_callback()

        logger.debug(f"Tracked performance for model {model_id}: {metrics}")

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

            # Trigger save
            if self._save_callback:
                self._save_callback()

            logger.debug("Updated metadata for model %s: keys=%s", model_id, list(metadata.keys()))
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
            Model ID

        Returns
        -------
        list[dict[str, Any]]
            Performance history

        """
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

        # Trigger save
        if self._save_callback:
            self._save_callback()

        logger.info(f"Rolled back {target} to model {to_model_id}")
        return True

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

        # Trigger save
        if self._save_callback:
            self._save_callback()

        logger.info(f"Retired model {model_id}")
        return True

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
