#!/usr/bin/env python3

"""
MLflow-based model registry implementation.

This module provides integration with MLflow for environments that
have MLflow tracking server available.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ml._imports import HAS_MLFLOW
from ml._imports import check_ml_dependencies
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.base import ModelRegistry


if HAS_MLFLOW:
    import mlflow
    from mlflow.tracking import MlflowClient


logger = logging.getLogger(__name__)


class MLflowModelRegistry(ModelRegistry):
    """
    MLflow-based model registry implementation.

    This registry integrates with MLflow Model Registry for
    enterprise environments with MLflow tracking server.

    Note: This is optional and requires MLflow to be installed.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        """
        Initialize MLflow model registry.

        Parameters
        ----------
        tracking_uri : Optional[str]
            MLflow tracking server URI
        registry_uri : Optional[str]
            MLflow model registry URI (defaults to tracking URI)
        """
        if not HAS_MLFLOW:
            check_ml_dependencies(["mlflow"])

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri or tracking_uri,
        )

        # Local cache for performance metrics (MLflow doesn't store these)
        self._performance_cache: dict[str, list[dict[str, Any]]] = {}
        self._deployment_cache: dict[str, list[str]] = {}

        logger.info(f"Initialized MLflowModelRegistry with URI: {tracking_uri}")

    def _get_model_version(self, model_name: str, version: str) -> Any:
        """Get MLflow model version object."""
        try:
            return self.client.get_model_version(model_name, version)
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None

    def register_model(
        self,
        model_path: Path,
        metadata: dict[str, Any],
        version: str | None = None,
    ) -> str:
        """
        Register a model in MLflow.

        Parameters
        ----------
        model_path : Path
            Path to the model file
        metadata : dict[str, Any]
            Model metadata
        version : Optional[str]
            Model version (auto-assigned by MLflow if not provided)

        Returns
        -------
        str
            Model version ID (format: "model_name/version")
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Extract model name from path or metadata
        model_name = metadata.get("model_name", model_path.stem)

        # Register model with MLflow
        try:
            # Log the model file
            with mlflow.start_run() as run:
                # Log metadata as parameters
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(key, value)

                # Log model artifact
                mlflow.log_artifact(str(model_path))

                # Register model version
                model_uri = f"runs:/{run.info.run_id}/model"
                result = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                )

                # Add tags to model version
                for key, value in metadata.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=result.version,
                        key=key,
                        value=str(value),
                    )

                model_id = f"{model_name}/{result.version}"
                logger.info(f"Registered model {model_id} in MLflow")
                return model_id

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deploy a model (transition to Production stage in MLflow).

        Parameters
        ----------
        model_id : str
            Model ID (format: "model_name/version")
        target : str
            Deployment target
        config : Optional[dict[str, Any]]
            Deployment configuration

        Returns
        -------
        bool
            True if deployment successful
        """
        try:
            model_name, version = model_id.split("/")

            # Transition model to Production stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=False,
            )

            # Store deployment info in cache
            if model_id not in self._deployment_cache:
                self._deployment_cache[model_id] = []
            self._deployment_cache[model_id].append(target)

            # Add deployment tags
            if config:
                self.client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="deployment_target",
                    value=target,
                )
                self.client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="deployment_config",
                    value=str(config),
                )

            logger.info(f"Deployed model {model_id} to {target}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False

    def get_active_models(self) -> list[ModelInfo]:
        """Get all models in Production stage."""
        active_models = []

        try:
            # Get all registered models
            for rm in self.client.search_registered_models():
                # Get production versions
                for mv in self.client.get_latest_versions(
                    rm.name,
                    stages=["Production"]
                ):
                    model_id = f"{mv.name}/{mv.version}"

                    # Skip models without source path
                    if mv.source is None:
                        logger.warning(f"Active model {model_id} has no source path, skipping")
                        continue

                    # Handle potentially None last_updated_timestamp
                    last_modified = (
                        mv.last_updated_timestamp / 1000
                        if mv.last_updated_timestamp is not None
                        else mv.creation_timestamp / 1000  # Fallback to creation time
                    )

                    # Create ModelInfo
                    model_info = ModelInfo(
                        model_id=model_id,
                        model_path=Path(mv.source),  # Safe now, checked above
                        version=str(mv.version),
                        metadata=dict(mv.tags),
                        deployment_status=DeploymentStatus.ACTIVE,
                        deployed_to=self._deployment_cache.get(model_id, []),
                        created_at=mv.creation_timestamp / 1000,
                        last_modified=last_modified,
                        performance_history=self._performance_cache.get(model_id, []),
                    )
                    active_models.append(model_info)

        except Exception as e:
            logger.error(f"Failed to get active models: {e}")

        return active_models

    def get_all_models(self) -> list[ModelInfo]:
        """Get all registered models."""
        all_models = []

        try:
            for rm in self.client.search_registered_models():
                for mv in self.client.search_model_versions(f"name='{rm.name}'"):
                    model_id = f"{mv.name}/{mv.version}"

                    # Skip models without source path
                    if mv.source is None:
                        logger.warning(f"Model {model_id} has no source path, skipping")
                        continue

                    # Map MLflow stage to DeploymentStatus
                    stage_map = {
                        "Production": DeploymentStatus.ACTIVE,
                        "Staging": DeploymentStatus.TESTING,
                        "Archived": DeploymentStatus.RETIRED,
                        "None": DeploymentStatus.INACTIVE,
                    }

                    # Handle potentially None current_stage
                    deployment_status = (
                        stage_map.get(mv.current_stage, DeploymentStatus.INACTIVE)
                        if mv.current_stage is not None
                        else DeploymentStatus.INACTIVE
                    )

                    # Handle potentially None last_updated_timestamp
                    last_modified = (
                        mv.last_updated_timestamp / 1000
                        if mv.last_updated_timestamp is not None
                        else mv.creation_timestamp / 1000  # Fallback to creation time
                    )

                    model_info = ModelInfo(
                        model_id=model_id,
                        model_path=Path(mv.source),  # Safe now, checked above
                        version=str(mv.version),
                        metadata=dict(mv.tags),
                        deployment_status=deployment_status,
                        deployed_to=self._deployment_cache.get(model_id, []),
                        created_at=mv.creation_timestamp / 1000,
                        last_modified=last_modified,
                        performance_history=self._performance_cache.get(model_id, []),
                    )
                    all_models.append(model_info)

        except Exception as e:
            logger.error(f"Failed to get all models: {e}")

        return all_models

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get information about a specific model."""
        try:
            model_name, version = model_id.split("/")
            mv = self._get_model_version(model_name, version)

            if mv is None:
                return None

            # Check for required source path
            if mv.source is None:
                logger.error(f"Model {model_id} has no source path")
                return None

            # Map stage to status
            stage_map = {
                "Production": DeploymentStatus.ACTIVE,
                "Staging": DeploymentStatus.TESTING,
                "Archived": DeploymentStatus.RETIRED,
                "None": DeploymentStatus.INACTIVE,
            }

            # Handle potentially None current_stage
            deployment_status = (
                stage_map.get(mv.current_stage, DeploymentStatus.INACTIVE)
                if mv.current_stage is not None
                else DeploymentStatus.INACTIVE
            )

            # Handle potentially None last_updated_timestamp
            last_modified = (
                mv.last_updated_timestamp / 1000
                if mv.last_updated_timestamp is not None
                else mv.creation_timestamp / 1000  # Fallback to creation time
            )

            return ModelInfo(
                model_id=model_id,
                model_path=Path(mv.source),  # Safe now, checked above
                version=str(mv.version),
                metadata=dict(mv.tags),
                deployment_status=deployment_status,
                deployed_to=self._deployment_cache.get(model_id, []),
                created_at=mv.creation_timestamp / 1000,
                last_modified=last_modified,
                performance_history=self._performance_cache.get(model_id, []),
            )

        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None

    def track_performance(
        self,
        model_id: str,
        metrics: dict[str, Any],
    ) -> None:
        """Track model performance metrics in local cache."""
        if "timestamp" not in metrics:
            metrics["timestamp"] = time.time()

        if model_id not in self._performance_cache:
            self._performance_cache[model_id] = []

        self._performance_cache[model_id].append(metrics)

        # Optionally log to MLflow as well
        try:
            model_name, version = model_id.split("/")
            with mlflow.start_run():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"live_{key}", value)

        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def get_performance_history(
        self,
        model_id: str,
    ) -> list[dict[str, Any]]:
        """Get performance history from cache."""
        return self._performance_cache.get(model_id, []).copy()

    def rollback(
        self,
        target: str,
        to_model_id: str,
    ) -> bool:
        """Rollback to a previous model version."""
        try:
            # Deploy the rollback model
            return self.deploy_model(to_model_id, target)

        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return False

    def retire_model(self, model_id: str) -> bool:
        """Retire a model (transition to Archived stage)."""
        try:
            model_name, version = model_id.split("/")

            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Archived",
            )

            # Clear from deployment cache
            if model_id in self._deployment_cache:
                del self._deployment_cache[model_id]

            logger.info(f"Retired model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to retire model: {e}")
            return False

    def configure_ab_test(
        self,
        models: list[str],
        split_ratio: float,
        duration_hours: int,
        target: str,
    ) -> dict[str, Any] | None:
        """
        Configure A/B test (both models to Staging).

        Note: Actual traffic splitting would be handled by the
        deployment infrastructure, not MLflow itself.
        """
        if len(models) != 2:
            logger.error("A/B test requires exactly 2 models")
            return None

        try:
            # Transition both models to Staging
            for model_id in models:
                model_name, version = model_id.split("/")
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Staging",
                )

            ab_config = {
                "model_a": models[0],
                "model_b": models[1],
                "split_ratio": split_ratio,
                "duration_hours": duration_hours,
                "target": target,
                "start_time": time.time(),
                "end_time": time.time() + (duration_hours * 3600),
            }

            logger.info(f"Configured A/B test for models {models}")
            return ab_config

        except Exception as e:
            logger.error(f"Failed to configure A/B test: {e}")
            return None

    def compare_models(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """Compare models based on performance metrics."""
        results = []

        for model_id in model_ids:
            history = self.get_performance_history(model_id)
            if history:
                # Get latest metric value
                for perf in reversed(history):
                    if metric in perf:
                        results.append({
                            "model_id": model_id,
                            metric: perf[metric],
                        })
                        break

        if not results:
            return None

        # Sort by metric
        results.sort(key=lambda x: x[metric], reverse=True)

        return {
            "metric": metric,
            "rankings": results,
            "best_model": results[0]["model_id"] if results else None,
        }
