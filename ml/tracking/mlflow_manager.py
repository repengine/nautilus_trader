# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Centralized MLflow management utilities for Nautilus ML infrastructure.

This module provides comprehensive MLflow management capabilities including experiment
tracking, model lifecycle management, and centralized operations for financial ML
workflows.

"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ml._imports import HAS_MLFLOW
from ml._imports import check_ml_dependencies
from ml._imports import mlflow
from ml.config.lightgbm_unified import MLflowConfig


if TYPE_CHECKING:
    import mlflow


class ModelStage(Enum):
    """
    Model lifecycle stages in MLflow registry.

    This enum defines the standard stages used in the MLflow model registry to track the
    lifecycle of deployed models.

    """

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class MLflowManager:
    """
    Centralized MLflow management for experiment tracking and model lifecycle.

    This class provides a comprehensive interface for MLflow operations including
    experiment management, run tracking, model registry operations, and cleanup
    utilities. It ensures consistent patterns across all ML workflows.

    Features:
    - Centralized experiment and run management
    - Context managers for automatic run lifecycle
    - Model registry operations with stage transitions
    - Artifact management and metadata tracking
    - Cleanup and retention policies
    - Performance metrics comparison
    - Graceful degradation when MLflow is unavailable

    Parameters
    ----------
    config : MLflowConfig
        Configuration for MLflow tracking and registry operations.

    """

    def __init__(self, config: MLflowConfig) -> None:
        """
        Initialize MLflow manager.

        Parameters
        ----------
        config : MLflowConfig
            MLflow configuration settings.

        """
        self.config = config
        self._mlflow: Any = None
        self._client: Any = None
        self._current_run_id: str | None = None
        self._experiment_id: str | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """
        Ensure MLflow is available and properly configured.

        Raises
        ------
        ImportError
            If MLflow is not available.

        """
        if not HAS_MLFLOW:
            check_ml_dependencies(["mlflow"])

        if not self._initialized:
            self._mlflow = mlflow

            # Configure MLflow
            if self.config.tracking_uri:
                self._mlflow.set_tracking_uri(self.config.tracking_uri)

            # Initialize client
            self._client = self._mlflow.tracking.MlflowClient(
                tracking_uri=self.config.tracking_uri,
            )

            # Setup experiment
            if self.config.experiment_name:
                self._setup_experiment()

            self._initialized = True

    def _setup_experiment(self) -> None:
        """
        Setup or create MLflow experiment.
        """
        try:
            experiment = self._mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                # Create new experiment
                self._experiment_id = self._mlflow.create_experiment(
                    name=self.config.experiment_name,
                    tags={
                        "framework": "nautilus_trader",
                        "created_by": "MLflowManager",
                        "purpose": "financial_ml",
                    },
                )
                print(f"Created new MLflow experiment: {self.config.experiment_name}")
            else:
                self._experiment_id = experiment.experiment_id
                print(f"Using existing MLflow experiment: {self.config.experiment_name}")

        except Exception as e:
            print(f"Warning: Could not setup experiment {self.config.experiment_name}: {e}")
            self._experiment_id = None

    @contextmanager
    def run_context(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Generator[str, None, None]:
        """
        Context manager for MLflow runs with automatic cleanup.

        This context manager ensures proper run lifecycle management with
        automatic start/end and cleanup on exceptions.

        Parameters
        ----------
        run_name : str | None, optional
            Name for the run. Auto-generated if None.
        tags : dict[str, str] | None, optional
            Tags to apply to the run.
        nested : bool, default False
            Whether this is a nested run within another run.

        Yields
        ------
        str
            The run ID of the active run.

        Examples
        --------
        >>> manager = MLflowManager(config)
        >>> with manager.run_context(run_name="training") as run_id:
        ...     mlflow.log_param("lr", 0.01)
        ...     mlflow.log_metric("accuracy", 0.95)

        """
        self._ensure_initialized()

        # Generate run name if not provided
        if run_name is None:
            timestamp = int(time.time())
            run_name = f"run_{timestamp}"

        # Default tags
        default_tags = {
            "framework": "nautilus_trader",
            "manager": "MLflowManager",
            "timestamp": str(int(time.time())),
        }
        all_tags = {**default_tags, **(tags or {})}

        # Start run
        run = self._mlflow.start_run(
            run_name=run_name,
            experiment_id=self._experiment_id,
            tags=all_tags,
            nested=nested,
        )
        run_id = run.info.run_id

        if nested:
            print(f"Started nested MLflow run: {run_name} (ID: {run_id})")
        else:
            self._current_run_id = run_id
            print(f"Started MLflow run: {run_name} (ID: {run_id})")

        try:
            yield run_id
        finally:
            # Always end the run
            self._mlflow.end_run()
            if not nested:
                self._current_run_id = None

    def log_training_session(
        self,
        model: Any,
        params: dict[str, Any],
        metrics: dict[str, float],
        feature_importance: dict[str, float] | None = None,
        feature_names: list[str] | None = None,
        artifacts: dict[str, Any] | None = None,
        model_signature: Any = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Log a complete training session to MLflow.

        This method provides a high-level interface for logging all aspects
        of a training session including parameters, metrics, model artifacts,
        and feature importance.

        Parameters
        ----------
        model : Any
            Trained model object (XGBoost, LightGBM, etc.).
        params : dict[str, Any]
            Training parameters and hyperparameters.
        metrics : dict[str, float]
            Training and validation metrics.
        feature_importance : dict[str, float] | None, optional
            Feature importance scores.
        feature_names : list[str] | None, optional
            List of feature names.
        artifacts : dict[str, Any] | None, optional
            Additional artifacts to log.
        model_signature : Any | None, optional
            MLflow model signature.
        run_name : str | None, optional
            Name for the run.
        tags : dict[str, str] | None, optional
            Additional tags for the run.

        Returns
        -------
        str
            The run ID of the logged session.

        """
        with self.run_context(run_name=run_name, tags=tags) as run_id:
            # Log parameters
            self._log_params_batch(params)

            # Log metrics
            self._log_metrics_batch(metrics)

            # Log feature importance
            if feature_importance:
                self._log_feature_importance(feature_importance, top_n=20)

            # Log model
            if self.config.log_model:
                self._log_model_generic(model, feature_names, model_signature)

            # Log artifacts
            if self.config.log_artifacts and artifacts:
                self._log_artifacts_batch(artifacts)

            # Log metadata
            self._log_session_metadata(feature_names, feature_importance)

        return run_id

    def _log_params_batch(self, params: dict[str, Any]) -> None:
        """
        Log parameters in batches to avoid MLflow limits.
        """
        # Filter serializable parameters
        loggable_params = {}
        for key, value in params.items():
            if isinstance(value, int | float | str | bool):
                loggable_params[key] = value
            elif value is None:
                loggable_params[key] = "None"
            else:
                loggable_params[key] = str(value)[:250]  # Truncate long strings

        # Log in batches
        batch_size = 100
        param_items = list(loggable_params.items())
        for i in range(0, len(param_items), batch_size):
            batch = dict(param_items[i : i + batch_size])
            self._mlflow.log_params(batch)

    def _log_metrics_batch(self, metrics: dict[str, float]) -> None:
        """
        Log metrics with validation and error handling.
        """
        loggable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, int | float) and np.isfinite(value):
                loggable_metrics[key] = float(value)
            elif not np.isfinite(value):
                print(f"Warning: Skipping non-finite metric {key}: {value}")

        if loggable_metrics:
            self._mlflow.log_metrics(loggable_metrics)

    def _log_feature_importance(
        self,
        feature_importance: dict[str, float],
        top_n: int = 20,
    ) -> None:
        """
        Log feature importance as metrics and artifacts.
        """
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        # Log top features as metrics
        importance_metrics = {}
        for i, (feature, importance) in enumerate(sorted_features[:top_n]):
            clean_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in feature)
            metric_name = f"importance_rank_{i+1:02d}_{clean_name}"[:64]
            importance_metrics[metric_name] = float(importance)

        if importance_metrics:
            self._mlflow.log_metrics(importance_metrics)

        # Log full importance as artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(feature_importance, f, indent=2, default=str)
            temp_path = f.name

        try:
            self._mlflow.log_artifact(temp_path, "feature_importance.json")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _log_model_generic(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        model_signature: Any = None,
    ) -> None:
        """
        Log model with automatic framework detection.
        """
        model_type = type(model).__name__.lower()

        try:
            if "xgboost" in model_type or hasattr(model, "get_booster"):
                self._mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    registered_model_name=(
                        self.config.model_name if self.config.register_model else None
                    ),
                    signature=model_signature,
                    await_registration_for=300,
                )
            elif "lightgbm" in model_type or hasattr(model, "booster_"):
                self._mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path="model",
                    registered_model_name=(
                        self.config.model_name if self.config.register_model else None
                    ),
                    signature=model_signature,
                )
            else:
                # Generic pickle logging
                self._mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=(
                        self.config.model_name if self.config.register_model else None
                    ),
                    signature=model_signature,
                )

            print("Model logged successfully")

        except Exception as e:
            print(f"Warning: Failed to log model: {e}")

    def _log_artifacts_batch(self, artifacts: dict[str, Any]) -> None:
        """
        Log multiple artifacts efficiently.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for name, content in artifacts.items():
                artifact_file = temp_path / f"{name}.json"

                try:
                    with open(artifact_file, "w", encoding="utf-8") as f:
                        if isinstance(content, dict | list):
                            json.dump(content, f, indent=2, default=str)
                        else:
                            json.dump({"content": str(content)}, f, indent=2)

                    self._mlflow.log_artifact(str(artifact_file))

                except Exception as e:
                    print(f"Warning: Failed to log artifact {name}: {e}")

    def _log_session_metadata(
        self,
        feature_names: list[str] | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> None:
        """
        Log session metadata as tags and metrics.
        """
        metadata = {
            "n_features": len(feature_names) if feature_names else 0,
            "timestamp": int(time.time()),
            "nautilus_version": "2.0.0",
        }

        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            metadata.update(
                {
                    "n_important_features": len(feature_importance),
                    "top_feature": sorted_features[0][0] if sorted_features else None,
                    "importance_sum": sum(abs(v) for v in feature_importance.values()),
                    "importance_mean": np.mean(list(feature_importance.values())),
                    "importance_std": np.std(list(feature_importance.values())),
                },
            )

        # Log as tags and metrics
        for key, value in metadata.items():
            if value is not None:
                try:
                    if isinstance(value, int | float) and np.isfinite(value):
                        self._mlflow.log_metric(f"meta_{key}", float(value))
                    self._mlflow.set_tag(key, str(value))
                except Exception as e:
                    print(f"Warning: Failed to log metadata {key}: {e}")

    def register_model(
        self,
        run_id: str,
        model_name: str | None = None,
        stage: ModelStage = ModelStage.STAGING,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Register model in MLflow model registry.

        Parameters
        ----------
        run_id : str
            Run ID containing the model to register.
        model_name : str | None, optional
            Name for the registered model. Uses config if None.
        stage : ModelStage, default ModelStage.STAGING
            Initial stage for the model version.
        description : str | None, optional
            Description for the model version.
        tags : dict[str, str] | None, optional
            Tags for the model version.

        Returns
        -------
        str
            The model version number.

        """
        self._ensure_initialized()

        model_name = model_name or self.config.model_name
        if not model_name:
            raise ValueError("Model name must be provided for registration")

        try:
            model_uri = f"runs:/{run_id}/model"

            # Create model version
            model_version = self._client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description=description,
            )

            version = model_version.version
            print(f"Created model version: {model_name} v{version}")

            # Add tags
            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(model_name, version, key, value)

            # Transition to stage
            if stage != ModelStage.NONE:
                self.transition_model_stage(model_name, version, stage)

            return str(version)

        except Exception as e:
            print(f"Error registering model: {e}")
            raise

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = False,
    ) -> None:
        """
        Transition model to a new stage.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        version : str
            Version number to transition.
        stage : ModelStage
            Target stage for the model.
        archive_existing : bool, default False
            Whether to archive existing versions in the target stage.

        """
        self._ensure_initialized()

        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value,
                archive_existing_versions=archive_existing,
            )
            print(f"Transitioned {model_name} v{version} to {stage.value}")

        except Exception as e:
            print(f"Error transitioning model stage: {e}")
            raise

    def load_model(
        self,
        model_name: str,
        stage: ModelStage = ModelStage.PRODUCTION,
    ) -> Any:
        """
        Load model from registry by stage.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        stage : ModelStage, default ModelStage.PRODUCTION
            Stage of the model to load.

        Returns
        -------
        Any
            Loaded model object.

        """
        self._ensure_initialized()

        try:
            model_uri = f"models:/{model_name}/{stage.value}"

            # Try different model flavors
            try:
                model = self._mlflow.xgboost.load_model(model_uri)
            except Exception:
                try:
                    model = self._mlflow.lightgbm.load_model(model_uri)
                except Exception:
                    model = self._mlflow.sklearn.load_model(model_uri)

            print(f"Loaded model: {model_name} ({stage.value})")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_model_by_version(self, model_name: str, version: str) -> Any:
        """
        Load specific model version from registry.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        version : str
            Version number to load.

        Returns
        -------
        Any
            Loaded model object.

        """
        self._ensure_initialized()

        try:
            model_uri = f"models:/{model_name}/{version}"

            # Try different model flavors
            try:
                model = self._mlflow.xgboost.load_model(model_uri)
            except Exception:
                try:
                    model = self._mlflow.lightgbm.load_model(model_uri)
                except Exception:
                    model = self._mlflow.sklearn.load_model(model_uri)

            print(f"Loaded model: {model_name} v{version}")
            return model

        except Exception as e:
            print(f"Error loading model version: {e}")
            raise

    def compare_models(
        self,
        model_names: list[str],
        metric_name: str,
        stage: ModelStage = ModelStage.PRODUCTION,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare models across different names or versions.

        Parameters
        ----------
        model_names : list[str]
            List of model names to compare.
        metric_name : str
            Metric to compare (e.g., "accuracy", "auc").
        stage : ModelStage, default ModelStage.PRODUCTION
            Stage to compare models in.

        Returns
        -------
        dict[str, dict[str, Any]]
            Comparison results with model metadata and metrics.

        """
        self._ensure_initialized()

        results = {}

        for model_name in model_names:
            try:
                # Get latest version in stage
                versions = self._client.get_latest_versions(
                    model_name,
                    stages=[stage.value],
                )

                if not versions:
                    print(f"No {stage.value} version found for {model_name}")
                    continue

                version = versions[0]
                run_id = version.run_id

                # Get run details
                run = self._client.get_run(run_id)

                results[model_name] = {
                    "version": version.version,
                    "run_id": run_id,
                    "stage": stage.value,
                    "metric_value": run.data.metrics.get(metric_name, None),
                    "creation_timestamp": version.creation_timestamp,
                    "description": version.description,
                    "tags": dict(version.tags) if version.tags else {},
                    "run_metrics": dict(run.data.metrics),
                    "run_params": dict(run.data.params),
                }

            except Exception as e:
                print(f"Error getting info for {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def cleanup_old_runs(
        self,
        max_runs: int = 100,
        experiment_name: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """
        Clean up old runs to prevent storage bloat.

        Parameters
        ----------
        max_runs : int, default 100
            Maximum number of runs to keep per experiment.
        experiment_name : str | None, optional
            Specific experiment to clean. If None, uses all experiments.
        dry_run : bool, default False
            If True, only report what would be deleted without deleting.

        Returns
        -------
        dict[str, int]
            Cleanup statistics.

        """
        self._ensure_initialized()

        stats = {"runs_examined": 0, "runs_deleted": 0, "experiments_processed": 0}

        try:
            # Get experiments to process
            if experiment_name:
                experiment = self._mlflow.get_experiment_by_name(experiment_name)
                experiments = [experiment] if experiment else []
            else:
                experiments = self._client.search_experiments()

            for experiment in experiments:
                if experiment is None:
                    continue

                stats["experiments_processed"] += 1

                # Get runs sorted by creation time (newest first)
                runs = self._client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["attribute.start_time DESC"],
                )

                stats["runs_examined"] += len(runs)

                if len(runs) <= max_runs:
                    continue

                # Mark runs for deletion
                runs_to_delete = runs[max_runs:]

                if not dry_run:
                    for run in runs_to_delete:
                        try:
                            self._client.delete_run(run.info.run_id)
                            stats["runs_deleted"] += 1
                        except Exception as e:
                            print(f"Warning: Could not delete run {run.info.run_id}: {e}")
                else:
                    stats["runs_deleted"] += len(runs_to_delete)
                    print(
                        f"Would delete {len(runs_to_delete)} runs from "
                        f"experiment {experiment.name}",
                    )

            if not dry_run and stats["runs_deleted"] > 0:
                print(f"Cleaned up {stats['runs_deleted']} old runs")

        except Exception as e:
            print(f"Error during cleanup: {e}")

        return stats

    def get_experiment_summary(self, experiment_name: str | None = None) -> dict[str, Any]:
        """
        Get summary statistics for an experiment.

        Parameters
        ----------
        experiment_name : str | None, optional
            Name of experiment to summarize. Uses config experiment if None.

        Returns
        -------
        dict[str, Any]
            Experiment summary including run counts, metrics, etc.

        """
        self._ensure_initialized()

        experiment_name = experiment_name or self.config.experiment_name
        if not experiment_name:
            raise ValueError("Experiment name must be provided")

        try:
            experiment = self._mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment {experiment_name} not found")

            # Get all runs
            runs = self._client.search_runs(
                experiment_ids=[experiment.experiment_id],
            )

            summary = {
                "experiment_name": experiment_name,
                "experiment_id": experiment.experiment_id,
                "total_runs": len(runs),
                "active_runs": sum(1 for r in runs if r.info.status == "RUNNING"),
                "completed_runs": sum(1 for r in runs if r.info.status == "FINISHED"),
                "failed_runs": sum(1 for r in runs if r.info.status == "FAILED"),
                "creation_time": experiment.creation_time,
                "tags": dict(experiment.tags) if experiment.tags else {},
            }

            # Get metric statistics if runs exist
            if runs:
                all_metrics: dict[str, list[float]] = {}
                for run in runs:
                    for metric_name, value in run.data.metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)

                metric_stats = {}
                for metric_name, values in all_metrics.items():
                    if values:
                        metric_stats[metric_name] = {
                            "count": len(values),
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values),
                        }

                summary["metric_statistics"] = metric_stats

            return summary

        except Exception as e:
            print(f"Error getting experiment summary: {e}")
            raise

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on MLflow connectivity and configuration.

        Returns
        -------
        dict[str, Any]
            Health status information.

        """
        status = {
            "mlflow_available": HAS_MLFLOW,
            "initialized": self._initialized,
            "tracking_uri": self.config.tracking_uri,
            "experiment_name": self.config.experiment_name,
            "connectivity": False,
            "experiment_exists": False,
        }

        if HAS_MLFLOW:
            try:
                self._ensure_initialized()

                # Test connectivity
                experiments = self._client.search_experiments()
                status["connectivity"] = True
                status["total_experiments"] = len(experiments)

                # Check if configured experiment exists
                if self.config.experiment_name:
                    experiment = self._mlflow.get_experiment_by_name(
                        self.config.experiment_name,
                    )
                    status["experiment_exists"] = experiment is not None
                    if experiment:
                        status["experiment_id"] = experiment.experiment_id

            except Exception as e:
                status["error"] = str(e)

        return status
