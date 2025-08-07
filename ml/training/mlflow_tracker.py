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
MLflow tracking integration for XGBoost model experiments.

This module provides comprehensive MLflow integration for experiment tracking, model
registry management, and artifact storage, specifically designed for financial machine
learning workflows with XGBoost models.

"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ml._imports import HAS_LIGHTGBM
from ml._imports import HAS_MLFLOW
from ml._imports import HAS_XGBOOST
from ml._imports import check_ml_dependencies
from ml._imports import mlflow
from ml.config.lightgbm_unified import MLflowConfig


if TYPE_CHECKING:
    import mlflow


class MLflowXGBoostTracker:
    """
    MLflow tracking and model registry integration for XGBoost models.

    This class provides comprehensive MLflow integration for financial ML workflows,
    including experiment tracking, model versioning, artifact management, and
    automatic model registration with proper metadata.

    Features:
    - Experiment and run management
    - Model registry integration with versioning
    - Artifact storage (feature importance, SHAP values, etc.)
    - Automatic model tagging and metadata
    - Model deployment stage management
    - Performance metrics tracking over time

    Parameters
    ----------
    config : MLflowConfig
        Configuration for MLflow tracking and registry.

    """

    def __init__(self, config: MLflowConfig) -> None:
        """
        Initialize MLflow tracker.

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

    def _ensure_mlflow(self) -> None:
        """
        Ensure MLflow is available and properly configured.
        """
        if not HAS_MLFLOW:
            check_ml_dependencies(["mlflow"])

        if self._mlflow is None:
            self._mlflow = mlflow

            # Configure MLflow
            self._mlflow.set_tracking_uri(self.config.tracking_uri)

            # Set or create experiment
            try:
                experiment = self._mlflow.set_experiment(self.config.experiment_name)
                self._experiment_id = experiment.experiment_id
                print(
                    f"Using MLflow experiment: {self.config.experiment_name} (ID: {self._experiment_id})",
                )
            except Exception as e:
                print(f"Warning: Could not set MLflow experiment: {e}")
                self._experiment_id = None

            # Initialize client for registry operations
            self._client = self._mlflow.tracking.MlflowClient()

            # Enable auto-logging if configured
            if self.config.auto_log:
                try:
                    self._mlflow.xgboost.autolog(
                        importance_types=["weight", "gain", "cover"],
                        log_input_examples=False,  # Can be large for financial data
                        log_model_signatures=True,
                        log_models=True,
                        disable=False,
                        exclusive=False,
                        disable_for_unsupported_versions=False,
                        silent=False,
                    )
                    print("XGBoost auto-logging enabled")
                except Exception as e:
                    print(f"Warning: Could not enable auto-logging: {e}")

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        """
        Start a new MLflow run.

        Parameters
        ----------
        run_name : str | None, optional
            Name for the run. Auto-generated if None.
        tags : dict[str, str] | None, optional
            Tags to apply to the run.

        Returns
        -------
        str
            The run ID of the started run.

        """
        self._ensure_mlflow()

        # Generate run name if not provided
        if run_name is None:
            timestamp = int(time.time())
            run_name = f"xgboost_run_{timestamp}"

        # Default tags
        default_tags = {
            "model_type": "xgboost",
            "framework": "nautilus_trader",
            "timestamp": str(int(time.time())),
        }

        # Merge with user tags
        all_tags = {**default_tags, **(tags or {})}

        # Start run
        run = self._mlflow.start_run(run_name=run_name, tags=all_tags)
        self._current_run_id = run.info.run_id

        print(f"Started MLflow run: {run_name} (ID: {self._current_run_id})")
        return self._current_run_id

    def log_training_run(
        self,
        model: Any,
        params: dict[str, Any],
        metrics: dict[str, float],
        feature_importance: dict[str, float],
        feature_names: list[str] | None = None,
        artifacts: dict[str, Any] | None = None,
        model_signature: Any = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Log a complete XGBoost training run to MLflow.

        Parameters
        ----------
        model : Any
            Trained XGBoost model.
        params : dict[str, Any]
            Model parameters used for training.
        metrics : dict[str, float]
            Training and validation metrics.
        feature_importance : dict[str, float]
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
            Tags to apply to the run.

        Returns
        -------
        str
            The run ID of the logged run.

        """
        self._ensure_mlflow()

        # Start run if not already active
        if not self._mlflow.active_run():
            self.start_run(run_name=run_name, tags=tags)

        try:
            # Log parameters
            self._log_parameters(params)

            # Log metrics
            self._log_metrics(metrics)

            # Log feature importance as metrics (top features only)
            self._log_feature_importance(feature_importance)

            # Log model
            if self.config.log_model:
                self._log_model(model, feature_names, model_signature)

            # Log artifacts
            if self.config.log_artifacts and artifacts:
                self._log_artifacts(artifacts)

            # Log additional metadata
            self._log_metadata(feature_names, feature_importance)

            print(f"✅ Training run logged to MLflow: {self._current_run_id}")

            return self._current_run_id or ""

        finally:
            # Always end the run
            if self._mlflow.active_run():
                self._mlflow.end_run()

    def _log_parameters(self, params: dict[str, Any]) -> None:
        """
        Log model parameters to MLflow.
        """
        # Filter out non-serializable parameters
        loggable_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                loggable_params[key] = value
            elif value is None:
                loggable_params[key] = "None"
            else:
                loggable_params[key] = str(value)

        # Log parameters in batches to avoid MLflow limits
        batch_size = 100
        for i in range(0, len(loggable_params), batch_size):
            batch = dict(list(loggable_params.items())[i : i + batch_size])
            self._mlflow.log_params(batch)

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """
        Log training metrics to MLflow.
        """
        # Convert all metrics to float and handle NaN/inf
        loggable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                loggable_metrics[key] = float(value)
            elif not np.isfinite(value):
                print(f"Warning: Skipping non-finite metric {key}: {value}")

        # Log metrics
        if loggable_metrics:
            self._mlflow.log_metrics(loggable_metrics)

    def _log_feature_importance(self, feature_importance: dict[str, float]) -> None:
        """
        Log feature importance scores as metrics.
        """
        # Log top 20 features to avoid cluttering the UI
        top_features = dict(list(feature_importance.items())[:20])

        importance_metrics = {}
        for feature, importance in top_features.items():
            # Clean feature name for MLflow (remove special characters)
            clean_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in feature)
            metric_name = f"importance_{clean_name}"
            importance_metrics[metric_name] = float(importance)

        if importance_metrics:
            self._mlflow.log_metrics(importance_metrics)

    def _log_model(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        model_signature: Any = None,
    ) -> None:
        """
        Log XGBoost model to MLflow.
        """
        if not HAS_XGBOOST:
            print("Warning: XGBoost not available, skipping model logging")
            return

        try:
            # Prepare input example if feature names are provided
            input_example = None
            if feature_names:
                # Create a small example with random data
                input_example = np.random.randn(1, len(feature_names))

            # Log model with XGBoost flavor
            self._mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                registered_model_name=(
                    self.config.model_name if self.config.register_model else None
                ),
                signature=model_signature,
                input_example=input_example,
                await_registration_for=300,  # Wait up to 5 minutes for registration
            )

            print("Model logged successfully")

        except Exception as e:
            print(f"Warning: Failed to log model: {e}")

    def _log_artifacts(self, artifacts: dict[str, Any]) -> None:
        """
        Log additional artifacts to MLflow.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for name, content in artifacts.items():
                artifact_path = temp_path / f"{name}.json"
                try:
                    with open(artifact_path, "w") as f:
                        if isinstance(content, dict):
                            json.dump(content, f, indent=2, default=str)
                        else:
                            json.dump({"content": str(content)}, f, indent=2)

                    self._mlflow.log_artifact(str(artifact_path))
                    print(f"Logged artifact: {name}")

                except Exception as e:
                    print(f"Warning: Failed to log artifact {name}: {e}")

    def _log_metadata(
        self,
        feature_names: list[str] | None = None,
        feature_importance: dict[str, float] | None = None,
    ) -> None:
        """
        Log additional metadata about the training run.
        """
        metadata = {
            "n_features": len(feature_names) if feature_names else 0,
            "timestamp": int(time.time()),
            "nautilus_version": "2.0.0",  # Could be dynamically obtained
        }

        if feature_importance:
            metadata.update(
                {
                    "n_important_features": len(feature_importance),
                    "top_feature": (
                        max(feature_importance.items(), key=lambda x: x[1])[0]
                        if feature_importance
                        else None
                    ),
                    "importance_sum": sum(feature_importance.values()),
                },
            )

        # Log as tags
        for key, value in metadata.items():
            if value is not None:
                self._mlflow.set_tag(key, str(value))

    def register_model(
        self,
        run_id: str,
        model_name: str | None = None,
        stage: str = "Staging",
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
        stage : str, default "Staging"
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
        self._ensure_mlflow()

        model_name = model_name or self.config.model_name
        if not model_name:
            raise ValueError("Model name must be provided for registration")

        try:
            # Model URI
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

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(model_name, version, key, value)

            # Transition to specified stage
            if stage != "None":
                self._client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage,
                    archive_existing_versions=False,
                )
                print(f"Transitioned model to {stage} stage")

            return str(version)

        except Exception as e:
            print(f"Error registering model: {e}")
            raise

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """
        Load model from MLflow registry.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        stage : str, default "Production"
            Stage of the model to load.

        Returns
        -------
        Any
            Loaded XGBoost model.

        """
        self._ensure_mlflow()

        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = self._mlflow.xgboost.load_model(model_uri)
            print(f"Loaded model: {model_name} ({stage})")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_model_by_version(self, model_name: str, version: str) -> Any:
        """
        Load specific model version from MLflow registry.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        version : str
            Version number to load.

        Returns
        -------
        Any
            Loaded XGBoost model.

        """
        self._ensure_mlflow()

        try:
            model_uri = f"models:/{model_name}/{version}"
            model = self._mlflow.xgboost.load_model(model_uri)
            print(f"Loaded model: {model_name} v{version}")
            return model

        except Exception as e:
            print(f"Error loading model version: {e}")
            raise

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """
        Get information about registered model and its versions.

        Parameters
        ----------
        model_name : str
            Name of the registered model.

        Returns
        -------
        dict[str, Any]
            Model information and version details.

        """
        self._ensure_mlflow()

        try:
            # Get model details
            model = self._client.get_registered_model(model_name)

            # Get all versions
            versions = self._client.get_latest_versions(
                model_name,
                stages=["None", "Staging", "Production", "Archived"],
            )

            model_info = {
                "name": model.name,
                "description": model.description,
                "tags": dict(model.tags) if model.tags else {},
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "description": v.description,
                        "run_id": v.run_id,
                        "creation_timestamp": v.creation_timestamp,
                        "tags": dict(v.tags) if v.tags else {},
                    }
                    for v in versions
                ],
            }

            return model_info

        except Exception as e:
            print(f"Error getting model info: {e}")
            raise

    def cleanup_old_runs(self, max_runs: int = 100, experiment_name: str | None = None) -> None:
        """
        Clean up old runs to prevent storage bloat.

        Parameters
        ----------
        max_runs : int, default 100
            Maximum number of runs to keep.
        experiment_name : str | None, optional
            Experiment name to clean. Uses config experiment if None.

        """
        self._ensure_mlflow()

        experiment_name = experiment_name or self.config.experiment_name

        try:
            experiment = self._mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"Experiment {experiment_name} not found")
                return

            # Get all runs, sorted by creation time
            runs = self._client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
            )

            if len(runs) <= max_runs:
                print(f"Only {len(runs)} runs found, no cleanup needed")
                return

            # Delete oldest runs
            runs_to_delete = runs[max_runs:]
            for run in runs_to_delete:
                self._client.delete_run(run.info.run_id)

            print(f"Cleaned up {len(runs_to_delete)} old runs")

        except Exception as e:
            print(f"Error during cleanup: {e}")


class MLflowLightGBMTracker:
    """
    MLflow tracking and model registry integration for LightGBM models.

    This class provides comprehensive MLflow integration for financial ML workflows,
    including experiment tracking, model versioning, artifact management, and
    automatic model registration with proper metadata for LightGBM models.

    Features:
    - Experiment and run management
    - Model registry integration with versioning
    - Artifact storage (feature importance, SHAP values, etc.)
    - Automatic model tagging and metadata
    - Model deployment stage management
    - Performance metrics tracking over time

    Parameters
    ----------
    config : MLflowConfig
        Configuration for MLflow tracking and registry.

    """

    def __init__(self, config: MLflowConfig) -> None:
        """
        Initialize MLflow tracker for LightGBM models.

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

    def _ensure_mlflow(self) -> None:
        """
        Ensure MLflow is available and properly configured.
        """
        if not HAS_MLFLOW:
            check_ml_dependencies(["mlflow"])
        if not HAS_LIGHTGBM:
            check_ml_dependencies(["lightgbm"])

        if self._mlflow is None:
            self._mlflow = mlflow

            # Configure MLflow
            self._mlflow.set_tracking_uri(self.config.tracking_uri)

            # Set or create experiment
            try:
                experiment = self._mlflow.set_experiment(self.config.experiment_name)
                self._experiment_id = experiment.experiment_id
            except Exception as e:
                print(f"Warning: Could not set experiment {self.config.experiment_name}: {e}")
                self._experiment_id = None

            # Initialize client
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(tracking_uri=self.config.tracking_uri)

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        """
        Start new MLflow run.

        Parameters
        ----------
        run_name : str | None, optional
            Name for the run.
        tags : dict[str, str] | None, optional
            Tags to set for the run.

        Returns
        -------
        str
            The run ID.

        """
        self._ensure_mlflow()

        # Enable autologging if configured
        if self.config.auto_log:
            self._mlflow.lightgbm.autolog(
                log_models=self.config.log_model,
                log_input_examples=False,
                log_model_signatures=True,
            )

        run = self._mlflow.start_run(run_name=run_name, tags=tags)
        self._current_run_id = run.info.run_id

        print(f"Started MLflow run: {self._current_run_id}")
        return self._current_run_id

    def end_run(self) -> None:
        """
        End current MLflow run.
        """
        if self._mlflow and self._current_run_id:
            self._mlflow.end_run()
            print(f"Ended MLflow run: {self._current_run_id}")
            self._current_run_id = None

    def log_config(self, config: Any) -> None:
        """
        Log training configuration.

        Parameters
        ----------
        config : Any
            Training configuration to log.

        """
        if not self._current_run_id:
            return

        try:
            # Convert config to dictionary for logging
            if hasattr(config, "__dict__"):
                config_dict = config.__dict__
            else:
                config_dict = {"config": str(config)}

            # Flatten nested configurations
            flat_config = {}
            for key, value in config_dict.items():
                if hasattr(value, "__dict__"):
                    # Nested config object
                    for nested_key, nested_value in value.__dict__.items():
                        flat_config[f"{key}_{nested_key}"] = nested_value
                else:
                    flat_config[key] = value

            # Log parameters
            for key, value in flat_config.items():
                if value is not None:
                    try:
                        self._mlflow.log_param(key, value)
                    except Exception as e:
                        print(f"Warning: Could not log parameter {key}: {e}")

        except Exception as e:
            print(f"Warning: Could not log config: {e}")

    def log_model(
        self,
        model: Any,
        training_time: float,
        feature_names: list[str] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        """
        Log LightGBM model and associated artifacts.

        Parameters
        ----------
        model : Any
            Trained LightGBM model.
        training_time : float
            Time spent training in seconds.
        feature_names : list[str] | None, optional
            Names of features.
        artifacts : dict[str, Any] | None, optional
            Additional artifacts to log.

        """
        if not self._current_run_id:
            return

        try:
            # Log model if enabled
            if self.config.log_model and not self.config.auto_log:
                self._mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path="model",
                    registered_model_name=(
                        self.config.model_name if self.config.register_model else None
                    ),
                )

            # Log training metrics
            self._mlflow.log_metric("training_time", training_time)
            self._mlflow.log_metric("num_features", model.num_feature())
            self._mlflow.log_metric("best_iteration", model.best_iteration)

            # Log feature importance
            if hasattr(model, "feature_importance"):
                importance = model.feature_importance(importance_type="gain")

                # Log top feature importances as metrics
                if feature_names and len(feature_names) == len(importance):
                    feature_importance = dict(zip(feature_names, importance.astype(float)))

                    # Log top 10 features
                    sorted_features = sorted(
                        feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    for i, (feature, imp) in enumerate(sorted_features[:10]):
                        self._mlflow.log_metric(f"feature_importance_rank_{i+1}", imp)
                        self._mlflow.set_tag(f"top_feature_{i+1}", feature)

            # Log artifacts if enabled and provided
            if self.config.log_artifacts and artifacts:
                self._log_artifacts(artifacts)

            # Log metadata
            self._log_metadata(feature_names, artifacts)

        except Exception as e:
            print(f"Warning: Error logging model: {e}")

    def _log_artifacts(self, artifacts: dict[str, Any]) -> None:
        """
        Log additional artifacts to MLflow.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, content in artifacts.items():
                try:
                    artifact_path = Path(temp_dir) / f"{name}.json"

                    with open(artifact_path, "w", encoding="utf-8") as f:
                        if isinstance(content, (dict, list)):
                            json.dump(content, f, indent=2, default=str)
                        else:
                            json.dump({"content": str(content)}, f, indent=2)

                    self._mlflow.log_artifact(str(artifact_path))
                    print(f"Logged artifact: {name}")

                except Exception as e:
                    print(f"Warning: Failed to log artifact {name}: {e}")

    def _log_metadata(
        self,
        feature_names: list[str] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        """
        Log additional metadata about the training run.
        """
        metadata = {
            "n_features": len(feature_names) if feature_names else 0,
            "timestamp": int(time.time()),
            "nautilus_version": "2.0.0",
            "model_type": "lightgbm",
        }

        if artifacts and "feature_importance" in artifacts:
            feature_importance = artifacts["feature_importance"]
            if isinstance(feature_importance, dict):
                metadata.update(
                    {
                        "n_important_features": len(feature_importance),
                        "top_feature": (
                            max(feature_importance.items(), key=lambda x: x[1])[0]
                            if feature_importance
                            else None
                        ),
                        "importance_sum": sum(feature_importance.values()),
                    },
                )

        # Log as tags
        for key, value in metadata.items():
            if value is not None:
                self._mlflow.set_tag(key, str(value))

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """
        Load model from MLflow registry.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        stage : str, default "Production"
            Stage of the model to load.

        Returns
        -------
        Any
            Loaded LightGBM model.

        """
        self._ensure_mlflow()

        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = self._mlflow.lightgbm.load_model(model_uri)
            print(f"Loaded LightGBM model: {model_name} ({stage})")
            return model

        except Exception as e:
            print(f"Error loading LightGBM model: {e}")
            raise

    def load_model_by_version(self, model_name: str, version: str) -> Any:
        """
        Load specific model version from MLflow registry.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        version : str
            Version number to load.

        Returns
        -------
        Any
            Loaded LightGBM model.

        """
        self._ensure_mlflow()

        try:
            model_uri = f"models:/{model_name}/{version}"
            model = self._mlflow.lightgbm.load_model(model_uri)
            print(f"Loaded LightGBM model: {model_name} v{version}")
            return model

        except Exception as e:
            print(f"Error loading LightGBM model version: {e}")
            raise

    def register_model(
        self,
        run_id: str,
        model_name: str | None = None,
        stage: str = "Staging",
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Register LightGBM model in MLflow model registry.

        Parameters
        ----------
        run_id : str
            Run ID containing the model to register.
        model_name : str | None, optional
            Name for the registered model. Uses config if None.
        stage : str, default "Staging"
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
        self._ensure_mlflow()

        model_name = model_name or self.config.model_name
        if not model_name:
            raise ValueError("Model name must be provided for registration")

        try:
            # Model URI
            model_uri = f"runs:/{run_id}/model"

            # Create model version
            model_version = self._client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description=description,
            )

            version = model_version.version
            print(f"Created LightGBM model version: {model_name} v{version}")

            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(model_name, version, key, value)

            # Transition to specified stage
            if stage != "None":
                self._client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage,
                    archive_existing_versions=False,
                )
                print(f"Transitioned LightGBM model to {stage} stage")

            return str(version)

        except Exception as e:
            print(f"Error registering LightGBM model: {e}")
            raise


# Explicit exports
__all__ = [
    "MLflowLightGBMTracker",
    "MLflowXGBoostTracker",
]
