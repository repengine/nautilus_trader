
"""
MLflow tracking integration for ML model experiments.

This module provides comprehensive MLflow integration for experiment tracking, model
registry management, and artifact storage, specifically designed for financial machine
learning workflows. Supports XGBoost, LightGBM, and scikit-learn models.

"""

from __future__ import annotations

import json
import logging
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
from ml.config.shared import MLflowConfig


if TYPE_CHECKING:
    from typing import Literal

    import mlflow

    MLFramework = Literal["xgboost", "lightgbm", "sklearn", "auto"]


# Configure module logger
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Unified MLflow tracking and model registry integration for ML models.

    This class provides comprehensive MLflow integration for financial ML workflows,
    supporting multiple ML frameworks (XGBoost, LightGBM, scikit-learn) through
    a single unified interface. Includes experiment tracking, model versioning,
    artifact management, and automatic model registration with proper metadata.

    Features:
    - Multi-framework support (XGBoost, LightGBM, scikit-learn)
    - Automatic framework detection from model type
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
    framework : str, default "auto"
        ML framework to use ("xgboost", "lightgbm", "sklearn", "auto").
        If "auto", will detect from model type.

    """

    def __init__(self, config: MLflowConfig, framework: MLFramework = "auto") -> None:
        """
        Initialize MLflow tracker.

        Parameters
        ----------
        config : MLflowConfig
            MLflow configuration settings.
        framework : str, default "auto"
            ML framework to use. Options: "xgboost", "lightgbm", "sklearn", "auto".
            If "auto", will detect from model type when logging.

        """
        self.config = config
        self.framework: MLFramework = framework
        self._mlflow: Any = None
        self._client: Any = None
        self._current_run_id: str | None = None
        self._experiment_id: str | None = None
        self._mlflow_module: Any = None  # Will be set to mlflow.xgboost, mlflow.lightgbm, etc.

    def _detect_framework(self, model: Any) -> str:
        """
        Detect ML framework from model type.

        Parameters
        ----------
        model : Any
            The model object to detect framework from.

        Returns
        -------
        str
            Detected framework name.

        Raises
        ------
        ValueError
            If framework cannot be detected.

        """
        model_type = type(model).__name__
        module_name = type(model).__module__ if hasattr(type(model), "__module__") else ""

        # Check for XGBoost
        if "xgboost" in module_name.lower() or "XGB" in model_type:
            return "xgboost"

        # Check for LightGBM
        if (
            "lightgbm" in module_name.lower()
            or "LGB" in model_type
            or model_type
            in [
                "Booster",
                "LGBMClassifier",
                "LGBMRegressor",
                "LGBMRanker",
            ]
        ):
            return "lightgbm"

        # Check for scikit-learn
        if "sklearn" in module_name.lower() or "scikit" in module_name.lower():
            return "sklearn"

        # Default to sklearn for unknown types
        logger.warning(f"Could not detect framework for {model_type}, defaulting to sklearn")
        return "sklearn"

    def _get_mlflow_module(self, framework: str | None = None) -> Any:
        """
        Get the appropriate MLflow module for the framework.

        Parameters
        ----------
        framework : str | None, optional
            Framework name. Uses self.framework if None.

        Returns
        -------
        Any
            MLflow module for the framework (e.g., mlflow.xgboost).

        """
        framework = framework or self.framework

        if framework == "xgboost":
            if not HAS_XGBOOST:
                check_ml_dependencies(["xgboost"])
            return self._mlflow.xgboost
        elif framework == "lightgbm":
            if not HAS_LIGHTGBM:
                check_ml_dependencies(["lightgbm"])
            return self._mlflow.lightgbm
        elif framework == "sklearn":
            return self._mlflow.sklearn
        else:
            # Default to sklearn
            return self._mlflow.sklearn

    def _enable_autolog(self) -> None:
        """
        Enable auto-logging for the configured framework.
        """
        if not self.config.auto_log or self.framework == "auto":
            return

        try:
            if self.framework == "xgboost":
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
                logger.info("XGBoost auto-logging enabled")
            elif self.framework == "lightgbm":
                self._mlflow.lightgbm.autolog(
                    log_models=self.config.log_model,
                    log_input_examples=False,
                    log_model_signatures=True,
                )
                logger.info("LightGBM auto-logging enabled")
            elif self.framework == "sklearn":
                self._mlflow.sklearn.autolog(
                    log_models=self.config.log_model,
                    log_input_examples=False,
                    log_model_signatures=True,
                )
                logger.info("Scikit-learn auto-logging enabled")
        except Exception as e:
            logger.warning(f"Could not enable auto-logging: {e}")

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
                logger.info(
                    f"Using MLflow experiment: {self.config.experiment_name} "
                    f"(ID: {self._experiment_id})",
                )
            except Exception as e:
                logger.warning(f"Could not set MLflow experiment: {e}")
                self._experiment_id = None

            # Initialize client for registry operations
            self._client = self._mlflow.tracking.MlflowClient()

            # Get the appropriate MLflow module and enable auto-logging if not auto-detect mode
            if self.framework != "auto":
                self._mlflow_module = self._get_mlflow_module(self.framework)

                # Enable auto-logging if configured
                if self.config.auto_log:
                    self._enable_autolog()

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
            framework_name = self.framework if self.framework != "auto" else "ml"
            run_name = f"{framework_name}_run_{timestamp}"

        # Default tags
        default_tags = {
            "model_type": self.framework if self.framework != "auto" else "unknown",
            "framework": "nautilus_trader",
            "timestamp": str(int(time.time())),
        }

        # Merge with user tags
        all_tags = {**default_tags, **(tags or {})}

        # Start run
        run = self._mlflow.start_run(run_name=run_name, tags=all_tags)
        self._current_run_id = run.info.run_id

        logger.info(f"Started MLflow run: {run_name} (ID: {self._current_run_id})")
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
        Log a complete ML training run to MLflow.

        Parameters
        ----------
        model : Any
            Trained ML model (XGBoost, LightGBM, or scikit-learn).
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

        # Auto-detect framework if needed and model is provided
        if self.framework == "auto" and model is not None:
            detected = self._detect_framework(model)
            # Type cast needed for mypy - _detect_framework returns valid framework strings
            self.framework = detected  # type: ignore[assignment]
            logger.info(f"Auto-detected framework: {self.framework}")

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

            logger.info(f"Training run logged to MLflow: {self._current_run_id}")

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
            if isinstance(value, int | float | str | bool):
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
            if isinstance(value, int | float) and np.isfinite(value):
                loggable_metrics[key] = float(value)
            elif not np.isfinite(value):
                logger.warning(f"Skipping non-finite metric {key}: {value}")

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
        Log ML model to MLflow with appropriate framework flavor.
        """
        # Ensure framework is set
        if self.framework == "auto":
            detected = self._detect_framework(model)
            # Type cast needed for mypy - _detect_framework returns valid framework strings
            self.framework = detected  # type: ignore[assignment]
            self._mlflow_module = self._get_mlflow_module(self.framework)

        try:
            # Prepare input example if feature names are provided
            input_example = None
            if feature_names:
                # Create a small example with random data
                rng = np.random.default_rng()
                input_example = rng.standard_normal((1, len(feature_names)))

            # Common parameters for all frameworks
            common_params = {
                "artifact_path": "model",
                "registered_model_name": (
                    self.config.model_name if self.config.register_model else None
                ),
                "signature": model_signature,
                "input_example": input_example,
            }

            # Log model with appropriate flavor
            if self.framework == "xgboost":
                if not HAS_XGBOOST:
                    logger.warning("XGBoost not available, skipping model logging")
                    return
                self._mlflow.xgboost.log_model(
                    xgb_model=model,
                    await_registration_for=300,  # Wait up to 5 minutes for registration
                    **common_params,
                )
            elif self.framework == "lightgbm":
                if not HAS_LIGHTGBM:
                    logger.warning("LightGBM not available, skipping model logging")
                    return
                self._mlflow.lightgbm.log_model(
                    lgb_model=model,
                    **common_params,
                )
            else:
                # Default to sklearn
                self._mlflow.sklearn.log_model(
                    sk_model=model,
                    **common_params,
                )

            logger.info(f"Model logged successfully using {self.framework} flavor")

        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

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
                    logger.debug(f"Logged artifact: {name}")

                except Exception as e:
                    logger.warning(f"Failed to log artifact {name}: {e}")

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
            logger.info(f"Created model version: {model_name} v{version}")

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
                logger.info(f"Transitioned model to {stage} stage")

            return str(version)

        except Exception as e:
            logger.error(f"Error registering model: {e}")
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
            Loaded ML model.

        """
        self._ensure_mlflow()

        try:
            model_uri = f"models:/{model_name}/{stage}"

            # Try to load with the configured framework
            if self.framework != "auto":
                mlflow_module = self._get_mlflow_module(self.framework)
                model = mlflow_module.load_model(model_uri)
                logger.info(f"Loaded {self.framework} model: {model_name} ({stage})")
                return model

            # Auto-detect framework from registered model
            # Try each framework in order
            for framework in ["xgboost", "lightgbm", "sklearn"]:
                try:
                    mlflow_module = self._get_mlflow_module(framework)
                    model = mlflow_module.load_model(model_uri)
                    logger.info(f"Loaded {framework} model: {model_name} ({stage})")
                    # Type cast needed for mypy - framework comes from our iteration
                    self.framework = framework  # type: ignore[assignment]
                    return model
                except Exception as e:
                    # Try next framework
                    logger.debug(f"Framework {framework} failed: {e}")
                    continue

            # If all fail, raise the last exception
            raise ValueError(f"Could not load model {model_name} with any supported framework")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            Loaded ML model.

        """
        self._ensure_mlflow()

        try:
            model_uri = f"models:/{model_name}/{version}"

            # Try to load with the configured framework
            if self.framework != "auto":
                mlflow_module = self._get_mlflow_module(self.framework)
                model = mlflow_module.load_model(model_uri)
                logger.info(f"Loaded {self.framework} model: {model_name} v{version}")
                return model

            # Auto-detect framework from registered model
            # Try each framework in order
            for framework in ["xgboost", "lightgbm", "sklearn"]:
                try:
                    mlflow_module = self._get_mlflow_module(framework)
                    model = mlflow_module.load_model(model_uri)
                    logger.info(f"Loaded {framework} model: {model_name} v{version}")
                    # Type cast needed for mypy - framework comes from our iteration
                    self.framework = framework  # type: ignore[assignment]
                    return model
                except Exception as e:
                    # Try next framework
                    logger.debug(f"Framework {framework} failed: {e}")
                    continue

            # If all fail, raise the last exception
            raise ValueError(
                f"Could not load model {model_name} v{version} with any supported framework",
            )

        except Exception as e:
            logger.error(f"Error loading model version: {e}")
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
            logger.error(f"Error getting model info: {e}")
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
                logger.warning(f"Experiment {experiment_name} not found")
                return

            # Get all runs, sorted by creation time
            runs = self._client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attribute.start_time DESC"],
            )

            if len(runs) <= max_runs:
                logger.info(f"Only {len(runs)} runs found, no cleanup needed")
                return

            # Delete oldest runs
            runs_to_delete = runs[max_runs:]
            for run in runs_to_delete:
                self._client.delete_run(run.info.run_id)

            logger.info(f"Cleaned up {len(runs_to_delete)} old runs")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Backward compatibility aliases
class MLflowXGBoostTracker(MLflowTracker):
    """
    Backward compatibility alias for MLflowTracker with XGBoost default.

    .. deprecated:: 2.1.0
        Use :class:`MLflowTracker` with framework="xgboost" instead.

    """

    def __init__(self, config: MLflowConfig) -> None:
        """
        Initialize XGBoost-specific MLflow tracker.

        Parameters
        ----------
        config : MLflowConfig
            MLflow configuration settings.

        """
        super().__init__(config, framework="xgboost")
        logger.warning(
            "MLflowXGBoostTracker is deprecated. "
            "Use MLflowTracker(config, framework='xgboost') instead.",
        )


class MLflowLightGBMTracker(MLflowTracker):
    """
    Backward compatibility alias for MLflowTracker with LightGBM default.

    .. deprecated:: 2.1.0
        Use :class:`MLflowTracker` with framework="lightgbm" instead.

    """

    def __init__(self, config: MLflowConfig) -> None:
        """
        Initialize LightGBM-specific MLflow tracker.

        Parameters
        ----------
        config : MLflowConfig
            MLflow configuration settings.

        """
        super().__init__(config, framework="lightgbm")
        logger.warning(
            "MLflowLightGBMTracker is deprecated. "
            "Use MLflowTracker(config, framework='lightgbm') instead.",
        )


# Explicit exports
__all__ = [
    "MLflowLightGBMTracker",  # Backward compatibility
    "MLflowTracker",  # Primary unified tracker
    "MLflowXGBoostTracker",  # Backward compatibility
]
